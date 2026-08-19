// SPDX-License-Identifier: MPL-2.0
//! [`InferenceGate`]: priority-aware admission control for outbound inference
//! requests. The rationale lives on the type, which is the part hosts see.

#[cfg(miri)]
use std::time::Instant;
use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    sync::{
        Arc,
        Mutex,
        MutexGuard,
        PoisonError,
        atomic::{
            AtomicU64,
            Ordering::Relaxed,
        },
    },
};

use metrics::{
    gauge,
    histogram,
};
#[cfg(not(miri))]
use minstant::Instant;
use tokio::sync::oneshot;

use crate::observability::{
    INFERENCE_GATE_CAPACITY,
    INFERENCE_GATE_QUEUED,
    INFERENCE_GATE_WAIT,
    INFERENCE_IN_FLIGHT,
};

tokio::task_local! {
    /// The gate and priority in effect for requests issued by this task.
    static ACTIVE_GATE: GateScope;
}

/// The scheduling priority of one inference request: higher is served first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Priority(u32);

impl Priority {
    /// The priority of a lane's first attempt, and the lowest one.
    pub const FIRST_ATTEMPT: Self = Self(1);

    /// The priority of the attempt of a lane, counting from 1.
    ///
    /// Monotonically increasing in `attempt`: the lane closest to exhausting
    /// its retry budget — the one that has already accumulated the most
    /// latency, and so is most likely to be the batch's tail — is served
    /// first.
    #[must_use]
    pub(crate) fn attempt(attempt: usize) -> Self {
        Self(u32::try_from(attempt).unwrap_or(u32::MAX))
    }
}

impl Default for Priority {
    fn default() -> Self {
        Self::FIRST_ATTEMPT
    }
}

/// The gate and priority captured from the task-local, ready to be
/// acquired against.
#[derive(Debug, Clone)]
pub(crate) struct GateScope {
    gate: InferenceGate,
    priority: Priority,
}

impl GateScope {
    /// The scope of the calling task, if it runs inside an
    pub(crate) fn current() -> Option<Self> {
        ACTIVE_GATE.try_with(Clone::clone).ok()
    }

    /// Wait for a slot at the endpoint.
    pub(crate) async fn acquire(self) -> GatePermit {
        self.gate.acquire(self.priority).await
    }
}

#[derive(Debug)]
struct GateInner {
    state: Mutex<State>,
    /// Hands out the tiebreaker that makes waiters of equal priority FIFO.
    next_seq: AtomicU64,
}

#[derive(Debug)]
struct State {
    capacity: u16,
    /// Permits handed out but not yet released.
    in_flight: u16,
    waiters: BinaryHeap<Waiter>,
}

/// A task parked on the gate, holding the channel its permit will arrive
/// through.
#[derive(Debug)]
struct Waiter {
    priority: Priority,
    seq: u64,
    tx: oneshot::Sender<GatePermit>,
}

impl Ord for Waiter {
    /// Max-heap order: highest priority first, and among equals the one that
    /// arrived first (lowest `seq`).
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

impl PartialOrd for Waiter {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Waiter {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq
    }
}

impl Eq for Waiter {}

/// A limit on how many inference requests may be resident at the endpoint at
/// once, enforced where the requests are actually sent.
///
/// Cheap to clone: every clone is a handle to the same budget.
/// [`crate::Runtime`] owns one and exposes its capacity as
/// [`crate::Runtime::set_max_in_flight`]; hosts do not normally construct one.
///
/// # Priority
///
/// Waiters are served highest-priority-first, FIFO within a priority, rather
/// than the plain FIFO a [`tokio::sync::Semaphore`] would give.
/// [`crate::Runtime`] uses the lane's attempt number as the priority, so a lane
/// on its fourth repair outranks a lane that has not started. Two reasons:
///
/// - The straggler *is* the tail. A batch finishes when its slowest lane does,
///   so queueing a repair behind freshly admitted work directly extends the
///   batch.
/// - A repair request is a prefix-extension of the request before it — same
///   preamble, same history, one more turn — so on a server with prefix caching
///   it is the cheapest prefill in the batch.
///
/// # How the gate reaches the HTTP client
///
/// Through a task-local, set by [`InferenceGate::scope`]. The request is issued
/// deep inside `rig`'s prompt loop. [`crate::MeteredHttpClient`] reads
/// it synchronously in `send`, on the caller's task, before building the future
/// it returns; the resulting handle is owned by that future, so it stays
/// correct even if the future is later polled elsewhere.
#[derive(Debug, Clone)]
pub(crate) struct InferenceGate(Arc<GateInner>);

impl InferenceGate {
    /// A gate admitting at most `capacity` concurrent requests.
    ///
    /// `capacity` is clamped to at least 1; a gate of zero could never make
    /// progress.
    #[must_use]
    pub(crate) fn new(capacity: u16) -> Self {
        Self(Arc::new(GateInner {
            state: Mutex::new(State {
                capacity: capacity.max(1),
                in_flight: 0,
                waiters: BinaryHeap::new(),
            }),
            next_seq: AtomicU64::new(0),
        }))
    }

    /// A gate that never blocks.
    #[must_use]
    pub(crate) fn unlimited() -> Self {
        Self::new(u16::MAX)
    }

    /// Change the limit.
    ///
    /// Raising it wakes the highest-priority waiters immediately. Lowering it
    /// never revokes a permit already handed out: the surplus drains as those
    /// requests finish, and no new one is admitted until `in_flight` is back
    /// under the new capacity.
    pub(crate) fn set_capacity(&self, capacity: u16) {
        let mut state = self.state();
        state.capacity = capacity.max(1);
        while state.in_flight < state.capacity {
            if self.grant_to_waiter(&mut state) {
                state.in_flight += 1;
            } else {
                break;
            }
        }
        Self::record(&state);
    }

    /// The current limit.
    #[must_use]
    pub(crate) fn capacity(&self) -> u16 {
        self.state().capacity
    }

    /// Requests currently resident at the endpoint.
    #[must_use]
    pub(crate) fn in_flight(&self) -> u16 {
        self.state().in_flight
    }

    /// Number of requests waiting for a slot.
    #[must_use]
    pub(crate) fn queued(&self) -> usize {
        self.state().waiters.len()
    }

    /// Run `fut` with this gate and `priority` installed as the scope,
    /// so every request `fut` issues — at any depth, including inside a
    /// provider's tool-calling loop — waits for a slot.
    ///
    /// The scope does not survive [`tokio::spawn`]: a task spawned inside
    /// `fut` starts with no gate and its requests go ungated. Re-enter
    /// the scope inside the spawned task if you need it there.
    pub(crate) async fn scope<F>(&self, priority: Priority, fut: F) -> F::Output
    where
        F: Future,
    {
        ACTIVE_GATE
            .scope(
                GateScope {
                    gate: self.clone(),
                    priority,
                },
                fut,
            )
            .await
    }

    /// Wait for a slot, then hold it until the returned permit is dropped.
    ///
    /// Cancellation-safe: dropping the returned future before it resolves
    /// removes this waiter from the queue without consuming a slot.
    pub(crate) async fn acquire(&self, priority: Priority) -> GatePermit {
        let t_wait = Instant::now();
        let rx = {
            let mut state = self.state();
            if state.in_flight < state.capacity {
                state.in_flight += 1;
                Self::record(&state);
                histogram!(INFERENCE_GATE_WAIT).record(0.0);
                return GatePermit {
                    gate: Some(Arc::clone(&self.0)),
                };
            }
            let (tx, rx) = oneshot::channel();
            state.waiters.push(Waiter {
                priority,
                seq: self.0.next_seq.fetch_add(1, Relaxed),
                tx,
            });
            Self::record(&state);
            rx
        };
        let permit = rx.await.expect(
            "a queued waiter is only ever removed by being granted a permit, and this borrow \
             keeps the gate alive until then",
        );
        histogram!(INFERENCE_GATE_WAIT).record(t_wait.elapsed().as_secs_f64());
        permit
    }

    /// Hand one slot to the highest-priority waiter, skipping waiters whose
    /// task was cancelled while queued.
    ///
    /// Returns whether a waiter took the slot.
    fn grant_to_waiter(&self, state: &mut State) -> bool {
        while let Some(waiter) = state.waiters.pop() {
            let permit = GatePermit {
                gate: Some(Arc::clone(&self.0)),
            };
            match waiter.tx.send(permit) {
                Ok(()) => return true,
                // The waiting task was dropped. Defuse the permit before it
                // falls out of scope: its `Drop` would re-enter `release`,
                // which wants the lock this function is holding.
                Err(mut orphan) => orphan.gate = None,
            }
        }
        false
    }

    /// Poison recovery rather than propagation: the gate's invariants are
    /// re-established on every operation, so a panic elsewhere in the process
    /// must not wedge every subsequent request behind a poisoned lock.
    fn state(&self) -> MutexGuard<'_, State> {
        self.0.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn record(state: &State) {
        // `unlimited()` is `u16::MAX`, which as an f64 gauge is meaningless.
        // Report it as 0 so a dashboard dividing in-flight by it gets no data
        // rather than a saturation of ~0 that reads as a starved endpoint.
        let capacity = if state.capacity == u16::MAX {
            0.0
        } else {
            state.capacity as f64
        };
        gauge!(INFERENCE_GATE_CAPACITY).set(capacity);
        gauge!(INFERENCE_IN_FLIGHT).set(state.in_flight as f64);
        gauge!(INFERENCE_GATE_QUEUED).set(state.waiters.len() as f64);
    }
}

/// A slot at the inference endpoint, released on drop.
#[derive(Debug)]
pub(crate) struct GatePermit {
    /// `None` once released, which is what makes `Drop` idempotent and lets
    /// [`InferenceGate::grant_to_waiter`] discard a permit it could not hand
    /// over without recursing into the lock.
    gate: Option<Arc<GateInner>>,
}

impl Drop for GatePermit {
    fn drop(&mut self) {
        let Some(inner) = self.gate.take() else {
            return;
        };
        let gate = InferenceGate(inner);
        let mut state = gate.state();
        // Pass the slot straight to a waiter where possible: releasing and
        // re-acquiring would let a newly arrived low-priority request win the
        // race against a repair that has been queued for a while.
        //
        // Only if we are still within capacity, though — after a shrink the
        // surplus has to actually drain.
        if state.in_flight > state.capacity || !gate.grant_to_waiter(&mut state) {
            state.in_flight -= 1;
        }
        InferenceGate::record(&state);
    }
}

#[cfg(test)]
mod tests {
    use std::{
        pin::Pin,
        sync::atomic::AtomicUsize,
        task::{
            Context,
            Poll,
            Waker,
        },
        time::Duration,
    };

    use super::*;

    /// Poll `fut` exactly once. `futures_util`'s `poll!` needs its
    /// `async-await` feature, which this crate does not enable.
    fn poll_once<F: Future>(fut: Pin<&mut F>) -> Poll<F::Output> {
        fut.poll(&mut Context::from_waker(Waker::noop()))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn capacity_bounds_concurrent_permits() {
        let gate = InferenceGate::new(2);
        let a = gate.acquire(Priority::FIRST_ATTEMPT).await;
        let _b = gate.acquire(Priority::FIRST_ATTEMPT).await;
        assert_eq!(gate.in_flight(), 2);

        let mut third = Box::pin(gate.acquire(Priority::FIRST_ATTEMPT));
        assert!(
            poll_once(third.as_mut()).is_pending(),
            "the third request must wait for a slot"
        );
        assert_eq!(gate.queued(), 1);

        drop(a);
        let _c = third.await;
        assert_eq!(gate.in_flight(), 2, "the slot was transferred, not doubled");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn higher_priority_waiters_are_served_first() {
        let gate = InferenceGate::new(1);
        let held = gate.acquire(Priority::FIRST_ATTEMPT).await;

        let order = Arc::new(Mutex::new(Vec::new()));
        // Queued fresh-first, so plain FIFO would produce [1, 1, 4, 2].
        let mut tasks = Vec::new();
        for priority in [1_usize, 1, 4, 2] {
            let gate = gate.clone();
            let order = Arc::clone(&order);
            tasks.push(tokio::spawn(async move {
                let _permit = gate.acquire(Priority::attempt(priority)).await;
                order.lock().expect("no panics in this test").push(priority);
            }));
        }
        // Let every task reach the queue before the slot frees up.
        while gate.queued() < 4 {
            tokio::task::yield_now().await;
        }

        drop(held);
        for task in tasks {
            task.await.expect("no panics in this test");
        }

        assert_eq!(
            *order.lock().expect("no panics in this test"),
            vec![4, 2, 1, 1],
            "repairs first, FIFO within a priority"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn a_cancelled_waiter_does_not_consume_a_slot() {
        let gate = InferenceGate::new(1);
        let held = gate.acquire(Priority::FIRST_ATTEMPT).await;

        let mut abandoned = Box::pin(gate.acquire(Priority::attempt(9)));
        assert!(poll_once(abandoned.as_mut()).is_pending());
        assert_eq!(gate.queued(), 1);
        // Highest priority, so the released slot is offered to it first.
        drop(abandoned);

        let waiter = tokio::spawn({
            let gate = gate.clone();
            async move { gate.acquire(Priority::FIRST_ATTEMPT).await }
        });
        while gate.queued() < 1 {
            tokio::task::yield_now().await;
        }
        drop(held);

        let permit = tokio::time::timeout(Duration::from_secs(5), waiter)
            .await
            .expect("the slot must skip the cancelled waiter, not be lost to it")
            .expect("no panics in this test");
        assert_eq!(gate.in_flight(), 1);
        drop(permit);
        assert_eq!(gate.in_flight(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn raising_capacity_wakes_waiters_and_lowering_it_drains() {
        let gate = InferenceGate::new(1);
        let held = gate.acquire(Priority::FIRST_ATTEMPT).await;
        let second = tokio::spawn({
            let gate = gate.clone();
            async move { gate.acquire(Priority::FIRST_ATTEMPT).await }
        });
        while gate.queued() < 1 {
            tokio::task::yield_now().await;
        }

        gate.set_capacity(2);
        let second = second.await.expect("no panics in this test");
        assert_eq!(gate.in_flight(), 2);

        // Shrinking below `in_flight` must not admit anyone on release.
        gate.set_capacity(1);
        let blocked = tokio::spawn({
            let gate = gate.clone();
            async move { gate.acquire(Priority::FIRST_ATTEMPT).await }
        });
        while gate.queued() < 1 {
            tokio::task::yield_now().await;
        }
        drop(held);
        assert_eq!(gate.in_flight(), 1, "the surplus slot is not handed on");
        assert_eq!(gate.queued(), 1);

        drop(second);
        let admitted = tokio::time::timeout(Duration::from_secs(5), blocked)
            .await
            .expect("back under capacity, so the waiter is admitted")
            .expect("no panics in this test");
        assert_eq!(gate.in_flight(), 1);
        drop(admitted);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_limit_holds_under_contention() {
        let gate = InferenceGate::new(4);
        let peak = Arc::new(AtomicUsize::new(0));
        let live = Arc::new(AtomicUsize::new(0));

        let mut tasks = Vec::new();
        for attempt in 0..64_usize {
            let gate = gate.clone();
            let peak = Arc::clone(&peak);
            let live = Arc::clone(&live);
            tasks.push(tokio::spawn(async move {
                let _permit = gate.acquire(Priority::attempt(attempt % 5 + 1)).await;
                let now = live.fetch_add(1, Relaxed) + 1;
                peak.fetch_max(now, Relaxed);
                tokio::task::yield_now().await;
                live.fetch_sub(1, Relaxed);
            }));
        }
        for task in tasks {
            task.await.expect("no panics in this test");
        }

        assert_eq!(peak.load(Relaxed), 4, "the capacity is never exceeded");
        assert_eq!(gate.in_flight(), 0, "every slot is returned");
        assert_eq!(gate.queued(), 0);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn requests_outside_a_scope_are_ungated() {
        assert!(GateScope::current().is_none());
        let gate = InferenceGate::new(1);
        gate.scope(Priority::attempt(3), async {
            let scope = GateScope::current().expect("inside a scope");
            assert_eq!(scope.priority, Priority::attempt(3));
            let _permit = scope.acquire().await;
            assert_eq!(gate.in_flight(), 1);
        })
        .await;
        assert!(GateScope::current().is_none());
    }
}
