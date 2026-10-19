// SPDX-License-Identifier: MPL-2.0
//! The reflection step: one model call per round that reviews the evaluated
//! attempts and proposes playbook deltas.
//!
//! ACE runs a Reflector (diagnose an attempt, tag the bullets it used) and a
//! Curator (turn the reflection into new bullets) as two calls. Here both
//! roles share one call per batch of attempts, because the evolve loop
//! already spends many calls per round and the scores make the diagnosis
//! cheap: the model compares measured numbers instead of guessing at ground
//! truth. The output is the same pair of deltas ACE produces, bullet tags and
//! `ADD` operations, and like the reference implementation only `ADD` is
//! supported.
//!
//! The step is sans-IO. [`crate::Memory::reflection_request`] hands out the
//! prompts, the host sends them to whatever model it likes, and
//! [`crate::Memory::apply_reflection`] merges the answer. The memory never
//! awaits, so a host keeps it on the one task that owns its population.

use std::fmt::Write as _;

use getset::{
    CopyGetters,
    Getters,
};
use serde::{
    Deserialize,
    Serialize,
};
use serde_json::Value;

use crate::{
    Archive,
    BulletId,
    EntrySeq,
    Failures,
    IslandId,
    Memory,
    Playbook,
    Rejection,
    Scope,
    Section,
    Tag,
    render::{
        EXPECT_WRITE,
        truncate_lines,
    },
};

/// When and how much the reflection step reviews.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct ReflectionConfig {
    /// [`crate::Memory::reflection_due`] turns true once this many new
    /// attempts await review. Set it to the number of lanes for one
    /// reflection per round.
    reflect_every: usize,
    /// The most attempts one request reviews, oldest first. The rest wait
    /// for the next request.
    max_reviewed: usize,
    /// The longest source quoted per attempt, in bytes.
    max_source_bytes: usize,
    /// How many of the best archive entries the request shows for
    /// reference, so the reviewer can tell a good attempt from a lucky one.
    reference_elites: usize,
}

impl ReflectionConfig {
    /// A configuration with the given limits; `reflect_every` and
    /// `max_reviewed` are raised to 1.
    #[must_use]
    pub fn new(
        reflect_every: usize,
        max_reviewed: usize,
        max_source_bytes: usize,
        reference_elites: usize,
    ) -> Self {
        Self {
            reflect_every: reflect_every.max(1),
            max_reviewed: max_reviewed.max(1),
            max_source_bytes,
            reference_elites,
        }
    }
}

impl Default for ReflectionConfig {
    fn default() -> Self {
        Self::new(8, 16, 3_000, 2)
    }
}

/// An attempt that awaits review, with the bullets its lane was shown.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Pending {
    seq: EntrySeq,
    island: IslandId,
    bullets: Vec<BulletId>,
}

impl Pending {
    pub(crate) fn new(seq: EntrySeq, island: IslandId, bullets: Vec<BulletId>) -> Self {
        Self {
            seq,
            island,
            bullets,
        }
    }

    pub(crate) fn seq(&self) -> EntrySeq {
        self.seq
    }

    pub(crate) fn island(&self) -> IslandId {
        self.island
    }
}

/// The id of a [`ReflectionRequest`], so an answer is applied at most once.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct RequestId(u32);

impl RequestId {
    pub(crate) fn new(n: u32) -> Self {
        Self(n)
    }

    /// The number.
    #[must_use]
    pub fn get(self) -> u32 {
        self.0
    }
}

/// The prompts of one reflection call. Send [`Self::system_prompt`] as the
/// system message and [`Self::user_prompt`] as the user message, then pass
/// the answer to [`crate::Memory::apply_reflection`].
#[derive(Debug, Clone, Getters, CopyGetters)]
#[must_use]
pub struct ReflectionRequest {
    /// The id.
    #[getset(get_copy = "pub")]
    id: RequestId,
    /// The island whose attempts it reviews.
    #[getset(get_copy = "pub")]
    island: IslandId,
    /// Where the bullets it adds go: the island's own lessons when there
    /// are several islands, else the global playbook.
    #[getset(get_copy = "pub")]
    scope: Scope,
    /// The system prompt.
    #[getset(get = "pub")]
    system_prompt: String,
    /// The user prompt.
    #[getset(get = "pub")]
    user_prompt: String,
    pending: Vec<Pending>,
    shown: Vec<BulletId>,
}

impl ReflectionRequest {
    /// The attempts this request reviews.
    pub fn reviewed(&self) -> impl Iterator<Item = EntrySeq> + '_ {
        self.pending.iter().map(|p| p.seq)
    }

    pub(crate) fn into_pending(self) -> Vec<Pending> {
        self.pending
    }
}

/// Why an answer could not be applied.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ReflectionError {
    /// The request was already applied or abandoned.
    #[error("reflection request {0:?} is not outstanding")]
    NotOutstanding(RequestId),
    /// The answer holds no JSON object with `operations` or `bullet_tags`.
    /// The request stays outstanding: apply another answer, or abandon it.
    #[error("the answer holds no reflection JSON object")]
    NoJson,
}

/// What [`crate::Memory::apply_reflection`] changed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Getters, CopyGetters)]
#[must_use]
pub struct CurationReport {
    /// The reviewer's reasoning, for logs.
    #[getset(get = "pub")]
    reasoning: String,
    /// The reviewer's key insight, for logs.
    #[getset(get = "pub")]
    key_insight: String,
    /// Bullets tagged helpful.
    #[getset(get_copy = "pub")]
    helpful: usize,
    /// Bullets tagged harmful.
    #[getset(get_copy = "pub")]
    harmful: usize,
    /// Bullets tagged neutral.
    #[getset(get_copy = "pub")]
    neutral: usize,
    /// Tags on bullets the reviewed lanes were not shown, unknown ids,
    /// repeated ids and unknown tag words. ACE only counts tags on bullets
    /// the attempt used.
    #[getset(get_copy = "pub")]
    ignored_tags: usize,
    /// The new bullets.
    #[getset(get = "pub")]
    added: Vec<BulletId>,
    /// The proposed bullets that were not added, with the reason.
    #[getset(get = "pub")]
    rejected: Vec<(String, Rejection)>,
    /// Operations other than `ADD`.
    #[getset(get_copy = "pub")]
    unsupported: usize,
    /// Bullets removed because they were refuted, see
    /// [`crate::PlaybookConfig::prune_margin`].
    #[getset(get = "pub")]
    pruned: Vec<BulletId>,
}

/// The JSON object the model answers with. Every field is optional so a
/// partial answer still yields what it has.
#[derive(Debug, Default, Deserialize)]
struct Answer {
    #[serde(default)]
    reasoning: Value,
    #[serde(default, alias = "key_insights")]
    key_insight: Value,
    #[serde(default)]
    bullet_tags: Vec<TagAnswer>,
    #[serde(default)]
    operations: Vec<Operation>,
}

#[derive(Debug, Deserialize)]
struct TagAnswer {
    #[serde(alias = "bullet_id", alias = "bullet")]
    id: String,
    tag: String,
}

#[derive(Debug, Deserialize)]
struct Operation {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    section: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    evidence: Vec<Value>,
}

/// The system prompt of every [`ReflectionRequest`]. It never changes, so a
/// host can build its reflection agent with it as the preamble up front.
pub const REFLECTION_SYSTEM_PROMPT: &str = "\
You are an expert analyst and a curator of knowledge. A code-evolution harness asks a model to \
write a Rust function for an optimization problem, compiles it and scores it. You review a batch \
of these evaluated attempts and maintain the playbook of lessons that every future attempt is \
shown.

Instructions:
- Compare each attempt with its parent and with the best attempts. Find why the score went up or \
down: the root cause, not the surface.
- Ground every judgement in the measured scores and reports. A small score difference can be \
noise.
- Tag every bullet listed under \"Bullets to tag\": \"helpful\" if attempts that were shown it did \
better because of it, \"harmful\" if it misled them, \"neutral\" if it had no visible effect.
- Propose only NEW lessons that are missing from the playbook. If similar advice exists, add only \
what complements it. Do not restate or rewrite the playbook.
- A lesson is one specific, actionable line: a strategy, formula, code pattern or mistake to \
avoid. Future attempts will not see these reports, so a lesson must stand on its own.
- Cite the attempts a lesson is drawn from as evidence, by number (12 for #12).
- Quality over quantity. If nothing new was learned, return an empty operations list.

Sections: strategies_and_insights, formulas_and_calculations, code_snippets_and_templates, \
common_mistakes_to_avoid, problem_solving_heuristics, context_clues_and_indicators, others.

Respond with one JSON object only, no markdown:
{
  \"reasoning\": \"[your analysis]\",
  \"key_insight\": \"[the most important thing to remember]\",
  \"bullet_tags\": [{\"id\": \"str-00001\", \"tag\": \"helpful\"}],
  \"operations\": [{\"type\": \"ADD\", \"section\": \"common_mistakes_to_avoid\", \
\"content\": \"[the lesson, without an id]\", \"evidence\": [12]}]
}";

/// Everything the user prompt is built from.
struct PromptInput<'a, I> {
    island: IslandId,
    task: &'a str,
    archive: &'a Archive<I>,
    playbook: &'a Playbook,
    failures: &'a Failures,
    pending: &'a [Pending],
    config: &'a ReflectionConfig,
}

/// Build the request that reviews `pending`, all from `island`, against the
/// state of `memory`. Bullets it adds go to `scope`.
pub(crate) fn request<I: PartialEq + Clone>(
    id: RequestId,
    memory: &Memory<I>,
    island: IslandId,
    scope: Scope,
    task: &str,
    pending: Vec<Pending>,
) -> ReflectionRequest {
    assert!(!pending.is_empty());
    assert!(pending.iter().all(|p| p.island == island));
    let input = PromptInput {
        island,
        task,
        archive: &memory.archive,
        playbook: &memory.playbook,
        failures: &memory.failures,
        pending: &pending,
        config: &memory.config.reflection,
    };
    let mut shown = Vec::from_iter(pending.iter().flat_map(|p| p.bullets.iter().copied()));
    shown.sort_unstable();
    shown.dedup();
    shown.retain(|b| input.playbook.get(*b).is_some());

    let mut user = String::new();
    writeln!(user, "## Problem\n\n{}\n", input.task.trim()).expect(EXPECT_WRITE);
    let lines = input.playbook.lines(island);
    writeln!(
        user,
        "## Current playbook ({} bullets)\n\n{}\n",
        input.playbook.bullets().len(),
        if lines.is_empty() {
            "(empty)\n"
        } else {
            &lines
        }
    )
    .expect(EXPECT_WRITE);
    user.push_str(&input.failures.render());
    push_reference(&mut user, &input);
    push_reviewed(&mut user, &input);
    user.push_str("## Bullets to tag\n\n");
    if shown.is_empty() {
        user.push_str("(none: the reviewed attempts were shown no bullets)\n");
    }
    for bullet in &shown {
        let bullet = input.playbook.get(*bullet).expect("retained above");
        writeln!(user, "[{}] :: {}", bullet.label(), bullet.content()).expect(EXPECT_WRITE);
    }

    ReflectionRequest {
        id,
        island,
        scope,
        system_prompt: REFLECTION_SYSTEM_PROMPT.to_string(),
        user_prompt: user,
        pending,
        shown,
    }
}

fn push_reference<I: PartialEq + Clone>(user: &mut String, input: &PromptInput<'_, I>) {
    let reviewed = Vec::from_iter(input.pending.iter().map(|p| p.seq));
    let elites = Vec::from_iter(
        input
            .archive
            .ranked(input.island)
            .into_iter()
            .filter(|e| !reviewed.contains(&e.seq()))
            .take(input.config.reference_elites),
    );
    if elites.is_empty() {
        return;
    }
    user.push_str("## Best attempts so far, for reference\n\n");
    for (rank, entry) in elites.into_iter().enumerate() {
        writeln!(
            user,
            "### {} (score {})\n{}",
            entry.seq(),
            entry.score().get(),
            entry.report().trim()
        )
        .expect(EXPECT_WRITE);
        if rank == 0 {
            let source = truncate_lines(entry.source().trim_end(), input.config.max_source_bytes);
            writeln!(user, "```rust\n{source}\n```").expect(EXPECT_WRITE);
        }
        user.push('\n');
    }
}

fn push_reviewed<I: PartialEq + Clone>(user: &mut String, input: &PromptInput<'_, I>) {
    user.push_str("## Attempts to review\n\n");
    for pending in input.pending {
        let entry = input
            .archive
            .by_seq(pending.seq)
            .expect("requests only hold attempts still in the archive");
        write!(user, "### {} (score {}", entry.seq(), entry.score().get()).expect(EXPECT_WRITE);
        for parent in entry.parents() {
            if let Some(parent) = input.archive.get(parent) {
                let delta = entry.score().get() - parent.score().get();
                write!(
                    user,
                    "; parent {} scored {}, delta {delta:+}",
                    parent.seq(),
                    parent.score().get()
                )
                .expect(EXPECT_WRITE);
            }
        }
        user.push_str(")\n");
        let labels = Vec::from_iter(
            pending
                .bullets
                .iter()
                .filter_map(|b| input.playbook.get(*b))
                .map(crate::Bullet::label),
        );
        let labels = if labels.is_empty() {
            "none".to_string()
        } else {
            labels.join(", ")
        };
        writeln!(user, "Bullets shown: {labels}\n{}", entry.report().trim()).expect(EXPECT_WRITE);
        let source = truncate_lines(entry.source().trim_end(), input.config.max_source_bytes);
        writeln!(user, "```rust\n{source}\n```\n").expect(EXPECT_WRITE);
    }
}

/// The last JSON object in `text` that has `operations` or `bullet_tags`.
///
/// Models wrap JSON in markdown fences, put prose around it, or think out
/// loud with draft objects first. The final answer comes last, so the last
/// object that parses wins.
fn parse_answer(text: &str) -> Option<Answer> {
    let mut found = None;
    for (start, _) in text.match_indices('{') {
        let mut stream = serde_json::Deserializer::from_str(&text[start..]).into_iter::<Value>();
        let Some(Ok(value)) = stream.next() else {
            continue;
        };
        let Value::Object(map) = &value else {
            continue;
        };
        if map.contains_key("operations") || map.contains_key("bullet_tags") {
            found = Some(value);
        }
    }
    serde_json::from_value(found?).ok()
}

/// Merge an answer into the playbook: tags first, then pruning, then the new
/// bullets, so a new bullet is checked for duplicates against the bullets
/// that survived.
///
/// # Errors
///
/// [`ReflectionError::NoJson`] if the answer holds no reflection object.
pub(crate) fn apply<I: PartialEq + Clone>(
    request: &ReflectionRequest,
    answer: &str,
    playbook: &mut Playbook,
    archive: &Archive<I>,
) -> Result<CurationReport, ReflectionError> {
    let answer = parse_answer(answer).ok_or(ReflectionError::NoJson)?;
    let mut report = CurationReport {
        reasoning: text_of(answer.reasoning),
        key_insight: text_of(answer.key_insight),
        ..CurationReport::default()
    };

    let mut tagged = Vec::with_capacity(answer.bullet_tags.len());
    for tag in answer.bullet_tags {
        let target = playbook
            .resolve(&tag.id)
            .filter(|b| request.shown.contains(b));
        let (Some(bullet), Some(verdict)) = (target, Tag::parse(&tag.tag)) else {
            report.ignored_tags += 1;
            continue;
        };
        if tagged.contains(&bullet) {
            report.ignored_tags += 1;
            continue;
        }
        tagged.push(bullet);
        assert!(playbook.tag(bullet, verdict), "resolved ids exist");
        match verdict {
            Tag::Helpful => report.helpful += 1,
            Tag::Harmful => report.harmful += 1,
            Tag::Neutral => report.neutral += 1,
        }
    }
    report.pruned = playbook.prune();

    for op in answer.operations {
        if !op.kind.trim().eq_ignore_ascii_case("add") {
            report.unsupported += 1;
            continue;
        }
        let evidence = evidence_of(&op.evidence, request, archive);
        match playbook.add(
            request.scope,
            Section::parse(&op.section),
            &op.content,
            evidence,
        ) {
            Ok(id) => report.added.push(id),
            Err(rejection) => report.rejected.push((op.content, rejection)),
        }
    }
    Ok(report)
}

fn text_of(value: Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text,
        other => other.to_string(),
    }
}

/// The cited attempts that exist: reviewed in this request, or still in the
/// archive (the reference elites). `12`, `"12"` and `"#12"` all cite #12.
fn evidence_of<I: PartialEq + Clone>(
    cited: &[Value],
    request: &ReflectionRequest,
    archive: &Archive<I>,
) -> Vec<EntrySeq> {
    let mut evidence = Vec::from_iter(
        cited
            .iter()
            .filter_map(|v| match v {
                Value::Number(n) => n.as_u64(),
                Value::String(s) => s.trim().trim_start_matches('#').parse().ok(),
                _ => None,
            })
            .map(EntrySeq::from_raw)
            .filter(|seq| request.reviewed().any(|r| r == *seq) || archive.by_seq(*seq).is_some()),
    );
    evidence.sort_unstable();
    evidence.dedup();
    evidence
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_answer_takes_the_last_reflection_object() {
        let text = "Thinking {not json} and a draft {\"operations\": []}\n\
            ```json\n{\"reasoning\": \"final\", \"operations\": [{\"type\": \"ADD\", \
            \"content\": \"x\"}]}\n```\nDone.";
        let answer = parse_answer(text).expect("parses");
        assert_eq!(text_of(answer.reasoning), "final");
        assert_eq!(answer.operations.len(), 1);
        assert_eq!(answer.operations[0].kind, "ADD");
        assert!(parse_answer("no json here").is_none());
        assert!(
            parse_answer("{\"other\": 1}").is_none(),
            "needs a reflection key"
        );
    }

    #[test]
    fn answer_fields_have_aliases() {
        let answer = parse_answer(
            "{\"key_insights\": [\"a\"], \"bullet_tags\": [{\"bullet_id\": \"str-00001\", \"tag\": \"helpful\"}]}",
        )
        .expect("parses");
        assert_eq!(text_of(answer.key_insight), "[\"a\"]");
        assert_eq!(answer.bullet_tags[0].id, "str-00001");
    }
}
