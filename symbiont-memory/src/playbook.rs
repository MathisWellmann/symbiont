// SPDX-License-Identifier: MPL-2.0
//! The playbook: itemized lessons with helpful and harmful counters.
//!
//! This is the context representation of ACE, *Agentic Context Engineering*
//! (Zhang et al., [arXiv:2510.04618](https://arxiv.org/abs/2510.04618),
//! reference code <https://github.com/ace-agent/ace>). A lesson is a bullet:
//! an id, two counters and one line of content, grouped into sections. The
//! model never rewrites the playbook. It proposes deltas (new bullets, tags
//! on shown bullets) and this module merges them with plain, deterministic
//! logic. That is what ACE credits for avoiding *context collapse*, where
//! iterative rewriting erodes detail, and *brevity bias*, where summaries drop
//! the specific insights.
//!
//! The "grow-and-refine" step of ACE appends new bullets, updates counters in
//! place and removes redundant ones. ACE compares bullets by embedding; this
//! crate has no embedding model, so it compares word sets (Jaccard). It also
//! removes bullets that were tagged harmful clearly more often than helpful,
//! which the paper names as the first line of defence against noisy
//! reflections.

use std::{
    collections::BTreeSet,
    fmt::Write as _,
};

use getset::{
    CopyGetters,
    Getters,
};
use serde::{
    Deserialize,
    Serialize,
};

use crate::{
    EntrySeq,
    IslandId,
    Scope,
    estimate_tokens,
    render::EXPECT_WRITE,
};

/// A playbook section. The set is ACE's default playbook layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Section {
    /// Approaches that worked and why.
    Strategies,
    /// Formulas and calculations the problem needs.
    Formulas,
    /// Reusable code patterns.
    CodeSnippets,
    /// Mistakes that cost score or failed to compile.
    Mistakes,
    /// General problem-solving heuristics.
    Heuristics,
    /// Signals in the evaluation report and what they indicate.
    ContextClues,
    /// Anything else.
    Others,
}

impl Section {
    /// Every section, in render order.
    pub const ALL: [Self; 7] = [
        Self::Strategies,
        Self::Formulas,
        Self::CodeSnippets,
        Self::Mistakes,
        Self::Heuristics,
        Self::ContextClues,
        Self::Others,
    ];

    /// The name the model uses for the section in an operation.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::Strategies => "strategies_and_insights",
            Self::Formulas => "formulas_and_calculations",
            Self::CodeSnippets => "code_snippets_and_templates",
            Self::Mistakes => "common_mistakes_to_avoid",
            Self::Heuristics => "problem_solving_heuristics",
            Self::ContextClues => "context_clues_and_indicators",
            Self::Others => "others",
        }
    }

    /// The prefix of the ids of bullets in this section, as in `str-00003`.
    #[must_use]
    pub fn prefix(self) -> &'static str {
        match self {
            Self::Strategies => "str",
            Self::Formulas => "calc",
            Self::CodeSnippets => "code",
            Self::Mistakes => "err",
            Self::Heuristics => "heur",
            Self::ContextClues => "ctx",
            Self::Others => "misc",
        }
    }

    fn header(self) -> &'static str {
        match self {
            Self::Strategies => "STRATEGIES & INSIGHTS",
            Self::Formulas => "FORMULAS & CALCULATIONS",
            Self::CodeSnippets => "CODE SNIPPETS & TEMPLATES",
            Self::Mistakes => "COMMON MISTAKES TO AVOID",
            Self::Heuristics => "PROBLEM-SOLVING HEURISTICS",
            Self::ContextClues => "CONTEXT CLUES & INDICATORS",
            Self::Others => "OTHERS",
        }
    }

    /// The section a model named, forgiving case, `&` and punctuation; a
    /// leading word is enough (`"mistakes"`, `"Strategies"`). An unknown name
    /// maps to [`Section::Others`], as a lesson in the wrong drawer is still
    /// worth keeping.
    #[must_use]
    pub fn parse(name: &str) -> Self {
        let wanted = slugify(name);
        if wanted.is_empty() {
            return Self::Others;
        }
        Self::ALL
            .into_iter()
            .find(|s| {
                let slug = s.slug();
                slug == wanted
                    || slug.starts_with(&wanted)
                    || slug.split('_').any(|word| word == wanted)
                    || slugify(s.header()) == wanted
            })
            .unwrap_or(Self::Others)
    }
}

fn slugify(name: &str) -> String {
    let mut slug = String::with_capacity(name.len());
    for word in name
        .to_lowercase()
        .replace('&', " and ")
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        if !slug.is_empty() {
            slug.push('_');
        }
        slug.push_str(word);
    }
    slug
}

/// The id of a bullet, unique within one playbook and never reused. Rendered
/// with its section prefix, `str-00003`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BulletId(u32);

impl BulletId {
    /// The number.
    #[must_use]
    pub fn get(self) -> u32 {
        self.0
    }

    /// The number of an id the model wrote, `str-00003` or `[str-00003]`.
    /// Whether the prefix fits the section is for the caller to check.
    pub(crate) fn parse_number(text: &str) -> Option<u32> {
        let text = text.trim().trim_start_matches('[').trim_end_matches(']');
        let (_, number) = text.rsplit_once('-')?;
        number.parse().ok()
    }
}

/// One lesson.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Getters, CopyGetters)]
pub struct Bullet {
    /// The id.
    #[getset(get_copy = "pub")]
    id: BulletId,
    /// The section.
    #[getset(get_copy = "pub")]
    section: Section,
    /// Where it applies.
    #[getset(get_copy = "pub")]
    scope: Scope,
    /// The lesson, one line.
    #[getset(get = "pub")]
    content: String,
    /// How often a reflection tagged it helpful.
    #[getset(get_copy = "pub")]
    helpful: u32,
    /// How often a reflection tagged it harmful.
    #[getset(get_copy = "pub")]
    harmful: u32,
    /// The archive entries the lesson was drawn from.
    #[getset(get = "pub")]
    evidence: Vec<EntrySeq>,
}

impl Bullet {
    /// The rendered id, `str-00003`.
    #[must_use]
    pub fn label(&self) -> String {
        format!("{}-{:05}", self.section.prefix(), self.id.0)
    }

    /// `helpful - harmful`, the rank of the bullet when space runs out.
    #[must_use]
    pub fn net(&self) -> i64 {
        i64::from(self.helpful) - i64::from(self.harmful)
    }
}

/// A reflection's verdict on a bullet the lanes were shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tag {
    /// It led to better attempts.
    Helpful,
    /// It led to worse attempts.
    Harmful,
    /// No visible effect.
    Neutral,
}

impl Tag {
    /// The tag a model wrote; `None` for anything but the three words.
    #[must_use]
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_lowercase().as_str() {
            "helpful" => Some(Self::Helpful),
            "harmful" => Some(Self::Harmful),
            "neutral" => Some(Self::Neutral),
            _ => None,
        }
    }
}

/// The limits of a [`Playbook`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, CopyGetters)]
#[getset(get_copy = "pub")]
pub struct PlaybookConfig {
    /// The most bullets kept per scope (the global playbook, and each
    /// island). A new bullet beyond it replaces the bullet of the same scope
    /// with the lowest [`Bullet::net`], the oldest among ties.
    max_bullets: usize,
    /// The longest bullet content accepted, in bytes. A lesson is one
    /// actionable line, not an essay.
    max_bullet_bytes: usize,
    /// The token budget of the rendered playbook. Bullets with the lowest
    /// net count are left out of the prompt first.
    playbook_tokens: usize,
    /// A bullet is removed once `harmful >= helpful + prune_margin`.
    prune_margin: u32,
    /// Two bullets whose word sets overlap at least this much (Jaccard, in
    /// percent) are duplicates; the newer one is not added.
    duplicate_percent: u8,
    /// Whether a new bullet must cite at least one reviewed attempt. A
    /// lesson that points at no measured result is how a hallucinated rule
    /// gets into the playbook.
    require_evidence: bool,
}

impl PlaybookConfig {
    /// A configuration with the given limits. `duplicate_percent` is capped
    /// at 100 and `prune_margin` raised to 1: a margin of 0 would remove a
    /// bullet that nobody tagged.
    #[must_use]
    pub fn new(
        max_bullets: usize,
        max_bullet_bytes: usize,
        playbook_tokens: usize,
        prune_margin: u32,
        duplicate_percent: u8,
        require_evidence: bool,
    ) -> Self {
        Self {
            max_bullets: max_bullets.max(1),
            max_bullet_bytes,
            playbook_tokens,
            prune_margin: prune_margin.max(1),
            duplicate_percent: duplicate_percent.min(100),
            require_evidence,
        }
    }
}

impl Default for PlaybookConfig {
    fn default() -> Self {
        Self::new(64, 400, 2_000, 3, 80, true)
    }
}

/// Why a proposed bullet was not added.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Rejection {
    /// The content was empty.
    #[error("empty content")]
    Empty,
    /// The content exceeded [`PlaybookConfig::max_bullet_bytes`].
    #[error("content of {0} bytes is too long")]
    TooLong(usize),
    /// The bullet cited no reviewed attempt, and the configuration requires
    /// evidence.
    #[error("no evidence among the reviewed attempts")]
    NoEvidence,
    /// An existing bullet already says the same.
    #[error("duplicate of {0}")]
    Duplicate(String),
}

/// The lessons learned so far.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Playbook {
    /// Ascending by id.
    bullets: Vec<Bullet>,
    next_id: u32,
    config: PlaybookConfig,
}

impl Playbook {
    /// An empty playbook.
    #[must_use]
    pub fn new(config: PlaybookConfig) -> Self {
        Self {
            bullets: Vec::new(),
            next_id: 1,
            config,
        }
    }

    /// Add a lesson, unless it is empty, too long, without evidence or a
    /// duplicate of a bullet visible in `scope`. A full scope makes room by
    /// dropping its weakest bullet.
    ///
    /// # Errors
    ///
    /// Returns the [`Rejection`] reason.
    pub fn add(
        &mut self,
        scope: Scope,
        section: Section,
        content: &str,
        evidence: Vec<EntrySeq>,
    ) -> Result<BulletId, Rejection> {
        let content = strip_label(content.trim());
        if content.is_empty() {
            return Err(Rejection::Empty);
        }
        if content.len() > self.config.max_bullet_bytes {
            return Err(Rejection::TooLong(content.len()));
        }
        if self.config.require_evidence && evidence.is_empty() {
            return Err(Rejection::NoEvidence);
        }
        if let Some(twin) = self.twin(scope, content, None) {
            return Err(Rejection::Duplicate(twin));
        }
        self.make_room(scope);

        let id = BulletId(self.next_id);
        self.next_id += 1;
        self.bullets.push(Bullet {
            id,
            section,
            scope,
            content: content.to_string(),
            helpful: 0,
            harmful: 0,
            evidence,
        });
        assert!(
            self.bullets.iter().filter(|b| b.scope == scope).count() <= self.config.max_bullets
        );
        assert!(self.bullets.windows(2).all(|w| w[0].id < w[1].id));
        Ok(id)
    }

    /// The label of a bullet that a new bullet in `scope` would duplicate:
    /// a global bullet, or one of the same island. `skip` excludes the bullet
    /// being moved.
    fn twin(&self, scope: Scope, content: &str, skip: Option<BulletId>) -> Option<String> {
        let words = word_set(content);
        self.bullets
            .iter()
            .filter(|b| Some(b.id) != skip)
            .filter(|b| b.scope == Scope::Global || b.scope == scope)
            .find(|b| {
                jaccard_percent(&words, &word_set(&b.content)) >= self.config.duplicate_percent
            })
            .map(Bullet::label)
    }

    /// Drop the weakest bullet of `scope` if the scope is full.
    fn make_room(&mut self, scope: Scope) {
        let in_scope = self.bullets.iter().filter(|b| b.scope == scope).count();
        if in_scope < self.config.max_bullets {
            return;
        }
        let (idx, _) = self
            .bullets
            .iter()
            .enumerate()
            .filter(|(_, b)| b.scope == scope)
            .min_by_key(|(_, b)| (b.net(), b.id))
            .expect("max_bullets is positive, so a full scope is not empty");
        self.bullets.remove(idx);
    }

    /// Move an island bullet to the global playbook, keeping its id and
    /// counters.
    ///
    /// # Errors
    ///
    /// [`Rejection::Duplicate`] if a global bullet already says the same.
    pub(crate) fn promote(&mut self, id: BulletId) -> Result<(), Rejection> {
        let bullet = self.get(id).expect("promote is called with known ids");
        assert_ne!(bullet.scope, Scope::Global);
        let content = bullet.content.clone();
        if let Some(twin) = self.twin(Scope::Global, &content, Some(id)) {
            return Err(Rejection::Duplicate(twin));
        }
        self.make_room(Scope::Global);
        let bullet = self
            .bullets
            .iter_mut()
            .find(|b| b.id == id)
            .expect("known, and making global room keeps island bullets");
        bullet.scope = Scope::Global;
        Ok(())
    }

    /// Remove every bullet of `island`.
    pub(crate) fn clear_island(&mut self, island: IslandId) {
        self.bullets.retain(|b| b.scope != Scope::Island(island));
    }

    /// Count a reflection's tag. Returns `false` for an unknown id.
    pub fn tag(&mut self, id: BulletId, tag: Tag) -> bool {
        let Some(bullet) = self.bullets.iter_mut().find(|b| b.id == id) else {
            return false;
        };
        match tag {
            Tag::Helpful => bullet.helpful += 1,
            Tag::Harmful => bullet.harmful += 1,
            Tag::Neutral => {}
        }
        true
    }

    /// Remove the bullets tagged harmful clearly more often than helpful,
    /// see [`PlaybookConfig::prune_margin`]. Returns their ids.
    pub fn prune(&mut self) -> Vec<BulletId> {
        let margin = self.config.prune_margin;
        let pruned = Vec::from_iter(
            self.bullets
                .iter()
                .filter(|b| b.harmful >= b.helpful.saturating_add(margin))
                .map(|b| b.id),
        );
        self.bullets.retain(|b| !pruned.contains(&b.id));
        pruned
    }

    /// The bullet with `id`.
    #[must_use]
    pub fn get(&self, id: BulletId) -> Option<&Bullet> {
        self.bullets.iter().find(|b| b.id == id)
    }

    /// The bullet a model referred to as `str-00003`: the number must exist
    /// and the prefix must be its section's.
    #[must_use]
    pub fn resolve(&self, label: &str) -> Option<BulletId> {
        let number = BulletId::parse_number(label)?;
        let bullet = self.get(BulletId(number))?;
        let label = label.trim().trim_start_matches('[');
        label
            .starts_with(bullet.section.prefix())
            .then_some(bullet.id)
    }

    /// Every bullet, ascending by id.
    #[must_use]
    pub fn bullets(&self) -> &[Bullet] {
        &self.bullets
    }

    /// The bullets of exactly `scope` as a prompt section, within
    /// [`PlaybookConfig::playbook_tokens`], and the bullets it shows.
    ///
    /// The lines use ACE's format, `[str-00003] helpful=2 harmful=0 ::
    /// lesson`, so the model sees how much each lesson has been confirmed.
    #[must_use]
    pub fn render(&self, scope: Scope) -> (String, Vec<BulletId>) {
        let header = match scope {
            Scope::Global => PLAYBOOK_HEADER,
            Scope::Island(_) => ISLAND_HEADER,
        };
        let lines = Vec::from_iter(
            self.bullets
                .iter()
                .filter(|b| b.scope == scope)
                .map(|b| (b, line(b))),
        );
        // Keep the strongest bullets that fit, then print in section order.
        let mut by_strength = Vec::from_iter(lines.iter());
        by_strength.sort_by(|(a, _), (b, _)| b.net().cmp(&a.net()).then(a.id.cmp(&b.id)));
        let mut used = estimate_tokens(header);
        let mut kept = Vec::with_capacity(lines.len());
        for (bullet, text) in by_strength {
            let cost = estimate_tokens(text) + SECTION_HEADER_TOKENS;
            if used + cost > self.config.playbook_tokens {
                continue;
            }
            used += cost;
            kept.push(bullet.id);
        }
        if kept.is_empty() {
            return (String::new(), kept);
        }
        kept.sort_unstable();

        let mut text = String::from(header);
        for section in Section::ALL {
            let mut in_section = lines
                .iter()
                .filter(|(b, _)| b.section == section && kept.contains(&b.id))
                .peekable();
            if in_section.peek().is_none() {
                continue;
            }
            writeln!(text, "### {}", section.header()).expect(EXPECT_WRITE);
            for (_, line) in in_section {
                text.push_str(line);
            }
            text.push('\n');
        }
        (text, kept)
    }
}

impl Playbook {
    /// Every bullet visible on `island` in ACE's line format, grouped by
    /// section, with no budget: the reviewer must see the whole playbook to
    /// avoid proposing a duplicate.
    pub(crate) fn lines(&self, island: IslandId) -> String {
        let mut text = String::new();
        for section in Section::ALL {
            let mut in_section = self
                .bullets
                .iter()
                .filter(|b| b.section == section && b.scope.visible_on(island))
                .peekable();
            if in_section.peek().is_none() {
                continue;
            }
            writeln!(text, "### {}", section.header()).expect(EXPECT_WRITE);
            for bullet in in_section {
                text.push_str(&line(bullet));
            }
        }
        text
    }
}

const PLAYBOOK_HEADER: &str = "## Playbook\n\n\
    Lessons from earlier rounds, as `[id] helpful=X harmful=Y :: lesson`. \
    The counts say how often a review confirmed or refuted a lesson.\n\n";

const ISLAND_HEADER: &str = "## Lessons of this island\n\n\
    Lessons only this sub-population has learned so far, in the same format.\n\n";

/// A generous share of a section header per bullet, so the budget holds
/// even when every kept bullet opens a section.
const SECTION_HEADER_TOKENS: usize = 8;

fn line(bullet: &Bullet) -> String {
    format!(
        "[{}] helpful={} harmful={} :: {}\n",
        bullet.label(),
        bullet.helpful,
        bullet.harmful,
        bullet.content
    )
}

/// The content without a `[id] helpful=X harmful=Y ::` prefix a model copied
/// from the rendered playbook although it was told not to.
fn strip_label(content: &str) -> &str {
    if content.starts_with('[')
        && let Some((_, rest)) = content.split_once("::")
    {
        return rest.trim();
    }
    content
}

fn word_set(text: &str) -> BTreeSet<String> {
    BTreeSet::from_iter(
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .map(str::to_string),
    )
}

fn jaccard_percent(a: &BTreeSet<String>, b: &BTreeSet<String>) -> u8 {
    let union = a.union(b).count();
    if union == 0 {
        return 100;
    }
    let shared = a.intersection(b).count();
    u8::try_from(shared * 100 / union).expect("a ratio of at most one, in percent")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seq() -> Vec<EntrySeq> {
        vec![EntrySeq::from_raw(0)]
    }

    fn playbook() -> Playbook {
        Playbook::new(PlaybookConfig::default())
    }

    #[test]
    fn section_parse_is_forgiving() {
        assert_eq!(
            Section::parse("strategies_and_insights"),
            Section::Strategies
        );
        assert_eq!(Section::parse("STRATEGIES & INSIGHTS"), Section::Strategies);
        assert_eq!(
            Section::parse("Common Mistakes to Avoid"),
            Section::Mistakes
        );
        assert_eq!(Section::parse("mistakes"), Section::Mistakes);
        assert_eq!(Section::parse("formulas"), Section::Formulas);
        assert_eq!(Section::parse("code"), Section::CodeSnippets);
        assert_eq!(Section::parse("something new"), Section::Others);
        assert_eq!(Section::parse(""), Section::Others);
    }

    #[test]
    fn add_assigns_labels_and_strips_copied_prefixes() {
        let mut pb = playbook();
        let a = pb.add(
            Scope::Global,
            Section::Mistakes,
            "Never divide by the window range.",
            seq(),
        );
        let a = a.expect("added");
        assert_eq!(pb.get(a).expect("known").label(), "err-00001");
        let b = pb
            .add(
                Scope::Global,
                Section::Strategies,
                "[str-00009] helpful=3 harmful=0 :: Cut losses at 2 ATR.",
                seq(),
            )
            .expect("added");
        assert_eq!(pb.get(b).expect("known").content(), "Cut losses at 2 ATR.");
        assert_eq!(pb.resolve("str-00002"), Some(b));
        assert_eq!(pb.resolve("[err-00001]"), Some(a));
        assert_eq!(
            pb.resolve("err-00002"),
            None,
            "prefix must match the section"
        );
        assert_eq!(pb.resolve("str-00099"), None);
        assert_eq!(pb.resolve("garbage"), None);
    }

    #[test]
    fn add_rejects_bad_bullets() {
        let mut pb = playbook();
        assert_eq!(
            pb.add(Scope::Global, Section::Others, "  ", seq()),
            Err(Rejection::Empty)
        );
        let long = "x".repeat(401);
        assert_eq!(
            pb.add(Scope::Global, Section::Others, &long, seq()),
            Err(Rejection::TooLong(401))
        );
        assert_eq!(
            pb.add(Scope::Global, Section::Others, "a lesson", Vec::new()),
            Err(Rejection::NoEvidence)
        );
        let _ = pb
            .add(
                Scope::Global,
                Section::Others,
                "Use a sieve for counting primes below n.",
                seq(),
            )
            .expect("added");
        assert_eq!(
            pb.add(
                Scope::Global,
                Section::Mistakes,
                "use a SIEVE for counting primes below n!",
                seq()
            ),
            Err(Rejection::Duplicate("misc-00001".to_string()))
        );
        let lax = PlaybookConfig::new(4, 400, 2_000, 3, 80, false);
        assert!(
            Playbook::new(lax)
                .add(Scope::Global, Section::Others, "a lesson", Vec::new())
                .is_ok()
        );
    }

    #[test]
    fn tags_count_and_prune_removes_refuted_bullets() {
        let mut pb = playbook();
        let a = pb
            .add(Scope::Global, Section::Strategies, "alpha beta", seq())
            .expect("added");
        let b = pb
            .add(Scope::Global, Section::Strategies, "gamma delta", seq())
            .expect("added");
        assert!(pb.tag(a, Tag::Helpful));
        assert!(pb.tag(a, Tag::Neutral));
        assert!(!pb.tag(BulletId(99), Tag::Helpful));
        for _ in 0..3 {
            assert!(pb.tag(b, Tag::Harmful));
        }
        assert_eq!(pb.get(a).expect("known").helpful(), 1);
        assert_eq!(pb.prune(), vec![b]);
        assert!(pb.get(b).is_none());
        assert!(pb.prune().is_empty(), "an untagged or helpful bullet stays");
    }

    #[test]
    fn a_full_playbook_drops_its_weakest_bullet() {
        let mut pb = Playbook::new(PlaybookConfig::new(2, 400, 2_000, 3, 80, true));
        let a = pb
            .add(Scope::Global, Section::Others, "one", seq())
            .expect("added");
        let b = pb
            .add(Scope::Global, Section::Others, "two", seq())
            .expect("added");
        assert!(pb.tag(a, Tag::Helpful));
        let c = pb
            .add(Scope::Global, Section::Others, "three", seq())
            .expect("added");
        assert_eq!(
            Vec::from_iter(pb.bullets().iter().map(Bullet::id)),
            vec![a, c]
        );
        assert!(pb.get(b).is_none());
    }

    #[test]
    fn render_groups_by_section_and_respects_the_budget() {
        let mut pb = playbook();
        assert_eq!(pb.render(Scope::Global), (String::new(), Vec::new()));
        let m = pb
            .add(
                Scope::Global,
                Section::Mistakes,
                "Do not index past the window.",
                seq(),
            )
            .expect("added");
        let s = pb
            .add(
                Scope::Global,
                Section::Strategies,
                "Trade only above the fee.",
                seq(),
            )
            .expect("added");
        let (text, shown) = pb.render(Scope::Global);
        assert_eq!(shown, vec![m, s]);
        let strategies = text.find("### STRATEGIES & INSIGHTS").expect("present");
        let mistakes = text.find("### COMMON MISTAKES TO AVOID").expect("present");
        assert!(strategies < mistakes, "sections render in a fixed order");
        assert!(text.contains("[str-00002] helpful=0 harmful=0 :: Trade only above the fee.\n"));

        assert!(pb.tag(m, Tag::Helpful));
        let one = estimate_tokens(PLAYBOOK_HEADER) + SECTION_HEADER_TOKENS + 20;
        let tight = Playbook {
            config: PlaybookConfig::new(64, 400, one, 3, 80, true),
            ..pb
        };
        let (text, shown) = tight.render(Scope::Global);
        assert_eq!(shown, vec![m], "the stronger bullet wins the space");
        assert!(!text.contains("fee"));
    }

    #[test]
    fn bullet_id_numbers_parse() {
        assert_eq!(BulletId::parse_number("str-00012"), Some(12));
        assert_eq!(BulletId::parse_number(" [code-7] "), Some(7));
        assert_eq!(BulletId::parse_number("12"), None);
        assert_eq!(BulletId::parse_number("str-x"), None);
    }

    #[test]
    fn tags_parse() {
        assert_eq!(Tag::parse(" Helpful "), Some(Tag::Helpful));
        assert_eq!(Tag::parse("HARMFUL"), Some(Tag::Harmful));
        assert_eq!(Tag::parse("neutral"), Some(Tag::Neutral));
        assert_eq!(Tag::parse("great"), None);
    }

    #[test]
    fn scopes_separate_bullets() {
        let mut pb = playbook();
        let (a, b) = (IslandId::new(0), IslandId::new(1));
        let global = pb
            .add(Scope::Global, Section::Others, "Shared lesson here.", seq())
            .expect("added");
        // An island bullet duplicating a global one is rejected ...
        assert_eq!(
            pb.add(
                Scope::Island(a),
                Section::Others,
                "shared lesson here",
                seq()
            ),
            Err(Rejection::Duplicate("misc-00001".to_string()))
        );
        let on_a = pb
            .add(
                Scope::Island(a),
                Section::Others,
                "Lesson of island a.",
                seq(),
            )
            .expect("added");
        // ... but two islands may learn the same thing independently.
        let on_b = pb
            .add(
                Scope::Island(b),
                Section::Others,
                "Lesson of island a.",
                seq(),
            )
            .expect("added");
        assert_eq!(pb.render(Scope::Global).1, vec![global]);
        assert_eq!(pb.render(Scope::Island(a)).1, vec![on_a]);
        assert!(
            pb.render(Scope::Island(a))
                .0
                .starts_with("## Lessons of this island")
        );
        assert!(pb.lines(a).contains("Shared lesson here."));
        assert!(!pb.lines(a).contains("misc-00003"));

        assert!(pb.promote(on_a).is_ok());
        assert_eq!(pb.get(on_a).expect("kept").scope(), Scope::Global);
        assert_eq!(
            pb.promote(on_b),
            Err(Rejection::Duplicate("misc-00002".to_string()))
        );
        pb.clear_island(b);
        assert!(pb.get(on_b).is_none());
        assert_eq!(pb.bullets().len(), 2);
    }
}
