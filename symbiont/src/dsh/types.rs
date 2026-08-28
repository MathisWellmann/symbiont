use serde::{
    Deserialize,
    Serialize,
};
use typed_builder::TypedBuilder;

/// The on-disk session-format version this exporter writes. The harness
/// refuses any other value outright, before it looks at the header shape.
const SESSION_FORMAT_VERSION: u32 = 0;

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum AgentPreset {
    #[default]
    Standard,
    Code,
    Minimal,
    Cordis,
}

#[derive(Debug, Clone, Serialize, Deserialize, TypedBuilder)]
#[serde(rename_all = "camelCase")]
pub(super) struct DshHeader<'a> {
    #[builder(default = "session")]
    r#type: &'a str,
    #[builder(default = SESSION_FORMAT_VERSION)]
    version: u32,
    id: String,
    created_at: u64,
    delegation_depth: u32,
    agent_preset: AgentPreset,
    #[serde(skip_serializing_if = "Option::is_none")]
    cwd: Option<&'a str>,
}
