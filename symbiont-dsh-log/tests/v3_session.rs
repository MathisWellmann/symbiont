// SPDX-License-Identifier: MPL-2.0
//! A persona-shaped v3 session written through the public API. The artifact is
//! kept at `target/debug/v3-sample/` for the out-of-process check against the
//! dsh session-format catalog (the authoritative v3 validator).
#![cfg(feature = "zstd")]
#![allow(unused_crate_dependencies)] // the zstd dep is what this test exercises

use serde::{
    Deserialize,
    Serialize,
};
use symbiont_dsh_log::{
    AssistantMessageData,
    ContentBlock,
    ContextForm,
    Event,
    LlmCallConfig,
    LogLine,
    Message,
    MessageSource,
    RequestHeaderData,
    RequestHeaderReason,
    Role,
    SessionHeaderLine,
    SessionTitleData,
    StepStartData,
    SystemMessageData,
    TitleSource,
    TokenUsage,
    ToolCallData,
    ToolResultData,
    ToolSchema,
    TurnEndData,
    TurnEndReason,
    TurnStartData,
    container::SessionArtifact,
};

/// The producer-side pattern the crate documents: `LogLine` plus the
/// producer's own event types, so a custom line serializes alongside the
/// harness lines.
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Line {
    Harness(LogLine),
    Custom(CustomLine),
}

#[derive(Serialize, Deserialize)]
struct CustomLine {
    #[serde(rename = "type")]
    event_type: String,
    seq: u64,
    time: i64,
    data: serde_json::Value,
    ignorable: bool,
}

#[test]
fn custom_line_deserializes() {
    let raw = r#"{"type":"persona/thought","seq":12,"time":1789560000012,"data":{"text":"x"},"ignorable":true}"#;
    let c: CustomLine = serde_json::from_str(raw).expect("a custom line parses");
    assert_eq!(c.event_type, "persona/thought");
}

#[test]
fn a_persona_session_serializes_the_v3_shape() {
    let t = 1_789_560_000_000i64;
    let root = std::env::temp_dir().join(format!("symbiont-dsh-log-v3-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let artifact =
        SessionArtifact::locate(&root, Some("/var/lib/monty-persona/jeff/work"), "jeff-1");
    let header = LogLine::Session(SessionHeaderLine {
        version: 3,
        id: "jeff-1".to_string(),
        created_at: t as u64,
        cwd: Some("/var/lib/monty-persona/jeff/work".to_string()),
        is_seeded: false,
        parent_session: None,
        origin: None,
        delegation_depth: 0,
        agent_preset: Some("persona".to_string()),
    });
    artifact.create(&header).expect("the header frame writes");

    let lines: Vec<Line> = vec![
        // turn 1 opens, step 1 opens, the system prompt anchors in it
        Line::Harness(LogLine::TurnStart(Event::new(
            0,
            t,
            TurnStartData { turn: 1 },
        ))),
        Line::Harness(LogLine::StepStart(Event::new(
            1,
            t + 1,
            StepStartData { turn: 1, step: 1 },
        ))),
        Line::Harness(LogLine::SystemMessage(
            Event::new(
                2,
                t + 2,
                SystemMessageData {
                    turn: 1,
                    step: 1,
                    message: Message {
                        id: "sys-1".to_string(),
                        role: Role::System,
                        content: vec![ContentBlock::Text {
                            text: "You are Jeff Dean, a persistent developer assistant.".into(),
                        }],
                        source: MessageSource::plugin("persona", ContextForm::Instructions),
                    },
                },
            )
            .on_surface(),
        )),
        // the boot observation: a plain user message
        Line::Harness(LogLine::UserMessage(
            Event::new(
                3,
                t + 3,
                Message {
                    id: "persona-note-0".to_string(),
                    role: Role::User,
                    content: vec![ContentBlock::Text {
                        text: "booted; fresh start".into(),
                    }],
                    source: MessageSource::user(),
                },
            )
            .on_surface(),
        )),
        // the request header, without its retired `system` member
        Line::Harness(LogLine::RequestHeader(Event::new(
            4,
            t + 4,
            RequestHeaderData {
                header: symbiont_dsh_log::EpochHeader {
                    config: LlmCallConfig {
                        provider: "desg0:8000".into(),
                        model: "RadixArk/Qwen3.8-27B-NVFP4".into(),
                        reasoning_effort: None,
                        temperature: Some(0.2),
                        max_tokens: Some(4096),
                        stop: None,
                    },
                    adapter_defaults: None,
                    tools: Some(vec![ToolSchema {
                        name: "python".into(),
                        description: "Run one ```python block in the persistent REPL.".into(),
                        parameters: serde_json::json!({"type": "object"}),
                    }]),
                },
                reason: RequestHeaderReason::Initial,
            },
        ))),
        // the model reply advertises its tool call in content
        Line::Harness(LogLine::AssistantMessage(
            Event::new(
                5,
                t + 5,
                AssistantMessageData {
                    turn: 1,
                    step: 1,
                    stream: vec![],
                    message: Message {
                        id: "persona-msg-1".to_string(),
                        role: Role::Assistant,
                        content: vec![
                            ContentBlock::Reasoning {
                                text: "Let me look at the workspace.".into(),
                            },
                            ContentBlock::ToolCall {
                                id: "persona-call-0".into(),
                                name: "python".into(),
                                arguments: r#"{"code":"import os; print(os.getcwd())"}"#.into(),
                            },
                        ],
                        source: MessageSource::model("desg0:8000", "RadixArk/Qwen3.8-27B-NVFP4"),
                    },
                    usage: Some(TokenUsage {
                        input_tokens: 1200,
                        output_tokens: 80,
                        cache_read_tokens: None,
                        cache_write_tokens: None,
                        reasoning_tokens: Some(40),
                    }),
                },
            )
            .on_surface(),
        )),
        // its log-only twin, then the tool result back on the surface
        Line::Harness(LogLine::ToolCall(Event::new(
            6,
            t + 6,
            ToolCallData {
                turn: 1,
                step: 1,
                call_id: "persona-call-0".into(),
                name: "python".into(),
                arguments: r#"{"code":"import os; print(os.getcwd())"}"#.into(),
            },
        ))),
        Line::Harness(LogLine::ToolResult(
            Event::new(
                7,
                t + 7,
                ToolResultData {
                    turn: 1,
                    step: 1,
                    message: Message {
                        id: "persona-msg-2".to_string(),
                        role: Role::User,
                        content: vec![ContentBlock::ToolResult {
                            tool_call_id: "persona-call-0".into(),
                            content: vec![ContentBlock::Text {
                                text: "/var/lib/monty-persona/jeff/work".into(),
                            }],
                            is_error: None,
                        }],
                        source: MessageSource::tool("persona-call-0"),
                    },
                    error: None,
                    meta: None,
                },
            )
            .on_surface(),
        )),
        Line::Harness(LogLine::StepEnd(Event::new(
            8,
            t + 8,
            StepStartData { turn: 1, step: 1 },
        ))),
        // a second step: the thought completes
        Line::Harness(LogLine::StepStart(Event::new(
            9,
            t + 9,
            StepStartData { turn: 1, step: 2 },
        ))),
        Line::Harness(LogLine::AssistantMessage(
            Event::new(
                10,
                t + 10,
                AssistantMessageData {
                    turn: 1,
                    step: 2,
                    stream: vec![],
                    message: Message {
                        id: "persona-msg-3".to_string(),
                        role: Role::Assistant,
                        content: vec![ContentBlock::Text {
                            text: "Workspace is healthy.".into(),
                        }],
                        source: MessageSource::model("desg0:8000", "RadixArk/Qwen3.8-27B-NVFP4"),
                    },
                    usage: None,
                },
            )
            .on_surface(),
        )),
        Line::Harness(LogLine::StepEnd(Event::new(
            11,
            t + 11,
            StepStartData { turn: 1, step: 2 },
        ))),
        // a custom event: legal in current-format logs with the marker
        Line::Custom(CustomLine {
            event_type: "persona/thought".into(),
            seq: 12,
            time: t + 12,
            data: serde_json::json!({"text": "Initialized the workspace."}),
            ignorable: true,
        }),
        Line::Harness(LogLine::SessionTitle(Event::new(
            13,
            t + 13,
            SessionTitleData {
                title: "jeff-1".into(),
                message_seqs: vec![],
                source: TitleSource::User,
            },
        ))),
        Line::Harness(LogLine::TurnEnd(Event::new(
            14,
            t + 14,
            TurnEndData {
                turn: 1,
                reason: TurnEndReason::Completed,
            },
        ))),
    ];

    artifact.append(&lines).expect("the batch writes");

    let all: Vec<Line> = artifact.read().expect("the container decodes");
    assert_eq!(all.len(), 16);

    // Keep a copy out in the open for the dsh catalog check, in the cargo
    // target directory the test binary itself was built into.
    let exe = std::env::current_exe().expect("the test binary has a path");
    let deps_dir = exe
        .parent()
        .expect("the test binary sits in its target directory");
    let debug_dir = deps_dir
        .parent()
        .expect("the deps directory sits in the target profile");
    let sample = debug_dir
        .join("v3-sample")
        .join(symbiont_dsh_log::container::SESSION_FILE_NAME);
    if let Some(dir) = sample.parent() {
        std::fs::create_dir_all(dir).expect("the sample dir writes");
    }
    std::fs::copy(artifact.path(), &sample).expect("the sample copies");
    eprintln!("v3 sample at {}", sample.display());

    std::fs::remove_dir_all(&root).expect("the test cleans up after itself");
}
