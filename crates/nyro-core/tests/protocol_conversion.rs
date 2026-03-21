use nyro_core::protocol::anthropic::stream::AnthropicResponseFormatter;
use nyro_core::protocol::anthropic::decoder::AnthropicDecoder;
use nyro_core::protocol::anthropic::encoder::AnthropicEncoder;
use nyro_core::protocol::openai::encoder::OpenAIEncoder;
use nyro_core::protocol::openai::responses::decoder::ResponsesDecoder;
use nyro_core::protocol::openai::responses::formatter::ResponsesResponseFormatter;
use nyro_core::protocol::semantic::reasoning::normalize_response_reasoning;
use nyro_core::protocol::semantic::tool_correlation::normalize_request_tool_results;
use nyro_core::protocol::types::{
    ContentBlock, InternalMessage, InternalRequest, InternalResponse, MessageContent, ResponseItem, Role,
    TokenUsage, ToolCall, ToolDef,
};
use nyro_core::protocol::{IngressDecoder, Protocol, ResponseFormatter};
use nyro_core::protocol::EgressEncoder;

#[test]
fn openai_to_anthropic_thinking_blocks() {
    let resp = InternalResponse {
        id: "msg_1".to_string(),
        model: "minimax-m2.7".to_string(),
        content: "hello".to_string(),
        reasoning_content: Some("reasoning summary".to_string()),
        tool_calls: vec![],
        response_items: None,
        stop_reason: Some("stop".to_string()),
        usage: TokenUsage {
            input_tokens: 10,
            output_tokens: 20,
        },
    };

    let out = AnthropicResponseFormatter.format_response(&resp);
    let content = out
        .get("content")
        .and_then(|v| v.as_array())
        .expect("content should be array");
    assert_eq!(content[0].get("type").and_then(|v| v.as_str()), Some("thinking"));
    assert_eq!(
        content[0].get("thinking").and_then(|v| v.as_str()),
        Some("reasoning summary")
    );
}

#[test]
fn openai_to_responses_reasoning_and_function_call_items() {
    let resp = InternalResponse {
        id: "resp_1".to_string(),
        model: "minimax-m2.7".to_string(),
        content: "done".to_string(),
        reasoning_content: Some("chain".to_string()),
        tool_calls: vec![ToolCall {
            id: "call_123".to_string(),
            name: "ls".to_string(),
            arguments: "{\"path\":\".\"}".to_string(),
        }],
        response_items: Some(vec![
            ResponseItem::Reasoning {
                text: "chain".to_string(),
            },
            ResponseItem::FunctionCall {
                call_id: "call_123".to_string(),
                name: "ls".to_string(),
                arguments: "{\"path\":\".\"}".to_string(),
            },
            ResponseItem::Message {
                text: "done".to_string(),
            },
        ]),
        stop_reason: Some("stop".to_string()),
        usage: TokenUsage::default(),
    };

    let out = ResponsesResponseFormatter.format_response(&resp);
    let output = out
        .get("output")
        .and_then(|v| v.as_array())
        .expect("output should be array");
    assert!(
        output
            .iter()
            .any(|item| item.get("type").and_then(|v| v.as_str()) == Some("reasoning"))
    );
    assert!(
        output
            .iter()
            .any(|item| item.get("type").and_then(|v| v.as_str()) == Some("function_call"))
    );
    assert!(
        output
            .iter()
            .any(|item| item.get("type").and_then(|v| v.as_str()) == Some("message"))
    );
}

#[test]
fn gemini_tool_result_correlation_success() {
    let mut req = InternalRequest {
        messages: vec![
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_abc".to_string(),
                    name: "read_file".to_string(),
                    arguments: "{\"path\":\"src/main.rs\"}".to_string(),
                }]),
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                    tool_use_id: "read_file".to_string(),
                    content: serde_json::json!({"ok": true}),
                }]),
                tool_calls: None,
                tool_call_id: None,
            },
        ],
        model: "minimax-m2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        source_protocol: Protocol::Gemini,
        extra: Default::default(),
    };

    normalize_request_tool_results(&mut req);
    assert_eq!(
        req.messages[1].tool_call_id.as_deref(),
        Some("call_abc"),
        "tool result should be correlated to previous assistant tool_call id"
    );
}

#[test]
fn minimax_reasoning_split_fallback_think_tag() {
    let mut resp = InternalResponse {
        id: "resp_2".to_string(),
        model: "minimax-m2.7".to_string(),
        content: "<think>plan first</think>run ls".to_string(),
        reasoning_content: None,
        tool_calls: vec![],
        response_items: None,
        stop_reason: Some("stop".to_string()),
        usage: TokenUsage::default(),
    };

    normalize_response_reasoning(&mut resp);
    assert_eq!(resp.reasoning_content.as_deref(), Some("plan first"));
    assert_eq!(resp.content, "run ls");
}

#[test]
fn non_reasoning_model_no_regression() {
    let mut resp = InternalResponse {
        id: "resp_3".to_string(),
        model: "plain-model".to_string(),
        content: "hello world".to_string(),
        reasoning_content: None,
        tool_calls: vec![],
        response_items: None,
        stop_reason: Some("stop".to_string()),
        usage: TokenUsage::default(),
    };

    normalize_response_reasoning(&mut resp);
    assert!(resp.reasoning_content.is_none());
    assert_eq!(resp.content, "hello world");
}

#[test]
fn anthropic_tool_result_decodes_to_tool_role() {
    let body = serde_json::json!({
        "model": "claude-sonnet",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call_abc",
                        "name": "read_file",
                        "input": {"path": "Cargo.toml"}
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_abc",
                        "content": {"ok": true}
                    }
                ]
            }
        ]
    });

    let req = AnthropicDecoder.decode_request(body).expect("decode anthropic request");
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.messages[1].role, Role::Tool);
    assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("call_abc"));
}

#[test]
fn anthropic_multi_tool_result_decodes_to_multiple_tool_messages() {
    let body = serde_json::json!({
        "model": "claude-sonnet",
        "max_tokens": 1024,
        "messages": [
            {
                "role": "assistant",
                "content": [
                    { "type": "tool_use", "id": "call_a", "name": "read_file", "input": {"path":"a"} },
                    { "type": "tool_use", "id": "call_b", "name": "read_file", "input": {"path":"b"} }
                ]
            },
            {
                "role": "user",
                "content": [
                    { "type": "tool_result", "tool_use_id": "call_a", "content": {"ok": true} },
                    { "type": "tool_result", "tool_use_id": "call_b", "content": {"ok": true} }
                ]
            }
        ]
    });
    let req = AnthropicDecoder.decode_request(body).expect("decode anthropic request");
    assert_eq!(req.messages.len(), 3);
    assert_eq!(req.messages[1].role, Role::Tool);
    assert_eq!(req.messages[2].role, Role::Tool);
    assert_eq!(req.messages[1].tool_call_id.as_deref(), Some("call_a"));
    assert_eq!(req.messages[2].tool_call_id.as_deref(), Some("call_b"));
}

#[test]
fn openai_encoder_injects_synthetic_tool_call_before_orphan_tool_result() {
    let req = InternalRequest {
        messages: vec![InternalMessage {
            role: Role::Tool,
            content: MessageContent::Text("{\"ok\":true}".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_orphan_1".to_string()),
        }],
        model: "minimax-m2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        source_protocol: Protocol::ResponsesAPI,
        extra: Default::default(),
    };

    let (body, _) = OpenAIEncoder
        .encode_request(&req)
        .expect("encode openai body");
    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .expect("messages array");
    assert_eq!(messages.len(), 2);
    assert_eq!(
        messages[0].get("role").and_then(|v| v.as_str()),
        Some("assistant")
    );
    assert_eq!(messages[1].get("role").and_then(|v| v.as_str()), Some("tool"));
    assert_eq!(
        messages[1].get("tool_call_id").and_then(|v| v.as_str()),
        Some("call_orphan_1")
    );
}

#[test]
fn openai_encoder_injects_adjacent_tool_call_for_non_adjacent_match() {
    let req = InternalRequest {
        messages: vec![
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text("will call".to_string()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_x".to_string(),
                    name: "ls".to_string(),
                    arguments: "{}".to_string(),
                }]),
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::User,
                content: MessageContent::Text("intermediate".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Text("{\"ok\":true}".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_x".to_string()),
            },
        ],
        model: "minimax-m2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        source_protocol: Protocol::ResponsesAPI,
        extra: Default::default(),
    };

    let (body, _) = OpenAIEncoder
        .encode_request(&req)
        .expect("encode openai body");
    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .expect("messages array");

    assert_eq!(messages.len(), 4);
    assert_eq!(
        messages[2].get("role").and_then(|v| v.as_str()),
        Some("assistant")
    );
    assert_eq!(messages[3].get("role").and_then(|v| v.as_str()), Some("tool"));
    let tool_id = messages[3]
        .get("tool_call_id")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert!(!tool_id.is_empty());
    let assistant_call_id = messages[2]
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|tc| tc.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(assistant_call_id, tool_id);
}

#[test]
fn openai_encoder_drops_intermediate_assistant_text_before_tool_result() {
    let req = InternalRequest {
        messages: vec![
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text("plan".to_string()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_keep".to_string(),
                    name: "exec_command".to_string(),
                    arguments: "{\"command\":\"ls -la\"}".to_string(),
                }]),
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text("extra text".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Text("{\"stdout\":\"...\"}".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_keep".to_string()),
            },
        ],
        model: "MiniMax-M2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        source_protocol: Protocol::ResponsesAPI,
        extra: Default::default(),
    };

    let (body, _) = OpenAIEncoder
        .encode_request(&req)
        .expect("encode openai body");
    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .expect("messages array");

    // intermediate assistant text should be dropped to keep tool_result adjacent
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0].get("role").and_then(|v| v.as_str()), Some("assistant"));
    assert_eq!(
        messages[1]
            .get("tool_calls")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|tc| tc.get("id"))
            .and_then(|v| v.as_str()),
        Some("call_keep")
    );
    assert_eq!(messages[2].get("role").and_then(|v| v.as_str()), Some("tool"));
    assert_eq!(
        messages[2].get("tool_call_id").and_then(|v| v.as_str()),
        Some("call_keep")
    );
}

#[test]
fn openai_encoder_remaps_duplicate_tool_call_ids() {
    let req = InternalRequest {
        messages: vec![
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_dup".to_string(),
                    name: "exec_command".to_string(),
                    arguments: "{}".to_string(),
                }]),
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_dup".to_string(),
                    name: "exec_command".to_string(),
                    arguments: "{}".to_string(),
                }]),
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Text("{\"ok\":true}".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_dup".to_string()),
            },
        ],
        model: "MiniMax-M2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        tools: None,
        tool_choice: None,
        source_protocol: Protocol::ResponsesAPI,
        extra: Default::default(),
    };

    let (body, _) = OpenAIEncoder
        .encode_request(&req)
        .expect("encode openai body");
    let messages = body
        .get("messages")
        .and_then(|v| v.as_array())
        .expect("messages array");

    let ids: Vec<String> = messages
        .iter()
        .filter_map(|m| m.get("tool_calls").and_then(|v| v.as_array()).and_then(|arr| arr.first()))
        .filter_map(|tc| tc.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
        .collect();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1]);

    let tool_id = messages
        .iter()
        .find(|m| m.get("role").and_then(|v| v.as_str()) == Some("tool"))
        .and_then(|m| m.get("tool_call_id"))
        .and_then(|v| v.as_str())
        .expect("tool_call_id exists");
    assert_eq!(tool_id, ids[1]);
}

#[test]
fn anthropic_encoder_maps_required_tool_choice_to_any() {
    let req = InternalRequest {
        messages: vec![InternalMessage {
            role: Role::User,
            content: MessageContent::Text("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }],
        model: "MiniMax-M2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: Some(256),
        top_p: None,
        tools: Some(vec![ToolDef {
            name: "exec_command".to_string(),
            description: Some("Execute command".to_string()),
            parameters: serde_json::json!({"type":"object","properties":{"command":{"type":"string"}}}),
        }]),
        tool_choice: Some(serde_json::json!("required")),
        source_protocol: Protocol::ResponsesAPI,
        extra: Default::default(),
    };

    let (body, _) = AnthropicEncoder
        .encode_request(&req)
        .expect("encode anthropic body");
    assert_eq!(
        body.get("tool_choice")
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str()),
        Some("any")
    );
}

#[test]
fn anthropic_encoder_maps_function_tool_choice_to_tool_name() {
    let req = InternalRequest {
        messages: vec![InternalMessage {
            role: Role::User,
            content: MessageContent::Text("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }],
        model: "MiniMax-M2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: Some(256),
        top_p: None,
        tools: Some(vec![ToolDef {
            name: "exec_command".to_string(),
            description: Some("Execute command".to_string()),
            parameters: serde_json::json!({"type":"object","properties":{"command":{"type":"string"}}}),
        }]),
        tool_choice: Some(serde_json::json!({
            "type":"function",
            "function":{"name":"exec_command"}
        })),
        source_protocol: Protocol::ResponsesAPI,
        extra: Default::default(),
    };

    let (body, _) = AnthropicEncoder
        .encode_request(&req)
        .expect("encode anthropic body");
    assert_eq!(
        body.get("tool_choice")
            .and_then(|v| v.get("type"))
            .and_then(|v| v.as_str()),
        Some("tool")
    );
    assert_eq!(
        body.get("tool_choice")
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str()),
        Some("exec_command")
    );
}

#[test]
fn anthropic_encoder_merges_consecutive_roles_and_drops_empty_text() {
    let req = InternalRequest {
        messages: vec![
            InternalMessage {
                role: Role::User,
                content: MessageContent::Text("first".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::User,
                content: MessageContent::Text("second".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text(String::new()),
                tool_calls: None,
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text("tool".to_string()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_1".to_string(),
                    name: "exec_command".to_string(),
                    arguments: "{}".to_string(),
                }]),
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Text("result".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_1".to_string()),
            },
        ],
        model: "MiniMax-M2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: Some(256),
        top_p: None,
        tools: None,
        tool_choice: None,
        source_protocol: Protocol::ResponsesAPI,
        extra: Default::default(),
    };

    let (body, _) = AnthropicEncoder
        .encode_request(&req)
        .expect("encode anthropic body");
    let msgs = body
        .get("messages")
        .and_then(|v| v.as_array())
        .expect("messages array");
    assert_eq!(msgs.len(), 3);
    assert_eq!(msgs[0].get("role").and_then(|v| v.as_str()), Some("user"));
    assert_eq!(msgs[1].get("role").and_then(|v| v.as_str()), Some("assistant"));
    assert_eq!(msgs[2].get("role").and_then(|v| v.as_str()), Some("user"));

    let first_blocks = msgs[0]
        .get("content")
        .and_then(|v| v.as_array())
        .expect("first content blocks");
    assert_eq!(first_blocks.len(), 2);
    assert_eq!(
        first_blocks[0].get("text").and_then(|v| v.as_str()),
        Some("first")
    );
    assert_eq!(
        first_blocks[1].get("text").and_then(|v| v.as_str()),
        Some("second")
    );
}

#[test]
fn responses_decoder_ignores_empty_message_content_item() {
    let body = serde_json::json!({
        "model": "MiniMax-M2.7-Code-Claude",
        "input": [
            { "type": "message", "role": "user", "content": [] },
            {
                "type": "message",
                "role": "user",
                "content": [{ "type": "input_text", "text": "帮我查看当前目录下有哪些文件" }]
            }
        ]
    });

    let req = ResponsesDecoder
        .decode_request(body)
        .expect("decode request should succeed");
    assert_eq!(req.messages.len(), 1);
    assert_eq!(req.messages[0].role, Role::User);
    assert_eq!(
        req.messages[0].content.as_text(),
        "帮我查看当前目录下有哪些文件"
    );
}

#[test]
fn openai_encoder_remaps_reused_tool_result_id_with_synthetic_adjacent_call() {
    let req = InternalRequest {
        messages: vec![
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text(String::new()),
                tool_calls: Some(vec![ToolCall {
                    id: "call_same".to_string(),
                    name: "exec_command".to_string(),
                    arguments: "{}".to_string(),
                }]),
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Text("ok1".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_same".to_string()),
            },
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text("intermediate".to_string()),
                tool_calls: None,
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Text("ok2".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_same".to_string()),
            },
        ],
        model: "gpt-4o-mini".to_string(),
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        tools: Some(vec![ToolDef {
            name: "exec_command".to_string(),
            description: None,
            parameters: serde_json::json!({"type":"object","properties":{}}),
        }]),
        tool_choice: None,
        source_protocol: Protocol::OpenAI,
        extra: Default::default(),
    };

    let (body, _) = OpenAIEncoder.encode_request(&req).expect("encode");
    let msgs = body
        .get("messages")
        .and_then(|v| v.as_array())
        .expect("messages");
    let mut tool_ids: Vec<String> = Vec::new();
    for msg in msgs {
        if msg.get("role").and_then(|v| v.as_str()) == Some("tool") {
            let id = msg
                .get("tool_call_id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            assert!(!id.is_empty());
            tool_ids.push(id);
        }
    }
    assert_eq!(tool_ids.len(), 2);
    assert_ne!(tool_ids[0], tool_ids[1]);
}

#[test]
fn openai_encoder_rewrites_multi_tool_call_history_to_adjacent_pairs() {
    let req = InternalRequest {
        messages: vec![
            InternalMessage {
                role: Role::Assistant,
                content: MessageContent::Text("".to_string()),
                tool_calls: Some(vec![
                    ToolCall {
                        id: "call_a".to_string(),
                        name: "Glob".to_string(),
                        arguments: "{}".to_string(),
                    },
                    ToolCall {
                        id: "call_b".to_string(),
                        name: "Bash".to_string(),
                        arguments: "{}".to_string(),
                    },
                ]),
                tool_call_id: None,
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Text("r1".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_a".to_string()),
            },
            InternalMessage {
                role: Role::Tool,
                content: MessageContent::Text("r2".to_string()),
                tool_calls: None,
                tool_call_id: Some("call_b".to_string()),
            },
        ],
        model: "MiniMax-M2.7".to_string(),
        stream: false,
        temperature: None,
        max_tokens: None,
        top_p: None,
        tools: Some(vec![ToolDef {
            name: "Glob".to_string(),
            description: None,
            parameters: serde_json::json!({"type":"object","properties":{}}),
        }]),
        tool_choice: None,
        source_protocol: Protocol::Anthropic,
        extra: Default::default(),
    };

    let (body, _) = OpenAIEncoder.encode_request(&req).expect("encode");
    let msgs = body
        .get("messages")
        .and_then(|v| v.as_array())
        .expect("messages");
    assert_eq!(msgs.len(), 4);
    assert_eq!(msgs[0].get("role").and_then(|v| v.as_str()), Some("assistant"));
    assert_eq!(msgs[1].get("role").and_then(|v| v.as_str()), Some("tool"));
    assert_eq!(msgs[2].get("role").and_then(|v| v.as_str()), Some("assistant"));
    assert_eq!(msgs[3].get("role").and_then(|v| v.as_str()), Some("tool"));
    let id1 = msgs[1].get("tool_call_id").and_then(|v| v.as_str()).unwrap_or("");
    let id2 = msgs[3].get("tool_call_id").and_then(|v| v.as_str()).unwrap_or("");
    let prev1 = msgs[0]
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|tc| tc.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let prev2 = msgs[2]
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .and_then(|arr| arr.first())
        .and_then(|tc| tc.get("id"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    assert_eq!(id1, prev1);
    assert_eq!(id2, prev2);
}
