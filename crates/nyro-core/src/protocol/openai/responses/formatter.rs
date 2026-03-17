use serde_json::Value;
use uuid::Uuid;

use crate::protocol::types::InternalResponse;
use crate::protocol::ResponseFormatter;

pub struct ResponsesResponseFormatter;

impl ResponseFormatter for ResponsesResponseFormatter {
    fn format_response(&self, resp: &InternalResponse) -> Value {
        let resp_id = if resp.id.is_empty() {
            format!("resp_{}", Uuid::new_v4().simple())
        } else {
            resp.id.clone()
        };

        serde_json::json!({
            "id": resp_id,
            "object": "response",
            "status": "completed",
            "model": resp.model,
            "output": [{
                "type": "message",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": resp.content,
                    "annotations": []
                }]
            }],
            "output_text": resp.content,
            "usage": {
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "total_tokens": resp.usage.input_tokens + resp.usage.output_tokens
            }
        })
    }
}
