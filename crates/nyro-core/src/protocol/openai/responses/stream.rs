use uuid::Uuid;

use crate::protocol::types::*;
use crate::protocol::{SseEvent, StreamFormatter};

pub struct ResponsesStreamFormatter {
    resp_id: String,
    msg_id: String,
    model: String,
    accumulated_text: String,
    usage: TokenUsage,
    started: bool,
    completed: bool,
}

impl ResponsesStreamFormatter {
    pub fn new() -> Self {
        Self {
            resp_id: format!("resp_{}", Uuid::new_v4().simple()),
            msg_id: format!("msg_{}", Uuid::new_v4().simple()),
            model: String::new(),
            accumulated_text: String::new(),
            usage: TokenUsage::default(),
            started: false,
            completed: false,
        }
    }

    fn emit_preamble(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::with_capacity(4);
        let model = if self.model.is_empty() {
            "unknown".to_string()
        } else {
            self.model.clone()
        };

        let created = serde_json::json!({
            "type": "response.created",
            "response": {
                "id": self.resp_id,
                "object": "response",
                "status": "in_progress",
                "model": model,
                "output": [],
                "output_text": ""
            }
        });
        events.push(SseEvent::new(
            Some("response.created"),
            created.to_string(),
        ));

        let in_progress = serde_json::json!({
            "type": "response.in_progress",
            "response": {
                "id": self.resp_id,
                "object": "response",
                "status": "in_progress"
            }
        });
        events.push(SseEvent::new(
            Some("response.in_progress"),
            in_progress.to_string(),
        ));

        let item_added = serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": self.msg_id,
                "status": "in_progress",
                "role": "assistant",
                "content": []
            }
        });
        events.push(SseEvent::new(
            Some("response.output_item.added"),
            item_added.to_string(),
        ));

        let part_added = serde_json::json!({
            "type": "response.content_part.added",
            "item_id": self.msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": []
            }
        });
        events.push(SseEvent::new(
            Some("response.content_part.added"),
            part_added.to_string(),
        ));

        events
    }

    fn emit_completed(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::with_capacity(4);

        let text_done = serde_json::json!({
            "type": "response.output_text.done",
            "item_id": self.msg_id,
            "output_index": 0,
            "content_index": 0,
            "text": self.accumulated_text
        });
        events.push(SseEvent::new(
            Some("response.output_text.done"),
            text_done.to_string(),
        ));

        let part_done = serde_json::json!({
            "type": "response.content_part.done",
            "item_id": self.msg_id,
            "output_index": 0,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": self.accumulated_text,
                "annotations": []
            }
        });
        events.push(SseEvent::new(
            Some("response.content_part.done"),
            part_done.to_string(),
        ));

        let item_done = serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": self.msg_id,
                "status": "completed",
                "role": "assistant",
                "content": [{
                    "type": "output_text",
                    "text": self.accumulated_text,
                    "annotations": []
                }]
            }
        });
        events.push(SseEvent::new(
            Some("response.output_item.done"),
            item_done.to_string(),
        ));

        let completed = serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": self.resp_id,
                "object": "response",
                "status": "completed",
                "model": self.model,
                "output": [{
                    "type": "message",
                    "id": self.msg_id,
                    "status": "completed",
                    "role": "assistant",
                    "content": [{
                        "type": "output_text",
                        "text": self.accumulated_text,
                        "annotations": []
                    }]
                }],
                "output_text": self.accumulated_text,
                "usage": {
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                    "total_tokens": self.usage.input_tokens + self.usage.output_tokens
                }
            }
        });
        events.push(SseEvent::new(
            Some("response.completed"),
            completed.to_string(),
        ));

        events
    }
}

impl StreamFormatter for ResponsesStreamFormatter {
    fn format_deltas(&mut self, deltas: &[StreamDelta]) -> Vec<SseEvent> {
        let mut events = Vec::new();

        for delta in deltas {
            match delta {
                StreamDelta::MessageStart { id, model } => {
                    if !id.is_empty() {
                        self.resp_id = id.clone();
                    }
                    self.model = model.clone();
                    if !self.started {
                        self.started = true;
                        events.extend(self.emit_preamble());
                    }
                }
                StreamDelta::TextDelta(text) => {
                    if !self.started {
                        self.started = true;
                        events.extend(self.emit_preamble());
                    }
                    self.accumulated_text.push_str(text);
                    let ev = serde_json::json!({
                        "type": "response.output_text.delta",
                        "item_id": self.msg_id,
                        "output_index": 0,
                        "content_index": 0,
                        "delta": text
                    });
                    events.push(SseEvent::new(
                        Some("response.output_text.delta"),
                        ev.to_string(),
                    ));
                }
                StreamDelta::Usage(u) => {
                    self.usage = u.clone();
                }
                StreamDelta::Done { .. } => {
                    if !self.completed {
                        self.completed = true;
                        events.extend(self.emit_completed());
                    }
                }
                _ => {}
            }
        }

        events
    }

    fn format_done(&mut self) -> Vec<SseEvent> {
        let mut events = Vec::new();
        if !self.completed {
            self.completed = true;
            events.extend(self.emit_completed());
        }
        events.push(SseEvent::new(None, "[DONE]"));
        events
    }

    fn usage(&self) -> TokenUsage {
        self.usage.clone()
    }
}
