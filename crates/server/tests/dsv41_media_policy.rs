//! The format must honor the live vision-handle admission decision.
mod common;

use gb_server::chat_format::ChatFormat;
use gb_server::formats::deepseek_v41::DeepseekV41Format;
use serde_json::json;

#[test]
fn dsv41_requires_a_vision_handle_for_every_image_shape() {
    let format = DeepseekV41Format;
    let text = json!([{"role":"user","content":"hello"}]);
    assert!(format.check_media(&text, false).is_ok());
    for part in [
        json!({"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}),
        json!({"type":"image","image":"data:image/png;base64,AAAA"}),
        json!({"type":"input_image","image_url":"data:image/png;base64,AAAA"}),
        json!({"image_url":{"url":"data:image/png;base64,AAAA"}}),
    ] {
        let messages = json!([{"role":"user","content":[part,{"type":"text","text":"Describe."}]}]);
        assert!(format.check_media(&messages, true).is_ok());
        assert!(format.check_media(&messages, false).is_err(), "no image may reach text-only rendering: {messages}");
    }
}

#[test]
fn dsv41_http_without_vision_rejects_images_before_rendering() {
    let server = common::start(common::Opts { format: Some(std::sync::Arc::new(DeepseekV41Format)), ..Default::default() });
    let request = json!({
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            {"type": "text", "text": "What color is the square?"}
        ]}],
        "max_tokens": 16,
        "chat_template_kwargs": {"enable_thinking": false}
    });
    for route in ["/v1/chat/completions", "/v1/chat/completions/render"] {
        let reply = common::http_timeout(server.port, "POST", route, Some(&request), std::time::Duration::from_secs(10));
        assert_eq!(reply.status, 400, "{route}: {}", reply.body);
        assert!(reply.json()["error"]["message"].as_str().unwrap().contains("image"));
    }
    assert_eq!(server.in_flight(), 0);
}

#[test]
fn dsv41_vision_does_not_enable_audio_or_video() {
    let format = DeepseekV41Format;
    for part in [
        json!({"type":"video_url","video_url":{"url":"data:video/mp4;base64,AAAA"}}),
        json!({"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}),
    ] {
        let messages = json!([{"role":"user","content":[part]}]);
        for allow_images in [false, true] {
            assert!(format.check_media(&messages, allow_images).is_err());
        }
    }
}
