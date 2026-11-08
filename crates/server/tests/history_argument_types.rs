use gb_server::{budget::Limits, field_policy::validate_chat};
use serde_json::json;
#[test]
fn history_arguments_strings_must_encode_objects_not_scalars_or_arrays() {
    for text in ["[]", "true", "12", "\"text\""] {
        let body = json!({"messages":[{"role":"assistant","content":null,"tool_calls":[{
            "id":"call1","type":"function","function":{"name":"run","arguments":text}}]}]});
        assert_eq!(validate_chat(&body, &Limits::default()).unwrap_err().status, 400);
    }
    let body = json!({"messages":[{"role":"assistant","content":null,"tool_calls":[{
        "id":"call1","type":"function","function":{"name":"run","arguments":" {\"value\":1}"}}]}]});
    validate_chat(&body, &Limits::default()).unwrap();
}
