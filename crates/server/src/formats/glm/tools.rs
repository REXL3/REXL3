//! GLM-5.3-Flash tool calls: the block syntax and the typing of argument values.
//!
//! The GLM chat template tells the model (and renders history) as
//!
//! ```text
//! <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value><arg_key>KEY2</arg_key><arg_value>VALUE2</arg_value></tool_call>
//! ```
//!
//! with no whitespace anywhere. A string argument is written verbatim (`{{ v if v is string }}`), any other value as
//! `json.dumps(v, ensure_ascii=False)`, so `3`, `2.5`, `true`, `null`, `[1, "x"]` and `{"k": "v"}` look the same as the strings that spell them:
//! the declared schema is the only way to tell them apart, and it decides here (see [`typed`]). Several calls follow one another,
//! and a call may have no arguments (`<tool_call>get_time</tool_call>`). The reader also accepts the whitespace some GLM releases
//! put after the name and between the tags (`<tool_call>f\n<arg_key>k</arg_key>\n<arg_value>v</arg_value>\n</tool_call>`), which cannot
//! change a value: string values are taken between the value tags exactly as written.
//!
//! A block that is not in this shape (a tag missing or out of order, an empty key, a name with spaces) is *not* a call: the splitter
//! leaves it in the visible text instead of running a tool with half its arguments.

use crate::tools::{ToolCall, ToolSchemas};
use serde_json::{Map, Value};

const KEY_OPEN: &str = "<arg_key>";
const KEY_CLOSE: &str = "</arg_key>";
const VALUE_OPEN: &str = "<arg_value>";
const VALUE_CLOSE: &str = "</arg_value>";

/// Parse the inside of one `<tool_call>` block. `None` when it is not a well-formed GLM call.
pub fn parse_call(block: &str, schemas: &ToolSchemas) -> Option<ToolCall> {
    let name_end = block.find(KEY_OPEN).unwrap_or(block.len());
    let name = block[..name_end].trim();
    if name.is_empty() || name.contains(char::is_whitespace) || name.contains(['<', '>']) {
        return None;
    }
    let mut rest = &block[name_end..];
    let mut args = Map::new();
    loop {
        rest = rest.trim_start();
        if rest.is_empty() {
            break;
        }
        let after = rest.strip_prefix(KEY_OPEN)?;
        let key_end = after.find(KEY_CLOSE)?;
        let key = after[..key_end].trim();
        if key.is_empty() || key.contains(['<', '>']) {
            return None;
        }
        let after = after[key_end + KEY_CLOSE.len()..].trim_start().strip_prefix(VALUE_OPEN)?;
        let value_end = after.find(VALUE_CLOSE)?;
        args.insert(key.to_string(), typed(schemas, name, key, &after[..value_end]));
        rest = &after[value_end + VALUE_CLOSE.len()..];
    }
    Some(ToolCall { name: name.to_string(), arguments: Value::Object(args).to_string() })
}

/// JSON kinds a schema admits, as bits.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct Kinds(u8);

impl Kinds {
    const NULL: u8 = 1;
    const BOOL: u8 = 2;
    const INTEGER: u8 = 4;
    const NUMBER: u8 = 8;
    const STRING: u8 = 16;
    const ARRAY: u8 = 32;
    const OBJECT: u8 = 64;
    /// The schema (or part of it) says nothing about the type.
    const UNKNOWN: u8 = 128;

    fn has(self, bit: u8) -> bool {
        self.0 & bit != 0
    }
    fn only_string(self) -> bool {
        self.0 == Self::STRING
    }
}

/// The kinds `schema` admits: `type` (a name or a list of names), `anyOf` / `oneOf` / `allOf` (their union), `enum` and `const` (the
/// kinds of the listed values, when there is no `type`), `$ref` into the function's `$defs`, and the shape keywords (`properties`,
/// `items`). Anything that says nothing counts as unknown.
fn kinds(schema: &Value, root: &Value, depth: usize) -> Kinds {
    let unknown = Kinds(Kinds::UNKNOWN);
    if depth > 8 {
        return unknown;
    }
    let Some(obj) = schema.as_object() else { return if schema == &Value::Bool(true) { unknown } else { Kinds(0) } };
    let mut k = 0u8;
    let name = |n: &str| match n {
        "null" => Kinds::NULL,
        "boolean" => Kinds::BOOL,
        "integer" => Kinds::INTEGER,
        "number" => Kinds::NUMBER,
        "string" => Kinds::STRING,
        "array" => Kinds::ARRAY,
        "object" => Kinds::OBJECT,
        _ => Kinds::UNKNOWN,
    };
    match obj.get("type") {
        Some(Value::String(t)) => k |= name(t),
        Some(Value::Array(ts)) => ts.iter().filter_map(|t| t.as_str()).for_each(|t| k |= name(t)),
        _ => {}
    }
    if k == 0 {
        let of_value = |v: &Value| match v {
            Value::Null => Kinds::NULL,
            Value::Bool(_) => Kinds::BOOL,
            Value::Number(n) if n.is_i64() || n.is_u64() => Kinds::INTEGER,
            Value::Number(_) => Kinds::NUMBER,
            Value::String(_) => Kinds::STRING,
            Value::Array(_) => Kinds::ARRAY,
            Value::Object(_) => Kinds::OBJECT,
        };
        if let Some(Value::Array(vs)) = obj.get("enum") {
            vs.iter().for_each(|v| k |= of_value(v));
        }
        if let Some(c) = obj.get("const") {
            k |= of_value(c);
        }
    }
    if k == 0 {
        for key in ["anyOf", "oneOf", "allOf"] {
            if let Some(Value::Array(parts)) = obj.get(key) {
                parts.iter().for_each(|p| k |= kinds(p, root, depth + 1).0);
            }
        }
    }
    if k == 0 {
        if let Some(target) =
            obj.get("$ref").and_then(|r| r.as_str()).and_then(|r| r.strip_prefix('#')).and_then(|p| root.pointer(p))
        {
            k |= kinds(target, root, depth + 1).0;
        }
    }
    if k == 0 {
        if obj.contains_key("properties") || obj.contains_key("additionalProperties") {
            k |= Kinds::OBJECT;
        } else if obj.contains_key("items") || obj.contains_key("prefixItems") {
            k |= Kinds::ARRAY;
        }
    }
    Kinds(if k == 0 { Kinds::UNKNOWN } else { k })
}

/// The value of an argument written as `raw`, from the parameter's declared schema.
///
/// - Only `string` declared (alone, or with `null`): the text as written, untrimmed, except that the exact text `null` of a nullable
///   string is `null` (the template writes `None` that way; the string `"null"` is the price of that reading).
/// - Otherwise the text, trimmed, is read as JSON and used when its kind is one the schema admits (an integer is a number; an integral
///   float such as `3.0` is an integer; `True` and `False` are booleans). A quoted JSON string is never used: strings are written bare.
/// - What does not read as an admitted kind stays the text when `string` is admitted, and also when it is not (an `integer` parameter given
///   `three` is the string `"three"`, never a coerced 0 or a dropped argument: the tool's own validation reports it).
/// - An undeclared parameter (or one whose schema says nothing) is JSON when it parses to something other than a string, else the text.
pub fn typed(schemas: &ToolSchemas, func: &str, key: &str, raw: &str) -> Value {
    let k = match schemas.param_schema(func, key) {
        Some((s, root)) => kinds(s, root, 0),
        None => Kinds(Kinds::UNKNOWN),
    };
    if k.only_string() {
        return Value::String(raw.to_string());
    }
    let text = raw.trim();
    let parsed: Option<Value> = serde_json::from_str(text).ok();
    let text_value = || Value::String(raw.to_string());
    if k.has(Kinds::UNKNOWN) {
        return parsed.filter(|v| !v.is_string()).unwrap_or_else(text_value);
    }
    match parsed {
        Some(Value::Null) if k.has(Kinds::NULL) => Value::Null,
        Some(Value::Bool(b)) if k.has(Kinds::BOOL) => Value::Bool(b),
        Some(Value::Number(n)) => {
            let is_int = n.is_i64() || n.is_u64();
            let integral = n.as_f64().filter(|f| f.fract() == 0.0 && f.abs() < 9.0e15);
            if k.has(Kinds::NUMBER) || (is_int && k.has(Kinds::INTEGER)) {
                Value::Number(n) // a number takes any JSON number; an integer takes an integer
            } else if let (true, Some(f)) = (k.has(Kinds::INTEGER), integral) {
                Value::from(f as i64) // 3.0 for an integer parameter
            } else {
                text_value()
            }
        }
        Some(v @ Value::Array(_)) if k.has(Kinds::ARRAY) => v,
        Some(v @ Value::Object(_)) if k.has(Kinds::OBJECT) => v,
        _ => match text.to_ascii_lowercase().as_str() {
            "true" if k.has(Kinds::BOOL) => Value::Bool(true),
            "false" if k.has(Kinds::BOOL) => Value::Bool(false),
            _ => text_value(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schemas() -> ToolSchemas {
        ToolSchemas::from_request(&json!({"tools": [
            {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "z": {"type": "null"},
                "a": {"type": "array"},
                "o": {"type": "object"},
                "opt_s": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "opt_i": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                "list_t": {"type": ["integer", "null"]},
                "si": {"anyOf": [{"type": "string"}, {"type": "integer"}]},
                "one": {"oneOf": [{"type": "boolean"}, {"type": "array", "items": {"type": "string"}}]},
                "e_s": {"enum": ["a", "b"]},
                "e_i": {"enum": [1, 2, 3]},
                "c": {"const": 7},
                "shaped": {"properties": {"x": {"type": "integer"}}},
                "ref": {"$ref": "#/$defs/Unit"},
                "ref_list": {"type": "array", "items": {"$ref": "#/$defs/Unit"}},
                "loop": {"$ref": "#/$defs/Loop"},
                "free": {"description": "anything"}
            }, "$defs": {"Unit": {"type": "string", "enum": ["c", "f"]}, "Loop": {"$ref": "#/$defs/Loop"}}}}}
        ]}))
        .unwrap()
    }

    fn args(raw_block: &str) -> Value {
        let c = parse_call(raw_block, &schemas()).unwrap_or_else(|| panic!("not a call: {raw_block:?}"));
        serde_json::from_str(&c.arguments).unwrap()
    }

    fn one(key: &str, raw: &str) -> Value {
        args(&format!("f{KEY_OPEN}{key}{KEY_CLOSE}{VALUE_OPEN}{raw}{VALUE_CLOSE}"))[key].clone()
    }

    #[test]
    fn a_call_with_typed_arguments_in_the_order_written() {
        let c = parse_call(
            "get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value><arg_key>days</arg_key><arg_value>3</arg_value><arg_key>c</arg_key><arg_value>true</arg_value>",
            &ToolSchemas::from_request(&json!({"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"properties": {
                "city": {"type": "string"}, "days": {"type": "integer"}, "c": {"type": "boolean"}}}}}]}))
            .unwrap(),
        )
        .unwrap();
        assert_eq!(c.name, "get_weather");
        assert_eq!(c.arguments, r#"{"city":"Paris","days":3,"c":true}"#);
    }

    #[test]
    fn declared_strings_are_verbatim_even_when_they_look_like_something_else() {
        assert_eq!(one("s", "12345"), json!("12345"));
        assert_eq!(one("s", "true"), json!("true"));
        assert_eq!(one("s", "null"), json!("null"));
        assert_eq!(one("s", "[1, 2]"), json!("[1, 2]"));
        assert_eq!(one("s", "  padded \n"), json!("  padded \n"), "a string value is not trimmed");
        assert_eq!(one("s", ""), json!(""));
        assert_eq!(one("s", "\"quoted\""), json!("\"quoted\""));
        assert_eq!(one("s", "line1\nline2 \u{e9}\u{4e2d}"), json!("line1\nline2 \u{e9}\u{4e2d}"));
    }

    #[test]
    fn numbers_booleans_nulls_and_json_values_are_typed_from_their_declaration() {
        assert_eq!(one("i", "3"), json!(3));
        assert_eq!(one("i", " -7 "), json!(-7));
        assert_eq!(one("i", "3.0"), json!(3), "an integral float is an integer");
        assert_eq!(one("i", "3.5"), json!("3.5"), "a fraction is not an integer: kept as text");
        assert_eq!(one("i", "three"), json!("three"));
        assert_eq!(one("n", "2.5"), json!(2.5));
        assert_eq!(one("n", "2"), json!(2));
        assert_eq!(one("n", "1e-07"), json!(1e-7));
        assert_eq!(one("b", "true"), json!(true));
        assert_eq!(one("b", "False"), json!(false));
        assert_eq!(one("b", "maybe"), json!("maybe"));
        assert_eq!(one("z", "null"), json!(null));
        assert_eq!(one("a", "[1, \"x\", null]"), json!([1, "x", null]));
        assert_eq!(one("o", "{\"k\": [1, {\"j\": true}]}"), json!({"k": [1, {"j": true}]}));
        assert_eq!(one("a", "{\"k\": 1}"), json!("{\"k\": 1}"), "an object is not an array");
        assert_eq!(one("a", "[1, "), json!("[1, "), "malformed JSON stays the text");
    }

    #[test]
    fn unions_enums_consts_refs_and_shapes() {
        // Optional[str]: the text is the string, except the exact text null
        assert_eq!(one("opt_s", "hello"), json!("hello"));
        assert_eq!(one("opt_s", "123"), json!("123"));
        assert_eq!(one("opt_s", "null"), json!(null));
        assert_eq!(one("opt_s", " hello "), json!(" hello "));
        assert_eq!(one("opt_i", "5"), json!(5));
        assert_eq!(one("opt_i", "null"), json!(null));
        assert_eq!(one("opt_i", "x"), json!("x"));
        assert_eq!(one("list_t", "5"), json!(5));
        assert_eq!(one("list_t", "null"), json!(null));
        // string | integer: a text that is an integer is the integer, any other text the string
        assert_eq!(one("si", "42"), json!(42));
        assert_eq!(one("si", "forty-two"), json!("forty-two"));
        assert_eq!(one("si", "\"42\""), json!("\"42\""), "quotes are text: strings are written bare");
        assert_eq!(one("one", "true"), json!(true));
        assert_eq!(one("one", "[\"a\", \"b\"]"), json!(["a", "b"]));
        assert_eq!(one("e_s", "a"), json!("a"));
        assert_eq!(one("e_i", "2"), json!(2));
        assert_eq!(one("c", "7"), json!(7));
        assert_eq!(one("shaped", "{\"x\": 1}"), json!({"x": 1}));
        assert_eq!(one("ref", "c"), json!("c"));
        assert_eq!(one("ref_list", "[\"c\", \"f\"]"), json!(["c", "f"]));
        assert_eq!(one("loop", "5"), json!(5), "a self-referencing $ref is unknown, not a hang: JSON when it is JSON");
        assert_eq!(one("free", "5"), json!(5));
        assert_eq!(one("free", "text"), json!("text"));
        assert_eq!(one("free", "\"q\""), json!("\"q\""));
        // not declared at all
        assert_eq!(one("undeclared", "5"), json!(5));
        assert_eq!(one("undeclared", "[1]"), json!([1]));
        assert_eq!(one("undeclared", "words"), json!("words"));
    }

    #[test]
    fn a_function_the_request_did_not_declare_still_parses_with_the_untyped_rule() {
        let c = parse_call(
            "g<arg_key>x</arg_key><arg_value>5</arg_value><arg_key>y</arg_key><arg_value>five</arg_value>",
            &ToolSchemas::default(),
        )
        .unwrap();
        assert_eq!(c.arguments, r#"{"x":5,"y":"five"}"#);
    }

    #[test]
    fn zero_arguments_whitespace_and_repeated_keys() {
        assert_eq!(parse_call("get_time", &schemas()).unwrap(), ToolCall { name: "get_time".into(), arguments: "{}".into() });
        assert_eq!(parse_call(" get_time\n", &schemas()).unwrap().name, "get_time");
        // the newline-separated layout of other GLM releases
        let c = parse_call(
            "f\n<arg_key>s</arg_key>\n<arg_value>x</arg_value>\n<arg_key>i</arg_key>\n<arg_value>2</arg_value>\n",
            &schemas(),
        )
        .unwrap();
        assert_eq!(c.arguments, r#"{"s":"x","i":2}"#);
        // the later value of a repeated key wins, in the first one's place
        assert_eq!(
            parse_call("f<arg_key>i</arg_key><arg_value>1</arg_value><arg_key>s</arg_key><arg_value>x</arg_value><arg_key>i</arg_key><arg_value>2</arg_value>", &schemas()).unwrap().arguments,
            r#"{"i":2,"s":"x"}"#
        );
        // a string value may contain tag-like text that is not the closing tag
        assert_eq!(one("s", "<b>x</b> <arg_key> a < b"), json!("<b>x</b> <arg_key> a < b"));
    }

    #[test]
    fn malformed_blocks_are_not_calls() {
        for bad in [
            "",
            "   ",
            "f<arg_key>k</arg_key>",                                                // no value
            "f<arg_key>k</arg_key><arg_value>v",                                    // value never closed
            "f<arg_key>k<arg_value>v</arg_value>",                                  // key never closed
            "f<arg_value>v</arg_value>",                                            // value without a key
            "f<arg_key></arg_key><arg_value>v</arg_value>",                         // empty key
            "f<arg_key>k</arg_key>text<arg_value>v</arg_value>",                    // text between the tags
            "f<arg_key>k</arg_key><arg_value>v</arg_value> trailing words",         // text after the last value
            "I will call the tool now<arg_key>k</arg_key><arg_value>v</arg_value>", // prose instead of a name
            "<arg_key>k</arg_key><arg_value>v</arg_value>",                         // no name
            "na me<arg_key>k</arg_key><arg_value>v</arg_value>",                    // a name with a space
            "f<arg_key>a</arg_key><arg_value>1</arg_value>junk</arg_key><arg_value>2</arg_value>", // stray text, then a pair that reads as one
        ] {
            assert!(parse_call(bad, &schemas()).is_none(), "{bad:?} must not be a call");
        }
    }
}
