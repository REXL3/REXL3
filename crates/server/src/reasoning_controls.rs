//! The reasoning controls of a chat request, driven only by the model's checked [`ReasoningCapabilities`] (`reasoning_caps.rs`): which
//! `chat_template_kwargs` and top-level fields a request may use, how an effort level maps onto template variables, what the server does
//! when a request says nothing about thinking (`--default-thinking`), and the reasoning token budget. Nothing here names a family; see
//! `docs/REASONING_CAPABILITIES.md`.

use crate::api::ApiError;
use crate::field_policy::bad;
use crate::reasoning_caps::{Alias, Control, EffortLevel, Place, ReasoningCapabilities, VarRole};
use serde_json::{Map, Value};

/// What the server does when a chat request names neither the thinking switch nor an effort level (`--default-thinking`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DefaultThinking {
    /// Insert the switch set to false, when the model has a switch that closes the reasoning block (the server's behaviour since #11,
    /// and the default: nothing changes unless the flag is given).
    #[default]
    Off,
    /// Insert nothing: the template's own default decides (`ReasoningCapabilities::template_thinks_by_default`).
    Template,
    /// `--default-reasoning-effort LEVEL`: a request that names no effort, and does not turn thinking off, gets this level, as if it had asked for it
    /// (a declared level sets its variables; a template choice variable `reasoning_effort` is set to the value). The name is leaked once at
    /// boot so the type stays `Copy`.
    Effort(&'static str),
}

impl DefaultThinking {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "off" => Some(Self::Off),
            "template" => Some(Self::Template),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Template => "template",
            Self::Effort(_) => "effort",
        }
    }

    /// The effort level a request that names nothing gets, when `--default-reasoning-effort` is set.
    pub fn effort(self) -> Option<&'static str> {
        match self {
            Self::Effort(level) => Some(level),
            _ => None,
        }
    }
}

/// Check `--default-reasoning-effort LEVEL` against the model: it must reason and take `level`, either as a declared effort level or as a
/// value of its template's `reasoning_effort` choice variable. Returns the `DefaultThinking` to run with.
pub fn default_effort(caps: &ReasoningCapabilities, level: &str) -> Result<DefaultThinking, String> {
    if !caps.reasons {
        return Err("this model does not reason, so it has no reasoning effort to default".into());
    }
    let as_variable = caps.var("reasoning_effort").filter(|v| v.is_choice && v.role == VarRole::Control);
    let ok = caps.level(level).is_some() || as_variable.is_some_and(|v| v.ty.check(&Value::String(level.to_string())).is_ok());
    if !ok {
        let mut names: Vec<String> = caps.level_names().iter().map(|s| s.to_string()).collect();
        if let Some(v) = as_variable {
            names.push(format!("{} {}", v.name, v.ty.check(&Value::Null).unwrap_err()));
        }
        return Err(if names.is_empty() {
            "this model takes no reasoning effort".into()
        } else {
            format!("\"{level}\" is not one of this model's efforts: {}", names.join(", "))
        });
    }
    Ok(DefaultThinking::Effort(Box::leak(level.to_string().into_boxed_str())))
}

/// The kwargs a request renders with, and its reasoning budget in thinking tokens (`None`: no budget).
#[derive(Clone, Debug, PartialEq)]
pub struct Resolved {
    pub kwargs: Map<String, Value>,
    pub budget: Option<usize>,
}

/// The top-level reasoning fields of the OpenAI-compatible shape: a model that does not offer one refuses it by name (rather than as an
/// unknown field), and so does `/v1/completions`.
pub const TOP_LEVEL_FIELDS: &[&str] = &["reasoning_effort", "max_reasoning_tokens"];

fn non_null<'a>(body: &'a Value, key: &str) -> Option<&'a Value> {
    body.get(key).filter(|v| !v.is_null())
}

/// Every top-level reasoning field the server recognises for this model: the OpenAI-compatible ones and the model's own alias tables
/// (the field allow-list takes these; [`validate_kwargs`] refuses by name the ones the model does not offer).
pub fn known_top_level(caps: &ReasoningCapabilities) -> Vec<String> {
    let mut out: Vec<String> = TOP_LEVEL_FIELDS.iter().map(|s| s.to_string()).collect();
    for n in caps.top_level_names() {
        if !out.contains(&n) {
            out.push(n);
        }
    }
    out
}

/// Refuse the top-level reasoning fields in `names` where they do nothing (`/v1/completions`: no template).
pub fn refuse_top_level(body: &Value, names: &[String], why: &str) -> Result<(), ApiError> {
    for f in names {
        if non_null(body, f).is_some() {
            return Err(bad(f, format!("{f} is not supported {why}")));
        }
    }
    Ok(())
}

/// What a model takes, for a refusal: "; this model takes: a, b" (or that it takes none).
fn takes(caps: &ReasoningCapabilities) -> String {
    let fields = caps.request_fields();
    if fields.is_empty() {
        "; this model has no reasoning controls".into()
    } else {
        format!("; this model takes: {}", fields.join(", "))
    }
}

fn check_alias_value(a: &Alias, v: &Value, caps: &ReasoningCapabilities) -> Result<(), ApiError> {
    let ok = match a.control {
        Control::Effort => v.as_str().is_some_and(|s| caps.level(s).is_some()),
        Control::Budget => v.as_u64().is_some_and(|n| usize::try_from(n).is_ok()),
        Control::Switch => v.is_boolean(),
    };
    if ok {
        return Ok(());
    }
    let what = match a.control {
        Control::Effort => format!("must be one of {}", caps.level_names().join(", ")),
        Control::Budget => "must be a non-negative integer".into(),
        Control::Switch => "must be a boolean".into(),
    };
    Err(bad(a.param(), format!("{} {what}", a.display())))
}

/// Validate the request's reasoning fields under `caps` (the shape check `field_policy::validate_chat_with` runs before anything is
/// rendered): the top-level fields this model does not offer are refused by name, every `chat_template_kwargs` key must be one the model
/// takes, with the right type.
pub fn validate_kwargs(body: &Value, caps: &ReasoningCapabilities) -> Result<(), ApiError> {
    for f in known_top_level(caps) {
        if non_null(body, &f).is_some() && caps.alias(Place::TopLevel, &f).is_none() {
            return Err(bad(&f, format!("{f} is not supported by this server{}", takes(caps))));
        }
    }
    let Some(kw) = non_null(body, "chat_template_kwargs") else { return Ok(()) };
    let kw = kw.as_object().ok_or_else(|| bad("chat_template_kwargs", "chat_template_kwargs must be an object"))?;
    for (k, v) in kw {
        if let Some(a) = caps.alias(Place::Kwarg, k) {
            check_alias_value(a, v, caps)?;
            continue;
        }
        let ty = match (&caps.switch, caps.var(k)) {
            (Some(s), _) if s.var == *k => crate::reasoning_caps::VarType::Boolean,
            (_, Some(var)) => var.ty.clone(),
            // tools is supported only at the request root: no cap/tool_choice bypass via template overrides.
            _ => {
                let names = caps.kwarg_names();
                let list = if names.is_empty() { "none".to_string() } else { names.join(", ") };
                return Err(bad(
                    "chat_template_kwargs",
                    format!("unsupported chat_template_kwargs.{k}; this model's template takes: {list}"),
                ));
            }
        };
        ty.check(v).map_err(|why| bad("chat_template_kwargs", format!("chat_template_kwargs.{k} {why}")))?;
    }
    Ok(())
}

/// The kwargs a chat request renders with, and its budget. Validates as [`validate_kwargs`] does, so it may be called on its own.
pub fn resolve(body: &Value, caps: &ReasoningCapabilities, default: DefaultThinking) -> Result<Resolved, ApiError> {
    resolve_with(body, caps, default, true)
}

/// Render what a request that names nothing would render under `default`, so a default the template refuses (an effort its own check
/// rejects, for a template whose variable takes any string) stops the server at boot instead of failing every such request.
pub fn check_default_renders(
    format: &dyn crate::chat_format::ChatFormat,
    caps: &ReasoningCapabilities,
    default: DefaultThinking,
) -> Result<(), String> {
    let kwargs = resolve(&serde_json::json!({}), caps, default).map_err(|e| format!("{e:?}"))?.kwargs;
    let messages = serde_json::json!([{"role": "user", "content": "hi"}]);
    format.render(&messages, &kwargs).map(|_| ()).map_err(|e| format!("the template refuses it: {e}"))
}

/// The reasoning budget of a chat request alone: [`resolve`] without copying the request's `tools` into the kwargs.
pub fn budget(body: &Value, caps: &ReasoningCapabilities, default: DefaultThinking) -> Result<Option<usize>, ApiError> {
    resolve_with(body, caps, default, false).map(|r| r.budget)
}

/// Whether a request that says nothing about thinking thinks, under `default`.
pub fn default_thinks(caps: &ReasoningCapabilities, default: DefaultThinking) -> bool {
    if let DefaultThinking::Effort(level) = default {
        return caps.reasons && caps.level(level).map_or(caps.template_thinks_by_default, |l| l.thinks);
    }
    let closes = caps.switch.as_ref().is_some_and(|s| s.closes_block);
    caps.reasons && caps.template_thinks_by_default && !(default == DefaultThinking::Off && closes)
}

/// The values the request gives for `control`, through any of its aliases, in alias order (top-level fields as they come, the kwarg ones
/// from `kw`).
fn given<'a>(
    body: &'a Value,
    kw: &'a Map<String, Value>,
    caps: &'a ReasoningCapabilities,
    control: Control,
) -> Vec<(&'a Alias, &'a Value)> {
    caps.aliases()
        .filter(|a| a.control == control)
        .filter_map(|a| {
            let v = match a.place {
                Place::TopLevel => non_null(body, &a.name),
                Place::Kwarg => kw.get(&a.name).filter(|v| !v.is_null()),
            }?;
            Some((a, v))
        })
        .collect()
}

/// The one value several aliases of a control agree on, or a refusal naming the first two that differ.
fn agree<'a>(given: &[(&'a Alias, &'a Value)]) -> Result<Option<(&'a Alias, &'a Value)>, ApiError> {
    let Some(&(a, v)) = given.first() else { return Ok(None) };
    if let Some(&(b, w)) = given.iter().find(|(_, w)| *w != v) {
        let show = |x: &Value| x.as_str().map_or_else(|| x.to_string(), |s| format!("\"{s}\""));
        return Err(bad(a.param(), format!("{} is {} but {} is {}; give one", a.display(), show(v), b.display(), show(w))));
    }
    Ok(Some((a, v)))
}

fn resolve_with(
    body: &Value,
    caps: &ReasoningCapabilities,
    default: DefaultThinking,
    with_tools: bool,
) -> Result<Resolved, ApiError> {
    validate_kwargs(body, caps)?;
    let mut kw = if with_tools {
        crate::api::chat_template_kwargs_base(body)?
    } else {
        non_null(body, "chat_template_kwargs").and_then(Value::as_object).cloned().unwrap_or_default()
    };
    // the effort: every alias that gives one must agree (the top-level values are checked here, the kwarg ones already were)
    let efforts = given(body, &kw, caps, Control::Effort);
    for (a, v) in efforts.iter().filter(|(a, _)| a.place == Place::TopLevel) {
        check_alias_value(a, v, caps)?;
    }
    let effort: Option<(Alias, EffortLevel)> = agree(&efforts)?
        .map(|(a, v)| (a.clone(), v.as_str().and_then(|s| caps.level(s)).cloned().expect("validated against the levels")));
    let budgets = given(body, &kw, caps, Control::Budget);
    for (a, v) in &budgets {
        check_alias_value(a, v, caps)?;
    }
    let explicit_budget = agree(&budgets)?.map(|(_, v)| v.as_u64().and_then(|n| usize::try_from(n).ok()).expect("validated"));
    let budget_alias = budgets.first().map(|(a, _)| (*a).clone());
    let switches = given(body, &kw, caps, Control::Switch);
    for (a, v) in &switches {
        check_alias_value(a, v, caps)?;
    }
    let switch_alias = agree(&switches)?.map(|(a, v)| (a.clone(), v.clone()));
    // the aliases are the server's, not the template's: they leave the kwargs (a level puts back what it sets)
    for a in caps.aliases().filter(|a| a.place == Place::Kwarg) {
        kw.remove(&a.name);
    }
    if let (Some((a, v)), Some(s)) = (switch_alias, &caps.switch) {
        match kw.get(&s.var) {
            Some(w) if *w != v => {
                return Err(bad(a.param(), format!("{} is {v} but chat_template_kwargs.{} is {w}; give one", a.display(), s.var)))
            }
            _ => {
                kw.insert(s.var.clone(), v);
            }
        }
    }
    // `--default-reasoning-effort`: a request that names no effort (no level, no choice variable) and does not turn thinking off gets the
    // configured level. Naming the switch on is not naming an effort: `enable_thinking: true` alone still gets the default.
    let turned_off = caps.switch.as_ref().is_some_and(|s| kw.get(&s.var) == Some(&Value::Bool(false)));
    let named_variable = caps.variables.iter().any(|v| v.is_choice && kw.contains_key(&v.name));
    let default_effort = default.effort().filter(|_| !turned_off && !named_variable);
    let effort = match (effort, default_effort) {
        (None, Some(name)) => caps
            .level(name)
            .zip(caps.aliases().find(|a| a.control == Control::Effort))
            .map(|(level, alias)| (alias.clone(), level.clone())),
        (given, _) => given,
    };
    let mut budget = explicit_budget;
    let switch_off = caps.switch.as_ref().is_some_and(|s| kw.get(&s.var) == Some(&Value::Bool(false)));
    // An explicit switch-off with a level that thinks: thinking off, the level's variables not set, no budget. This is what such a request
    // rendered before levels existed (the template never read the effort), so it stays byte-identical.
    let off_wins = switch_off && effort.as_ref().is_some_and(|(_, l)| l.thinks);
    if off_wins {
        // the switches stay as the request gave them
    } else if let Some((a, level)) = &effort {
        for (key, want) in level.sets.iter().chain(level.requires.iter()) {
            match kw.get(key) {
                Some(got) if got != want => {
                    return Err(bad(
                        &a.name,
                        format!(
                            "{} \"{}\" sets chat_template_kwargs.{key} to {want}, but the request gives {got}; give one",
                            a.name, level.name
                        ),
                    ))
                }
                _ => {}
            }
        }
        for (key, v) in &level.sets {
            kw.insert(key.clone(), v.clone());
        }
        if budget.is_none() {
            budget = level.default_budget;
        }
    } else if let Some(name) = default_effort.filter(|name| caps.level(name).is_none()) {
        // a template that takes the effort as a plain variable: set it; thinking stays the template's own default
        if let Some(v) = caps.var("reasoning_effort").filter(|v| v.is_choice) {
            kw.insert(v.name.clone(), Value::String(name.to_string()));
        }
    } else if default == DefaultThinking::Off {
        if let Some(s) = caps.switch.as_ref().filter(|s| s.closes_block) {
            let chose = kw.contains_key(&s.var) || caps.variables.iter().any(|v| v.is_choice && kw.contains_key(&v.name));
            if !chose {
                kw.insert(s.var.clone(), Value::Bool(false));
            }
        }
    }
    // what the template will do
    let thinking = caps.reasons
        && match caps.switch.as_ref().filter(|s| s.closes_block) {
            Some(s) => kw.get(&s.var).and_then(Value::as_bool).unwrap_or(caps.template_thinks_by_default),
            None => caps.template_thinks_by_default,
        };
    if let Some(a) = budget_alias.filter(|_| !thinking) {
        return Err(bad(a.param(), format!("{} needs thinking on ({})", a.display(), how_to_think(caps))));
    }
    Ok(Resolved { kwargs: kw, budget })
}

/// How a request turns thinking on, for the budget refusal: `enable_thinking true, or a reasoning_effort other than "none"`.
fn how_to_think(caps: &ReasoningCapabilities) -> String {
    let mut ways = Vec::new();
    if let Some(s) = caps.switch.as_ref().filter(|s| s.closes_block) {
        ways.push(format!("{} true", s.var));
    }
    if let Some(a) = caps.aliases().find(|a| a.control == Control::Effort) {
        let off: Vec<String> = caps.levels.iter().filter(|l| !l.thinks).map(|l| format!("\"{}\"", l.name)).collect();
        ways.push(if off.is_empty() { format!("a {}", a.name) } else { format!("a {} other than {}", a.name, off.join(" or ")) });
    }
    if ways.is_empty() {
        "this model's template decides".into()
    } else {
        ways.join(", or ")
    }
}

/// The note a response carries when the client asked for thinking off and the prompt opens a reasoning block anyway: the switch is a
/// template variable, and a template that always opens a block cannot honour it. Nothing is invented for the prompt; the response says so.
pub fn thinking_ignored(body: &Value, caps: &ReasoningCapabilities, in_reasoning: bool) -> Option<String> {
    if !in_reasoning {
        return None;
    }
    let s = caps.switch.as_ref()?;
    let kw = non_null(body, "chat_template_kwargs").and_then(Value::as_object);
    let off_kw = kw.and_then(|k| k.get(&s.var)) == Some(&Value::Bool(false));
    let off_alias = caps.aliases().filter(|a| a.control == Control::Switch).any(|a| match a.place {
        Place::TopLevel => non_null(body, &a.name) == Some(&Value::Bool(false)),
        Place::Kwarg => kw.and_then(|k| k.get(&a.name)) == Some(&Value::Bool(false)),
    });
    (off_kw || off_alias)
        .then(|| format!("{}=false has no effect for this model: its chat template always opens a reasoning block", s.var))
}

/// Whether `var` is an advertised control (not an inert, accepted-only variable).
pub fn is_control(caps: &ReasoningCapabilities, var: &str) -> bool {
    caps.switch.as_ref().is_some_and(|s| s.var == var) || caps.var(var).is_some_and(|v| v.role == VarRole::Control)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formats::nemotron_omni::{reasoning_capabilities as nemotron, LOW_EFFORT_REASONING_TOKENS};
    use serde_json::json;

    fn r(body: Value) -> Result<Resolved, ApiError> {
        resolve(&body, &nemotron(), DefaultThinking::Off)
    }
    fn kw(body: Value) -> Value {
        Value::Object(r(body).unwrap().kwargs)
    }
    fn legacy() -> ReasoningCapabilities {
        ReasoningCapabilities::legacy()
    }

    #[test]
    fn efforts_map_onto_the_templates_switches() {
        assert_eq!(kw(json!({})), json!({"enable_thinking": false}), "the server default stays off");
        assert_eq!(kw(json!({"reasoning_effort": "none"})), json!({"enable_thinking": false}));
        assert_eq!(kw(json!({"reasoning_effort": "low"})), json!({"enable_thinking": true, "medium_effort": true}));
        assert_eq!(kw(json!({"reasoning_effort": "medium"})), json!({"enable_thinking": true, "medium_effort": true}));
        assert_eq!(kw(json!({"reasoning_effort": "high"})), json!({"enable_thinking": true}));
        assert_eq!(kw(json!({"reasoning_effort": "xhigh"})), json!({"enable_thinking": true}));
        // the kwargs route is the same
        assert_eq!(
            kw(json!({"chat_template_kwargs": {"reasoning_effort": "medium"}})),
            json!({"enable_thinking": true, "medium_effort": true})
        );
        assert_eq!(r(json!({"reasoning_effort": "low"})).unwrap().budget, Some(LOW_EFFORT_REASONING_TOKENS));
        assert_eq!(r(json!({"reasoning_effort": "low", "max_reasoning_tokens": 64})).unwrap().budget, Some(64));
        for e in ["none", "medium", "high", "xhigh"] {
            assert_eq!(r(json!({"reasoning_effort": e})).unwrap().budget, None, "{e}");
        }
        assert_eq!(r(json!({"reasoning_effort": "high", "max_reasoning_tokens": 0})).unwrap().budget, Some(0));
    }

    #[test]
    fn the_template_default_inserts_nothing() {
        let t = |b: Value| resolve(&b, &nemotron(), DefaultThinking::Template).unwrap();
        assert_eq!(Value::Object(t(json!({})).kwargs), json!({}));
        assert_eq!(t(json!({"max_reasoning_tokens": 5})).budget, Some(5), "the template's default is thinking on");
        let l = resolve(&json!({}), &legacy(), DefaultThinking::Template).unwrap();
        assert_eq!(Value::Object(l.kwargs), json!({}));
    }

    #[test]
    fn explicit_switches_pass_through_and_conflicts_are_refused() {
        assert_eq!(
            kw(
                json!({"chat_template_kwargs": {"enable_thinking": true, "medium_effort": true, "truncate_history_thinking": false}})
            ),
            json!({"enable_thinking": true, "medium_effort": true, "truncate_history_thinking": false})
        );
        // preserve_thinking is accepted and passed through unchanged (the template does not read it): byte-identical to before
        assert_eq!(
            kw(json!({"chat_template_kwargs": {"preserve_thinking": true}})),
            json!({"preserve_thinking": true, "enable_thinking": false})
        );
        // an explicit enable_thinking false wins over a level that thinks: what such a request always rendered (thinking off, no tag)
        for e in ["low", "medium", "high", "xhigh"] {
            for route in [json!({"reasoning_effort": e}), json!({"chat_template_kwargs": {"reasoning_effort": e}})] {
                let mut b = route.clone();
                b["chat_template_kwargs"]["enable_thinking"] = json!(false);
                let got = r(b.clone()).unwrap();
                assert_eq!(Value::Object(got.kwargs), json!({"enable_thinking": false}), "{b}");
                assert_eq!(got.budget, None, "{b}");
            }
        }
        for body in [
            json!({"reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": true}}),
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"enable_thinking": false}, "max_reasoning_tokens": 5}),
            json!({"reasoning_effort": "high", "chat_template_kwargs": {"medium_effort": true}}),
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"medium_effort": false}}),
            json!({"reasoning_effort": "low", "chat_template_kwargs": {"reasoning_effort": "high"}}),
        ] {
            assert_eq!(r(body.clone()).unwrap_err().status, 400, "{body}");
        }
        // the same value twice is fine
        assert!(r(
            json!({"reasoning_effort": "high", "chat_template_kwargs": {"reasoning_effort": "high", "enable_thinking": true}})
        )
        .is_ok());
    }

    #[test]
    fn unknown_keys_values_and_types_are_refused_listing_what_is_supported() {
        let e = r(json!({"chat_template_kwargs": {"thinking": true}})).unwrap_err();
        assert_eq!(e.status, 400);
        for k in ["enable_thinking", "medium_effort", "truncate_history_thinking", "preserve_thinking", "reasoning_effort"] {
            assert!(e.message.contains(k), "{}", e.message);
        }
        let e = r(json!({"reasoning_effort": "maximum"})).unwrap_err();
        assert!(e.message.contains("none, low, medium, high, xhigh"), "{}", e.message);
        assert_eq!(e.param.as_deref(), Some("reasoning_effort"));
        assert!(r(json!({"chat_template_kwargs": {"reasoning_effort": "minimal"}})).is_err());
        assert!(r(json!({"chat_template_kwargs": {"medium_effort": "yes"}})).is_err());
        assert!(r(json!({"chat_template_kwargs": {"tools": []}})).is_err(), "tools only at the request root");
        for b in [json!(-1), json!(1.5), json!("8")] {
            assert!(r(json!({"max_reasoning_tokens": b})).is_err());
        }
        let e = r(json!({"max_reasoning_tokens": 8})).unwrap_err();
        assert!(e.message.contains("needs thinking on"), "{}", e.message);
        assert!(r(json!({"max_reasoning_tokens": 8, "reasoning_effort": "none"})).is_err());
    }

    #[test]
    fn the_budget_alone_is_the_budget_of_the_whole_resolution() {
        let tools = json!([{"type": "function", "function": {"name": "f"}}]);
        for body in [
            json!({"tools": tools, "reasoning_effort": "low"}),
            json!({"tools": tools, "reasoning_effort": "high", "max_reasoning_tokens": 9}),
            json!({"chat_template_kwargs": {"reasoning_effort": "low"}, "max_reasoning_tokens": 2}),
            json!({}),
        ] {
            for d in [DefaultThinking::Off, DefaultThinking::Template] {
                let whole = resolve(&body, &nemotron(), d).unwrap().budget;
                assert_eq!(budget(&body, &nemotron(), d).unwrap(), whole, "{body}");
            }
        }
    }

    #[test]
    fn the_legacy_policy_is_unchanged() {
        let l = |b: Value| resolve(&b, &legacy(), DefaultThinking::Off);
        assert_eq!(Value::Object(l(json!({})).unwrap().kwargs), json!({"enable_thinking": false}));
        assert_eq!(
            Value::Object(l(json!({"chat_template_kwargs": {"reasoning_effort": "low"}})).unwrap().kwargs),
            json!({"reasoning_effort": "low"})
        );
        assert!(l(json!({"chat_template_kwargs": {"medium_effort": true}})).is_err());
        assert!(l(json!({"chat_template_kwargs": {"reasoning_effort": "none"}})).is_err());
        let e = l(json!({"reasoning_effort": "high"})).unwrap_err();
        assert!(e.message.starts_with("reasoning_effort is not supported by this server; this model takes: "), "{}", e.message);
        assert!(l(json!({"max_reasoning_tokens": 3})).is_err());
    }

    #[test]
    fn a_default_effort_is_what_a_request_that_names_nothing_gets() {
        let low = default_effort(&legacy(), "low").unwrap();
        assert_eq!(low.name(), "effort");
        assert_eq!(low.effort(), Some("low"));
        let go = |body: Value, d: DefaultThinking| Value::Object(resolve(&body, &legacy(), d).unwrap().kwargs);
        // a template that takes the effort as a plain variable: the variable is set, the switch is not touched, thinking is the template's
        assert_eq!(go(json!({}), low), json!({"reasoning_effort": "low"}));
        assert!(default_thinks(&legacy(), low));
        // thinking explicitly off keeps its own; the switch on without an effort is not naming an effort, so it gets the default
        assert_eq!(go(json!({"chat_template_kwargs": {"enable_thinking": false}}), low), json!({"enable_thinking": false}));
        assert_eq!(
            go(json!({"chat_template_kwargs": {"enable_thinking": true}}), low),
            json!({"enable_thinking": true, "reasoning_effort": "low"})
        );
        // a request that names an effort keeps its own
        assert_eq!(go(json!({"chat_template_kwargs": {"reasoning_effort": "high"}}), low), json!({"reasoning_effort": "high"}));
        // other template variables do not count as a choice
        assert_eq!(
            go(json!({"chat_template_kwargs": {"preserve_thinking": true}}), low),
            json!({"preserve_thinking": true, "reasoning_effort": "low"})
        );
        // without the flag nothing changes: the original policy
        assert_eq!(go(json!({}), DefaultThinking::Off), json!({"enable_thinking": false}));
    }

    #[test]
    fn a_default_effort_that_is_a_declared_level_acts_as_the_request_asking_for_it() {
        let caps = nemotron();
        let low = default_effort(&caps, "low").unwrap();
        let asked = Value::Object(resolve(&json!({"reasoning_effort": "low"}), &caps, DefaultThinking::Off).unwrap().kwargs);
        let defaulted = resolve(&json!({}), &caps, low).unwrap();
        assert_eq!(Value::Object(defaulted.kwargs), asked);
        assert_eq!(defaulted.budget, resolve(&json!({"reasoning_effort": "low"}), &caps, DefaultThinking::Off).unwrap().budget);
        // an explicit level or switch wins
        let high = resolve(&json!({"reasoning_effort": "high"}), &caps, low).unwrap();
        assert_eq!(high.kwargs, resolve(&json!({"reasoning_effort": "high"}), &caps, DefaultThinking::Off).unwrap().kwargs);
        let off = resolve(&json!({"chat_template_kwargs": {"enable_thinking": false}}), &caps, low).unwrap();
        assert_eq!(
            off.kwargs,
            resolve(&json!({"chat_template_kwargs": {"enable_thinking": false}}), &caps, DefaultThinking::Off).unwrap().kwargs
        );
    }

    #[test]
    fn a_default_effort_the_model_does_not_take_is_refused_at_boot() {
        assert!(default_effort(&legacy(), "extreme").unwrap_err().contains("not one of this model's efforts"));
        assert!(default_effort(&ReasoningCapabilities::none(), "low").unwrap_err().contains("does not reason"));
        assert!(default_effort(&nemotron(), "bogus").is_err());
    }
}
