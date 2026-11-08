//! What reasoning/thinking controls a model offers, as data: the [`ReasoningCapabilities`] descriptor a chat format declares, checked
//! against the model's own chat template and tokenizer at load ([`derive`], [`ReasoningSupport`]). The request path
//! (`reasoning_controls.rs`), the response splitter and reasoning-token count (`reasoning.rs`), the reasoning budget and the "thinking
//! ignored" note (`http.rs`) read only the checked descriptor; none of them names a family. See `docs/REASONING_CAPABILITIES.md`.
//!
//! A family declares its descriptor in its chat-format module (`ChatFormat::reasoning_capabilities`). A format that declares none but
//! exposes its template (`ChatFormat::chat_template`) gets a conservative descriptor inferred from the template ([`infer`]): only the
//! reasoning variables the template reads, no effort levels, a budget only when the closing marker is found and is one token. A format
//! that exposes neither gets the server's original policy ([`ReasoningCapabilities::legacy`]), byte for byte.

use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

/// The type of a template variable a request may set in `chat_template_kwargs`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VarType {
    Boolean,
    /// One of these strings.
    Enum {
        values: Vec<String>,
    },
    /// Any string: the template validates it (a value it rejects is refused with the template's own message).
    String,
}

impl VarType {
    /// `Ok` when `v` has this type; otherwise the reason, phrased after `chat_template_kwargs.NAME`.
    pub fn check(&self, v: &Value) -> Result<(), String> {
        match self {
            VarType::Boolean if v.is_boolean() => Ok(()),
            VarType::Boolean => Err("must be a boolean".into()),
            VarType::Enum { values } if v.as_str().is_some_and(|s| values.iter().any(|x| x == s)) => Ok(()),
            VarType::Enum { values } => Err(format!("must be one of {}", values.join(", "))),
            VarType::String if v.is_string() => Ok(()),
            VarType::String => Err("must be a string".into()),
        }
    }
}

/// What a declared variable is to the server.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum VarRole {
    /// A reasoning control the template reads: checked against the template at load and advertised.
    Control,
    /// Accepted (typed) and passed through for compatibility, never advertised: the template need not read it (a request that worked
    /// before keeps working, and renders as before).
    Inert,
}

/// One template variable a request may set through `chat_template_kwargs`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TemplateVar {
    pub name: String,
    #[serde(flatten)]
    pub ty: VarType,
    pub role: VarRole,
    /// What the template uses when the request does not give it, when the family knows (exposure and documentation only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub template_default: Option<Value>,
    /// Giving it is the client's own thinking choice: `--default-thinking off` then inserts nothing.
    #[serde(skip)]
    pub is_choice: bool,
}

impl TemplateVar {
    pub fn new(name: &str, ty: VarType) -> Self {
        Self { name: name.into(), ty, role: VarRole::Control, template_default: None, is_choice: false }
    }
    pub fn boolean(name: &str) -> Self {
        Self::new(name, VarType::Boolean)
    }
    pub fn one_of(name: &str, values: &[&str]) -> Self {
        Self::new(name, VarType::Enum { values: values.iter().map(|s| s.to_string()).collect() })
    }
    pub fn default_value(mut self, v: Value) -> Self {
        self.template_default = Some(v);
        self
    }
    pub fn inert(mut self) -> Self {
        self.role = VarRole::Inert;
        self
    }
    pub fn choice(mut self) -> Self {
        self.is_choice = true;
        self
    }
}

/// The template variable that turns thinking on and off.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ThinkingSwitch {
    /// Its name in `chat_template_kwargs` (`enable_thinking`).
    pub var: String,
    /// `false` closes the reasoning block in the prompt (the reply is all content). When the template always opens a block, the switch is
    /// still accepted (a client that sends it keeps working), the model thinks anyway and the response carries a note
    /// (`X-Rexl3-Note`); the server's default inserts nothing.
    pub closes_block: bool,
}

/// One effort level of `reasoning_effort` (or any [`Control::Effort`] alias) and what it does to the template.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EffortLevel {
    pub name: String,
    /// The prompt opens a reasoning block at this level.
    pub thinks: bool,
    /// Template variables the level sets (inserted into the kwargs). A request that gives one of them with another value is refused.
    pub sets: Map<String, Value>,
    /// Template variables the level requires but does not insert (the template's default already has them): a request that gives one with
    /// another value is refused.
    #[serde(skip_serializing_if = "Map::is_empty")]
    pub requires: Map<String, Value>,
    /// The reasoning budget (thinking tokens) of a request at this level that names none.
    pub default_budget: Option<usize>,
}

impl EffortLevel {
    pub fn new(name: &str, thinks: bool) -> Self {
        Self { name: name.into(), thinks, sets: Map::new(), requires: Map::new(), default_budget: None }
    }
    pub fn sets(mut self, var: &str, v: impl Into<Value>) -> Self {
        self.sets.insert(var.into(), v.into());
        self
    }
    pub fn requires(mut self, var: &str, v: impl Into<Value>) -> Self {
        self.requires.insert(var.into(), v.into());
        self
    }
    pub fn budget(mut self, tokens: usize) -> Self {
        self.default_budget = Some(tokens);
        self
    }
}

/// What a request field means to the reasoning pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    /// An effort level's name ([`EffortLevel`]).
    Effort,
    /// A hard reasoning budget: a non-negative integer of thinking tokens.
    Budget,
    /// The thinking switch, as a boolean ([`ThinkingSwitch`]).
    Switch,
}

/// Where a request field sits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Place {
    /// A top-level field of the request body.
    TopLevel,
    /// A key of `chat_template_kwargs` that is not handed to the template as such.
    Kwarg,
}

/// One external API's name for a control.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Alias {
    pub place: Place,
    pub name: String,
    pub control: Control,
}

impl Alias {
    /// How a message names it: `reasoning_effort`, `chat_template_kwargs.reasoning_effort`.
    pub fn display(&self) -> String {
        match self.place {
            Place::TopLevel => self.name.clone(),
            Place::Kwarg => format!("chat_template_kwargs.{}", self.name),
        }
    }
    /// The `param` of an error about it.
    pub fn param(&self) -> &str {
        match self.place {
            Place::TopLevel => &self.name,
            Place::Kwarg => "chat_template_kwargs",
        }
    }
}

/// The request fields one external API shape uses for the reasoning controls. More shapes are more tables in a descriptor: the request
/// pipeline, the field allow-list and the exposure read them, nothing else changes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AliasTable {
    pub name: String,
    pub aliases: Vec<Alias>,
}

impl AliasTable {
    pub fn new(name: &str, aliases: &[(Place, &str, Control)]) -> Self {
        Self {
            name: name.into(),
            aliases: aliases.iter().map(|&(place, n, control)| Alias { place, name: n.into(), control }).collect(),
        }
    }

    /// The OpenAI-compatible shape: top-level `reasoning_effort` (also accepted as `chat_template_kwargs.reasoning_effort`, the vLLM way)
    /// and `max_reasoning_tokens`.
    pub fn openai() -> Self {
        Self::new(
            "openai",
            &[
                (Place::TopLevel, "reasoning_effort", Control::Effort),
                (Place::Kwarg, "reasoning_effort", Control::Effort),
                (Place::TopLevel, "max_reasoning_tokens", Control::Budget),
            ],
        )
    }
}

/// The text that opens and closes the reasoning block.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Markers {
    pub open: String,
    pub close: String,
}

impl Markers {
    pub fn new(open: &str, close: &str) -> Self {
        Self { open: open.into(), close: close.into() }
    }
}

/// The reasoning/thinking controls a model offers. Owned by the chat-format layer: a family declares it (`ChatFormat::reasoning_capabilities`),
/// the server checks it against the template ([`derive`]) and drives every reasoning decision of a request from the result.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReasoningCapabilities {
    /// The model reasons at all.
    pub reasons: bool,
    pub markers: Option<Markers>,
    pub switch: Option<ThinkingSwitch>,
    /// The template's own default (the generation prompt opens a block when the request says nothing). Checked by a probe render.
    pub template_thinks_by_default: bool,
    /// The other template variables a request may set.
    pub variables: Vec<TemplateVar>,
    /// The effort levels, in the order they are listed.
    pub levels: Vec<EffortLevel>,
    /// A hard reasoning budget (`max_reasoning_tokens`): the engine forces the closing marker, so it must be one token.
    pub budget: bool,
    pub aliases: Vec<AliasTable>,
}

impl ReasoningCapabilities {
    /// A model with no reasoning: every reasoning field is refused.
    pub fn none() -> Self {
        Self {
            reasons: false,
            markers: None,
            switch: None,
            template_thinks_by_default: false,
            variables: Vec::new(),
            levels: Vec::new(),
            budget: false,
            aliases: Vec::new(),
        }
    }

    /// The server's original policy (before reasoning capabilities): `enable_thinking` and `preserve_thinking` (booleans) and
    /// `reasoning_effort` in {low, medium, high, xhigh} in `chat_template_kwargs`, handed to the template as they are; no top-level
    /// field, no budget. Used for a format that declares no descriptor and exposes no template.
    pub fn legacy() -> Self {
        Self {
            reasons: true,
            markers: Some(Markers::new("<think>", "</think>")),
            switch: Some(ThinkingSwitch { var: "enable_thinking".into(), closes_block: true }),
            template_thinks_by_default: true,
            variables: vec![
                TemplateVar::boolean("preserve_thinking"),
                TemplateVar::one_of("reasoning_effort", &["low", "medium", "high", "xhigh"]).choice(),
            ],
            levels: Vec::new(),
            budget: false,
            aliases: Vec::new(),
        }
    }

    pub fn var(&self, name: &str) -> Option<&TemplateVar> {
        self.variables.iter().find(|v| v.name == name)
    }

    pub fn level(&self, name: &str) -> Option<&EffortLevel> {
        self.levels.iter().find(|l| l.name == name)
    }

    pub fn aliases(&self) -> impl Iterator<Item = &Alias> {
        self.aliases.iter().flat_map(|t| t.aliases.iter())
    }

    pub fn alias(&self, place: Place, name: &str) -> Option<&Alias> {
        self.aliases().find(|a| a.place == place && a.name == name)
    }

    /// The `chat_template_kwargs` keys a request may give, in the order they are listed: the switch, the variables, the kwarg aliases.
    pub fn kwarg_names(&self) -> Vec<String> {
        let mut out: Vec<String> = self.switch.iter().map(|s| s.var.clone()).collect();
        out.extend(self.variables.iter().map(|v| v.name.clone()));
        for a in self.aliases().filter(|a| a.place == Place::Kwarg) {
            if !out.contains(&a.name) {
                out.push(a.name.clone());
            }
        }
        out
    }

    /// The top-level request fields this model takes.
    pub fn top_level_names(&self) -> Vec<String> {
        let mut out = Vec::new();
        for a in self.aliases().filter(|a| a.place == Place::TopLevel) {
            if !out.contains(&a.name) {
                out.push(a.name.clone());
            }
        }
        out
    }

    /// Every reasoning field a request may use with this model, as messages name them.
    pub fn request_fields(&self) -> Vec<String> {
        let mut out = self.top_level_names();
        out.extend(self.kwarg_names().into_iter().map(|k| format!("chat_template_kwargs.{k}")));
        out
    }

    /// The level names, in order.
    pub fn level_names(&self) -> Vec<&str> {
        self.levels.iter().map(|l| l.name.as_str()).collect()
    }
}

// ------------------------------------------------------------------------------------------------------------------ the check at load

/// Render one user message with these kwargs: whether the reply would start inside a reasoning block, or the template's error.
pub type Probe<'a> = dyn Fn(&Map<String, Value>) -> Result<bool, String> + 'a;

/// What the server learned about a model's template, for [`derive`]. Pure data and two callbacks, so the check runs without a server.
pub struct TemplateFacts<'a> {
    /// The top-level variables the template reads (`None`: no template to inspect).
    pub reads: Option<BTreeSet<String>>,
    /// The template's source, when there is one.
    pub source: Option<&'a str>,
    /// Render one user message with these kwargs: whether the reply would start inside a reasoning block, or the template's error.
    pub probe: &'a Probe<'a>,
    /// A marker's id, when it is exactly one token.
    pub token_id: &'a dyn Fn(&str) -> Option<u32>,
    /// The closing marker the chat format names (`ChatFormat::reasoning_close`), used when inferring.
    pub close_hint: &'a str,
}

/// Where the checked descriptor came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// The family declared it.
    Declared,
    /// Inferred from the template (the family declared none).
    Inferred,
    /// The server's original policy (no descriptor, no template to inspect).
    Legacy,
}

/// The outcome of the check.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Every claim of the declared descriptor holds on the template.
    Ok,
    /// Some claims do not hold: what the template cannot honour is not offered (fail-closed); `findings` say what.
    Mismatch,
    /// Some claims do not hold, and `--reasoning-unverified` keeps them offered anyway.
    Overridden,
    /// Inferred from the template.
    Inferred,
    /// There was no template to check against.
    Unchecked,
}

impl Verdict {
    pub fn name(self) -> &'static str {
        match self {
            Verdict::Ok => "ok",
            Verdict::Mismatch => "mismatch",
            Verdict::Overridden => "overridden",
            Verdict::Inferred => "inferred",
            Verdict::Unchecked => "unchecked",
        }
    }
}

/// A model's reasoning capabilities after the check: what the server offers, and why.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReasoningSupport {
    /// What the request path enforces and `/v1/models` advertises.
    pub caps: ReasoningCapabilities,
    pub source: Source,
    pub verdict: Verdict,
    /// Declared controls (variables, levels, the switch's ability to close the block) the template never reads or does not honour.
    pub unread: Vec<String>,
    /// Reasoning variables the template reads that the descriptor does not offer (informational: they are not accepted).
    pub undeclared: Vec<String>,
    /// Declared things not offered, each with the reason.
    pub findings: Vec<String>,
    pub open_id: Option<u32>,
    pub close_id: Option<u32>,
    /// Why a reasoning budget is not offered, when it is not.
    pub budget_refusal: Option<String>,
}

/// Reasoning variables commonly read by chat templates, and their type: the catalogue inference draws from (a variable a template reads
/// that is not here is never accepted from a request, so `tools` and the like cannot be overridden through `chat_template_kwargs`).
pub const CATALOGUE: &[(&str, &str)] = &[
    ("enable_thinking", "boolean"),
    ("reasoning_effort", "string"),
    ("medium_effort", "boolean"),
    ("truncate_history_thinking", "boolean"),
    ("preserve_thinking", "boolean"),
    ("clear_thinking", "boolean"),
];

/// The switch's name when a template reads it.
const SWITCH: &str = "enable_thinking";

fn kw(pairs: &[(&str, Value)]) -> Map<String, Value> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// Check `declared` against the template (`facts`) and resolve the markers' ids. `unverified` (`--reasoning-unverified`) keeps a declared
/// claim the template does not support offered (the verdict says so); the budget still needs a one-token closing marker, since the
/// engine forces that token.
pub fn derive(declared: Option<ReasoningCapabilities>, facts: &TemplateFacts<'_>, unverified: bool) -> ReasoningSupport {
    let (caps, source, verdict, unread, undeclared, findings) = match (declared, &facts.reads) {
        (None, None) => (ReasoningCapabilities::legacy(), Source::Legacy, Verdict::Unchecked, vec![], vec![], vec![]),
        (None, Some(reads)) => (infer(reads, facts), Source::Inferred, Verdict::Inferred, vec![], vec![], vec![]),
        (Some(d), None) => (d, Source::Declared, Verdict::Unchecked, vec![], vec![], vec![]),
        (Some(d), Some(reads)) => {
            let (checked, unread, findings) = check(&d, reads, facts);
            // the template variables the descriptor offers (not its aliases: an alias is the server's, never the template's)
            let mut known: BTreeSet<String> = d.switch.iter().map(|s| s.var.clone()).collect();
            known.extend(d.variables.iter().map(|v| v.name.clone()));
            for l in &d.levels {
                known.extend(l.sets.keys().cloned());
            }
            let undeclared = CATALOGUE
                .iter()
                .map(|(n, _)| *n)
                .filter(|n| reads.contains(*n) && !known.contains(*n))
                .map(String::from)
                .collect();
            match (findings.is_empty(), unverified) {
                (true, _) => (checked, Source::Declared, Verdict::Ok, unread, undeclared, findings),
                (false, false) => (checked, Source::Declared, Verdict::Mismatch, unread, undeclared, findings),
                (false, true) => {
                    // keep the declared claims; the template default is a fact, not a claim
                    let mut d = d;
                    d.template_thinks_by_default = checked.template_thinks_by_default;
                    (d, Source::Declared, Verdict::Overridden, unread, undeclared, findings)
                }
            }
        }
    };
    finish(caps, source, verdict, unread, undeclared, findings, facts)
}

/// Resolve the markers' ids and what follows from them (the budget, the aliases that need a level, a budget or a switch).
fn finish(
    mut caps: ReasoningCapabilities,
    source: Source,
    mut verdict: Verdict,
    unread: Vec<String>,
    undeclared: Vec<String>,
    mut findings: Vec<String>,
    facts: &TemplateFacts<'_>,
) -> ReasoningSupport {
    let (open_id, close_id) = match &caps.markers {
        Some(m) => ((facts.token_id)(&m.open), (facts.token_id)(&m.close)),
        None => (None, None),
    };
    let mut budget_refusal = None;
    if !caps.reasons {
        budget_refusal = Some("this model does not reason".to_string());
    } else if !caps.budget {
        budget_refusal = Some("this model's reasoning capabilities declare no reasoning budget".to_string());
    } else if close_id.is_none() {
        let close = caps.markers.as_ref().map_or("(none)", |m| m.close.as_str());
        budget_refusal =
            Some(format!("a reasoning budget needs the closing marker ({close}) to be one token, and it is not for this model"));
        findings.push(format!("budget: {}", budget_refusal.as_deref().unwrap_or_default()));
        if verdict == Verdict::Ok {
            verdict = Verdict::Mismatch;
        }
    }
    if budget_refusal.is_some() {
        caps.budget = false;
    }
    let (levels, budget, switch) = (!caps.levels.is_empty(), caps.budget, caps.switch.is_some());
    for t in &mut caps.aliases {
        t.aliases.retain(|a| match a.control {
            Control::Effort => levels,
            Control::Budget => budget,
            Control::Switch => switch,
        });
    }
    caps.aliases.retain(|t| !t.aliases.is_empty());
    ReasoningSupport { caps, source, verdict, unread, undeclared, findings, open_id, close_id, budget_refusal }
}

/// The declared descriptor with what the template cannot honour taken out, the names it never reads, and the findings.
fn check(
    d: &ReasoningCapabilities,
    reads: &BTreeSet<String>,
    facts: &TemplateFacts<'_>,
) -> (ReasoningCapabilities, Vec<String>, Vec<String>) {
    let mut c = d.clone();
    let (mut unread, mut findings) = (Vec::new(), Vec::new());
    // the template's own default
    match (facts.probe)(&Map::new()) {
        Ok(opens) if opens != c.template_thinks_by_default => {
            findings.push(format!(
                "template default: declared thinking {}, the template's generation prompt {}",
                if c.template_thinks_by_default { "on" } else { "off" },
                if opens { "opens a reasoning block" } else { "does not open a reasoning block" }
            ));
            c.template_thinks_by_default = opens;
        }
        Ok(_) => {}
        Err(e) => findings.push(format!("template default: the probe render failed: {e}")),
    }
    // the switch
    if let Some(s) = c.switch.as_mut().filter(|s| s.closes_block) {
        let honoured = if !reads.contains(&s.var) {
            unread.push(s.var.clone());
            findings.push(format!("{}: the template never reads it, so it cannot turn thinking off", s.var));
            false
        } else {
            match (facts.probe)(&kw(&[(&s.var, json!(false))])) {
                Ok(false) => true,
                Ok(true) => {
                    findings.push(format!("{}: false still opens a reasoning block", s.var));
                    false
                }
                Err(e) => {
                    findings.push(format!("{}: false does not render: {e}", s.var));
                    false
                }
            }
        };
        // fail-closed: the switch is still accepted, but the server no longer claims it turns thinking off (the note says so)
        s.closes_block = honoured;
    }
    // the controls
    c.variables.retain(|v| {
        let keep = v.role == VarRole::Inert || reads.contains(&v.name);
        if !keep {
            unread.push(v.name.clone());
            findings.push(format!("{}: the template never reads it", v.name));
        }
        keep
    });
    // the levels: every variable a level sets must be read, and the level must render as it claims
    c.levels.retain(|l| {
        if let Some(k) = l.sets.keys().find(|k| !reads.contains(*k)) {
            findings.push(format!("level {}: sets {k}, which the template never reads", l.name));
            return false;
        }
        match (facts.probe)(&l.sets) {
            Ok(opens) if opens == l.thinks => true,
            Ok(opens) => {
                findings.push(format!(
                    "level {}: claims thinking {}, the template renders {}",
                    l.name,
                    if l.thinks { "on" } else { "off" },
                    if opens { "an open block" } else { "no open block" }
                ));
                false
            }
            Err(e) => {
                findings.push(format!("level {}: the template refuses it: {e}", l.name));
                false
            }
        }
    });
    // the markers: the template must spell the closing marker (it is the text that splits a reply)
    if let (Some(m), Some(src)) = (&c.markers, facts.source) {
        if c.reasons && !src.contains(&m.close) {
            findings.push(format!("markers: the template never writes {}", m.close));
            c.budget = false;
        }
    }
    unread.dedup();
    (c, unread, findings)
}

/// The conservative descriptor of a template whose family declares none: only the catalogued reasoning variables it reads, no effort
/// levels, the switch only when `enable_thinking: false` renders a prompt that does not open a block, a budget only when the template
/// writes the format's closing marker (and [`finish`] finds it is one token).
pub fn infer(reads: &BTreeSet<String>, facts: &TemplateFacts<'_>) -> ReasoningCapabilities {
    let close = facts.close_hint;
    let open = close.replacen("</", "<", 1);
    let writes_markers = facts.source.is_some_and(|s| s.contains(close) && s.contains(&open));
    let default_opens = (facts.probe)(&Map::new()).unwrap_or(false);
    let on_opens = reads.contains(SWITCH) && (facts.probe)(&kw(&[(SWITCH, json!(true))])).unwrap_or(false);
    let reasons = writes_markers || default_opens || on_opens;
    if !reasons {
        return ReasoningCapabilities::none();
    }
    let switch = reads
        .contains(SWITCH)
        .then(|| ThinkingSwitch { var: SWITCH.into(), closes_block: (facts.probe)(&kw(&[(SWITCH, json!(false))])) == Ok(false) });
    let variables = CATALOGUE
        .iter()
        .filter(|(n, _)| *n != SWITCH && reads.contains(*n))
        .map(|(n, ty)| {
            let v = if *ty == "boolean" { TemplateVar::boolean(n) } else { TemplateVar::new(n, VarType::String) };
            // an effort-like variable is the client's own choice, as the original policy treated `reasoning_effort`
            if *n == "reasoning_effort" {
                v.choice()
            } else {
                v
            }
        })
        .collect();
    ReasoningCapabilities {
        reasons,
        markers: Some(Markers::new(&open, close)),
        switch,
        template_thinks_by_default: default_opens,
        variables,
        levels: Vec::new(),
        budget: writes_markers,
        aliases: vec![AliasTable::new("openai", &[(Place::TopLevel, "max_reasoning_tokens", Control::Budget)])],
    }
}

impl ReasoningSupport {
    /// Check a chat format's declared descriptor against its template and the tokenizer (the server's boot path).
    pub fn for_format(format: &dyn crate::chat_format::ChatFormat, tok: &crate::tokenizer::Tok, unverified: bool) -> Self {
        let template = format.chat_template();
        let messages = json!([{"role": "user", "content": "hi"}]);
        let probe =
            |k: &Map<String, Value>| format.render(&messages, k).map(|r| r.starts_in_reasoning).map_err(|e| e.to_string());
        let token_id = |t: &str| tok.token_to_id(t);
        let facts = TemplateFacts {
            reads: template.map(|t| t.variables()),
            source: template.map(|t| t.source()),
            probe: &probe,
            token_id: &token_id,
            close_hint: format.reasoning_close(),
        };
        derive(format.reasoning_capabilities(), &facts, unverified)
    }

    /// The closing marker, for the splitter (`</think>` when the model declares none).
    pub fn close(&self) -> &str {
        self.caps.markers.as_ref().map_or("</think>", |m| m.close.as_str())
    }

    /// The `rexl3 event=reasoning_capabilities` boot line.
    pub fn log_line(&self) -> String {
        let list = |v: &[String]| if v.is_empty() { "-".to_string() } else { v.join(",") };
        let c = &self.caps;
        let id = |i: Option<u32>| i.map_or("-".to_string(), |i| i.to_string());
        format!(
            "rexl3 event=reasoning_capabilities source={} verdict={} reasons={} open={} open_id={} close={} close_id={} switch={} \
             template_default={} controls={} levels={} budget={} top_level={} unread={} undeclared={} findings={}",
            serde_json::to_value(self.source).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default(),
            self.verdict.name(),
            c.reasons,
            c.markers.as_ref().map_or("-", |m| m.open.as_str()),
            id(self.open_id),
            c.markers.as_ref().map_or("-", |m| m.close.as_str()),
            id(self.close_id),
            c.switch.as_ref().map_or("-".to_string(), |s| format!("{}:{}", s.var, if s.closes_block { "closes" } else { "ignored" })),
            if c.template_thinks_by_default { "thinking" } else { "off" },
            list(&c.kwarg_names()),
            list(&c.level_names().iter().map(|s| s.to_string()).collect::<Vec<_>>()),
            c.budget,
            list(&c.top_level_names()),
            list(&self.unread),
            list(&self.undeclared),
            serde_json::to_string(&self.findings).unwrap_or_default(),
        )
    }

    /// The object `/v1/models` carries under `reasoning`: what a client may send and what it does.
    pub fn exposure(&self, default: crate::reasoning_controls::DefaultThinking) -> Value {
        let c = &self.caps;
        let mut controls: Vec<Value> = Vec::new();
        if let Some(s) = &c.switch {
            controls.push(json!({"name": s.var, "type": "boolean", "role": "switch", "closes_block": s.closes_block}));
        }
        for v in c.variables.iter().filter(|v| v.role == VarRole::Control) {
            controls.push(serde_json::to_value(v).unwrap_or(Value::Null));
        }
        let effective = crate::reasoning_controls::default_thinks(c, default);
        let mut default_json =
            json!({"template_thinks": c.template_thinks_by_default, "server": default.name(), "thinks": effective});
        if let Some(level) = default.effort() {
            default_json["effort"] = json!(level);
        }
        json!({
            "reasons": c.reasons,
            "source": self.source,
            "verdict": self.verdict.name(),
            "markers": c.markers.as_ref().map(|m| json!({"open": m.open, "close": m.close, "open_id": self.open_id, "close_id": self.close_id})),
            "controls": controls,
            "accepted_inert": c.variables.iter().filter(|v| v.role == VarRole::Inert).map(|v| v.name.clone()).collect::<Vec<_>>(),
            "levels": c.levels,
            "default": default_json,
            "budget": {"supported": c.budget, "refusal": self.budget_refusal},
            "request_fields": c.request_fields(),
            "aliases": c.aliases,
        })
    }

    /// The `rexl3_reasoning_info` labels: one series, always 1.
    pub fn metric_labels(&self, default: crate::reasoning_controls::DefaultThinking) -> Vec<(&'static str, String)> {
        let c = &self.caps;
        vec![
            ("source", serde_json::to_value(self.source).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default()),
            ("verdict", self.verdict.name().to_string()),
            ("reasons", c.reasons.to_string()),
            ("controls", c.kwarg_names().join(",")),
            ("levels", c.level_names().join(",")),
            ("budget", c.budget.to_string()),
            ("default_thinking", default.name().to_string()),
            ("close", c.markers.as_ref().map_or(String::new(), |m| m.close.clone())),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::template::ChatTemplate;

    fn facts_of<'a>(t: &'a ChatTemplate, probe: &'a Probe<'a>, ids: &'a dyn Fn(&str) -> Option<u32>) -> TemplateFacts<'a> {
        TemplateFacts { reads: Some(t.variables()), source: Some(t.source()), probe, token_id: ids, close_hint: "</think>" }
    }

    fn opens(t: &ChatTemplate) -> impl Fn(&Map<String, Value>) -> Result<bool, String> + '_ {
        move |k| {
            t.render(&json!([{"role": "user", "content": "hi"}]), true, k)
                .map(|s| s.trim_end().ends_with("<think>"))
                .map_err(|e| e.to_string())
        }
    }

    const NEMO: &str = include_str!("../../../testdata/nemotron_omni/chat_template.jinja");
    const QWEN: &str = include_str!("../tests/fixtures/qwen38_chat_template.jinja");

    #[test]
    fn the_scan_finds_variables_read_in_their_own_reassignment() {
        let n = ChatTemplate::new(NEMO).unwrap().variables();
        for v in ["enable_thinking", "medium_effort", "truncate_history_thinking", "messages", "tools"] {
            assert!(n.contains(v), "{v}: {n:?}");
        }
        assert!(!n.contains("preserve_thinking") && !n.contains("reasoning_effort"), "{n:?}");
        let q = ChatTemplate::new(QWEN).unwrap().variables();
        for v in ["enable_thinking", "reasoning_effort", "preserve_thinking"] {
            assert!(q.contains(v), "{v}: {q:?}");
        }
        let synth =
            ChatTemplate::new("{% set x = x if x is defined else 1 %}{% macro m() %}{{ y }}{% endmacro %}{{ m() }}").unwrap();
        let s = synth.variables();
        assert!(s.contains("x") && s.contains("y"), "{s:?}");
        // attributes, keyword arguments, filters, tests, loop targets and locals assigned before they are read are not context reads
        let t = ChatTemplate::new(
            "{% set ns = namespace(k=1) %}{{ ns.k }}{% for m in items %}{{ m | tojson(indent=2) }}{% endfor %}{% if z is defined %}{% endif %}",
        )
        .unwrap();
        let l = crate::template::lexical_reads(t.source());
        assert_eq!(l.iter().map(String::as_str).collect::<Vec<_>>(), ["items", "namespace", "z"], "{l:?}");
    }

    #[test]
    fn a_claim_the_template_cannot_honour_is_not_offered_unless_overridden() {
        let t = ChatTemplate::new(
            "{{ messages[0].content }}{% if enable_thinking is false %}<think></think>{% else %}<think>{% endif %}",
        )
        .unwrap();
        let p = opens(&t);
        let ids = |s: &str| (s == "</think>").then_some(7);
        let d = ReasoningCapabilities {
            variables: vec![TemplateVar::boolean("deep_mode")],
            levels: vec![
                EffortLevel::new("deep", true).sets("deep_mode", true),
                EffortLevel::new("off", false).sets("enable_thinking", false),
            ],
            aliases: vec![AliasTable::openai()],
            budget: true,
            ..ReasoningCapabilities::legacy()
        };
        let s = derive(Some(d.clone()), &facts_of(&t, &p, &ids), false);
        assert_eq!(s.verdict, Verdict::Mismatch);
        assert_eq!(s.unread, vec!["deep_mode"]);
        assert!(s.caps.var("deep_mode").is_none());
        assert_eq!(s.caps.level_names(), vec!["off"]);
        assert_eq!(s.close_id, Some(7));
        assert!(s.caps.budget);
        let o = derive(Some(d), &facts_of(&t, &p, &ids), true);
        assert_eq!(o.verdict, Verdict::Overridden);
        assert!(o.caps.var("deep_mode").is_some() && o.caps.level("deep").is_some());
        assert!(!o.findings.is_empty());
    }

    #[test]
    fn no_template_is_the_original_policy_and_a_plain_template_reasons_not() {
        let p = |_: &Map<String, Value>| Ok(false);
        let ids = |_: &str| None;
        let none = TemplateFacts { reads: None, source: None, probe: &p, token_id: &ids, close_hint: "</think>" };
        let s = derive(None, &none, false);
        assert_eq!((s.source, s.verdict), (Source::Legacy, Verdict::Unchecked));
        assert_eq!(s.caps, ReasoningCapabilities::legacy());
        let t = ChatTemplate::new("{% for m in messages %}{{ m.content }}{% endfor %}").unwrap();
        let p = opens(&t);
        let s = derive(None, &facts_of(&t, &p, &ids), false);
        assert_eq!(s.verdict, Verdict::Inferred);
        assert!(!s.caps.reasons && s.caps.request_fields().is_empty() && !s.caps.budget);
    }
}
