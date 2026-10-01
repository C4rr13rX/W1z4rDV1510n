//! Every `integrate_autonomous` call site takes its parameters from
//! `w1z4rd_brain::answer_path`, so the gated configuration and the shipped one
//! cannot drift apart again.
//!
//! # The defect this guards
//!
//! The operator rule (2026-09-30) is that the node's answer routes use the same
//! path the scorecard measures, and nothing enforced it. Measured 2026-10-01 by
//! walking the four call sites and collecting each call's paren-balanced
//! argument list:
//!
//! ```text
//! site                                       fabric_conf  depth  visit
//! crates/brain/examples/scorecard.rs:410       100.0        3     200   <- the GATED number
//! crates/node/src/brain_api.rs:5221              0.0        4     200   <- /brain/chat
//! crates/node/src/bin/brain_server.rs:2398       0.0        4     200   <- /chat
//! crates/node/src/api.rs:8287 (_tuned)           0.10       3      16   <- hypothesis loop
//! ```
//!
//! Three configurations for one question. `fabric_confidence_threshold` is
//! compared against a 0..1 confidence (`src/grounding.rs:15`, `:52`, `:75`), so
//! `100.0` is unreachable: the scorecard measured integration with the
//! fabric-confidence arm OFF while both chat routes ran it wide open at `0.0`.
//! Every `wrong = 0.0` in `docs/scorecard-baseline.json` was therefore a true
//! statement about a strictly more abstemious brain than the node ships.
//!
//! # Why a source scan
//!
//! The fault is a VALUE drifting across four files, which is structural and
//! per-site: a behavioural test of one route says nothing about the other
//! three, and a route that regresses to a literal would still answer
//! correctly on a small world. The same reasoning, and the same shape, as
//! `read_only_window_parity` in `crates/node/src/brain_api.rs`.
//!
//! # It cannot pass vacuously
//!
//! This repository's most expensive recurring mistake is a guard keyed on
//! evidence the source can no longer produce, and a scan is the easiest place
//! to make it: a renamed file, a moved route, or a regex that stops matching
//! all report zero violations. So the scan asserts it FOUND the call sites it
//! is pointed at — every file must contribute at least one — and
//! `every_file_is_readable_and_nonempty` fails on a path that has moved rather
//! than silently skipping it.
//!
//! Run it:
//!   cargo test -p w1z4rd-brain --test answer_path_parameters_are_shared

/// The files that may call `integrate_autonomous` or
/// `integrate_autonomous_tuned`, relative to the workspace root.
///
/// Read at runtime rather than `include_str!`d because they live in two other
/// crates, and a test in `crates/brain` cannot `include_str!` across a crate
/// boundary without hard-coding `../..` into the macro, which breaks the
/// moment anything moves. A missing path fails loudly below.
const CALL_SITE_FILES: [&str; 4] = [
    "crates/brain/examples/scorecard.rs",
    "crates/node/src/brain_api.rs",
    "crates/node/src/bin/brain_server.rs",
    "crates/node/src/api.rs",
];

/// `CARGO_MANIFEST_DIR` is `<root>/crates/brain`, so the workspace root is two
/// levels up. Derived rather than assumed from the CWD, which differs between
/// `cargo test` and a test binary run directly.
fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("CARGO_MANIFEST_DIR is <root>/crates/brain, so it has two parents")
        .to_path_buf()
}

/// One call's argument list: everything between the opening paren of
/// `integrate_autonomous(` / `integrate_autonomous_tuned(` and its match,
/// with line comments stripped.
///
/// Paren-balanced rather than line-based, because three of the four sites
/// spread their arguments over six lines and one puts them on one. Comments
/// are stripped FIRST: `/*chain_max_depth*/ 4` and `// default 16` both
/// contain digits that would otherwise read as arguments.
fn call_arguments(source: &str) -> Vec<String> {
    // Strip `//` to end-of-line and `/* ... */` anywhere. Done on the whole
    // file so a block comment spanning lines cannot leak.
    let mut clean = String::with_capacity(source.len());
    let bytes: Vec<char> = source.chars().collect();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '/' {
            while i < bytes.len() && bytes[i] != '\n' {
                i += 1;
            }
        } else if bytes[i] == '/' && i + 1 < bytes.len() && bytes[i + 1] == '*' {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == '*' && bytes[i + 1] == '/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len());
            // A block comment can sit BETWEEN two arguments, so it must leave
            // a separator behind or `/*a*/0.0,` would glue onto the previous
            // token.
            clean.push(' ');
        } else {
            clean.push(bytes[i]);
            i += 1;
        }
    }

    let mut out = Vec::new();
    // `integrate_autonomous_tuned` starts with `integrate_autonomous`, so
    // matching the shorter name finds both. The paren walk then takes
    // whichever argument list actually follows.
    let needle = "integrate_autonomous";
    let mut from = 0usize;
    while let Some(hit) = clean[from..].find(needle) {
        let at = from + hit;
        from = at + needle.len();
        let Some(open_rel) = clean[at..].find('(') else { continue };
        let open = at + open_rel;
        // Only a call: the chars between the name and the paren must be the
        // optional `_tuned` and nothing else. This is what keeps a doc
        // reference or a `fn integrate_autonomous(` declaration out.
        let between = &clean[at + needle.len()..open];
        if !between.is_empty() && between != "_tuned" {
            continue;
        }
        let mut depth = 0i32;
        let mut end = open;
        for (k, c) in clean[open..].char_indices() {
            if c == '(' {
                depth += 1;
            } else if c == ')' {
                depth -= 1;
                if depth == 0 {
                    end = open + k;
                    break;
                }
            }
        }
        if end > open {
            out.push(clean[open + 1..end].to_string());
        }
        from = end.max(from);
    }
    out
}

#[test]
fn every_file_is_readable_and_nonempty() {
    let root = workspace_root();
    for rel in CALL_SITE_FILES {
        let path = root.join(rel);
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{rel} must exist and be readable from {path:?}: {e}"));
        assert!(
            text.len() > 1024,
            "{rel} read back {} bytes, which is not a source file -- the path has moved and \
             the scan below would report zero violations against nothing",
            text.len(),
        );
    }
}

/// The scan: no call site may pass a bare numeric literal for any of the three
/// drifting parameters.
///
/// Checked as "no numeric literal in the argument list" rather than "the right
/// constant is present", because the failure mode is a NEW literal appearing,
/// and a site could satisfy the positive form while adding a fourth
/// configuration beside it. Pool ids are identifiers at all four sites
/// (`QUERY_POOL`, `POOL_TEXT`, `action_pool`, `POOL_ACTION_ID`), so a digit in
/// one of these argument lists is a tuning parameter and nothing else.
#[test]
fn no_call_site_passes_a_numeric_literal() {
    let root = workspace_root();
    let mut violations: Vec<String> = Vec::new();
    let mut calls_total = 0usize;
    for rel in CALL_SITE_FILES {
        let text = std::fs::read_to_string(root.join(rel)).expect("checked by the test above");
        let calls = call_arguments(&text);
        assert!(
            !calls.is_empty(),
            "{rel} contributed no integrate_autonomous call. Either the call moved to another \
             file -- update CALL_SITE_FILES -- or this scan has gone vacuous.",
        );
        calls_total += calls.len();
        for args in calls {
            // Split on top-level commas so the offending ARGUMENT is named
            // rather than the whole call.
            let mut depth = 0i32;
            let mut field = String::new();
            let mut fields: Vec<String> = Vec::new();
            for c in args.chars() {
                match c {
                    '(' | '[' | '<' => {
                        depth += 1;
                        field.push(c);
                    }
                    ')' | ']' | '>' => {
                        depth -= 1;
                        field.push(c);
                    }
                    ',' if depth == 0 => fields.push(std::mem::take(&mut field)),
                    _ => field.push(c),
                }
            }
            fields.push(field);
            for f in fields {
                let t = f.trim();
                if t.is_empty() {
                    continue;
                }
                // A literal STARTS with a digit. `POOL_ACTION_ID` and
                // `w1z4rd_brain::ANSWER_CHAIN_MAX_VISIT` contain digits and
                // start with letters.
                if t.starts_with(|c: char| c.is_ascii_digit()) {
                    violations.push(format!("{rel}: argument `{t}`"));
                }
            }
        }
    }
    assert!(
        calls_total >= CALL_SITE_FILES.len(),
        "found {calls_total} calls across {} files; the scan needs at least one each",
        CALL_SITE_FILES.len(),
    );
    assert!(
        violations.is_empty(),
        "integrate_autonomous parameters must come from w1z4rd_brain::answer_path so the gated \
         configuration and the shipped one cannot drift. {} literal argument(s):\n  {}",
        violations.len(),
        violations.join("\n  "),
    );
}

/// The scan is only worth running if it can fail. This proves it on a sample
/// of the exact text that was in `bin/brain_server.rs` before the fix.
#[test]
fn the_scan_detects_the_drift_it_was_written_for() {
    let before = r#"
    let xpool = brain.integrate_autonomous(
        POOL_TEXT,
        POOL_ACTION,
        /*fabric_threshold*/ 0.0,
        /*chain_max_depth*/ 4,
        /*chain_max_visit*/ 200,
    );
"#;
    let calls = call_arguments(before);
    assert_eq!(calls.len(), 1, "one call in the sample, got {calls:?}");
    let args = &calls[0];
    for literal in ["0.0", "4", "200"] {
        assert!(
            args.contains(literal),
            "the pre-fix literal {literal} must survive comment stripping, got `{args}`",
        );
    }
    assert!(
        !args.contains("fabric_threshold"),
        "`/*fabric_threshold*/` is a comment and must be stripped, got `{args}`",
    );

    let after = r#"
    let xpool = brain.integrate_autonomous(
        POOL_TEXT,
        POOL_ACTION,
        w1z4rd_brain::ANSWER_FABRIC_CONFIDENCE_THRESHOLD,
        w1z4rd_brain::ANSWER_CHAIN_MAX_DEPTH,
        w1z4rd_brain::ANSWER_CHAIN_MAX_VISIT,
    );
"#;
    let fixed = call_arguments(after);
    assert_eq!(fixed.len(), 1, "one call in the sample, got {fixed:?}");
    for field in fixed[0].split(',') {
        let t = field.trim();
        assert!(
            t.is_empty() || !t.starts_with(|c: char| c.is_ascii_digit()),
            "the fixed form must carry no literal, but `{t}` starts with a digit",
        );
    }
}

/// A declaration is not a call. `brain.rs` DECLARES both functions, and if the
/// walk counted a declaration's parameter list the scan would fire on
/// `f32`/`usize` type names forever -- or worse, pass because a declaration
/// has no digits and so inflate `calls_total` for free.
#[test]
fn a_declaration_is_not_counted_as_a_call() {
    let decl = r#"
    pub fn integrate_autonomous(
        &mut self,
        query_pool: PoolId,
        target_pool: PoolId,
        fabric_confidence_threshold: f32,
        chain_max_depth: usize,
        chain_max_visit: usize,
    ) -> AnswerWithGrounding {
"#;
    // A declaration DOES match the paren walk -- `integrate_autonomous(` is
    // the same text -- so what this pins is that it carries no digit-leading
    // argument and therefore cannot produce a false violation.
    for args in call_arguments(decl) {
        for field in args.split(',') {
            let t = field.trim();
            assert!(
                t.is_empty() || !t.starts_with(|c: char| c.is_ascii_digit()),
                "a declaration's parameter `{t}` must not read as a literal argument",
            );
        }
    }
    // And a prose mention with no parenthesis contributes nothing at all.
    assert!(
        call_arguments("/// `integrate_autonomous` hardcodes 0.70 and").is_empty()
            || !call_arguments("//! integrate_autonomous hardcodes 0.70")
                .iter()
                .any(|a| a.contains("0.70")),
        "a doc mention must not be scanned as a call",
    );
}
