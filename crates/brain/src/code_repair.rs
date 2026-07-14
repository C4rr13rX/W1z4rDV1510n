//! Deterministic executor for repair relations selected by the neurofabric.
//! The relation is the learned action; this module applies it to the current
//! raw source so novel identifiers and formatting are preserved.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CodeRepairRelation {
    ReplaceOperator { from: String, to: String },
    ReplaceText { from: String, to: String },
    GuardEmpty { fallback: String },
    IncrementStoredCount { amount: i64 },
    PowerSelf { exponent: usize },
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum CodeRepairError {
    #[error("repair relation is invalid JSON: {0}")]
    InvalidRelation(String),
    #[error("the selected relation does not match the current source")]
    NoApplicableSite,
}

pub fn apply_code_repair_relation(
    source: &str,
    encoded_relation: &[u8],
) -> Result<String, CodeRepairError> {
    let relation: CodeRepairRelation = serde_json::from_slice(encoded_relation)
        .map_err(|error| CodeRepairError::InvalidRelation(error.to_string()))?;
    apply_relation(source, &relation)
}

pub fn apply_relation(
    source: &str,
    relation: &CodeRepairRelation,
) -> Result<String, CodeRepairError> {
    match relation {
        CodeRepairRelation::ReplaceOperator { from, to } => {
            if from.is_empty() || from == to { return Err(CodeRepairError::NoApplicableSite); }
            let needle = format!(" {} ", from);
            let replacement = format!(" {} ", to);
            replace_once_in_relevant_line(source, &needle, &replacement)
        }
        CodeRepairRelation::ReplaceText { from, to } => {
            if from.is_empty() || from == to || !source.contains(from) {
                Err(CodeRepairError::NoApplicableSite)
            } else {
                Ok(source.replacen(from, to, 1))
            }
        }
        CodeRepairRelation::GuardEmpty { fallback } => {
            let parameter = first_python_parameter(source)
                .ok_or(CodeRepairError::NoApplicableSite)?;
            let mut changed = false;
            let lines = source.lines().map(|line| {
                if !changed {
                    let trimmed = line.trim_start();
                    if let Some(expression) = trimmed.strip_prefix("return ") {
                        if !expression.contains(" if ") {
                            let indent = &line[..line.len() - trimmed.len()];
                            changed = true;
                            return format!("{}return {} if {} else {}", indent, expression,
                                           parameter, fallback);
                        }
                    }
                }
                line.to_string()
            }).collect::<Vec<_>>().join("\n");
            if changed { Ok(preserve_final_newline(source, lines)) }
            else { Err(CodeRepairError::NoApplicableSite) }
        }
        CodeRepairRelation::IncrementStoredCount { amount } => {
            let suffix = format!(" + {}", amount);
            let mut changed = false;
            let lines = source.lines().map(|line| {
                if !changed && line.contains(".get(") && !line.trim_end().ends_with(&suffix) {
                    changed = true;
                    format!("{}{}", line, suffix)
                } else { line.to_string() }
            }).collect::<Vec<_>>().join("\n");
            if changed { Ok(preserve_final_newline(source, lines)) }
            else { Err(CodeRepairError::NoApplicableSite) }
        }
        CodeRepairRelation::PowerSelf { exponent } => {
            if *exponent < 2 { return Err(CodeRepairError::NoApplicableSite); }
            let parameter = first_python_parameter(source)
                .ok_or(CodeRepairError::NoApplicableSite)?;
            let expression = std::iter::repeat(parameter).take(*exponent)
                .collect::<Vec<_>>().join(" * ");
            let mut changed = false;
            let lines = source.lines().map(|line| {
                if !changed && line.trim_start().starts_with("return ") {
                    let indent = &line[..line.len() - line.trim_start().len()];
                    changed = true;
                    format!("{}return {}", indent, expression)
                } else { line.to_string() }
            }).collect::<Vec<_>>().join("\n");
            if changed { Ok(preserve_final_newline(source, lines)) }
            else { Err(CodeRepairError::NoApplicableSite) }
        }
    }
}

fn replace_once_in_relevant_line(
    source: &str,
    needle: &str,
    replacement: &str,
) -> Result<String, CodeRepairError> {
    let mut changed = false;
    let lines = source.lines().map(|line| {
        if !changed && (line.trim_start().starts_with("return ") || line.contains(" if "))
            && line.contains(needle)
        {
            changed = true;
            line.replacen(needle, replacement, 1)
        } else { line.to_string() }
    }).collect::<Vec<_>>().join("\n");
    if changed { Ok(preserve_final_newline(source, lines)) }
    else { Err(CodeRepairError::NoApplicableSite) }
}

fn first_python_parameter(source: &str) -> Option<&str> {
    let header = source.lines().find(|line| line.trim_start().starts_with("def "))?;
    let inside = header.split_once('(')?.1.split_once(')')?.0;
    inside.split(',').next().map(str::trim).filter(|value| !value.is_empty())
}

fn preserve_final_newline(source: &str, mut result: String) -> String {
    if source.ends_with('\n') { result.push('\n'); }
    result
}
