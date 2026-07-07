//! Transient typed logical composition over independently learned pathways.
//!
//! The workspace owns no long-term knowledge. Callers copy grounded relations
//! from the EEM/fabric into it, run deterministic joins, inspect provenance,
//! and discard it. Only externally observed outcomes may later consolidate a
//! derived relation into the brain.

use std::collections::{BTreeMap, BTreeSet};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TypedValue {
    pub kind: String,
    pub value: String,
}

impl TypedValue {
    pub fn new(kind: impl Into<String>, value: impl Into<String>) -> Self {
        Self { kind: kind.into(), value: value.into() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternValue {
    Constant(TypedValue),
    Variable { name: String, kind: String },
}

impl PatternValue {
    pub fn var(name: impl Into<String>, kind: impl Into<String>) -> Self {
        Self::Variable { name: name.into(), kind: kind.into() }
    }
    pub fn constant(kind: impl Into<String>, value: impl Into<String>) -> Self {
        Self::Constant(TypedValue::new(kind, value))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationPattern {
    pub predicate: String,
    pub arguments: Vec<PatternValue>,
}

impl RelationPattern {
    pub fn new(predicate: impl Into<String>, arguments: Vec<PatternValue>) -> Self {
        Self { predicate: predicate.into(), arguments }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroundedRelation {
    pub predicate: String,
    pub arguments: Vec<TypedValue>,
    pub confidence: f32,
    pub provenance: BTreeSet<String>,
}

impl GroundedRelation {
    pub fn new(predicate: impl Into<String>, arguments: Vec<TypedValue>,
               confidence: f32, source: impl Into<String>) -> Self {
        Self { predicate: predicate.into(), arguments,
               confidence: confidence.clamp(0.0, 1.0),
               provenance: BTreeSet::from([source.into()]) }
    }
    fn key(&self) -> (String, Vec<TypedValue>) {
        (self.predicate.clone(), self.arguments.clone())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositionRule {
    pub name: String,
    pub premises: Vec<RelationPattern>,
    pub conclusion: RelationPattern,
}

#[derive(Debug, Clone, Default)]
pub struct TransientWorkspace {
    facts: Vec<GroundedRelation>,
}

type Bindings = BTreeMap<String, TypedValue>;

impl TransientWorkspace {
    pub fn new(facts: impl IntoIterator<Item = GroundedRelation>) -> Self {
        let mut workspace = Self::default();
        for fact in facts { workspace.insert(fact); }
        workspace
    }

    pub fn facts(&self) -> &[GroundedRelation] { &self.facts }

    pub fn query(&self, predicate: &str) -> impl Iterator<Item = &GroundedRelation> {
        self.facts.iter().filter(move |fact| fact.predicate == predicate)
    }

    /// Deterministic forward chaining. No reference to the source EEM/Brain is
    /// retained or mutated, so all derived facts disappear with this value.
    pub fn resolve(&mut self, rules: &[CompositionRule], max_rounds: usize) {
        for _ in 0..max_rounds {
            let snapshot = self.facts.clone();
            let mut derived = Vec::new();
            for rule in rules {
                let mut matches = Vec::new();
                join_premises(&rule.premises, &snapshot, 0, Bindings::new(),
                              1.0, BTreeSet::new(), &mut matches);
                for (bindings, confidence, mut provenance) in matches {
                    let Some(arguments) = instantiate(&rule.conclusion.arguments, &bindings) else { continue };
                    provenance.insert(format!("rule:{}", rule.name));
                    derived.push(GroundedRelation {
                        predicate: rule.conclusion.predicate.clone(), arguments,
                        confidence, provenance,
                    });
                }
            }
            let before = self.facts.len();
            for fact in derived { self.insert(fact); }
            if self.facts.len() == before { break; }
        }
    }

    fn insert(&mut self, fact: GroundedRelation) {
        if let Some(existing) = self.facts.iter_mut().find(|f| f.key() == fact.key()) {
            if fact.confidence > existing.confidence { existing.confidence = fact.confidence; }
            existing.provenance.extend(fact.provenance);
        } else {
            self.facts.push(fact);
            self.facts.sort_by_key(GroundedRelation::key);
        }
    }
}

fn join_premises(premises: &[RelationPattern], facts: &[GroundedRelation], index: usize,
                 bindings: Bindings, confidence: f32, provenance: BTreeSet<String>,
                 out: &mut Vec<(Bindings, f32, BTreeSet<String>)>) {
    if index == premises.len() {
        out.push((bindings, confidence, provenance));
        return;
    }
    let pattern = &premises[index];
    for fact in facts.iter().filter(|f| f.predicate == pattern.predicate
                                   && f.arguments.len() == pattern.arguments.len()) {
        let mut next = bindings.clone();
        if !unify(&pattern.arguments, &fact.arguments, &mut next) { continue; }
        let mut sources = provenance.clone();
        sources.extend(fact.provenance.iter().cloned());
        join_premises(premises, facts, index + 1, next,
                      confidence.min(fact.confidence), sources, out);
    }
}

fn unify(pattern: &[PatternValue], values: &[TypedValue], bindings: &mut Bindings) -> bool {
    for (slot, value) in pattern.iter().zip(values) {
        match slot {
            PatternValue::Constant(expected) if expected != value => return false,
            PatternValue::Constant(_) => {}
            PatternValue::Variable { name: _, kind } if kind != &value.kind => return false,
            PatternValue::Variable { name, .. } => match bindings.get(name) {
                Some(bound) if bound != value => return false,
                Some(_) => {}
                None => { bindings.insert(name.clone(), value.clone()); }
            },
        }
    }
    true
}

fn instantiate(pattern: &[PatternValue], bindings: &Bindings) -> Option<Vec<TypedValue>> {
    pattern.iter().map(|slot| match slot {
        PatternValue::Constant(value) => Some(value.clone()),
        PatternValue::Variable { name, .. } => bindings.get(name).cloned(),
    }).collect()
}
