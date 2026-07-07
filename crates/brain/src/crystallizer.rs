//! Generic structural role induction from repeated sensor frames.
//!
//! A sensor supplies a channel and typed tokens. Across two experiences with
//! the same shape, invariant positions define the relation's structure and
//! varying positions become typed roles. No domain vocabulary is hard-coded.

use serde::{Deserialize, Serialize};
use crate::workspace::{GroundedRelation, TypedValue};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFrame {
    pub channel: String,
    pub tokens: Vec<TypedValue>,
    pub provenance: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum TemplateSlot {
    Constant(TypedValue),
    Role { index: usize, kind: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RelationTemplate {
    channel: String,
    slots: Vec<TemplateSlot>,
    observations: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SemanticCrystallizer {
    seeds: Vec<SemanticFrame>,
    templates: Vec<RelationTemplate>,
}

impl SemanticCrystallizer {
    pub fn template_count(&self) -> usize { self.templates.len() }

    /// Observe one outcome-confirmed frame. Returns newly crystallized
    /// relations, including the seed experience when a template first forms.
    pub fn observe(&mut self, frame: SemanticFrame) -> Vec<GroundedRelation> {
        if let Some(index) = self.templates.iter().position(|t| matches_template(t, &frame)) {
            self.templates[index].observations = self.templates[index].observations.saturating_add(1);
            return vec![instantiate(&self.templates[index], &frame)];
        }

        let best = self.seeds.iter().enumerate()
            .filter(|(_, seed)| seed.channel == frame.channel && seed.tokens.len() == frame.tokens.len())
            .filter_map(|(index, seed)| induce(seed, &frame).map(|template| {
                let constants = template.slots.iter().filter(|s| matches!(s, TemplateSlot::Constant(_))).count();
                (constants, index, template)
            }))
            .max_by_key(|(constants, _, _)| *constants);
        if let Some((_, seed_index, template)) = best {
            let seed = self.seeds.remove(seed_index);
            let relations = vec![instantiate(&template, &seed), instantiate(&template, &frame)];
            self.templates.push(template);
            relations
        } else {
            self.seeds.push(frame);
            Vec::new()
        }
    }

    pub fn recognize(&self, frame: &SemanticFrame) -> Vec<GroundedRelation> {
        self.templates.iter().filter(|t| matches_template(t, frame))
            .map(|t| instantiate(t, frame)).collect()
    }
}

fn induce(left: &SemanticFrame, right: &SemanticFrame) -> Option<RelationTemplate> {
    let mut slots = Vec::with_capacity(left.tokens.len());
    let mut constants = 0;
    let mut roles = 0;
    for (index, (a, b)) in left.tokens.iter().zip(&right.tokens).enumerate() {
        if a == b {
            constants += 1;
            slots.push(TemplateSlot::Constant(a.clone()));
        } else if a.kind == b.kind {
            roles += 1;
            slots.push(TemplateSlot::Role { index, kind: a.kind.clone() });
        } else {
            return None;
        }
    }
    // Reject both exact duplicates and completely unconstrained coincidence.
    if constants == 0 || roles == 0 { return None; }
    Some(RelationTemplate { channel: left.channel.clone(), slots, observations: 2 })
}

fn matches_template(template: &RelationTemplate, frame: &SemanticFrame) -> bool {
    template.channel == frame.channel && template.slots.len() == frame.tokens.len()
        && template.slots.iter().zip(&frame.tokens).all(|(slot, value)| match slot {
            TemplateSlot::Constant(expected) => expected == value,
            TemplateSlot::Role { kind, .. } => kind == &value.kind,
        })
}

fn instantiate(template: &RelationTemplate, frame: &SemanticFrame) -> GroundedRelation {
    let arguments = template.slots.iter().filter_map(|slot| match slot {
        TemplateSlot::Role { index, .. } => frame.tokens.get(*index).cloned(),
        TemplateSlot::Constant(_) => None,
    }).collect();
    GroundedRelation::new(template.channel.clone(), arguments,
        1.0 - 1.0 / (template.observations as f32 + 1.0), frame.provenance.clone())
}
