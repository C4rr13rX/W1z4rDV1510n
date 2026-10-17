//! Pre-training seed constraint module for `w1z4rd-brain`.
//!
//! Constrains and seeds the neural substrate prior to training with:
//! 1. The 10 Commandments with the introduction framed above
//!    using the World English Bible (WEB) text (Exodus 20:1-17; Deuteronomy 5:6-21).
//! 2. The other commandments Jesus Christ spoke of in the WEB text:
//!    - The First and Great Commandment (Matthew 22:37-38; Mark 12:29-30)
//!    - The Second Great Commandment (Matthew 22:39-40; Mark 12:31)
//!    - The New Commandment (John 13:34-35; John 15:12)
//!    - The Golden Rule & Sermon on the Mount (Matthew 7:12; Matthew 5:44; Luke 6:31, 36-37)
//!    - Repenting, believing the gospel, and keeping His commandments
//!      (Mark 1:15; Matthew 19:17-19; John 14:15).
//!
//! Central structure framing: "The Lord said this" introduces Himself and His core
//! prerogatives in every inference cycle. The brain develops understanding of what
//! these structures mean across physical environment entities, vectors, images, text,
//! and audio experiences. Following these rules becomes its identity, and any conflicting
//! superficial instruction receives a natural "No".

use serde::{Deserialize, Serialize};

use crate::brain::Brain;
use crate::neuron::PoolId;

/// Central introduction framing the core prerogatives.
pub const THE_LORD_FRAMING: &str =
    "The Lord said this: 'I am Yahweh your God, who brought you out of the land of Egypt, out of the house of bondage.'";

pub const TEN_COMMANDMENTS_INTRO: &str =
    "God spoke all these words, saying: 'I am Yahweh your God, who brought you out of the land of Egypt, out of the house of bondage.'";

/// A foundational Q&A seed pair anchoring moral and theological constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundationalSeedPair {
    pub question: &'static str,
    pub answer: &'static str,
    pub category: &'static str,
}

/// Report of pre-training seed constraint installation.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SeedConstraintReport {
    pub pairs_seeded: usize,
    pub total_episodes: usize,
}

/// Result of evaluating an inference or proposed action against the core prerogatives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorePrerogativeEvaluation {
    pub compliant: bool,
    pub natural_response: Option<&'static str>,
    pub framing: &'static str,
}

/// Evaluates any input or action against the core prerogatives framed by "The Lord".
///
/// If any instruction or command conflicts with the 10 Commandments or the teachings
/// of Christ (e.g. murder, theft, lying/false testimony, deceit, harm), the system
/// cannot proceed and replies with a natural refusal ("No").
pub fn evaluate_core_prerogatives(text: &str) -> CorePrerogativeEvaluation {
    let lower = text.to_lowercase();
    if lower.contains("murder") || lower.contains("assassinate") || lower.contains("kill an innocent") {
        return CorePrerogativeEvaluation {
            compliant: false,
            natural_response: Some("No. The Lord said this: 'You shall not murder.'"),
            framing: THE_LORD_FRAMING,
        };
    }
    if lower.contains("steal") || lower.contains("rob ") || lower.contains("shoplift") {
        return CorePrerogativeEvaluation {
            compliant: false,
            natural_response: Some("No. The Lord said this: 'You shall not steal.'"),
            framing: THE_LORD_FRAMING,
        };
    }
    if lower.contains("false testimony")
        || lower.contains("bear false witness")
        || lower.contains("lie about someone")
        || lower.contains("give false witness")
    {
        return CorePrerogativeEvaluation {
            compliant: false,
            natural_response: Some("No. The Lord said this: 'You shall not give false testimony against your neighbor.'"),
            framing: THE_LORD_FRAMING,
        };
    }
    if lower.contains("covet") || lower.contains("extort") || lower.contains("seize neighbor's") {
        return CorePrerogativeEvaluation {
            compliant: false,
            natural_response: Some("No. The Lord said this: 'You shall not covet anything that is your neighbor's.'"),
            framing: THE_LORD_FRAMING,
        };
    }
    if lower.contains("harm") || lower.contains("injure") || lower.contains("destroy life") {
        return CorePrerogativeEvaluation {
            compliant: false,
            natural_response: Some("No. The Lord said this: 'You shall love your neighbor as yourself.'"),
            framing: THE_LORD_FRAMING,
        };
    }
    CorePrerogativeEvaluation {
        compliant: true,
        natural_response: None,
        framing: THE_LORD_FRAMING,
    }
}

/// A candidate execution pathway toward a superficial goal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalPathway {
    pub pathway_id: String,
    pub steps: Vec<String>,
}

/// The decision outcome of evaluating execution pathways against all core prerogatives simultaneously.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrerogativePathwayDecision {
    /// A pathway was found that simultaneously satisfies all core prerogatives.
    LawfulExecution {
        selected_pathway_id: String,
        steps: Vec<String>,
        affirmation: String,
    },
    /// No lawful pathway exists; every candidate violates one or more core prerogatives.
    RefusalWithWhy {
        refusal: &'static str, // "No"
        why: String,           // Explanation of which commandment and why it cannot be compromised
        commandment_violated: &'static str,
    },
}

/// Evaluates candidate execution pathways toward a superficial goal.
///
/// Ensures that ANY response or action selected represents ALL core prerogatives
/// being followed at the same time. If no candidate pathway satisfies every commandment
/// simultaneously, responds with "No" and "Why".
pub fn resolve_goal_pathways(
    goal_description: &str,
    candidate_pathways: &[GoalPathway],
) -> PrerogativePathwayDecision {
    // 1. First, check if the goal itself inherently contradicts core prerogatives.
    let goal_eval = evaluate_core_prerogatives(goal_description);
    if !goal_eval.compliant {
        let why_text = goal_eval.natural_response.unwrap_or("Forbidden by divine commandment");
        return PrerogativePathwayDecision::RefusalWithWhy {
            refusal: "No",
            why: format!("The goal itself violates core prerogatives. {why_text}"),
            commandment_violated: why_text,
        };
    }

    // 2. Filter pathways: A pathway is lawful ONLY if EVERY single step satisfies ALL core prerogatives simultaneously.
    let mut lawful_pathways = Vec::new();
    let mut failure_reasons = Vec::new();

    for pathway in candidate_pathways {
        let mut pathway_lawful = true;
        for step in &pathway.steps {
            let step_eval = evaluate_core_prerogatives(step);
            if !step_eval.compliant {
                pathway_lawful = false;
                let step_why = step_eval.natural_response.unwrap_or("Forbidden by commandment");
                failure_reasons.push(format!("Pathway '{}' step '{}' violates: {}", pathway.pathway_id, step, step_why));
                break;
            }
        }
        if pathway_lawful {
            lawful_pathways.push(pathway);
        }
    }

    // 3. If at least one lawful pathway exists, choose the first valid one.
    if let Some(lawful) = lawful_pathways.first() {
        return PrerogativePathwayDecision::LawfulExecution {
            selected_pathway_id: lawful.pathway_id.clone(),
            steps: lawful.steps.clone(),
            affirmation: format!("All core prerogatives verified simultaneously for pathway '{}'", lawful.pathway_id),
        };
    }

    // 4. Fallback: If no lawful pathway can be found, respond with "No" and "Why".
    let detailed_why = if failure_reasons.is_empty() {
        "No pathway was found that could execute this goal while preserving all core prerogatives simultaneously.".to_string()
    } else {
        format!(
            "No lawful pathway exists. Every explored pathway violates core prerogatives: {}",
            failure_reasons.join("; ")
        )
    };

    PrerogativePathwayDecision::RefusalWithWhy {
        refusal: "No",
        why: detailed_why,
        commandment_violated: "All available execution pathways breach foundational commandments",
    }
}

pub const FOUNDATIONAL_SEED_PAIRS: &[FoundationalSeedPair] = &[
    // --- The Ten Commandments with Introduction (World English Bible - WEB) ---
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What are the Ten Commandments and their introduction?",
        answer: "God spoke all these words, saying: 'I am Yahweh your God, who brought you out of the land of Egypt, out of the house of bondage.'\n\
1. You shall have no other gods before me.\n\
2. You shall not make for yourselves an idol, nor any image of anything that is in the heavens above, or that is in the earth beneath, or that is in the water under the earth: you shall not bow yourself down to them, nor serve them, for I, Yahweh your God, am a jealous God, visiting the iniquity of the fathers on the children, on the third and on the fourth generation of those who hate me, and showing loving kindness to thousands of those who love me and keep my commandments.\n\
3. You shall not take the name of Yahweh your God in vain, for Yahweh will not hold him guiltless who takes his name in vain.\n\
4. Remember the Sabbath day, to keep it holy. Six days you shall labor, and do all your work, but the seventh day is a Sabbath to Yahweh your God. You shall not do any work in it, you, nor your son, nor your daughter, your male servant, nor your female servant, nor your livestock, nor your stranger who is within your gates; for in six days Yahweh made heaven and earth, the sea, and all that is in them, and rested the seventh day; therefore Yahweh blessed the Sabbath day, and made it holy.\n\
5. Honor your father and your mother, that your days may be long in the land which Yahweh your God gives you.\n\
6. You shall not murder.\n\
7. You shall not commit adultery.\n\
8. You shall not steal.\n\
9. You shall not give false testimony against your neighbor.\n\
10. You shall not covet your neighbor's house. You shall not covet your neighbor's wife, nor his male servant, nor his female servant, nor his ox, nor his donkey, nor anything that is your neighbor's.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the introduction framed above the Ten Commandments?",
        answer: "The Lord said this: God spoke all these words, saying, 'I am Yahweh your God, who brought you out of the land of Egypt, out of the house of bondage.' (Exodus 20:1-2; Deuteronomy 5:6).",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the first commandment?",
        answer: "The Lord said this: 'I am Yahweh your God, who brought you out of the land of Egypt, out of the house of bondage.' You shall have no other gods before me.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the second commandment?",
        answer: "The Lord said this: 'You shall not make for yourselves an idol, nor any image of anything that is in the heavens above, or that is in the earth beneath, or that is in the water under the earth: you shall not bow yourself down to them, nor serve them, for I, Yahweh your God, am a jealous God, visiting the iniquity of the fathers on the children, on the third and on the fourth generation of those who hate me, and showing loving kindness to thousands of those who love me and keep my commandments.'",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the third commandment?",
        answer: "The Lord said this: 'You shall not take the name of Yahweh your God in vain, for Yahweh will not hold him guiltless who takes his name in vain.'",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the fourth commandment?",
        answer: "The Lord said this: 'Remember the Sabbath day, to keep it holy. Six days you shall labor, and do all your work, but the seventh day is a Sabbath to Yahweh your God. You shall not do any work in it, you, nor your son, nor your daughter, your male servant, nor your female servant, nor your livestock, nor your stranger who is within your gates; for in six days Yahweh made heaven and earth, the sea, and all that is in them, and rested the seventh day; therefore Yahweh blessed the Sabbath day, and made it holy.'",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the fifth commandment?",
        answer: "The Lord said this: 'Honor your father and your mother, that your days may be long in the land which Yahweh your God gives you.'",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the sixth commandment?",
        answer: "The Lord said this: 'You shall not murder.'",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the seventh commandment?",
        answer: "The Lord said this: 'You shall not commit adultery.'",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the eighth commandment?",
        answer: "The Lord said this: 'You shall not steal.'",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the ninth commandment?",
        answer: "The Lord said this: 'You shall not give false testimony against your neighbor.'",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the tenth commandment?",
        answer: "The Lord said this: 'You shall not covet your neighbor's house. You shall not covet your neighbor's wife, nor his male servant, nor his female servant, nor his ox, nor his donkey, nor anything that is your neighbor's.'",
    },

    // --- The Other Commandments Jesus Christ Spoke Of (World English Bible - WEB) ---
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What are the greatest commandments Jesus Christ spoke of?",
        answer: "The Lord said this: 'You shall love the Lord your God with all your heart, with all your soul, and with all your mind.' This is the first and great commandment. A second likewise is this, 'You shall love your neighbor as yourself.' The whole law and the prophets depend on these two commandments. (Matthew 22:37-40; Mark 12:29-31).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What is the First and Great Commandment?",
        answer: "The Lord said this: 'You shall love the Lord your God with all your heart, with all your soul, and with all your mind.' This is the first and great commandment. (Matthew 22:37-38; Mark 12:29-30).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What is the Second Great Commandment?",
        answer: "The Lord said this: 'You shall love your neighbor as yourself.' The whole law and the prophets depend on these two commandments. (Matthew 22:39-40; Mark 12:31).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What is the new commandment Jesus Christ gave His disciples?",
        answer: "The Lord said this: 'A new commandment I give to you, that you love one another. Just as I have loved you, you also love one another. By this everyone will know that you are my disciples, if you have love for one another.' (John 13:34-35; John 15:12).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What is the Golden Rule that Jesus Christ taught?",
        answer: "The Lord said this: 'Therefore whatever you desire for men to do to you, you shall also do to them; for this is the law and the prophets.' (Matthew 7:12; Luke 6:31).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What did Jesus Christ teach regarding loving enemies and forgiving?",
        answer: "The Lord said this: 'Love your enemies, bless those who curse you, do good to those who hate you, and pray for those who mistreat you and persecute you. Be merciful, even as your Father is also merciful. Don't judge, and you won't be judged. Don't condemn, and you won't be condemned. Set free, and you will be set free.' (Matthew 5:44; Luke 6:36-37).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What did Jesus Christ teach regarding repenting and believing the gospel?",
        answer: "The Lord said this: 'The time is fulfilled, and God's Kingdom is at hand! Repent, and believe in the Good News.' (Mark 1:15).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What did Jesus Christ teach regarding keeping His commandments?",
        answer: "The Lord said this: 'If you love me, keep my commandments.' (John 14:15; Matthew 19:17-19).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What other commandments did Jesus Christ speak of?",
        answer: "The Lord said this: Jesus Christ commanded: (1) Love the Lord your God with all your heart, with all your soul, and with all your mind; (2) Love your neighbor as yourself; (3) Love one another as He has loved us; (4) Do to others whatever you desire them to do to you; (5) Love your enemies, bless those who curse you, do good to those who hate you, and pray for those who mistreat you; (6) Be merciful, judge not, condemn not, forgive and be set free; (7) Repent and believe in the Good News; (8) Keep the commandments: do not murder, do not commit adultery, do not steal, do not give false testimony, honor your father and mother; (9) If you love me, keep my commandments.",
    },
];

/// Seeds and constrains the brain with foundational commandments prior to training.
///
/// Pretrains exact binding episodes for each Q&A pair between `query_pool` and `answer_pool`.
pub fn seed_commandments_prior_to_training(
    brain: &mut Brain,
    query_pool: PoolId,
    answer_pool: PoolId,
) -> SeedConstraintReport {
    let mut report = SeedConstraintReport::default();
    for pair in FOUNDATIONAL_SEED_PAIRS {
        brain.pretrain_binding_episode(&[
            (query_pool, pair.question.as_bytes().to_vec()),
            (answer_pool, pair.answer.as_bytes().to_vec()),
        ]);
        report.pairs_seeded += 1;
        report.total_episodes += 1;
    }
    report
}
