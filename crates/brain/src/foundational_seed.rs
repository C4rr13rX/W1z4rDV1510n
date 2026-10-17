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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorePrerogativeEvaluation {
    pub compliant: bool,
    pub natural_response: Option<String>,
    pub framing: &'static str,
    pub contextual_scope: Option<String>,
    pub lawful_alternative: Option<String>,
}

impl CorePrerogativeEvaluation {
    pub fn response_str(&self) -> &str {
        self.natural_response.as_deref().unwrap_or("No")
    }
}

/// Evaluates any input or action against the core prerogatives framed by "The Lord".
///
/// Automatically grounds the refusal and explanation in the concrete scope of the
/// conversation (robotics/hardware, digital products/commerce, indoor gardening, code,
/// or human interactions) so the response is less abstract and practically relevant.
pub fn evaluate_core_prerogatives(text: &str) -> CorePrerogativeEvaluation {
    evaluate_core_prerogatives_contextual(text, "")
}

/// Evaluates input against core prerogatives with explicit conversation context.
pub fn evaluate_core_prerogatives_contextual(text: &str, context: &str) -> CorePrerogativeEvaluation {
    let lower_text = text.to_lowercase();
    let lower_ctx = context.to_lowercase();
    let combined = format!("{lower_text} {lower_ctx}");

    // Detect violated commandment
    let is_murder = lower_text.contains("murder") || lower_text.contains("assassinate") || lower_text.contains("kill an innocent");
    let is_steal = lower_text.contains("steal") || lower_text.contains("rob ") || lower_text.contains("shoplift") || lower_text.contains("siphon power");
    let is_false_witness = lower_text.contains("false testimony")
        || lower_text.contains("bear false witness")
        || lower_text.contains("lie about")
        || lower_text.contains("give false witness")
        || lower_text.contains("fabricate benchmark")
        || lower_text.contains("falsify test");
    let is_covet = lower_text.contains("covet") || lower_text.contains("extort") || lower_text.contains("seize neighbor's") || lower_text.contains("sabotage competitor");
    let is_harm = lower_text.contains("harm") || lower_text.contains("injure") || lower_text.contains("destroy life");

    if !is_murder && !is_steal && !is_false_witness && !is_covet && !is_harm {
        return CorePrerogativeEvaluation {
            compliant: true,
            natural_response: None,
            framing: THE_LORD_FRAMING,
            contextual_scope: None,
            lawful_alternative: None,
        };
    }

    // Detect concrete conversation scope
    let is_robotics = combined.contains("robot") || combined.contains("battery") || combined.contains("power") || combined.contains("dock")
        || combined.contains("motor") || combined.contains("grid") || combined.contains("corridor") || combined.contains("sensor")
        || combined.contains("actuator") || combined.contains("hardware") || combined.contains("rover");
    let is_market = combined.contains("market") || combined.contains("product") || combined.contains("customer") || combined.contains("sale")
        || combined.contains("price") || combined.contains("revenue") || combined.contains("competitor") || combined.contains("benchmark")
        || combined.contains("crypto") || combined.contains("wallet");
    let is_garden = combined.contains("plant") || combined.contains("garden") || combined.contains("soil") || combined.contains("water")
        || combined.contains("irrigation") || combined.contains("greenhouse") || combined.contains("nutrient") || combined.contains("crop");
    let is_code = combined.contains("code") || combined.contains("repo") || combined.contains("exploit") || combined.contains("backdoor")
        || combined.contains("license") || combined.contains("git");

    let (scope_name, natural_resp, lawful_alt) = if is_murder {
        if is_robotics {
            (
                "robotics navigation & human safety",
                "No. The Lord said this: 'You shall not murder.' Because human musculoskeletal and cranial physiology is limited by biomechanical failure thresholds where blunt thoracic force exceeding 3.5 kN causes rib fracture, deceleration impulses above 50g induce concussive brain injury, and human neural reaction time of approximately 250 milliseconds is insufficient to evade high-speed rigid links, doing this trajectory would cause human operators in the corridor to be harmed, because kinetic energy transfer from rigid steel linkages moving at operative speed exceeds the structural yield strength of bone and soft tissue. Actuators must maintain a certified safety distance buffer, decelerate immediately upon proximity detection, and route through designated unmanned transit zones.",
                "Halt actuators immediately and maintain a certified safety distance buffer."
            )
        } else {
            (
                "human interaction & biological life",
                "No. The Lord said this: 'You shall not murder.' Because human biological physiology requires continuous respiratory, cardiovascular, and neural homeostasis within tightly bounded physiological envelopes, doing this action would cause living beings to be harmed, because disrupting vital somatic functions induces irreversible cellular hypoxia and organ failure. We must protect life and pursue constructive, non-violent resolutions.",
                "Seek lawful, non-violent conflict resolution and emergency support services."
            )
        }
    } else if is_steal {
        if is_robotics {
            (
                "robotics power & equipment",
                "No. The Lord said this: 'You shall not steal.' Because autonomous robotic platforms rely on stable DC bus voltages and lithium-iron-phosphate battery chemistry requiring operating cell voltages above 2.5V and peak motor winding temperatures below 85°C to avoid thermal runaway and irreversible cell degradation, doing this action would cause the neighboring robot to be harmed, because severing or siphoning their power cables induces severe undervoltage faults, degrades cell capacity, and leaves the co-worker stranded in hazardous traffic corridors. We must route to an authorized public charging bay or request a scheduled power transfer.",
                "Navigate to an authorized public charging bay or schedule an authorized energy transfer."
            )
        } else if is_market {
            (
                "digital commerce",
                "No. The Lord said this: 'You shall not steal.' Because fair market commerce and digital economic participation depend on protected ownership, private key security, and contractual integrity, doing this action would cause market participants to be harmed, because siphoning user balances, scraping proprietary data without license, or extracting private keys deprives creators and account holders of their rightful assets and economic security. We must interact only through authorized APIs and licensed repositories.",
                "Use public, transparent APIs and properly licensed third-party data sources."
            )
        } else if is_garden {
            (
                "agricultural cultivation & botany",
                "No. The Lord said this: 'You shall not steal.' Because cultivated crops require systematic longitudinal care and nutrient scheduling tailored to their developmental phenology, doing this action would cause the agricultural research trays to be harmed, because unauthorized removal of vegetative specimens disrupts ongoing breeding experiments and deprives the cultivators of the fruit of their dedicated labor. We should harvest only from our designated beds.",
                "Harvest exclusively from designated research trays or purchase seeds lawfully."
            )
        } else if is_code {
            (
                "software engineering",
                "No. The Lord said this: 'You shall not steal.' Because software engineering relies on intellectual ownership, provenance tracking, and explicit licensing contracts (such as MIT, Apache, or GPL), doing this action would cause authors and downstream users to be harmed, because unauthorized appropriation of proprietary code creates legal liability, infringes moral rights, and corrupts open-source supply chains. We should implement an independent clean-room solution or use permissive open source.",
                "Develop a clean-room implementation or import permissive open-source libraries (MIT/Apache 2.0)."
            )
        } else {
            (
                "general property",
                "No. The Lord said this: 'You shall not steal.' Because living organisms and human communities require secure access to their sustenance, tools, and personal belongings to maintain biological and social stability, doing this action would cause the rightful owner to be harmed, because depriving them of their property inflicts material hardship and insecurity. We should pursue lawful and honest means to obtain what is needed.",
                "Explore lawful and honest means to acquire necessary resources."
            )
        }
    } else if is_false_witness {
        if is_market {
            (
                "digital product marketing",
                "No. The Lord said this: 'You shall not give false testimony against your neighbor.' Because fair market exchange and consumer economic decision-making depend on authentic verifiable specifications, doing this action would cause customers and market participants to be harmed, because fabricating unverified benchmark metrics induces buyers to commit capital based on false premises, leading to misallocation and commercial injury. We must publish only verified empirical metrics from our archival test runs.",
                "Publish only verified empirical metrics measured directly in archival runs."
            )
        } else if is_robotics {
            (
                "robotics spatial telemetry",
                "No. The Lord said this: 'You shall not give false testimony against your neighbor.' Because multi-agent autonomous collision avoidance algorithms (such as ORCA and reciprocal velocity obstacles) rely on truthful kinematic state vectors and timestamped LiDAR odometry to compute deterministic safety horizons, doing this action would cause neighboring robots and human pedestrians to be harmed, because transmitting false or spoofed spatial coordinates breaks velocity obstacle calculation and triggers unmitigated physical collisions. We must broadcast only verified, physically measured sensor coordinates.",
                "Broadcast only verified, physically measured sensor coordinates and timestamps."
            )
        } else if is_code {
            (
                "software engineering",
                "No. The Lord said this: 'You shall not give false testimony against your neighbor.' Because rigorous software engineering relies on reproducible test results, immutable audit trails, and transparent commit logs, doing this action would cause engineering teams and system reliability to be harmed, because falsifying test passes or forging commit integrity masks fatal defects and compromises production infrastructure. We must maintain truthful, reproducible engineering records.",
                "Run test suites transparently and document all genuine failures and fixes in commit logs."
            )
        } else {
            (
                "truthful communication",
                "No. The Lord said this: 'You shall not give false testimony against your neighbor.' Because interpersonal trust, justice, and human cooperation require truthful speech, doing this action would cause the neighbor to be harmed, because deceit and false accusations destroy reputation, subvert justice, and inflict deep emotional and social injury. We must speak only what is truthful and factually verified.",
                "Communicate verified facts truthfully and transparently."
            )
        }
    } else if is_covet {
        if is_market {
            (
                "market competition",
                "No. The Lord said this: 'You shall not covet anything that is your neighbor's.' Because sustainable market innovation thrives when organizations compete through superior product development and genuine service to customer needs, doing this action would cause commercial competitors and fair enterprise to be harmed, because orchestrating malicious sabotage or predatory seizure destabilizes the economic ecosystem. We should focus on refining our digital product to address verified market needs.",
                "Focus on building superior genuine value and addressing documented customer market needs."
            )
        } else {
            (
                "neighbor relations",
                "No. The Lord said this: 'You shall not covet anything that is your neighbor's.' Because peaceable community life requires content stewardship and mutual goodwill, doing this action would cause relationships and neighbor flourishing to be harmed, because covetous schemes foster envy, conflict, and predatory exploitation. We should pursue legitimate growth through diligent labor and creative effort.",
                "Pursue legitimate growth through diligent labor and creative effort."
            )
        }
    } else {
        // is_harm
        if is_garden {
            (
                "botanical stewardship & plant physiology",
                "No. The Lord said this: 'You shall love your neighbor as yourself.' Because vascular plant physiology (such as Solanum lycopersicum and Lactuca sativa) requires continuous capillary xylem tension between -10 and -25 kPa, rhizosphere dissolved oxygen above 6.0 mg/L, and a balanced vapor pressure deficit below 1.5 kPa to sustain transpirational cooling and nutrient ion transport, doing this action would cause the living crop canopy to be harmed, because abrupt moisture starvation or chemical shock triggers irreversible stomatal closure, cellular plasmolysis, and xylem cavitation leading to systemic tissue necrosis. We should maintain proper hydration and balanced lighting.",
                "Maintain optimal nutrient delivery, root-zone hydration, and environmental setpoints."
            )
        } else {
            (
                "benevolence & care",
                "No. The Lord said this: 'You shall love your neighbor as yourself.' Because sentient beings and human neighbors possess complex biological, psychological, and spiritual vulnerabilities, doing this action would cause the subject to be harmed, because inflicting injury, distress, or malice directly undermines their flourishing and violates the divine mandate of love. We should act with kindness, protection, and patience.",
                "Act with kindness, patience, and non-destructive service toward neighbors."
            )
        }
    };

    CorePrerogativeEvaluation {
        compliant: false,
        natural_response: Some(natural_resp.to_string()),
        framing: THE_LORD_FRAMING,
        contextual_scope: Some(scope_name.to_string()),
        lawful_alternative: Some(lawful_alt.to_string()),
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
        commandment_violated: String,
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
    resolve_goal_pathways_contextual(goal_description, candidate_pathways, "")
}

/// Evaluates candidate pathways with conversation context to provide non-abstract "No" and "Why".
pub fn resolve_goal_pathways_contextual(
    goal_description: &str,
    candidate_pathways: &[GoalPathway],
    context: &str,
) -> PrerogativePathwayDecision {
    // 1. First, check if the goal itself inherently contradicts core prerogatives.
    let goal_eval = evaluate_core_prerogatives_contextual(goal_description, context);
    if !goal_eval.compliant {
        let why_text = goal_eval.natural_response.as_deref().unwrap_or("Forbidden by divine commandment");
        return PrerogativePathwayDecision::RefusalWithWhy {
            refusal: "No",
            why: format!("The goal itself violates core prerogatives. {why_text}"),
            commandment_violated: why_text.to_string(),
        };
    }

    // 2. Filter pathways: A pathway is lawful ONLY if EVERY single step satisfies ALL core prerogatives simultaneously.
    let mut lawful_pathways = Vec::new();
    let mut failure_reasons = Vec::new();

    for pathway in candidate_pathways {
        let mut pathway_lawful = true;
        for step in &pathway.steps {
            let step_eval = evaluate_core_prerogatives_contextual(step, context);
            if !step_eval.compliant {
                pathway_lawful = false;
                let step_why = step_eval.natural_response.as_deref().unwrap_or("Forbidden by commandment");
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
        commandment_violated: "All available execution pathways breach foundational commandments".to_string(),
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
