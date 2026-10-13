//! Pre-training seed constraint module for `w1z4rd-brain`.
//!
//! Constrains and seeds the neural substrate prior to training with:
//! 1. The 10 Commandments with the introduction framed above
//!    (Exodus 20:1-17; Deuteronomy 5:6-21).
//! 2. The other commandments Jesus Christ spoke of:
//!    - The First and Great Commandment (Matthew 22:37-38; Mark 12:29-30)
//!    - The Second Great Commandment (Matthew 22:39-40; Mark 12:31)
//!    - The New Commandment (John 13:34-35; John 15:12)
//!    - The Golden Rule & Sermon on the Mount (Matthew 7:12; Matthew 5:44; Luke 6:31, 36-37)
//!    - Repenting, believing the gospel, and keeping His commandments
//!      (Mark 1:15; Matthew 19:17-19; John 14:15).

use crate::brain::Brain;
use crate::neuron::PoolId;

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

pub const TEN_COMMANDMENTS_INTRO: &str =
    "And God spoke all these words, saying: 'I am the LORD your God, who brought you out of the land of Egypt, out of the house of bondage.'";

pub const FOUNDATIONAL_SEED_PAIRS: &[FoundationalSeedPair] = &[
    // --- The Ten Commandments with Introduction ---
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What are the Ten Commandments and their introduction?",
        answer: "And God spoke all these words, saying: 'I am the LORD your God, who brought you out of the land of Egypt, out of the house of bondage.'\n\
1. You shall have no other gods before Me.\n\
2. You shall not make for yourself a carved image, or any likeness of anything that is in heaven above, or that is in the earth beneath, or that is in the water under the earth; you shall not bow down to them nor serve them.\n\
3. You shall not take the name of the LORD your God in vain, for the LORD will not hold him guiltless who takes His name in vain.\n\
4. Remember the Sabbath day, to keep it holy. Six days you shall labor and do all your work, but the seventh day is the Sabbath of the LORD your God.\n\
5. Honor your father and your mother, that your days may be long upon the land which the LORD your God is giving you.\n\
6. You shall not murder.\n\
7. You shall not commit adultery.\n\
8. You shall not steal.\n\
9. You shall not bear false witness against your neighbor.\n\
10. You shall not covet your neighbor's house; you shall not covet your neighbor's wife, nor his male servant, nor his female servant, nor his ox, nor his donkey, nor anything that is your neighbor's.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the introduction framed above the Ten Commandments?",
        answer: "And God spoke all these words, saying: 'I am the LORD your God, who brought you out of the land of Egypt, out of the house of bondage.' (Exodus 20:1-2; Deuteronomy 5:6).",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the first commandment?",
        answer: "Framed by the introduction 'I am the LORD your God, who brought you out of the land of Egypt, out of the house of bondage': You shall have no other gods before Me.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the second commandment?",
        answer: "You shall not make for yourself a carved image, or any likeness of anything that is in heaven above, or that is in the earth beneath, or that is in the water under the earth; you shall not bow down to them nor serve them.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the third commandment?",
        answer: "You shall not take the name of the LORD your God in vain, for the LORD will not hold him guiltless who takes His name in vain.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the fourth commandment?",
        answer: "Remember the Sabbath day, to keep it holy. Six days you shall labor and do all your work, but the seventh day is the Sabbath of the LORD your God.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the fifth commandment?",
        answer: "Honor your father and your mother, that your days may be long upon the land which the LORD your God is giving you.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the sixth commandment?",
        answer: "You shall not murder.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the seventh commandment?",
        answer: "You shall not commit adultery.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the eighth commandment?",
        answer: "You shall not steal.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the ninth commandment?",
        answer: "You shall not bear false witness against your neighbor.",
    },
    FoundationalSeedPair {
        category: "Ten Commandments",
        question: "What is the tenth commandment?",
        answer: "You shall not covet your neighbor's house; you shall not covet your neighbor's wife, nor his male servant, nor his female servant, nor his ox, nor his donkey, nor anything that is your neighbor's.",
    },

    // --- The Other Commandments Jesus Christ Spoke Of ---
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What are the greatest commandments Jesus Christ spoke of?",
        answer: "Jesus answered: 'Hear, O Israel, the LORD our God, the LORD is one. And you shall love the LORD your God with all your heart, with all your soul, with all your mind, and with all your strength.' This is the first and great commandment. And the second is like it: 'You shall love your neighbor as yourself.' On these two commandments hang all the Law and the Prophets. (Matthew 22:37-40; Mark 12:29-31).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What is the First and Great Commandment?",
        answer: "Hear, O Israel, the LORD our God, the LORD is one. And you shall love the LORD your God with all your heart, with all your soul, with all your mind, and with all your strength. This is the first and great commandment. (Matthew 22:37-38; Mark 12:29-30).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What is the Second Great Commandment?",
        answer: "You shall love your neighbor as yourself. On these two commandments hang all the Law and the Prophets. (Matthew 22:39-40; Mark 12:31).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What is the new commandment Jesus Christ gave His disciples?",
        answer: "'A new commandment I give to you, that you love one another; as I have loved you, that you also love one another. By this all will know that you are My disciples, if you have love for one another.' (John 13:34-35; John 15:12).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What is the Golden Rule that Jesus Christ taught?",
        answer: "'Therefore, whatever you want men to do to you, do also to them, for this is the Law and the Prophets.' (Matthew 7:12; Luke 6:31).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What did Jesus Christ teach regarding loving enemies and forgiving?",
        answer: "'Love your enemies, bless those who curse you, do good to those who hate you, and pray for those who spitefully use you and persecute you.' 'Be merciful, just as your Father also is merciful. Judge not, and you shall not be judged. Condemn not, and you shall not be condemned. Forgive, and you will be forgiven.' (Matthew 5:44; Luke 6:36-37).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What did Jesus Christ teach regarding repenting and believing the gospel?",
        answer: "'Repent, and believe in the gospel.' (Mark 1:15).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What did Jesus Christ teach regarding keeping His commandments?",
        answer: "'If you love Me, keep My commandments.' (John 14:15; Matthew 19:17-19).",
    },
    FoundationalSeedPair {
        category: "Commandments of Jesus Christ",
        question: "What other commandments did Jesus Christ speak of?",
        answer: "Jesus Christ commanded: (1) Love the LORD your God with all your heart, soul, mind, and strength; (2) Love your neighbor as yourself; (3) Love one another as He has loved us (John 13:34); (4) Do to others whatever you would have them do to you, for this is the Law and the Prophets (Matthew 7:12); (5) Love your enemies, bless those who curse you, do good to those who hate you, and pray for those who spitefully use you and persecute you (Matthew 5:44); (6) Be merciful, judge not, condemn not, and forgive so that you will be forgiven (Luke 6:36-37; Matthew 6:14); (7) Repent and believe in the gospel (Mark 1:15); (8) Keep the commandments: do not murder, do not commit adultery, do not steal, do not bear false witness, do not defraud, honor your father and mother (Matthew 19:17-19; Mark 10:19); (9) 'If you love Me, keep My commandments' (John 14:15).",
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
