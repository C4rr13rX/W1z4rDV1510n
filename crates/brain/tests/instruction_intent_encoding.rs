use w1z4rd_brain::{AtomEncoding, InstructionIntentEncoding};

#[test]
fn square_paraphrases_share_intent_but_cube_is_distinct() {
    let encoding = InstructionIntentEncoding { prefix: "intent".into() };
    let square_a = encoding.atomize(b"Fix square so it returns a number multiplied by itself.");
    let square_b = encoding.atomize(b"Make square compute the product of its argument with itself.");
    let cube = encoding.atomize(b"Make cube calculate the third power of its input.");
    assert_eq!(square_a, square_b);
    assert_ne!(square_a, cube);
}

#[test]
fn unsupported_intents_do_not_alias_trained_power_features() {
    let encoding = InstructionIntentEncoding { prefix: "intent".into() };
    let factorial = encoding.atomize(b"Fix factorial so it computes factorial.");
    let square = encoding.atomize(b"Fix square so it multiplies by itself.");
    assert_ne!(factorial, square);
}

#[test]
fn bare_k12_concept_does_not_fire_a_coding_intent() {
    let encoding = InstructionIntentEncoding { prefix: "intent".into() };
    assert!(encoding.atomize(b"square").is_empty());
    assert!(!encoding.atomize(b"Write a function that returns a square.").is_empty());
}

#[test]
fn word_count_paraphrases_share_state_intent() {
    let encoding = InstructionIntentEncoding { prefix: "intent".into() };
    let trained = encoding.atomize(b"Implement a function that returns a dict of word -> count.");
    let novel = encoding.atomize(b"Produce a function mapping every word to its occurrence count.");
    assert_eq!(trained, novel);
    assert!(!trained.is_empty());
}

#[test]
fn repair_verb_paraphrases_retain_the_same_intent() {
    let encoding = InstructionIntentEncoding { prefix: "intent".into() };
    let fixed = encoding.atomize(b"Fix is_negative so values below zero are recognized.");
    let corrected = encoding.atomize(b"Correct is_negative to recognize values less than zero.");
    assert_eq!(fixed, corrected);
    assert!(!fixed.is_empty());
}

#[test]
fn language_and_task_are_independent_cofiring_features() {
    let encoding = InstructionIntentEncoding { prefix: "intent".into() };
    let rust = encoding.atomize(b"Create Rust code computing the second power of a number.");
    assert!(rust.iter().any(|label| label == "intent:LANGUAGE:RUST"));
    assert!(rust.iter().any(|label| label == "intent:POWER_SELF:2"));

    let javascript = encoding.atomize(b"Write a JavaScript function that returns a square.");
    assert!(javascript.iter().any(|label| label == "intent:LANGUAGE:JAVASCRIPT"));
    assert!(!javascript.iter().any(|label| label == "intent:LANGUAGE:JAVA"));
}

#[test]
fn average_paraphrases_share_language_and_task_features() {
    let encoding = InstructionIntentEncoding { prefix: "intent".into() };
    let average = encoding.atomize(b"Implement avg_list in Python to return the average.");
    let mean = encoding.atomize(b"Write Python that calculates the arithmetic mean.");
    assert_eq!(average, mean);
    assert!(average.iter().any(|label| label == "intent:MATH:AVERAGE"));
}
