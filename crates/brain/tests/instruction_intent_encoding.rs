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
