#!/usr/bin/env python3
"""
build_concept_dataset.py — Build developmental vocabulary dataset
=================================================================
Produces data/foundation/concept_dataset.jsonl — one record per concept:

  {
    "concept":   "apple",
    "level":     1,            // 0=infant, 1=toddler, 2=pre-K, 3=K
    "category":  "food/fruit",
    "definition": "An apple is a round fruit...",
    "qa_pairs":  [{"q": "What is an apple?", "a": "..."}],
    "synonyms":  ["fruit", "red apple", "green apple"],
    "open_images_class": "/m/014j1m",   // Open Images class ID (for image fetch)
    "wiki_title": "Apple",              // matched Simple English Wikipedia title
  }

Steps:
  1. Build concept taxonomy from seed categories + WordNet expansion
  2. Rank by developmental frequency (wordfreq)
  3. Match each concept to Simple English Wikipedia article (for definition)
  4. Map concept to Open Images v7 class where available
  5. Generate QA pairs from definition

Usage:
  python scripts/build_concept_dataset.py [--out data/foundation] [--max 10000]

Dependencies:
  pip install nltk wordfreq requests
  python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path

try:
    from nltk.corpus import wordnet as wn
except ImportError:
    sys.exit("Missing: pip install nltk && python -c \"import nltk; nltk.download('wordnet')\"")

try:
    from wordfreq import word_frequency
except ImportError:
    sys.exit("Missing: pip install wordfreq")

# ---------------------------------------------------------------------------
# Developmental taxonomy seed
# Organized by level (0=infant, 1=toddler, 2=pre-K, 3=kindergarten)
# Each entry: (concept, category, level)
# WordNet will expand most noun categories with hyponyms
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Level -1: Pre-linguistic first words (English-speaking environment)
# The 50 most statistically common first words in English from CDI research.
# These are the words a child hears and says BEFORE they understand definitions.
# Training these first grounds the phonological patterns of English.
# ---------------------------------------------------------------------------

FIRST_WORDS = [
    # Social words (heard constantly in infancy)
    ("no", "social_word", -1), ("yes", "social_word", -1),
    ("hi", "social_word", -1), ("bye", "social_word", -1),
    ("please", "social_word", -1), ("thank you", "social_word", -1),
    ("sorry", "social_word", -1), ("okay", "social_word", -1),
    ("hello", "social_word", -1), ("ow", "social_word", -1),
    ("uh oh", "social_word", -1), ("yay", "social_word", -1),
    ("wow", "social_word", -1), ("shh", "social_word", -1),

    # Immediate people and pets
    ("mama", "people", -1), ("dada", "people", -1),
    ("baby", "people", -1), ("doggie", "people", -1), ("kitty", "people", -1),

    # Most-heard action/state words
    ("more", "action", -1), ("all gone", "action", -1),
    ("up", "spatial", -1), ("down", "spatial", -1),
    ("go", "action", -1), ("stop", "action", -1), ("come", "action", -1),
    ("look", "action", -1), ("see", "action", -1), ("get", "action", -1),
    ("give", "action", -1), ("want", "action", -1), ("open", "action", -1),
    ("night night", "action", -1), ("eat", "action", -1),
    ("drink", "action", -1), ("potty", "action", -1),

    # Immediate objects (first nouns)
    ("ball", "toy", -1), ("bottle", "home", -1), ("cup", "home", -1),
    ("book", "learning", -1), ("car (toy)", "toy", -1),
    ("shoe", "clothing", -1), ("hat", "clothing", -1),
    ("milk", "food", -1), ("juice", "food", -1), ("cookie", "food", -1),
    ("apple", "food", -1), ("banana", "food", -1),
    ("water", "food", -1),

    # Properties (first adjectives)
    ("hot", "property", -1), ("cold", "property", -1),
    ("big", "property", -1), ("little", "property", -1),
    ("mine", "property", -1), ("wet", "property", -1), ("dirty", "property", -1),
]

# ---------------------------------------------------------------------------
# Misconception → correction pairs
# Key belief-breaks that should be part of the dataset at level 3+.
# Format: (concept_key, misconception, correction, level)
# ---------------------------------------------------------------------------

MISCONCEPTIONS = {
    # Physics
    "gravity": (
        "Heavier objects fall faster than lighter ones.",
        "All objects fall at the same rate in a vacuum, regardless of weight. "
        "Galileo showed this by dropping cannonballs of different sizes. "
        "Air resistance can slow lighter objects, but gravity pulls equally."
    ),
    "energy": (
        "Energy can be created or destroyed.",
        "Energy cannot be created or destroyed, only converted from one form to another. "
        "This is the law of conservation of energy. When you burn wood, chemical energy "
        "becomes heat and light — none disappears."
    ),
    "heat": (
        "Cold is a thing that enters objects to make them cold.",
        "Cold is not a substance — it is the absence of heat. Heat always flows from "
        "hotter objects to colder ones. When you feel cold, heat is leaving your body."
    ),
    # Biology
    "evolution": (
        "Animals deliberately change to become better suited to their environment.",
        "Evolution happens through random variation and natural selection. Individual "
        "animals do not choose to evolve. Variations happen randomly, and those that "
        "help survival get passed on. No animal is 'trying' to evolve."
    ),
    "brain": (
        "We only use 10% of our brain.",
        "This is a myth. Brain scans show that virtually all parts of the brain are "
        "active at some point. Different areas handle different tasks like vision, "
        "movement, memory, and language — all of it gets used."
    ),
    "plant": (
        "Plants get their food from the soil.",
        "Plants make their own food from sunlight, water, and carbon dioxide through "
        "photosynthesis. The soil provides minerals and water, but the food energy "
        "comes from light, not from the ground."
    ),
    # Earth science
    "sun": (
        "The sun moves across the sky.",
        "The sun does not move across the sky — Earth rotates on its axis once every "
        "24 hours. From Earth's surface it looks like the sun is moving, but it is "
        "we who are spinning."
    ),
    "season": (
        "Seasons happen because Earth moves closer to and farther from the sun.",
        "Seasons are caused by Earth's tilted axis, not its distance from the sun. "
        "In summer, your part of Earth is tilted toward the sun, giving more direct "
        "sunlight. Earth is actually slightly closer to the sun in winter for the "
        "northern hemisphere."
    ),
    # Chemistry / matter
    "water": (
        "Ice, liquid water, and steam are different substances.",
        "Ice, liquid water, and steam are all the same substance — H2O — just in "
        "different states. Adding or removing heat changes the state but not what "
        "the substance is."
    ),
    "oxygen": (
        "We breathe in oxygen and breathe out carbon dioxide entirely.",
        "We breathe in air, which is mostly nitrogen (78%) and oxygen (21%). We use "
        "the oxygen and breathe out air that is still mostly nitrogen, plus more "
        "carbon dioxide and some unused oxygen."
    ),
    # Statistics / probability
    "probability": (
        "If a coin has come up heads 5 times in a row, tails is 'due' next.",
        "Each coin flip is independent. The coin has no memory. The probability of "
        "heads is always 50%, no matter what just happened. This is the gambler's "
        "fallacy — past random events do not affect future ones."
    ),
    # Physics (advanced)
    "light (science)": (
        "Light travels instantly.",
        "Light travels very fast (about 300,000 km per second) but not instantly. "
        "Sunlight takes about 8 minutes to reach Earth. Light from distant stars "
        "takes years or millions of years — we see stars as they were in the past."
    ),
    "atom": (
        "Atoms are the smallest things that exist.",
        "Atoms are made of even smaller particles: protons and neutrons in the "
        "nucleus, and electrons around the outside. Protons and neutrons are made "
        "of even smaller particles called quarks."
    ),
    # Cognitive
    "memory": (
        "Memory works like a video recording that we play back exactly.",
        "Memory is reconstructive, not reproductive. Each time we remember something, "
        "we partly rebuild it. Memories can change, merge with other memories, or be "
        "influenced by things we learned afterward. Eyewitness accounts are often wrong."
    ),
}


SEED_TAXONOMY = [

    # ── Level 0: Infant (0-18 months) ─────────────────────────────────
    # Body parts (first words a child learns by pointing)
    ("eye", "body", 0), ("ear", "body", 0), ("nose", "body", 0),
    ("mouth", "body", 0), ("hand", "body", 0), ("foot", "body", 0),
    ("head", "body", 0), ("hair", "body", 0), ("face", "body", 0),
    ("finger", "body", 0), ("toe", "body", 0), ("arm", "body", 0),
    ("leg", "body", 0), ("belly", "body", 0), ("back", "body", 0),
    ("knee", "body", 0), ("elbow", "body", 0), ("neck", "body", 0),
    ("shoulder", "body", 0), ("chest", "body", 0), ("thumb", "body", 0),
    ("lip", "body", 0), ("tooth", "body", 0), ("tongue", "body", 0),
    ("cheek", "body", 0), ("chin", "body", 0), ("forehead", "body", 0),
    ("eyebrow", "body", 0), ("eyelid", "body", 0), ("palm", "body", 0),
    ("wrist", "body", 0), ("ankle", "body", 0), ("heel", "body", 0),
    ("nail", "body", 0), ("skin", "body", 0), ("bone", "body", 0),
    ("heart", "body", 0), ("brain", "body", 0), ("blood", "body", 0),
    ("stomach", "body", 0), ("lung", "body", 0),

    # Immediate family
    ("mother", "people", 0), ("father", "people", 0), ("baby", "people", 0),
    ("child", "people", 0), ("person", "people", 0), ("boy", "people", 0),
    ("girl", "people", 0), ("man", "people", 0), ("woman", "people", 0),

    # Core objects an infant sees daily
    ("bed", "home", 0), ("chair", "home", 0), ("table", "home", 0),
    ("door", "home", 0), ("window", "home", 0), ("floor", "home", 0),
    ("cup", "home", 0), ("spoon", "home", 0), ("ball", "toy", 0),
    ("book", "learning", 0), ("bottle", "home", 0), ("blanket", "home", 0),
    ("light", "home", 0), ("shoe", "clothing", 0), ("sock", "clothing", 0),

    # Basic food (infant)
    ("milk", "food", 0), ("water", "food", 0), ("apple", "food", 0),
    ("banana", "food", 0), ("bread", "food", 0), ("egg", "food", 0),
    ("rice", "food", 0),

    # First animals
    ("dog", "animal", 0), ("cat", "animal", 0), ("bird", "animal", 0),
    ("fish", "animal", 0), ("duck", "animal", 0),

    # Basic nature
    ("sun", "nature", 0), ("moon", "nature", 0), ("star", "nature", 0),
    ("sky", "nature", 0), ("rain", "nature", 0), ("tree", "nature", 0),
    ("flower", "nature", 0), ("grass", "nature", 0),

    # First actions (Level 0 verbs — things babies do)
    ("eat", "action", 0), ("sleep", "action", 0), ("cry", "action", 0),
    ("smile", "action", 0), ("sit", "action", 0), ("stand", "action", 0),
    ("walk", "action", 0), ("run", "action", 0), ("play", "action", 0),
    ("fall", "action", 0), ("drink", "action", 0), ("hold", "action", 0),

    # ── Level 1: Toddler (18 months – 3 years) ───────────────────────
    # Expanded animals
    ("horse", "animal", 1), ("cow", "animal", 1), ("pig", "animal", 1),
    ("sheep", "animal", 1), ("chicken", "animal", 1), ("rabbit", "animal", 1),
    ("mouse", "animal", 1), ("frog", "animal", 1), ("snake", "animal", 1),
    ("bear", "animal", 1), ("lion", "animal", 1), ("tiger", "animal", 1),
    ("elephant", "animal", 1), ("monkey", "animal", 1), ("deer", "animal", 1),
    ("fox", "animal", 1), ("wolf", "animal", 1), ("owl", "animal", 1),
    ("eagle", "animal", 1), ("butterfly", "animal", 1), ("bee", "animal", 1),
    ("ant", "animal", 1), ("worm", "animal", 1), ("spider", "animal", 1),
    ("turtle", "animal", 1), ("penguin", "animal", 1), ("giraffe", "animal", 1),
    ("zebra", "animal", 1), ("dolphin", "animal", 1), ("whale", "animal", 1),
    ("shark", "animal", 1), ("octopus", "animal", 1), ("crab", "animal", 1),
    ("lobster", "animal", 1), ("snail", "animal", 1), ("caterpillar", "animal", 1),
    ("dragonfly", "animal", 1), ("ladybug", "animal", 1), ("grasshopper", "animal", 1),
    ("parrot", "animal", 1), ("flamingo", "animal", 1), ("crocodile", "animal", 1),
    ("hippopotamus", "animal", 1), ("rhinoceros", "animal", 1), ("gorilla", "animal", 1),
    ("kangaroo", "animal", 1), ("koala", "animal", 1), ("panda", "animal", 1),
    ("camel", "animal", 1), ("llama", "animal", 1), ("seal", "animal", 1),

    # Expanded food
    ("orange", "food", 1), ("grape", "food", 1), ("strawberry", "food", 1),
    ("watermelon", "food", 1), ("peach", "food", 1), ("pear", "food", 1),
    ("cherry", "food", 1), ("lemon", "food", 1), ("mango", "food", 1),
    ("pineapple", "food", 1), ("avocado", "food", 1), ("coconut", "food", 1),
    ("carrot", "food", 1), ("potato", "food", 1), ("tomato", "food", 1),
    ("onion", "food", 1), ("garlic", "food", 1), ("corn", "food", 1),
    ("broccoli", "food", 1), ("spinach", "food", 1), ("lettuce", "food", 1),
    ("cucumber", "food", 1), ("pepper", "food", 1), ("mushroom", "food", 1),
    ("pumpkin", "food", 1), ("celery", "food", 1), ("pea", "food", 1),
    ("bean", "food", 1), ("cheese", "food", 1), ("butter", "food", 1),
    ("juice", "food", 1), ("cake", "food", 1), ("cookie", "food", 1),
    ("candy", "food", 1), ("chocolate", "food", 1), ("pizza", "food", 1),
    ("sandwich", "food", 1), ("soup", "food", 1), ("pasta", "food", 1),
    ("cereal", "food", 1), ("yogurt", "food", 1), ("ice cream", "food", 1),
    ("sugar", "food", 1), ("salt", "food", 1), ("meat", "food", 1),
    ("chicken (meat)", "food", 1), ("fish (food)", "food", 1), ("honey", "food", 1),
    ("peanut butter", "food", 1), ("jam", "food", 1),

    # Vehicles
    ("car", "vehicle", 1), ("truck", "vehicle", 1), ("bus", "vehicle", 1),
    ("train", "vehicle", 1), ("airplane", "vehicle", 1), ("boat", "vehicle", 1),
    ("bicycle", "vehicle", 1), ("motorcycle", "vehicle", 1),
    ("helicopter", "vehicle", 1), ("submarine", "vehicle", 1),
    ("ambulance", "vehicle", 1), ("fire truck", "vehicle", 1),
    ("police car", "vehicle", 1), ("taxi", "vehicle", 1), ("ship", "vehicle", 1),
    ("rocket", "vehicle", 1), ("tractor", "vehicle", 1),
    ("skateboard", "vehicle", 1), ("scooter", "vehicle", 1),

    # Clothing
    ("shirt", "clothing", 1), ("pants", "clothing", 1), ("dress", "clothing", 1),
    ("hat", "clothing", 1), ("coat", "clothing", 1), ("glove", "clothing", 1),
    ("glasses", "clothing", 1), ("belt", "clothing", 1), ("jacket", "clothing", 1),
    ("boot", "clothing", 1), ("sandal", "clothing", 1), ("sneaker", "clothing", 1),
    ("scarf", "clothing", 1), ("sweater", "clothing", 1), ("skirt", "clothing", 1),
    ("pajama", "clothing", 1), ("diaper", "clothing", 1), ("swimsuit", "clothing", 1),
    ("uniform", "clothing", 1), ("tie", "clothing", 1), ("apron", "clothing", 1),
    ("backpack", "clothing", 1), ("purse", "clothing", 1), ("umbrella", "clothing", 1),

    # Home objects
    ("house", "home", 1), ("room", "home", 1), ("wall", "home", 1),
    ("ceiling", "home", 1), ("roof", "home", 1), ("sofa", "home", 1),
    ("lamp", "home", 1), ("mirror", "home", 1), ("clock", "home", 1),
    ("phone", "home", 1), ("television", "home", 1), ("computer", "home", 1),
    ("refrigerator", "home", 1), ("stove", "home", 1), ("sink", "home", 1),
    ("toilet", "home", 1), ("bathtub", "home", 1), ("shower", "home", 1),
    ("pillow", "home", 1), ("towel", "home", 1), ("soap", "home", 1),
    ("toothbrush", "home", 1), ("comb", "home", 1), ("key", "home", 1),
    ("lock", "home", 1), ("stairs", "home", 1), ("garage", "home", 1),
    ("fence", "home", 1), ("mailbox", "home", 1), ("trash can", "home", 1),
    ("plate", "home", 1), ("bowl", "home", 1), ("fork", "home", 1),
    ("knife", "home", 1), ("pot", "home", 1), ("pan", "home", 1),
    ("oven", "home", 1), ("microwave", "home", 1), ("blender", "home", 1),
    ("broom", "home", 1), ("mop", "home", 1), ("vacuum", "home", 1),
    ("scissors", "home", 1), ("needle", "home", 1), ("thread", "home", 1),
    ("hammer", "home", 1), ("nail (fastener)", "home", 1),

    # Nature (expanded)
    ("cloud", "nature", 1), ("wind", "nature", 1), ("snow", "nature", 1),
    ("ice", "nature", 1), ("fire", "nature", 1), ("earth", "nature", 1),
    ("rock", "nature", 1), ("sand", "nature", 1), ("ocean", "nature", 1),
    ("river", "nature", 1), ("mountain", "nature", 1), ("lake", "nature", 1),
    ("forest", "nature", 1), ("desert", "nature", 1), ("island", "nature", 1),
    ("beach", "nature", 1), ("volcano", "nature", 1), ("cave", "nature", 1),
    ("leaf", "nature", 1), ("seed", "nature", 1), ("root", "nature", 1),
    ("branch", "nature", 1), ("trunk", "nature", 1), ("bush", "nature", 1),
    ("mud", "nature", 1), ("puddle", "nature", 1), ("shadow", "nature", 1),
    ("rainbow", "nature", 1), ("lightning", "nature", 1), ("thunder", "nature", 1),
    ("storm", "nature", 1), ("wave", "nature", 1), ("soil", "nature", 1),
    ("air", "nature", 1), ("fog", "nature", 1), ("frost", "nature", 1),

    # Colors
    ("red", "color", 1), ("blue", "color", 1), ("green", "color", 1),
    ("yellow", "color", 1), ("orange (color)", "color", 1), ("purple", "color", 1),
    ("pink", "color", 1), ("brown", "color", 1), ("black", "color", 1),
    ("white", "color", 1), ("gray", "color", 1), ("gold", "color", 1),
    ("silver", "color", 1), ("turquoise", "color", 1), ("beige", "color", 1),

    # Shapes
    ("circle", "shape", 1), ("square", "shape", 1), ("triangle", "shape", 1),
    ("rectangle", "shape", 1), ("oval", "shape", 1), ("star (shape)", "shape", 1),
    ("heart (shape)", "shape", 1), ("diamond (shape)", "shape", 1),
    ("cube", "shape", 1), ("sphere", "shape", 1), ("cylinder", "shape", 1),
    ("cone", "shape", 1), ("pyramid", "shape", 1),

    # Numbers and quantity
    ("one", "number", 1), ("two", "number", 1), ("three", "number", 1),
    ("four", "number", 1), ("five", "number", 1), ("six", "number", 1),
    ("seven", "number", 1), ("eight", "number", 1), ("nine", "number", 1),
    ("ten", "number", 1), ("zero", "number", 1), ("hundred", "number", 1),
    ("thousand", "number", 1), ("half", "number", 1), ("many", "number", 1),
    ("few", "number", 1), ("empty", "number", 1), ("full", "number", 1),

    # Size / property adjectives
    ("big", "property", 1), ("small", "property", 1), ("tall", "property", 1),
    ("short", "property", 1), ("long", "property", 1), ("wide", "property", 1),
    ("thin", "property", 1), ("thick", "property", 1), ("heavy", "property", 1),
    ("light (weight)", "property", 1), ("hot", "property", 1), ("cold", "property", 1),
    ("warm", "property", 1), ("cool", "property", 1), ("fast", "property", 1),
    ("slow", "property", 1), ("loud", "property", 1), ("quiet", "property", 1),
    ("hard", "property", 1), ("soft", "property", 1), ("rough", "property", 1),
    ("smooth", "property", 1), ("sharp", "property", 1), ("flat", "property", 1),
    ("round", "property", 1), ("straight", "property", 1), ("curved", "property", 1),
    ("clean", "property", 1), ("dirty", "property", 1), ("wet", "property", 1),
    ("dry", "property", 1), ("new", "property", 1), ("old", "property", 1),
    ("broken", "property", 1), ("open", "property", 1), ("closed", "property", 1),

    # Emotions (toddler learns emotional vocabulary)
    ("happy", "emotion", 1), ("sad", "emotion", 1), ("angry", "emotion", 1),
    ("scared", "emotion", 1), ("surprised", "emotion", 1), ("excited", "emotion", 1),
    ("tired", "emotion", 1), ("hungry", "emotion", 1), ("thirsty", "emotion", 1),
    ("sick", "emotion", 1), ("hurt", "emotion", 1), ("bored", "emotion", 1),
    ("love", "emotion", 1), ("hate", "emotion", 1), ("afraid", "emotion", 1),
    ("shy", "emotion", 1), ("proud", "emotion", 1), ("confused", "emotion", 1),

    # ── Level 2: Pre-Kindergarten (3–4 years) ────────────────────────
    # People and roles
    ("teacher", "people", 2), ("doctor", "people", 2), ("nurse", "people", 2),
    ("police officer", "people", 2), ("firefighter", "people", 2),
    ("farmer", "people", 2), ("cook", "people", 2), ("driver", "people", 2),
    ("pilot", "people", 2), ("scientist", "people", 2), ("artist", "people", 2),
    ("singer", "people", 2), ("athlete", "people", 2), ("soldier", "people", 2),
    ("president", "people", 2), ("king", "people", 2), ("queen", "people", 2),
    ("friend", "people", 2), ("neighbor", "people", 2), ("stranger", "people", 2),
    ("sister", "people", 2), ("brother", "people", 2), ("grandmother", "people", 2),
    ("grandfather", "people", 2), ("uncle", "people", 2), ("aunt", "people", 2),
    ("cousin", "people", 2), ("twin", "people", 2),

    # Places
    ("school", "place", 2), ("park", "place", 2), ("store", "place", 2),
    ("hospital", "place", 2), ("library", "place", 2), ("church", "place", 2),
    ("beach (place)", "place", 2), ("farm", "place", 2), ("zoo", "place", 2),
    ("museum", "place", 2), ("restaurant", "place", 2), ("bank", "place", 2),
    ("post office", "place", 2), ("airport", "place", 2), ("train station", "place", 2),
    ("stadium", "place", 2), ("theater", "place", 2), ("market", "place", 2),
    ("factory", "place", 2), ("city", "place", 2), ("town", "place", 2),
    ("village", "place", 2), ("country", "place", 2),

    # Time concepts
    ("day", "time", 2), ("night", "time", 2), ("morning", "time", 2),
    ("afternoon", "time", 2), ("evening", "time", 2), ("today", "time", 2),
    ("tomorrow", "time", 2), ("yesterday", "time", 2), ("week", "time", 2),
    ("month", "time", 2), ("year", "time", 2), ("hour", "time", 2),
    ("minute", "time", 2), ("second (time)", "time", 2), ("season", "time", 2),
    ("spring", "time", 2), ("summer", "time", 2), ("autumn", "time", 2),
    ("winter", "time", 2), ("Monday", "time", 2), ("Tuesday", "time", 2),
    ("Wednesday", "time", 2), ("Thursday", "time", 2), ("Friday", "time", 2),
    ("Saturday", "time", 2), ("Sunday", "time", 2), ("January", "time", 2),
    ("February", "time", 2), ("March", "time", 2), ("April", "time", 2),
    ("May", "time", 2), ("June", "time", 2), ("July", "time", 2),
    ("August", "time", 2), ("September", "time", 2), ("October", "time", 2),
    ("November", "time", 2), ("December", "time", 2),

    # Spatial/relational concepts
    ("inside", "spatial", 2), ("outside", "spatial", 2), ("above", "spatial", 2),
    ("below", "spatial", 2), ("next to", "spatial", 2), ("between", "spatial", 2),
    ("front", "spatial", 2), ("back (position)", "spatial", 2),
    ("left", "spatial", 2), ("right", "spatial", 2), ("up", "spatial", 2),
    ("down", "spatial", 2), ("near", "spatial", 2), ("far", "spatial", 2),
    ("beginning", "spatial", 2), ("end", "spatial", 2), ("middle", "spatial", 2),
    ("top", "spatial", 2), ("bottom", "spatial", 2), ("corner", "spatial", 2),
    ("edge", "spatial", 2), ("center", "spatial", 2), ("side", "spatial", 2),

    # Actions (pre-K verbs)
    ("jump", "action", 2), ("climb", "action", 2), ("swim", "action", 2),
    ("fly", "action", 2), ("throw", "action", 2), ("catch", "action", 2),
    ("push", "action", 2), ("pull", "action", 2), ("kick", "action", 2),
    ("dance", "action", 2), ("sing", "action", 2), ("draw", "action", 2),
    ("paint", "action", 2), ("write", "action", 2), ("read", "action", 2),
    ("count", "action", 2), ("build", "action", 2), ("break", "action", 2),
    ("help", "action", 2), ("share", "action", 2), ("clean", "action", 2),
    ("cook", "action", 2), ("wash", "action", 2), ("plant", "action", 2),
    ("grow", "action", 2), ("cut", "action", 2), ("open", "action", 2),
    ("close", "action", 2), ("give", "action", 2), ("take", "action", 2),
    ("find", "action", 2), ("lose", "action", 2), ("make", "action", 2),
    ("fix", "action", 2), ("hide", "action", 2), ("look", "action", 2),
    ("listen", "action", 2), ("smell", "action", 2), ("touch", "action", 2),
    ("taste", "action", 2), ("breathe", "action", 2), ("think", "action", 2),
    ("know", "action", 2), ("remember", "action", 2), ("forget", "action", 2),
    ("teach", "action", 2), ("learn", "action", 2), ("work", "action", 2),
    ("rest", "action", 2), ("wait", "action", 2), ("start", "action", 2),
    ("stop", "action", 2), ("turn", "action", 2), ("carry", "action", 2),
    ("lift", "action", 2), ("drop", "action", 2), ("pour", "action", 2),
    ("fill", "action", 2), ("mix", "action", 2), ("buy", "action", 2),
    ("sell", "action", 2), ("pay", "action", 2), ("save", "action", 2),
    ("send", "action", 2), ("receive", "action", 2), ("call", "action", 2),
    ("answer", "action", 2), ("ask", "action", 2), ("tell", "action", 2),
    ("show", "action", 2), ("point", "action", 2), ("choose", "action", 2),
    ("want", "action", 2), ("need", "action", 2), ("like", "action", 2),
    ("love (action)", "action", 2), ("hate (action)", "action", 2),
    ("agree", "action", 2), ("disagree", "action", 2), ("change", "action", 2),

    # Basic science (pre-K)
    ("plant", "science", 2), ("animal", "science", 2), ("insect", "science", 2),
    ("reptile", "science", 2), ("mammal", "science", 2), ("bird (animal)", "science", 2),
    ("water (science)", "science", 2), ("ice (science)", "science", 2),
    ("steam", "science", 2), ("solid", "science", 2), ("liquid", "science", 2),
    ("gas", "science", 2), ("heat", "science", 2), ("light (science)", "science", 2),
    ("sound", "science", 2), ("magnet", "science", 2), ("electricity", "science", 2),
    ("gravity", "science", 2), ("weight", "science", 2), ("size", "science", 2),
    ("temperature", "science", 2), ("color (science)", "science", 2),
    ("seed (science)", "science", 2), ("root (science)", "science", 2),
    ("photosynthesis", "science", 2),

    # Social concepts
    ("family", "social", 2), ("home (concept)", "social", 2),
    ("community", "social", 2), ("rule", "social", 2), ("law", "social", 2),
    ("right (concept)", "social", 2), ("responsibility", "social", 2),
    ("kindness", "social", 2), ("respect", "social", 2), ("honesty", "social", 2),
    ("fairness", "social", 2), ("teamwork", "social", 2), ("cooperation", "social", 2),
    ("safety", "social", 2), ("danger", "social", 2), ("problem", "social", 2),
    ("solution", "social", 2), ("idea", "social", 2), ("question", "social", 2),
    ("answer (concept)", "social", 2),

    # ── Level 3: Kindergarten (4–5 years) ────────────────────────────
    # Science concepts
    ("energy", "science", 3), ("matter", "science", 3), ("force", "science", 3),
    ("motion", "science", 3), ("friction", "science", 3), ("pressure", "science", 3),
    ("oxygen", "science", 3), ("carbon dioxide", "science", 3),
    ("photosynthesis", "science", 3), ("ecosystem", "science", 3),
    ("habitat", "science", 3), ("food chain", "science", 3),
    ("predator", "science", 3), ("prey", "science", 3),
    ("adaptation", "science", 3), ("evolution", "science", 3),
    ("cell", "science", 3), ("organism", "science", 3),
    ("bacteria", "science", 3), ("virus", "science", 3),
    ("planet", "science", 3), ("solar system", "science", 3),
    ("galaxy", "science", 3), ("universe", "science", 3),
    ("atmosphere", "science", 3), ("weather", "science", 3),
    ("climate", "science", 3), ("erosion", "science", 3),
    ("earthquake", "science", 3), ("hurricane", "science", 3),

    # Math concepts
    ("addition", "math", 3), ("subtraction", "math", 3),
    ("multiplication", "math", 3), ("division", "math", 3),
    ("number", "math", 3), ("digit", "math", 3), ("fraction", "math", 3),
    ("measurement", "math", 3), ("length", "math", 3), ("width", "math", 3),
    ("height", "math", 3), ("area", "math", 3), ("perimeter", "math", 3),
    ("pattern", "math", 3), ("graph", "math", 3), ("chart", "math", 3),
    ("equation", "math", 3), ("symbol", "math", 3),

    # Language arts
    ("alphabet", "language", 3), ("letter", "language", 3),
    ("word", "language", 3), ("sentence", "language", 3),
    ("paragraph", "language", 3), ("story", "language", 3),
    ("poem", "language", 3), ("rhyme", "language", 3),
    ("syllable", "language", 3), ("vowel", "language", 3),
    ("consonant", "language", 3), ("noun", "language", 3),
    ("verb", "language", 3), ("adjective", "language", 3),
    ("punctuation", "language", 3), ("question mark", "language", 3),
    ("exclamation", "language", 3), ("comma", "language", 3),
    ("language", "language", 3), ("communicate", "language", 3),
    ("read (action)", "language", 3), ("write (action)", "language", 3),

    # Social studies
    ("history", "social", 3), ("culture", "social", 3),
    ("tradition", "social", 3), ("government", "social", 3),
    ("democracy", "social", 3), ("election", "social", 3),
    ("map", "social", 3), ("globe", "social", 3), ("continent", "social", 3),
    ("ocean (geography)", "social", 3), ("country (place)", "social", 3),
    ("capital", "social", 3), ("flag", "social", 3), ("citizen", "social", 3),
    ("economy", "social", 3), ("trade", "social", 3), ("money", "social", 3),
    ("job", "social", 3), ("career", "social", 3), ("invention", "social", 3),
    ("technology", "social", 3), ("tool", "social", 3),

    # Health and body (K-level)
    ("muscle", "health", 3), ("skeleton", "health", 3), ("nerve", "health", 3),
    ("immune system", "health", 3), ("digestion", "health", 3),
    ("nutrition", "health", 3), ("vitamin", "health", 3),
    ("exercise", "health", 3), ("hygiene", "health", 3),
    ("medicine", "health", 3), ("disease", "health", 3),
    ("infection", "health", 3), ("allergy", "health", 3),

    # Abstract concepts (K-level)
    ("time", "abstract", 3), ("space", "abstract", 3),
    ("cause", "abstract", 3), ("effect", "abstract", 3),
    ("compare", "abstract", 3), ("contrast", "abstract", 3),
    ("classify", "abstract", 3), ("predict", "abstract", 3),
    ("observe", "abstract", 3), ("experiment", "abstract", 3),
    ("measure", "abstract", 3), ("record", "abstract", 3),
    ("conclusion", "abstract", 3), ("evidence", "abstract", 3),
    ("theory", "abstract", 3), ("opinion", "abstract", 3),
    ("fact", "abstract", 3), ("fiction", "abstract", 3),
    ("imagination", "abstract", 3), ("creativity", "abstract", 3),
    ("memory", "abstract", 3), ("dream", "abstract", 3),
    ("belief", "abstract", 3), ("value", "abstract", 3),
    ("culture", "abstract", 3), ("symbol", "abstract", 3),
    ("meaning", "abstract", 3), ("purpose", "abstract", 3),
    ("goal", "abstract", 3), ("plan", "abstract", 3),
    ("decision", "abstract", 3), ("choice", "abstract", 3),
    ("consequence", "abstract", 3), ("change", "abstract", 3),
    ("growth", "abstract", 3), ("cycle", "abstract", 3),
    ("process", "abstract", 3), ("system", "abstract", 3),
    ("structure", "abstract", 3), ("function", "abstract", 3),
]

# ---------------------------------------------------------------------------
# Open Images v7 label map — concept → OI class ID
# (600 categories, CC-BY 4.0)
# ---------------------------------------------------------------------------

OI_LABEL_MAP = {
    "dog": "/m/0bt9lr", "cat": "/m/01yrx", "bird": "/m/015p6",
    "fish": "/m/0fish", "horse": "/m/03k3r", "cow": "/m/01xq0k1",
    "pig": "/m/068zj", "sheep": "/m/07bgp", "chicken": "/m/09b5t",
    "duck": "/m/09ddx", "rabbit": "/m/06mf6", "mouse": "/m/04rmv",
    "frog": "/m/09ld4", "snake": "/m/078jl", "bear": "/m/01dws",
    "lion": "/m/096mb", "tiger": "/m/07xpf", "elephant": "/m/0bwd_0j",
    "monkey": "/m/04q1j", "deer": "/m/09kx5", "fox": "/m/0cdn1",
    "owl": "/m/04n6w", "eagle": "/m/015x4r", "butterfly": "/m/0csby",
    "bee": "/m/09dzwx", "ant": "/m/0k4j", "spider": "/m/07_wnh",
    "turtle": "/m/09j5n", "penguin": "/m/05z87", "giraffe": "/m/035r7c",
    "zebra": "/m/0898b", "dolphin": "/m/02hj4", "whale": "/m/084zz",
    "shark": "/m/0by6g", "octopus": "/m/06c54", "crab": "/m/01lcw4",
    "apple": "/m/014j1m", "banana": "/m/09qck", "orange": "/m/0grw1",
    "grape": "/m/0fjh7", "strawberry": "/m/07fbm7", "watermelon": "/m/07j7r",
    "carrot": "/m/0dj6p", "potato": "/m/05vtc", "tomato": "/m/07j87",
    "bread": "/m/098f4v", "egg": "/m/0dqv4", "milk": "/m/01c8br",
    "car": "/m/0k4j", "truck": "/m/07r04", "bus": "/m/01bjv",
    "train": "/m/07jdr", "airplane": "/m/02yvhj", "boat": "/m/019jd",
    "bicycle": "/m/0199g", "motorcycle": "/m/04_sv",
    "chair": "/m/01mzpv", "table": "/m/04bcr3", "bed": "/m/03fp41",
    "sofa": "/m/04kkgm", "lamp": "/m/0ph39", "clock": "/m/01x3z",
    "book": "/m/0bt_c3", "cup": "/m/02jz0l", "bottle": "/m/04dr76w",
    "plate": "/m/02p5f1q", "fork": "/m/0cmx8", "knife": "/m/04ctx",
    "spoon": "/m/0cmf2", "bowl": "/m/04yx4",
    "house": "/m/03jm5", "door": "/m/02dgv", "window": "/m/0d4v4",
    "tree": "/m/07j7r", "flower": "/m/0c9ph5", "grass": "/m/05s2s",
    "sun": "/m/06q74", "moon": "/m/052_cx", "star": "/m/06ngk",
    "cloud": "/m/01ctsf", "fire": "/m/0302g", "water": "/m/0838f",
    "mountain": "/m/09d_r", "beach": "/m/0c3gq",
    "shirt": "/m/01n4qj", "pants": "/m/0fly7", "dress": "/m/01d40f",
    "hat": "/m/02tfl0", "shoe": "/m/01rkbr", "glasses": "/m/000jt",
    "backpack": "/m/0159h6", "umbrella": "/m/0hnnb",
    "person": "/m/01g317", "man": "/m/04yx4", "woman": "/m/03bt1vf",
    "boy": "/m/01bl7v", "girl": "/m/05r655", "baby": "/m/0323sq",
    "face": "/m/0dzf4", "hand": "/m/0k65p", "eye": "/m/014sv8",
    "nose": "/m/0k0pj", "ear": "/m/039xj_", "mouth": "/m/0283dt1",
    "hair": "/m/03q69", "tooth": "/m/012n7d",
    "pizza": "/m/0663v", "sandwich": "/m/06nwz", "cake": "/m/0fszt",
    "cookie": "/m/0jy4k", "ice cream": "/m/06_72j", "chocolate": "/m/01wydv",
    "coffee": "/m/02p0tk3", "wine": "/m/081qc",
    "school": "/m/06_y0by", "hospital": "/m/03gq5hm", "library": "/m/0d4w1",
    "park": "/m/019h78", "zoo": "/m/0f3kl", "museum": "/m/05gqfk",
    "pencil": "/m/0jyfg", "pen": "/m/07s6nbt", "scissors": "/m/01lsmm",
    "ball": "/m/018xm", "toy": "/m/0138tl",
    "map": "/m/0cmf2", "flag": "/m/02wv6h6", "money": "/m/0hgxn",
    "hammer": "/m/03wvsk", "key": "/m/0d2v0", "lock": "/m/02d9qx",
}

# ---------------------------------------------------------------------------

def get_wordnet_definition(word: str) -> str:
    """Get a clean plain-English definition from WordNet."""
    clean = re.sub(r"\s*\(.*?\)", "", word).strip().lower()
    synsets = wn.synsets(clean, pos=wn.NOUN) or wn.synsets(clean, pos=wn.VERB) \
              or wn.synsets(clean, pos=wn.ADJ) or wn.synsets(clean)
    if not synsets:
        return ""
    defn = synsets[0].definition()
    # Capitalize first letter
    if defn:
        defn = defn[0].upper() + defn[1:]
    return defn


def make_definition_sentence(concept: str, category: str, wn_defn: str) -> str:
    """Produce a child-friendly definition sentence."""
    clean = re.sub(r"\s*\(.*?\)", "", concept).strip()
    if wn_defn:
        return f"{clean.capitalize()} is {wn_defn.lower().rstrip('.')}."
    # Fallback by category
    fallbacks = {
        "body":     f"{clean.capitalize()} is a part of the body.",
        "animal":   f"A {clean} is an animal.",
        "food":     f"{clean.capitalize()} is a type of food.",
        "vehicle":  f"A {clean} is a vehicle used for transportation.",
        "clothing": f"{clean.capitalize()} is a type of clothing people wear.",
        "home":     f"A {clean} is something found in a home.",
        "nature":   f"{clean.capitalize()} is found in nature.",
        "color":    f"{clean.capitalize()} is a color.",
        "shape":    f"A {clean} is a shape.",
        "number":   f"{clean.capitalize()} is a number.",
        "action":   f"To {clean} is something people do.",
        "emotion":  f"{clean.capitalize()} is a feeling.",
        "people":   f"A {clean} is a type of person.",
        "place":    f"A {clean} is a type of place.",
        "time":     f"{clean.capitalize()} is related to time.",
        "spatial":  f"{clean.capitalize()} describes where something is.",
        "science":  f"{clean.capitalize()} is a science concept.",
        "math":     f"{clean.capitalize()} is a math concept.",
        "language": f"{clean.capitalize()} is a language concept.",
        "social":   f"{clean.capitalize()} is a social concept.",
        "health":   f"{clean.capitalize()} is related to health.",
        "abstract": f"{clean.capitalize()} is an idea or concept.",
        "property": f"{clean.capitalize()} describes how something looks or feels.",
    }
    return fallbacks.get(category, f"{clean.capitalize()} is a concept.")


def make_qa_pairs(concept: str, definition: str, wn_defn: str) -> list[dict]:
    """Generate multiple QA pairs for a concept."""
    clean = re.sub(r"\s*\(.*?\)", "", concept).strip()
    pairs = []

    # Primary: what is X?
    if definition:
        pairs.append({
            "qa_id":         str(uuid.uuid4()),
            "question":      f"What is {clean.lower()}?",
            "answer":        definition,
            "confidence":    0.90,
            "review_status": "foundation_taxonomy",
        })

    # WordNet extra definition as second QA pair
    if wn_defn and wn_defn.lower() != definition.lower()[:len(wn_defn)].lower():
        pairs.append({
            "qa_id":         str(uuid.uuid4()),
            "question":      f"Can you describe {clean.lower()}?",
            "answer":        wn_defn[0].upper() + wn_defn[1:].rstrip('.') + ".",
            "confidence":    0.80,
            "review_status": "foundation_taxonomy",
        })

    # WordNet examples as a third QA pair
    synsets = wn.synsets(clean.lower())
    if synsets and synsets[0].examples():
        ex = synsets[0].examples()[0]
        pairs.append({
            "qa_id":         str(uuid.uuid4()),
            "question":      f"Give an example of {clean.lower()}.",
            "answer":        ex[0].upper() + ex[1:] if ex else "",
            "confidence":    0.70,
            "review_status": "foundation_taxonomy",
        })

    return [p for p in pairs if p.get("answer", "").strip()]


def build_dataset(max_concepts: int, out_path: Path) -> None:
    seen = set()
    records = []

    # ── Level -1: pre-linguistic first words ─────────────────────────────
    for concept, category, level in FIRST_WORDS:
        clean_key = re.sub(r"\s*\(.*?\)", "", concept).strip().lower()
        if clean_key in seen:
            continue
        seen.add(clean_key)
        # First words don't need definitions — they need image + audio grounding
        # Use a minimal sentence that a caregiver would actually say
        caregiver_sentences = {
            "no": "No means stop or do not do that.",
            "yes": "Yes means something is true or you agree.",
            "more": "More means you want another one or extra of something.",
            "all gone": "All gone means there is none left.",
            "up": "Up means higher, above, or going higher.",
            "down": "Down means lower, below, or going lower.",
            "go": "Go means move or start moving.",
            "stop": "Stop means do not move or do not continue.",
            "hot": "Hot means something has a lot of heat and can burn you.",
            "cold": "Cold means something does not have much heat and feels cool.",
            "big": "Big means something is large in size.",
            "little": "Little means something is small in size.",
            "wet": "Wet means covered with water or another liquid.",
            "dirty": "Dirty means something needs to be cleaned.",
            "mama": "Mama is the word for mother — the woman who takes care of you.",
            "dada": "Dada is the word for father — the man who takes care of you.",
            "ball": "A ball is a round toy you can throw, kick, or roll.",
            "milk": "Milk is a white drink that comes from cows or mothers.",
            "water": "Water is a clear liquid you drink to stay alive.",
            "hi": "Hi is a word you say when you meet someone.",
            "bye": "Bye is a word you say when you leave someone.",
            "please": "Please is a word you say to ask for something politely.",
            "thank you": "Thank you is a word you say when someone gives you something or helps you.",
        }
        definition = caregiver_sentences.get(clean_key,
            f"{concept.capitalize()} is one of the first words you learn in English.")
        qa_pairs = [{
            "qa_id":         str(uuid.uuid4()),
            "question":      f"What does {clean_key} mean?",
            "answer":        definition,
            "confidence":    0.95,
            "review_status": "first_words",
        }]
        records.append({
            "id":            f"concept_{len(records):05d}",
            "concept":       concept,
            "clean":         clean_key,
            "category":      category,
            "level":         level,
            "frequency":     word_frequency(clean_key, "en"),
            "definition":    definition,
            "qa_pairs":      qa_pairs,
            "synonyms":      [],
            "open_images_class": OI_LABEL_MAP.get(clean_key, ""),
            "misconception": None,
            "correction":    None,
            "images":        [],
            "wiki_title":    "",
            "wiki_text":     "",
        })

    for concept, category, level in SEED_TAXONOMY:
        if len(records) >= max_concepts:
            break
        clean_key = re.sub(r"\s*\(.*?\)", "", concept).strip().lower()
        if clean_key in seen:
            continue
        seen.add(clean_key)

        wn_defn   = get_wordnet_definition(concept)
        definition = make_definition_sentence(concept, category, wn_defn)
        qa_pairs  = make_qa_pairs(concept, definition, wn_defn)
        oi_class  = OI_LABEL_MAP.get(clean_key, "")

        # Get synonyms / related words from WordNet
        synonyms = []
        synsets = wn.synsets(clean_key)
        if synsets:
            for lemma in synsets[0].lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != clean_key:
                    synonyms.append(name)
        synonyms = synonyms[:5]

        # Frequency rank (higher = more common in English)
        freq = word_frequency(clean_key, "en")

        misconception_data = MISCONCEPTIONS.get(clean_key)
        misconception = misconception_data[0] if misconception_data else None
        correction    = misconception_data[1] if misconception_data else None

        # Add a misconception-correction QA pair if available
        if misconception_data:
            qa_pairs.append({
                "qa_id":         str(uuid.uuid4()),
                "question":      f"Is it true that {misconception_data[0][:60].rstrip('.')}?",
                "answer":        f"No. {misconception_data[1]}",
                "confidence":    0.95,
                "review_status": "misconception_correction",
            })

        record = {
            "id":            f"concept_{len(records):05d}",
            "concept":       concept,
            "clean":         clean_key,
            "category":      category,
            "level":         level,
            "frequency":     freq,
            "definition":    definition,
            "qa_pairs":      qa_pairs,
            "synonyms":      synonyms,
            "open_images_class": oi_class,
            "misconception": misconception,
            "correction":    correction,
            "images":        [],
            "wiki_title":    "",   # filled by match_wikipedia step
            "wiki_text":     "",   # filled by match_wikipedia step
        }
        records.append(record)

    print(f"  Seed taxonomy: {len(records):,} concepts")

    # ── WordNet expansion ──────────────────────────────────────────────────
    # For noun categories, expand with hyponyms (more specific concepts)
    expandable = {"animal", "food", "vehicle", "clothing", "home", "nature",
                  "place", "health", "science"}
    expansion_roots = [
        ("animal", "animal.n.01", 2),
        ("food", "food.n.01", 2),
        ("vehicle", "vehicle.n.01", 2),
        ("clothing", "clothing.n.01", 2),
        ("plant", "plant.n.02", 2),
        ("tool", "tool.n.01", 3),
        ("furniture", "furniture.n.01", 2),
        ("musical_instrument", "musical_instrument.n.01", 3),
        ("sport", "sport.n.01", 3),
        ("game", "game.n.01", 3),
        ("weapon", "weapon.n.01", 3),
        ("container", "container.n.01", 2),
        ("device", "device.n.01", 3),
        ("structure", "structure.n.01", 3),
    ]

    def expand_synset(synset_name: str, category: str, level: int,
                      depth: int = 2) -> None:
        try:
            root = wn.synset(synset_name)
        except Exception:
            return
        for hypo in root.hyponyms():
            for lemma in hypo.lemmas():
                word = lemma.name().replace("_", " ").lower()
                if word in seen:
                    continue
                if not word.isascii():
                    continue
                if len(records) >= max_concepts:
                    return
                freq = word_frequency(word, "en")
                if freq < 1e-6:   # skip very rare words
                    continue
                # Skip English function words that slipped through hyponym expansion
                _FUNCTION_WORDS = {
                    "a", "an", "the", "and", "or", "but", "if", "in", "on",
                    "at", "to", "of", "for", "with", "by", "from", "up", "down",
                    "out", "as", "it", "its", "is", "are", "was", "were", "be",
                    "been", "being", "have", "has", "had", "do", "does", "did",
                    "will", "would", "could", "should", "may", "might", "shall",
                    "can", "not", "no", "so", "yet", "nor", "after", "before",
                    "well", "just", "also", "then", "than", "that", "this",
                    "these", "those", "he", "she", "we", "you", "they", "i",
                    "me", "him", "her", "us", "them", "what", "which", "who",
                    "whom", "whose", "when", "where", "why", "how", "all",
                    "both", "each", "few", "more", "most", "other", "some",
                    "such", "into", "through", "during", "about", "per",
                    "once", "over", "under", "again", "there", "here",
                    "any", "only", "same", "own", "very",
                }
                if word in _FUNCTION_WORDS or len(word) <= 1:
                    continue
                # Require at least one noun or adjective sense for expansion words
                if not (wn.synsets(word, pos=wn.NOUN) or wn.synsets(word, pos=wn.ADJ)):
                    continue
                seen.add(word)
                wn_defn   = hypo.definition() or ""
                if wn_defn:
                    wn_defn = wn_defn[0].upper() + wn_defn[1:]
                definition = make_definition_sentence(word, category, wn_defn)
                qa_pairs  = make_qa_pairs(word, definition, wn_defn)
                oi_class  = OI_LABEL_MAP.get(word, "")
                records.append({
                    "id":            f"concept_{len(records):05d}",
                    "concept":       word,
                    "clean":         word,
                    "category":      category,
                    "level":         level,
                    "frequency":     freq,
                    "definition":    definition,
                    "misconception": None,
                    "correction":    None,
                    "images":        [],
                    "qa_pairs":      qa_pairs,
                    "synonyms":      [],
                    "open_images_class": oi_class,
                    "wiki_title":    "",
                    "wiki_text":     "",
                })
            if depth > 1:
                expand_synset(hypo.name(), category, level, depth - 1)

    print("  Expanding with WordNet hyponyms...")
    for cat_name, synset_name, level in expansion_roots:
        if len(records) >= max_concepts:
            break
        expand_synset(synset_name, cat_name, level)
    print(f"  After expansion: {len(records):,} concepts")

    # ── Sort by level then frequency ──────────────────────────────────────
    records.sort(key=lambda r: (r["level"], -r["frequency"]))

    # ── Match Wikipedia definitions ───────────────────────────────────────
    print("  Matching Simple English Wikipedia definitions...")
    wiki_jsonl = out_path.parent / "simple_wiki_articles.jsonl"
    if wiki_jsonl.exists():
        # Build a title → first_paragraph lookup (streaming, memory-efficient)
        title_index: dict[str, str] = {}
        with open(wiki_jsonl, encoding="utf-8") as wfh:
            for line in wfh:
                try:
                    rec = json.loads(line)
                    title_index[rec["title"].lower()] = rec["text"][:600]
                except Exception:
                    pass
        matched = 0
        for r in records:
            key = r["clean"]
            if key in title_index:
                r["wiki_title"] = key.title()
                r["wiki_text"]  = title_index[key][:600]
                matched += 1
        print(f"  Wikipedia matches: {matched:,}/{len(records):,}")
    else:
        print("  (simple_wiki_articles.jsonl not found — skipping wiki match)")

    # ── Write output ──────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDataset written: {out_path}")
    print(f"  {len(records):,} concepts across levels 0-3")
    by_level = {}
    for r in records:
        by_level.setdefault(r["level"], 0)
        by_level[r["level"]] += 1
    for lv in sorted(by_level):
        names = {-1: "first-words", 0: "infant", 1: "toddler", 2: "pre-K", 3: "kindergarten"}
        print(f"  Level {lv} ({names.get(lv, str(lv))}): {by_level[lv]:,}")
    with_images = sum(1 for r in records if r["open_images_class"])
    print(f"  Concepts with Open Images class: {with_images:,}")
    with_wiki = sum(1 for r in records if r["wiki_title"])
    print(f"  Concepts with Wikipedia text: {with_wiki:,}")


# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build developmental concept dataset")
    parser.add_argument("--out",  default="data/foundation/concept_dataset.jsonl")
    parser.add_argument("--max",  type=int, default=10000,
                        help="Max concepts to include (default 10000)")
    args = parser.parse_args()
    print(f"Building concept dataset (max={args.max:,})...")
    build_dataset(args.max, Path(args.out))


if __name__ == "__main__":
    main()
