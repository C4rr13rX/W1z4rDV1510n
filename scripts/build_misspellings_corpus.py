#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_misspellings_corpus.py -- Stage 41: English Misspellings & Spelling Correction

Trains the node to:
  1. Recognize 100+ common and uncommon English misspellings
  2. Respond correctly to misspelled INPUT (understand intent, answer the question)
  3. Correct spelling when explicitly asked with a scope marker
  4. Explain WHY words are commonly misspelled (rules, mnemonics)
  5. Handle homophones and commonly confused words

SCOPE PRINCIPLE (prevents over-training):
  - Section A/B/D: spell-check questions ALWAYS contain an explicit scope marker
    ("correct the spelling of", "fix the spelling in", "spell-check this", etc.)
  - Section C: misspelled-input questions have NO correction marker -- the node
    learns to understand intent and answer correctly despite the spelling errors.
  The node learns: marker present -> give spelling correction;
                   no marker     -> understand and answer the question.

Usage:
  python scripts/build_misspellings_corpus.py --node localhost:8090
  python scripts/build_misspellings_corpus.py --repeats 100
  python scripts/build_misspellings_corpus.py --stages 41 --repeats 50
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import urllib.request
import urllib.error

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BATCH_SIZE = 10

# ---------------------------------------------------------------------------
# SECTION A: Single-word spelling correction (explicit scope marker in Q)
# ---------------------------------------------------------------------------
SPELLING_WORD_PAIRS: list[tuple[str, str]] = [
    ("What is the correct spelling of 'recieve'?",
     "The correct spelling is 'receive'. The misspelling 'recieve' swaps the vowels. Remember: 'i before e except after c' — the letter c comes first, so it is 'cei' not 'cie'."),
    ("How do you spell 'accomodation'? Is that correct?",
     "The correct spelling is 'accommodation' — with double c and double m. The misspelling 'accomodation' is missing one m. Memory tip: the word accommodates two sets of double letters: cc and mm."),
    ("Is 'definately' a correct spelling?",
     "No. The correct spelling is 'definitely'. 'Definately' is one of the most common misspellings in English. The word comes from 'definite' plus the suffix '-ly', so there is no 'a' in it."),
    ("Correct the spelling of 'seperate'.",
     "The correct spelling is 'separate'. The misspelling 'seperate' replaces the first 'a' with an 'e'. Memory tip: there is 'a rat' in sep-a-rat-e."),
    ("Fix the spelling of 'occured'.",
     "The correct spelling is 'occurred' — with double c and double r. 'Occured' is missing the extra r. When adding '-ed' to 'occur', the final r is doubled because the last syllable is stressed."),
    ("Is 'neccessary' spelled correctly?",
     "No. The correct spelling is 'necessary' — one c and double s. Memory tip: a shirt has one Collar and two Sleeves — necessary has one c and two s's."),
    ("How do you spell 'embarrasment'?",
     "The correct spelling is 'embarrassment' — double r and double s. Common misspelling: 'embarrasment' (only one r). Think of it as being really, really (double r) embarrassed with a sharp sensation (double s)."),
    ("Correct the spelling of 'priviledge'.",
     "The correct spelling is 'privilege' — no 'd'. The misspelling 'priviledge' inserts a d before the g. There is no 'ledge' in privilege."),
    ("What is the correct spelling of 'independant'?",
     "The correct spelling is 'independent' — ending in '-ent', not '-ant'. The misspelling 'independant' uses the wrong suffix. Related words: independence, independently."),
    ("Is 'wierd' spelled correctly?",
     "No. The correct spelling is 'weird'. 'Wierd' violates the spelling. Weird is one of the exceptions to the 'i before e' rule — it is 'ei' even though no c precedes it. Weird is just... weird."),
    ("How do you spell 'beleive'? Fix the spelling.",
     "The correct spelling is 'believe'. 'Beleive' reverses the 'ie'. Remember: 'i before e except after c' — believe has no c before the ie, so it follows the standard rule: bel-i-e-ve."),
    ("Correct the spelling of 'arguement'.",
     "The correct spelling is 'argument' — no 'e' between 'argu' and 'ment'. The misspelling 'arguement' adds an unnecessary e. When 'argue' takes a suffix starting with a consonant, the final e is dropped."),
    ("Fix the spelling of 'grammer'.",
     "The correct spelling is 'grammar' — ending in '-ar', not '-er'. The misspelling 'grammer' uses the wrong vowel. Memory tip: a grammAR student gets an A."),
    ("Is 'tommorrow' correct spelling?",
     "No. The correct spelling is 'tomorrow' — one m and double r. Common misspelling: 'tommorrow' doubles the m incorrectly. Think: to + morrow (morrow is an archaic word for the next day)."),
    ("Correct the spelling of 'freind'.",
     "The correct spelling is 'friend'. 'Freind' swaps the ie to ei. Apply 'i before e except after c': no c precedes, so it is fri-end, not frei-nd."),
    ("What is the correct spelling of 'enviroment'?",
     "The correct spelling is 'environment' — with an 'n' before the 'ment': environ-ment. 'Enviroment' drops the second n. The root word is 'environ' (meaning surroundings)."),
    ("How do you spell 'calender'? Is it the same as 'calendar'?",
     "These are different words. 'Calendar' (ending in -ar) is the system for organizing dates. 'Calender' (ending in -er) is a machine for pressing cloth or paper. If you mean the date-tracking tool, the correct spelling is 'calendar'."),
    ("Fix the spelling of 'rythm'.",
     "The correct spelling is 'rhythm' — with 'hyt' in the middle. 'Rythm' omits the first h. Rhythm is unusual in English for having no vowels in its last five letters. Memory trick: Rhythm Helps Your Two Hips Move."),
    ("Correct the spelling of 'vaccum'.",
     "The correct spelling is 'vacuum' — with double u. The misspelling 'vaccum' doubles the c instead. Think: a vacuum has a lot of empty space, and the double u represents that emptiness."),
    ("Is 'existance' spelled correctly?",
     "No. The correct spelling is 'existence' — ending in '-ence', not '-ance'. Related words: exist, existent, nonexistent — all use the 'e' form."),
    ("How do you spell 'conscence'? Fix the spelling.",
     "The correct spelling is 'conscience' — con-sci-ence. The silent 'sc' and the 'ie' make this tricky. 'Conscence' is missing the 'i'. Memory: your conscience is like a science of the self."),
    ("Correct the spelling of 'restaraunt'.",
     "The correct spelling is 'restaurant'. Both 'restaraunt' and 'resturant' are common misspellings. The 'au' appears in the middle: rest-au-rant. Think of the French origin: a place to 'restore' you."),
    ("Fix the spelling of 'pronounciation'.",
     "The correct spelling is 'pronunciation' — there is no 'o' after 'pron'. 'Pronounciation' is extremely common because we base it on the verb 'pronounce', but the noun form drops the second o: pro-nun-ci-a-tion."),
    ("Is 'perseverence' spelled correctly?",
     "No. The correct spelling is 'perseverance' — ending in '-ance'. The misspelling 'perseverence' uses '-ence'. Memory tip: perseverANCE — think of an ANCE (dance) as persevering through the steps."),
    ("Correct the spelling of 'millenium'.",
     "The correct spelling is 'millennium' — double l and double n. 'Millenium' is missing one n. From Latin: mille (thousand) + annum (year) — two n's from annum."),
    ("What is the correct spelling of 'suprise'?",
     "The correct spelling is 'surprise' — with the first 'r': sur-prise. 'Suprise' drops that r. The prefix is 'sur-' (over/above), not 'su-'."),
    ("Fix the spelling of 'occassion'.",
     "The correct spelling is 'occasion' — double c, single s. 'Occassion' doubles the s incorrectly. The cc is present because of the prefix oc- (a form of ob-) plus the Latin root casio."),
    ("Correct the spelling of 'liason'.",
     "The correct spelling is 'liaison' — with two i's: li-ai-son. 'Liason' drops the first i. Liaison comes from French and retains the unusual vowel sequence."),
    ("Is 'questionaire' spelled correctly?",
     "No. The correct spelling is 'questionnaire' — double n. 'Questionaire' has only one n. The word comes from French questionnaire, which keeps both n's."),
    ("How do you spell 'mispell'? Fix the spelling.",
     "The correct spelling is 'misspell' — double s. The prefix 'mis-' attaches to 'spell', giving mis-spell with two s's. Ironically, 'misspell' is itself frequently misspelled."),
    ("Correct the spelling of 'maintenence'.",
     "The correct spelling is 'maintenance' — main-te-nance. 'Maintenence' and 'maintanance' are both common errors. Think of the verb 'maintain' plus the suffix '-ance': maintain-ance, then drop the final 'ain'."),
    ("What is the correct spelling of 'harrass'?",
     "The correct spelling is 'harass' — one r and double s. 'Harrass' doubles the r incorrectly. Memory: one r, two s's — harASS."),
    ("Fix the spelling of 'judgement'.",
     "Both 'judgment' and 'judgement' are accepted spellings. American English prefers 'judgment' (no e after judg). British English commonly uses 'judgement'. In legal contexts, American usage strongly favors 'judgment'."),
    ("Correct the spelling of 'foriegn'.",
     "The correct spelling is 'foreign'. 'Foriegn' swaps the 'ei' to 'ie'. Foreign is one of many exceptions to the 'i before e' rule — it uses 'ei' with no preceding c."),
    ("Is 'publically' spelled correctly?",
     "No. The correct spelling is 'publicly' — no 'al' before the '-ly'. 'Publically' adds an extra syllable. The adjective 'public' takes '-ly' directly: public + ly = publicly."),
    ("How do you spell 'garentee'? Fix the spelling.",
     "The correct spelling is 'guarantee' — gua-ran-tee. 'Garentee' and 'guarentee' are both wrong. The 'gua' opening is unusual in English; the word comes from Old French garantie."),
    ("Correct the spelling of 'fourty'.",
     "The correct spelling is 'forty' — no 'u'. 'Fourty' adds a u by analogy with 'four', but the number forty drops the u. All other multiples of ten also drop unusual vowels: not 'fivety' but fifty."),
    ("Fix the spelling of 'truely'.",
     "The correct spelling is 'truly' — no 'e'. 'Truely' keeps the e from 'true', but when adding '-ly' to words ending in silent e, the e is dropped: true -> truly, whole -> wholly."),
    ("Is 'untill' correct spelling?",
     "No. The correct spelling is 'until' — one l at the end. 'Untill' doubles the l by analogy with 'till', but the standard spelling of the conjunction meaning 'up to the time that' is 'until' with one l."),
    ("Correct the spelling of 'acheive'.",
     "The correct spelling is 'achieve'. 'Acheive' swaps the e and i. Apply 'i before e except after c': achieve has no c before 'ie', so the i comes first: ach-ie-ve."),
]

# ---------------------------------------------------------------------------
# SECTION B: Sentence-level spelling correction (explicit scope marker in Q)
# ---------------------------------------------------------------------------
SPELLING_SENTENCE_PAIRS: list[tuple[str, str]] = [
    ("Fix the spelling in this sentence: I definately recieved the letter yesterday.",
     "Corrected: 'I definitely received the letter yesterday.' Corrections: 'definately' -> 'definitely' (no 'a' in the word); 'recieved' -> 'received' (i before e except after c)."),
    ("Please spell-check this: The goverment needs to addres the enviroment.",
     "Corrected: 'The government needs to address the environment.' Corrections: 'goverment' -> 'government' (missing n); 'addres' -> 'address' (double d and double s); 'enviroment' -> 'environment' (missing n before ment)."),
    ("Correct the spelling in: She was embarased and felt wierd about the whole situatoin.",
     "Corrected: 'She was embarrassed and felt weird about the whole situation.' Corrections: 'embarased' -> 'embarrassed' (double r, double s); 'wierd' -> 'weird' (ei not ie); 'situatoin' -> 'situation' (transposed letters)."),
    ("Fix the misspelled words: The comittee reccomended a seperate commitee for the occassion.",
     "Corrected: 'The committee recommended a separate committee for the occasion.' Corrections: 'comittee' -> 'committee' (double m, double t, double e); 'reccomended' -> 'recommended' (one c, double m); 'seperate' -> 'separate' (a not e in middle); 'occassion' -> 'occasion' (double c, single s)."),
    ("Spell-check this sentence: Its a privelege to recieve an independant recomendation.",
     "Corrected: 'It's a privilege to receive an independent recommendation.' Corrections: 'Its' -> 'It's' (contraction needs apostrophe); 'privelege' -> 'privilege'; 'recieve' -> 'receive'; 'independant' -> 'independent'; 'recomendation' -> 'recommendation' (double m)."),
    ("Fix the spelling: The bussiness man argueed that the maintanance was neccessary.",
     "Corrected: 'The businessman argued that the maintenance was necessary.' Corrections: 'bussiness' -> 'business' (one s in middle); 'argueed' -> 'argued' (no double e); 'maintanance' -> 'maintenance'; 'neccessary' -> 'necessary' (one c, double s)."),
    ("Please correct this: I beleive the rythm of the musick is beatiful.",
     "Corrected: 'I believe the rhythm of the music is beautiful.' Corrections: 'beleive' -> 'believe' (i before e); 'rythm' -> 'rhythm' (h after r); 'musick' -> 'music' (no k); 'beatiful' -> 'beautiful' (eau not ea)."),
    ("Fix the misspellings: Tomorow I will visit the libary and do some reserch.",
     "Corrected: 'Tomorrow I will visit the library and do some research.' Corrections: 'Tomorow' -> 'Tomorrow' (double r); 'libary' -> 'library' (missing r in middle); 'reserch' -> 'research' (ea not just e)."),
    ("Spell-check: The sientist made an importent discrovery about the atmosfere.",
     "Corrected: 'The scientist made an important discovery about the atmosphere.' Corrections: 'sientist' -> 'scientist' (c before i in sci); 'importent' -> 'important' (ant not ent); 'discrovery' -> 'discovery' (no r after disc); 'atmosfere' -> 'atmosphere' (ph not f, ere not fere)."),
    ("Fix the spelling in this text: She was absolutly suprised by the anouncment.",
     "Corrected: 'She was absolutely surprised by the announcement.' Corrections: 'absolutly' -> 'absolutely' (e before ly); 'suprised' -> 'surprised' (r before p: sur-prised); 'anouncment' -> 'announcement' (double n, e before ment)."),
    ("Correct the misspelled words: The foriegn minister gave a persuation speach.",
     "Corrected: 'The foreign minister gave a persuasion speech.' Corrections: 'foriegn' -> 'foreign' (ei not ie); 'persuation' -> 'persuasion' (sua not tua); 'speach' -> 'speech' (ee not ea)."),
    ("Please fix: I have knowlege of the apropriate proceedure.",
     "Corrected: 'I have knowledge of the appropriate procedure.' Corrections: 'knowlege' -> 'knowledge' (dge at end); 'apropriate' -> 'appropriate' (double p); 'proceedure' -> 'procedure' (one e after proc)."),
    ("Spell-check this sentence: The millenium celerbation was truely magnificant.",
     "Corrected: 'The millennium celebration was truly magnificent.' Corrections: 'millenium' -> 'millennium' (double l, double n); 'celerbation' -> 'celebration' (bra not brr, no transposition); 'truely' -> 'truly' (drop e before ly); 'magnificant' -> 'magnificent' (ent not ant)."),
    ("Fix the spelling: He reccomends we aqcuire a guarentee before the negociation.",
     "Corrected: 'He recommends we acquire a guarantee before the negotiation.' Corrections: 'reccomends' -> 'recommends' (one c, double m); 'aqcuire' -> 'acquire' (c before q is wrong: acq); 'guarentee' -> 'guarantee' (gua-ran-tee); 'negociation' -> 'negotiation' (t not c before iation)."),
    ("Correct the spelling: The inteligent studdent acheived excellant grades in sceince.",
     "Corrected: 'The intelligent student achieved excellent grades in science.' Corrections: 'inteligent' -> 'intelligent' (double l); 'studdent' -> 'student' (one d); 'acheived' -> 'achieved' (ie not ei); 'excellant' -> 'excellent' (ent not ant); 'sceince' -> 'science' (ie not ei, no transposition)."),
]

# ---------------------------------------------------------------------------
# SECTION C: Understanding misspelled INPUT (NO scope marker in Q)
# The node answers the question correctly despite the misspellings.
# ---------------------------------------------------------------------------
MISSPELLED_INPUT_PAIRS: list[tuple[str, str]] = [
    ("Wat is the captial of Frnce?",
     "The capital of France is Paris."),
    ("How do u spel acomodate?",
     "The correct spelling is 'accommodate' — with double c and double m."),
    ("Wuts the difrence betwen there their and theyre?",
     "'There' refers to a place or existence. 'Their' shows possession (belonging to them). 'They're' is a contraction of 'they are'."),
    ("Cna you explane wat photosinthesis is?",
     "Photosynthesis is the process plants use to convert sunlight, water, and carbon dioxide into glucose and oxygen."),
    ("Wut is AI and how does mchine lerning wrk?",
     "Artificial intelligence is the simulation of human cognitive processes by computers. Machine learning is a subset of AI where systems learn patterns from data to improve performance without being explicitly programmed for each task."),
    ("Who invnted the telefone?",
     "Alexander Graham Bell is credited with inventing and patenting the first practical telephone in 1876."),
    ("Is the erth realy rund?",
     "Yes. Earth is an oblate spheroid — slightly flattened at the poles and bulging at the equator — but for most purposes it is described as round."),
    ("Wat happend in WW2?",
     "World War II was a global conflict from 1939 to 1945. It involved most of the world's nations and resulted in approximately 70 to 85 million deaths. It ended with the defeat of Nazi Germany and Imperial Japan."),
    ("Ow do u calculate the area of a circel?",
     "The area of a circle is pi times the radius squared: A = pi * r^2. For example, a circle with radius 5 has an area of approximately 78.54 square units."),
    ("Wut iz gravty and y do things fall?",
     "Gravity is the force of attraction between objects with mass. On Earth, it pulls objects toward the planet's center with an acceleration of 9.8 metres per second squared, which is why things fall when dropped."),
    ("Can u tel me abot the blak hole?",
     "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape from it once past the event horizon."),
    ("Hw mny planits r in the solar sistm?",
     "There are eight recognized planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune."),
    ("Wats the diferance between viruses and bactria?",
     "Bacteria are single-celled living organisms that can reproduce on their own. Viruses are non-living particles that require a host cell to replicate. Bacteria can often be treated with antibiotics; viruses cannot."),
    ("Pls explane how the inernet wrks.",
     "The internet is a global network of interconnected computers communicating through standardized protocols like TCP/IP. When you visit a website, your device sends a request through your internet provider to a server, which sends back data your browser displays."),
    ("Waht is quantom mechanicks?",
     "Quantum mechanics is the branch of physics describing the behavior of matter and energy at the smallest scales, where particles exhibit both wave-like and particle-like properties."),
    ("Hw do u make bred?",
     "Basic bread is made by mixing flour, water, yeast, and salt into a dough, letting it rise (ferment), shaping it, letting it rise again, and baking it in an oven at around 220 degrees Celsius until golden and hollow-sounding when tapped."),
    ("Wut is a derivitave in calculus?",
     "A derivative measures how a function changes as its input changes. It is the instantaneous rate of change of the function at a given point, represented as dy/dx or f'(x)."),
    ("Who waz marie curie and wat did she do?",
     "Marie Curie was a Polish-French physicist and chemist who pioneered research on radioactivity. She was the first woman to win a Nobel Prize and the only person to win Nobel Prizes in two different sciences — Physics (1903) and Chemistry (1911)."),
    ("Wat dose DNA stand for and wut is it?",
     "DNA stands for deoxyribonucleic acid. It is the molecule that carries genetic instructions for the development, functioning, growth, and reproduction of all living organisms."),
    ("Explane the diferense between speed and velosity.",
     "Speed is a scalar quantity — it measures how fast something moves regardless of direction. Velocity is a vector quantity — it measures speed in a specific direction. For example, 60 km/h is a speed; 60 km/h northward is a velocity."),
    ("Hw do computirs store informashun?",
     "Computers store information as binary data — sequences of 0s and 1s. At the hardware level, this is represented by electrical states (on/off) in transistors. Data is stored in RAM (temporary) or on drives (permanent) using these binary patterns."),
    ("Wut is the Hebbian lerning rool?",
     "The Hebbian learning rule states that neurons which fire together wire together. It strengthens connections between neurons that are simultaneously active, forming the basis of associative learning."),
    ("Iz ther life on other plannets?",
     "No confirmed extraterrestrial life has been found yet. Scientists consider it plausible given the vast number of planets in the universe and the discovery of Earth-like exoplanets, but no definitive evidence exists as of 2026."),
    ("Wat does the imune sistem do?",
     "The immune system is the body's defense network that identifies and destroys pathogens, foreign substances, and abnormal cells, protecting the organism from infection and disease."),
    ("How do u find the pythagorean threorm?",
     "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides: a squared plus b squared equals c squared."),
]

# ---------------------------------------------------------------------------
# SECTION D: Homophones and commonly confused words (explicit scope marker)
# ---------------------------------------------------------------------------
HOMOPHONE_PAIRS: list[tuple[str, str]] = [
    ("What is the difference between 'their', 'there', and 'they're'?",
     "'Their' is a possessive pronoun showing ownership: 'their car'. 'There' refers to a place or introduces a sentence: 'over there', 'there is'. 'They're' is a contraction of 'they are': 'they're coming'. Test: replace with 'they are' — if it works, use they're; if it names a place, use there; if it shows ownership, use their."),
    ("Explain the difference between 'your' and 'you're'.",
     "'Your' is a possessive pronoun: 'your book', 'your idea'. 'You're' is a contraction of 'you are': 'you're right', 'you're welcome'. Test: say the sentence with 'you are' — if it sounds correct, use you're; otherwise use your."),
    ("What is the difference between 'its' and 'it's'?",
     "'Its' is a possessive pronoun: 'the dog wagged its tail'. 'It's' is a contraction of 'it is' or 'it has': 'it's raining', 'it's been a long day'. Test: replace with 'it is' or 'it has' — if correct, use it's; otherwise use its."),
    ("Correct usage: when do you use 'affect' vs 'effect'?",
     "'Affect' is usually a verb meaning to influence: 'The weather affected my mood'. 'Effect' is usually a noun meaning the result: 'The effect of exercise is improved health'. Memory trick: Affect is the Action (verb), Effect is the End result (noun). Exceptions: 'effect' can be a verb meaning to bring about ('effect change'), and 'affect' can be a psychology noun."),
    ("Explain the difference between 'accept' and 'except'.",
     "'Accept' means to receive or agree to something: 'I accept your offer'. 'Except' means excluding or apart from: 'everyone except me'. They sound similar but are completely different: accept (to take in), except (to leave out)."),
    ("What is the difference between 'principle' and 'principal'?",
     "'Principle' is a noun meaning a rule or belief: 'a moral principle'. 'Principal' can be a noun (the head of a school, or the main sum of money) or adjective (main, primary): 'the principal reason'. Memory: your principAL is your pAL; a principLE is a ruLE."),
    ("Explain 'complement' vs 'compliment'.",
     "'Complement' means to complete or go well with: 'the wine complements the meal'. 'Compliment' means an expression of praise: 'she gave me a compliment'. Memory: compLEment = compLEte; compLIment = I Like it."),
    ("Correct usage: 'fewer' vs 'less'.",
     "Use 'fewer' with countable nouns (things you can count individually): 'fewer apples', 'fewer people'. Use 'less' with uncountable nouns (quantities treated as a whole): 'less water', 'less time'. Test: can you count the individual items? If yes, use fewer."),
    ("What is the difference between 'who' and 'whom'?",
     "'Who' is used as a subject (performs the action): 'Who called?' 'Whom' is used as an object (receives the action): 'To whom did you speak?' Test: substitute 'he/she' vs 'him/her' — if 'he' fits, use who; if 'him' fits, use whom."),
    ("Explain the difference between 'lay' and 'lie'.",
     "'Lay' is a transitive verb requiring an object — it means to place something: 'Lay the book on the table'. 'Lie' is an intransitive verb meaning to recline: 'I need to lie down'. The confusion increases with past tenses: the past tense of 'lie' is 'lay' ('I lay down yesterday'), and the past tense of 'lay' is 'laid' ('I laid the book down')."),
    ("What is the difference between 'i.e.' and 'e.g.'?",
     "'i.e.' (from Latin id est, meaning 'that is') introduces a restatement or clarification: 'the largest planet, i.e., Jupiter'. 'e.g.' (from Latin exempli gratia, meaning 'for example') introduces one or more examples: 'large planets, e.g., Jupiter and Saturn'. Memory: i.e. = In other words Exactly; e.g. = Example Given."),
    ("Correct usage: 'then' vs 'than'.",
     "'Then' relates to time or sequence: 'We ate dinner, then watched a movie'. 'Than' is used for comparisons: 'She is taller than me'. A common test: 'then' is about when; 'than' is about comparison."),
    ("Explain the difference between 'further' and 'farther'.",
     "'Farther' refers to physical distance: 'The store is farther than I thought'. 'Further' refers to figurative or metaphorical distance and also means 'additional' or 'to a greater degree': 'We need to discuss this further'. If you can measure it in miles or metres, use farther; otherwise use further."),
    ("What is the difference between 'imply' and 'infer'?",
     "'Imply' means to suggest something indirectly — the speaker or writer implies: 'Her silence implied agreement'. 'Infer' means to draw a conclusion from evidence — the listener or reader infers: 'I inferred from her silence that she agreed'. Memory: the Implier sends the message; the Inferrer receives and interprets it."),
    ("Correct usage: 'that' vs 'which'.",
     "'That' introduces restrictive clauses — essential information that defines which thing is meant: 'The car that I bought is red' (specifies which car). 'Which' introduces non-restrictive clauses — additional, non-essential information set off by commas: 'My car, which I bought last year, is red'. If removing the clause changes the meaning, use 'that'; if it just adds information, use 'which'."),
]

# ---------------------------------------------------------------------------
# SECTION E: Spelling rules and patterns
# ---------------------------------------------------------------------------
SPELLING_RULES_PAIRS: list[tuple[str, str]] = [
    ("Explain the 'i before e except after c' spelling rule.",
     "The rule 'i before e except after c' applies when the vowel combination makes a long 'ee' sound. Examples following the rule: believe, achieve, field, piece (ie after consonant); receive, ceiling, deceive (ei after c). Exceptions: weird, seize, either, neither, protein, species. The rule is a helpful guideline but has many exceptions."),
    ("When do you double the final consonant before adding a suffix?",
     "Double the final consonant when: (1) the word ends in a single consonant, (2) preceded by a single vowel, (3) the last syllable is stressed, and (4) the suffix begins with a vowel. Examples: run -> running, begin -> beginning, prefer -> preferred. Do NOT double if the last syllable is unstressed: happen -> happening, listen -> listened."),
    ("Explain when to drop the silent 'e' before a suffix.",
     "Drop the silent final e when adding a suffix that begins with a vowel: love -> loving, hope -> hoping, true -> truly. Keep the silent e when adding a suffix that begins with a consonant: hope -> hopeful, love -> lovely. Exceptions: judgment (not judgement in American English), acknowledgment."),
    ("What is the rule for adding '-ful', '-less', '-ness' to words?",
     "These suffixes attach directly to the base word without spelling changes in most cases: hope + ful = hopeful, care + less = careless, dark + ness = darkness. Exception: words ending in y change y to i before these suffixes: happy -> happiness, beauty -> beautiful. Words ending in silent e keep the e before consonant suffixes: hopeful, useful."),
    ("Explain the difference between '-ible' and '-able' endings.",
     "Both endings mean 'capable of' or 'worthy of'. General guidelines: '-able' is more common and often attaches to complete English words (readable, breakable, fashionable). '-ible' often attaches to Latin roots that are not complete English words (visible, audible, credible, flexible). When in doubt, '-able' is more likely correct for newer words."),
    ("When do you use '-ance'/'-ancy' vs '-ence'/'-ency'?",
     "These endings follow the preceding consonant or vowel. After hard c (k sound) or g, use '-ance': significance, elegance. After soft c (s sound) or g, use '-ence': licence, intelligence. After most other letters, patterns are less predictable and often must be memorized: resistance vs persistence, relevance vs excellence. When in doubt, look it up."),
    ("Explain the 'all' prefix rule in spelling.",
     "When 'all' is a prefix, it typically loses one l: already (not allready), although (not allthough), altogether (not alltogether), almost (not allmost), always (not allways), almighty (not allmighty). However, when 'all' stands alone as a word before another word, both l's remain: all right, all day."),
    ("What is the rule for possessive apostrophes?",
     "To form possessives: (1) Singular nouns: add apostrophe + s: cat's, James's, boss's. (2) Plural nouns ending in s: add apostrophe only: cats', players', bosses'. (3) Plural nouns not ending in s: add apostrophe + s: children's, men's, geese's. (4) Possessive pronouns never use apostrophes: its, hers, theirs, ours, yours, whose."),
]

ALL_PAIRS = (
    SPELLING_WORD_PAIRS
    + SPELLING_SENTENCE_PAIRS
    + MISSPELLED_INPUT_PAIRS
    + HOMOPHONE_PAIRS
    + SPELLING_RULES_PAIRS
)

STAGES = {41: "English misspellings, spelling correction, homophones, and understanding misspelled input"}


def post(node: str, path: str, body: dict, timeout: int = 20) -> dict:
    data = json.dumps(body).encode()
    url = f"http://{node}{path}"
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as exc:
        return {"error": str(exc)}


def ingest_batch(node: str, pairs: list[tuple[str, str]], book_id: str) -> dict:
    candidates = [
        {
            "qa_id": str(uuid.uuid4()),
            "question": q,
            "answer": a,
            "book_id": book_id,
            "confidence": 0.94,
            "evidence": "Stage 41 English misspellings and spelling correction corpus",
            "review_status": "approved",
        }
        for q, a in pairs
    ]
    return post(node, "/qa/ingest", {"candidates": candidates, "pool": "correction"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="localhost:8090")
    parser.add_argument("--stages", default="41")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Times to repeat the full corpus (default 100 = ~13000 ingestions)")
    args = parser.parse_args()

    total = len(ALL_PAIRS)
    print(f"Stage 41 — English Misspellings Corpus")
    print(f"  {total} QA pairs x {args.repeats} repeats = {total * args.repeats} total ingestions")
    print(f"  Breakdown: {len(SPELLING_WORD_PAIRS)} word corrections, "
          f"{len(SPELLING_SENTENCE_PAIRS)} sentence corrections, "
          f"{len(MISSPELLED_INPUT_PAIRS)} misspelled-input understanding, "
          f"{len(HOMOPHONE_PAIRS)} homophones, "
          f"{len(SPELLING_RULES_PAIRS)} spelling rules")
    print(f"  Node: {args.node}\n")

    errors = 0
    batches_done = 0
    total_batches = (total * args.repeats + BATCH_SIZE - 1) // BATCH_SIZE

    for repeat in range(args.repeats):
        book_id = f"misspellings_r{repeat:04d}"
        offset = (repeat * 13) % total
        shuffled = ALL_PAIRS[offset:] + ALL_PAIRS[:offset]

        for i in range(0, len(shuffled), BATCH_SIZE):
            batch = shuffled[i:i + BATCH_SIZE]
            result = ingest_batch(args.node, batch, book_id)
            batches_done += 1
            if result.get("error"):
                errors += 1
                if errors <= 5:
                    print(f"  ERROR batch {batches_done}: {result['error']}", file=sys.stderr)
            elif batches_done % 50 == 0 or batches_done == total_batches:
                pct = batches_done / total_batches * 100
                print(f"  [{batches_done}/{total_batches}] {pct:.1f}%", flush=True)
            time.sleep(0.05)

    print(f"\nDone. {batches_done} batches, {errors} errors.")
    print("Node trained on misspelling recognition, correction, and understanding.")


if __name__ == "__main__":
    main()
