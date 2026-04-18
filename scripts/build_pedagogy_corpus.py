#!/usr/bin/env python3
"""build_pedagogy_corpus.py — Stage 35: Educational Pedagogy & Curriculum Design

Trains the model on the science of learning and how to design exceptional
educational curricula — drawing from cognitive psychology, neuroscience,
and evidence-based pedagogy.

Sources:
  - 38 evidence-based learning principles (hardcoded, with mechanism + application)
  - 5 major curriculum design frameworks (ADDIE, UbD, Gagné, PBL, UDL)
  - Bloom's Revised Taxonomy (complete with verbs and examples)
  - ~65 Wikipedia articles on learning science and educational psychology
  - PubMed/NCBI peer-reviewed research on learning and memory

Stage 35: Evidence-based pedagogy and advanced curriculum design

Usage:
  python scripts/build_pedagogy_corpus.py --stages 35 --node localhost:8090
"""

import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ---------------------------------------------------------------------------
# Evidence-Based Learning Principles
# Each entry: principle, mechanism (WHY it works), application (HOW to use),
# strength of evidence, and example.
# ---------------------------------------------------------------------------

LEARNING_PRINCIPLES = [
    {
        'name': 'Spaced Repetition (Distributed Practice)',
        'evidence': 'Strongest in cognitive psychology — replicated hundreds of times since Ebbinghaus (1885)',
        'mechanism': 'The "forgetting curve" shows memory decays exponentially. Re-studying just as memory fades (spacing) forces retrieval effort, which strengthens the memory trace via reconsolidation. Massed practice (cramming) produces short-term performance but poor long-term retention.',
        'application': 'Schedule reviews at increasing intervals: 1 day → 3 days → 1 week → 2 weeks → 1 month. Systems like Anki implement optimal spacing algorithms. For curriculum design: return to foundational concepts in spiral fashion throughout a course. "Introduce, review, master" across weeks — not within a single session.',
        'example': 'Students who studied vocabulary for 30 min on 3 separate days outperformed students who studied for 90 continuous minutes by 200% on a 1-week retention test.',
    },
    {
        'name': 'Retrieval Practice (The Testing Effect)',
        'evidence': 'Roediger & Karpicke (2006); hundreds of replications. One of the most robust effects in memory research.',
        'mechanism': 'The act of retrieving information from memory — not passively re-reading — strengthens the memory. Effortful retrieval triggers protein synthesis and synaptic strengthening. Re-reading creates an illusion of knowing without building durable traces.',
        'application': 'End every lesson with a no-stakes quiz. Use flashcards. Have students write down everything they remember before reviewing notes ("brain dump"). Free recall > multiple choice > re-reading in terms of retention benefit. Low-stakes frequent testing beats one high-stakes exam for learning.',
        'example': 'Students who tested themselves after reading a passage retained 50% more one week later than students who re-read the passage three more times.',
    },
    {
        'name': 'Interleaving (Mixed Practice)',
        'evidence': 'Rohrer, Taylor (2007); Kornell & Bjork (2008). Particularly robust for mathematics and categorization learning.',
        'mechanism': 'Interleaving different problem types forces learners to identify WHICH strategy to use, not just execute a known strategy. This discrimination training improves both recognition and retrieval. The difficulty feels higher but produces better long-term learning.',
        'application': 'Mix problem types within a practice session rather than completing all problems of type A before moving to type B. In a math course: one derivative problem, then an integral, then a limit — rather than 20 derivatives in a row. Warning: feels harder and slower — students and teachers often resist it despite evidence.',
        'example': 'Students who practiced interleaved math problems scored 43% higher on a shuffled final test than blocked-practice students (Rohrer 2014).',
    },
    {
        'name': 'Elaborative Interrogation',
        'evidence': 'Dunlosky et al. (2013) rated high utility. Robust across subjects.',
        'mechanism': 'Asking "why" and "how" forces learners to connect new information to prior knowledge, creating richer memory traces with more retrieval pathways. Passive reading creates isolated facts; elaboration weaves them into a semantic network.',
        'application': 'Have students explain WHY a fact is true, not just THAT it is true. "Why does a metal spoon feel colder than a wooden spoon at the same temperature?" Encourage "self-explanation" — explain each step of a worked example to yourself.',
        'example': 'Students who generated explanations for facts (e.g., "Why would whales have blubber?") remembered them 2× better than students who simply read the answers.',
    },
    {
        'name': 'Concrete Examples',
        'evidence': 'Dunlosky et al. (2013). Universally effective across domains.',
        'mechanism': 'Abstract principles are hard to encode and retrieve because they lack perceptual anchors. Concrete examples provide specific instances that can be mentally "touched," making the abstract principle more accessible. Multiple examples help learners extract the underlying structure.',
        'application': 'Always present abstract concepts with 2-3 concrete examples from different domains. Then ask students to generate their own examples. The principle: concept → concrete → abstract → transfer.',
        'example': '"Sunk cost fallacy" is remembered better when taught with: "You\'ve eaten half a bad meal but feel obligated to finish because you paid for it."',
    },
    {
        'name': 'Dual Coding (Visual + Verbal)',
        'evidence': 'Paivio (1971, 2006). Mayer\'s multimedia learning research (2001). Strong evidence base.',
        'mechanism': 'Information encoded in both verbal (language) and visual (image/diagram) channels creates two independent memory traces. Either can trigger the other during recall, effectively doubling retrieval pathways. Verbal-only encoding leaves learners with a single fragile trace.',
        'application': 'Pair explanations with diagrams, charts, or illustrations. Have students draw what they\'re learning (concept maps, timeline diagrams, process flowcharts). Don\'t just add decorative images — use explanatory visuals that show STRUCTURE or PROCESS.',
        'example': 'Students who received a multimedia explanation (text + animation) of how a bicycle pump works solved transfer problems 70% better than text-only students (Mayer 1997).',
    },
    {
        'name': 'Worked Examples (for Novices)',
        'evidence': 'Sweller & Cooper (1985); the "worked example effect." Robust in STEM education.',
        'mechanism': 'Novices lack the schemas needed to solve problems independently. Wrestling with unsolvable problems overloads working memory with search processes, leaving no capacity for schema formation. Worked examples offload search, freeing capacity to observe structure.',
        'application': 'For beginners: present fully worked examples before asking students to solve problems. Use "fading" — gradually remove steps as competence grows (worked → partially worked → problem to solve). For experts: worked examples become less helpful (expertise reversal effect) — shift to open-ended problems.',
        'example': 'Novice algebra students learned 2× more from studying worked examples than from solving equivalent problems (Sweller & Cooper 1985).',
    },
    {
        'name': 'Cognitive Load Management',
        'evidence': 'Sweller (1988) — foundational in instructional design. Extensive empirical support.',
        'mechanism': 'Working memory (conscious attention) holds only ~4 items simultaneously. Cognitive load comes in three types: Intrinsic (inherent complexity of content), Extraneous (poor design — wasted mental effort), Germane (productive schema-building). Good instruction minimizes extraneous, manages intrinsic, maximizes germane.',
        'application': 'Reduce split-attention (don\'t put diagram and its labels far apart). Remove redundancy (don\'t narrate text the student can read). Segment complex material into chunks. Present prerequisite knowledge before complex applications. Use worked examples for novices. Goal-free problems reduce extraneous load.',
        'example': 'Students learning geometry from diagrams with integrated labels performed 72% better than those with separate text (split-attention effect eliminated).',
    },
    {
        'name': 'Zone of Proximal Development (ZPD)',
        'evidence': 'Vygotsky (1978). Foundational in developmental psychology and education.',
        'mechanism': 'Learning is maximally efficient just beyond current ability — in the "stretch zone." Too easy: no new schema formation. Too hard: cognitive overload, no schema extraction possible. The ZPD is the gap between what a learner can do alone and what they can do with guidance.',
        'application': 'Calibrate challenge continuously. Use pre-assessments to find current level. Provide scaffolding (temporary support) that fades as competence develops. In AI tutoring: adaptive difficulty systems. Avoid the "too easy plateau" — it feels comfortable but wastes learning time.',
        'example': 'Children learned to solve logic puzzles significantly better when peers or tutors guided them through problems just beyond their solo ability.',
    },
    {
        'name': 'Scaffolding and Gradual Release',
        'evidence': 'Wood, Bruner & Ross (1976). Extensively supported in instructional research.',
        'mechanism': 'Scaffolding = temporary support structures that enable learners to accomplish what they cannot yet do independently. As competence develops, scaffolds are removed. The gradual release model: "I do, we do, you do."',
        'application': 'Start with full support (teacher models), move to guided practice (student with help), then independent practice. In writing: provide sentence starters → paragraph frames → full essays. In coding: provide template → provide hints → full project. Never remove scaffolding too early.',
        'example': '"I do, we do, you do" structure in literacy instruction produced significantly better reading outcomes than sink-or-swim independent practice.',
    },
    {
        'name': 'Growth Mindset',
        'evidence': 'Dweck (2006). Extensive research on attribution theory and achievement.',
        'mechanism': 'Fixed mindset: "Intelligence is innate — I either have it or I don\'t." Growth mindset: "Ability is developed through effort and strategy." Learners with growth mindset persist longer after failure, choose harder challenges, and recover better from setbacks. Neural basis: growth mindset learners show more error-monitoring brain activity (Moser et al. 2011).',
        'application': 'Praise process and strategy, not innate ability. Say "You worked hard and found a better strategy" not "You\'re so smart." Teach that the brain changes with practice (neuroplasticity). Reframe failure as information. Use "Not yet" instead of "Failed." Avoid ability tracking/grouping that signals fixed ceiling.',
        'example': 'Students taught growth mindset + study skills showed significant improvement in math grades vs. control group (Blackwell et al. 2007).',
    },
    {
        'name': 'Deliberate Practice',
        'evidence': 'Ericsson, Krampe & Tesch-Römer (1993). Foundation of expertise research.',
        'mechanism': 'Expertise is built by deliberately targeting weaknesses just beyond current ability, with immediate precise feedback. Mindless repetition of comfortable tasks produces automation, not improvement. The brain adapts specifically to challenges it faces.',
        'application': 'Identify specific weaknesses (not general "practice more"). Design practice that isolates and targets those weaknesses. Require immediate detailed feedback. Push to edge of ability, then slightly beyond. Expert musicians practice weak passages slowly and repeatedly — they don\'t just play the whole piece.',
        'example': 'Chess grandmasters spent 4× more time on deliberate study of difficult positions than average players, with same total practice hours (Charness et al. 2005).',
    },
    {
        'name': 'Metacognition (Thinking About Thinking)',
        'evidence': 'Flavell (1979); extensive cross-domain support. One of the highest-impact skills.',
        'mechanism': 'Metacognitive learners monitor their own understanding, detect confusion, and adjust strategies. Novices often experience the "illusion of knowing" — they feel they understand while reading but cannot recall. Self-monitoring catches this gap.',
        'application': 'Teach students to predict their own test performance before seeing results, then compare. Use "muddiest point" — what\'s still unclear? Teach self-explanation ("Can I explain this in my own words?"). Encourage planning before and reflection after study sessions. Metacognitive journaling.',
        'example': 'Students trained in metacognitive monitoring showed 20-25% improvement in problem-solving across domains (Schraw 1998).',
    },
    {
        'name': 'Generative Learning',
        'evidence': 'Fiorella & Mayer (2015) — 8 well-validated generative learning strategies.',
        'mechanism': 'Generating a representation (summary, map, drawing, explanation) forces active integration and organization of knowledge. Passive reception builds weak traces; generation builds strong, organized schema.',
        'application': '8 strategies (all evidence-supported): Summarizing, Mapping (concept maps), Drawing, Imagining, Self-Testing, Self-Explaining, Teaching Others, Enacting. The Feynman Technique: explain a concept in simple terms → identify gaps → review → simplify more.',
        'example': 'Students who drew diagrams while studying outperformed those who re-read by 30% on transfer questions (Schwamborn et al. 2010).',
    },
    {
        'name': 'Desirable Difficulties',
        'evidence': 'Bjork & Bjork (2011). Counterintuitive and frequently resisted by learners/teachers.',
        'mechanism': 'Conditions that feel harder and produce poorer immediate performance often produce BETTER long-term retention and transfer. The struggle builds stronger memory. Conditions that feel easier (massed practice, re-reading) create fluency illusions without building durable knowledge.',
        'application': 'Implement: spacing (not cramming), interleaving (not blocking), retrieval practice (not re-reading), variable practice conditions (not fixed conditions). When learning feels easy and smooth, it\'s often not sticking. Educate learners that difficulty is a good sign.',
        'example': 'Students trained on variable tennis serves (different speeds/spins) performed worse during practice but 70% better in a transfer test than blocked-practice students.',
    },
    {
        'name': 'Emotional State and Motivation in Learning',
        'evidence': 'Fredrickson broaden-and-build (2001); Pekrun\'s control-value theory (2006).',
        'mechanism': 'Positive emotions broaden attention and cognitive flexibility; negative emotions (anxiety, threat) narrow attention to defensive processing. Optimal learning occurs under "challenge" (high stakes + high confidence), not "threat" (high stakes + low confidence). Curiosity, interest, and enjoyment are the best learning emotions.',
        'application': 'Create safe environments where mistakes are expected and valuable. Make stakes feel meaningful but not threatening. Use novelty and surprise to trigger curiosity. Structure activities so students experience achievable mastery moments regularly. Address math/writing anxiety explicitly — it consumes working memory.',
        'example': 'Math anxiety was shown to directly reduce working memory capacity during math tasks, explaining performance deficits beyond knowledge gaps (Beilock 2010).',
    },
    {
        'name': 'Curiosity and the Information Gap Theory',
        'evidence': 'Loewenstein (1994); Kang et al. (2009). Active area in educational neuroscience.',
        'mechanism': 'Curiosity is triggered by an "information gap" — awareness that you are missing information you care about. Dopamine system activates. Curiosity enhances encoding and long-term memory for the sought information. Prior knowledge paradoxically increases curiosity (you know enough to know what you don\'t know).',
        'application': 'Start lessons with an intriguing question, surprising fact, or paradox BEFORE presenting information. "Why does time slow down near black holes?" before teaching relativity. Use cliffhangers. Let students generate questions before reading. The question must be answerable — complete mystery produces frustration, not curiosity.',
        'example': 'Students who received trivia questions (even when they couldn\'t answer them) showed better memory for the answers plus incidental information presented nearby (Gruber et al. 2014).',
    },
    {
        'name': 'Sleep and Memory Consolidation',
        'evidence': 'Walker (2017); Stickgold (2005). Robust neuroscience — multiple mechanisms identified.',
        'mechanism': 'Memory consolidation occurs primarily during sleep. Slow-wave sleep: replays hippocampal memories and transfers to neocortex for long-term storage. REM sleep: integrates new learning with existing knowledge, supports creative insight and problem-solving. Sleep deprivation blocks memory consolidation — the learning is "lost".',
        'application': 'Schedule intensive learning before sleep periods, not just before tests. Naps (especially 20-90 minutes including REM) boost retention. Teach students that sleep IS studying. Final exam cramming all night is counterproductive — consolidation happens during sleep, not during reviewing.',
        'example': 'Students who slept after learning a grammar rule and tested 12 hours later retained 2× more than those who stayed awake for 12 hours before testing.',
    },
    {
        'name': 'Transfer of Learning',
        'evidence': 'Thorndike & Woodworth (1901) through modern research. Central challenge of education.',
        'mechanism': 'Near transfer: applying learning in similar contexts. Far transfer: applying in novel domains — the holy grail of education. Transfer requires understanding PRINCIPLES (abstract structure), not just procedures. Variable practice, comparing multiple examples, and abstracting the common structure promotes transfer.',
        'application': 'Don\'t just teach procedure — explicitly teach the underlying principle and when to apply it. Use multiple varied examples and ask students to identify what\'s the same. Present problems in varied surface forms. Ask "Where else does this principle apply?" after every major concept.',
        'example': 'Students who compared two analogous examples of a negotiation principle transferred the strategy to a novel scenario 2× more often than those who studied only one example (Gentner et al. 2003).',
    },
    {
        'name': 'Social Learning and Peer Instruction',
        'evidence': 'Bandura (1977); Mazur (1997) peer instruction. Strong classroom evidence.',
        'mechanism': 'Learning through observation of models and peer explanation. Explaining to a peer forces deeper encoding ("protégé effect"). Listening to a peer who recently learned something can be more effective than expert explanation — peer has recently navigated the same confusions.',
        'application': 'Think-pair-share. Peer instruction (students argue for their answer with a neighbor before seeing correct answer). Jigsaw learning (each student becomes expert in one piece, teaches others). Student-generated examples and problems. Collaborative concept mapping.',
        'example': 'Mazur\'s peer instruction in physics increased conceptual understanding gains by 2× compared to traditional lecture (Crouch & Mazur 2001).',
    },
    {
        'name': 'Mastery Learning',
        'evidence': 'Bloom (1968); extensive meta-analyses show effect sizes 1.0-2.0 standard deviations.',
        'mechanism': 'Standard instruction assumes fixed time with variable outcomes — some students fall behind and never recover. Mastery learning fixes outcomes (everyone must reach 80-90% before moving on) and varies time. Prerequisite gaps prevent future learning; mastery eliminates the accumulation of gaps.',
        'application': 'Define mastery criteria before teaching. Provide varied instructional pathways. Check for mastery with formative assessments before progressing. Students who don\'t reach mastery receive corrective instruction, then re-assess. Highly effective but requires flexible pacing (challenging in fixed-time school schedules).',
        'example': 'Bloom (1984): Students in mastery learning conditions performed 98th percentile compared to conventional instruction students at 50th percentile.',
    },
    {
        'name': 'Neuroplasticity and Experience-Dependent Change',
        'evidence': 'Kandel (Nobel 2000); Draganski et al. (2004) structural brain change; extensive neuroscience.',
        'mechanism': 'Hebbian plasticity: "neurons that fire together wire together." LTP (long-term potentiation) strengthens synaptic connections through repeated co-activation. Practice produces measurable structural changes: increased grey matter density, myelination of white matter tracts, synaptic pruning of weak connections. The brain is not fixed at birth.',
        'application': 'Frame learning as literally changing the brain\'s physical structure. Use this to motivate growth mindset. Emphasize deliberate practice — the type of practice matters (target weaknesses, require effort). Intensity + recovery > continuous effort. Sleep, exercise, and nutrition support neuroplasticity.',
        'example': 'Medical students showed measurable increases in hippocampal grey matter density during intense exam preparation, reversing after (Draganski et al. 2006).',
    },
    {
        'name': 'Exercise and Cognitive Function',
        'evidence': 'Ratey (2008); Hillman et al. (2008). Consistent cross-species evidence.',
        'mechanism': 'Aerobic exercise increases BDNF (brain-derived neurotrophic factor) — promotes neurogenesis and synaptic plasticity in hippocampus. Increases blood flow, norepinephrine, and dopamine. Acute exercise improves attention and executive function for 30-120 minutes.',
        'application': 'Schedule physical activity before cognitively demanding learning sessions. Even a 20-minute walk improves attention and retention. Schools with daily PE show better academic performance. For curriculum design: incorporate movement breaks, active learning activities, standing/movement options.',
        'example': 'Students who exercised before vocabulary learning showed 20% better vocabulary retention than sedentary students (Hötting & Röder 2013).',
    },
    {
        'name': 'Feedback Timing and Quality',
        'evidence': 'Hattie & Timperley (2007); Kulhavy & Stock (1989). Complex — depends on task type.',
        'mechanism': 'For procedural tasks: immediate feedback prevents error consolidation. For conceptual tasks: delayed feedback can produce desirable difficulty (student must struggle to self-assess before seeing answer). Feedback must address the task, not just the person. "Your answer was wrong" < "This step produced an error because you didn\'t account for X" < "Here\'s a strategy to find the error yourself."',
        'application': 'Provide task-level feedback (what\'s right/wrong and why). Avoid person-level feedback ("you\'re smart/dumb"). For simple skills: immediate. For understanding: consider delay to encourage self-evaluation first. Make feedback actionable — a learner should know exactly what to do differently.',
        'example': 'Hattie meta-analysis: feedback has effect size of 0.73 (very high). But negative feedback can have negative effect size — quality and framing matter enormously.',
    },
    {
        'name': 'Mnemonics and Memory Systems',
        'evidence': 'Yates (1966) method of loci; Bellezza (1981); consistent empirical support.',
        'mechanism': 'Mnemonics create artificial but vivid associations that leverage existing memory structures (spatial memory, emotional memory) for material that lacks inherent structure. Method of loci exploits the hippocampus\'s strong spatial memory system.',
        'application': 'Method of loci (memory palace): visualize items along a familiar path. Acronyms: ROYGBIV for spectrum colors. Keyword method for foreign vocabulary: connect sound of new word to a vivid image of its meaning. Rhymes and songs. Most effective for lists, ordered sequences, and arbitrary associations.',
        'example': 'World memory champions memorize decks of cards in under 20 seconds using method of loci — demonstrating the extreme power of spatial-visual encoding.',
    },
    {
        'name': 'Flow State in Learning',
        'evidence': 'Csikszentmihalyi (1990, 1997). Neuroscience of flow: Kotler (2014).',
        'mechanism': 'Flow is optimal experience: total absorption, loss of self-consciousness, intrinsic reward. Occurs when challenge matches skill at a high level. Neurologically: norepinephrine + dopamine + anandamide + serotonin — attention, reward, and lateral prefrontal cortex partially deactivates (transient hypofrontality).',
        'application': 'Design activities with clear goals, immediate feedback, and challenge level just above current skill. Remove external distractions. Provide autonomy and control. Games naturally produce flow conditions — gamification can leverage this. Flow is productive for practicing existing skills; difficult problems requiring analytical thought may not reach flow.',
        'example': 'Students in flow-conducive classroom conditions showed higher intrinsic motivation and better retention than in anxiety-inducing or boring conditions (Shernoff et al. 2003).',
    },
    {
        'name': 'Self-Determination Theory (Intrinsic Motivation)',
        'evidence': 'Deci & Ryan (1985, 2000). Extensively replicated across cultures.',
        'mechanism': 'Three innate psychological needs: Autonomy (agency, choice), Competence (efficacy, mastery), Relatedness (belonging, connection). When these are met, intrinsic motivation flourishes. When thwarted, extrinsic motivation (rewards/punishment) provides short-term compliance but undermines long-term engagement.',
        'application': 'Autonomy: offer meaningful choices (topic, format, path). Competence: design for achievable challenge, celebrate growth. Relatedness: build community, collaborative learning, real audience for work. Minimize controlling language ("you must", "you have to"). Provide rationales for required tasks.',
        'example': 'Students given choice in how to demonstrate learning showed 40% higher engagement and better retention than those with no choice (Patall et al. 2008).',
    },
    {
        'name': 'Chunking and Working Memory',
        'evidence': 'Miller (1956) "magical number 7±2"; modern revision to ~4 chunks (Cowan 2001).',
        'mechanism': 'Working memory can hold ~4 chunks simultaneously. A "chunk" is a unit of meaningful organized information — a chess grandmaster sees a board position as 5 chunks; a novice sees 32 individual pieces. Expertise = ability to form larger, richer chunks, allowing more to fit in working memory.',
        'application': 'Present information in organized groups of 3-4. Use advance organizers to prime chunking. Develop vocabulary and conceptual frameworks early — these become the "chunks" that enable complex reasoning later. Don\'t require learners to hold too much new information in mind simultaneously.',
        'example': 'Phone numbers are memorized as 3 chunks (555) (867) (5309) — not 10 individual digits. Meaningfully chunked information is 3× more memorable.',
    },
    {
        'name': 'Prior Knowledge Activation',
        'evidence': 'Ausubel (1968) assimilation theory; Dochy et al. (1999) meta-analysis.',
        'mechanism': 'New information is encoded by connecting it to existing knowledge structures (schemas). Without a schema to attach to, information is encoded as isolated facts — easily forgotten. Activating relevant prior knowledge creates "hooks" for new learning. Incorrect prior knowledge (misconceptions) actively blocks correct understanding.',
        'application': 'Start every lesson with activities that activate relevant prior knowledge: KWL charts, warm-up questions, pre-tests. Explicitly bridge from familiar to unfamiliar ("This is like X, but with the key difference Y"). Address common misconceptions directly — ignoring them means they persist.',
        'example': 'Students given a brief overview of a passage\'s topic before reading recalled 2× more information than students who read without an advance organizer (Mayer 1979).',
    },
    {
        'name': 'Narrative and Story Structure',
        'evidence': 'Schank & Abelson (1977) scripts; Willingham (2009) cognitive science of education.',
        'mechanism': 'The brain is fundamentally a story-processing machine. Narrative structure (character, conflict, resolution) matches the brain\'s causal-temporal reasoning systems. Information embedded in story is remembered far better than equivalent information in list form. Emotional engagement during encoding enhances consolidation.',
        'application': 'Structure content as stories where possible. "The scientist who discovered X was trying to solve Y when she noticed Z." Use historical narratives to teach scientific concepts. Case-based learning: present a real problem/patient/situation, then teach the concepts needed to resolve it.',
        'example': 'Information presented in story form was recalled with 50-70% accuracy vs. 10-15% for equivalent information in expository form (Graesser et al. 1994).',
    },
    {
        'name': 'Multimedia Learning Principles (Mayer)',
        'evidence': 'Mayer (2001, 2009). 15+ coherence principles for multimedia instruction.',
        'mechanism': 'Humans process visual and verbal information in separate cognitive channels, each with limited capacity. Effective multimedia aligns these channels and respects their limits. Key principle: words and pictures together > either alone — when the pictures are explanatory (not decorative).',
        'application': 'Key Mayer principles: (1) Coherence: cut irrelevant material. (2) Signaling: highlight key points. (3) Contiguity: put words near related pictures. (4) Segmenting: let learner control pace. (5) Pre-training: teach names/characteristics before complex animation. (6) Modality: audio + visuals > text + visuals (dual-channel). Avoid redundancy: narrating exactly what is on the slide forces two channels to process identical info.',
        'example': 'Students who received narration + animation scored 50% better than text + animation students on transfer tests, due to freed visual channel capacity.',
    },
    {
        'name': 'Priming and Pre-Questions',
        'evidence': 'Rothkopf (1966) mathemagenic behaviors; Pressley et al. (1992). Consistent support.',
        'mechanism': 'Questions presented BEFORE content (advance questions) prime the brain to selectively attend to relevant information. They activate schema, orient attention, and signal what matters. Information relevant to pre-questions is processed more deeply.',
        'application': 'Open every lesson with 2-3 focus questions that students should be able to answer by the end. Post these questions visibly throughout. Have students predict/hypothesize before revealing information. Even wrong predictions enhance encoding of the correction.',
        'example': 'Students given adjunct questions before reading a passage recalled 30% more targeted content than students who only read the passage.',
    },
    {
        'name': 'Variation Theory',
        'evidence': 'Marton & Booth (1997); extensive application in mathematics education.',
        'mechanism': 'Learning requires discerning what varies and what is invariant. Understanding a concept means understanding what can change (parameters) while the structure remains the same, and what cannot change without destroying the concept. Contrast between examples reveals the critical features.',
        'application': 'When teaching a concept, systematically vary one feature at a time while holding others constant. Show non-examples alongside examples — contrast reveals what is essential. In math: show the same problem with different surface features to help students see the underlying structure.',
        'example': 'Students who studied carefully varied examples of fractions (changing numerator, then denominator, then both) understood the concept more deeply than those who saw random examples.',
    },
    {
        'name': 'Formative Assessment (Assessment for Learning)',
        'evidence': 'Black & Wiliam (1998) meta-analysis: effect sizes 0.4-0.7. One of the highest-yield practices.',
        'mechanism': 'Formative assessment provides real-time information about learning gaps to BOTH teacher and student, enabling instruction to adapt. Summative assessment evaluates after the fact; formative assessment changes learning while it is happening.',
        'application': 'Use exit tickets, think-pair-share, hinge questions, mini-whiteboards, digital polls. Goal: identify who is confused about what, so instruction can respond. NOT graded — should feel safe. Design "hinge questions" that perfectly discriminate between students who understand and students with specific misconceptions.',
        'example': 'Classes using systematic formative assessment showed 0.5 SD improvement in achievement — equivalent to 2 grade levels in a single year (Black & Wiliam 1998).',
    },
    {
        'name': 'Productive Failure',
        'evidence': 'Kapur (2010, 2016). Counterintuitive findings from Singapore research.',
        'mechanism': 'Letting students struggle with a problem before teaching the solution (Preparation for Future Learning) activates prior knowledge, generates hypotheses, and surfaces deep structure of the problem. The subsequent instruction then finds a prepared, engaged mind with context for the answer.',
        'application': 'Present a challenging problem BEFORE teaching the relevant concept. Let students attempt and fail. Then teach. Students will better understand and remember the taught concept because they appreciate the problem it solves. Contrast with direct instruction first — less memorable without the productive struggle.',
        'example': 'Students who failed to solve problems before instruction outperformed students who received instruction before problems on conceptual transfer tests (Kapur 2010).',
    },
    {
        'name': 'Spaced vs. Massed Repetition of Examples',
        'evidence': 'Kornell & Bjork (2008). Robust in perceptual and conceptual learning.',
        'mechanism': 'When learning to categorize (e.g., art styles, species, math problem types), seeing examples of different categories INTERLEAVED is better than seeing all examples of one category grouped. The juxtaposition helps learners extract what distinguishes each category.',
        'application': 'When teaching classification (diagnosis, grammar patterns, math problem types): interleave examples from different categories rather than blocking. Add non-examples to sharpen category boundaries.',
        'example': 'Art students who studied interleaved paintings by different artists later classified new paintings more accurately than those who studied each artist\'s works in a block.',
    },
]


# ---------------------------------------------------------------------------
# Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001)
# ---------------------------------------------------------------------------

BLOOMS_TAXONOMY = """
=== BLOOM'S REVISED TAXONOMY OF EDUCATIONAL OBJECTIVES ===

Bloom's Revised Taxonomy (Anderson & Krathwohl, 2001) organizes cognitive skills into six
hierarchical levels. Lower levels are prerequisites for higher levels. Effective curriculum
design deliberately targets higher levels — most traditional education over-emphasizes
Remembering and Understanding at the expense of deeper thinking.

LEVEL 1: REMEMBER (Recall)
Retrieving relevant knowledge from long-term memory.
Action verbs: define, recall, list, identify, name, recognize, state, describe, match, repeat
Example assessment: "List the five phases of mitosis."
Limitation: Students can remember facts without understanding them. Pure recall is the
lowest form of learning — necessary but insufficient.

LEVEL 2: UNDERSTAND (Comprehension)
Constructing meaning from instructional messages.
Action verbs: explain, paraphrase, classify, compare, summarize, interpret, infer, give examples
Example assessment: "Explain in your own words why the sky is blue."
Key indicator: Can the student explain it differently than the textbook? If only exact
textbook words work, it's still Level 1.

LEVEL 3: APPLY (Application)
Carrying out or using a procedure in a given situation.
Action verbs: use, solve, demonstrate, calculate, execute, implement, perform, construct
Example assessment: "Calculate the pH of a 0.01 M HCl solution."
Key: Application requires recognizing WHICH procedure to use (not just how to use it).

LEVEL 4: ANALYZE (Analysis)
Breaking material into parts and determining how they relate.
Action verbs: differentiate, organize, compare, contrast, distinguish, examine, deconstruct, attribute
Example assessment: "Compare the mechanisms of action of aspirin and ibuprofen. What are the
structural differences that lead to their different selectivity profiles?"
Key: Students must see the relationships BETWEEN parts, not just identify the parts.

LEVEL 5: EVALUATE (Evaluation)
Making judgments based on criteria and standards.
Action verbs: judge, critique, defend, assess, prioritize, evaluate, argue, debate, justify
Example assessment: "Evaluate the experimental design of this study. What are its limitations
and how might they affect the conclusions?"
Key: Evaluation requires applying criteria — not just opinion. "I think X" is not evaluation;
"X meets criteria A, B, C but fails D because..." is evaluation.

LEVEL 6: CREATE (Synthesis)
Putting elements together to form a coherent new whole; reorganizing into a new pattern.
Action verbs: design, construct, develop, formulate, compose, plan, produce, propose, invent
Example assessment: "Design an experiment to test whether caffeine improves working memory.
Include your hypothesis, procedure, controls, and analysis plan."
Key: The product is novel to the student. It requires integrating across multiple domains.

DESIGNING WITH BLOOM'S:
- Most courses spend 80% of time at Levels 1-2. Aim for 50%+ at Levels 3-6.
- Use Bloom's to write learning objectives: "Students will be able to [VERB] [CONTENT]"
  where the verb signals the cognitive level.
- Align assessments to objectives: if you teach at Level 3 (Apply) but test at Level 1
  (Recall), you're measuring memory not learning.
- Build upward: students cannot Analyze before they Understand; cannot Evaluate before they Apply.
- Higher levels consolidate lower levels — creating forces you to also remember, understand,
  apply, and analyze.

KNOWLEDGE DIMENSIONS (2nd axis of Bloom's Revised Taxonomy):
  Factual knowledge: basic terminology, specific facts
  Conceptual knowledge: classifications, principles, theories, structures
  Procedural knowledge: skills, algorithms, techniques, methods
  Metacognitive knowledge: self-knowledge, strategies, cognitive tasks
Each Bloom's level can operate on any knowledge type, creating a 4×6 grid for lesson design.
"""


# ---------------------------------------------------------------------------
# Curriculum Design Frameworks
# ---------------------------------------------------------------------------

CURRICULUM_FRAMEWORKS = """
=== CURRICULUM DESIGN FRAMEWORKS ===

--- FRAMEWORK 1: UNDERSTANDING BY DESIGN (UbD / Backward Design) ---
Wiggins & McTighe (1998, 2005). Used widely in K-12 and university curriculum.

CORE PRINCIPLE: Start with desired OUTCOMES, then design ASSESSMENTS, then design INSTRUCTION.
Most teachers do the opposite (teach content, then figure out what to test).

THREE STAGES:
Stage 1 — Identify Desired Results
  • What enduring understandings should students have?
  • What essential questions will guide inquiry?
  • What knowledge and skills must students acquire?
  Essential question example: "How do numbers help us describe the world?" (mathematics)

Stage 2 — Determine Acceptable Evidence
  • What will students DO to demonstrate understanding?
  • Performance tasks: authentic, complex, require transfer
  • Other evidence: quizzes, tests, observations, self-assessments
  GRASPS framework for tasks: Goal, Role, Audience, Situation, Product, Standards

Stage 3 — Plan Learning Experiences and Instruction
  WHERE are we going? (clarity of goals and expectations)
  HOOK and HOLD attention (engaging introduction)
  EQUIP students with knowledge, skills, experience
  RETHINK and REVISE (opportunities to reflect and improve)
  EVALUATE student work and self-assess
  TAILOR to individual needs
  ORGANIZE for maximum engagement and effectiveness

KEY PRINCIPLE — "Twin Sins" of curriculum design:
  Activity-focused: "Fun activity! (But what understanding does it produce?)"
  Coverage-focused: "We must get through Chapter 12 by Friday (whether understood or not)"
  Both produce forgettable education without transferable understanding.

---

--- FRAMEWORK 2: ADDIE (Instructional Systems Design) ---
Originally developed for US military training. Standard in corporate/professional learning.

ANALYSIS: Who are the learners? What do they already know? What are the learning gaps?
  What constraints exist (time, tech, budget)? What is the desired performance?
DESIGN: Write measurable learning objectives. Determine assessment strategy. Choose
  instructional strategy. Develop storyboard/blueprint. Sequence content.
DEVELOPMENT: Create content, media, exercises. Build into delivery platform.
IMPLEMENTATION: Deliver instruction. Manage learner experience. Support facilitators.
EVALUATION:
  Kirkpatrick Level 1: Did learners like it? (Satisfaction)
  Kirkpatrick Level 2: Did learners learn? (Knowledge/skill gain)
  Kirkpatrick Level 3: Are learners applying it on the job? (Behavior change)
  Kirkpatrick Level 4: Did it impact organizational outcomes? (Results)

---

--- FRAMEWORK 3: GAGNÉ'S NINE EVENTS OF INSTRUCTION ---
Robert Gagné (1965). "Conditions of Learning." Highly structured, evidence-based sequence.

EVENT 1: GAIN ATTENTION — Start with novelty, problem, surprising fact, conflict.
  Neurological basis: orienting response activates ascending arousal system.

EVENT 2: INFORM OBJECTIVES — Tell learners what they'll be able to DO.
  "By the end, you'll be able to explain why vaccines sometimes fail and design a better delivery."

EVENT 3: STIMULATE RECALL OF PRIOR LEARNING — Connect to what they already know.
  "Last week we covered how the immune system responds to pathogens. Today builds directly on that."

EVENT 4: PRESENT CONTENT — Clear, logically sequenced, chunked, with visuals.
  Respect cognitive load. Present in small segments.

EVENT 5: PROVIDE LEARNING GUIDANCE — Worked examples, analogies, mnemonics, elaboration.
  Show HOW to encode and organize, not just present the information.

EVENT 6: ELICIT PERFORMANCE — Have learners practice the target skill immediately.
  Retrieval practice while learning. Reduces forgetting and reveals misconceptions.

EVENT 7: PROVIDE FEEDBACK — Specific, timely, actionable. Address WHAT and WHY.
  Not just "correct/incorrect" but "you did X, the issue is Y, try Z next time."

EVENT 8: ASSESS PERFORMANCE — Test transfer, not just recall.
  Match assessment to objective level (Bloom's).

EVENT 9: ENHANCE RETENTION AND TRANSFER — Spaced review. Varied practice contexts.
  "How does this apply in your daily life? Where else does this principle appear?"

---

--- FRAMEWORK 4: PROJECT-BASED LEARNING (PBL) ---
Buck Institute for Education; Larmer & Mergendoller (2015). Best for complex, transferable skills.

CORE: Students learn by working for an extended time on a challenging real-world problem
or question, culminating in a public product or presentation.

GOLD STANDARD PBL ELEMENTS:
1. Challenging problem or question — authentic, complex, requires sustained inquiry
2. Sustained inquiry — research, iterative process, not one-shot
3. Authenticity — real-world context, real tools, real audience
4. Student voice and choice — meaningful decisions, not just compliance
5. Reflection — metacognitive checkpoints throughout
6. Critique and revision — quality products require multiple drafts
7. Public product — presentation to real audience, not just teacher

DRIVING QUESTION structure: "How might we [solve problem] for [specific audience] so that [impact]?"
Example: "How might we design a water filtration system for a rural community that has no electricity?"

WHEN TO USE: Complex interdisciplinary skills; when intrinsic motivation matters;
when transfer is the goal; for students with existing foundational knowledge.

WHEN NOT TO USE: Initial acquisition of prerequisite skills (direct instruction is more efficient);
time-constrained curricula; when prerequisites are missing.

---

--- FRAMEWORK 5: UNIVERSAL DESIGN FOR LEARNING (UDL) ---
CAST (2011); Based on neuroscience of learning variation. Required in many educational contexts.

CORE PRINCIPLE: All learners are variable. Design for the range from the start — don't
retrofit for "special needs" after designing for a mythical "average learner."

THREE PRINCIPLES (UDL Guidelines 2.0):

PRINCIPLE 1: MULTIPLE MEANS OF REPRESENTATION (the "what" of learning)
  • Offer information in multiple formats (text, audio, video, tactile)
  • Activate background knowledge
  • Support language/vocabulary/symbols comprehension
  Goal: provide learners with flexible options for acquiring and comprehending information

PRINCIPLE 2: MULTIPLE MEANS OF ACTION AND EXPRESSION (the "how" of learning)
  • Vary methods for response and navigation
  • Multiple tools for construction and composition
  • Support planning and strategy development
  Goal: provide flexible options for demonstrating what is known

PRINCIPLE 3: MULTIPLE MEANS OF ENGAGEMENT (the "why" of learning)
  • Offer choices to optimize individual challenge and relevance
  • Optimize autonomy and minimize threats/distractions
  • Foster collaboration and community
  • Self-regulation and metacognition support
  Goal: develop purposeful, motivated learners

PRACTICAL IMPLEMENTATION:
  Represent: offer text + audio + video. Use multiple examples. Pre-teach vocabulary.
  Express: allow essays OR presentations OR models OR videos.
  Engage: offer relevant choices. Make stakes clear but non-threatening. Co-design with learners.
"""


# ---------------------------------------------------------------------------
# Advanced Curriculum Design Principles for AI-Generated Courses
# ---------------------------------------------------------------------------

AI_CURRICULUM_DESIGN = """
=== PRINCIPLES FOR DESIGNING ADVANCED, MEMORABLE, EFFORTFUL COURSES ===

When designing a course on any subject, apply these principles to create an experience
that is engaging, deeply understood, long-remembered, and practically useful.

STRUCTURE:
1. Start with why — lead with a compelling real-world problem the course will solve.
2. Build backward from desired performance — what will students be able to DO?
3. Spiral curriculum — revisit core concepts at increasing depth (Bruner, 1960).
4. Progress from concrete → abstract → transfer → creation.
5. Align every activity, assessment, and objective (Biggs constructive alignment).

ENGAGEMENT:
6. Open every module with an intriguing question or surprising phenomenon (curiosity gap).
7. Use narrative arcs — the history of a discovery, the story behind a concept.
8. Deliver regular "minimum surprise" moments that confirm growing competence.
9. Design for flow: challenge should slightly exceed current skill at every point.
10. Create communities of practice — students learn from each other's expertise.

ENCODING:
11. Multiple modalities per concept: verbal explanation + visual + worked example + student practice.
12. Connect every abstract concept to a concrete real-world referent.
13. Explicitly state relevance: "You need this because..."
14. Create elaborative questions that force connection to existing knowledge.
15. Teach metacognitive monitoring: "Do you understand this well enough to explain it simply?"

RETENTION:
16. Implement mandatory spaced repetition — no optional review, required revisitation.
17. End every session with retrieval (no notes): "Write everything you remember."
18. Use low-stakes quizzing throughout — never wait for high-stakes exams to assess.
19. Interleave topics within sessions, not just across sessions.
20. Build cumulative assessments — later tests always include earlier material.

TRANSFER:
21. Present each concept in at least 3 different contexts.
22. Ask "Where else does this apply?" after every major principle.
23. Use case studies from different domains to teach the same underlying principle.
24. Teach the domain's "moves" and "lenses" — how experts think, not just what they know.
25. Require students to generate novel applications, not just reproduce known ones.

MOTIVATION:
26. Communicate mastery progress visibly and frequently.
27. Provide autonomy in at least one significant dimension (topic, format, pace, partner).
28. Connect course material to students' existing goals and identities.
29. Celebrate struggle and error as the mechanism of learning, not its failure.
30. Build in regular milestones with authentic audiences for student work.

ASSESSMENT:
31. Every assessment should TEACH as well as evaluate (test as learning event).
32. Include self-assessment and peer assessment alongside expert assessment.
33. Design performance tasks that require integration across the entire course.
34. Provide rubrics that describe LEVELS OF MASTERY, not just pass/fail.
35. Use portfolio assessment to capture growth over time.
"""


# ---------------------------------------------------------------------------
# Wikipedia topics on learning science and educational psychology
# ---------------------------------------------------------------------------

PEDAGOGY_WIKI_TOPICS = {
    'Learning Science': [
        'Learning', 'Memory consolidation', 'Long-term potentiation', 'Working memory',
        'Cognitive load', 'Spaced repetition', 'Retrieval practice', 'Interleaving',
        'Transfer appropriate processing', 'Encoding specificity principle',
        'Levels of processing', 'Elaborative encoding', 'Dual-coding theory',
        'Forgetting curve', 'Testing effect', 'Desirable difficulty',
        'Generation effect', 'Elaborative interrogation',
    ],
    'Cognitive Psychology': [
        'Cognitive psychology', 'Attention', 'Executive functions', 'Short-term memory',
        'Long-term memory', 'Episodic memory', 'Semantic memory', 'Procedural memory',
        'Metacognition', 'Cognitive bias', 'Chunking (psychology)', 'Mental model',
        'Schema (psychology)', 'Priming (psychology)', 'Implicit memory',
        'Working memory model', 'Miller\'s law',
    ],
    'Educational Psychology': [
        'Educational psychology', 'Zone of proximal development', 'Scaffolding (education)',
        'Mastery learning', 'Constructivism (philosophy of education)',
        'Social learning theory', 'Self-determination theory', 'Growth mindset',
        'Fixed mindset', 'Motivation in education', 'Attribution theory',
        'Expectancy-value theory', 'Self-efficacy', 'Learned helplessness',
        'Bloom\'s taxonomy', 'Universal Design for Learning',
    ],
    'Neuroscience of Learning': [
        'Neuroplasticity', 'Hebbian theory', 'Hippocampus', 'Prefrontal cortex',
        'Amygdala', 'Dopamine', 'Memory and sleep', 'Brain-derived neurotrophic factor',
        'Neurogenesis', 'Synaptic pruning', 'Myelination', 'Default mode network',
    ],
    'Curriculum and Instruction': [
        'Curriculum', 'Instructional design', 'Learning objectives', 'Backward design',
        'Direct instruction', 'Inquiry-based learning', 'Problem-based learning',
        'Project-based learning', 'Flipped classroom', 'Differentiated instruction',
        'Formative assessment', 'Summative assessment', 'Authentic assessment',
        'Feedback (education)',
    ],
    'Learning Theory': [
        'Behaviorism', 'Cognitivism (psychology)', 'Constructivism (learning theory)',
        'Connectivism', 'Experiential learning', 'Kolb\'s experiential learning',
        'Situated learning', 'Communities of practice', 'Sociocultural theory',
        'Multiple intelligences', 'Learning styles', 'Andragogy',
    ],
}

PEDAGOGY_PUBMED_QUERIES = [
    'spaced repetition learning memory retention',
    'retrieval practice testing effect long-term retention',
    'interleaving blocked practice mathematics learning',
    'cognitive load theory instructional design',
    'worked examples novice learner STEM',
    'metacognition academic achievement intervention',
    'growth mindset neuroplasticity academic performance',
    'sleep memory consolidation learning',
    'formative assessment student achievement classroom',
    'peer instruction active learning physics',
    'problem-based learning medical education outcomes',
    'project-based learning student motivation engagement',
    'elaborative interrogation self-explanation learning',
    'dual coding visual verbal learning',
    'desirable difficulties long-term retention',
    'spaced practice interleaving mathematics',
    'feedback timing quality learning outcomes',
    'mastery learning achievement outcomes meta-analysis',
    'curiosity learning hippocampus memory',
    'emotion motivation academic achievement neuroscience',
    'exercise cognitive function BDNF learning',
    'zone of proximal development scaffolding',
    'transfer of learning far transfer instruction',
    'multimedia learning principles Mayer',
    'generative learning strategies summarizing mapping',
    'deliberate practice expertise skill acquisition',
    'prior knowledge activation schema learning',
    'narrative learning story comprehension memory',
    'Universal Design for Learning disability inclusion',
    'self-determination theory autonomy competence relatedness',
    'productive failure preparation for future learning',
    'variation theory mathematics education concept learning',
    'flow state learning intrinsic motivation',
    'social learning peer teaching learning outcomes',
    'chunking working memory expertise',
    'mnemonics memory technique method of loci',
    'advance organizer prior knowledge comprehension',
    'flipped classroom active learning outcomes',
    'inquiry-based learning science education',
    'Bloom taxonomy higher order thinking',
]


# ---------------------------------------------------------------------------
# Helpers (reused from other corpus builders)
# ---------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.mount('http://', HTTPAdapter(max_retries=retry))
    return s


def _train(node: str, text: str, session: requests.Session) -> bool:
    try:
        r = session.post(f'http://{node}/train', json={'text': text}, timeout=30)
        return r.status_code == 200
    except Exception as e:
        print(f'  [WARN] train error: {e}')
        return False


def _wiki_content(title: str, session: requests.Session) -> str:
    url = 'https://en.wikipedia.org/w/api.php'
    params = {
        'action': 'query', 'prop': 'extracts', 'exintro': False,
        'explaintext': True, 'titles': title, 'format': 'json',
    }
    try:
        r = session.get(url, params=params, timeout=20)
        if r.status_code == 200:
            pages = r.json().get('query', {}).get('pages', {})
            for page in pages.values():
                return page.get('extract', '')[:6000]
    except Exception:
        pass
    return ''


def _search_pubmed(query: str, max_results: int, session: requests.Session,
                   api_key: str = '') -> list:
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
    params = {
        'db': 'pubmed', 'term': query, 'retmax': max_results,
        'retmode': 'json', 'sort': 'relevance',
    }
    if api_key:
        params['api_key'] = api_key
    try:
        r = session.get(base, params=params, timeout=20)
        if r.status_code == 200:
            return r.json().get('esearchresult', {}).get('idlist', [])
    except Exception:
        pass
    return []


def _fetch_abstracts(pmids: list, session: requests.Session, api_key: str = '') -> str:
    if not pmids:
        return ''
    base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
    params = {
        'db': 'pubmed', 'id': ','.join(pmids),
        'retmode': 'text', 'rettype': 'abstract',
    }
    if api_key:
        params['api_key'] = api_key
    try:
        r = session.get(base, params=params, timeout=30)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return ''


def _load_checkpoint(path: Path) -> set:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def _save_checkpoint(path: Path, done: set):
    path.write_text(json.dumps(sorted(done)))


def _format_principle(p: dict) -> str:
    return f"""EVIDENCE-BASED LEARNING PRINCIPLE: {p['name']}

Evidence Level: {p['evidence']}

MECHANISM — WHY IT WORKS:
{p['mechanism']}

HOW TO APPLY:
{p['application']}

EXAMPLE:
{p['example']}
"""


# ---------------------------------------------------------------------------
# Stage 35 runner
# ---------------------------------------------------------------------------

def train_stage35(node: str, data_dir: str, args):
    out = Path(data_dir) / 'training' / 'stage35'
    out.mkdir(parents=True, exist_ok=True)
    ckpt_principles = out / 'checkpoint_principles.json'
    ckpt_wiki = out / 'checkpoint_wiki.json'
    ckpt_pubmed = out / 'checkpoint_pubmed.json'
    session = _session()
    api_key = getattr(args, 'ncbi_api_key', '')
    rate = 0.12 if api_key else 0.38

    trained_principles = _load_checkpoint(ckpt_principles)
    trained_wiki = _load_checkpoint(ckpt_wiki)
    trained_pubmed = _load_checkpoint(ckpt_pubmed)

    # --- Part 1: Learning principles ---
    print('[Stage 35] Training evidence-based learning principles...')
    for p in LEARNING_PRINCIPLES:
        key = p['name']
        if key in trained_principles:
            continue
        text = _format_principle(p)
        if _train(node, text, session):
            trained_principles.add(key)
            print(f"  ✓ Principle: {p['name']}")
        time.sleep(0.05)
    _save_checkpoint(ckpt_principles, trained_principles)

    # --- Part 2: Bloom's taxonomy ---
    bloom_key = '__blooms_taxonomy__'
    if bloom_key not in trained_wiki:
        print("[Stage 35] Training Bloom's Revised Taxonomy...")
        if _train(node, BLOOMS_TAXONOMY, session):
            trained_wiki.add(bloom_key)
            print("  ✓ Bloom's Taxonomy trained.")
        _save_checkpoint(ckpt_wiki, trained_wiki)

    # --- Part 3: Curriculum frameworks ---
    fw_key = '__curriculum_frameworks__'
    if fw_key not in trained_wiki:
        print('[Stage 35] Training curriculum design frameworks...')
        if _train(node, CURRICULUM_FRAMEWORKS, session):
            trained_wiki.add(fw_key)
            print('  ✓ Curriculum frameworks trained.')
        _save_checkpoint(ckpt_wiki, trained_wiki)

    # --- Part 4: AI curriculum design principles ---
    ai_key = '__ai_curriculum_design__'
    if ai_key not in trained_wiki:
        print('[Stage 35] Training advanced curriculum design principles...')
        if _train(node, AI_CURRICULUM_DESIGN, session):
            trained_wiki.add(ai_key)
            print('  ✓ AI curriculum design guide trained.')
        _save_checkpoint(ckpt_wiki, trained_wiki)

    # --- Part 5: Wikipedia learning science articles ---
    print('[Stage 35] Training Wikipedia learning science articles...')
    total = sum(len(v) for v in PEDAGOGY_WIKI_TOPICS.values())
    done = 0
    for branch, topics in PEDAGOGY_WIKI_TOPICS.items():
        for topic in topics:
            key = f'wiki:{topic}'
            if key in trained_wiki:
                done += 1
                continue
            text = _wiki_content(topic, session)
            if text:
                full = f'LEARNING SCIENCE — {branch.upper()}\nTopic: {topic}\n\n{text}'
                if _train(node, full, session):
                    trained_wiki.add(key)
                    done += 1
                    print(f'  [{done}/{total}] ✓ {branch}: {topic}')
            time.sleep(0.15)
        _save_checkpoint(ckpt_wiki, trained_wiki)

    # --- Part 6: PubMed peer-reviewed learning research ---
    print('[Stage 35] Fetching PubMed research on learning science...')
    max_q = getattr(args, 'max_per_query', 30)
    for query in PEDAGOGY_PUBMED_QUERIES:
        key = f'pubmed:{query}'
        if key in trained_pubmed:
            continue
        pmids = _search_pubmed(
            f'{query} AND (Review[ptyp] OR "Systematic Review"[ptyp] OR "Meta-Analysis"[ptyp])',
            max_q, session, api_key
        )
        if pmids:
            for i in range(0, len(pmids), 20):
                batch = pmids[i:i+20]
                raw = _fetch_abstracts(batch, session, api_key)
                if raw:
                    text = (
                        'PEER-REVIEWED RESEARCH: Learning Science & Educational Psychology\n'
                        f'Query: {query}\n\n{raw[:12000]}'
                    )
                    _train(node, text, session)
                time.sleep(rate)
        trained_pubmed.add(key)
        _save_checkpoint(ckpt_pubmed, trained_pubmed)
        print(f'  ✓ PubMed: {query} ({len(pmids)} articles)')
        time.sleep(rate)

    print(f'[Stage 35] Complete. Principles: {len(trained_principles)}, '
          f'Articles: {done}, PubMed queries: {len(trained_pubmed)}')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description='Build pedagogy & curriculum design corpus (Stage 35)')
    p.add_argument('--stages', default='35')
    p.add_argument('--node', default='localhost:8090')
    p.add_argument('--data-dir', default='D:/w1z4rdv1510n-data')
    p.add_argument('--max-per-query', type=int, default=30)
    p.add_argument('--ncbi-api-key', default='')
    return p.parse_args()


def main():
    args = _parse_args()
    stages = [int(s.strip()) for s in args.stages.split(',')]
    if 35 in stages:
        train_stage35(args.node, args.data_dir, args)


if __name__ == '__main__':
    main()
