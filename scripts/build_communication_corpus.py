#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_communication_corpus.py -- Stage 42: Long-Form, Run-On & Grammar Training

Trains the node to:
  1. Answer long, detailed questions with appropriately comprehensive responses
  2. Give SHORT direct answers to long/winding questions when the answer is simple
  3. Correct run-on sentences with proper punctuation when asked
  4. Fix grammar, spelling, and punctuation errors on request
  5. Understand run-on and poorly-structured input and respond to the intent
  6. Correct and rewrite text when explicitly asked

SCOPE PRINCIPLE (prevents over-training):
  - Correction sections use explicit scope markers ("fix the grammar", "correct the
    punctuation", "rewrite this sentence", "clean up this text", etc.)
  - Understanding sections have run-on/poorly-phrased questions with NO correction
    marker -- the node answers the question, not the structure
  - The node learns: correction marker present -> fix the text;
                     no marker -> understand and answer the question

Usage:
  python scripts/build_communication_corpus.py --node localhost:8090
  python scripts/build_communication_corpus.py --repeats 100
  python scripts/build_communication_corpus.py --stages 42 --repeats 50
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
# SECTION A: Long questions with long, comprehensive answers
# ---------------------------------------------------------------------------
LONG_Q_LONG_A_PAIRS: list[tuple[str, str]] = [
    ("Can you explain in detail the entire process of photosynthesis, including both the light-dependent and light-independent reactions, what molecules are involved, where in the plant cell each reaction takes place, and what the final products are?",
     "Photosynthesis converts sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in chloroplasts and has two stages. The light-dependent reactions occur in the thylakoid membranes: chlorophyll absorbs sunlight and splits water molecules (photolysis), releasing oxygen as a byproduct and generating ATP and NADPH as energy carriers. These energy molecules then power the light-independent reactions (the Calvin cycle), which take place in the stroma. In the Calvin cycle, carbon dioxide is fixed by the enzyme RuBisCO and combined with the ATP and NADPH from the first stage to build glyceraldehyde-3-phosphate (G3P), a three-carbon sugar. G3P is then used to synthesize glucose and regenerate ribulose bisphosphate (RuBP) to continue the cycle. The net equation is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2. Photosynthesis not only feeds plants but produces nearly all the oxygen in Earth's atmosphere."),
    ("I want to understand machine learning thoroughly -- can you explain what it is, how neural networks work, what training and backpropagation mean, what overfitting is and how to prevent it, and how deep learning differs from classical machine learning?",
     "Machine learning is a branch of AI where systems learn patterns from data rather than being explicitly programmed. Classical ML algorithms like linear regression, decision trees, and SVMs learn a mapping from input features to outputs using statistical methods. Deep learning uses neural networks with many layers. A neural network consists of interconnected nodes (neurons) organized in layers: an input layer, one or more hidden layers, and an output layer. Each connection has a weight. During training, the network makes predictions, compares them to the correct answers using a loss function, and adjusts weights to reduce the error. This adjustment uses backpropagation: the error is propagated backward through the network, computing gradients via the chain rule of calculus, and weights are updated using gradient descent. Overfitting occurs when a model learns the training data too well, including its noise, and performs poorly on new data. Prevention methods include: regularization (L1/L2 penalties), dropout (randomly disabling neurons during training), early stopping (halting training when validation loss increases), and using more training data. Deep learning excels at tasks like image recognition, natural language processing, and speech because its hierarchical layers can learn complex hierarchical features automatically."),
    ("What is the history of the internet from its origins to today, including who invented it, what ARPANET was, how the World Wide Web differs from the internet, what major milestones changed how people use it, and where it stands today?",
     "The internet evolved over several decades. In the late 1960s, the U.S. Department of Defense funded ARPANET, the first packet-switched network connecting universities and research institutions. The key innovation was packet switching: data is broken into packets that travel independently and reassemble at the destination. In 1983, ARPANET adopted TCP/IP, the foundational protocol suite still used today. This is the moment often cited as the birth of the modern internet. The World Wide Web, invented by Tim Berners-Lee in 1989-1991, is a layer built on top of the internet: it is a system of interlinked hypertext documents accessed through browsers using HTTP and URLs. The internet is the infrastructure (cables, routers, protocols); the Web is a service running on it. Major milestones include: the first web browser (Mosaic, 1993), the commercialization of the internet in the mid-1990s, Google's founding (1998), broadband replacing dial-up in the 2000s, social media emergence (Facebook 2004, Twitter 2006, YouTube 2005), the smartphone revolution (iPhone 2007), and cloud computing. Today, approximately 5.5 billion people use the internet. It underlies global commerce, communication, science, and culture, with emerging challenges around privacy, security, misinformation, and AI integration."),
    ("Can you explain the entire water cycle in detail, including evaporation, condensation, precipitation, surface runoff, groundwater, transpiration from plants, and how human activity affects the cycle?",
     "The water cycle describes the continuous movement of water through Earth's systems. Evaporation occurs when solar energy heats surface water in oceans, lakes, and rivers, converting liquid water to water vapor. Plants contribute through transpiration, releasing water vapor through pores in their leaves (combined, these are called evapotranspiration). Water vapor rises into the atmosphere, cools, and condenses around tiny particles (dust, pollen) to form clouds and fog -- this is condensation. When water droplets in clouds combine and grow heavy enough, they fall as precipitation: rain, snow, sleet, or hail depending on temperature. Precipitation that falls on land takes several paths. Surface runoff flows over land into rivers, streams, and lakes, eventually reaching the ocean. Some water infiltrates the soil, becoming soil moisture. Deeper infiltration creates groundwater in aquifers, which can remain stored for thousands of years or discharge into rivers and springs. Snow and ice in glaciers and polar ice caps represent a frozen reservoir that melts slowly over time. Human activities disrupt the cycle significantly: deforestation reduces transpiration and increases runoff; urbanization creates impermeable surfaces that increase runoff and reduce infiltration; irrigation diverts water from natural channels; climate change accelerates evaporation and intensifies precipitation extremes; groundwater extraction depletes aquifers faster than they recharge."),
    ("Explain the theory of evolution by natural selection in full detail -- what Darwin observed, the core mechanism, genetic underpinning, evidence from the fossil record and DNA, and how speciation occurs.",
     "Charles Darwin, observing diverse wildlife during his Beagle voyage (1831-1836), particularly in the Galapagos Islands, noticed that species varied geographically and seemed adapted to local conditions. He and Alfred Russel Wallace independently formulated natural selection. The mechanism has four components: (1) Variation -- individuals in a population differ in traits. (2) Heredity -- traits are passed from parent to offspring. (3) Selection -- individuals with traits better suited to the environment survive and reproduce more successfully (survival of the fittest). (4) Time -- over generations, favorable traits become more common. The genetic underpinning was missing from Darwin's original theory and was supplied by Mendel's genetics and, later, DNA science. Mutations in DNA introduce new variation. Sexual reproduction shuffles existing variation. The modern synthesis unified Darwinian selection with Mendelian genetics. Evidence comes from multiple sources: the fossil record shows gradual changes and transitional forms (e.g., Tiktaalik bridging fish and tetrapods); comparative anatomy reveals homologous structures (the same bones in human hands, whale flippers, bat wings); vestigial organs (the human appendix, whale hip bones) reflect evolutionary history; DNA comparisons show that species sharing recent common ancestors share more DNA; direct observation has documented evolution in real time in bacteria, viruses, and isolated populations. Speciation -- the formation of new species -- occurs when populations become reproductively isolated (by geography, behavior, or biology) and diverge over time until they can no longer interbreed."),
    ("I would like to understand the entire process of how a computer compiles and runs a program -- from typing source code to the CPU executing instructions -- including parsing, compiling, linking, loading, and what happens at the hardware level.",
     "When you write source code and run it, a multi-stage process transforms human-readable text into hardware instructions. First, the compiler's lexer scans the source text and converts it into tokens (keywords, identifiers, operators). The parser then analyzes the token sequence to build an abstract syntax tree (AST) representing the program's structure. Semantic analysis checks that the AST is meaningful (types match, variables are declared, etc.). The compiler then generates an intermediate representation and applies optimization passes to improve efficiency. Finally, it generates machine code -- binary instructions specific to the target CPU architecture. A linker combines the compiled object files with library code, resolving references between modules (e.g., calls to printf) and producing an executable file. When you run the executable, the operating system's loader reads the file, allocates memory, maps code and data into the process's virtual address space, and jumps to the entry point. At the hardware level, the CPU's fetch-decode-execute cycle takes over: it fetches the next instruction from memory (using the program counter register), decodes the binary instruction to understand what operation to perform, and executes it using the ALU (for arithmetic/logic), memory controller (for loads/stores), or other units. Modern CPUs execute multiple instructions simultaneously via pipelining and out-of-order execution, making this process vastly faster than the sequential description implies."),
    ("Can you explain in comprehensive detail what happens in the human body during exercise -- cardiovascular response, muscle physiology, energy systems, hormonal changes, and long-term adaptations from regular training?",
     "During exercise, virtually every system in the body responds and adapts. Cardiovascular system: heart rate increases (mediated by adrenaline and reduced parasympathetic tone) to deliver more oxygen to working muscles. Stroke volume increases. Blood vessels in working muscles dilate while vessels to digestive organs constrict, redirecting blood flow. Cardiac output can increase from 5 L/min at rest to 25+ L/min in intense exercise. Muscle physiology: skeletal muscle fibers contract when motor neurons fire, releasing acetylcholine at the neuromuscular junction. Calcium floods the cell, allowing actin-myosin cross-bridge cycling that generates force. Different fiber types are recruited progressively: slow-twitch (Type I) fibers first (efficient, fatigue-resistant), then fast-twitch (Type II) for high intensity. Energy systems: the phosphocreatine system provides immediate energy for the first 10 seconds; anaerobic glycolysis (breaking down glucose without oxygen) sustains effort for up to 2 minutes, producing lactate; aerobic respiration (oxidative phosphorylation in mitochondria) powers sustained exercise using glucose and fatty acids. Hormonal changes: adrenaline (epinephrine) and noradrenaline surge, increasing heart rate and mobilizing energy. Cortisol rises during prolonged exercise, mobilizing glucose. Growth hormone is released. Long-term adaptations from regular training include: increased heart size and stroke volume, more mitochondria in muscle cells, greater capillary density, improved fat oxidation efficiency, increased VO2 max, stronger tendons and bones, and better lactate threshold."),
    ("Explain the entire process of how vaccines work, from the initial exposure of the immune system to the vaccine through to long-term memory and protection, including the difference between live-attenuated, inactivated, subunit, and mRNA vaccines.",
     "Vaccines exploit the immune system's ability to learn and remember pathogens. When a vaccine is administered, it introduces an antigen -- a molecule the immune system can recognize as foreign -- without causing disease. The innate immune system responds first: macrophages and dendritic cells engulf the antigen, process it, and migrate to lymph nodes, displaying antigen fragments on their surface (antigen presentation). This activates the adaptive immune system. T cells recognize the presented antigens. Helper T cells (CD4+) activate B cells and cytotoxic T cells (CD8+). B cells proliferate and differentiate into plasma cells that produce antibodies -- proteins that bind specifically to the antigen. Cytotoxic T cells are primed to kill infected cells. After the initial response clears, most immune cells die, but memory B cells and memory T cells persist long-term. On re-exposure to the real pathogen, these memory cells respond rapidly -- within hours rather than weeks -- producing large quantities of antibodies and activated T cells before the infection can take hold. Vaccine types differ in what they introduce: live-attenuated vaccines (MMR, chickenpox) use weakened live pathogens -- they produce strong, long-lasting immunity but cannot be given to immunocompromised individuals. Inactivated vaccines (flu shot, polio IPV) use killed pathogens -- safer but may need boosters. Subunit vaccines (hepatitis B, HPV) use only specific proteins from the pathogen. mRNA vaccines (COVID-19) deliver instructions for cells to produce a pathogen protein (e.g., spike protein), which the immune system then learns to recognize -- no live pathogen or DNA involved."),
]

# ---------------------------------------------------------------------------
# SECTION B: Long or winding questions with SHORT direct answers
# ---------------------------------------------------------------------------
LONG_Q_SHORT_A_PAIRS: list[tuple[str, str]] = [
    ("I've been thinking about this for a while and I've heard different things from different people and I'm not sure who to believe but I'm really curious what is the capital city of France because I want to plan a trip there?",
     "Paris."),
    ("So I was having a conversation with my friend the other day and we got into a debate and neither of us could agree on the answer and we looked it up but we got confused because there were different sources saying different things so we decided to just ask you -- what is the boiling point of water at sea level?",
     "100 degrees Celsius (212 degrees Fahrenheit) at sea level."),
    ("I know this is probably a simple question but I always get confused about this and I want to make sure I have it right once and for all -- how many days are in a leap year?",
     "366 days."),
    ("My teacher mentioned this in class and I wrote it down but I can't read my own handwriting now and I really need to know for my homework that's due tomorrow -- what is the chemical formula for water?",
     "H2O."),
    ("I've been trying to learn more about space and the solar system because I find it fascinating and I've been reading a lot of articles online and watching documentaries but I keep seeing different numbers and I just want a clear straight answer -- how many moons does Earth have?",
     "One. Earth has one natural satellite, the Moon."),
    ("My coworker and I got into a really long argument about this today and we both had strong opinions and we searched online but couldn't agree on which result to trust and it's been bothering me all day -- who wrote Romeo and Juliet?",
     "William Shakespeare."),
    ("I know there's a lot of nuance to this topic and people have written entire books about it and there are many different perspectives and I don't want to oversimplify but at a very basic level and in just a few words if possible -- what is the speed of sound in air?",
     "Approximately 343 metres per second (1,235 km/h) at 20 degrees Celsius."),
    ("I was telling someone about this the other day and I realized I wasn't actually 100 percent sure of the answer so even though I've known this fact since school and I'm a little embarrassed to ask I just want to confirm -- what planet is closest to the Sun?",
     "Mercury."),
    ("So there's this concept that I keep hearing about in my software engineering job and everyone seems to assume I know what it means and I've been nodding along but honestly I never fully understood it and I've been meaning to ask for a long time -- what does API stand for?",
     "Application Programming Interface."),
    ("I read an article about this recently and it mentioned a specific number but I can't remember whether it was the value in Celsius or Fahrenheit or some other scale and I need to know the exact value in Celsius -- what is normal human body temperature?",
     "Approximately 37 degrees Celsius (98.6 degrees Fahrenheit), though normal range is 36.1 to 37.2 degrees Celsius."),
    ("My daughter asked me this question at dinner last night and I thought I knew the answer but then I second-guessed myself and her brother said a completely different answer and we ended up spending twenty minutes arguing about it -- how many continents are there on Earth?",
     "Seven: Africa, Antarctica, Asia, Australia (Oceania), Europe, North America, and South America."),
    ("I've seen this term in a lot of job descriptions recently and I always assumed I understood what it meant but when I actually tried to explain it to someone I realized I wasn't sure I could define it clearly -- what does 'machine learning' mean in simple terms?",
     "Machine learning is a method of teaching computers to learn patterns from data so they can make decisions or predictions without being explicitly programmed for each task."),
    ("I know this might sound like a very basic question for someone who works in technology like I do but I keep using these terms interchangeably and my colleague pointed out that they're actually different things and now I want to know the real answer -- what is the difference between the internet and the World Wide Web?",
     "The internet is the global physical network of cables, routers, and servers. The World Wide Web is a service that runs on top of it -- a system of interlinked web pages accessed through browsers using HTTP. The internet can carry email, video calls, and other services besides the Web."),
    ("I'm writing an essay and I need to include a brief mention of this and I want to make absolutely sure I get the date right because my professor is strict about historical accuracy and marked me down last time for getting a date wrong -- in what year did World War II end?",
     "1945. Germany surrendered on May 8 (V-E Day) and Japan surrendered on September 2 (V-J Day)."),
    ("My boss asked me to include this statistic in a presentation I'm preparing and I've seen different figures in different places and I'm not sure which one is most accurate but I need to give some kind of answer -- approximately how many countries are there in the world?",
     "195 countries are recognized by the United Nations, including 193 UN member states and 2 observer states (Vatican City and Palestine)."),
]

# ---------------------------------------------------------------------------
# SECTION C: Run-on sentence correction (explicit scope marker in Q)
# ---------------------------------------------------------------------------
RUNON_CORRECTION_PAIRS: list[tuple[str, str]] = [
    ("Fix the punctuation and break this into proper sentences: I went to the store and I bought milk and I also got some bread and then I went home and made a sandwich it was really good.",
     "I went to the store and bought milk and bread. When I got home, I made a sandwich. It was really good."),
    ("Correct the punctuation: The weather was really nice yesterday so I decided to go for a walk and I saw my neighbor and we talked for a while and then I kept walking to the park and I sat on a bench for a long time.",
     "The weather was really nice yesterday, so I decided to go for a walk. I saw my neighbor and we talked for a while. Then I kept walking to the park and sat on a bench for a long time."),
    ("Rewrite this with proper punctuation and sentence structure: The company released a new product last month and it sold really well and the CEO was very happy about it and they decided to expand production and hire more workers.",
     "The company released a new product last month, and it sold really well. The CEO was very happy about it, so they decided to expand production and hire more workers."),
    ("Fix the run-on sentence: I need to finish this report by Friday but I also have a meeting on Thursday and my computer keeps crashing and I lost some of my work and I'm really stressed about the deadline.",
     "I need to finish this report by Friday, but I also have a meeting on Thursday. My computer keeps crashing and I lost some of my work, so I'm really stressed about the deadline."),
    ("Correct the punctuation and grammar: She studied really hard for the exam and she got a great score and her parents were proud of her and she celebrated with her friends that night and they went out for dinner.",
     "She studied really hard for the exam and got a great score. Her parents were proud of her. That night, she celebrated with her friends by going out for dinner."),
    ("Rewrite with proper sentences: The project was interesting and challenging and we worked on it for three months and there were a lot of obstacles along the way but we figured out solutions to all of them and in the end it turned out really well.",
     "The project was interesting and challenging. We worked on it for three months, encountering many obstacles along the way. However, we found solutions to all of them, and in the end, it turned out really well."),
    ("Fix this run-on: I woke up late this morning and I was rushing to get ready and I couldn't find my keys and I missed the bus and I had to call a cab and I was still fifteen minutes late to work.",
     "I woke up late this morning and rushed to get ready. I couldn't find my keys, missed the bus, and had to call a cab. I was still fifteen minutes late to work."),
    ("Correct the punctuation: The new software update fixed several bugs and it also added some useful features and users are generally happy with the changes but some people complained about the new interface and the company is collecting feedback.",
     "The new software update fixed several bugs and added some useful features. Users are generally happy with the changes, though some have complained about the new interface. The company is collecting feedback."),
    ("Rewrite this properly: I've been learning to cook lately and I started with simple recipes and now I'm trying more complicated ones and last week I made pasta from scratch and it took a long time but it tasted amazing and my family loved it.",
     "I've been learning to cook lately, starting with simple recipes and progressing to more complicated ones. Last week, I made pasta from scratch. It took a long time, but it tasted amazing and my family loved it."),
    ("Fix the run-on and punctuation: The research shows that exercise improves mental health and it also reduces the risk of heart disease and it helps with weight management and people who exercise regularly tend to live longer and they report higher levels of happiness.",
     "Research shows that exercise improves mental health, reduces the risk of heart disease, and helps with weight management. People who exercise regularly tend to live longer and report higher levels of happiness."),
    ("Correct this sentence: I need to call the doctor and make an appointment and I also need to pick up my prescription and I have to do the grocery shopping and the house needs cleaning and I have a meeting at three.",
     "I need to call the doctor, make an appointment, and pick up my prescription. I also have to do the grocery shopping, clean the house, and attend a meeting at three."),
    ("Fix the punctuation and structure: The concert was amazing and the band played for three hours and they did two encores and the crowd was incredible and everyone was singing along and I lost my voice from cheering.",
     "The concert was amazing. The band played for three hours and did two encores. The crowd was incredible -- everyone was singing along, and I lost my voice from cheering."),
    ("Rewrite with proper sentences: The report needs several revisions and the data section is unclear and the conclusion doesn't follow logically from the findings and the formatting is inconsistent and I think we need another week to get it right.",
     "The report needs several revisions. The data section is unclear, the conclusion does not follow logically from the findings, and the formatting is inconsistent. I think we need another week to get it right."),
    ("Correct the run-on: Technology has changed the way we communicate and it has made the world more connected and it has also created new problems like misinformation and privacy concerns and society is still figuring out how to deal with all of these issues.",
     "Technology has changed the way we communicate and made the world more connected. However, it has also created new problems, such as misinformation and privacy concerns. Society is still figuring out how to deal with these issues."),
    ("Fix this: The dog ran out of the yard and across the street and into the neighbor's garden and knocked over their flower pots and the neighbor came outside and was upset but the dog just wagged its tail and the neighbor ended up laughing.",
     "The dog ran out of the yard, across the street, and into the neighbor's garden, knocking over their flower pots. The neighbor came outside upset, but the dog just wagged its tail -- and the neighbor ended up laughing."),
    ("Correct the punctuation: I started a new job last month and it's been a learning curve and my colleagues are very helpful and the work is challenging but interesting and I think I'm going to enjoy it once I settle in.",
     "I started a new job last month, and it's been a learning curve. My colleagues are very helpful, and the work is challenging but interesting. I think I'm going to enjoy it once I settle in."),
    ("Rewrite properly: Climate change is a serious problem and governments need to act now and companies need to change their practices and individuals can also make a difference by reducing their carbon footprint and the time for action is running out.",
     "Climate change is a serious problem. Governments need to act now, and companies need to change their practices. Individuals can also make a difference by reducing their carbon footprint. Time is running out."),
    ("Fix the run-on: She graduated from university last year and she got a job at a tech company and she moved to a new city and she bought her first apartment and it has been a big year for her and she is proud of everything she accomplished.",
     "She graduated from university last year, got a job at a tech company, and moved to a new city. She also bought her first apartment. It has been a big year for her, and she is proud of everything she accomplished."),
]

# ---------------------------------------------------------------------------
# SECTION D: Grammar and punctuation correction (explicit scope marker in Q)
# ---------------------------------------------------------------------------
GRAMMAR_CORRECTION_PAIRS: list[tuple[str, str]] = [
    ("Correct the grammar: Me and my friend went to the store yesterday.",
     "Corrected: 'My friend and I went to the store yesterday.' Rule: use 'I' as the subject, not 'me'. Test: remove 'my friend and' -- 'I went to the store' is correct; 'Me went to the store' is not."),
    ("Fix the grammar: There's a lot of peoples here today.",
     "Corrected: 'There are a lot of people here today.' Corrections: 'peoples' -> 'people' (people is already plural; 'peoples' refers to multiple ethnic groups); 'There's' -> 'There are' (subject-verb agreement: 'a lot of people' is plural)."),
    ("Correct this sentence: She don't know nothing about it.",
     "Corrected: 'She doesn't know anything about it.' Corrections: 'don't' -> 'doesn't' (third-person singular subject requires 's'); double negative 'don't know nothing' -> 'doesn't know anything' (two negatives cancel out in standard English)."),
    ("Fix the grammar: I should of told you sooner.",
     "Corrected: 'I should have told you sooner.' 'Should of' is a common error caused by the contracted pronunciation of 'should have' (should've). The correct modal auxiliary is 'have', not 'of'."),
    ("Correct the grammar in: Between you and I, the meeting was a waste of time.",
     "Corrected: 'Between you and me, the meeting was a waste of time.' 'Between' is a preposition and requires object pronouns: 'me', not 'I'. Rule: after prepositions (between, for, to, with, from), use object pronouns: me, him, her, us, them."),
    ("Fix the grammar: The data shows that our hypothesis were correct.",
     "Corrected: 'The data show that our hypothesis was correct.' 'Data' is the plural of 'datum' and takes a plural verb in formal English: 'data show'. 'Hypothesis' is singular, so 'was' (not 'were') is correct."),
    ("Correct this: Each of the students have their own desk.",
     "Corrected: 'Each of the students has their own desk.' 'Each' is grammatically singular and requires a singular verb: 'has', not 'have'. 'Their' is acceptable here as a gender-neutral singular possessive."),
    ("Fix the grammar: Who did you give the package to?",
     "Formally corrected: 'To whom did you give the package?' In formal English, 'whom' is used as the object of the preposition 'to'. However, 'Who did you give the package to?' is widely accepted in informal and conversational English."),
    ("Correct the grammar: The team have played well all season.",
     "Both versions are acceptable depending on dialect. American English treats collective nouns like 'team' as singular: 'The team has played well.' British English commonly uses the plural: 'The team have played well.' Both are correct in their respective varieties."),
    ("Fix the grammar and punctuation: Its important to remember, that hard work pays off.",
     "Corrected: 'It's important to remember that hard work pays off.' Corrections: 'Its' -> 'It's' (contraction of 'it is' needs apostrophe); remove the comma before 'that' (no comma before a restrictive 'that' clause)."),
    ("Correct the sentence: I wish I was taller.",
     "In formal English: 'I wish I were taller.' The subjunctive mood ('were', not 'was') is used after 'wish' to express hypothetical or contrary-to-fact conditions. In informal speech, 'was' is widely used and broadly accepted."),
    ("Fix the dangling modifier: Running down the street, the rain began to fall.",
     "Corrected: 'Running down the street, I noticed the rain begin to fall.' Or: 'As I ran down the street, the rain began to fall.' The original sentence has a dangling modifier -- 'running down the street' appears to modify 'the rain', which cannot run. The subject doing the running must be stated."),
    ("Correct the grammar: The reason is because I was sick.",
     "Corrected: 'The reason is that I was sick.' 'The reason is because' is redundant -- 'because' means 'for the reason that', so 'the reason is because' doubles the causal meaning. Use 'the reason is that' instead."),
    ("Fix the grammar: Less people attended the event than last year.",
     "Corrected: 'Fewer people attended the event than last year.' Use 'fewer' with countable nouns (people, items, errors). Use 'less' with uncountable quantities (time, water, money)."),
    ("Correct the sentence: Neither the manager nor the employees was available.",
     "Corrected: 'Neither the manager nor the employees were available.' With 'neither...nor' and 'either...or', the verb agrees with the closer subject. 'Employees' is plural and closest to the verb, so use 'were'."),
]

# ---------------------------------------------------------------------------
# SECTION E: Understanding run-on/poorly-phrased input (NO correction marker)
# The node answers the question -- it does NOT comment on structure.
# ---------------------------------------------------------------------------
RUNON_UNDERSTANDING_PAIRS: list[tuple[str, str]] = [
    ("hey so I was wondering and I know this might be a basic question but I've been curious about it for a while what exactly is the difference between a virus and a bacteria",
     "Bacteria are single-celled living organisms that can reproduce independently. Viruses are non-living particles that need a host cell to replicate. Bacteria can often be treated with antibiotics; viruses cannot."),
    ("okay so I've been thinking about this and I want to understand how does the brain actually store memories like is it in one place or is it spread out and how do you retrieve a memory",
     "Memories are distributed across multiple brain regions, not stored in one place. The hippocampus is critical for forming new memories. Long-term memories are encoded through changes in synaptic connections between neurons -- stronger connections make memories easier to retrieve. Retrieval is the process of reactivating those same neural patterns."),
    ("I've been learning Python and I keep hearing about this thing called object-oriented programming and I know it's important but I don't really get it can you explain what it is and why it matters",
     "Object-oriented programming (OOP) is a way of organizing code around objects -- data structures that combine data (attributes) and functions (methods). For example, a 'Car' object might have attributes like color and speed, and methods like accelerate() and brake(). OOP helps organize large codebases, makes code reusable, and models real-world relationships naturally."),
    ("so my doctor mentioned something about cholesterol and I know it's important for health but I'm not really sure what it is exactly and what the difference is between good and bad cholesterol",
     "Cholesterol is a waxy substance in your blood that your body needs to build cells. LDL cholesterol (low-density lipoprotein) is called 'bad' because high levels deposit in artery walls, increasing heart disease risk. HDL cholesterol (high-density lipoprotein) is 'good' because it carries cholesterol away from arteries back to the liver."),
    ("okay I know everyone talks about climate change but I'm genuinely confused about what exactly causes it and why scientists think it's getting worse and what the actual evidence is",
     "Climate change is primarily caused by greenhouse gases -- mainly carbon dioxide and methane -- trapping heat in Earth's atmosphere. Human activities like burning fossil fuels and deforestation have significantly increased these gas concentrations since the Industrial Revolution. Evidence includes rising global average temperatures, shrinking ice sheets, rising sea levels, shifting weather patterns, and ocean acidification, all corroborated by multiple independent measurement systems."),
    ("I keep hearing about inflation and interest rates and I feel like I should understand this stuff but whenever I try to read about it my eyes glaze over can you just explain what inflation actually is and how interest rates relate to it",
     "Inflation is the rate at which prices for goods and services rise over time, reducing purchasing power. When inflation is high, central banks raise interest rates to make borrowing more expensive, which slows spending and investment, cooling the economy and reducing price pressure. When inflation is low, interest rates are cut to encourage borrowing and growth."),
    ("so I want to start exercising but I don't really know where to begin and I've heard so many different things about cardio versus weights and I'm not sure what's actually best for my overall health",
     "Both cardio and strength training benefit health in different ways. Cardio (running, cycling, swimming) improves heart and lung function, burns calories, and reduces cardiovascular disease risk. Strength training builds muscle, improves metabolism, and supports bone density. Current guidelines recommend both: at least 150 minutes of moderate cardio per week plus strength training twice a week. Starting with any movement is better than waiting for the perfect plan."),
    ("I've been trying to eat healthier and I keep reading about macronutrients and micronutrients but I don't fully understand what those terms mean and how they're different from each other",
     "Macronutrients are the three major energy-providing nutrients: carbohydrates, proteins, and fats. The body needs them in large quantities. Micronutrients are vitamins and minerals -- essential for health but needed in small amounts. Examples: vitamin C, vitamin D, calcium, iron. Both are necessary; macros provide energy, micros support biochemical functions."),
    ("I need to understand how the stock market works because I want to start investing but every time I try to learn about it there's so much jargon and I get overwhelmed can you give me a basic explanation",
     "A stock represents a small ownership share in a company. The stock market is where these shares are bought and sold. Companies list shares publicly to raise capital. Investors buy shares hoping the company grows and shares increase in value. Prices change based on supply and demand, which is influenced by company performance, economic conditions, and investor sentiment. Major stock exchanges include the NYSE and NASDAQ."),
    ("my friend told me about blockchain and crypto and I've heard those words so many times but honestly I'm still not sure what blockchain actually is at its core can you just explain the concept clearly",
     "A blockchain is a distributed ledger -- a record of transactions that is duplicated across many computers simultaneously. Each batch of transactions is a 'block', and blocks are linked in sequence, forming a 'chain'. Because the ledger is distributed and each block references the previous one, altering past records would require changing every subsequent block across thousands of computers simultaneously, making it extremely tamper-resistant."),
    ("I work in software and I keep hearing about microservices and I kind of understand the general idea but I'm not sure exactly how it differs from a regular monolithic application and when you'd want to use one approach over the other",
     "A monolithic application is built as a single deployable unit -- all functions (user interface, business logic, database) are tightly coupled. A microservices architecture splits the application into small, independently deployable services, each responsible for one function and communicating over APIs. Microservices scale more flexibly and allow teams to deploy independently, but add complexity in service communication, monitoring, and orchestration. Monoliths are simpler for small teams and early-stage products; microservices suit large teams and applications that need to scale components independently."),
    ("so I've been reading about sleep and I know it's important but I want to understand what's actually happening in my brain when I sleep like what are the different stages and why do we need them",
     "Sleep has two main types: NREM (non-rapid eye movement) and REM (rapid eye movement), cycling roughly every 90 minutes. NREM has three stages: N1 (light sleep, transition), N2 (consolidated sleep, memory consolidation begins), and N3 (deep/slow-wave sleep, physical restoration, immune function, growth hormone release). REM sleep features dreaming, emotional processing, and memory consolidation. The brain cycles through these stages multiple times per night. Skipping sleep impairs memory, immune function, metabolism, and cognitive performance."),
]

# ---------------------------------------------------------------------------
# SECTION F: Correction and rewrite requests (explicit scope markers)
# ---------------------------------------------------------------------------
CORRECTION_REWRITE_PAIRS: list[tuple[str, str]] = [
    ("Correct the spelling and punctuation of this text: i wanted to ask if your comming to the party on saturday because i dont no if theres enough room and i need to let sarah no soon",
     "Corrected: 'I wanted to ask if you're coming to the party on Saturday, because I don't know if there's enough room and I need to let Sarah know soon.' Corrections: capitalized 'I', 'Saturday', 'Sarah'; 'your' -> 'you're'; 'comming' -> 'coming'; 'dont' -> \"don't\"; 'no' -> 'know' (twice); 'theres' -> \"there's\"; added comma."),
    ("Clean up this text and fix all errors: the report was suppose to be submitted by friday but john forgot to attatch the spreadsheat and know we have to wait until monday to get the complet version",
     "Corrected: 'The report was supposed to be submitted by Friday, but John forgot to attach the spreadsheet and now we have to wait until Monday to get the complete version.' Corrections: capitals for sentence start and proper nouns; 'suppose' -> 'supposed'; 'attatch' -> 'attach'; 'spreadsheat' -> 'spreadsheet'; 'know' -> 'now'; 'complet' -> 'complete'."),
    ("Rewrite this with correct grammar and punctuation: its been a really good year for the company we launched two new products and hired 50 new employees and our revenue increased by thirty percent and where planning to expand to three new markets next year",
     "Corrected: \"It's been a really good year for the company. We launched two new products and hired 50 new employees, and our revenue increased by thirty percent. We're planning to expand to three new markets next year.\" Key fixes: 'its' -> \"it's\"; sentence breaks added; 'where' -> \"we're\"; commas added."),
    ("Fix the grammar, spelling, and punctuation: Me and him went to see the new movie last nite and it was way better then the last one we seen together the effects was incredible",
     "Corrected: 'He and I went to see the new movie last night, and it was way better than the last one we had seen together. The effects were incredible.' Corrections: 'Me and him' -> 'He and I' (subject case); 'nite' -> 'night'; 'then' -> 'than' (comparison); 'we seen' -> 'we had seen'; 'effects was' -> 'effects were'."),
    ("Please correct all errors in this message: thankyou for you're email i recieved it this morning and i will defenately respond with the infomation you requested before end of buisness today",
     "Corrected: 'Thank you for your email. I received it this morning and will definitely respond with the information you requested before end of business today.' Corrections: 'thankyou' -> 'Thank you' (two words, capitalized); 'you're' -> 'your'; 'recieved' -> 'received'; 'defenately' -> 'definitely'; 'infomation' -> 'information'; 'buisness' -> 'business'."),
    ("Rewrite this professionally with correct grammar and spelling: i think the reason why the project failed is because we didnt have enough resourses and the timeline was to short and nobody was really sure who was in charge and communications between teams wasnt great",
     "Rewritten: 'The project failed for several reasons: insufficient resources, an unrealistic timeline, unclear ownership, and poor inter-team communication.' Or in fuller form: 'I believe the project failed because we lacked sufficient resources, the timeline was too short, accountability was unclear, and communication between teams was inadequate.'"),
    ("Fix the spelling and grammar: The new policy effect alot of employes particularly those who work in the field and we should of consulted with them before makeing any descisions",
     "Corrected: 'The new policy affects a lot of employees, particularly those who work in the field, and we should have consulted with them before making any decisions.' Corrections: 'effect' -> 'affects' (verb); 'alot' -> 'a lot'; 'employes' -> 'employees'; 'should of' -> 'should have'; 'makeing' -> 'making'; 'descisions' -> 'decisions'."),
    ("Correct this email draft: Dear Mr Smith I am writting to followup on are meeting last week regarding the proposel we discused. I beleive the terms we aggred on are mutualy benificial and I look foward to recieving you're formal response. Yours Faithfully",
     "Corrected: 'Dear Mr. Smith, I am writing to follow up on our meeting last week regarding the proposal we discussed. I believe the terms we agreed on are mutually beneficial, and I look forward to receiving your formal response. Yours faithfully,' Corrections: 'writting' -> 'writing'; 'followup' -> 'follow up'; 'are' -> 'our'; 'proposel' -> 'proposal'; 'discused' -> 'discussed'; 'beleive' -> 'believe'; 'aggred' -> 'agreed'; 'mutualy' -> 'mutually'; 'benificial' -> 'beneficial'; 'foward' -> 'forward'; 'recieving' -> 'receiving'; 'you're' -> 'your'."),
    ("Rewrite this with proper structure and correct all errors: i went to the interview this morning and i think it went pretty good there was three interviewers and they asked me alot of questions about my experience and i answered them as best as i could and now i just gotta wait and see",
     "Rewritten: 'I went to the interview this morning and I think it went well. There were three interviewers who asked me many questions about my experience. I answered them as best I could and now I just have to wait and see.' Key corrections: 'pretty good' -> 'well' (adverb needed); 'there was' -> 'there were' (plural); 'alot' -> 'many'; 'gotta' -> 'have to'."),
]

ALL_PAIRS = (
    LONG_Q_LONG_A_PAIRS
    + LONG_Q_SHORT_A_PAIRS
    + RUNON_CORRECTION_PAIRS
    + GRAMMAR_CORRECTION_PAIRS
    + RUNON_UNDERSTANDING_PAIRS
    + CORRECTION_REWRITE_PAIRS
)

STAGES = {42: "Long-form Q&A, run-on sentence correction, grammar/punctuation, and understanding informal input"}


def post(node: str, path: str, body: dict, timeout: int = 30) -> dict:
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
            "confidence": 0.93,
            "evidence": "Stage 42 communication corpus: long-form, run-on, grammar correction",
            "review_status": "approved",
        }
        for q, a in pairs
    ]
    return post(node, "/qa/ingest", {"candidates": candidates})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="localhost:8090")
    parser.add_argument("--stages", default="42")
    parser.add_argument("--repeats", type=int, default=100,
                        help="Times to repeat the full corpus (default 100)")
    args = parser.parse_args()

    total = len(ALL_PAIRS)
    print(f"Stage 42 -- Communication Corpus")
    print(f"  {total} QA pairs x {args.repeats} repeats = {total * args.repeats} total ingestions")
    print(f"  Breakdown: {len(LONG_Q_LONG_A_PAIRS)} long Q/long A, "
          f"{len(LONG_Q_SHORT_A_PAIRS)} long Q/short A, "
          f"{len(RUNON_CORRECTION_PAIRS)} run-on corrections, "
          f"{len(GRAMMAR_CORRECTION_PAIRS)} grammar corrections, "
          f"{len(RUNON_UNDERSTANDING_PAIRS)} run-on understanding, "
          f"{len(CORRECTION_REWRITE_PAIRS)} text correction/rewrite")
    print(f"  Node: {args.node}\n")

    errors = 0
    batches_done = 0
    total_batches = (total * args.repeats + BATCH_SIZE - 1) // BATCH_SIZE

    for repeat in range(args.repeats):
        book_id = f"communication_r{repeat:04d}"
        offset = (repeat * 17) % total
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
    print("Node trained on long-form Q&A, run-on correction, grammar, and informal input understanding.")


if __name__ == "__main__":
    main()
