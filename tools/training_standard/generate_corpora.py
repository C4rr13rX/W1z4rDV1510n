"""tools/training_standard/generate_corpora.py — produce thousands of training pairs.

Programmatic corpus generators for every registry script.  Templates are
parameterised across:
  - arithmetic / data-structure / string operations × multiple languages
  - common errors × their fixes
  - dialog patterns × topic substitutions
  - planning patterns × project shapes
  - summary patterns × paragraph variations

Each generator writes its output to data/training/<script_id>.jsonl,
OVERWRITING any seed file present.  Targets are sized so the cross-pool
fabric has enough co-occurrence to clear the runner's keyword + AST
benchmark gates.

Usage:
    python -m tools.training_standard.generate_corpora        # generate all
    python -m tools.training_standard.generate_corpora --only code_gen_python_001
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Deterministic so reruns produce the same corpus (training reproducibility).
random.seed(20260514)

_HERE         = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent.parent
_DATA_DIR     = _PROJECT_ROOT / "data" / "training"
_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _write(path: Path, rows: list[dict]) -> None:
    """Write a JSONL file.  Each row must have 'prompt' and 'response'."""
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>5d}  {path.relative_to(_PROJECT_ROOT)}")


# ── Conversation basics ──────────────────────────────────────────────────

_TOPICS = [
    # ── Architecture-internal concepts ───────────────────────────────
    ("a neuron",            "the basic unit of my fabric — dendrites receive, axons emit, fires above threshold"),
    ("a concept neuron",    "a composite neuron whose members are other neurons; fires when its members fire"),
    ("a synapse",           "a directional connection between two neurons with a weight that strengthens via Hebbian co-firing"),
    ("Hebbian learning",    "the rule that synapses between simultaneously-active neurons strengthen; cells that fire together wire together"),
    ("the multi-pool fabric","an N-pool architecture where each pool represents one modality and cross-pool synapses bind concepts across modalities"),
    ("a cross-pool edge",   "a weighted connection between a concept in one pool and a concept in another; sign can be excitatory or inhibitory"),
    ("inhibitory synapses", "edges that subtract from a target's activation when the source fires; implement categorical exclusion"),
    ("excitatory synapses", "edges that add to a target's activation when the source fires; the default learning sign"),
    ("contrastive Hebbian", "after pairing A with B, build inhibitory edges from A to every other existing concept in B's pool"),
    ("dopamine",            "a neuromodulator that gates plasticity: when present, recent traces are captured into permanent consolidation"),
    ("norepinephrine",      "a neuromodulator triggered by novelty; boosts learning rate and lowers activation thresholds"),
    ("acetylcholine",       "a neuromodulator gating plasticity multiplicatively; 1.0 means normal learning, lower means consolidation mode"),
    ("serotonin",           "a neuromodulator that shifts the homeostatic target activation level; mood baseline"),
    ("a pool",              "a collection of neurons representing one modality — keyboard text, image pixels, audio features, etc."),
    ("online learning",     "training that happens continuously with every observation, not in offline batch passes"),
    ("temporal binding",    "wiring concepts from a previous observation in a session to the current observation's concepts"),
    ("a hierarchical concept", "a parent concept whose members are themselves concepts; fires when any child fires"),
    ("the chat endpoint",   "POST /chat that takes text and returns the multi-pool decode plus a char-chain fallback"),
    ("the sensor endpoint", "POST /sensor/observe that takes raw bytes or text and trains the matching modality pool"),
    ("a checkpoint",        "the act of persisting hot-tier neurons and cross-pool synapses to disk so restart preserves state"),
    ("a regression test",   "a test that locks in known-good behavior so a future change doesn't accidentally break it"),
    ("a benchmark",         "a measured contract a training script promises to satisfy after training"),
    ("the wizard node",     "the W1z4rD V1510n process that runs the neural fabric and answers /chat requests"),
    # ── Programming languages ────────────────────────────────────────
    ("Python",              "a high-level interpreted language with clean syntax used for scripts, data work, web backends, and machine learning"),
    ("Rust",                "a systems programming language with strong memory safety guarantees via ownership and borrow checking"),
    ("JavaScript",          "the language of the web; runs in browsers and on the server via Node.js"),
    ("TypeScript",          "a strict superset of JavaScript that adds static types; compiles down to plain JavaScript"),
    ("Go",                  "a compiled language designed at Google with goroutines for concurrency and a small standard library"),
    ("Java",                "an object-oriented language that runs on the JVM; widely used in enterprise backends and Android"),
    ("C",                   "a low-level systems language; small, fast, close to the hardware, the foundation of Unix and most other languages"),
    ("C++",                 "an object-oriented extension of C with templates, classes, and zero-cost abstractions"),
    ("C#",                  "a Microsoft language similar to Java; runs on .NET and is the default for Windows desktop and Unity games"),
    ("Ruby",                "a dynamic interpreted language with elegant syntax; the engine behind Rails web framework"),
    ("PHP",                 "a server-side scripting language that powers a huge fraction of the web, including WordPress"),
    ("Swift",               "Apple's modern language for iOS, macOS, watchOS, and tvOS development"),
    ("Kotlin",              "a JVM language that interoperates with Java; the preferred language for Android development"),
    ("Scala",               "a JVM language blending object-oriented and functional programming; used at scale by companies like Twitter"),
    ("Haskell",             "a purely functional language with strong static types and lazy evaluation"),
    ("Erlang",              "a functional language built for concurrent, fault-tolerant systems; powers WhatsApp and many telecom backends"),
    ("Elixir",              "a functional language built on the Erlang virtual machine with friendlier syntax"),
    ("Clojure",             "a Lisp dialect that runs on the JVM with immutable data structures and excellent concurrency primitives"),
    ("OCaml",               "a statically-typed functional language with pattern matching and an emphasis on safety"),
    ("Lua",                 "a small embeddable scripting language; common in game engines and configuration"),
    ("Perl",                "a dynamic language with powerful text-processing and a long history in Unix administration"),
    ("R",                   "a language and environment for statistical computing and graphics"),
    ("Julia",               "a high-performance dynamic language designed for numerical and scientific computing"),
    ("MATLAB",              "a proprietary numerical computing environment widely used in engineering and research"),
    ("SQL",                 "a declarative language for querying and modifying relational databases"),
    ("Bash",                "the default shell on most Linux and macOS systems; used for scripting and one-line commands"),
    ("PowerShell",          "the default shell on modern Windows; pipes structured objects instead of plain text"),
    ("Zsh",                 "a modern Unix shell with strong tab completion and customisation; the default on macOS"),
    ("Fish",                "a user-friendly shell with sensible defaults and inline autosuggestions"),
    ("HTML",                "the markup language that structures content on the web"),
    ("CSS",                 "the styling language for the web; controls colour, layout, and visual presentation of HTML"),
    ("WebAssembly",         "a binary instruction format for a stack-based virtual machine; lets compiled languages run in the browser at near-native speed"),
    # ── Web / networking ─────────────────────────────────────────────
    ("HTTP",                "the Hypertext Transfer Protocol; the request-response protocol the web is built on"),
    ("HTTPS",               "HTTP wrapped in TLS encryption; the standard for production web traffic"),
    ("REST",                "Representational State Transfer; an architectural style for web APIs based on resources and HTTP verbs"),
    ("GraphQL",             "a query language and runtime for APIs where the client specifies exactly what data it wants"),
    ("WebSocket",           "a protocol that keeps a long-lived bidirectional connection between client and server"),
    ("JSON",                "a lightweight text format for structured data; the default body format for modern web APIs"),
    ("XML",                 "a structured text format that predated JSON; still widely used in document formats and SOAP APIs"),
    ("DNS",                 "the Domain Name System; translates human-readable hostnames into IP addresses"),
    ("TCP",                 "the Transmission Control Protocol; provides reliable ordered byte streams over IP"),
    ("UDP",                 "the User Datagram Protocol; unreliable but low-overhead packet delivery over IP"),
    ("a TLS handshake",     "the negotiation that establishes a TLS-encrypted channel between client and server"),
    ("CORS",                "Cross-Origin Resource Sharing; the browser policy that controls which origins can call which APIs"),
    ("OAuth",               "an authorisation framework that lets third parties access an account without seeing the password"),
    ("a JWT",               "a JSON Web Token; a signed token used for stateless authentication"),
    ("a cookie",            "a small piece of state the server asks the browser to store and send back on subsequent requests"),
    ("a session",           "server-side state associated with a particular client, usually keyed by a session cookie"),
    ("a CDN",               "a Content Delivery Network; caches static assets at edge locations close to users"),
    ("a load balancer",     "a server that distributes incoming requests across a pool of backend instances"),
    ("a reverse proxy",     "a server in front of one or more backends that forwards requests; common for TLS termination and caching"),
    ("a webhook",           "an HTTP callback the server makes when an event occurs; the inverse of polling"),
    # ── Data structures ──────────────────────────────────────────────
    ("an array",            "a contiguous block of memory holding fixed-size elements indexed by integer position"),
    ("a list",              "an ordered collection of elements; in Python it grows dynamically, in Rust it is Vec, in C it is often a linked list"),
    ("a stack",             "a LIFO collection: the last item pushed is the first item popped"),
    ("a queue",             "a FIFO collection: items are added at the back and removed from the front"),
    ("a deque",             "a double-ended queue; supports efficient push and pop at both ends"),
    ("a set",               "an unordered collection of unique elements with fast membership tests"),
    ("a map",               "a key-to-value lookup structure; called dict in Python, HashMap in Rust, object in JavaScript"),
    ("a tree",              "a hierarchical data structure with a root node and child branches"),
    ("a binary search tree", "a tree where each node has at most two children and the left subtree is smaller than the right"),
    ("a heap",              "a tree-shaped structure that keeps the smallest (or largest) element at the root for fast min/max access"),
    ("a hash table",        "a data structure that maps keys to values using a hash function for average-O(1) lookup"),
    ("a linked list",       "a sequence of nodes each pointing to the next; cheap insertions but linear-time random access"),
    ("a graph",             "a collection of nodes connected by edges; directed or undirected, weighted or unweighted"),
    ("a trie",              "a tree where each node represents a character; common for autocomplete and dictionary lookups"),
    ("a Bloom filter",      "a probabilistic structure that answers 'is x in the set?' with no false negatives but possible false positives"),
    # ── Algorithms ───────────────────────────────────────────────────
    ("recursion",           "a function that calls itself; the base case stops the recursion, the recursive case reduces the problem"),
    ("dynamic programming", "solving a problem by combining solutions to overlapping subproblems and memoising results"),
    ("breadth-first search","a graph traversal that explores all neighbours of a node before moving deeper"),
    ("depth-first search",  "a graph traversal that goes as deep as possible before backtracking"),
    ("binary search",       "a search algorithm that halves the search space at each step on a sorted array"),
    ("merge sort",          "a divide-and-conquer sort that splits, sorts the halves, and merges them back in O(n log n)"),
    ("quicksort",           "a divide-and-conquer sort that partitions around a pivot; average O(n log n), worst O(n²)"),
    ("Dijkstra's algorithm","a shortest-path algorithm for weighted graphs with non-negative edge weights"),
    ("the A* algorithm",    "a best-first search that uses a heuristic to estimate distance to the goal"),
    ("a greedy algorithm",  "an algorithm that makes the locally optimal choice at each step in hope of a globally optimal result"),
    # ── Databases ────────────────────────────────────────────────────
    ("a relational database","a database that stores data in tables with rows and columns linked by foreign keys"),
    ("a NoSQL database",    "a database that abandons the strict relational model; document, key-value, columnar, or graph"),
    ("a transaction",       "a unit of database work that either fully commits or fully rolls back"),
    ("ACID",                "Atomicity, Consistency, Isolation, Durability — the guarantees a relational database transaction provides"),
    ("an index",            "a database structure that speeds up lookups at the cost of slower writes and more storage"),
    ("a foreign key",       "a column that references the primary key of another table, enforcing relational integrity"),
    ("a primary key",       "a column or set of columns that uniquely identifies a row in a table"),
    ("a join",              "an operation that combines rows from two tables based on matching column values"),
    ("a query plan",        "the engine's chosen strategy for executing a SQL query; inspect it to find missing indexes"),
    ("a deadlock",          "two transactions each holding a lock the other needs; the database aborts one to resolve it"),
    # ── Operating systems ────────────────────────────────────────────
    ("a process",           "an instance of a running program with its own memory, file descriptors, and lifecycle"),
    ("a thread",            "an execution unit within a process; threads share memory and need synchronisation"),
    ("a file descriptor",   "an integer handle the kernel gives a process for an open file, socket, or pipe"),
    ("a socket",            "a network endpoint a process opens to send or receive data"),
    ("a pipe",              "a unidirectional byte stream between two file descriptors, often between processes in a shell pipeline"),
    ("a system call",       "a request a user-space program makes to the kernel to perform a privileged operation"),
    ("virtual memory",      "the illusion of a large contiguous address space per process backed by physical RAM and swap"),
    ("a page fault",        "a hardware exception when a process accesses a virtual page not currently in physical RAM"),
    ("a context switch",    "the kernel saving one thread's state and loading another's to share the CPU"),
    ("a daemon",            "a long-running background process, typically detached from a terminal"),
    # ── Machine learning ─────────────────────────────────────────────
    ("a neural network",    "a layered model of weighted connections trained by adjusting weights to minimise an error signal"),
    ("backpropagation",     "the algorithm that propagates loss gradients backwards through a neural network to update its weights"),
    ("gradient descent",    "an optimisation method that moves parameters in the direction that most decreases the loss"),
    ("overfitting",         "when a model memorises training data details that don't generalise to unseen examples"),
    ("regularisation",      "techniques that discourage overly complex models, e.g. L1, L2, or dropout"),
    ("supervised learning", "training on labelled examples — each input comes with the desired output"),
    ("unsupervised learning","training without labels; the model finds structure in the input distribution itself"),
    ("reinforcement learning","an agent learns to act in an environment by receiving rewards or penalties"),
    ("a transformer",       "an attention-based neural network architecture that powers modern large language models"),
    ("an embedding",        "a dense vector representation of a discrete item like a word, sentence, or user"),
    # ── Math ─────────────────────────────────────────────────────────
    ("the mean",            "the arithmetic average: sum of values divided by their count"),
    ("the median",          "the middle value of a sorted list; robust to outliers compared to the mean"),
    ("the standard deviation","a measure of how spread out values are around the mean"),
    ("the variance",        "the square of the standard deviation; the average squared distance from the mean"),
    ("a derivative",        "the instantaneous rate of change of a function at a point"),
    ("an integral",         "the area under a curve; the inverse of differentiation"),
    ("a vector",            "an ordered list of numbers representing a point or direction in some-dimensional space"),
    ("a matrix",            "a 2D array of numbers; the central object of linear algebra"),
    ("the dot product",     "a scalar value computed by multiplying corresponding entries of two vectors and summing"),
    ("an eigenvalue",       "a scalar λ such that for a matrix A there is a vector v with A·v = λ·v"),
    # ── Software-engineering practice ────────────────────────────────
    ("version control",     "the practice of recording every change to a codebase so any prior state can be restored"),
    ("git",                 "a distributed version-control system; every clone is a full repository with full history"),
    ("a pull request",      "a proposed set of changes submitted for review before merging into the main branch"),
    ("code review",         "the practice of reading another engineer's diff before it lands to catch bugs and share knowledge"),
    ("continuous integration","automated build and test runs that fire on every push to detect breakage early"),
    ("continuous deployment", "automated promotion of changes from CI all the way to production after passing tests"),
    ("a unit test",         "a test that exercises a small isolated piece of code, ideally a single function"),
    ("an integration test", "a test that exercises multiple components together to catch interaction bugs"),
    ("test-driven development","writing the test first, watching it fail, then writing the code to make it pass"),
    ("refactoring",         "changing the structure of code without changing its observable behaviour"),
    ("a code smell",        "a pattern in code that suggests a deeper problem; long methods, duplicated logic, primitive obsession"),
    ("technical debt",      "the accumulated cost of taking shortcuts; refactor regularly to keep it manageable"),
    ("a stack trace",       "the chain of function calls that led to an error; read top-down to find the root cause"),
    ("logging",             "emitting structured records of what a program is doing; the foundation of observability"),
    ("a debugger",          "a tool that pauses program execution so you can inspect state and step through code"),
    ("a profiler",          "a tool that measures where a program spends its time or allocates its memory"),
    ("a static analyser",   "a tool that inspects source code without running it; finds bugs, unused code, style issues"),
    ("a linter",            "a static analyser focused on style and common mistakes; runs in the editor and CI"),
    ("idempotency",         "the property that calling an operation multiple times has the same effect as calling it once"),
    ("a race condition",    "a bug where the result depends on unpredictable timing of concurrent operations"),
    ("a deadlock",          "two threads each waiting for a lock the other holds; neither can proceed"),
    ("a livelock",          "two threads each yielding to the other in a way that prevents real progress"),
    ("a memory leak",       "memory allocated but never freed; the process's footprint grows without bound"),
    ("a buffer overflow",   "writing past the end of a buffer; the classic source of security exploits in C code"),
    ("dependency injection","passing a component's collaborators in from outside instead of constructing them internally"),
    ("an API",              "an Application Programming Interface; the contract a piece of software exposes to its callers"),
    ("a CLI",               "a Command-Line Interface; a text-driven interaction model versus a GUI"),
    # ── DevOps / infra ───────────────────────────────────────────────
    ("Docker",              "a container runtime that packages an application with its dependencies into a portable image"),
    ("Kubernetes",          "a container orchestrator that schedules and manages containerised workloads across a cluster"),
    ("infrastructure as code","describing infrastructure in version-controlled config so it can be reproduced reliably"),
    ("Terraform",           "a tool for declaring cloud infrastructure as code; works across AWS, GCP, Azure, and others"),
    ("Ansible",             "a configuration-management tool that runs idempotent playbooks over SSH"),
    ("a CI pipeline",       "a sequence of automated jobs run on every change to build, test, and deploy software"),
    ("a deployment",        "the act of promoting a new build into a running environment, ideally with zero downtime"),
    ("a rolling deploy",    "replacing instances of an old version with the new one gradually, keeping the service available"),
    ("a canary release",    "deploying a new version to a small percentage of traffic to detect problems before full rollout"),
    ("an SLA",              "a service-level agreement: a contractual promise about availability or response time"),
    ("an SLI",              "a service-level indicator: a measured metric of how the service is performing"),
    ("an SLO",              "a service-level objective: an internal target for an SLI"),
    # ── Security ─────────────────────────────────────────────────────
    ("encryption",          "transforming readable data into ciphertext using a key so only authorised parties can recover it"),
    ("symmetric encryption","encryption where the same key encrypts and decrypts; fast but key distribution is tricky"),
    ("asymmetric encryption","encryption where a public key encrypts and a private key decrypts; the basis of TLS"),
    ("hashing",             "a one-way function that maps data to a fixed-size digest; can't be reversed in practice"),
    ("a salt",              "random data added to a password before hashing so identical passwords don't have identical hashes"),
    ("SQL injection",       "an attack where user-supplied data is interpreted as SQL; defend with parameterised queries"),
    ("cross-site scripting", "injecting executable script into a page so it runs in another user's browser; escape outputs to prevent"),
    ("CSRF",                "Cross-Site Request Forgery: tricking a logged-in user's browser into making an unwanted request"),
    ("two-factor authentication","requiring a second factor (a code, hardware key) in addition to a password"),
    ("a zero-day",          "a vulnerability that is exploited before the vendor releases a fix"),
]

_GREETINGS = [
    ("Hi",                  "Hello.  What would you like to work on?"),
    ("Hello",               "Hi.  I am W1z4rD.  Tell me what you need."),
    ("Hey",                 "Hey.  Ready when you are."),
    ("Good morning",        "Good morning.  Let's get started."),
    ("Good evening",        "Good evening.  How can I help?"),
    ("Are you there",       "Yes, I am here."),
    ("Are you online",      "Yes, online and learning."),
    ("Are you ready",       "Yes — ask anything."),
]

_PHRASING = [
    "What is {topic}?",
    "Tell me about {topic}.",
    "Explain {topic}.",
    "Describe {topic}.",
    "Give me a definition of {topic}.",
    "How would you define {topic}?",
    "Can you explain {topic} to me?",
    "What does {topic} mean?",
    "I want to learn about {topic}.",
    "Help me understand {topic}.",
]


def gen_conversation_basics() -> list[dict]:
    rows: list[dict] = []
    # Identity / greeting block
    for q, a in _GREETINGS:
        rows.append({"prompt": q, "response": a})
        rows.append({"prompt": q + ".", "response": a})
        rows.append({"prompt": q + "!", "response": a})
    # Topic × phrasing combinations
    for topic, definition in _TOPICS:
        for ph in _PHRASING:
            rows.append({
                "prompt":   ph.format(topic=topic),
                "response": f"{topic.capitalize()} is {definition}.",
            })
        # Direct topic prompt
        rows.append({
            "prompt":   topic.capitalize() + "?",
            "response": f"{topic.capitalize()} is {definition}.",
        })
    # Self-references
    self_pairs = [
        ("Who are you?",           "I am W1z4rD, a multi-pool Hebbian vision node."),
        ("What are you?",          "I am W1z4rD — a neural fabric that learns online from every observation."),
        ("What do you do?",        "I learn from conversations and multimodal inputs and respond from what I have learned."),
        ("Are you still learning?","Yes.  Every observation updates my synapse weights; there is no fully-trained state."),
        ("Are you fully trained?", "No.  Training is online and continuous."),
        ("How were you built?",    "I am a multi-pool Hebbian fabric written in Rust, with a Vue chat front end and a Django service layer."),
        ("Who made you?",          "I am part of the W1z4rD V1510n project."),
        ("What can you do?",       "I can answer questions, generate code, plan tasks, diagnose errors, and learn from each correction."),
        ("Can you write code?",    "Yes, in Python, Rust, JavaScript, Bash, and PowerShell.  I can also diagnose errors I encounter."),
        ("Are you an AI?",         "I am a neural fabric — an AI in the sense that I learn associations and answer from them."),
        ("How do you learn?",      "Hebbian.  When two concepts fire together their synapse strengthens; over time concepts cluster by co-occurrence."),
        ("Thanks",                 "You're welcome.  Tell me when you want to keep going."),
        ("Thank you",              "You're welcome."),
        ("Goodbye",                "Goodbye.  I will keep learning while you're away."),
        ("Bye",                    "Bye."),
        ("Are you online?",        "Yes, online and responding."),
        ("Can you remember?",      "Yes — within a session, temporal binding wires sequential observations together."),
    ]
    for q, a in self_pairs:
        rows.append({"prompt": q, "response": a})
    return rows


# ── Conversation dialog (multi-turn) ─────────────────────────────────────

_DIALOG_THEMES = [
    ("python",      "Python",     "A high-level interpreted language; clean syntax, batteries-included standard library."),
    ("rust",        "Rust",       "A systems language with ownership-based memory safety and zero-cost abstractions."),
    ("javascript",  "JavaScript", "The language of the web; ubiquitous in browsers and on the server via Node.js."),
    ("typescript",  "TypeScript", "A typed superset of JavaScript that compiles to plain JavaScript."),
    ("go",          "Go",         "A compiled language with goroutines for concurrency and a small simple standard library."),
    ("java",        "Java",       "An object-oriented JVM language used heavily in enterprise backends and Android."),
    ("c",           "C",          "The original portable systems language; small, fast, close to the hardware."),
    ("c++",         "C++",        "An object-oriented extension of C with templates and zero-cost abstractions."),
    ("ruby",        "Ruby",       "A dynamic interpreted language with elegant syntax; the engine behind Rails."),
    ("kotlin",      "Kotlin",     "A JVM language that interoperates with Java and is the preferred Android language."),
    ("bash",        "Bash",       "The dominant Unix shell; pipes, redirection, glob patterns, POSIX utilities."),
    ("powershell",  "PowerShell", "The structured-object shell from Microsoft; pipes pass objects, not text."),
    ("html",        "HTML",       "The markup language that structures content on the web."),
    ("css",         "CSS",        "The styling language for the web; controls colour, layout, and visual presentation."),
    ("sql",         "SQL",        "A declarative language for querying and modifying relational databases."),
    ("hebbian learning", "Hebbian learning", "When two neurons fire together the synapse between them strengthens — the basic plasticity rule."),
    ("neurons",     "neurons",    "Cells with dendrites for input, axons for output; fire above threshold and emit a signal."),
    ("multi-pool",  "the multi-pool fabric", "N independent neuron pools — one per modality — linked by cross-pool synapses."),
    ("training",    "training",   "The process of presenting paired observations so the fabric updates synapse weights."),
    ("synapses",    "synapses",   "Weighted connections between neurons; the weights are what the fabric actually learns."),
    ("inhibition",  "inhibition", "Edges that reduce a target's activation when the source fires; the source of categorical exclusion."),
    ("excitation",  "excitation", "Edges that increase a target's activation when the source fires; the default learning sign."),
    ("dopamine",    "dopamine",   "A neuromodulator that gates plasticity; when it spikes, recent traces consolidate into permanent connections."),
    ("git",         "Git",        "A distributed version-control system; every clone has the full history of the project."),
    ("docker",      "Docker",     "A container runtime that packages an app with all of its dependencies into a portable image."),
    ("kubernetes",  "Kubernetes", "A container orchestrator that schedules and manages workloads across a cluster of machines."),
    ("http",        "HTTP",       "The Hypertext Transfer Protocol; the request-response protocol the web is built on."),
    ("rest apis",   "REST APIs",  "Resource-oriented web APIs that use HTTP verbs (GET, POST, PUT, DELETE) and JSON bodies."),
    ("graphql",     "GraphQL",    "A query language for APIs where the client specifies exactly which fields it needs."),
    ("websockets",  "WebSockets", "A protocol that keeps a long-lived bidirectional connection open between client and server."),
    ("json",        "JSON",       "A lightweight text format for structured data; the default body format for modern web APIs."),
    ("oauth",       "OAuth",      "An authorisation framework that lets third parties access an account without seeing the password."),
    ("jwt",         "JWT",        "A JSON Web Token; a signed token used for stateless authentication."),
    ("regression tests", "regression tests", "Tests that lock in known-good behaviour so a future change doesn't break it accidentally."),
    ("unit tests",  "unit tests", "Tests that exercise a small isolated piece of code, ideally a single function."),
    ("integration tests", "integration tests", "Tests that exercise multiple components together to catch interaction bugs."),
    ("recursion",   "recursion",  "A function that calls itself; the base case stops the recursion, the recursive case reduces the problem."),
    ("trees",       "trees",      "Hierarchical data structures with a root node and child branches."),
    ("graphs",      "graphs",     "Collections of nodes connected by edges; directed or undirected, weighted or unweighted."),
    ("hash tables", "hash tables","Data structures that map keys to values using a hash function for average-O(1) lookup."),
    ("a linked list", "a linked list", "A sequence of nodes each pointing to the next; cheap insertions, linear-time random access."),
    ("dynamic programming", "dynamic programming", "Solving a problem by combining solutions to overlapping subproblems and memoising results."),
    ("binary search", "binary search", "A search algorithm that halves the search space at each step on a sorted array."),
    ("merge sort",  "merge sort", "A divide-and-conquer sort that splits, sorts the halves, and merges back in O(n log n)."),
    ("relational databases", "relational databases", "Databases that store data in tables with rows and columns linked by foreign keys."),
    ("acid",        "ACID",       "Atomicity, Consistency, Isolation, Durability — the guarantees a relational transaction provides."),
    ("indexes",     "indexes",    "Database structures that speed up lookups at the cost of slower writes and more storage."),
    ("transactions","transactions","Units of database work that either fully commit or fully roll back."),
    ("threads",     "threads",    "Execution units within a process; threads share memory and need synchronisation."),
    ("processes",   "processes",  "Instances of running programs with their own memory, file descriptors, and lifecycle."),
    ("encryption",  "encryption", "Transforming readable data into ciphertext using a key so only authorised parties can recover it."),
    ("hashing",     "hashing",    "A one-way function that maps data to a fixed-size digest; can't be reversed in practice."),
]


def gen_conversation_dialog() -> list[dict]:
    rows: list[dict] = []
    for key, name, def_ in _DIALOG_THEMES:
        # Two-turn pattern
        rows.append({
            "prompt":   f"User: What is {key}? Assistant:",
            "response": f"{name} is {def_}",
        })
        rows.append({
            "prompt":   f"User: What is {key}? Assistant: {name} is {def_} User: Tell me more. Assistant:",
            "response": f"Most importantly, {name.lower()} is what we use day to day — practical and well-supported.",
        })
        rows.append({
            "prompt":   f"User: What is {key}? Assistant: {name} is {def_} User: Why is it useful? Assistant:",
            "response": f"{name} is useful because it solves real problems with idioms that many people already know.",
        })
        rows.append({
            "prompt":   f"User: What is {key}? Assistant: {name} is {def_} User: Give me an example. Assistant:",
            "response": f"A typical example of {name.lower()} would show its core idioms in a few short lines.",
        })
    # Conversational continuation
    closers = [
        ("Thanks", "You're welcome."),
        ("Got it", "Glad I could help."),
        ("Cool", "Anytime."),
        ("Awesome", "Let me know what's next."),
        ("Nice", "Glad that worked."),
    ]
    for cl, resp in closers:
        for key, name, _ in _DIALOG_THEMES:
            rows.append({
                "prompt":   f"User: What is {key}? Assistant: {name}. User: {cl}. Assistant:",
                "response": resp,
            })
    return rows


# ── Code-gen Python ──────────────────────────────────────────────────────

_PY_ARITH = [
    ("add", "a + b", "return the sum of two numbers"),
    ("sub", "a - b", "return the difference of two numbers"),
    ("mul", "a * b", "return the product of two numbers"),
    ("div", "a / b if b != 0 else 0", "return the quotient, or 0 if dividing by zero"),
    ("mod", "a % b", "return a modulo b"),
    ("pow", "a ** b", "return a raised to the power b"),
    ("max_of", "a if a > b else b", "return the larger of two numbers"),
    ("min_of", "a if a < b else b", "return the smaller of two numbers"),
    ("abs_val", "n if n >= 0 else -n", "return the absolute value of n"),
    ("square", "n * n", "return n squared"),
    ("cube", "n * n * n", "return n cubed"),
    ("double", "n * 2", "return n doubled"),
    ("triple", "n * 3", "return n tripled"),
    ("half", "n / 2", "return half of n"),
    ("negate", "-n", "return the negative of n"),
    ("is_even", "n % 2 == 0", "return True if n is even"),
    ("is_odd", "n % 2 == 1", "return True if n is odd"),
    ("is_positive", "n > 0", "return True if n is positive"),
    ("is_negative", "n < 0", "return True if n is negative"),
    ("is_zero", "n == 0", "return True if n is zero"),
]


def gen_code_gen_python() -> list[dict]:
    rows: list[dict] = []
    # Arithmetic and predicate functions
    for name, body, desc in _PY_ARITH:
        argline = "(n)" if "n" in body and "a" not in body else "(a, b)"
        params = "n" if argline == "(n)" else "a, b"
        prompts = [
            f"Write a Python function `{name}{argline}` that {desc}.",
            f"In Python, write a function called {name} that {desc}.",
            f"Python: define {name}{argline} -- {desc}.",
            f"Give me a Python function {name} that {desc}.",
        ]
        response = f"def {name}({params}):\n    return {body}"
        for p in prompts:
            rows.append({"prompt": p, "response": response})
    # List/iterable utilities
    list_ops = [
        ("sum_list",      "sum the elements of a list",            "def sum_list(xs):\n    total = 0\n    for x in xs:\n        total = total + x\n    return total"),
        ("avg_list",      "return the average of a list of numbers", "def avg_list(xs):\n    if not xs:\n        return 0\n    return sum(xs) / len(xs)"),
        ("max_list",      "return the maximum of a list",          "def max_list(xs):\n    if not xs:\n        return None\n    best = xs[0]\n    for x in xs[1:]:\n        if x > best:\n            best = x\n    return best"),
        ("min_list",      "return the minimum of a list",          "def min_list(xs):\n    if not xs:\n        return None\n    best = xs[0]\n    for x in xs[1:]:\n        if x < best:\n            best = x\n    return best"),
        ("count_list",    "return the number of elements in a list", "def count_list(xs):\n    n = 0\n    for _ in xs:\n        n = n + 1\n    return n"),
        ("reverse_list",  "return a reversed copy of a list",      "def reverse_list(xs):\n    return list(reversed(xs))"),
        ("dedupe_list",   "remove duplicates from a list, preserving order", "def dedupe_list(xs):\n    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out"),
        ("sort_list",     "return a sorted copy of a list",        "def sort_list(xs):\n    return sorted(xs)"),
        ("filter_even",   "return only the even numbers from a list", "def filter_even(xs):\n    return [x for x in xs if x % 2 == 0]"),
        ("filter_odd",    "return only the odd numbers from a list",  "def filter_odd(xs):\n    return [x for x in xs if x % 2 == 1]"),
        ("first_n",       "return the first n elements of a list",  "def first_n(xs, n):\n    return xs[:n]"),
        ("last_n",        "return the last n elements of a list",   "def last_n(xs, n):\n    return xs[-n:]"),
        ("concat_lists",  "concatenate two lists",                  "def concat_lists(a, b):\n    return a + b"),
        ("any_positive",  "return True if any element is positive", "def any_positive(xs):\n    for x in xs:\n        if x > 0:\n            return True\n    return False"),
        ("all_positive",  "return True if every element is positive", "def all_positive(xs):\n    for x in xs:\n        if x <= 0:\n            return False\n    return True"),
        ("contains",      "return True if a list contains the given value", "def contains(xs, v):\n    for x in xs:\n        if x == v:\n            return True\n    return False"),
        ("index_of",      "return the index of the first occurrence, or -1", "def index_of(xs, v):\n    for i, x in enumerate(xs):\n        if x == v:\n            return i\n    return -1"),
        ("zip_lists",     "zip two lists into a list of pairs",     "def zip_lists(a, b):\n    return list(zip(a, b))"),
        ("flatten",       "flatten a list of lists one level deep", "def flatten(xss):\n    out = []\n    for xs in xss:\n        for x in xs:\n            out.append(x)\n    return out"),
        ("range_to_list", "return a list of integers from 0 to n exclusive", "def range_to_list(n):\n    return list(range(n))"),
    ]
    for name, desc, body in list_ops:
        prompts = [
            f"Write a Python function `{name}` that will {desc}.",
            f"Python function: {desc}.  Call it {name}.",
            f"Implement {name} in Python — it should {desc}.",
            f"Give me Python code for {name}: {desc}.",
        ]
        for p in prompts:
            rows.append({"prompt": p, "response": body})
    # String ops
    str_ops = [
        ("reverse_string",   "reverse a string",                                  "def reverse_string(s):\n    return s[::-1]"),
        ("is_palindrome",    "return True if a string reads the same forwards and backwards", "def is_palindrome(s):\n    return s == s[::-1]"),
        ("count_chars",      "return a dict of character → count",                "def count_chars(s):\n    out = {}\n    for c in s:\n        out[c] = out.get(c, 0) + 1\n    return out"),
        ("count_words",      "return a dict of word → count",                     "def count_words(s):\n    out = {}\n    for w in s.split():\n        out[w] = out.get(w, 0) + 1\n    return out"),
        ("uppercase",        "return a string in upper case",                     "def uppercase(s):\n    return s.upper()"),
        ("lowercase",        "return a string in lower case",                     "def lowercase(s):\n    return s.lower()"),
        ("title_case",       "return a string in title case",                     "def title_case(s):\n    return s.title()"),
        ("strip_whitespace", "strip leading and trailing whitespace",             "def strip_whitespace(s):\n    return s.strip()"),
        ("repeat_string",    "repeat a string n times",                           "def repeat_string(s, n):\n    return s * n"),
        ("starts_with",      "return True if string starts with a prefix",        "def starts_with(s, p):\n    return s.startswith(p)"),
        ("ends_with",        "return True if string ends with a suffix",          "def ends_with(s, p):\n    return s.endswith(p)"),
        ("contains_substr",  "return True if string contains a substring",        "def contains_substr(s, sub):\n    return sub in s"),
        ("char_at",          "return the character at index i, or empty string",  "def char_at(s, i):\n    if 0 <= i < len(s):\n        return s[i]\n    return ''"),
        ("string_length",    "return the length of a string",                     "def string_length(s):\n    n = 0\n    for _ in s:\n        n = n + 1\n    return n"),
        ("words_in",         "return the words of a string as a list",            "def words_in(s):\n    return s.split()"),
        ("join_words",       "join a list of strings with a single space",        "def join_words(xs):\n    return ' '.join(xs)"),
        ("first_word",       "return the first word of a string",                 "def first_word(s):\n    parts = s.split()\n    if parts:\n        return parts[0]\n    return ''"),
        ("last_word",        "return the last word of a string",                  "def last_word(s):\n    parts = s.split()\n    if parts:\n        return parts[-1]\n    return ''"),
    ]
    for name, desc, body in str_ops:
        prompts = [
            f"Write a Python function `{name}` that will {desc}.",
            f"Python: implement {name} to {desc}.",
            f"Define a Python function {name}: {desc}.",
        ]
        for p in prompts:
            rows.append({"prompt": p, "response": body})
    # Classic algorithms
    classics = [
        ("fib",          "Write a Python function `fib(n)` that returns the nth Fibonacci number using a loop.",
                          "def fib(n):\n    if n <= 0:\n        return 0\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
        ("factorial",    "Write a Python function `factorial(n)` that returns n!.",
                          "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result = result * i\n    return result"),
        ("gcd",          "Write a Python function `gcd(a, b)` using the Euclidean algorithm.",
                          "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a"),
        ("is_prime",     "Write a Python function `is_prime(n)` that returns True if n is prime.",
                          "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n    while i * i <= n:\n        if n % i == 0:\n            return False\n        i = i + 1\n    return True"),
        ("primes_upto",  "Write a Python function that returns all primes less than or equal to n.",
                          "def primes_upto(n):\n    out = []\n    for k in range(2, n + 1):\n        prime = True\n        i = 2\n        while i * i <= k:\n            if k % i == 0:\n                prime = False\n                break\n            i = i + 1\n        if prime:\n            out.append(k)\n    return out"),
        ("binary_search","Write a Python function `binary_search(arr, target)` that returns the index or -1.",
                          "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1"),
        ("bubble_sort",  "Write a Python function `bubble_sort(xs)` that sorts a list in place using bubble sort.",
                          "def bubble_sort(xs):\n    n = len(xs)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if xs[j] > xs[j+1]:\n                xs[j], xs[j+1] = xs[j+1], xs[j]\n    return xs"),
        ("merge_sort",   "Write a Python function `merge_sort(xs)` that returns a sorted list using merge sort.",
                          "def merge_sort(xs):\n    if len(xs) <= 1:\n        return xs\n    mid = len(xs) // 2\n    left = merge_sort(xs[:mid])\n    right = merge_sort(xs[mid:])\n    out = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            out.append(left[i]); i += 1\n        else:\n            out.append(right[j]); j += 1\n    out.extend(left[i:])\n    out.extend(right[j:])\n    return out"),
        ("reverse_linked_list",
                         "Write a Python function that reverses a singly linked list.  Each node has `value` and `next` attributes.",
                         "def reverse_linked_list(head):\n    prev = None\n    while head:\n        nxt = head.next\n        head.next = prev\n        prev = head\n        head = nxt\n    return prev"),
        ("fizzbuzz",     "Write a Python function `fizzbuzz(n)` that returns a list of fizzbuzz values up to n.",
                          "def fizzbuzz(n):\n    out = []\n    for i in range(1, n + 1):\n        if i % 15 == 0:\n            out.append('FizzBuzz')\n        elif i % 3 == 0:\n            out.append('Fizz')\n        elif i % 5 == 0:\n            out.append('Buzz')\n        else:\n            out.append(str(i))\n    return out"),
        ("word_freq",    "Write a Python function that takes a string and returns a dict of word -> count.",
                          "def word_freq(text):\n    out = {}\n    for w in text.split():\n        out[w] = out.get(w, 0) + 1\n    return out"),
        ("anagram",      "Write a Python function that returns True if two strings are anagrams.",
                          "def is_anagram(a, b):\n    return sorted(a) == sorted(b)"),
        ("read_file",    "Write a Python function `read_text(path)` that returns the contents of a text file.",
                          "def read_text(path):\n    with open(path, 'r', encoding='utf-8') as f:\n        return f.read()"),
        ("write_file",   "Write a Python function `write_text(path, content)` that writes a string to a file.",
                          "def write_text(path, content):\n    with open(path, 'w', encoding='utf-8') as f:\n        f.write(content)"),
        ("json_load",    "Write a Python function that loads a JSON file and returns the parsed object.",
                          "import json\n\ndef load_json(path):\n    with open(path, 'r', encoding='utf-8') as f:\n        return json.load(f)"),
        ("json_dump",    "Write a Python function that writes an object to a JSON file with 2-space indent.",
                          "import json\n\ndef dump_json(path, obj):\n    with open(path, 'w', encoding='utf-8') as f:\n        json.dump(obj, f, indent=2)"),
        ("http_get",     "Write a Python function using requests that performs a GET and returns the JSON body.",
                          "import requests\n\ndef http_get_json(url):\n    r = requests.get(url, timeout=10)\n    r.raise_for_status()\n    return r.json()"),
        ("class_point",  "Write a Python class Point with x and y attributes and a distance() method to another Point.",
                          "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def distance(self, other):\n        dx = self.x - other.x\n        dy = self.y - other.y\n        return (dx * dx + dy * dy) ** 0.5"),
        ("decorator_timer", "Write a Python decorator @timed that prints how long the wrapped function took.",
                            "import time\n\ndef timed(fn):\n    def wrapper(*args, **kwargs):\n        t0 = time.time()\n        result = fn(*args, **kwargs)\n        print(f'{fn.__name__} took {time.time() - t0:.3f}s')\n        return result\n    return wrapper"),
        ("context_manager", "Write a Python context manager class FileWriter that opens a file for writing on enter and closes it on exit.",
                            "class FileWriter:\n    def __init__(self, path):\n        self.path = path\n        self.f = None\n\n    def __enter__(self):\n        self.f = open(self.path, 'w', encoding='utf-8')\n        return self.f\n\n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if self.f:\n            self.f.close()\n        return False"),
    ]
    for name, prompt, code in classics:
        rows.append({"prompt": prompt, "response": code})
        rows.append({"prompt": prompt.replace("Write", "Implement"), "response": code})
        rows.append({"prompt": prompt.replace("Write", "Give me"),   "response": code})

    # Extended classics — dozens of common patterns.
    extras = [
        ("collatz",    "Write a Python function `collatz(n)` that returns the Collatz sequence starting at n.",
                       "def collatz(n):\n    seq = [n]\n    while n != 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n        seq.append(n)\n    return seq"),
        ("power_of_two","Write a Python function `is_power_of_two(n)` that returns True iff n is a positive power of 2.",
                       "def is_power_of_two(n):\n    return n > 0 and (n & (n - 1)) == 0"),
        ("lcm",        "Write a Python function `lcm(a, b)` returning the least common multiple of two integers.",
                       "def lcm(a, b):\n    from math import gcd\n    return a * b // gcd(a, b)"),
        ("digits_sum", "Write a Python function `digit_sum(n)` returning the sum of decimal digits of n.",
                       "def digit_sum(n):\n    n = abs(n)\n    total = 0\n    while n > 0:\n        total += n % 10\n        n //= 10\n    return total"),
        ("reverse_int","Write a Python function `reverse_int(n)` that reverses the digits of an integer.",
                       "def reverse_int(n):\n    sign = -1 if n < 0 else 1\n    n = abs(n)\n    out = 0\n    while n > 0:\n        out = out * 10 + n % 10\n        n //= 10\n    return sign * out"),
        ("is_armstrong", "Write a Python function `is_armstrong(n)` that returns True iff n equals the sum of its digits each raised to the power of the digit count.",
                         "def is_armstrong(n):\n    s = str(n)\n    k = len(s)\n    return sum(int(d) ** k for d in s) == n"),
        ("hamming_distance", "Write a Python function `hamming(a, b)` returning the number of positions at which two equal-length strings differ.",
                             "def hamming(a, b):\n    if len(a) != len(b):\n        raise ValueError('strings must be equal length')\n    return sum(x != y for x, y in zip(a, b))"),
        ("levenshtein", "Write a Python function `levenshtein(a, b)` returning the edit distance between two strings.",
                        "def levenshtein(a, b):\n    if not a:\n        return len(b)\n    if not b:\n        return len(a)\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        cur = [i] + [0] * len(b)\n        for j, cb in enumerate(b, 1):\n            cost = 0 if ca == cb else 1\n            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + cost)\n        prev = cur\n    return prev[-1]"),
        ("caesar",     "Write a Python function `caesar(s, shift)` that performs a Caesar cipher on letters.",
                       "def caesar(s, shift):\n    out = []\n    for c in s:\n        if c.isupper():\n            out.append(chr((ord(c) - 65 + shift) % 26 + 65))\n        elif c.islower():\n            out.append(chr((ord(c) - 97 + shift) % 26 + 97))\n        else:\n            out.append(c)\n    return ''.join(out)"),
        ("run_length", "Write a Python function `rle(s)` that performs run-length encoding on a string.",
                       "def rle(s):\n    if not s:\n        return ''\n    out = []\n    prev = s[0]\n    count = 1\n    for c in s[1:]:\n        if c == prev:\n            count += 1\n        else:\n            out.append(f'{prev}{count}')\n            prev = c\n            count = 1\n    out.append(f'{prev}{count}')\n    return ''.join(out)"),
        ("matrix_transpose", "Write a Python function `transpose(m)` that returns the transpose of a 2D list.",
                             "def transpose(m):\n    if not m:\n        return []\n    return [[row[i] for row in m] for i in range(len(m[0]))]"),
        ("matrix_multiply",  "Write a Python function `matmul(a, b)` that returns the matrix product of two 2D lists.",
                             "def matmul(a, b):\n    rows_a, cols_a = len(a), len(a[0])\n    rows_b, cols_b = len(b), len(b[0])\n    if cols_a != rows_b:\n        raise ValueError('incompatible shapes')\n    out = [[0] * cols_b for _ in range(rows_a)]\n    for i in range(rows_a):\n        for j in range(cols_b):\n            for k in range(cols_a):\n                out[i][j] += a[i][k] * b[k][j]\n    return out"),
        ("dot_product", "Write a Python function `dot(a, b)` returning the dot product of two equal-length lists.",
                        "def dot(a, b):\n    if len(a) != len(b):\n        raise ValueError('length mismatch')\n    return sum(x * y for x, y in zip(a, b))"),
        ("euclid_dist", "Write a Python function `euclid(p, q)` returning the Euclidean distance between two points represented as tuples.",
                        "def euclid(p, q):\n    return sum((a - b) ** 2 for a, b in zip(p, q)) ** 0.5"),
        ("manhattan_dist","Write a Python function `manhattan(p, q)` returning the Manhattan distance between two points.",
                          "def manhattan(p, q):\n    return sum(abs(a - b) for a, b in zip(p, q))"),
        ("group_by_key","Write a Python function `group_by(items, key)` returning a dict of key → list of items.",
                        "def group_by(items, key):\n    out = {}\n    for it in items:\n        k = key(it)\n        out.setdefault(k, []).append(it)\n    return out"),
        ("chunk",       "Write a Python function `chunk(lst, n)` that splits a list into chunks of size n.",
                        "def chunk(lst, n):\n    return [lst[i:i+n] for i in range(0, len(lst), n)]"),
        ("rotate_list", "Write a Python function `rotate(lst, k)` that rotates a list k positions to the right.",
                        "def rotate(lst, k):\n    if not lst:\n        return lst\n    k = k % len(lst)\n    return lst[-k:] + lst[:-k]"),
        ("running_total","Write a Python function `running_total(xs)` returning a list of cumulative sums.",
                          "def running_total(xs):\n    total = 0\n    out = []\n    for x in xs:\n        total += x\n        out.append(total)\n    return out"),
        ("moving_average","Write a Python function `moving_average(xs, w)` returning a simple moving average with window w.",
                           "def moving_average(xs, w):\n    if w <= 0 or len(xs) < w:\n        return []\n    out = []\n    s = sum(xs[:w])\n    out.append(s / w)\n    for i in range(w, len(xs)):\n        s += xs[i] - xs[i - w]\n        out.append(s / w)\n    return out"),
        ("counter_top_n","Write a Python function `top_n(text, n)` returning the n most frequent words in text.",
                          "def top_n(text, n):\n    from collections import Counter\n    return Counter(text.split()).most_common(n)"),
        ("set_jaccard", "Write a Python function `jaccard(a, b)` returning the Jaccard similarity of two sets.",
                        "def jaccard(a, b):\n    if not a and not b:\n        return 1.0\n    inter = len(a & b)\n    union = len(a | b)\n    return inter / union if union else 0.0"),
        ("levenshtein_threshold","Write a Python function `within_edit_distance(a, b, k)` that returns True iff Levenshtein(a, b) <= k.",
                                  "def within_edit_distance(a, b, k):\n    if abs(len(a) - len(b)) > k:\n        return False\n    prev = list(range(len(b) + 1))\n    for i, ca in enumerate(a, 1):\n        cur = [i] + [0] * len(b)\n        for j, cb in enumerate(b, 1):\n            cost = 0 if ca == cb else 1\n            cur[j] = min(cur[j-1] + 1, prev[j] + 1, prev[j-1] + cost)\n        prev = cur\n        if min(prev) > k:\n            return False\n    return prev[-1] <= k"),
        ("rabin_karp",  "Write a Python function `rabin_karp(text, pattern)` returning the index of pattern in text, or -1.",
                        "def rabin_karp(text, pattern):\n    n, m = len(text), len(pattern)\n    if m == 0 or m > n:\n        return 0 if m == 0 else -1\n    base = 256\n    mod = 1_000_000_007\n    ph = th = 0\n    high = pow(base, m - 1, mod)\n    for i in range(m):\n        ph = (ph * base + ord(pattern[i])) % mod\n        th = (th * base + ord(text[i])) % mod\n    for i in range(n - m + 1):\n        if ph == th and text[i:i+m] == pattern:\n            return i\n        if i < n - m:\n            th = ((th - ord(text[i]) * high) * base + ord(text[i+m])) % mod\n            th %= mod\n    return -1"),
        ("quicksort",   "Write a Python function `quicksort(xs)` that returns a sorted list using quicksort.",
                        "def quicksort(xs):\n    if len(xs) <= 1:\n        return xs\n    pivot = xs[len(xs) // 2]\n    left  = [x for x in xs if x <  pivot]\n    mid   = [x for x in xs if x == pivot]\n    right = [x for x in xs if x >  pivot]\n    return quicksort(left) + mid + quicksort(right)"),
        ("heapsort",    "Write a Python function `heapsort(xs)` that returns a sorted copy using heapq.",
                        "import heapq\n\ndef heapsort(xs):\n    h = list(xs)\n    heapq.heapify(h)\n    return [heapq.heappop(h) for _ in range(len(h))]"),
        ("insertion_sort", "Write a Python function `insertion_sort(xs)` that sorts a list in place using insertion sort.",
                           "def insertion_sort(xs):\n    for i in range(1, len(xs)):\n        v = xs[i]\n        j = i - 1\n        while j >= 0 and xs[j] > v:\n            xs[j+1] = xs[j]\n            j -= 1\n        xs[j+1] = v\n    return xs"),
        ("selection_sort", "Write a Python function `selection_sort(xs)` that sorts a list in place using selection sort.",
                           "def selection_sort(xs):\n    n = len(xs)\n    for i in range(n):\n        m = i\n        for j in range(i + 1, n):\n            if xs[j] < xs[m]:\n                m = j\n        xs[i], xs[m] = xs[m], xs[i]\n    return xs"),
        ("dfs",         "Write a Python function `dfs(graph, start)` that returns nodes visited in DFS order.",
                        "def dfs(graph, start):\n    visited = []\n    stack = [start]\n    seen = set()\n    while stack:\n        n = stack.pop()\n        if n in seen:\n            continue\n        seen.add(n)\n        visited.append(n)\n        for nbr in reversed(graph.get(n, [])):\n            if nbr not in seen:\n                stack.append(nbr)\n    return visited"),
        ("bfs",         "Write a Python function `bfs(graph, start)` that returns nodes visited in BFS order.",
                        "def bfs(graph, start):\n    from collections import deque\n    visited = []\n    seen = {start}\n    q = deque([start])\n    while q:\n        n = q.popleft()\n        visited.append(n)\n        for nbr in graph.get(n, []):\n            if nbr not in seen:\n                seen.add(nbr)\n                q.append(nbr)\n    return visited"),
        ("dijkstra",    "Write a Python function `dijkstra(graph, start)` for a graph of {node: [(neighbour, weight), ...]} returning shortest distances.",
                        "import heapq\n\ndef dijkstra(graph, start):\n    dist = {start: 0}\n    pq = [(0, start)]\n    while pq:\n        d, n = heapq.heappop(pq)\n        if d > dist.get(n, float('inf')):\n            continue\n        for nbr, w in graph.get(n, []):\n            nd = d + w\n            if nd < dist.get(nbr, float('inf')):\n                dist[nbr] = nd\n                heapq.heappush(pq, (nd, nbr))\n    return dist"),
        ("topological_sort", "Write a Python function `topo_sort(graph)` returning a topological ordering of a DAG.",
                             "def topo_sort(graph):\n    indeg = {n: 0 for n in graph}\n    for n in graph:\n        for nbr in graph[n]:\n            indeg[nbr] = indeg.get(nbr, 0) + 1\n    from collections import deque\n    q = deque([n for n, d in indeg.items() if d == 0])\n    out = []\n    while q:\n        n = q.popleft()\n        out.append(n)\n        for nbr in graph.get(n, []):\n            indeg[nbr] -= 1\n            if indeg[nbr] == 0:\n                q.append(nbr)\n    return out if len(out) == len(indeg) else []"),
        ("knapsack",    "Write a Python function `knapsack(weights, values, capacity)` returning the maximum value achievable.",
                        "def knapsack(weights, values, capacity):\n    n = len(weights)\n    dp = [[0] * (capacity + 1) for _ in range(n + 1)]\n    for i in range(1, n + 1):\n        for w in range(capacity + 1):\n            if weights[i-1] <= w:\n                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])\n            else:\n                dp[i][w] = dp[i-1][w]\n    return dp[n][capacity]"),
        ("longest_common_subseq","Write a Python function `lcs(a, b)` returning the length of the longest common subsequence.",
                                  "def lcs(a, b):\n    n, m = len(a), len(b)\n    dp = [[0] * (m + 1) for _ in range(n + 1)]\n    for i in range(n):\n        for j in range(m):\n            if a[i] == b[j]:\n                dp[i+1][j+1] = dp[i][j] + 1\n            else:\n                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])\n    return dp[n][m]"),
        ("longest_palindrome","Write a Python function `longest_palindrome(s)` returning the longest palindromic substring.",
                               "def longest_palindrome(s):\n    if not s:\n        return ''\n    start = end = 0\n    for i in range(len(s)):\n        for l, r in ((i, i), (i, i + 1)):\n            while l >= 0 and r < len(s) and s[l] == s[r]:\n                if r - l > end - start:\n                    start, end = l, r\n                l -= 1\n                r += 1\n    return s[start:end+1]"),
        ("trie",        "Write a Python class Trie supporting insert(word) and contains(word).",
                        "class Trie:\n    def __init__(self):\n        self.root = {}\n\n    def insert(self, word):\n        node = self.root\n        for c in word:\n            node = node.setdefault(c, {})\n        node['#'] = True\n\n    def contains(self, word):\n        node = self.root\n        for c in word:\n            if c not in node:\n                return False\n            node = node[c]\n        return '#' in node"),
        ("union_find",  "Write a Python class UnionFind with find and union operations.",
                        "class UnionFind:\n    def __init__(self, n):\n        self.parent = list(range(n))\n        self.rank = [0] * n\n\n    def find(self, x):\n        while self.parent[x] != x:\n            self.parent[x] = self.parent[self.parent[x]]\n            x = self.parent[x]\n        return x\n\n    def union(self, a, b):\n        ra, rb = self.find(a), self.find(b)\n        if ra == rb:\n            return False\n        if self.rank[ra] < self.rank[rb]:\n            ra, rb = rb, ra\n        self.parent[rb] = ra\n        if self.rank[ra] == self.rank[rb]:\n            self.rank[ra] += 1\n        return True"),
        ("lru_cache",   "Write a Python class LRUCache(capacity) with get and put operations in O(1) amortised.",
                        "from collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.cap = capacity\n\n    def get(self, key):\n        if key not in self.cache:\n            return None\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.cap:\n            self.cache.popitem(last=False)"),
        ("threadsafe_counter","Write a Python class Counter that is threadsafe using threading.Lock.",
                              "import threading\n\nclass SafeCounter:\n    def __init__(self):\n        self.lock = threading.Lock()\n        self.n = 0\n\n    def inc(self):\n        with self.lock:\n            self.n += 1\n\n    def value(self):\n        with self.lock:\n            return self.n"),
        ("retry_decorator", "Write a Python decorator @retry(times) that retries a function up to `times` times on exception.",
                            "def retry(times):\n    def deco(fn):\n        def wrapper(*args, **kwargs):\n            last_exc = None\n            for _ in range(times):\n                try:\n                    return fn(*args, **kwargs)\n                except Exception as e:\n                    last_exc = e\n            raise last_exc\n        return wrapper\n    return deco"),
        ("singleton",   "Write a Python decorator @singleton that turns a class into a singleton.",
                        "def singleton(cls):\n    instances = {}\n    def get():\n        if cls not in instances:\n            instances[cls] = cls()\n        return instances[cls]\n    return get"),
        ("argparse_cli","Write a Python script that parses a --name string and --count integer from the command line and prints them.",
                        "import argparse\n\ndef main():\n    p = argparse.ArgumentParser()\n    p.add_argument('--name', required=True)\n    p.add_argument('--count', type=int, default=1)\n    args = p.parse_args()\n    for _ in range(args.count):\n        print(args.name)\n\nif __name__ == '__main__':\n    main()"),
        ("subprocess_run","Write a Python function `run_cmd(cmd)` that runs a shell command via subprocess and returns (returncode, stdout, stderr).",
                           "import subprocess\n\ndef run_cmd(cmd):\n    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)\n    return r.returncode, r.stdout, r.stderr"),
        ("env_get",     "Write a Python function `env_or(name, default)` that returns os.environ[name] or default.",
                        "import os\n\ndef env_or(name, default):\n    return os.environ.get(name, default)"),
        ("path_walk",   "Write a Python function `find_all(path, suffix)` returning every file under path ending in suffix.",
                        "import os\n\ndef find_all(path, suffix):\n    out = []\n    for root, _, files in os.walk(path):\n        for f in files:\n            if f.endswith(suffix):\n                out.append(os.path.join(root, f))\n    return out"),
        ("csv_read",    "Write a Python function `read_csv(path)` returning a list of dicts using csv.DictReader.",
                        "import csv\n\ndef read_csv(path):\n    with open(path, newline='', encoding='utf-8') as f:\n        return list(csv.DictReader(f))"),
        ("csv_write",   "Write a Python function `write_csv(path, rows)` that writes a list of dicts to CSV.",
                        "import csv\n\ndef write_csv(path, rows):\n    if not rows:\n        return\n    with open(path, 'w', newline='', encoding='utf-8') as f:\n        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))\n        w.writeheader()\n        for r in rows:\n            w.writerow(r)"),
        ("yaml_read",   "Write a Python function `read_yaml(path)` returning the parsed YAML object.",
                        "import yaml\n\ndef read_yaml(path):\n    with open(path, 'r', encoding='utf-8') as f:\n        return yaml.safe_load(f)"),
        ("sqlite_open", "Write a Python function `connect_sqlite(path)` that returns a SQLite connection with row_factory set to sqlite3.Row.",
                        "import sqlite3\n\ndef connect_sqlite(path):\n    conn = sqlite3.connect(path)\n    conn.row_factory = sqlite3.Row\n    return conn"),
        ("flask_hello", "Write a Python Flask app with a single GET / route returning JSON {'ok': True}.",
                        "from flask import Flask, jsonify\n\napp = Flask(__name__)\n\n@app.get('/')\ndef root():\n    return jsonify({'ok': True})\n\nif __name__ == '__main__':\n    app.run()"),
        ("fastapi_hello","Write a Python FastAPI app with a single GET / route returning {'ok': True}.",
                         "from fastapi import FastAPI\n\napp = FastAPI()\n\n@app.get('/')\ndef root():\n    return {'ok': True}"),
        ("threadpool",  "Write a Python function that runs a list of tasks in a ThreadPoolExecutor and returns their results.",
                        "from concurrent.futures import ThreadPoolExecutor\n\ndef run_parallel(fn, items, workers=8):\n    with ThreadPoolExecutor(max_workers=workers) as ex:\n        return list(ex.map(fn, items))"),
        ("async_sleep_all","Write a Python asyncio function that sleeps in parallel for each duration in a list.",
                            "import asyncio\n\nasync def sleep_all(durations):\n    await asyncio.gather(*(asyncio.sleep(d) for d in durations))"),
        ("dataclass_user","Write a Python dataclass User with id, name, email fields and a method to_dict.",
                          "from dataclasses import dataclass, asdict\n\n@dataclass\nclass User:\n    id: int\n    name: str\n    email: str\n\n    def to_dict(self):\n        return asdict(self)"),
        ("enum_color",  "Write a Python Enum Color with RED, GREEN, BLUE members.",
                        "from enum import Enum\n\nclass Color(Enum):\n    RED = 'red'\n    GREEN = 'green'\n    BLUE = 'blue'"),
        ("namedtuple",  "Write a Python namedtuple Point with x and y fields and use it.",
                        "from collections import namedtuple\n\nPoint = namedtuple('Point', ['x', 'y'])"),
        ("regex_emails","Write a Python function `extract_emails(text)` that returns a list of email addresses found via regex.",
                        "import re\n\ndef extract_emails(text):\n    return re.findall(r\"[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,}\", text)"),
        ("regex_phone", "Write a Python function `extract_phones(text)` that returns US-style phone numbers found in text.",
                        "import re\n\ndef extract_phones(text):\n    return re.findall(r\"\\(?\\d{3}\\)?[\\s\\-]?\\d{3}[\\s\\-]?\\d{4}\", text)"),
        ("hash_sha256", "Write a Python function `sha256_of(path)` returning the SHA-256 hex digest of a file.",
                        "import hashlib\n\ndef sha256_of(path):\n    h = hashlib.sha256()\n    with open(path, 'rb') as f:\n        for chunk in iter(lambda: f.read(8192), b''):\n            h.update(chunk)\n    return h.hexdigest()"),
        ("uuid4",       "Write a Python function `new_id()` returning a string UUID4.",
                        "import uuid\n\ndef new_id():\n    return str(uuid.uuid4())"),
        ("retry_http",  "Write a Python function `get_with_retry(url, attempts=3)` that retries a GET with exponential backoff.",
                        "import time\nimport requests\n\ndef get_with_retry(url, attempts=3):\n    delay = 1.0\n    last = None\n    for _ in range(attempts):\n        try:\n            r = requests.get(url, timeout=10)\n            r.raise_for_status()\n            return r.json()\n        except Exception as e:\n            last = e\n            time.sleep(delay)\n            delay *= 2\n    raise last"),
        ("logger_setup","Write a Python function `get_logger(name)` that returns a logger writing JSON-line records to stdout.",
                        "import json\nimport logging\nimport sys\n\nclass JsonFormatter(logging.Formatter):\n    def format(self, record):\n        return json.dumps({\n            'time':  self.formatTime(record),\n            'level': record.levelname,\n            'name':  record.name,\n            'msg':   record.getMessage(),\n        })\n\ndef get_logger(name):\n    log = logging.getLogger(name)\n    log.setLevel(logging.INFO)\n    h = logging.StreamHandler(sys.stdout)\n    h.setFormatter(JsonFormatter())\n    log.addHandler(h)\n    return log"),
        ("pytest_fixture", "Write a pytest fixture and a sample test that uses it.",
                           "import pytest\n\n@pytest.fixture\ndef sample_user():\n    return {'id': 1, 'name': 'test'}\n\ndef test_user_id(sample_user):\n    assert sample_user['id'] == 1"),
        ("contextlib_closing","Write a Python function that wraps a resource without a context manager using contextlib.closing.",
                              "from contextlib import closing\nimport sqlite3\n\ndef row_count(path, table):\n    with closing(sqlite3.connect(path)) as c:\n        return c.execute(f'select count(*) from {table}').fetchone()[0]"),
        ("ttl_cache",   "Write a Python decorator @ttl_cache(seconds) that caches a function's return value for `seconds` seconds.",
                        "import time\n\ndef ttl_cache(seconds):\n    def deco(fn):\n        cache = {}\n        def wrapper(*args):\n            now = time.time()\n            if args in cache and now - cache[args][0] < seconds:\n                return cache[args][1]\n            v = fn(*args)\n            cache[args] = (now, v)\n            return v\n        return wrapper\n    return deco"),
        ("rate_limit",  "Write a Python class RateLimiter(rate_per_sec) that uses token-bucket to gate calls.",
                        "import time\nimport threading\n\nclass RateLimiter:\n    def __init__(self, rate_per_sec):\n        self.rate = rate_per_sec\n        self.tokens = rate_per_sec\n        self.last = time.time()\n        self.lock = threading.Lock()\n\n    def acquire(self):\n        with self.lock:\n            now = time.time()\n            self.tokens = min(self.rate, self.tokens + (now - self.last) * self.rate)\n            self.last = now\n            while self.tokens < 1:\n                time.sleep(0.01)\n                now2 = time.time()\n                self.tokens += (now2 - self.last) * self.rate\n                self.last = now2\n            self.tokens -= 1"),
        ("send_email",  "Write a Python function `send_email(to, subject, body, smtp_host)` using smtplib.",
                        "import smtplib\nfrom email.mime.text import MIMEText\n\ndef send_email(to, subject, body, smtp_host='localhost'):\n    msg = MIMEText(body)\n    msg['Subject'] = subject\n    msg['From']    = 'noreply@example.com'\n    msg['To']      = to\n    with smtplib.SMTP(smtp_host) as s:\n        s.send_message(msg)"),
        ("xml_parse",   "Write a Python function `parse_xml(path)` returning the root ElementTree node.",
                        "import xml.etree.ElementTree as ET\n\ndef parse_xml(path):\n    return ET.parse(path).getroot()"),
        ("zip_extract", "Write a Python function `extract_zip(src, dest)` that extracts a zip archive.",
                        "import zipfile\n\ndef extract_zip(src, dest):\n    with zipfile.ZipFile(src) as z:\n        z.extractall(dest)"),
        ("gzip_compress","Write a Python function `gzip_file(src)` that gzips a file alongside it.",
                         "import gzip\nimport shutil\n\ndef gzip_file(src):\n    with open(src, 'rb') as f_in, gzip.open(src + '.gz', 'wb') as f_out:\n        shutil.copyfileobj(f_in, f_out)"),
        ("base64_encode","Write a Python function `b64encode_file(path)` returning the base64 string of a file's contents.",
                         "import base64\n\ndef b64encode_file(path):\n    with open(path, 'rb') as f:\n        return base64.b64encode(f.read()).decode('ascii')"),
        ("json_pretty", "Write a Python function `pretty(obj)` returning a pretty-printed JSON string of obj.",
                        "import json\n\ndef pretty(obj):\n    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)"),
        ("deep_get",    "Write a Python function `deep_get(obj, path, default=None)` that navigates a dotted path.",
                        "def deep_get(obj, path, default=None):\n    cur = obj\n    for part in path.split('.'):\n        if isinstance(cur, dict) and part in cur:\n            cur = cur[part]\n        else:\n            return default\n    return cur"),
        ("merge_dicts", "Write a Python function `deep_merge(a, b)` that returns a deep-merged dict.",
                        "def deep_merge(a, b):\n    out = dict(a)\n    for k, v in b.items():\n        if isinstance(v, dict) and isinstance(out.get(k), dict):\n            out[k] = deep_merge(out[k], v)\n        else:\n            out[k] = v\n    return out"),
        ("partition",   "Write a Python function `partition(xs, pred)` returning (matches, rest) based on a predicate.",
                        "def partition(xs, pred):\n    yes, no = [], []\n    for x in xs:\n        (yes if pred(x) else no).append(x)\n    return yes, no"),
        ("unique_by",   "Write a Python function `unique_by(xs, key)` keeping first occurrence by key.",
                        "def unique_by(xs, key):\n    seen = set()\n    out = []\n    for x in xs:\n        k = key(x)\n        if k not in seen:\n            seen.add(k)\n            out.append(x)\n    return out"),
        ("zip_longest", "Write a Python function `zip_longest_fill(a, b, fill=None)` similar to itertools.zip_longest.",
                        "def zip_longest_fill(a, b, fill=None):\n    out = []\n    for i in range(max(len(a), len(b))):\n        x = a[i] if i < len(a) else fill\n        y = b[i] if i < len(b) else fill\n        out.append((x, y))\n    return out"),
    ]
    for name, prompt, code in extras:
        rows.append({"prompt": prompt, "response": code})
        rows.append({"prompt": prompt.replace("Write", "Implement"), "response": code})
        rows.append({"prompt": prompt.replace("Write", "Give me"),   "response": code})
        rows.append({"prompt": prompt.replace("Write", "Show me"),   "response": code})
    return rows


# ── Code-gen Rust ────────────────────────────────────────────────────────

_RUST_ARITH = [
    ("add",  "(a: i32, b: i32) -> i32",  "a + b",        "return the sum of two i32s"),
    ("sub",  "(a: i32, b: i32) -> i32",  "a - b",        "return the difference of two i32s"),
    ("mul",  "(a: i32, b: i32) -> i32",  "a * b",        "return the product of two i32s"),
    ("max_of","(a: i32, b: i32) -> i32", "if a > b { a } else { b }", "return the larger of two i32s"),
    ("min_of","(a: i32, b: i32) -> i32", "if a < b { a } else { b }", "return the smaller of two i32s"),
    ("square","(n: i32) -> i32",         "n * n",        "return n squared"),
    ("cube",  "(n: i32) -> i32",         "n * n * n",    "return n cubed"),
    ("abs_i32","(n: i32) -> i32",        "if n >= 0 { n } else { -n }", "return the absolute value"),
    ("is_even","(n: i32) -> bool",       "n % 2 == 0",   "return true if n is even"),
    ("is_odd","(n: i32) -> bool",        "n % 2 != 0",   "return true if n is odd"),
    ("is_positive","(n: i32) -> bool",   "n > 0",        "return true if n is positive"),
    ("double","(n: i32) -> i32",         "n * 2",        "double a number"),
    ("triple","(n: i32) -> i32",         "n * 3",        "triple a number"),
    ("negate","(n: i32) -> i32",         "-n",           "negate a number"),
]


def gen_code_gen_rust() -> list[dict]:
    rows: list[dict] = []
    for name, sig, body, desc in _RUST_ARITH:
        prompts = [
            f"Write a Rust function `{name}{sig}` that {desc}.",
            f"In Rust, define {name}{sig} -- {desc}.",
            f"Rust function {name}{sig}: {desc}.",
            f"Give me a Rust function called {name} that takes the right arguments and {desc}.",
        ]
        response = f"fn {name}{sig} {{\n    return {body};\n}}"
        for p in prompts:
            rows.append({"prompt": p, "response": response})
    # Classics
    classics = [
        ("fib", "(n: u32) -> u64", "Write a Rust function fib(n: u32) -> u64 that returns the nth Fibonacci number.",
                "fn fib(n: u32) -> u64 {\n    let mut a: u64 = 0;\n    let mut b: u64 = 1;\n    for _ in 0..n {\n        let t = a + b;\n        a = b;\n        b = t;\n    }\n    return a;\n}"),
        ("factorial", "(n: u32) -> u64", "Write a Rust function factorial(n: u32) -> u64.",
                "fn factorial(n: u32) -> u64 {\n    let mut result: u64 = 1;\n    let mut i: u32 = 2;\n    while i <= n {\n        result = result * (i as u64);\n        i = i + 1;\n    }\n    return result;\n}"),
        ("gcd", "(a: u64, b: u64) -> u64", "Write a Rust gcd function using the Euclidean algorithm.",
                "fn gcd(a: u64, b: u64) -> u64 {\n    let mut x = a;\n    let mut y = b;\n    while y != 0 {\n        let t = y;\n        y = x % y;\n        x = t;\n    }\n    return x;\n}"),
        ("is_palindrome", "(s: &str) -> bool", "Write a Rust function is_palindrome that returns true if a string reads the same backwards.",
                "fn is_palindrome(s: &str) -> bool {\n    let chars: Vec<char> = s.chars().collect();\n    let n = chars.len();\n    let mut i = 0;\n    while i < n / 2 {\n        if chars[i] != chars[n - 1 - i] { return false; }\n        i = i + 1;\n    }\n    return true;\n}"),
        ("reverse_vec", "(v: &mut Vec<i32>)", "Write a Rust function that reverses a Vec<i32> in place.",
                "fn reverse_vec(v: &mut Vec<i32>) {\n    let n = v.len();\n    for i in 0..n / 2 {\n        v.swap(i, n - 1 - i);\n    }\n}"),
        ("sum_vec", "(v: &Vec<i32>) -> i32", "Write a Rust function that sums a Vec<i32>.",
                "fn sum_vec(v: &Vec<i32>) -> i32 {\n    let mut total: i32 = 0;\n    for x in v.iter() {\n        total = total + x;\n    }\n    return total;\n}"),
        ("max_vec", "(v: &Vec<i32>) -> i32", "Write a Rust function that returns the maximum of a Vec<i32>.",
                "fn max_vec(v: &Vec<i32>) -> i32 {\n    let mut best: i32 = v[0];\n    for x in v.iter() {\n        if *x > best { best = *x; }\n    }\n    return best;\n}"),
        ("count_in", "(v: &Vec<i32>, target: i32) -> usize", "Write a Rust function that counts occurrences of target in a vec.",
                "fn count_in(v: &Vec<i32>, target: i32) -> usize {\n    let mut n: usize = 0;\n    for x in v.iter() {\n        if *x == target { n = n + 1; }\n    }\n    return n;\n}"),
        ("read_file", "(path: &str) -> std::io::Result<String>", "Write a Rust function that reads a file to a String returning a Result.",
                "use std::fs;\nuse std::io;\n\nfn read_file(path: &str) -> io::Result<String> {\n    let s = fs::read_to_string(path)?;\n    return Ok(s);\n}"),
        ("write_file", "(path: &str, content: &str) -> std::io::Result<()>", "Write a Rust function that writes a string to a file.",
                "use std::fs;\nuse std::io;\n\nfn write_file(path: &str, content: &str) -> io::Result<()> {\n    fs::write(path, content)?;\n    return Ok(());\n}"),
        ("struct_point", "Point", "Define a Rust struct Point with x and y i32 fields and an impl block with a new() constructor.",
                "struct Point {\n    x: i32,\n    y: i32,\n}\n\nimpl Point {\n    fn new(x: i32, y: i32) -> Point {\n        Point { x, y }\n    }\n}"),
        ("enum_shape", "Shape", "Define a Rust enum Shape with Circle(f64) and Rect(f64,f64) variants and a function area.",
                "enum Shape {\n    Circle(f64),\n    Rect(f64, f64),\n}\n\nfn area(s: &Shape) -> f64 {\n    match s {\n        Shape::Circle(r) => 3.14159265 * r * r,\n        Shape::Rect(w, h) => w * h,\n    }\n}"),
    ]
    for name, sig, prompt, code in classics:
        rows.append({"prompt": prompt, "response": code})
        rows.append({"prompt": prompt.replace("Write a Rust", "Implement a Rust"), "response": code})
        rows.append({"prompt": prompt.replace("Write", "Give me"), "response": code})
    return rows


# ── Code-gen JavaScript ──────────────────────────────────────────────────

_JS_ARITH = [
    ("add", "a, b", "a + b", "return the sum of two numbers"),
    ("sub", "a, b", "a - b", "return the difference"),
    ("mul", "a, b", "a * b", "return the product"),
    ("div", "a, b", "b === 0 ? 0 : a / b", "return the quotient or 0 if dividing by zero"),
    ("max_of", "a, b", "a > b ? a : b", "return the larger of two numbers"),
    ("min_of", "a, b", "a < b ? a : b", "return the smaller of two numbers"),
    ("square", "n", "n * n", "return n squared"),
    ("cube", "n", "n * n * n", "return n cubed"),
    ("double", "n", "n * 2", "return n doubled"),
    ("isEven", "n", "n % 2 === 0", "return true if n is even"),
    ("isOdd",  "n", "n % 2 !== 0", "return true if n is odd"),
]


def gen_code_gen_javascript() -> list[dict]:
    rows: list[dict] = []
    for name, params, body, desc in _JS_ARITH:
        prompts = [
            f"Write a JavaScript function `{name}` that {desc}.",
            f"In JavaScript, write {name}({params}) that {desc}.",
            f"JavaScript: define a function {name} -- {desc}.",
            f"Give me a JavaScript function called {name} that {desc}.",
        ]
        response = f"function {name}({params}) {{\n    return {body};\n}}"
        for p in prompts:
            rows.append({"prompt": p, "response": response})
    classics = [
        ("doubleAll", "Write a JavaScript function that takes an array of numbers and returns a new array with each doubled.",
                      "function doubleAll(arr) {\n    return arr.map(x => x * 2);\n}"),
        ("filterEven", "Write a JavaScript function that returns only the even numbers from an array.",
                       "function filterEven(arr) {\n    return arr.filter(x => x % 2 === 0);\n}"),
        ("sumArr", "Write a JavaScript function that sums an array of numbers.",
                   "function sumArr(arr) {\n    return arr.reduce((a, b) => a + b, 0);\n}"),
        ("avgArr", "Write a JavaScript function that returns the average of an array.",
                   "function avgArr(arr) {\n    if (arr.length === 0) return 0;\n    return arr.reduce((a, b) => a + b, 0) / arr.length;\n}"),
        ("maxArr", "Write a JavaScript function that returns the maximum of an array.",
                   "function maxArr(arr) {\n    return Math.max(...arr);\n}"),
        ("flatten", "Write a JavaScript function flatten that flattens an array one level deep.",
                    "function flatten(arr) {\n    return arr.reduce((acc, x) => acc.concat(x), []);\n}"),
        ("dedupe", "Write a JavaScript function that removes duplicates from an array preserving order.",
                   "function dedupe(arr) {\n    const seen = new Set();\n    const out = [];\n    for (const x of arr) {\n        if (!seen.has(x)) {\n            seen.add(x);\n            out.push(x);\n        }\n    }\n    return out;\n}"),
        ("reverseStr", "Write a JavaScript function that reverses a string.",
                       "function reverseStr(s) {\n    return s.split('').reverse().join('');\n}"),
        ("isPalindrome", "Write a JavaScript function that returns true if a string is a palindrome.",
                         "function isPalindrome(s) {\n    return s === s.split('').reverse().join('');\n}"),
        ("wordFreq", "Write a JavaScript function that counts word frequencies in a string, returning an object.",
                     "function wordFreq(text) {\n    const out = {};\n    for (const w of text.split(/\\s+/)) {\n        if (!w) continue;\n        out[w] = (out[w] || 0) + 1;\n    }\n    return out;\n}"),
        ("getJson", "Write an async JavaScript function getJson(url) that fetches JSON and returns the parsed object.",
                    "async function getJson(url) {\n    const resp = await fetch(url);\n    return await resp.json();\n}"),
        ("postJson", "Write an async JavaScript function postJson(url, body) that POSTs JSON and returns the response JSON.",
                     "async function postJson(url, body) {\n    const resp = await fetch(url, {\n        method: 'POST',\n        headers: { 'Content-Type': 'application/json' },\n        body: JSON.stringify(body),\n    });\n    return await resp.json();\n}"),
        ("debounce", "Write a JavaScript debounce function debounce(fn, ms) that returns a debounced version of fn.",
                     "function debounce(fn, ms) {\n    let t = null;\n    return function (...args) {\n        if (t) clearTimeout(t);\n        t = setTimeout(() => fn(...args), ms);\n    };\n}"),
        ("throttle", "Write a JavaScript throttle function throttle(fn, ms).",
                     "function throttle(fn, ms) {\n    let last = 0;\n    return function (...args) {\n        const now = Date.now();\n        if (now - last >= ms) {\n            last = now;\n            return fn(...args);\n        }\n    };\n}"),
        ("memoize", "Write a JavaScript memoize function that caches the results of a unary function by argument.",
                    "function memoize(fn) {\n    const cache = new Map();\n    return function (x) {\n        if (cache.has(x)) return cache.get(x);\n        const v = fn(x);\n        cache.set(x, v);\n        return v;\n    };\n}"),
        ("classPoint", "Write a JavaScript class Point with x and y fields and a distance(other) method.",
                       "class Point {\n    constructor(x, y) {\n        this.x = x;\n        this.y = y;\n    }\n    distance(other) {\n        const dx = this.x - other.x;\n        const dy = this.y - other.y;\n        return Math.sqrt(dx * dx + dy * dy);\n    }\n}"),
        ("fib", "Write a JavaScript function fib(n) that returns the nth Fibonacci number using a loop.",
                "function fib(n) {\n    let a = 0, b = 1;\n    for (let i = 0; i < n; i++) {\n        const t = a + b;\n        a = b;\n        b = t;\n    }\n    return a;\n}"),
        ("factorial", "Write a JavaScript function factorial(n).",
                      "function factorial(n) {\n    let result = 1;\n    for (let i = 2; i <= n; i++) result *= i;\n    return result;\n}"),
    ]
    for name, prompt, code in classics:
        rows.append({"prompt": prompt, "response": code})
        rows.append({"prompt": prompt.replace("Write", "Implement"), "response": code})
        rows.append({"prompt": prompt.replace("Write", "Give me"),   "response": code})
    return rows


# ── Code-gen Bash ────────────────────────────────────────────────────────

def gen_code_gen_bash() -> list[dict]:
    rows: list[dict] = []
    pairs = [
        ("Write a bash one-liner that finds all .log files modified in the last day and greps them for the word ERROR.",
         "find . -name '*.log' -mtime -1 -exec grep -H ERROR {} \\;"),
        ("Write a bash one-liner that prints the disk usage of each subdirectory sorted by size.",
         "du -sh */ 2>/dev/null | sort -h"),
        ("Write a bash one-liner that prints the top 10 most-frequent words in $1.",
         "tr -s '[:space:]' '\\n' < \"$1\" | sort | uniq -c | sort -nr | head -n 10"),
        ("Write a bash script that retries curl https://example.com up to 3 times with a 5-second backoff.",
         "for i in 1 2 3; do\n    if curl -fsSL https://example.com; then\n        break\n    fi\n    sleep 5\ndone"),
        ("Write a bash function tag_release that creates a git tag matching the current ISO date.",
         "tag_release() {\n    local d\n    d=$(date +%Y-%m-%d)\n    git tag \"release-$d\"\n    git push origin \"release-$d\"\n}"),
        ("Write a bash one-liner that lists all files larger than 100M under /var.",
         "find /var -type f -size +100M -ls"),
        ("Write a bash script that watches a directory $1 for new files and prints each filename when one appears.",
         "while true; do\n    inotifywait -e create -q \"$1\" --format '%f'\ndone"),
        ("Write a bash one-liner that counts the number of lines in every .py file under the current directory.",
         "find . -name '*.py' -exec wc -l {} +"),
        ("Write a bash script that takes a directory argument and prints each file modified in the last hour.",
         "find \"$1\" -type f -mmin -60 -printf '%TY-%Tm-%Td %TH:%TM  %p\\n'"),
        ("Write a bash function gitclean that removes all branches merged into main except main.",
         "gitclean() {\n    git branch --merged main | grep -v '^\\*\\| main$' | xargs -r git branch -d\n}"),
        ("Write a bash one-liner that shows the 10 largest files under $1.",
         "find \"$1\" -type f -printf '%s %p\\n' | sort -nr | head -n 10"),
        ("Write a bash one-liner that prints the kernel version and uptime.",
         "echo \"$(uname -r) -- $(uptime -p)\""),
        ("Write a bash script that backs up a directory $1 to a timestamped tar.gz under /tmp.",
         "ts=$(date +%Y%m%d_%H%M%S)\ntar -czf \"/tmp/backup_$ts.tar.gz\" \"$1\""),
        ("Write a bash function http_status that takes a URL and prints just the HTTP status code returned.",
         "http_status() {\n    curl -s -o /dev/null -w '%{http_code}\\n' \"$1\"\n}"),
        ("Write a bash one-liner that finds duplicate lines in $1 sorted by occurrence count desc.",
         "sort \"$1\" | uniq -c | sort -nr"),
        ("Write a bash script that lists every process consuming more than 10% CPU.",
         "ps aux | awk '$3 > 10.0 { print }'"),
        ("Write a bash one-liner that removes empty directories recursively under $1.",
         "find \"$1\" -depth -type d -empty -delete"),
        ("Write a bash function svc_running that returns 0 if a systemd service $1 is active.",
         "svc_running() {\n    systemctl is-active --quiet \"$1\"\n}"),
        ("Write a bash one-liner that prints the public IP of the current host.",
         "curl -s https://ifconfig.me"),
        ("Write a bash one-liner that finds all .git directories under $1.",
         "find \"$1\" -type d -name '.git'"),
        ("Write a bash function port_in_use that returns 0 if a TCP port $1 is bound locally.",
         "port_in_use() {\n    ss -ltn \"sport = :$1\" | grep -q LISTEN\n}"),
        ("Write a bash one-liner that prints the total memory and free memory in MiB.",
         "free -m | awk '/Mem:/ {print \"total=\" $2 \"  free=\" $4}'"),
        ("Write a bash script that loops over arguments and echoes each one on its own line.",
         "for arg in \"$@\"; do\n    echo \"$arg\"\ndone"),
        ("Write a bash function safe_rm that asks for confirmation before deleting.",
         "safe_rm() {\n    read -p \"delete $1? [y/N] \" yn\n    if [ \"$yn\" = \"y\" ]; then rm -rf -- \"$1\"; fi\n}"),
    ]
    for prompt, code in pairs:
        rows.append({"prompt": prompt, "response": code})
        rows.append({"prompt": prompt.replace("Write", "Give me"),   "response": code})
        rows.append({"prompt": prompt.replace("Write", "Show me"),   "response": code})
    return rows


# ── Code-gen PowerShell ──────────────────────────────────────────────────

def gen_code_gen_powershell() -> list[dict]:
    rows: list[dict] = []
    pairs = [
        ("Write a PowerShell one-liner that lists all running services with the word 'sql' in their name.",
         "Get-Service | Where-Object { $_.Status -eq 'Running' -and $_.Name -like '*sql*' }"),
        ("Write a PowerShell function Get-PageStatus that invokes Invoke-WebRequest on a URL and returns the status code, handling errors with try/catch.",
         "function Get-PageStatus {\n    param([string]$Url)\n    try {\n        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -ErrorAction Stop\n        return $r.StatusCode\n    } catch {\n        return $_.Exception.Response.StatusCode.value__\n    }\n}"),
        ("Write a PowerShell command that finds files larger than 100MB under C:\\Users.",
         "Get-ChildItem -Path 'C:\\Users' -Recurse -File | Where-Object { $_.Length -gt 100MB }"),
        ("Write a PowerShell function Set-EnvVar that sets a user-scope environment variable persistently.",
         "function Set-EnvVar {\n    param([string]$Name, [string]$Value)\n    [Environment]::SetEnvironmentVariable($Name, $Value, 'User')\n}"),
        ("Write a PowerShell one-liner that prints the top 10 CPU-consuming processes.",
         "Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 Name, CPU, Id"),
        ("Write a PowerShell function Test-Port that returns true if a TCP port is open on a host.",
         "function Test-Port {\n    param([string]$Host, [int]$Port)\n    try {\n        $c = New-Object System.Net.Sockets.TcpClient\n        $c.Connect($Host, $Port)\n        $c.Close()\n        return $true\n    } catch {\n        return $false\n    }\n}"),
        ("Write a PowerShell command that lists Windows services that start automatically but are not running.",
         "Get-Service | Where-Object { $_.StartType -eq 'Automatic' -and $_.Status -ne 'Running' }"),
        ("Write a PowerShell function Stop-OldProcesses that kills processes named $Name running longer than $Hours hours.",
         "function Stop-OldProcesses {\n    param([string]$Name, [int]$Hours)\n    Get-Process -Name $Name -ErrorAction SilentlyContinue | Where-Object {\n        $_.StartTime -lt (Get-Date).AddHours(-$Hours)\n    } | Stop-Process -Force\n}"),
        ("Write a PowerShell one-liner that prints disk usage by drive in GB.",
         "Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{n='UsedGB';e={[math]::Round(($_.Used/1GB),1)}}, @{n='FreeGB';e={[math]::Round(($_.Free/1GB),1)}}"),
        ("Write a PowerShell command that exports an event log to CSV.",
         "Get-EventLog -LogName Application -Newest 1000 | Export-Csv -Path 'app.csv' -NoTypeInformation"),
        ("Write a PowerShell function Get-FileHashSimple that returns the SHA256 hash of a file.",
         "function Get-FileHashSimple {\n    param([string]$Path)\n    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash\n}"),
        ("Write a PowerShell one-liner that shows the last 20 PowerShell history entries.",
         "Get-History | Select-Object -Last 20"),
        ("Write a PowerShell function Install-Module-If-Missing that installs a module only if it is not already installed.",
         "function Install-ModuleIfMissing {\n    param([string]$Name)\n    if (-not (Get-Module -ListAvailable -Name $Name)) {\n        Install-Module -Name $Name -Force -Scope CurrentUser\n    }\n}"),
        ("Write a PowerShell one-liner that prints the list of installed hotfixes sorted by install date desc.",
         "Get-HotFix | Sort-Object InstalledOn -Descending"),
        ("Write a PowerShell function Wait-Url that polls a URL every 5 seconds until the response is 200, then returns.",
         "function Wait-Url {\n    param([string]$Url)\n    while ($true) {\n        try {\n            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -ErrorAction Stop\n            if ($r.StatusCode -eq 200) { return $true }\n        } catch {}\n        Start-Sleep -Seconds 5\n    }\n}"),
    ]
    for prompt, code in pairs:
        rows.append({"prompt": prompt, "response": code})
        rows.append({"prompt": prompt.replace("Write", "Implement"), "response": code})
        rows.append({"prompt": prompt.replace("Write", "Give me"),   "response": code})
    return rows


# ── Agent planning ───────────────────────────────────────────────────────

def gen_agent_planning() -> list[dict]:
    rows: list[dict] = []
    pairs = [
        ("Plan how to scaffold a Python project named 'foo' with a src directory, tests directory, and pyproject.toml.  Output numbered steps.",
         "1. mkdir foo && cd foo\n2. mkdir src tests\n3. touch src/__init__.py tests/__init__.py\n4. write pyproject.toml with project.name = 'foo'\n5. write a simple src/main.py\n6. write a sanity test in tests/test_basic.py\n7. run pytest -q to confirm the test passes"),
        ("A pytest just failed with AssertionError: expected 5, got 4.  Plan the steps to diagnose and fix it.  Output numbered steps.",
         "1. read the failing test to find which value is asserted\n2. read the function under test\n3. add a print or breakpoint to inspect the actual value\n4. identify the off-by-one or wrong-operator bug\n5. fix the code\n6. rerun pytest -q to confirm the test passes\n7. run the full suite to make sure no other tests broke"),
        ("Plan how to deploy a Django app behind waitress on Windows.  Output numbered steps.",
         "1. pip install waitress\n2. write run_waitress.py that imports the wsgi application and calls serve()\n3. test locally on a non-privileged port\n4. configure Windows firewall to allow the port\n5. create a startup script or Task Scheduler entry to launch run_waitress.py at boot\n6. run python manage.py collectstatic --noinput\n7. configure whitenoise middleware so waitress serves /static/"),
        ("Plan how to add a new endpoint /api/health to a Django app.  Output numbered steps.",
         "1. open the app's urls.py\n2. import a new view HealthView\n3. add path('api/health', HealthView.as_view())\n4. in views.py define HealthView(View) with a get() method returning JsonResponse({'ok': True})\n5. write a unit test in tests.py\n6. run python manage.py test\n7. curl http://localhost:8000/api/health to confirm"),
        ("Plan how to investigate a memory leak in a long-running Python process.  Output numbered steps.",
         "1. use the stdlib tracemalloc module to snapshot allocations\n2. snapshot at startup and again after the suspected leak window\n3. diff the two snapshots, sort by total size\n4. read the implicated code path; check for objects retained by globals, caches, or open file handles\n5. fix the retention (close files, bound the cache, drop the reference)\n6. rerun with the snapshot diff to verify the allocation no longer grows"),
        ("Plan how to add CI to a GitHub project using GitHub Actions.  Output numbered steps.",
         "1. mkdir -p .github/workflows\n2. create .github/workflows/ci.yml\n3. trigger on push and pull_request to main\n4. set up the language runtime (e.g. actions/setup-python@v5)\n5. install dependencies (pip install -e . or npm ci)\n6. run the test command (pytest -q or npm test)\n7. push and confirm the workflow runs green in the Actions tab"),
        ("Plan how to add a new feature flag to a service.  Output numbered steps.",
         "1. add the flag name to your configuration source of truth (env, settings.py, feature-flag service)\n2. wrap the new code path in a conditional that reads the flag\n3. default the flag to off so existing behavior is unchanged\n4. add a unit test that exercises both branches\n5. document the flag in the project README\n6. ship; flip the flag for a small percentage; observe; expand"),
        ("Plan how to migrate a SQLite database to Postgres.  Output numbered steps.",
         "1. provision a Postgres instance and create the target database\n2. dump the SQLite schema; adjust SQL dialect for Postgres (autoincrement, types)\n3. apply the adjusted schema to Postgres\n4. export each table from SQLite as CSV\n5. import each CSV into Postgres using COPY\n6. point the app's DATABASE_URL at Postgres in a staging environment\n7. run the full test suite + smoke test before flipping production"),
        ("Plan how to set up a development environment for a new Python project.  Output numbered steps.",
         "1. install pyenv or rely on the system python; pick a python version (e.g. 3.13)\n2. python -m venv .venv && source .venv/bin/activate\n3. pip install -e .[dev]\n4. configure pre-commit if the repo uses it\n5. run the tests to confirm a clean baseline\n6. open the project in your editor with the .venv selected"),
        ("Plan how to debug a 500 error from a Django view.  Output numbered steps.",
         "1. set DEBUG=True locally and reproduce the request\n2. read the traceback at the top of the response or in the runserver log\n3. identify the file:line raising the exception\n4. inspect the request data (params, body, user) at that frame\n5. fix the bug in the view or the underlying helper\n6. write a regression test that reproduces the bad request\n7. set DEBUG=False before deploying"),
        ("Plan how to add structured logging to a Python service.  Output numbered steps.",
         "1. add a JSON-formatter logging.Handler to logging configuration\n2. set the root logger level to INFO (DEBUG for dev)\n3. import logging in each module and getLogger(__name__)\n4. add context (request id, user id) via a LoggerAdapter or filter\n5. ship logs to stdout so a log collector (journald, loki, splunk) can pick them up\n6. document the field schema so log dashboards can be built"),
        ("Plan how to dockerise a Python web service.  Output numbered steps.",
         "1. write a Dockerfile based on python:3.13-slim\n2. COPY requirements.txt and run pip install -r requirements.txt\n3. COPY the project source\n4. set CMD to the production server (gunicorn or waitress)\n5. write a .dockerignore so .venv / .git aren't copied\n6. docker build -t myservice .\n7. docker run --rm -p 8000:8000 myservice and curl localhost:8000"),
        ("Plan how to set up nightly database backups for a Postgres deployment.  Output numbered steps.",
         "1. create a backup user with only the necessary read privileges\n2. write a script that runs pg_dump with a timestamped output filename\n3. compress the dump (e.g. pg_dump | gzip)\n4. upload to off-site storage (S3, B2, GCS)\n5. schedule the script with cron (or a managed scheduler)\n6. write a corresponding restore script and test it on staging once a quarter"),
        ("Plan how to add OAuth login to a web app.  Output numbered steps.",
         "1. register the application with the OAuth provider; obtain client_id and client_secret\n2. configure callback URLs in both the provider and the app config\n3. add a login route that redirects to the provider's authorize endpoint\n4. add a callback route that exchanges the code for an access token\n5. fetch the user's profile and create or look up the local user record\n6. set a secure HttpOnly session cookie\n7. test login, logout, and revocation paths"),
        ("Plan how to write a load test for a JSON API.  Output numbered steps.",
         "1. pick a load testing tool (k6, locust, vegeta)\n2. enumerate the endpoints and their typical request mix\n3. parameterise inputs from a CSV or generator\n4. start at low concurrency to establish a baseline latency\n5. ramp up until you observe degradation; record the breakpoint\n6. capture p50/p95/p99 latency and error rate at each level\n7. share results with the team and file follow-up tickets for the worst bottlenecks"),
        ("Plan how to enable HTTPS on a server using Let's Encrypt.  Output numbered steps.",
         "1. install certbot for your platform\n2. ensure port 80 is reachable from the public internet (for HTTP-01 challenge)\n3. run certbot certonly --webroot -w /var/www/html -d example.com\n4. configure the web server to use the issued cert + key paths\n5. redirect all HTTP to HTTPS\n6. schedule certbot renew via cron or systemd timer\n7. test with sslshopper or ssllabs.com to confirm the chain"),
        ("Plan how to onboard a new developer to the codebase.  Output numbered steps.",
         "1. give them access to the repo and any required services\n2. pair them with a buddy for the first week\n3. point them at the README, CONTRIBUTING, and an onboarding doc\n4. assign a small starter ticket they can ship end-to-end\n5. review their PR with extra care; explain conventions\n6. add their feedback to the onboarding doc so the next person has it easier"),
    ]
    extras = [
        ("Plan how to add type checking to an existing Python codebase.  Output numbered steps.",
         "1. pip install mypy (or pyright) in the dev dependencies\n2. create a mypy.ini or pyproject.toml [tool.mypy] section with sensible defaults\n3. start strict on a small module; gradually expand --strict to the whole package\n4. add type hints incrementally; use # type: ignore for known false positives\n5. wire mypy into CI so new code can't regress\n6. document the typing standards in CONTRIBUTING"),
        ("Plan how to migrate a Flask app to FastAPI.  Output numbered steps.",
         "1. inventory every route, blueprint, and middleware in the Flask app\n2. install FastAPI + Uvicorn; create a minimal FastAPI app side-by-side\n3. port routes one at a time; convert Flask request/response idioms to FastAPI types\n4. for each route, add pydantic models for request and response\n5. port middleware to FastAPI's middleware hooks or dependencies\n6. add equivalent tests for each ported route\n7. cut over by switching the WSGI/ASGI entrypoint; keep Flask around for rollback"),
        ("Plan how to add monitoring to a microservice.  Output numbered steps.",
         "1. emit Prometheus metrics for request count, latency histogram, and error count\n2. configure Prometheus to scrape the service's /metrics endpoint\n3. add health (/healthz) and readiness (/readyz) probes\n4. send structured JSON logs to stdout for the log collector\n5. instrument outgoing calls (DB, HTTP) with timing metrics\n6. build Grafana dashboards for the SLI metrics\n7. set alerts for SLO violations"),
        ("Plan how to refactor a 1000-line Python file into a package.  Output numbered steps.",
         "1. read through and group functions/classes by responsibility\n2. mkdir a new package directory with __init__.py\n3. move each coherent group into a sub-module\n4. update imports across the codebase to the new paths\n5. run the test suite to catch the inevitable misses\n6. iterate until the file is empty; delete it; commit the refactor as a single PR for review"),
        ("Plan how to handle long-running background jobs in a web app.  Output numbered steps.",
         "1. pick a job queue (Celery + Redis, RQ, or built-in async tasks for small loads)\n2. define the task functions in a tasks module\n3. update the web handler to enqueue work and return a job id immediately\n4. expose a status endpoint that reports queued/running/done/error\n5. configure the worker process and supervise it (systemd / kubernetes)\n6. add metrics for queue depth and job latency\n7. test failure paths (task raises, worker dies mid-job)"),
        ("Plan how to add internationalisation to a Django app.  Output numbered steps.",
         "1. enable USE_I18N=True and add LocaleMiddleware\n2. wrap translatable strings with gettext / gettext_lazy\n3. run python manage.py makemessages -l <code> for each locale\n4. translate the .po files; commit them\n5. compile with python manage.py compilemessages\n6. add a language picker in the UI; persist choice in user profile or cookie"),
        ("Plan how to write a parser for a small custom language.  Output numbered steps.",
         "1. define the grammar in BNF or PEG notation; sketch the AST node types\n2. write a tokenizer (lexer) that turns characters into typed tokens\n3. write a recursive-descent parser that consumes tokens and emits AST nodes\n4. write unit tests for each grammar rule\n5. add error messages that point at the offending source location\n6. add a pretty-printer that round-trips AST → source for tests"),
        ("Plan how to debug a flaky test.  Output numbered steps.",
         "1. reproduce the flake by running the test in a loop\n2. capture logs and timing for each failure\n3. identify the source of nondeterminism (clock, random, network, ordering)\n4. either remove the nondeterminism (seed the RNG, freeze time, mock the network) or make the assertion tolerant of it\n5. rerun in a loop to confirm the flake is gone\n6. add a note in the test or commit about what the flake was"),
        ("Plan how to upgrade a project to a new major version of a framework.  Output numbered steps.",
         "1. read the framework's migration guide for the target major version\n2. pin the new framework version and resolve dependency conflicts\n3. run the test suite; categorise failures by root cause\n4. fix each category systematically; commit per logical batch\n5. add tests that cover behaviours the migration guide highlighted\n6. deploy to staging; run smoke tests and observe metrics for 24 h\n7. promote to production with a rollback plan"),
        ("Plan how to introduce a new library to the codebase.  Output numbered steps.",
         "1. evaluate alternatives; document the choice and trade-offs\n2. add the library to the project's dependency manifest with a tight version bound\n3. add a thin wrapper module so the rest of the code doesn't depend on the library's surface directly\n4. write tests against the wrapper, mocking the library where appropriate\n5. ship the first concrete use case behind a feature flag\n6. roll it out; monitor for issues; expand usage"),
        ("Plan how to onboard a service to centralised logging.  Output numbered steps.",
         "1. pick a log shipper (filebeat, fluentbit, vector)\n2. configure the service to emit structured JSON to stdout\n3. install and configure the shipper to forward stdout to the central pipeline\n4. add log fields the central index expects (service, env, request_id, user_id)\n5. verify logs appear in the central system within ~10 s of emission\n6. build a dashboard with the common queries; share it with the team"),
        ("Plan how to debug a slow database query.  Output numbered steps.",
         "1. capture the query and a representative parameter set\n2. run EXPLAIN ANALYZE in the database to get the query plan\n3. look for sequential scans on large tables, missing or unused indexes, or expensive sorts\n4. add the missing index, or rewrite the query so the planner picks a better path\n5. measure the new latency; compare to the old\n6. commit the index migration; deploy; verify in production"),
        ("Plan how to add a feature flag to a service.  Output numbered steps.",
         "1. add the flag definition to your feature-flag source of truth (env, config, LaunchDarkly)\n2. wrap the new behaviour in a conditional that reads the flag\n3. default it off so existing behaviour is preserved\n4. cover both branches with tests\n5. deploy; flip the flag for a small slice; observe; expand\n6. once stable, remove the conditional and the flag"),
        ("Plan how to investigate a sudden traffic spike.  Output numbered steps.",
         "1. confirm the spike in metrics dashboards; identify which endpoint(s)\n2. cross-reference with deployment timeline and external announcements\n3. check upstream call patterns and user agents in the access logs\n4. correlate with downstream metrics (DB load, queue depth)\n5. if abusive, rate-limit; if organic, scale horizontally\n6. write a brief post-incident note for the team"),
        ("Plan how to break up a monolith into services.  Output numbered steps.",
         "1. inventory the monolith's bounded contexts; find natural seams\n2. pick the smallest, least-coupled context to extract first\n3. give it a clear API contract before extracting; let the monolith call it in-process\n4. when the contract is stable, extract the code to its own deployable\n5. switch the monolith to call it over the network\n6. repeat with the next context; keep the monolith deployable until the last piece is out"),
        ("Plan how to investigate a security incident.  Output numbered steps.",
         "1. preserve evidence: take snapshots of logs, disk, memory before changing anything\n2. determine scope: which systems are affected, which credentials may have been compromised\n3. contain: rotate credentials, isolate affected hosts, block ingress as needed\n4. eradicate: remove the attacker's foothold (web shells, scheduled tasks, persistence)\n5. recover: rebuild affected systems from known-good baselines\n6. post-mortem: document the timeline, root cause, and prevention plan"),
        ("Plan how to add an idempotency key to a write API.  Output numbered steps.",
         "1. require the client to send an Idempotency-Key header on POSTs\n2. on receipt, look up the key in a short-TTL store (e.g. Redis)\n3. if present, return the previously-computed response\n4. if absent, perform the work, store (key → response, status) for the TTL window, return the response\n5. handle concurrent retries by serialising on the key (lock or upsert)\n6. document the contract: same key + same body returns the same response"),
        ("Plan how to write an SDK for a third-party API.  Output numbered steps.",
         "1. read the API docs thoroughly; note auth, pagination, error codes, rate limits\n2. design the SDK surface around the user's mental model, not the wire model\n3. implement a thin transport layer (retries, auth header, timeouts)\n4. build typed resource classes on top of the transport\n5. write unit tests against recorded fixtures (vcr-style)\n6. write integration tests that hit a sandbox\n7. publish with semantic versioning; document migration when the upstream API changes"),
    ]
    pairs = pairs + extras
    for prompt, plan in pairs:
        rows.append({"prompt": prompt, "response": plan})
        rows.append({"prompt": prompt.replace("Plan how to", "Outline how to"), "response": plan})
        rows.append({"prompt": prompt.replace("Plan how to", "Walk me through"), "response": plan})
        rows.append({"prompt": prompt.replace("Plan how to", "Give me a step-by-step for"),
                       "response": plan})
    return rows


# ── Error diagnosis ──────────────────────────────────────────────────────

def gen_error_diagnosis() -> list[dict]:
    rows: list[dict] = []
    pairs = [
        ("I got: NameError: name 'x' is not defined. My code: print(x). What's wrong?",
         "You're using x before it's been assigned. Fix: assign x first, e.g. x = 5; print(x)."),
        ("I got: ModuleNotFoundError: No module named 'requests'. How do I fix it?",
         "Install the package with pip install requests, or pip install -r requirements.txt if it's listed."),
        ("I got: TypeError: 'int' object is not iterable. My code: for i in 5: pass.",
         "You can't iterate over an int. Use range(5) instead: for i in range(5): pass."),
        ("I got: IndentationError: expected an indented block.",
         "After a colon-ending line like if/def/for, the next line must be indented. Add 4 spaces (or a tab) before the body."),
        ("I got: SyntaxError: f-string: empty expression not allowed. My code: f'value: {}'.",
         "Empty braces in an f-string are not allowed. Put the expression inside: f'value: {x}'."),
        ("I got: PermissionError: [Errno 13] Permission denied. How do I fix it?",
         "The user running the script lacks permission for that path. Run with elevated privileges, change file ownership, or write to a directory you own."),
        ("I got: ConnectionRefusedError on port 8090.",
         "Nothing is listening on that port. Start the service that should bind to 8090, or correct the port in the client to match where the server is actually listening."),
        ("Rust compiler says: cannot borrow `*x` as mutable because it is also borrowed as immutable.",
         "You have an immutable borrow still live when you take the mutable one. End the immutable borrow first (let it go out of scope) before taking &mut x."),
        ("JavaScript: Uncaught TypeError: Cannot read property 'foo' of undefined.",
         "The object you're dereferencing is undefined. Add a guard (if (obj) obj.foo) or use optional chaining: obj?.foo."),
        ("I got: AttributeError: 'NoneType' object has no attribute 'split'.",
         "A function returned None and you treated it as a string. Check the function's return value; ensure it returns a string in every code path."),
        ("I got: KeyError: 'foo'.  My code: d['foo'].",
         "The dictionary doesn't contain that key. Use d.get('foo') to return None instead, or check 'foo' in d before accessing."),
        ("I got: IndexError: list index out of range.",
         "You're accessing an index that doesn't exist. Check len(list) before accessing arr[i]; or use a safer pattern like itertools.islice."),
        ("I got: ValueError: invalid literal for int() with base 10: 'abc'.",
         "You're passing a non-numeric string to int(). Validate or strip the input first; consider int(s) inside a try/except ValueError."),
        ("I got: ZeroDivisionError: division by zero.",
         "Add a guard: if b != 0: result = a / b else: result = 0  (or whatever the right semantics are)."),
        ("I got: RecursionError: maximum recursion depth exceeded.",
         "Your recursive function never reaches a base case, or the input is too deep. Add a base case, switch to iteration, or sys.setrecursionlimit(...) if appropriate."),
        ("I got: FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'.",
         "The file doesn't exist where the script is looking. Check the working directory (os.getcwd()) and adjust the path, or use an absolute path."),
        ("I got: UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff.",
         "The file isn't UTF-8. Open it with the right encoding (open(path, 'r', encoding='latin-1')) or with errors='replace' if you can tolerate lossy decode."),
        ("Rust: expected struct `String`, found `&str`.",
         "You're passing a string slice where an owned String is required. Call .to_string() or .to_owned() to convert."),
        ("Rust: cannot move out of borrowed content.",
         "You're trying to take ownership through a reference. Either clone (let owned = (*r).clone()) or change the function to take &T instead of T."),
        ("Rust: trait `MyTrait` is not implemented for `i32`.",
         "Either implement the trait for i32 (impl MyTrait for i32 { ... }) or pass a type that already implements it."),
        ("JavaScript: SyntaxError: Unexpected token 'export'.",
         "The file is being parsed as a script, not a module. Use 'type': 'module' in package.json, or rename to .mjs, or convert export to module.exports."),
        ("JavaScript: ReferenceError: fetch is not defined.",
         "fetch is available in browsers and modern Node (>=18). On older Node, install node-fetch and import it explicitly."),
        ("JavaScript: Promise rejected with no .catch.  Unhandled rejection.",
         "Attach a .catch(err => ...) to the promise chain, or wrap an await in try/catch when in an async function."),
        ("Bash: command not found.",
         "The binary isn't on PATH or isn't installed. which <cmd> to check; install the package or add the install dir to PATH."),
        ("Bash: bind: address already in use.",
         "Another process holds that port. ss -lntp or lsof -i :PORT to find it, then stop or kill that process before rebinding."),
        ("Bash: Permission denied on chmod.",
         "You're not the owner of the file or don't have the necessary capability. Use sudo, or check the file ownership with ls -l and adjust as needed."),
        ("git: refusing to merge unrelated histories.",
         "The two branches don't share history. If you're sure you want to merge them: git pull --allow-unrelated-histories or git merge other --allow-unrelated-histories."),
        ("git: fatal: not a git repository.",
         "You're not inside a git working tree. cd into the repo or git init to create one."),
        ("npm: ENOENT: no such file or directory, open 'package.json'.",
         "Run the npm command from the directory that contains package.json, or npm init -y to create one."),
        ("pip: error: subprocess-exited-with-error during build of <pkg>.",
         "A native extension failed to compile. Install the build tooling (apt install build-essential / xcode-select --install / VS Build Tools) and ensure the right development headers are present."),
        ("Postgres: relation 'foo' does not exist.",
         "The table isn't in the schema you're querying. Check the search_path, or fully qualify the name (schema.foo); make sure your migrations have actually run."),
        ("HTTP 401 Unauthorized from /api/...",
         "The request lacks valid credentials. Include an Authorization header with a valid token, or log in to obtain a session cookie first."),
        ("HTTP 403 Forbidden.",
         "You're authenticated but not allowed to access this resource. Check the user's role/permissions; the policy may need updating, or you need to use an admin account."),
        ("HTTP 404 Not Found.",
         "The URL doesn't match any route. Verify the path; check the trailing slash; confirm the service is the right one for that path."),
        ("HTTP 500 Internal Server Error.",
         "The server crashed handling your request. Read the server logs; reproduce locally with DEBUG on; fix the underlying exception."),
        ("HTTP 502 Bad Gateway.",
         "A reverse proxy in front of the app received a bad response from the upstream. Check the upstream service's logs; it may be down, restarting, or returning malformed data."),
        ("HTTP 504 Gateway Timeout.",
         "The reverse proxy didn't get a response in time. The upstream is slow or hung. Increase the proxy timeout for slow endpoints, or speed up the slow path."),
        ("HTTP 429 Too Many Requests.",
         "You've exceeded a rate limit. Slow your client down; respect the Retry-After header if present; consider batching or caching."),
    ]
    extra_pairs = [
        ("Python: OSError: [Errno 28] No space left on device.",
         "The disk holding the target path is full. Free up space (df -h to find it), point the script at a roomier filesystem, or rotate logs."),
        ("Python: BrokenPipeError when piping to head.",
         "Downstream (head) closed before you finished writing. Either ignore SIGPIPE, catch BrokenPipeError, or wrap output in contextlib.suppress(BrokenPipeError)."),
        ("Python: TypeError: unhashable type: 'list'.",
         "You used a list as a dict key or set element. Convert it to a tuple first: d[tuple(my_list)] = ..."),
        ("Python: TypeError: 'NoneType' object is not callable.",
         "You tried to call something that's actually None. A function returned None where you expected a function; check imports or rebinding."),
        ("Python: AssertionError without message.",
         "Add a message to the assert (assert cond, 'why this should hold') so future failures explain themselves."),
        ("Python: pip error: externally-managed-environment.",
         "Modern distros block system-wide pip installs. Use a virtualenv (python -m venv .venv && source .venv/bin/activate && pip install ...) or pipx for tools."),
        ("Python: UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192' in position N: character maps to <undefined>.",
         "Windows console can't print that arrow with cp1252.  Set PYTHONUTF8=1, or call sys.stdout.reconfigure(encoding='utf-8'), or replace the char with '->'."),
        ("Python: ImportError: attempted relative import with no known parent package.",
         "Run the module with python -m package.module, not python package/module.py.  The '-m' form makes Python recognise the parent package."),
        ("Python: dictionary changed size during iteration.",
         "You're mutating the dict while iterating it. Iterate over list(d.items()) instead, or build a new dict and assign it back at the end."),
        ("Rust: borrowed value does not live long enough.",
         "You held a reference past the lifetime of the value it points to. Either own the value (clone or move), restructure so the value outlives the reference, or use a 'static reference if it really should be global."),
        ("Rust: use of moved value.",
         "You moved a value into a function or another binding, then tried to use the original. Either clone before moving, or borrow with & instead of moving."),
        ("Rust: cannot find function `foo` in this scope.",
         "The function isn't imported or doesn't exist at that path. Check `use my_crate::foo;` or rename to the correct symbol."),
        ("Rust: mismatched types: expected `String`, found `&str`.",
         "Convert with .to_string() or .to_owned(), or change the signature to take &str if owning isn't needed."),
        ("Rust: panicked at 'index out of bounds'.",
         "Check the slice length before indexing, or use get(i) which returns Option instead of panicking."),
        ("Rust: panicked at 'unwrap on a None value'.",
         "Don't unwrap. Match on the Option, use ? to propagate, or default with .unwrap_or(default)."),
        ("Rust: error[E0382]: borrow of moved value.",
         "Same family as 'use of moved value'. Clone before moving, or take a reference (&x) rather than passing x by value."),
        ("Rust: error[E0277]: the trait `Send` is not implemented for `Rc<...>`.",
         "Rc isn't thread-safe. Switch to Arc<...> when the value crosses thread boundaries."),
        ("Rust: lifetime parameter not declared.",
         "Add the lifetime to the function signature: fn foo<'a>(x: &'a str) -> &'a str."),
        ("JS: Error: listen EADDRINUSE: address already in use :::3000.",
         "Another process is bound to port 3000.  lsof -i :3000 to find it, kill it, or pick a different PORT environment variable."),
        ("JS: TypeError: Cannot destructure property 'foo' of 'undefined' as it is undefined.",
         "The thing you're destructuring is undefined. Add a fallback: const { foo = 0 } = obj || {}; or guard with obj?."),
        ("JS: SyntaxError: await is only valid in async functions and the top level bodies of modules.",
         "Either mark the enclosing function async, or move the await into a top-level await in an ESM file."),
        ("JS: Error: Maximum call stack size exceeded.",
         "Your recursion has no base case or is too deep.  Convert to iteration, add a base case, or use a trampoline."),
        ("JS: ReferenceError: require is not defined in ES module scope.",
         "Files with .mjs or 'type': 'module' use import; either rename to .cjs, switch to import syntax, or change package.json type."),
        ("JS: Error: connect ECONNREFUSED 127.0.0.1:8080.",
         "Nothing is listening on 8080.  Start the upstream service, or correct the URL to where it actually listens."),
        ("npm: ERESOLVE unable to resolve dependency tree.",
         "A peer dependency conflict.  Either update versions to compatible ones, use --legacy-peer-deps as a temporary workaround, or upgrade to a npm version that resolves it."),
        ("yarn: error An unexpected error occurred: \"EACCES: permission denied\".",
         "yarn doesn't have permission to write to its cache or to node_modules.  Run as the directory owner, fix the cache directory permissions, or use a fresh project location you own."),
        ("Bash: line N: unexpected EOF while looking for matching `\"'.",
         "An unclosed double-quote in the script.  Search for the offending line and close the quote."),
        ("Bash: -bash: !\"#$: event not found.",
         "Bash history expansion (the !) interpreted your string.  Wrap the string in single quotes or escape the ! as \\!."),
        ("Bash: integer expression expected.",
         "[ -eq is comparing strings.  Use [[ $a -eq $b ]] (and quote inputs), or test [[ $a == $b ]] for string compare."),
        ("Bash: too many arguments.",
         "An unquoted variable expanded to multiple words.  Quote it: [[ \"$var\" == 'x' ]]."),
        ("Bash: jobs: command not found in a script.",
         "Job control doesn't work in non-interactive shells by default.  set -m to enable, or skip job-control commands in scripts."),
        ("Git: error: failed to push some refs; updates were rejected because the remote contains work.",
         "Pull first (git pull --rebase) to integrate the remote work, then push again.  Force-push only if you own that branch and understand the cost."),
        ("Git: fatal: refusing to merge unrelated histories.",
         "The branches don't share history.  git pull --allow-unrelated-histories if you really want to merge them.  Otherwise rebase or cherry-pick instead."),
        ("Git: warning: LF will be replaced by CRLF the next time Git touches it.",
         "Cosmetic, not an error.  Configure core.autocrlf to your preference (input on macOS/Linux, true on Windows), or add a .gitattributes."),
        ("Git: detached HEAD after checkout.",
         "You're on a commit, not a branch.  git switch -c new_branch to start a branch from here, or git switch existing_branch to go back."),
        ("Docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock.",
         "Docker daemon isn't running, or your user isn't in the docker group.  sudo systemctl start docker, then add yourself to the docker group and re-login."),
        ("Docker: OCI runtime exec failed: exec failed: no such file or directory.",
         "The CMD or ENTRYPOINT references a binary that doesn't exist in the image.  Verify the path inside the image; use docker run --entrypoint sh -ti image to poke around."),
        ("Docker: image not found.",
         "Either the image name is misspelled, or you haven't pulled it, or you haven't built it locally.  docker pull <name> or docker build -t <name> ."),
        ("Kubernetes: ImagePullBackOff for my pod.",
         "kubectl describe pod <name>.  The image name is wrong, the tag doesn't exist, or your registry credentials aren't configured.  Fix the image reference or imagePullSecret."),
        ("Kubernetes: CrashLoopBackOff for my pod.",
         "kubectl logs <pod> --previous to see why it crashed.  Usually a config issue or a missing dependency.  Fix the app, redeploy."),
        ("Postgres: FATAL: password authentication failed for user 'app'.",
         "Wrong password, wrong username, or pg_hba.conf is restricting that connection.  Verify credentials, check pg_hba for the right auth method."),
        ("Postgres: ERROR: deadlock detected.",
         "Two transactions are waiting on each other's locks.  Order your operations so locks are acquired in a consistent order across all code paths, or shorten transaction scope."),
        ("Postgres: ERROR: relation 'foo' does not exist.",
         "Schema mismatch: the table isn't in the search_path you're using.  Either qualify the name (schema.foo) or alter the role's search_path."),
        ("Redis: MISCONF Redis is configured to save RDB snapshots but it's currently unable to persist.",
         "Disk is full or the data directory isn't writable.  Free space, fix permissions, or temporarily run with stop-writes-on-bgsave-error no."),
        ("nginx: emerg: bind() to 0.0.0.0:80 failed (98: Address already in use).",
         "Another process holds port 80.  Stop it (apache2, another nginx), or change nginx's listen directive to a free port."),
        ("nginx: 502 Bad Gateway from upstream.",
         "Upstream is down, unreachable, or returning malformed responses.  curl the upstream URL from the nginx host to confirm; restart the upstream service."),
        ("Python: requests.exceptions.SSLError: HTTPSConnectionPool ... certificate verify failed.",
         "Either the server's cert is invalid, your CA bundle is stale, or there's a TLS-intercepting proxy.  Update certifi, point requests at the right CA bundle, or fix the upstream cert."),
        ("Python: requests.exceptions.ReadTimeout.",
         "The server didn't respond within the timeout.  Increase the timeout (requests.get(url, timeout=30)) only after confirming the upstream is healthy and the right URL."),
        ("Python: socket.gaierror: [Errno -2] Name or service not known.",
         "DNS lookup failed for that hostname.  Check the hostname, the system DNS config (/etc/resolv.conf), and your network."),
        ("Python: psycopg.OperationalError: could not connect to server.",
         "Verify the Postgres host/port are reachable from the client (telnet host 5432), the server is up, and the pg_hba.conf allows your client IP."),
        ("Python: AssertionError: Tensor.dim() must be 2 or 3.",
         "You passed the wrong shape into a torch operation.  Print x.shape to confirm and reshape / unsqueeze as needed."),
        ("Python: AttributeError: module 'cv2' has no attribute 'COLOR_BGR2RGB'.",
         "Likely an incomplete OpenCV install.  pip install opencv-python (the headless variant doesn't expose all enums in some builds)."),
        ("Python: 'NoneType' object has no attribute 'shape' from cv2.imread.",
         "cv2.imread returns None when the path is missing or unreadable.  Print and check the path; ensure the file actually exists."),
        ("Make: missing separator (tab expected).",
         "Make uses TAB characters to indent recipe lines.  Your editor inserted spaces.  Replace with a real tab."),
        ("Cargo: error: failed to run `rustc` to learn about target-specific information.",
         "Probably a corrupted rustup toolchain.  rustup default stable && rustup update."),
        ("Cargo: warning: unused import.",
         "Cosmetic.  Remove the import or, if you actually use it, the compiler isn't seeing the usage path; check #[cfg(test)] gates."),
        ("Docker compose: services.web.environment.X is not allowed.",
         "Your compose file is using a key the spec rejects.  Look up the correct key under environment in the docs (usually it's a list or map, not nested keys)."),
        ("PowerShell: Cannot bind argument to parameter 'Path' because it is an empty string.",
         "The variable you passed was empty.  Check the variable was actually set; consider [string]::IsNullOrEmpty($var) before calling."),
        ("PowerShell: The term 'curl' is not recognised as a name of a cmdlet.",
         "PowerShell aliases curl to Invoke-WebRequest by default but the alias may be removed.  Either use curl.exe explicitly, install curl, or use Invoke-WebRequest -Uri."),
    ]
    pairs = pairs + extra_pairs
    for prompt, fix in pairs:
        rows.append({"prompt": prompt, "response": fix})
        rows.append({"prompt": prompt + "  Walk me through the fix.", "response": fix})
        rows.append({"prompt": "Help me with this error: " + prompt, "response": fix})
        rows.append({"prompt": prompt + "  What does it mean and how do I solve it?", "response": fix})
    return rows


# ── Long-context summarisation ───────────────────────────────────────────

_SUMMARY_PASSAGES = [
    ("The W1z4rD V1510n node is a Hebbian neural fabric with multi-pool architecture. Pools represent modalities — keyboard text, image pixels, audio features — and cross-pool synapses bind concepts across modalities. Training is online: every observation updates synapse weights via dopamine-flush consolidation.",
     "W1z4rD is an online-learning Hebbian fabric with multi-pool architecture; cross-pool synapses bind concepts across modalities and dopamine-flush consolidation reinforces them with every observation."),
    ("Cross-pool synapses can be excitatory or inhibitory. Excitatory edges fire the target concept when the source concept activates. Inhibitory edges suppress the target. Contrastive Hebbian learning builds inhibitions to competitors: when training apple ↔ apple_image, every previously-existing target concept that didn't participate becomes an inhibitory target.",
     "Cross-pool synapses come in two signs: excitatory ones activate the target, inhibitory ones suppress it. Contrastive Hebbian training adds inhibitory edges from each new concept to existing competitor concepts."),
    ("Hierarchical concepts are concepts whose members are themselves concepts, not atoms. A parent like 'fruit' has children 'apple' and 'banana'. The parent fires when any child fires (bottom-up generalisation) and when the parent fires the children get top-down expectation activations.",
     "Hierarchical concepts have concepts as members instead of atoms. The parent fires bottom-up from any active child, and parent activation propagates back down as expectations for the children."),
    ("Temporal binding wires the previous observation's concepts to the current observation's concepts. Per-session caches keep prev_concepts per pool. A 6-hour idle TTL evicts unused sessions. Within-pool sequence edges and cross-pool temporal edges form the time-axis of the fabric.",
     "Temporal binding wires concepts from the previous observation in a session to the current observation, building within-pool sequence edges and cross-pool temporal links, with a 6-hour idle TTL."),
    ("The training runner uses TOML scripts in the registry. Each declares its inputs, benchmarks, and regression_protects references. The runner POSTs benchmarks against the node's /chat endpoint, scores them with keyword + AST checks, and emits JSONL events the UI's Live Training panel polls.",
     "Training scripts are TOML files declaring inputs, benchmarks, and regression protections. The runner probes /chat with each benchmark, scores responses, and writes JSONL events that the wizard-chat Live Training panel polls."),
    ("Negative-pair training is the explicit anti-example case. Polarity flag negative tells multi_pool_train_pair to store inhibitory cross-edges between the paired concepts instead of excitatory ones. The contrastive Hebbian pass is skipped because an anti-example doesn't define new category structure.",
     "Negative-pair training stores inhibitory edges between paired concepts and skips contrastive learning, encoding explicit anti-examples like 'X is not Y'."),
    ("Within-pool lateral inhibition wires inhibitory synapses between concept neurons inside the same pool so a winning concept actively suppresses its peers during propagation. This delivers winner-take-all behaviour at the concept layer.",
     "Within-pool lateral inhibition connects competing concept neurons with inhibitory synapses so the winning concept suppresses peers during propagation — concept-level winner-take-all."),
    ("The wizard-chat UI is a Vue 3 single-page app served by Django. It polls /api/wizard-chat/status/ for live brain state and /api/wizard-chat/training/live/ for training events. The Live Training panel below the chat renders pool tiles, an event stream, and a concept-graph sketch.",
     "The wizard-chat UI is a Vue 3 SPA on Django that polls status + training endpoints; the Live Training panel shows pool tiles, the event stream, and a concept-graph sketch below the chat thread."),
    ("Multi-pool decoding queries cross-pool edges from the source concept and unions targets that fire above the activation threshold. For hierarchical sources, the decoder bypasses within-pool propagation and reads tgt_seeds directly to surface multi-child predictions equally.",
     "Multi-pool decoding walks cross-pool edges from the active source concept. Hierarchical sources bypass within-pool propagation so each child contributes its atoms equally."),
    ("Audio playback in the wizard chat triggers a translucent canvas overlay driven by AudioContext.createAnalyser(). The overlay reads the FFT spectrum, modulates per-band luminance bars, and overlays a top-down halo on the strongest band — a phonetic light effect.",
     "Wizard-chat audio playback feeds a Web Audio AnalyserNode; the canvas overlay modulates per-band luminance and a top-down halo by the FFT spectrum for a phonetic light overlay."),
    ("Git is a distributed version-control system. Every clone of a repository is itself a full repository with the complete history. Branches are cheap labels on commits and merges either fast-forward or produce a merge commit recording both parents.",
     "Git is distributed: every clone holds the full history. Branches are cheap commit labels; merges fast-forward or record both parents."),
    ("HTTP is a stateless request-response protocol. The client sends a verb (GET, POST, PUT, DELETE), a path, headers, and an optional body. The server responds with a status code, headers, and an optional body. Cookies and sessions add the appearance of state.",
     "HTTP is stateless — the client sends verb + path + headers + optional body; the server returns status + headers + body. Cookies and sessions simulate state."),
    ("A relational database stores data in tables with rows and columns. Tables relate to each other via foreign keys. Transactions provide ACID guarantees: atomicity, consistency, isolation, and durability. Indexes speed up lookups at the cost of slower writes.",
     "Relational databases store data in linked tables. Transactions are ACID. Indexes trade write speed for faster lookups."),
    ("A Docker container is a process running inside an isolated namespace with its own filesystem, network stack, and process tree. Containers are built from images, which are layered filesystems described by a Dockerfile.",
     "A container is an isolated process with its own filesystem and network. Containers come from layered images built from a Dockerfile."),
    ("Kubernetes is a container orchestrator. Pods are the smallest deployable unit, containing one or more containers that share a network namespace. Deployments manage replica sets of pods. Services expose pods as network endpoints.",
     "Kubernetes schedules containers via pods; deployments manage replicas; services give them stable network endpoints."),
    ("A REST API exposes resources via HTTP verbs. GET reads, POST creates, PUT replaces, PATCH modifies, DELETE removes. URLs are nouns identifying resources. Responses are typically JSON bodies with appropriate status codes.",
     "REST APIs expose resources through HTTP verbs (GET/POST/PUT/PATCH/DELETE) on noun-shaped URLs, with JSON responses."),
    ("Asynchronous I/O lets a single thread handle many concurrent operations without blocking. The runtime maintains an event loop and a queue of pending tasks. When an awaited operation completes, the runtime resumes its task.",
     "Async I/O concurrency: one thread, an event loop, and a queue of tasks that yield while waiting and resume when their I/O completes."),
    ("A unit test isolates a small piece of behaviour and asserts it produces the expected output. Fixtures provide reusable setup. Mocks stand in for collaborators so the test depends only on the unit under test.",
     "Unit tests isolate behaviour with fixtures for setup and mocks for collaborators so each test exercises one thing."),
    ("A continuous-integration pipeline runs automated checks on every push. Typical stages include linting, type-checking, unit tests, integration tests, and security scans. If any stage fails the pipeline blocks the merge.",
     "CI runs lint, type-check, unit + integration tests, and security scans on every push; failures block merges."),
    ("Encryption protects confidentiality by transforming data with a key into ciphertext only the key holder can read. Symmetric encryption uses the same key for both operations; asymmetric encryption uses a public/private key pair, the basis of TLS.",
     "Encryption uses keys to hide data. Symmetric encryption shares one key; asymmetric uses public/private pairs and underpins TLS."),
    ("A hash function maps arbitrary input to a fixed-size digest in a one-way fashion. Good cryptographic hashes are infeasible to reverse and rarely collide. Use a salt with hashes for storing passwords so identical inputs don't share an output.",
     "Hash functions produce a fixed-size digest one-way. For passwords, add a salt so equal inputs don't share an output."),
    ("OAuth is an authorisation framework. The user grants the app permission via the identity provider; the app receives an access token rather than the password. Tokens have scopes and expire; refresh tokens obtain new ones.",
     "OAuth lets apps act on a user's behalf via tokens (with scopes + expiry) issued by an identity provider — never the password."),
    ("A neural network is a function from inputs to outputs parameterised by weights. Training adjusts the weights via gradient descent to reduce a loss measured against expected outputs. Layers stack to build representations.",
     "Neural networks parameterise input-to-output functions with weights; training updates weights via gradient descent on a loss."),
    ("Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule from outputs back through the network. Modern frameworks build a computation graph and differentiate it automatically.",
     "Backprop chain-rules gradients of the loss back through the network. Modern frameworks auto-differentiate a recorded graph."),
    ("Reinforcement learning trains an agent to maximise cumulative reward by acting in an environment. The agent observes states, picks actions, and receives rewards. Policy gradients and Q-learning are two common families.",
     "RL trains an agent to maximise reward by acting in an environment. Policy gradients and Q-learning are the main algorithm families."),
    ("Linear regression finds the line that minimises squared error from the data points. With multiple features it generalises to a hyperplane. Regularisation (L1, L2) discourages large weights and helps generalisation.",
     "Linear regression minimises squared error over the data; multi-feature variant fits a hyperplane; regularisation curbs weight magnitude."),
    ("A graph is a collection of nodes connected by edges. Edges can be directed or undirected, weighted or unweighted. Common operations are traversal (BFS, DFS), shortest path (Dijkstra), and connectivity (union-find).",
     "Graphs are nodes + edges (directed/weighted variants). Common ops: BFS/DFS traversal, Dijkstra for shortest path, union-find for connectivity."),
    ("A dynamic programming problem can be solved by storing the results of subproblems and reusing them. Classic examples include longest common subsequence, knapsack, and edit distance. The key is finding the right state and recurrence.",
     "DP stores subproblem results to avoid recomputation. Classic examples: LCS, knapsack, edit distance. Define the state and the recurrence."),
    ("A B-tree is a self-balancing tree that keeps data sorted and allows logarithmic insertion, deletion, and search. Database indexes typically use B-tree variants to make point and range queries fast.",
     "B-trees are balanced search trees with logarithmic ops; databases use B-tree variants for indexes that speed up point and range queries."),
    ("Cache invalidation is the problem of knowing when cached data is stale. Strategies include time-based TTL, event-driven invalidation when source data changes, and write-through caches that update both source and cache atomically.",
     "Cache invalidation is hard. Options: TTL, event-driven invalidation, or write-through that updates source + cache together."),
    ("Idempotency means an operation has the same effect whether it's applied once or many times. PUT and DELETE are idempotent in HTTP; POST is not. Idempotency keys let clients retry write operations safely.",
     "Idempotent operations produce the same effect no matter how many times you apply them. PUT/DELETE are idempotent; idempotency keys make POSTs safe to retry."),
    ("A race condition is a bug where the outcome depends on the unpredictable timing of concurrent operations. Locks, mutexes, and atomic operations prevent races by enforcing an ordering on critical sections.",
     "Race conditions: outcome depends on timing of concurrent ops. Prevent with locks, mutexes, or atomics around critical sections."),
    ("A deadlock occurs when two or more threads each hold a resource and wait for another that another thread holds. Prevent it by establishing a consistent global lock-acquisition order across all code paths.",
     "Deadlock: threads wait for resources each other holds. Avoid by ordering lock acquisitions consistently across the codebase."),
    ("A memory leak is allocated memory that is no longer reachable through the program's data structures but also has not been freed. In garbage-collected languages it usually manifests as a slow-growing data structure like a global cache.",
     "Memory leaks: allocated memory you can no longer reach but also haven't freed. In GC languages, growing global caches are a common cause."),
]


def gen_long_context_summary() -> list[dict]:
    rows: list[dict] = []
    for passage, summary in _SUMMARY_PASSAGES:
        prompts = [
            f"Summarise: {passage}",
            f"Summarize this: {passage}",
            f"Give me a one-paragraph summary of: {passage}",
            f"Briefly summarise the following: {passage}",
            f"What does this say in a sentence?  {passage}",
            f"TL;DR: {passage}",
        ]
        for p in prompts:
            rows.append({"prompt": p, "response": summary})
    # Mixed-paragraph longer prompts
    for p1, s1 in _SUMMARY_PASSAGES:
        for p2, s2 in _SUMMARY_PASSAGES:
            if p1 == p2:
                continue
            combined = f"{p1}  {p2}"
            summary  = f"{s1}  {s2}"
            rows.append({
                "prompt":   f"Summarise these two passages: {combined}",
                "response": summary,
            })
            # Keep the corpus bounded — about 100 paired examples here.
            if len(rows) > 800:
                return rows
    return rows


# ── Multimodal pairs (text only — image bytes are seeded separately) ────

def gen_multimodal_pairs() -> list[dict]:
    """The actual image+caption training runs through /sensor/observe with
    bytes_b64; this corpus is the text-side reinforcement so the
    keyboard_text concept for each label is well-formed before the image
    pair-up happens."""
    rows: list[dict] = []
    objects = ["apple", "banana", "ball", "leaf", "cup", "book", "orange",
                "pear", "grape", "carrot", "tomato", "potato", "lettuce",
                "lemon", "lime", "peach", "plum", "kiwi", "mango", "melon"]
    for obj in objects:
        for phr in _PHRASING:
            rows.append({
                "prompt":   phr.format(topic=f"a {obj}"),
                "response": f"A {obj} is a familiar object.  Visually it has a recognisable shape and colour.",
            })
        rows.append({
            "prompt":   f"What does a {obj} look like?",
            "response": f"A {obj} has a recognisable visible appearance — its colour, shape, and texture make it identifiable."
        })
        rows.append({
            "prompt":   f"Describe a {obj}.",
            "response": f"A {obj} is an object with a particular colour and shape that distinguishes it from other items."
        })
    return rows


# ── Driver ───────────────────────────────────────────────────────────────

_GENERATORS = {
    "conversation_basics_001":   gen_conversation_basics,
    "conversation_dialog_001":   gen_conversation_dialog,
    "code_gen_python_001":       gen_code_gen_python,
    "code_gen_rust_001":         gen_code_gen_rust,
    "code_gen_javascript_001":   gen_code_gen_javascript,
    "code_gen_bash_001":         gen_code_gen_bash,
    "code_gen_powershell_001":   gen_code_gen_powershell,
    "agent_planning_001":        gen_agent_planning,
    "error_diagnosis_001":       gen_error_diagnosis,
    "long_context_summary_001":  gen_long_context_summary,
    "multimodal_pairs_001":      gen_multimodal_pairs,
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", default="", help="generate only this script id")
    args = p.parse_args(argv)

    targets = [(k, v) for k, v in _GENERATORS.items()
                if not args.only or k == args.only]
    if not targets:
        print(f"no generator matches {args.only!r}", file=sys.stderr)
        return 2

    total = 0
    for script_id, gen in targets:
        rows = gen()
        # Light shuffle so consecutive observations don't all share a
        # template prefix (Hebbian fabric benefits from varied ordering).
        random.shuffle(rows)
        out = _DATA_DIR / f"{script_id}.jsonl"
        _write(out, rows)
        total += len(rows)

    print(f"\ngenerated {total} training pairs across {len(targets)} script(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
