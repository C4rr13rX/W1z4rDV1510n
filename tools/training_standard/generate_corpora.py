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
    ("Python",              "a high-level programming language with clean syntax used for scripts, data work, and web backends"),
    ("Rust",                "a systems programming language with strong memory safety guarantees via ownership and borrow checking"),
    ("JavaScript",          "the language of the web; runs in browsers and in Node.js for backend services"),
    ("Bash",                "the default shell on most Linux and macOS systems; used for scripting and one-line commands"),
    ("PowerShell",          "the default shell on modern Windows; pipes structured objects instead of plain text"),
    ("a regression test",   "a test that locks in known-good behavior so a future change doesn't accidentally break it"),
    ("a benchmark",         "a measured contract a training script promises to satisfy after training"),
    ("the wizard node",     "the W1z4rD V1510n process that runs the neural fabric and answers /chat requests"),
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
    ("python",     "Python", "A high-level interpreted language; clean syntax, batteries-included standard library."),
    ("rust",       "Rust",   "A systems language with ownership-based memory safety and zero-cost abstractions."),
    ("javascript", "JavaScript", "The language of the web; ubiquitous in browsers and on the server via Node.js."),
    ("bash",       "Bash",   "The dominant Unix shell; pipes, redirection, glob patterns, and POSIX utilities."),
    ("powershell", "PowerShell", "The structured-object shell from Microsoft; pipes pass objects, not text."),
    ("hebbian learning", "Hebbian learning", "When two neurons fire together the synapse between them strengthens — the basic plasticity rule."),
    ("neurons", "neurons", "Cells with dendrites for input, axons for output; fire above threshold and emit a signal."),
    ("multi-pool", "the multi-pool fabric", "N independent neuron pools — one per modality — linked by cross-pool synapses."),
    ("training",  "training", "The process of presenting paired observations so the fabric updates synapse weights."),
    ("synapses",  "synapses", "Weighted connections between neurons; the weights are what the fabric actually learns."),
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
        # alternate phrasings
        rows.append({"prompt": prompt.replace("Write", "Implement"), "response": code})
        rows.append({"prompt": prompt.replace("Write", "Give me"),   "response": code})
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
