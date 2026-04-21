#!/usr/bin/env python3
# coding: utf-8
"""
build_language_corpus.py -- Language, literature and technical documentation corpus

Stages:
  18 -- Project Gutenberg: literature, philosophy, science, math history
  19 -- Wikibooks: computing, programming, science textbooks
  20 -- MDN Web Docs: HTML/CSS/JavaScript reference (CC BY-SA)
  21 -- Rust Book + Go Tour + Python Tutorial (official, MIT/open)
  22 -- IETF RFCs: HTTP, TLS, WebSocket, TCP/IP, web standards

All sources are free/open-licensed. No untested code is trained --
code blocks from docs are labeled as examples, not verified executables.

Usage:
  python scripts/build_language_corpus.py --stages 18,19,20,21,22 --node localhost:8090
"""

import argparse, html, json, re, sys, time, urllib.request, urllib.error, urllib.parse
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = Path('D:/w1z4rdv1510n-data')

STAGES = {
    18: 'Project Gutenberg: literature, philosophy, science, math history',
    19: 'Wikibooks: computing, programming, mathematics textbooks',
    20: 'MDN Web Docs: HTML/CSS/JavaScript reference',
    21: 'Rust Book, Go Tour, Python Tutorial (official docs)',
    22: 'IETF RFCs: HTTP, TLS, WebSocket, TCP/IP, DNS, OAuth',
}

# -- HTTP helpers ---------------------------------------------------------------

def _get(url: str, timeout=20, retries=3) -> bytes | None:
    headers = {'User-Agent': 'W1z4rDV1510n-training/1.0 (educational corpus builder)'}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410):
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def _get_json(url: str) -> dict | None:
    data = _get(url)
    if data:
        try:
            return json.loads(data.decode('utf-8', errors='replace'))
        except Exception:
            pass
    return None


def _get_text(url: str) -> str | None:
    data = _get(url)
    if data:
        return data.decode('utf-8', errors='replace')
    return None


def _strip_html(text: str) -> str:
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _train_text(text: str, node: str):
    if not text or not text.strip():
        return
    payload = json.dumps({'text': text}).encode()
    req = urllib.request.Request(
        f'http://{node}/media/train',
        data=payload,
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


# -- Stage 18: Project Gutenberg -----------------------------------------------

# Categories to pull from Gutendex (free Project Gutenberg API)
GUTENBERG_SEARCHES = [
    ('philosophy',          30, 'Philosophy and logic -- classical texts'),
    ('mathematics',         20, 'Mathematics history and foundations'),
    ('physics',             15, 'Classical physics texts'),
    ('astronomy',           10, 'Astronomy and cosmology'),
    ('natural history',     10, 'Natural history and biology'),
    ('electricity',         10, 'Electricity and electromagnetism'),
    ('logic',               15, 'Formal logic and reasoning'),
    ('psychology',          10, 'Early psychology and cognitive science'),
    ('mechanics',           10, 'Classical mechanics'),
    ('engineering',         10, 'Engineering and technology'),
    ('language',            10, 'Linguistics and language study'),
    ('grammar',             10, 'Grammar and language structure'),
    ('fiction',             20, 'Classic fiction for language pattern modeling'),
    ('poetry',              10, 'Classic poetry for linguistic richness'),
    ('drama',               10, 'Classical drama and dialogue'),
]

def _gutenberg_text_url(book: dict) -> str | None:
    fmts = book.get('formats', {})
    for key in ('text/plain; charset=utf-8', 'text/plain; charset=us-ascii', 'text/plain'):
        if key in fmts:
            return fmts[key]
    return None


def _extract_gutenberg_passages(raw: str, max_chars=40000) -> list[str]:
    """Strip Gutenberg header/footer, split into ~1000-char passages."""
    # Strip standard header/footer
    start = re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG', raw, re.I)
    end   = re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG',   raw, re.I)
    if start:
        raw = raw[start.end():]
    if end:
        raw = raw[:end.start()]

    raw = raw[:max_chars]
    # Split on double newlines into paragraphs
    paras = [p.strip() for p in re.split(r'\n\s*\n', raw) if len(p.strip()) > 100]
    passages = []
    buf = ''
    for p in paras:
        buf += p + '\n\n'
        if len(buf) >= 800:
            passages.append(buf.strip())
            buf = ''
    if buf.strip():
        passages.append(buf.strip())
    return passages


def build_gutenberg_corpus(node: str, args):
    print(f'\n[Stage 18] Project Gutenberg\n')
    trained = 0
    seen_ids = set()

    for topic, max_books, label in GUTENBERG_SEARCHES:
        print(f'  Topic: {topic} ({max_books} books max)...')
        page = 1
        fetched = 0
        while fetched < max_books:
            url = f'https://gutendex.com/books/?languages=en&topic={urllib.parse.quote(topic)}&page={page}'
            data = _get_json(url)
            if not data or not data.get('results'):
                break
            for book in data['results']:
                if fetched >= max_books:
                    break
                bid = book.get('id')
                if bid in seen_ids:
                    continue
                seen_ids.add(bid)
                title   = book.get('title', 'Unknown')
                authors = ', '.join(a['name'] for a in book.get('authors', []))
                txt_url = _gutenberg_text_url(book)
                if not txt_url:
                    continue
                raw = _get_text(txt_url)
                if not raw or len(raw) < 500:
                    continue
                passages = _extract_gutenberg_passages(raw, max_chars=args.gutenberg_chars)
                header = f'Project Gutenberg text -- "{title}" by {authors}. Category: {label}.\n\n'
                for passage in passages[:args.gutenberg_passages]:
                    try:
                        _train_text(header + passage, node)
                        trained += 1
                    except Exception as e:
                        print(f'    [WARN] train failed: {e}')
                fetched += 1
                print(f'    [{fetched}/{max_books}] {title[:60]} -- {len(passages)} passages')
                time.sleep(0.5)
            if not data.get('next'):
                break
            page += 1
            time.sleep(0.3)

    print(f'  Stage 18 done -- {trained} passages trained from {len(seen_ids)} books')


# -- Stage 19: Wikibooks --------------------------------------------------------

WIKIBOOKS_TITLES = [
    # Programming languages
    'Python Programming', 'JavaScript', 'C Programming', 'C++ Programming',
    'Go Programming', 'Bash Shell Scripting', 'Ruby Programming',
    'Haskell', 'Scheme Programming', 'Prolog',
    # Systems + CS
    'Algorithm Design', 'Data Structures', 'Computer Programming',
    'Discrete Mathematics', 'Introduction to Computer Information Systems',
    'Computer Architecture', 'Operating System Design',
    'Communication Networks', 'Cryptography',
    # Mathematics
    'Linear Algebra', 'Calculus', 'Abstract Algebra', 'Real Analysis',
    'Probability Theory', 'Statistics',
    # Engineering + electronics
    'Digital Circuits', 'Electronics', 'Embedded Systems',
    'Robotics', 'Control Systems',
    # Web
    'HTML', 'Cascading Style Sheets', 'Web Development',
    # Databases
    'Structured Query Language', 'SQL Exercises',
    # Software engineering
    'Software Engineering', 'Design Patterns', 'Object-Oriented Programming',
    # AI/ML
    'Introduction to Artificial Intelligence', 'Machine Learning',
    'Artificial Neural Networks',
    # Science
    'Physics Equations', 'Chemistry', 'Biology',
    'Sensory Systems', 'Human Physiology',
    # Language
    'Regular Expressions', 'LaTeX',
]


def _wikibooks_extract(title: str) -> str | None:
    api = (
        'https://en.wikibooks.org/w/api.php'
        f'?action=query&titles={urllib.parse.quote(title)}'
        '&prop=extracts&explaintext=1&exsectionformat=plain&format=json'
    )
    data = _get_json(api)
    if not data:
        return None
    pages = data.get('query', {}).get('pages', {})
    for page in pages.values():
        extract = page.get('extract', '')
        if len(extract) > 200:
            return extract
    return None


def build_wikibooks_corpus(node: str, args):
    print(f'\n[Stage 19] Wikibooks\n')
    trained = 0
    for title in WIKIBOOKS_TITLES:
        text = _wikibooks_extract(title)
        if not text:
            print(f'  SKIP {title}')
            time.sleep(0.2)
            continue
        # Split into ~1000-char chunks
        chunks = [text[i:i+1200] for i in range(0, min(len(text), args.wikibooks_chars), 1200)]
        header = f'Wikibooks -- "{title}". Free educational textbook.\n\n'
        count = 0
        for chunk in chunks:
            if len(chunk.strip()) < 100:
                continue
            try:
                _train_text(header + chunk.strip(), node)
                trained += 1
                count += 1
            except Exception as e:
                print(f'  [WARN] {e}')
        print(f'  {title[:55]}: {count} chunks')
        time.sleep(0.4)
    print(f'  Stage 19 done -- {trained} chunks trained')


# -- Stage 20: MDN Web Docs -----------------------------------------------------

# Popular CSS properties to document
CSS_PROPERTIES = [
    'display', 'position', 'float', 'clear', 'overflow', 'visibility',
    'flex', 'flex-direction', 'flex-wrap', 'justify-content', 'align-items',
    'align-self', 'flex-grow', 'flex-shrink', 'flex-basis', 'order',
    'grid', 'grid-template-columns', 'grid-template-rows', 'grid-area',
    'grid-gap', 'gap', 'grid-column', 'grid-row',
    'margin', 'padding', 'border', 'border-radius', 'outline',
    'width', 'height', 'min-width', 'max-width', 'min-height', 'max-height',
    'box-sizing', 'box-shadow', 'text-shadow',
    'color', 'background', 'background-color', 'background-image',
    'background-size', 'background-position', 'background-repeat',
    'font', 'font-family', 'font-size', 'font-weight', 'font-style',
    'line-height', 'letter-spacing', 'text-align', 'text-decoration',
    'text-transform', 'vertical-align', 'white-space', 'word-break',
    'opacity', 'transform', 'transition', 'animation', 'animation-name',
    'animation-duration', 'animation-timing-function', 'animation-delay',
    'z-index', 'top', 'right', 'bottom', 'left',
    'cursor', 'pointer-events', 'user-select',
    'content', 'counter-reset', 'counter-increment',
    'filter', 'backdrop-filter', 'mix-blend-mode',
    'clip-path', 'mask', 'perspective', 'transform-origin',
    'scroll-behavior', 'scroll-snap-type', 'overscroll-behavior',
    'resize', 'appearance', 'all', 'var', 'calc', 'clamp',
]

# HTML elements to document
HTML_ELEMENTS = [
    'div', 'span', 'p', 'a', 'img', 'input', 'button', 'form',
    'h1', 'h2', 'h3', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th',
    'section', 'article', 'header', 'footer', 'nav', 'main', 'aside',
    'figure', 'figcaption', 'canvas', 'video', 'audio', 'source',
    'script', 'style', 'link', 'meta', 'template', 'slot',
    'select', 'option', 'textarea', 'label', 'fieldset', 'legend',
    'details', 'summary', 'dialog', 'progress', 'meter', 'output',
    'iframe', 'object', 'embed', 'picture', 'svg', 'math',
    'blockquote', 'code', 'pre', 'kbd', 'samp', 'var',
    'strong', 'em', 'mark', 'del', 'ins', 'sub', 'sup',
    'abbr', 'cite', 'time', 'address', 'data',
]

# JavaScript built-ins and Web APIs
JS_TOPICS = [
    # Array methods
    'Array/map', 'Array/filter', 'Array/reduce', 'Array/forEach', 'Array/find',
    'Array/findIndex', 'Array/some', 'Array/every', 'Array/flat', 'Array/flatMap',
    'Array/includes', 'Array/from', 'Array/of', 'Array/sort', 'Array/splice',
    'Array/slice', 'Array/join', 'Array/concat', 'Array/push', 'Array/pop',
    # String methods
    'String/split', 'String/replace', 'String/replaceAll', 'String/includes',
    'String/startsWith', 'String/endsWith', 'String/trim', 'String/padStart',
    'String/padEnd', 'String/repeat', 'String/slice', 'String/substring',
    'String/match', 'String/matchAll', 'String/at', 'String/template_literals',
    # Object
    'Object/keys', 'Object/values', 'Object/entries', 'Object/assign',
    'Object/freeze', 'Object/create', 'Object/defineProperty',
    'Object/fromEntries', 'Object/hasOwn',
    # Async
    'Promise', 'Promise/all', 'Promise/allSettled', 'Promise/race',
    'Promise/any', 'Promise/then', 'Promise/catch',
    'AsyncFunction', 'Statements/async_function',
    # Modern JS
    'Operators/Destructuring_assignment', 'Operators/Spread_syntax',
    'Operators/Optional_chaining', 'Operators/Nullish_coalescing',
    'Functions/Arrow_functions', 'Classes', 'Statements/import',
    'Statements/export',
    # Built-ins
    'JSON', 'Map', 'Set', 'WeakMap', 'WeakSet', 'Symbol',
    'Proxy', 'Reflect', 'Generator', 'RegExp',
    'Number', 'Math', 'Date', 'Error',
    # Web APIs
    'fetch', 'localStorage', 'sessionStorage', 'URL', 'URLSearchParams',
    'EventTarget', 'CustomEvent',
]


def _fetch_mdn_page(path: str) -> str | None:
    url = f'https://developer.mozilla.org/en-US/docs/{path}'
    data = _get(url)
    if not data:
        return None
    text = data.decode('utf-8', errors='replace')
    # Extract main article content between <article> tags
    m = re.search(r'<article[^>]*>(.*?)</article>', text, re.S)
    if not m:
        m = re.search(r'<div[^>]+class="[^"]*content[^"]*"[^>]*>(.*?)</div>', text, re.S)
    if m:
        text = m.group(1)
    # Strip scripts and styles
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.S)
    # Extract code examples
    code_blocks = re.findall(r'<code[^>]*>(.*?)</code>', text, re.S)
    # Strip all HTML
    clean = _strip_html(text)
    # Re-attach code blocks as labeled examples
    if code_blocks:
        examples = '\n'.join(f'Example: {_strip_html(c)}' for c in code_blocks[:5] if len(c.strip()) > 5)
        clean = clean + '\n\n' + examples
    return clean if len(clean) > 100 else None


def build_mdn_corpus(node: str, args):
    print(f'\n[Stage 20] MDN Web Docs\n')
    trained = 0

    print('  CSS properties...')
    for prop in CSS_PROPERTIES:
        text = _fetch_mdn_page(f'Web/CSS/{prop}')
        if text:
            try:
                _train_text(
                    f'MDN Web Docs -- CSS property: {prop}.\n'
                    f'Source: developer.mozilla.org (CC BY-SA 2.5)\n\n{text[:2000]}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] {e}')
        time.sleep(0.3)
    print(f'  CSS: {len(CSS_PROPERTIES)} properties processed')

    print('  HTML elements...')
    for elem in HTML_ELEMENTS:
        text = _fetch_mdn_page(f'Web/HTML/Element/{elem}')
        if text:
            try:
                _train_text(
                    f'MDN Web Docs -- HTML element: <{elem}>.\n'
                    f'Source: developer.mozilla.org (CC BY-SA 2.5)\n\n{text[:2000]}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] {e}')
        time.sleep(0.3)
    print(f'  HTML: {len(HTML_ELEMENTS)} elements processed')

    print('  JavaScript reference...')
    for topic in JS_TOPICS:
        text = _fetch_mdn_page(f'Web/JavaScript/Reference/Global_Objects/{topic}')
        if not text:
            text = _fetch_mdn_page(f'Web/JavaScript/Reference/{topic}')
        if text:
            try:
                name = topic.split('/')[-1].replace('_', ' ')
                _train_text(
                    f'MDN Web Docs -- JavaScript: {name}.\n'
                    f'Source: developer.mozilla.org (CC BY-SA 2.5)\n\n{text[:2000]}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] {e}')
        time.sleep(0.3)
    print(f'  JS: {len(JS_TOPICS)} topics processed')

    print(f'  Stage 20 done -- {trained} MDN pages trained')


# -- Stage 21: Official language docs ------------------------------------------

# Rust Book chapters (GitHub raw)
RUST_BOOK_CHAPTERS = [
    'ch00-00-introduction.md',
    'ch01-00-getting-started.md', 'ch01-02-hello-world.md',
    'ch02-00-guessing-game-tutorial.md',
    'ch03-00-common-programming-concepts.md',
    'ch03-01-variables-and-mutability.md', 'ch03-02-data-types.md',
    'ch03-03-how-functions-work.md', 'ch03-05-control-flow.md',
    'ch04-00-understanding-ownership.md', 'ch04-01-what-is-ownership.md',
    'ch04-02-references-and-borrowing.md', 'ch04-03-slices.md',
    'ch05-00-structs.md', 'ch05-01-defining-structs.md',
    'ch05-03-method-syntax.md',
    'ch06-00-enums.md', 'ch06-01-defining-an-enum.md',
    'ch06-02-match.md', 'ch06-03-if-let.md',
    'ch07-00-managing-growing-projects.md',
    'ch08-00-common-collections.md', 'ch08-01-vectors.md',
    'ch08-02-strings.md', 'ch08-03-hash-maps.md',
    'ch09-00-error-handling.md', 'ch09-01-unrecoverable-errors-with-panic.md',
    'ch09-02-recoverable-errors-with-result.md',
    'ch10-00-generics.md', 'ch10-01-syntax.md', 'ch10-02-traits.md',
    'ch10-03-lifetime-syntax.md',
    'ch11-00-testing.md', 'ch11-01-writing-tests.md',
    'ch12-00-an-io-project.md',
    'ch13-00-functional-features.md', 'ch13-01-closures.md',
    'ch13-02-iterators.md',
    'ch15-00-smart-pointers.md', 'ch15-01-box.md', 'ch15-02-deref.md',
    'ch16-00-concurrency.md', 'ch16-01-threads.md', 'ch16-02-message-passing.md',
    'ch17-00-oop.md', 'ch17-01-what-is-oo.md', 'ch17-02-trait-objects.md',
    'ch18-00-patterns.md', 'ch19-00-advanced-features.md',
    'ch20-00-final-project-a-web-server.md',
]

# Python tutorial sections
PYTHON_TUTORIAL_SECTIONS = [
    'appetite', 'interpreter', 'introduction', 'controlflow',
    'datastructures', 'modules', 'inputoutput', 'errors',
    'classes', 'stdlib', 'stdlib2', 'venv', 'whatnow',
]

# Go Tour sections (markdown on GitHub)
GO_TOUR_PAGES = [
    'welcome/1', 'basics/1', 'basics/2', 'basics/3', 'basics/4',
    'flowcontrol/1', 'moretypes/1', 'methods/1', 'interfaces/1',
    'concurrency/1',
]


def build_official_docs_corpus(node: str, args):
    print(f'\n[Stage 21] Official language documentation\n')
    trained = 0

    # Rust Book
    print('  Rust Book (rust-lang/book @ GitHub)...')
    base = 'https://raw.githubusercontent.com/rust-lang/book/main/src/'
    for ch in RUST_BOOK_CHAPTERS:
        text = _get_text(base + ch)
        if not text or len(text) < 100:
            time.sleep(0.2)
            continue
        # Strip markdown image links and HTML
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'<[^>]+>', '', text)
        chunks = [text[i:i+1500] for i in range(0, min(len(text), 12000), 1500)]
        chapter_name = ch.replace('.md', '').replace('-', ' ')
        for chunk in chunks:
            if len(chunk.strip()) < 80:
                continue
            try:
                _train_text(
                    f'The Rust Programming Language (official book, MIT/Apache-2.0).\n'
                    f'Chapter: {chapter_name}\n\n{chunk.strip()}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] {e}')
        time.sleep(0.3)
    print(f'  Rust Book: {len(RUST_BOOK_CHAPTERS)} chapters')

    # Python Tutorial
    print('  Python Tutorial (docs.python.org)...')
    for section in PYTHON_TUTORIAL_SECTIONS:
        url = f'https://docs.python.org/3/tutorial/{section}.html'
        data = _get(url)
        if not data:
            time.sleep(0.3)
            continue
        html_text = data.decode('utf-8', errors='replace')
        # Extract body
        m = re.search(r'<div[^>]+class="[^"]*body[^"]*"[^>]*>(.*?)</div>', html_text, re.S)
        raw = m.group(1) if m else html_text
        raw = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', raw, flags=re.S)
        clean = _strip_html(raw)
        chunks = [clean[i:i+1500] for i in range(0, min(len(clean), 15000), 1500)]
        for chunk in chunks:
            if len(chunk.strip()) < 80:
                continue
            try:
                _train_text(
                    f'Python 3 Official Tutorial (docs.python.org, PSF License).\n'
                    f'Section: {section}\n\n{chunk.strip()}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] {e}')
        time.sleep(0.4)
    print(f'  Python Tutorial: {len(PYTHON_TUTORIAL_SECTIONS)} sections')

    # Go specification
    print('  Go Language Specification...')
    go_spec = _get_text('https://go.dev/ref/spec')
    if go_spec:
        go_spec = _strip_html(go_spec)
        chunks = [go_spec[i:i+1500] for i in range(0, min(len(go_spec), 30000), 1500)]
        for chunk in chunks:
            if len(chunk.strip()) < 80:
                continue
            try:
                _train_text(
                    f'Go Programming Language Specification (go.dev, BSD license).\n\n{chunk.strip()}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] {e}')
    print(f'  Go Specification processed')

    # Go Effective Guide
    print('  Effective Go...')
    eff_go = _get_text('https://go.dev/doc/effective_go')
    if eff_go:
        eff_go = _strip_html(eff_go)
        chunks = [eff_go[i:i+1500] for i in range(0, min(len(eff_go), 25000), 1500)]
        for chunk in chunks:
            if len(chunk.strip()) < 80:
                continue
            try:
                _train_text(
                    f'Effective Go -- idiomatic Go programming guide (go.dev, BSD).\n\n{chunk.strip()}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] {e}')

    # MDN JavaScript Guide
    print('  MDN JavaScript Guide...')
    js_guide_sections = [
        'Introduction', 'Grammar_and_types', 'Control_flow_and_error_handling',
        'Loops_and_iteration', 'Functions', 'Expressions_and_operators',
        'Numbers_and_dates', 'Text_formatting', 'Indexed_collections',
        'Keyed_collections', 'Working_with_objects', 'Using_classes',
        'Promises', 'Iterators_and_generators', 'Meta_programming',
        'Modules', 'Regular_expressions',
    ]
    for section in js_guide_sections:
        url = f'https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/{section}'
        data = _get(url)
        if not data:
            time.sleep(0.3)
            continue
        raw = data.decode('utf-8', errors='replace')
        raw = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', raw, flags=re.S)
        clean = _strip_html(raw)
        # Find the main content (usually long paragraph blocks)
        if len(clean) > 500:
            try:
                _train_text(
                    f'MDN JavaScript Guide -- {section.replace("_", " ")} '
                    f'(developer.mozilla.org, CC BY-SA 2.5).\n\n{clean[200:3000]}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] {e}')
        time.sleep(0.4)

    print(f'  Stage 21 done -- {trained} doc sections trained')


# -- Stage 22: IETF RFCs --------------------------------------------------------

RFC_LIST = [
    # HTTP
    (9110, 'HTTP Semantics'),
    (9112, 'HTTP/1.1'),
    (9113, 'HTTP/2'),
    (9114, 'HTTP/3'),
    (7540, 'HTTP/2 (original)'),
    (7230, 'HTTP/1.1 Message Syntax'),
    (7231, 'HTTP/1.1 Semantics and Content'),
    (7235, 'HTTP/1.1 Authentication'),
    # Security
    (8446, 'TLS 1.3'),
    (5246, 'TLS 1.2'),
    (7519, 'JSON Web Tokens (JWT)'),
    (6749, 'OAuth 2.0'),
    (7636, 'OAuth 2.0 PKCE'),
    (7617, 'HTTP Basic Authentication'),
    # Web protocols
    (6455, 'WebSocket Protocol'),
    (7807, 'Problem Details for HTTP APIs'),
    (6570, 'URI Templates'),
    (3986, 'URI Generic Syntax'),
    (7396, 'JSON Merge Patch'),
    (6902, 'JSON Patch'),
    # Data formats
    (8259, 'JSON'),
    (4648, 'Base64 Encoding'),
    (7159, 'JSON (original)'),
    # Transport
    (793,  'TCP'),
    (791,  'IP'),
    (768,  'UDP'),
    (826,  'ARP'),
    # Application
    (2822, 'Internet Message Format (Email)'),
    (5321, 'SMTP'),
    (1034, 'DNS Concepts'),
    (1035, 'DNS Implementation'),
    (4122, 'UUID'),
    (5905, 'NTP Network Time Protocol'),
    # CORS / web security
    (6454, 'Web Origin Concept'),
]


def _extract_rfc_sections(text: str) -> list[str]:
    """Extract Abstract + Introduction + key sections from RFC plain text."""
    sections = []
    # Abstract
    m = re.search(r'Abstract\s*\n+(.*?)(?=\n\s*\n\s*[A-Z0-9]|\nTable of Contents|\n1\.)', text, re.S)
    if m:
        sections.append('Abstract:\n' + m.group(1).strip())
    # Introduction section
    m = re.search(r'\n1\.\s+Introduction\s*\n+(.*?)(?=\n\d+\.\s+[A-Z]|\Z)', text, re.S)
    if m:
        intro = m.group(1).strip()
        sections.append('1. Introduction:\n' + intro[:3000])
    # Try to get next 2 sections
    section_matches = list(re.finditer(r'\n([2-5])\.\s+(\w[^\n]{3,60})\n(.*?)(?=\n[2-9]\.\s+[A-Z]|\Z)', text, re.S))
    for sm in section_matches[:3]:
        sec_text = sm.group(3).strip()[:2000]
        sections.append(f'{sm.group(1)}. {sm.group(2).strip()}:\n{sec_text}')
    return sections


def build_rfc_corpus(node: str, args):
    print(f'\n[Stage 22] IETF RFCs\n')
    trained = 0

    for rfc_num, title in RFC_LIST:
        url = f'https://www.rfc-editor.org/rfc/rfc{rfc_num}.txt'
        text = _get_text(url)
        if not text:
            print(f'  SKIP RFC {rfc_num}')
            time.sleep(0.5)
            continue
        sections = _extract_rfc_sections(text)
        if not sections:
            # Fall back to first 3000 chars
            sections = [text[500:3500]]
        for section in sections:
            if len(section.strip()) < 100:
                continue
            try:
                _train_text(
                    f'IETF RFC {rfc_num} -- {title}.\n'
                    f'Source: rfc-editor.org (public domain)\n\n{section.strip()}',
                    node
                )
                trained += 1
            except Exception as e:
                print(f'  [WARN] RFC {rfc_num}: {e}')
        print(f'  RFC {rfc_num} ({title[:40]}): {len(sections)} sections')
        time.sleep(0.5)

    print(f'  Stage 22 done -- {trained} RFC sections trained')


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Build language + docs corpus')
    ap.add_argument('--stages',             default='18,19,20,21,22')
    ap.add_argument('--node',               default='localhost:8090')
    ap.add_argument('--gutenberg-chars',    type=int, default=60000,
                    help='Max chars to read per Gutenberg book')
    ap.add_argument('--gutenberg-passages', type=int, default=20,
                    help='Max passages per Gutenberg book')
    ap.add_argument('--wikibooks-chars',    type=int, default=15000,
                    help='Max chars to read per Wikibooks page')
    args = ap.parse_args()

    stages = [int(s.strip()) for s in args.stages.split(',')]
    node   = args.node

    print('=' * 70)
    print('  W1z4rD V1510n -- Language + Documentation Corpus Builder')
    print('=' * 70)
    print(f'  Node:    http://{node}')
    print(f'  Stages:  {stages}')
    print()
    for s in stages:
        print(f'  Stage {s}: {STAGES.get(s, "?")}')
    print()

    if 18 in stages:
        build_gutenberg_corpus(node, args)
    if 19 in stages:
        build_wikibooks_corpus(node, args)
    if 20 in stages:
        build_mdn_corpus(node, args)
    if 21 in stages:
        build_official_docs_corpus(node, args)
    if 22 in stages:
        build_rfc_corpus(node, args)

    print('\n' + '=' * 70)
    print('  Language corpus build complete.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
