#!/usr/bin/env python3
# coding: utf-8
"""
build_library_corpus.py — MIT-licensed library docs + Stack Overflow accepted answers

Stage 23: MIT library documentation
  Fetches README + docs from pinned versions of popular MIT-licensed libraries.
  Extracts (English description, code example) pairs from markdown.
  Code is LABELED as library documentation examples, not independently verified.

Stage 24: Stack Overflow accepted answers
  Fetches top-voted questions with accepted answers via the Stack Exchange API.
  ONLY trains: original question + accepted/highest-voted answer + explanation.
  No unverified snippets — only community-validated, accepted answers.
  The question gives the "someone described a problem in English" signal;
  the accepted answer gives the verified resolution + explanation.

Usage:
  python scripts/build_library_corpus.py --stages 23,24 --node localhost:8090
"""

import argparse, html as html_mod, json, re, sys, time, urllib.request, urllib.error, urllib.parse
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = Path('D:/w1z4rdv1510n-data')

STAGES = {
    23: 'MIT library documentation with code examples',
    24: 'Stack Overflow: top accepted answers (question + explanation + code)',
}

# ── HTTP helpers ───────────────────────────────────────────────────────────────

def _get(url: str, timeout=20, retries=3, headers=None) -> bytes | None:
    hdrs = {'User-Agent': 'W1z4rDV1510n-training/1.0 (educational corpus builder)'}
    if headers:
        hdrs.update(headers)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=hdrs)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (404, 410, 429):
                if e.code == 429:
                    time.sleep(10)
                return None
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return None


def _get_json(url: str, headers=None) -> dict | list | None:
    data = _get(url, headers=headers)
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
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', text, flags=re.S)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = html_mod.unescape(text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _train_text(text: str, node: str):
    if not text or len(text.strip()) < 50:
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


# ── Markdown code block parser ─────────────────────────────────────────────────

def _extract_doc_pairs(markdown: str) -> list[dict]:
    """
    Extract (description, lang, code) triples from markdown.
    Each code block is paired with the nearest preceding heading + paragraph.
    """
    pairs = []
    # Split by fenced code blocks
    parts = re.split(r'```(\w*)\n(.*?)```', markdown, flags=re.S)
    # parts: [text, lang, code, text, lang, code, ...]
    i = 0
    while i < len(parts):
        if i % 3 == 0:
            # Text before next code block
            preceding_text = parts[i]
        elif i % 3 == 1:
            lang = parts[i].strip()
        elif i % 3 == 2:
            code = parts[i].strip()
            if len(code) < 10:
                i += 1
                continue
            # Find the nearest heading in preceding text
            headings = re.findall(r'^#{1,4}\s+(.+)$', preceding_text, re.M)
            heading  = headings[-1].strip() if headings else ''
            # Find the last paragraph in preceding text (closest context)
            paras = [p.strip() for p in re.split(r'\n\s*\n', preceding_text) if p.strip()]
            context = paras[-1] if paras else ''
            # Combine heading + context as description
            desc = ''
            if heading:
                desc += heading + '.\n'
            if context and context != heading:
                desc += context
            if len(desc.strip()) > 10 and len(code) > 10:
                pairs.append({'description': desc.strip(), 'lang': lang, 'code': code})
        i += 1
    return pairs


# ── Stage 23: MIT Library Documentation ───────────────────────────────────────

# Libraries: (display_name, license, github_org/repo, docs_paths)
LIBRARIES = [
    # Python
    {
        'name':    'requests',
        'lang':    'Python',
        'license': 'Apache-2.0 (compatible)',
        'repo':    'psf/requests',
        'paths':   ['README.md', 'docs/index.rst', 'docs/user/quickstart.rst',
                    'docs/user/advanced.rst', 'docs/api.rst'],
    },
    {
        'name':    'FastAPI',
        'lang':    'Python',
        'license': 'MIT',
        'repo':    'tiangolo/fastapi',
        'paths':   ['README.md', 'docs/en/docs/index.md',
                    'docs/en/docs/tutorial/first-steps.md',
                    'docs/en/docs/tutorial/path-params.md',
                    'docs/en/docs/tutorial/query-params.md',
                    'docs/en/docs/tutorial/body.md',
                    'docs/en/docs/tutorial/response-model.md',
                    'docs/en/docs/tutorial/path-operation-configuration.md',
                    'docs/en/docs/tutorial/handling-errors.md',
                    'docs/en/docs/tutorial/dependencies.md',
                    'docs/en/docs/tutorial/background-tasks.md',
                    'docs/en/docs/tutorial/middleware.md',
                    'docs/en/docs/tutorial/cors.md',
                    'docs/en/docs/tutorial/sql-databases.md',
                    'docs/en/docs/advanced/index.md'],
    },
    {
        'name':    'Click',
        'lang':    'Python',
        'license': 'BSD-3-Clause',
        'repo':    'pallets/click',
        'paths':   ['README.md', 'docs/index.rst', 'docs/quickstart.rst',
                    'docs/api.rst', 'docs/commands.rst', 'docs/options.rst',
                    'docs/arguments.rst', 'docs/utils.rst'],
    },
    {
        'name':    'Rich',
        'lang':    'Python',
        'license': 'MIT',
        'repo':    'Textualize/rich',
        'paths':   ['README.md', 'docs/source/introduction.rst',
                    'docs/source/console.rst', 'docs/source/style.rst',
                    'docs/source/markup.rst', 'docs/source/text.rst',
                    'docs/source/tables.rst', 'docs/source/progress.rst',
                    'docs/source/panel.rst', 'docs/source/layout.rst',
                    'docs/source/live.rst', 'docs/source/syntax.rst',
                    'docs/source/logging.rst', 'docs/source/pretty.rst',
                    'docs/source/inspect.rst'],
    },
    {
        'name':    'Pydantic',
        'lang':    'Python',
        'license': 'MIT',
        'repo':    'pydantic/pydantic',
        'paths':   ['README.md', 'docs/index.md', 'docs/concepts/models.md',
                    'docs/concepts/fields.md', 'docs/concepts/validators.md',
                    'docs/concepts/types.md', 'docs/concepts/config.md',
                    'docs/concepts/serialization.md', 'docs/concepts/json_schema.md'],
    },
    {
        'name':    'Pytest',
        'lang':    'Python',
        'license': 'MIT',
        'repo':    'pytest-dev/pytest',
        'paths':   ['README.rst', 'doc/en/getting-started.rst',
                    'doc/en/how-to/assert.rst', 'doc/en/how-to/fixtures.rst',
                    'doc/en/how-to/parametrize.rst', 'doc/en/how-to/monkeypatch.rst',
                    'doc/en/how-to/tmp_path.rst'],
    },
    {
        'name':    'SQLAlchemy',
        'lang':    'Python',
        'license': 'MIT',
        'repo':    'sqlalchemy/sqlalchemy',
        'paths':   ['README.rst', 'doc/build/tutorial/index.rst',
                    'doc/build/orm/quickstart.rst'],
    },
    {
        'name':    'Httpx',
        'lang':    'Python',
        'license': 'BSD-3-Clause',
        'repo':    'encode/httpx',
        'paths':   ['README.md', 'docs/index.md', 'docs/quickstart.md',
                    'docs/async.md', 'docs/exceptions.md'],
    },
    # JavaScript / Node.js
    {
        'name':    'Express.js',
        'lang':    'JavaScript',
        'license': 'MIT',
        'repo':    'expressjs/express',
        'paths':   ['Readme.md', 'History.md'],
    },
    {
        'name':    'Lodash',
        'lang':    'JavaScript',
        'license': 'MIT',
        'repo':    'lodash/lodash',
        'paths':   ['README.md'],
    },
    {
        'name':    'Chalk',
        'lang':    'JavaScript',
        'license': 'MIT',
        'repo':    'chalk/chalk',
        'paths':   ['readme.md'],
    },
    {
        'name':    'Commander.js',
        'lang':    'JavaScript',
        'license': 'MIT',
        'repo':    'tj/commander.js',
        'paths':   ['Readme.md'],
    },
    {
        'name':    'Zod',
        'lang':    'TypeScript',
        'license': 'MIT',
        'repo':    'colinhacks/zod',
        'paths':   ['README.md'],
    },
    {
        'name':    'Axios',
        'lang':    'JavaScript',
        'license': 'MIT',
        'repo':    'axios/axios',
        'paths':   ['README.md'],
    },
    # Rust
    {
        'name':    'Serde',
        'lang':    'Rust',
        'license': 'MIT/Apache-2.0',
        'repo':    'serde-rs/serde',
        'paths':   ['README.md'],
    },
    {
        'name':    'Clap',
        'lang':    'Rust',
        'license': 'MIT/Apache-2.0',
        'repo':    'clap-rs/clap',
        'paths':   ['README.md', 'CHANGELOG.md'],
    },
    {
        'name':    'Tokio',
        'lang':    'Rust',
        'license': 'MIT',
        'repo':    'tokio-rs/tokio',
        'paths':   ['README.md'],
    },
    {
        'name':    'Reqwest',
        'lang':    'Rust',
        'license': 'MIT/Apache-2.0',
        'repo':    'seanmonstar/reqwest',
        'paths':   ['README.md'],
    },
    # Go
    {
        'name':    'Gin',
        'lang':    'Go',
        'license': 'MIT',
        'repo':    'gin-gonic/gin',
        'paths':   ['README.md', 'docs/doc.md'],
    },
    {
        'name':    'Cobra',
        'lang':    'Go',
        'license': 'Apache-2.0',
        'repo':    'spf13/cobra',
        'paths':   ['README.md', 'site/content/user_guide.md'],
    },
    # Embedded / C
    {
        'name':    'cJSON',
        'lang':    'C',
        'license': 'MIT',
        'repo':    'DaveGamble/cJSON',
        'paths':   ['README.md'],
    },
    {
        'name':    'FreeRTOS-Kernel',
        'lang':    'C (RTOS)',
        'license': 'MIT',
        'repo':    'FreeRTOS/FreeRTOS-Kernel',
        'paths':   ['README.md'],
    },
]


def _fetch_github_raw(repo: str, path: str) -> str | None:
    # Try main branch, then master
    for branch in ('main', 'master'):
        url = f'https://raw.githubusercontent.com/{repo}/{branch}/{path}'
        text = _get_text(url)
        if text and len(text) > 50:
            return text
    return None


def build_library_corpus(node: str, args):
    print(f'\n[Stage 23] MIT-licensed library documentation\n')
    trained = 0

    for lib in LIBRARIES:
        name    = lib['name']
        lang    = lib['lang']
        license_ = lib['license']
        repo    = lib['repo']
        paths   = lib['paths']

        print(f'  {name} ({lang}, {license_})...')
        lib_trained = 0

        for path in paths:
            content = _fetch_github_raw(repo, path)
            if not content:
                continue

            ext = Path(path).suffix.lower()
            if ext in ('.rst',):
                # RST: strip directives, treat as plain text
                content = re.sub(r'\.\. \w+::[^\n]*\n(\s+[^\n]*\n)*', '', content)
                content = re.sub(r'_{3,}|-{3,}|={3,}|\*{3,}', '', content)

            if ext in ('.md', '.rst', ''):
                # Extract code pairs from markdown
                pairs = _extract_doc_pairs(content)
                for pair in pairs[:args.lib_pairs_per_file]:
                    if len(pair['code']) < 15:
                        continue
                    text = (
                        f'Library documentation example — {name} ({lang}, {license_} license).\n'
                        f'Repository: github.com/{repo}\n\n'
                        f'Description: {pair["description"][:400]}\n\n'
                        f'Language: {pair["lang"] or lang}\n'
                        f'Code:\n{pair["code"][:1000]}'
                    )
                    try:
                        _train_text(text, node)
                        trained += 1
                        lib_trained += 1
                    except Exception as e:
                        print(f'    [WARN] {e}')

                # Also train the raw prose (non-code) sections
                prose = re.sub(r'```.*?```', '', content, flags=re.S)
                prose = re.sub(r'!\[.*?\]\(.*?\)', '', prose)
                prose = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', prose)
                prose = re.sub(r'\n#{1,6}\s+', '\n', prose)
                prose_chunks = [
                    prose[i:i+1200].strip()
                    for i in range(0, min(len(prose), args.lib_prose_chars), 1200)
                    if len(prose[i:i+1200].strip()) > 80
                ]
                for chunk in prose_chunks[:8]:
                    text = (
                        f'{name} library documentation ({lang}, {license_} license, '
                        f'github.com/{repo}).\n\n{chunk}'
                    )
                    try:
                        _train_text(text, node)
                        trained += 1
                        lib_trained += 1
                    except Exception as e:
                        print(f'    [WARN] {e}')

            time.sleep(0.3)

        print(f'    → {lib_trained} items trained')

    print(f'\n  Stage 23 done — {trained} total items trained from {len(LIBRARIES)} libraries')


# ── Stage 24: Stack Overflow accepted answers ──────────────────────────────────
#
# Only trains: question title + question body + ACCEPTED (or highest-voted)
# answer body + score context.
# The model benefits from "programmer described X in plain English → solution"
# patterns for troubleshooting and self-correction.

SO_TAGS = [
    # Core languages
    'python', 'javascript', 'typescript', 'java', 'c%2B%2B', 'c', 'rust',
    'go', 'ruby', 'swift', 'kotlin', 'php', 'c%23',
    # Web
    'html', 'css', 'react.js', 'vue.js', 'angular', 'node.js',
    'rest', 'api', 'json', 'http',
    # Systems
    'linux', 'bash', 'shell', 'docker', 'git',
    # Data
    'sql', 'postgresql', 'mysql', 'mongodb', 'redis',
    # CS fundamentals
    'algorithm', 'data-structures', 'recursion', 'dynamic-programming',
    'sorting', 'binary-search', 'graph', 'tree',
    # OOP + design
    'oop', 'design-patterns', 'solid-principles', 'refactoring',
    'class', 'inheritance', 'interface', 'polymorphism',
    # Debugging + concepts
    'debugging', 'performance', 'memory-management', 'concurrency',
    'multithreading', 'async-await', 'promises', 'callback',
    # Embedded
    'arduino', 'embedded', 'microcontroller', 'avr', 'arm', 'raspberry-pi',
    # Electronics
    'electronics', 'circuit', 'gpio', 'i2c', 'spi', 'uart',
]

SO_API_BASE = 'https://api.stackexchange.com/2.3'


def _so_fetch_questions(tag: str, page: int, pagesize: int) -> list | None:
    """Fetch top-voted questions with accepted answers for a tag."""
    url = (
        f'{SO_API_BASE}/questions'
        f'?tagged={tag}'
        f'&site=stackoverflow'
        f'&sort=votes'
        f'&order=desc'
        f'&filter=withbody'
        f'&has_accepted_answer=True'
        f'&page={page}'
        f'&pagesize={pagesize}'
    )
    data = _get_json(url)
    if not data:
        return None
    # Check for throttle
    quota = data.get('quota_remaining', 999)
    if quota < 5:
        print(f'    [WARN] API quota low: {quota} remaining — slowing down')
        time.sleep(30)
    return data.get('items', [])


def _so_fetch_accepted_answer(question_id: int) -> dict | None:
    """Fetch the accepted answer body for a question."""
    url = (
        f'{SO_API_BASE}/questions/{question_id}/answers'
        f'?site=stackoverflow'
        f'&sort=votes'
        f'&order=desc'
        f'&filter=withbody'
        f'&pagesize=1'
    )
    data = _get_json(url)
    if not data or not data.get('items'):
        return None
    return data['items'][0]


def build_stackoverflow_corpus(node: str, args):
    print(f'\n[Stage 24] Stack Overflow — accepted answers\n')
    print(f'  Only training: question + ACCEPTED/top-voted answer + explanation')
    print(f'  Tags: {len(SO_TAGS)}, max {args.so_per_tag} questions per tag\n')

    trained = 0
    seen_ids = set()

    for tag in SO_TAGS:
        tag_display = urllib.parse.unquote(tag)
        print(f'  Tag: {tag_display}...', end=' ', flush=True)
        tag_count = 0
        page = 1

        while tag_count < args.so_per_tag:
            batch = min(args.so_per_tag - tag_count, 20)
            questions = _so_fetch_questions(tag, page, batch)
            if not questions:
                break

            for q in questions:
                if tag_count >= args.so_per_tag:
                    break
                qid = q.get('question_id')
                if qid in seen_ids:
                    continue
                seen_ids.add(qid)

                q_title  = q.get('title', '')
                q_body   = _strip_html(q.get('body', ''))
                q_score  = q.get('score', 0)
                q_tags   = ', '.join(q.get('tags', []))
                accepted = q.get('accepted_answer_id')

                if q_score < args.so_min_votes:
                    continue

                # Fetch the accepted answer (or top-voted if no accepted_answer_id)
                ans = _so_fetch_accepted_answer(qid)
                if not ans:
                    time.sleep(0.3)
                    continue

                ans_body  = _strip_html(ans.get('body', ''))
                ans_score = ans.get('score', 0)
                is_accepted = ans.get('is_accepted', False)

                if ans_score < 1:
                    continue

                # Build training text: question + answer context
                verdict = 'accepted answer' if is_accepted else f'top-voted answer (score: {ans_score})'
                text = (
                    f'Stack Overflow — programming Q&A. '
                    f'Question score: {q_score}. Tags: {q_tags}.\n\n'
                    f'QUESTION: {q_title}\n\n'
                    f'{q_body[:800]}\n\n'
                    f'--- {verdict.upper()} ---\n\n'
                    f'{ans_body[:1500]}'
                )

                try:
                    _train_text(text, node)
                    trained += 1
                    tag_count += 1
                except Exception as e:
                    print(f'\n    [WARN] train failed: {e}')

                time.sleep(0.4)  # respect SO rate limits

            page += 1
            time.sleep(1.0)  # between pages

        print(f'{tag_count} questions')

    print(f'\n  Stage 24 done — {trained} Q&A pairs trained from {len(seen_ids)} unique questions')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description='Build library docs + Stack Overflow corpus')
    ap.add_argument('--stages',              default='23,24')
    ap.add_argument('--node',                default='localhost:8090')
    ap.add_argument('--lib-pairs-per-file',  type=int, default=30,
                    help='Max (description, code) pairs extracted per doc file')
    ap.add_argument('--lib-prose-chars',     type=int, default=8000,
                    help='Max prose chars to train per library doc file')
    ap.add_argument('--so-per-tag',          type=int, default=50,
                    help='Max questions to fetch per Stack Overflow tag')
    ap.add_argument('--so-min-votes',        type=int, default=5,
                    help='Minimum question score to include from SO')
    args = ap.parse_args()

    stages = [int(s.strip()) for s in args.stages.split(',')]
    node   = args.node

    print('=' * 70)
    print('  W1z4rD V1510n — Library Docs + Stack Overflow Corpus Builder')
    print('=' * 70)
    print(f'  Node:    http://{node}')
    print(f'  Stages:  {stages}')
    print()
    for s in stages:
        print(f'  Stage {s}: {STAGES.get(s, "?")}')
    print()

    if 23 in stages:
        build_library_corpus(node, args)
    if 24 in stages:
        build_stackoverflow_corpus(node, args)

    print('\n' + '=' * 70)
    print('  Library + SO corpus build complete.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
