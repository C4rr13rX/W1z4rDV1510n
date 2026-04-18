#!/usr/bin/env python3
"""
build_libretexts_corpus.py — Stage 25
Comprehensive LibreTexts open textbook corpus: all 13 domains, all bookshelves.

Uses the Mindtouch CXone REST API to walk every book → chapter → section,
extracts plain text from page HTML, and trains via the node's /media/train endpoint.

Resume-safe: each domain checkpoints trained page IDs to disk.
All 13 LibreTexts libraries:
  math, phys, chem, bio, geo, eng, stats, biz, socialsci, human, med, workforce, k12

Usage:
  python scripts/build_libretexts_corpus.py --node localhost:8090
  python scripts/build_libretexts_corpus.py --domains biz,human --max-pages 1000
"""

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = 'D:/w1z4rdv1510n-data'
DEFAULT_NODE     = 'localhost:8090'

UA = ('W1z4rDV1510n-LibreTexts/1.0 '
      '(https://github.com/C4rr13rX/W1z4rDV1510n; adamedsall@gmail.com; '
      'educational AI training; polite crawler)')

DOMAINS = {
    'math':       'https://math.libretexts.org/@api/0.1',
    'phys':       'https://phys.libretexts.org/@api/0.1',
    'chem':       'https://chem.libretexts.org/@api/0.1',
    'bio':        'https://bio.libretexts.org/@api/0.1',
    'geo':        'https://geo.libretexts.org/@api/0.1',
    'eng':        'https://eng.libretexts.org/@api/0.1',
    'stats':      'https://stats.libretexts.org/@api/0.1',
    'biz':        'https://biz.libretexts.org/@api/0.1',
    'socialsci':  'https://socialsci.libretexts.org/@api/0.1',
    'human':      'https://human.libretexts.org/@api/0.1',
    'med':        'https://med.libretexts.org/@api/0.1',
    'workforce':  'https://workforce.libretexts.org/@api/0.1',
    'k12':        'https://k12.libretexts.org/@api/0.1',
}

DOMAIN_LABELS = {
    'math':      'Mathematics',
    'phys':      'Physics',
    'chem':      'Chemistry',
    'bio':       'Biology',
    'geo':       'Geosciences',
    'eng':       'Engineering',
    'stats':     'Statistics & Data Science',
    'biz':       'Business',
    'socialsci': 'Social Sciences',
    'human':     'Humanities',
    'med':       'Medicine & Health Sciences',
    'workforce': 'Workforce & Technical Education',
    'k12':       'K-12 Education',
}

STAGES = {25: 'LibreTexts comprehensive corpus — all 13 domains, all bookshelves'}

# Walk depth: 0=Bookshelves root, 1=Book, 2=Unit/Chapter, 3=Section, 4=Subsection
MAX_DEPTH    = 5
MIN_CONTENT  = 300   # minimum plain-text chars to bother training
MAX_CONTENT  = 7000  # chars per page sent to train (cap to avoid huge prompts)
RATE_DELAY   = 0.40  # seconds between LibreTexts API calls (be polite)
TRAIN_DELAY  = 0.10  # seconds between /media/train calls
CKPT_EVERY   = 50    # save checkpoint every N trained pages


# ── HTTP session ───────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=4,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET', 'POST'],
    )
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.mount('http://',  HTTPAdapter(max_retries=retry))
    s.headers.update({'User-Agent': UA, 'Accept': 'application/json'})
    return s


# ── HTML / text helpers ────────────────────────────────────────────────────────

_TAG_RE  = re.compile(r'<[^>]+>')
_WS_RE   = re.compile(r'[ \t\r\n]+')
_MATH_RE = re.compile(r'\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$[^\n$]+\$', re.S)

_HTML_ENTITIES = {
    '&nbsp;': ' ', '&#160;': ' ', '&lt;': '<', '&gt;': '>',
    '&amp;': '&', '&quot;': '"', '&#39;': "'", '&apos;': "'",
    '&mdash;': '—', '&ndash;': '–', '&hellip;': '...',
    '&ldquo;': '"', '&rdquo;': '"', '&lsquo;': "'", '&rsquo;': "'",
}

def _html_to_text(html: str) -> str:
    text = _MATH_RE.sub(' [math expression] ', html)
    text = _TAG_RE.sub(' ', text)
    for ent, ch in _HTML_ENTITIES.items():
        text = text.replace(ent, ch)
    # Decode remaining numeric entities
    text = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), text)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    text = _WS_RE.sub(' ', text).strip()
    return text


# ── LibreTexts API helpers ─────────────────────────────────────────────────────

def _get_json(session: requests.Session, url: str, params: dict = None, timeout: int = 25):
    """Fetch JSON from LibreTexts API; returns None on any error."""
    p = {'dream.out.format': 'json'}
    if params:
        p.update(params)
    try:
        r = session.get(url, params=p, timeout=timeout)
        if r.status_code in (401, 403, 404):
            return None
        r.raise_for_status()
        return r.json()
    except requests.exceptions.JSONDecodeError:
        return None
    except Exception as e:
        print(f'    [WARN] GET {url}: {type(e).__name__}: {e}', flush=True)
        return None


def _as_list(val) -> list:
    """Mindtouch returns a dict when there's one item, list when many."""
    if val is None:
        return []
    if isinstance(val, dict):
        return [val]
    if isinstance(val, list):
        return val
    return []


def _get_subpages(session: requests.Session, base_api: str, page_ref: str) -> list:
    """
    Return list of direct subpages for page_ref (numeric ID or '=Path').
    Each item: {'id': str, 'title': str}
    """
    data = _get_json(session, f'{base_api}/pages/{page_ref}/subpages')
    time.sleep(RATE_DELAY)
    if not data:
        return []
    pages = _as_list(data.get('page'))
    result = []
    for p in pages:
        pid   = str(p.get('@id', '')).strip()
        title = (p.get('title') or p.get('@title') or '').strip()
        if pid and pid.isdigit():
            result.append({'id': pid, 'title': title or f'Page {pid}'})
    return result


def _get_page_text(session: requests.Session, base_api: str, page_id: str) -> str:
    """Return stripped plain text of a page's content body, or '' if empty/error."""
    data = _get_json(session, f'{base_api}/pages/{page_id}/contents')
    time.sleep(RATE_DELAY)
    if not data:
        return ''
    body = data.get('body', '')
    if isinstance(body, list):
        # Multiple body sections (revision history, etc.)
        parts = []
        for b in body:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(b.get('#text', '') or b.get('body', '') or '')
        body = ' '.join(parts)
    elif isinstance(body, dict):
        body = body.get('#text', '') or body.get('body', '') or ''
    if not isinstance(body, str):
        return ''
    return _html_to_text(body)


# ── Training helper ────────────────────────────────────────────────────────────

def _train(text: str, node: str, session: requests.Session) -> bool:
    """POST text to node /media/train. Returns True on success."""
    try:
        r = session.post(
            f'http://{node}/media/train',
            data=json.dumps({'text': text}),
            headers={'Content-Type': 'application/json'},
            timeout=15,
        )
        return r.status_code == 200
    except Exception as e:
        print(f'    [WARN] train: {e}', flush=True)
        return False


# ── Domain walker ──────────────────────────────────────────────────────────────

def _walk_domain(api_session: requests.Session, train_session: requests.Session,
                 domain: str, base_api: str, label: str, node: str,
                 out_dir: Path, max_pages: int) -> int:
    """
    DFS walk of all bookshelves in one domain.
    Trains each content page once, resumes from checkpoint.
    Returns number of new pages trained this run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'
    trained: set = set()

    if ckpt_path.exists():
        try:
            trained = set(json.loads(ckpt_path.read_text(encoding='utf-8')))
            print(f'  [{domain}] Resuming — {len(trained)} pages already trained',
                  flush=True)
        except Exception:
            pass

    def save_ckpt():
        ckpt_path.write_text(json.dumps(sorted(trained)), encoding='utf-8')

    ok = 0

    # Enumerate top-level books from Bookshelves
    books = _get_subpages(api_session, base_api, '=Bookshelves')
    if not books:
        print(f'  [{domain}] No books found (API unreachable?)', flush=True)
        return 0

    print(f'  [{domain}] {len(books)} books in bookshelves', flush=True)

    # DFS stack: (page_id, breadcrumb_list, depth)
    # Reversed so first book is processed first when popping from end
    stack = deque()
    for book in reversed(books):
        stack.append((book['id'], [label, book['title']], 1))

    visited: set = set()

    while stack and ok < max_pages:
        page_id, breadcrumb, depth = stack.pop()

        if page_id in visited:
            continue
        visited.add(page_id)

        # Fetch and train page content (skip if already trained this or a prior run)
        if page_id not in trained:
            text = _get_page_text(api_session, base_api, page_id)
            if text and len(text) >= MIN_CONTENT:
                crumb = ' > '.join(breadcrumb)
                train_str = (
                    f'LibreTexts open textbook — {label}\n'
                    f'Topic: {crumb}\n\n'
                    f'{text[:MAX_CONTENT]}'
                )
                if _train(train_str, node, train_session):
                    trained.add(page_id)
                    ok += 1
                    time.sleep(TRAIN_DELAY)
                    if ok % CKPT_EVERY == 0:
                        save_ckpt()
                        print(f'  [{domain}] {ok} pages trained...', flush=True)
                    if ok >= max_pages:
                        break

        # Push subpages onto stack (DFS continues into this page's children)
        if depth < MAX_DEPTH:
            subpages = _get_subpages(api_session, base_api, page_id)
            for sp in reversed(subpages):
                if sp['id'] not in visited:
                    stack.append((sp['id'], breadcrumb + [sp['title']], depth + 1))

    save_ckpt()
    print(f'  [{domain}] Done — {ok} new pages trained ({len(trained)} total)',
          flush=True)
    return ok


# ── Stage 25 orchestrator ──────────────────────────────────────────────────────

def build_libretexts_corpus(out_dir: Path, node: str,
                            max_pages_per_domain: int = 5000,
                            domains_filter: list = None) -> list:
    """Stage 25: Walk all 13 LibreTexts domains, train every textbook page."""
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [(d, DOMAINS[d]) for d in DOMAINS
               if domains_filter is None or d in domains_filter]

    api_session   = _make_session()
    train_session = _make_session()
    items = []

    print(f'\n  LibreTexts corpus: {len(targets)} domain(s), '
          f'up to {max_pages_per_domain} pages each', flush=True)

    for domain, base_api in targets:
        label = DOMAIN_LABELS[domain]
        print(f'\n  ══ {label} ({domain}) ══', flush=True)
        n = _walk_domain(
            api_session, train_session,
            domain, base_api, label, node,
            out_dir / domain, max_pages_per_domain,
        )
        items.append({
            'stage': 25,
            'domain': domain,
            'label': label,
            'pages_trained': n,
            'type': 'libretexts_page',
            'modality': 'text',
            'tags': ['textbook', 'open-access', 'libretexts', domain],
        })

    total = sum(i['pages_trained'] for i in items)
    print(f'\n  LibreTexts corpus total: {total} pages across {len(items)} domain(s)',
          flush=True)
    return items


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description='Stage 25: Comprehensive LibreTexts open-textbook corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(f'  {n}: {d}' for n, d in STAGES.items()),
    )
    ap.add_argument('--stages',   default='25',
                    help='Comma-separated stage numbers (only 25 currently)')
    ap.add_argument('--node',     default=DEFAULT_NODE,
                    help='Node address host:port (default: %(default)s)')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                    help='Root data directory (default: %(default)s)')
    ap.add_argument('--max-pages', type=int, default=5000,
                    help='Max pages per domain (default %(default)s; 0 = unlimited)')
    ap.add_argument('--domains', default=None,
                    help=('Comma-separated subset of domains to run, e.g. biz,human '
                          '(default: all 13: ' + ','.join(DOMAINS) + ')'))
    args = ap.parse_args()

    stages  = {int(s.strip()) for s in args.stages.split(',')}
    domains = ([d.strip() for d in args.domains.split(',')]
               if args.domains else None)
    if domains:
        unknown = [d for d in domains if d not in DOMAINS]
        if unknown:
            print(f'Unknown domain(s): {unknown}. Valid: {list(DOMAINS)}')
            sys.exit(1)

    max_pages = args.max_pages if args.max_pages > 0 else 10_000_000
    train_dir = Path(args.data_dir) / 'training'

    print('LibreTexts Corpus Builder — Stage 25')
    print(f'  Node     : {args.node}')
    print(f'  Data dir : {args.data_dir}')
    print(f'  Stages   : {sorted(stages)}')
    print(f'  Domains  : {domains or "all 13"}')
    print(f'  Max pages: {max_pages:,}')

    all_items: dict = {}

    if 25 in stages:
        print('\n[Stage 25] LibreTexts comprehensive corpus')
        all_items[25] = build_libretexts_corpus(
            train_dir / 'stage25_libretexts',
            args.node,
            max_pages_per_domain=max_pages,
            domains_filter=domains,
        )

    # Write manifest
    manifest = [item for items in all_items.values() for item in items]
    mpath = train_dir / 'stage25_manifest.json'
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nManifest → {mpath}')
    print('Done.')


if __name__ == '__main__':
    main()
