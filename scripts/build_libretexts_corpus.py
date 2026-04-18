#!/usr/bin/env python3
"""
build_libretexts_corpus.py — Stage 25
Comprehensive LibreTexts open textbook corpus: all 13 domains, all bookshelves.

Uses the Mindtouch CXone REST API to walk every book → chapter → section.
For each page:
  • Trains the full text via {"modality": "text", "text": "..."}
  • Extracts every <figure>/<img> block, downloads the image, and trains it
    paired with its caption + surrounding paragraph context via
    {"modality": "page", "data_b64": "...", "text": "..."} so the image pixels
    and their textual description co-activate in the same Hebbian pass.

Resume-safe: each domain checkpoints trained page IDs to disk.
All 13 LibreTexts libraries:
  math, phys, chem, bio, geo, eng, stats, biz, socialsci, human, med, workforce, k12

Usage:
  python scripts/build_libretexts_corpus.py --node localhost:8090
  python scripts/build_libretexts_corpus.py --domains biz,human --max-pages 1000
  python scripts/build_libretexts_corpus.py --no-images   # text-only (faster)
"""

import argparse
import base64
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
MAX_DEPTH      = 5
MIN_CONTENT    = 300       # min plain-text chars to bother training a page
MAX_CONTENT    = 7000      # chars per page text (cap to avoid huge payloads)
MAX_IMG_BYTES  = 2_097_152 # 2 MiB per image
RATE_DELAY     = 0.40      # seconds between LibreTexts API calls
TRAIN_DELAY    = 0.10      # seconds between /media/train calls
CKPT_EVERY     = 50        # checkpoint every N pages


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


# ── HTML parsing ───────────────────────────────────────────────────────────────

_MATH_RE   = re.compile(r'\\\(.*?\\\)|\\\[.*?\\\]|\$\$.*?\$\$|\$[^\n$]+\$', re.S)
_TAG_RE    = re.compile(r'<[^>]+>')
_WS_RE     = re.compile(r'[ \t\r\n]+')
_FIGURE_RE = re.compile(r'<figure\b[^>]*>(.*?)</figure\s*>', re.S | re.I)
_IMG_RE    = re.compile(r'<img\b[^>]*/?>',  re.I)
_FIGCAP_RE = re.compile(r'<figcaption\b[^>]*>(.*?)</figcaption\s*>', re.S | re.I)
_SRC_RE    = re.compile(r'\bsrc=["\']([^"\']+)["\']',  re.I)
_ALT_RE    = re.compile(r'\balt=["\']([^"\']*)["\']',  re.I)

_HTML_ENTITIES = {
    '&nbsp;': ' ', '&#160;': ' ', '&lt;': '<', '&gt;': '>',
    '&amp;': '&', '&quot;': '"', '&#39;': "'", '&apos;': "'",
    '&mdash;': '—', '&ndash;': '–', '&hellip;': '...',
    '&ldquo;': '"', '&rdquo;': '"', '&lsquo;': "'", '&rsquo;': "'",
}

def _strip(html: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    t = _MATH_RE.sub(' [math] ', html)
    t = _TAG_RE.sub(' ', t)
    for ent, ch in _HTML_ENTITIES.items():
        t = t.replace(ent, ch)
    t = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), t)
    t = re.sub(r'&[a-zA-Z]+;', ' ', t)
    return _WS_RE.sub(' ', t).strip()


def _parse_page(html: str) -> tuple:
    """
    Split an HTML page into plain text + a list of figure dicts.

    Returns:
      plain_text : str  — full page text with [Figure N: caption] placeholders
      figures    : list — each item:
                     { 'src': str, 'alt': str, 'caption': str,
                       'before': str (≤600 chars of text before the figure),
                       'after':  str (≤400 chars of text after the figure) }
    """
    # Split HTML at figure block boundaries so we can collect before/after context.
    parts = _FIGURE_RE.split(html)
    # split() with one group produces [before, group1, after, group2, ...]
    # i.e. even indices are text, odd indices are figure inner HTML

    text_segments: list[str] = []
    figures: list[dict] = []

    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Plain HTML segment — strip standalone <img> tags too
            def _replace_standalone_img(m):
                alt_m = _ALT_RE.search(m.group(0))
                alt = alt_m.group(1).strip() if alt_m else ''
                # Track src-only images (no caption) so we can train them if wanted
                src_m = _SRC_RE.search(m.group(0))
                if src_m and not _is_skip_src(src_m.group(1)):
                    figures.append({
                        'src': src_m.group(1),
                        'alt': alt,
                        'caption': alt,
                        '_part_idx': len(text_segments),
                        '_inline': True,
                    })
                return f' [Image: {alt}] ' if alt else ' '

            cleaned = _IMG_RE.sub(_replace_standalone_img, part)
            text_segments.append(_strip(cleaned))
        else:
            # Figure inner HTML
            figure_html = part
            img_m = _IMG_RE.search(figure_html)
            if not img_m:
                text_segments.append('')
                continue
            src_m = _SRC_RE.search(img_m.group(0))
            if not src_m:
                text_segments.append('')
                continue
            src = src_m.group(1)
            alt_m = _ALT_RE.search(img_m.group(0))
            alt = alt_m.group(1).strip() if alt_m else ''
            cap_m = _FIGCAP_RE.search(figure_html)
            caption = _strip(cap_m.group(1)) if cap_m else alt

            n = len([f for f in figures if not f.get('_inline')]) + 1
            figures.append({
                'src': src,
                'alt': alt,
                'caption': caption,
                '_part_idx': len(text_segments),
                '_inline': False,
            })
            text_segments.append(f'[Figure {n}: {caption}]' if caption else '[Figure]')

    # Attach before/after context to each figure using the surrounding text segments
    for fig in figures:
        idx = fig.pop('_part_idx')
        before = ' '.join(text_segments[:idx])
        after  = ' '.join(text_segments[idx + 1:])
        fig['before'] = before[-600:].strip()
        fig['after']  = after[:400].strip()

    plain_text = _WS_RE.sub(' ', ' '.join(text_segments)).strip()
    return plain_text, figures


def _is_skip_src(src: str) -> bool:
    """Return True for image URLs we should not download (SVG, icons, spacers)."""
    low = src.lower().split('?')[0]
    if low.endswith('.svg'):
        return True
    skip_keywords = ('spacer', 'blank', 'pixel', 'dot.', 'icon', '1x1', 'logo',
                     'favicon', 'spinner', 'loading', 'arrow')
    return any(kw in low for kw in skip_keywords)


# ── LibreTexts API helpers ─────────────────────────────────────────────────────

def _get_json(session: requests.Session, url: str, params: dict = None,
              timeout: int = 25):
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
    if val is None:   return []
    if isinstance(val, dict):  return [val]
    if isinstance(val, list):  return val
    return []


def _get_subpages(session: requests.Session, base_api: str, page_ref: str) -> list:
    data = _get_json(session, f'{base_api}/pages/{page_ref}/subpages')
    time.sleep(RATE_DELAY)
    if not data:
        return []
    result = []
    for p in _as_list(data.get('page')):
        pid   = str(p.get('@id', '')).strip()
        title = (p.get('title') or p.get('@title') or '').strip()
        if pid and pid.isdigit():
            result.append({'id': pid, 'title': title or f'Page {pid}'})
    return result


def _get_raw_html(session: requests.Session, base_api: str, page_id: str) -> str:
    """Fetch raw HTML body of a page from the LibreTexts API."""
    data = _get_json(session, f'{base_api}/pages/{page_id}/contents')
    time.sleep(RATE_DELAY)
    if not data:
        return ''
    body = data.get('body', '')
    if isinstance(body, list):
        parts = []
        for b in body:
            if isinstance(b, str):
                parts.append(b)
            elif isinstance(b, dict):
                parts.append(b.get('#text', '') or b.get('body', '') or '')
        body = ' '.join(parts)
    elif isinstance(body, dict):
        body = body.get('#text', '') or body.get('body', '') or ''
    return body if isinstance(body, str) else ''


def _download_image(session: requests.Session, url: str) -> bytes | None:
    """Download an image; returns bytes or None if unsuitable/too large."""
    if _is_skip_src(url):
        return None
    try:
        r = session.get(url, timeout=20)
        if not r.ok:
            return None
        ct = r.headers.get('content-type', '')
        if 'svg' in ct or 'html' in ct or 'text' in ct:
            return None
        if len(r.content) > MAX_IMG_BYTES:
            return None
        return r.content if r.content else None
    except Exception:
        return None


# ── Training helpers ───────────────────────────────────────────────────────────

def _train_text(text: str, node: str, session: requests.Session) -> bool:
    try:
        r = session.post(
            f'http://{node}/media/train',
            data=json.dumps({'modality': 'text', 'text': text}),
            headers={'Content-Type': 'application/json'},
            timeout=15,
        )
        return r.status_code == 200
    except Exception as e:
        print(f'    [WARN] train text: {e}', flush=True)
        return False


def _train_figure(img_bytes: bytes, fig: dict, crumb: str, label: str,
                  node: str, session: requests.Session) -> bool:
    """
    Train one figure image paired with its caption + surrounding context
    as a 'page' modality co-activation (image pixels ↔ text labels).
    """
    caption = (fig.get('caption') or fig.get('alt') or '').strip()
    before  = fig.get('before', '').strip()
    after   = fig.get('after',  '').strip()

    parts = [f'LibreTexts {label} — figure.', f'Topic: {crumb}']
    if caption:
        parts.append(f'Caption: {caption}')
    if before:
        parts.append(f'Context before: ...{before}')
    if after:
        parts.append(f'Context after: {after}...')

    payload = {
        'modality': 'page',
        'data_b64': base64.b64encode(img_bytes).decode('ascii'),
        'text': '\n'.join(parts),
    }
    try:
        r = session.post(
            f'http://{node}/media/train',
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'},
            timeout=30,
        )
        return r.status_code == 200
    except Exception as e:
        print(f'    [WARN] train figure: {e}', flush=True)
        return False


# ── Domain walker ──────────────────────────────────────────────────────────────

def _walk_domain(api_session: requests.Session, train_session: requests.Session,
                 domain: str, base_api: str, label: str, node: str,
                 out_dir: Path, max_pages: int, train_images: bool) -> int:
    """
    DFS walk all bookshelves for one domain.
    Each page: train text + train every figure as image+text pair.
    Returns number of new pages trained this run.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'
    trained: set = set()

    if ckpt_path.exists():
        try:
            trained = set(json.loads(ckpt_path.read_text(encoding='utf-8')))
            print(f'  [{domain}] Resuming — {len(trained)} pages already done',
                  flush=True)
        except Exception:
            pass

    def save_ckpt():
        ckpt_path.write_text(json.dumps(sorted(trained)), encoding='utf-8')

    ok = 0

    books = _get_subpages(api_session, base_api, '=Bookshelves')
    if not books:
        print(f'  [{domain}] No books found (API unreachable?)', flush=True)
        return 0

    print(f'  [{domain}] {len(books)} books found', flush=True)

    # DFS stack: (page_id, breadcrumb_list, depth)
    stack = deque()
    for book in reversed(books):
        stack.append((book['id'], [label, book['title']], 1))

    visited: set = set()

    while stack and ok < max_pages:
        page_id, breadcrumb, depth = stack.pop()

        if page_id in visited:
            continue
        visited.add(page_id)

        if page_id not in trained:
            raw = _get_raw_html(api_session, base_api, page_id)
            if raw:
                text, figures = _parse_page(raw)
                crumb = ' > '.join(breadcrumb)

                # Train page text
                if text and len(text) >= MIN_CONTENT:
                    train_str = (
                        f'LibreTexts open textbook — {label}\n'
                        f'Topic: {crumb}\n\n'
                        f'{text[:MAX_CONTENT]}'
                    )
                    if _train_text(train_str, node, train_session):
                        trained.add(page_id)
                        ok += 1
                        time.sleep(TRAIN_DELAY)
                        if ok % CKPT_EVERY == 0:
                            save_ckpt()
                            print(f'  [{domain}] {ok} pages trained...', flush=True)

                # Train figures (image + caption + context)
                if train_images and figures:
                    figs_trained = 0
                    for fig in figures:
                        src = fig.get('src', '')
                        if not src:
                            continue
                        img = _download_image(api_session, src)
                        if img:
                            _train_figure(img, fig, crumb, label, node,
                                          train_session)
                            figs_trained += 1
                            time.sleep(TRAIN_DELAY)
                    if figs_trained:
                        pass  # logged in summary

                if ok >= max_pages:
                    break

        # Push children
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
                            domains_filter: list = None,
                            train_images: bool = True) -> list:
    """Stage 25: Walk all 13 LibreTexts domains, train text + figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [(d, DOMAINS[d]) for d in DOMAINS
               if domains_filter is None or d in domains_filter]

    api_session   = _make_session()
    train_session = _make_session()
    items = []

    print(f'\n  LibreTexts corpus: {len(targets)} domain(s), '
          f'up to {max_pages_per_domain} pages each, '
          f'images={"on" if train_images else "off"}', flush=True)

    for domain, base_api in targets:
        label = DOMAIN_LABELS[domain]
        print(f'\n  ══ {label} ({domain}) ══', flush=True)
        n = _walk_domain(
            api_session, train_session,
            domain, base_api, label, node,
            out_dir / domain, max_pages_per_domain, train_images,
        )
        items.append({
            'stage': 25,
            'domain': domain,
            'label': label,
            'pages_trained': n,
            'type': 'libretexts_page',
            'modality': 'text+image',
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
    ap.add_argument('--stages',    default='25')
    ap.add_argument('--node',      default=DEFAULT_NODE,
                    help='Node host:port (default: %(default)s)')
    ap.add_argument('--data-dir',  default=DEFAULT_DATA_DIR)
    ap.add_argument('--max-pages', type=int, default=5000,
                    help='Max pages per domain (0 = unlimited, default %(default)s)')
    ap.add_argument('--domains',   default=None,
                    help=('Comma-separated subset, e.g. biz,human '
                          '(default: all 13: ' + ','.join(DOMAINS) + ')'))
    ap.add_argument('--no-images', action='store_true',
                    help='Skip image download/training (text-only, faster)')
    args = ap.parse_args()

    stages  = {int(s.strip()) for s in args.stages.split(',')}
    domains = ([d.strip() for d in args.domains.split(',')]
               if args.domains else None)
    if domains:
        unknown = [d for d in domains if d not in DOMAINS]
        if unknown:
            print(f'Unknown domain(s): {unknown}. Valid: {list(DOMAINS)}')
            sys.exit(1)

    max_pages    = args.max_pages if args.max_pages > 0 else 10_000_000
    train_images = not args.no_images
    train_dir    = Path(args.data_dir) / 'training'

    print('LibreTexts Corpus Builder — Stage 25')
    print(f'  Node     : {args.node}')
    print(f'  Data dir : {args.data_dir}')
    print(f'  Stages   : {sorted(stages)}')
    print(f'  Domains  : {domains or "all 13"}')
    print(f'  Max pages: {max_pages:,}')
    print(f'  Images   : {"yes" if train_images else "no (--no-images)"}')

    all_items: dict = {}

    if 25 in stages:
        print('\n[Stage 25] LibreTexts comprehensive corpus')
        all_items[25] = build_libretexts_corpus(
            train_dir / 'stage25_libretexts',
            args.node,
            max_pages_per_domain=max_pages,
            domains_filter=domains,
            train_images=train_images,
        )

    manifest = [item for items in all_items.values() for item in items]
    mpath = train_dir / 'stage25_manifest.json'
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nManifest → {mpath}')
    print('Done.')


if __name__ == '__main__':
    main()
