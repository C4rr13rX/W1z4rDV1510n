#!/usr/bin/env python3
"""
build_biodiversity_corpus.py -- Stage 26
Plant and animal visual-identification corpus via multimodal Hebbian training.

Sources:
  * iNaturalist API  -- species list, photos, observation counts
  * iNaturalist taxa -- full Linnaean taxonomy via ancestor_ids batch lookup
  * Wikipedia REST   -- species description (opening paragraph)

For each species the node trains:
  1. {"modality": "page", "data_b64": <photo>, "text": <name + taxonomy + description>}
     -- image pixels co-activate with the binomial name, common name, kingdom,
       order, family, and description in one Hebbian pass.
  2. {"modality": "text", "text": <full taxonomy block>}
     -- text-only pass so taxonomy associations form even for photo-less species.

Covers 11 iconic taxon groups (sorted by observation count so best-known
species are trained first):
  Plantae . Aves . Insecta . Mammalia . Reptilia . Amphibia
  Actinopterygii . Arachnida . Mollusca . Fungi . Animalia

Usage:
  python scripts/build_biodiversity_corpus.py --node localhost:8090
  python scripts/build_biodiversity_corpus.py --max-per-group 500
  python scripts/build_biodiversity_corpus.py --no-images   # text-only, faster
"""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -- Configuration --------------------------------------------------------------

DEFAULT_DATA_DIR = 'D:/w1z4rdv1510n-data'
DEFAULT_NODE     = 'localhost:8090'

UA = ('W1z4rDV1510n-Biodiversity/1.0 '
      '(https://github.com/C4rr13rX/W1z4rDV1510n; adamedsall@gmail.com; '
      'educational AI training; polite crawler)')

STAGES = {26: 'Biodiversity visual-ID corpus -- plants, animals, fungi via iNaturalist + Wikipedia'}

# iNaturalist iconic_taxa group names (in processing order)
ICONIC_GROUPS = [
    'Plantae',        # flowering plants, ferns, mosses
    'Aves',           # birds
    'Insecta',        # insects
    'Mammalia',       # mammals
    'Fungi',          # fungi, lichens
    'Reptilia',       # reptiles
    'Arachnida',      # spiders, scorpions, mites
    'Actinopterygii', # ray-finned fish
    'Amphibia',       # frogs, salamanders
    'Mollusca',       # snails, clams, octopus
    'Animalia',       # other animals (cnidarians, worms, crustaceans, etc.)
]

GROUP_LABELS = {
    'Plantae':        'Plants',
    'Aves':           'Birds',
    'Insecta':        'Insects',
    'Mammalia':       'Mammals',
    'Fungi':          'Fungi & Lichens',
    'Reptilia':       'Reptiles',
    'Arachnida':      'Spiders & Arachnids',
    'Actinopterygii': 'Fish (ray-finned)',
    'Amphibia':       'Amphibians',
    'Mollusca':       'Molluscs',
    'Animalia':       'Other Animals',
}

INAT_BASE  = 'https://api.inaturalist.org/v1'
WIKI_BASE  = 'https://en.wikipedia.org/api/rest_v1/page/summary'

MAX_IMG_BYTES  = 2_097_152   # 2 MiB
INAT_DELAY     = 0.35        # s between iNaturalist API calls
WIKI_DELAY     = 0.12        # s between Wikipedia calls
IMG_DELAY      = 0.15        # s between image downloads
TRAIN_DELAY    = 0.10        # s between /media/train calls
ANCS_BATCH     = 40          # max ancestor IDs per batch lookup
CKPT_EVERY     = 100         # checkpoint every N species trained

TAXONOMY_RANKS = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus']


# -- HTTP sessions --------------------------------------------------------------

def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=['GET', 'POST'])
    s.mount('https://', HTTPAdapter(max_retries=retry))
    s.mount('http://',  HTTPAdapter(max_retries=retry))
    s.headers['User-Agent'] = UA
    return s


# -- Ancestor taxonomy cache ----------------------------------------------------

_anc_cache: dict[int, dict] = {}   # id -> {rank, name}

def _fetch_ancestors(ids: list[int], session: requests.Session):
    """Batch-fetch ancestor taxa by ID and populate _anc_cache."""
    missing = [i for i in ids if i not in _anc_cache]
    if not missing:
        return
    for batch_start in range(0, len(missing), ANCS_BATCH):
        batch = missing[batch_start: batch_start + ANCS_BATCH]
        url = f'{INAT_BASE}/taxa/{",".join(str(i) for i in batch)}'
        try:
            r = session.get(url, params={'per_page': len(batch)}, timeout=20)
            r.raise_for_status()
            for t in r.json().get('results', []):
                _anc_cache[t['id']] = {'rank': t.get('rank', ''), 'name': t.get('name', '')}
        except Exception as e:
            print(f'    [WARN] ancestor lookup: {e}', flush=True)
        time.sleep(INAT_DELAY)


def _build_taxonomy(taxon: dict, session: requests.Session) -> dict:
    """
    Return dict of rank->name for the main Linnaean ranks using ancestor_ids.
    Caches lookups so repeated ranks (e.g. Plantae) are free.
    """
    anc_ids = taxon.get('ancestor_ids', [])
    if anc_ids:
        _fetch_ancestors(anc_ids, session)
    taxonomy = {}
    for aid in anc_ids:
        info = _anc_cache.get(aid, {})
        rank = info.get('rank', '')
        if rank in TAXONOMY_RANKS:
            taxonomy[rank] = info.get('name', '')
    # Always set species from the taxon itself
    taxonomy['species'] = taxon.get('name', '')
    return taxonomy


# -- Wikipedia helper -----------------------------------------------------------

_wiki_cache: dict[str, str] = {}

def _wiki_summary(name: str, session: requests.Session) -> str:
    """Fetch opening paragraph from Wikipedia for a scientific name."""
    if name in _wiki_cache:
        return _wiki_cache[name]
    title = name.replace(' ', '_')
    try:
        r = session.get(f'{WIKI_BASE}/{title}', timeout=15)
        if r.status_code == 404:
            _wiki_cache[name] = ''
            return ''
        r.raise_for_status()
        extract = r.json().get('extract', '')
        _wiki_cache[name] = extract[:1200]
        return _wiki_cache[name]
    except Exception:
        _wiki_cache[name] = ''
        return ''
    finally:
        time.sleep(WIKI_DELAY)


# -- Image download -------------------------------------------------------------

def _download_image(url: str, session: requests.Session) -> bytes | None:
    if not url:
        return None
    try:
        r = session.get(url, timeout=20)
        if not r.ok:
            return None
        ct = r.headers.get('content-type', '')
        if 'html' in ct or 'text' in ct or 'svg' in ct:
            return None
        if len(r.content) > MAX_IMG_BYTES:
            return None
        return r.content or None
    except Exception:
        return None
    finally:
        time.sleep(IMG_DELAY)


# -- Training helpers -----------------------------------------------------------

def _build_train_text(taxon: dict, taxonomy: dict, description: str) -> str:
    """Build the text payload for a species training item."""
    sci_name    = taxon.get('name', '')
    common_name = taxon.get('preferred_common_name', '') or sci_name
    group_label = GROUP_LABELS.get(taxon.get('iconic_taxon_name', ''), 'Life')

    # Binomial breakdown
    parts = sci_name.split()
    genus   = parts[0] if parts else ''
    epithet = parts[1] if len(parts) > 1 else ''

    lines = [
        f'Visual identification: {common_name} ({sci_name})',
        f'Scientific name (binomial nomenclature): {sci_name}',
        f'Common name: {common_name}',
        f'Group: {group_label}',
    ]

    # Linnaean taxonomy
    for rank in TAXONOMY_RANKS:
        if taxonomy.get(rank):
            lines.append(f'{rank.capitalize()}: {taxonomy[rank]}')

    if genus and epithet:
        lines.append(f'Genus: {genus}  |  Species epithet: {epithet}')

    if description:
        lines.append(f'\nDescription: {description}')

    return '\n'.join(lines)


def _train_page(img_bytes: bytes | None, text: str,
                node: str, session: requests.Session) -> bool:
    if img_bytes:
        payload = {
            'modality': 'page',
            'data_b64': base64.b64encode(img_bytes).decode('ascii'),
            'text': text,
        }
    else:
        payload = {'modality': 'text', 'text': text}
    try:
        r = session.post(
            f'http://{node}/media/train',
            data=json.dumps(payload),
            headers={'Content-Type': 'application/json'},
            timeout=30,
        )
        return r.status_code == 200
    except Exception as e:
        print(f'    [WARN] train: {e}', flush=True)
        return False


# -- Group processor ------------------------------------------------------------

def _process_group(group: str, out_dir: Path, node: str,
                   max_species: int, train_images: bool,
                   inat_session: requests.Session,
                   train_session: requests.Session) -> int:
    """
    Fetch up to max_species from one iNaturalist iconic group,
    train each as image+text (or text-only).  Returns count of species trained.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'
    state = {'trained': [], 'next_page': 1}

    if ckpt_path.exists():
        try:
            state = json.loads(ckpt_path.read_text(encoding='utf-8'))
            print(f'  [{group}] Resuming -- {len(state["trained"])} already done',
                  flush=True)
        except Exception:
            pass

    trained_ids: set = set(state['trained'])

    def save():
        state['trained'] = sorted(trained_ids)
        ckpt_path.write_text(json.dumps(state), encoding='utf-8')

    ok = 0
    page = state.get('next_page', 1)

    while ok < max_species:
        params = {
            'iconic_taxa[]': group,
            'quality_grade': 'research',
            'per_page':      500,
            'page':          page,
        }
        try:
            r = inat_session.get(f'{INAT_BASE}/observations/species_counts',
                                 params=params, timeout=30)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f'  [{group}] page {page} fetch error: {e}', flush=True)
            break
        time.sleep(INAT_DELAY)

        results = data.get('results', [])
        if not results:
            break   # no more species

        for item in results:
            if ok >= max_species:
                break
            taxon = item.get('taxon', {})
            tid   = taxon.get('id')
            if not tid or tid in trained_ids:
                continue

            # Taxonomy (cached)
            taxonomy = _build_taxonomy(taxon, inat_session)

            # Wikipedia description
            description = _wiki_summary(taxon.get('name', ''), inat_session)

            # Build text payload
            text = _build_train_text(taxon, taxonomy, description)

            # Download photo
            img_bytes = None
            if train_images:
                photo_url = (taxon.get('default_photo') or {}).get('medium_url', '')
                if photo_url:
                    img_bytes = _download_image(photo_url, inat_session)

            # Train
            if _train_page(img_bytes, text, node, train_session):
                trained_ids.add(tid)
                ok += 1
                time.sleep(TRAIN_DELAY)
                if ok % CKPT_EVERY == 0:
                    state['next_page'] = page
                    save()
                    print(f'  [{group}] {ok} species trained...', flush=True)

        page += 1
        state['next_page'] = page

    save()
    print(f'  [{group}] Done -- {ok} species trained ({len(trained_ids)} total)',
          flush=True)
    return ok


# -- Stage 26 orchestrator ------------------------------------------------------

def build_biodiversity_corpus(out_dir: Path, node: str,
                              max_per_group: int = 2000,
                              groups_filter: list = None,
                              train_images: bool = True) -> list:
    """Stage 26: Multimodal species identification corpus."""
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = [g for g in ICONIC_GROUPS
              if groups_filter is None or g in groups_filter]

    inat_session  = _make_session()
    train_session = _make_session()
    items = []

    print(f'\n  Biodiversity corpus: {len(groups)} group(s), '
          f'up to {max_per_group} species each, '
          f'images={"on" if train_images else "off"}', flush=True)

    for group in groups:
        label = GROUP_LABELS[group]
        print(f'\n  == {label} ({group}) ==', flush=True)
        n = _process_group(
            group, out_dir / group.lower(), node,
            max_per_group, train_images,
            inat_session, train_session,
        )
        items.append({
            'stage': 26,
            'group': group,
            'label': label,
            'species_trained': n,
            'type': 'species_identification',
            'modality': 'text+image',
            'tags': ['biodiversity', 'species', 'taxonomy', group.lower()],
        })

    total = sum(i['species_trained'] for i in items)
    print(f'\n  Biodiversity corpus total: {total} species across {len(items)} group(s)',
          flush=True)
    return items


# -- Entry point ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Stage 26: Biodiversity visual-identification corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(f'  {n}: {d}' for n, d in STAGES.items()),
    )
    ap.add_argument('--stages',        default='26')
    ap.add_argument('--node',          default=DEFAULT_NODE,
                    help='Node host:port (default: %(default)s)')
    ap.add_argument('--data-dir',      default=DEFAULT_DATA_DIR)
    ap.add_argument('--max-per-group', type=int, default=2000,
                    help='Max species per iconic group (default %(default)s; 0=unlimited)')
    ap.add_argument('--groups',        default=None,
                    help=('Comma-separated subset, e.g. Plantae,Aves '
                          f'(default: all {len(ICONIC_GROUPS)}: '
                          + ','.join(ICONIC_GROUPS) + ')'))
    ap.add_argument('--no-images',     action='store_true',
                    help='Text-only training (skip photo downloads)')
    args = ap.parse_args()

    stages  = {int(s.strip()) for s in args.stages.split(',')}
    groups  = ([g.strip() for g in args.groups.split(',')]
               if args.groups else None)
    if groups:
        unknown = [g for g in groups if g not in ICONIC_GROUPS]
        if unknown:
            print(f'Unknown group(s): {unknown}. Valid: {ICONIC_GROUPS}')
            sys.exit(1)

    max_per = args.max_per_group if args.max_per_group > 0 else 10_000_000
    train_dir = Path(args.data_dir) / 'training'

    print('Biodiversity Corpus Builder -- Stage 26')
    print(f'  Node        : {args.node}')
    print(f'  Data dir    : {args.data_dir}')
    print(f'  Stages      : {sorted(stages)}')
    print(f'  Groups      : {groups or f"all {len(ICONIC_GROUPS)}"}')
    print(f'  Max/group   : {max_per:,}')
    print(f'  Images      : {"yes" if not args.no_images else "no"}')

    all_items: dict = {}

    if 26 in stages:
        print('\n[Stage 26] Biodiversity visual-identification corpus')
        all_items[26] = build_biodiversity_corpus(
            train_dir / 'stage26_biodiversity',
            args.node,
            max_per_group=max_per,
            groups_filter=groups,
            train_images=not args.no_images,
        )

    manifest = [item for items in all_items.values() for item in items]
    mpath = train_dir / 'stage26_manifest.json'
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nManifest -> {mpath}')
    print('Done.')


if __name__ == '__main__':
    main()
