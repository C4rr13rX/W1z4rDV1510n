#!/usr/bin/env python3
"""
build_bible_corpus.py -- Stage 29
World English Bible (WEB) -- complete 66-book corpus.

Source: scrollmapper/bible_databases on GitHub (public domain CSV)
  https://raw.githubusercontent.com/scrollmapper/bible_databases/master/csv/t_web.csv
  Columns: id, b (book 1-66), c (chapter), v (verse), t (text)

Fallback: bible-api.com chapter-by-chapter fetching.

Training format per chapter:
  "World English Bible -- Genesis, Chapter 1

   1 In the beginning, God created the heavens and the earth.
   2 The earth was formless and empty...
   ..."

Each chapter is trained as one item so verse numbers appear in context,
giving the model the full book -> chapter -> verse -> text association.

Also trains book-level overviews (genre, themes, key passages) so the model
understands the structure and purpose of each book.

Usage:
  python scripts/build_bible_corpus.py --node localhost:8090
  python scripts/build_bible_corpus.py --books Genesis,John,Psalms
"""

import argparse, csv, io, json, sys, time
from pathlib import Path
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_DATA_DIR = 'D:/w1z4rdv1510n-data'
DEFAULT_NODE     = 'localhost:8090'
UA = 'W1z4rDV1510n-Bible/1.0 (adamedsall@gmail.com; educational AI training)'

WEB_CSV_URL = ('https://raw.githubusercontent.com/scrollmapper/'
               'bible_databases/master/csv/t_web.csv')

STAGES = {29: 'World English Bible -- complete 66-book corpus, book/chapter/verse/text'}

# -- Book catalogue -------------------------------------------------------------
# (book_number, name, testament, abbrev)
BOOKS = [
    # -- Old Testament ---------------------------------------------------------
    (1,  'Genesis',          'OT', 'Gen'),
    (2,  'Exodus',           'OT', 'Exo'),
    (3,  'Leviticus',        'OT', 'Lev'),
    (4,  'Numbers',          'OT', 'Num'),
    (5,  'Deuteronomy',      'OT', 'Deu'),
    (6,  'Joshua',           'OT', 'Jos'),
    (7,  'Judges',           'OT', 'Jdg'),
    (8,  'Ruth',             'OT', 'Rut'),
    (9,  '1 Samuel',         'OT', '1Sa'),
    (10, '2 Samuel',         'OT', '2Sa'),
    (11, '1 Kings',          'OT', '1Ki'),
    (12, '2 Kings',          'OT', '2Ki'),
    (13, '1 Chronicles',     'OT', '1Ch'),
    (14, '2 Chronicles',     'OT', '2Ch'),
    (15, 'Ezra',             'OT', 'Ezr'),
    (16, 'Nehemiah',         'OT', 'Neh'),
    (17, 'Esther',           'OT', 'Est'),
    (18, 'Job',              'OT', 'Job'),
    (19, 'Psalms',           'OT', 'Psa'),
    (20, 'Proverbs',         'OT', 'Pro'),
    (21, 'Ecclesiastes',     'OT', 'Ecc'),
    (22, 'Song of Solomon',  'OT', 'Sng'),
    (23, 'Isaiah',           'OT', 'Isa'),
    (24, 'Jeremiah',         'OT', 'Jer'),
    (25, 'Lamentations',     'OT', 'Lam'),
    (26, 'Ezekiel',          'OT', 'Eze'),
    (27, 'Daniel',           'OT', 'Dan'),
    (28, 'Hosea',            'OT', 'Hos'),
    (29, 'Joel',             'OT', 'Joe'),
    (30, 'Amos',             'OT', 'Amo'),
    (31, 'Obadiah',          'OT', 'Oba'),
    (32, 'Jonah',            'OT', 'Jon'),
    (33, 'Micah',            'OT', 'Mic'),
    (34, 'Nahum',            'OT', 'Nah'),
    (35, 'Habakkuk',         'OT', 'Hab'),
    (36, 'Zephaniah',        'OT', 'Zep'),
    (37, 'Haggai',           'OT', 'Hag'),
    (38, 'Zechariah',        'OT', 'Zec'),
    (39, 'Malachi',          'OT', 'Mal'),
    # -- New Testament ---------------------------------------------------------
    (40, 'Matthew',          'NT', 'Mat'),
    (41, 'Mark',             'NT', 'Mrk'),
    (42, 'Luke',             'NT', 'Luk'),
    (43, 'John',             'NT', 'Jhn'),
    (44, 'Acts',             'NT', 'Act'),
    (45, 'Romans',           'NT', 'Rom'),
    (46, '1 Corinthians',    'NT', '1Co'),
    (47, '2 Corinthians',    'NT', '2Co'),
    (48, 'Galatians',        'NT', 'Gal'),
    (49, 'Ephesians',        'NT', 'Eph'),
    (50, 'Philippians',      'NT', 'Php'),
    (51, 'Colossians',       'NT', 'Col'),
    (52, '1 Thessalonians',  'NT', '1Th'),
    (53, '2 Thessalonians',  'NT', '2Th'),
    (54, '1 Timothy',        'NT', '1Ti'),
    (55, '2 Timothy',        'NT', '2Ti'),
    (56, 'Titus',            'NT', 'Tit'),
    (57, 'Philemon',         'NT', 'Phm'),
    (58, 'Hebrews',          'NT', 'Heb'),
    (59, 'James',            'NT', 'Jas'),
    (60, '1 Peter',          'NT', '1Pe'),
    (61, '2 Peter',          'NT', '2Pe'),
    (62, '1 John',           'NT', '1Jn'),
    (63, '2 John',           'NT', '2Jn'),
    (64, '3 John',           'NT', '3Jn'),
    (65, 'Jude',             'NT', 'Jud'),
    (66, 'Revelation',       'NT', 'Rev'),
]

BOOK_BY_NUM  = {b[0]: b for b in BOOKS}   # num -> (num, name, testament, abbrev)
BOOK_BY_NAME = {b[1]: b for b in BOOKS}   # name -> tuple

# Brief genre/theme overview per book for the book-level training item
BOOK_OVERVIEWS = {
    'Genesis':         'Law / Torah. Creation, Adam and Eve, Noah\'s flood, the patriarchs Abraham, Isaac, Jacob, Joseph. Themes: origins, covenant, promise.',
    'Exodus':          'Law / Torah. Israel\'s slavery in Egypt, Moses, the ten plagues, Passover, crossing the Red Sea, the Ten Commandments, the Tabernacle.',
    'Leviticus':       'Law / Torah. Priestly laws, sacrifices, cleanness codes, Day of Atonement, holiness laws. Themes: holiness, worship, atonement.',
    'Numbers':         'Law / Torah. Israel\'s wilderness wandering, census, Balaam. Themes: faith, obedience, consequences of rebellion.',
    'Deuteronomy':     'Law / Torah. Moses\' farewell speeches, restatement of the Law, covenant renewal, death of Moses. Themes: obedience, love, remembrance.',
    'Joshua':          'History. Conquest of Canaan under Joshua, division of the land. Themes: faithfulness, obedience, God\'s promises fulfilled.',
    'Judges':          'History. Cycle of apostasy, oppression, repentance, deliverance. Judges: Deborah, Gideon, Samson. Themes: sin cycle, grace, leadership.',
    'Ruth':            'History / Short story. Ruth\'s loyalty to Naomi, kinsman-redeemer Boaz. Themes: loyalty (hesed), redemption, inclusion.',
    '1 Samuel':        'History. Samuel the prophet, King Saul\'s rise and fall, young David. Themes: transition from judges to monarchy, obedience vs sacrifice.',
    '2 Samuel':        'History. David\'s reign, Bathsheba, Absalom\'s rebellion. Themes: covenant with David, consequences of sin, restoration.',
    '1 Kings':         'History. Solomon\'s wisdom and temple, the divided kingdom, Elijah vs prophets of Baal. Themes: wisdom, idolatry, prophetic ministry.',
    '2 Kings':         'History. Elisha, fall of Israel (722 BC), fall of Judah (586 BC). Themes: covenant consequences, faithfulness of prophets.',
    '1 Chronicles':    'History. Genealogies from Adam, David\'s preparations for the Temple. Themes: worship, divine appointment.',
    '2 Chronicles':    'History. Solomon\'s temple, kings of Judah, destruction and exile. Themes: worship, repentance, divine judgment.',
    'Ezra':            'History. Return from Babylonian exile, rebuilding the Temple under Zerubbabel and Ezra. Themes: restoration, law.',
    'Nehemiah':        'History. Nehemiah rebuilds Jerusalem\'s walls, covenant renewal. Themes: prayer, perseverance, community reform.',
    'Esther':          'History. Esther saves Jewish people from Haman\'s plot. Themes: providence, courage, Jewish identity.',
    'Job':             'Wisdom / Poetry. Job\'s suffering, friends\' arguments, God\'s answer from the whirlwind. Themes: theodicy, suffering, sovereignty.',
    'Psalms':          'Poetry / Worship. 150 songs and prayers covering lament, praise, thanksgiving, wisdom, and messianic expectation. Themes: worship, trust, suffering, redemption.',
    'Proverbs':        'Wisdom / Poetry. Practical wisdom for daily life: family, work, speech, wealth, integrity. Themes: wisdom, fear of the Lord.',
    'Ecclesiastes':    'Wisdom / Philosophy. "Vanity of vanities" -- the Preacher\'s search for meaning. Themes: meaning, mortality, fear God and keep His commandments.',
    'Song of Solomon': 'Poetry / Love. Celebration of romantic love between a bride and groom. Themes: love, beauty, commitment.',
    'Isaiah':          'Prophecy. Judgment on Israel and nations, Servant Songs, messianic prophecies (Isaiah 53), comfort for exiles. Themes: holiness, salvation.',
    'Jeremiah':        'Prophecy. Jeremiah\'s call, Judah\'s sins, fall of Jerusalem, the New Covenant (Jer 31). Themes: judgment, weeping prophet, hope.',
    'Lamentations':    'Poetry. Five poems mourning the destruction of Jerusalem. Themes: grief, confession, hope in God\'s mercies (3:22-23).',
    'Ezekiel':         'Prophecy. Visions (chariot throne, valley of dry bones, new temple). Themes: glory of God, individual responsibility, restoration.',
    'Daniel':          'Prophecy / Apocalyptic. Daniel in Babylon, fiery furnace, lions\' den, four kingdoms vision, Son of Man. Themes: faithfulness, sovereignty, apocalypse.',
    'Hosea':           'Prophecy. Hosea\'s marriage as metaphor for God\'s relationship with unfaithful Israel. Themes: covenant love, spiritual adultery, redemption.',
    'Joel':            'Prophecy. Locust plague, call to repentance, promise of the Spirit (Joel 2:28, cited in Acts 2). Themes: Day of the Lord.',
    'Amos':            'Prophecy. Social justice, judgment on Israel\'s exploitation of the poor. Themes: justice, righteousness.',
    'Obadiah':         'Prophecy. Judgment against Edom for betraying Israel. Shortest OT book. Themes: pride, divine justice.',
    'Jonah':           'Narrative / Prophecy. Jonah\'s flight, the great fish, Nineveh\'s repentance. Themes: mercy, mission, divine compassion beyond Israel.',
    'Micah':           'Prophecy. Judgment and hope, messianic birthplace (Bethlehem, Mic 5:2), justice (Mic 6:8). Themes: justice, mercy, walking humbly.',
    'Nahum':           'Prophecy. Judgment on Nineveh (Assyria). Themes: God\'s justice, comfort for the oppressed.',
    'Habakkuk':        'Prophecy. Dialogue with God about evil: "the just shall live by faith" (Hab 2:4). Themes: theodicy, faith, sovereignty.',
    'Zephaniah':       'Prophecy. Day of the Lord, judgment, restoration. Themes: judgment, humility, restoration.',
    'Haggai':          'Prophecy. Exhortation to rebuild the Temple after the return from exile. Themes: priorities, God\'s presence.',
    'Zechariah':       'Prophecy. Eight visions, messianic prophecies (triumphal entry, 30 pieces of silver, piercing). Themes: restoration, messiah.',
    'Malachi':         'Prophecy. Final OT book: covenant faithfulness, tithing, coming messenger (Elijah). Themes: faithfulness, preparation for the Messiah.',
    'Matthew':         'Gospel. Jesus as the Jewish Messiah, fulfillment of OT prophecy. Sermon on the Mount, kingdom parables, Great Commission. Audience: Jewish Christians.',
    'Mark':            'Gospel. Fastest-paced Gospel, Jesus as Servant and suffering Son of God. Action-focused, frequent use of "immediately". Audience: Roman.',
    'Luke':            'Gospel. Most complete life of Christ, written to Theophilus. Parables of Prodigal Son, Good Samaritan. Women, outcasts, Holy Spirit. Audience: Gentile.',
    'John':            'Gospel. Theological Gospel: seven "I am" statements, signs, prologue (In the beginning was the Word). Audience: universal. John 3:16.',
    'Acts':            'History. Acts of the Holy Spirit through the apostles. Pentecost, Paul\'s missionary journeys, spread of the Church to Rome.',
    'Romans':          'Epistle. Paul\'s systematic theology: sin, justification by faith, sanctification, election, Christian living. Romans 3:23, 5:8, 8:28, 12:1-2.',
    '1 Corinthians':   'Epistle. Church problems: divisions, immorality, spiritual gifts, resurrection. Famous: love chapter (ch.13), resurrection (ch.15).',
    '2 Corinthians':   'Epistle. Paul\'s defense of his ministry, thorn in the flesh, generous giving, strength in weakness.',
    'Galatians':       'Epistle. "Magna Carta of Christian liberty." Justification by faith alone, not works of the law. Fruit of the Spirit (5:22-23).',
    'Ephesians':       'Epistle. Spiritual blessings in Christ, unity in the body, armor of God (6:10-18), marriage and family.',
    'Philippians':     'Epistle. Joy in all circumstances, humility (ch.2 Christ hymn), contentment, "I can do all things through Christ" (4:13).',
    'Colossians':      'Epistle. Supremacy of Christ (1:15-20), warning against false philosophy, Christian living.',
    '1 Thessalonians': 'Epistle. Paul\'s affection for the church, comfort about the dead, Second Coming (4:13-18).',
    '2 Thessalonians': 'Epistle. Clarification about the Day of the Lord, man of lawlessness, call to work.',
    '1 Timothy':       'Epistle. Pastoral letter: church order, qualifications for leaders, prayer, false teachers.',
    '2 Timothy':       'Epistle. Paul\'s final letter before execution: preach the word, Scripture is God-breathed (3:16), finish the race.',
    'Titus':           'Epistle. Pastoral letter: church leadership in Crete, sound doctrine, good works.',
    'Philemon':        'Epistle. Paul\'s appeal for runaway slave Onesimus. Themes: forgiveness, reconciliation, Christian brotherhood.',
    'Hebrews':         'Epistle. Jesus as superior to angels, Moses, Aaron -- the perfect high priest. Hall of Faith (ch.11). Themes: faith, perseverance.',
    'James':           'Epistle. Practical Christianity: faith and works, taming the tongue, prayer, care for the poor. "Faith without works is dead" (2:26).',
    '1 Peter':         'Epistle. Hope in suffering, living as strangers in the world, submission, "living stones" (2:4-5).',
    '2 Peter':         'Epistle. Warning against false teachers, promise of Christ\'s return, Scripture\'s divine origin (1:20-21).',
    '1 John':          'Epistle. Tests of genuine faith: obedience, love, belief in Christ. "God is love" (4:8). Assurance of salvation (5:13).',
    '2 John':          'Epistle. Warning against false teachers, walk in truth and love.',
    '3 John':          'Epistle. Commendation of Gaius, warning about Diotrephes.',
    'Jude':            'Epistle. Warning against apostasy and false teachers, contend for the faith.',
    'Revelation':      'Apocalyptic. John\'s visions: seven churches, seals/trumpets/bowls, fall of Babylon, millennium, New Jerusalem. Themes: victory of Christ, perseverance.',
}


# -- HTTP session ---------------------------------------------------------------

def _make_session():
    s = requests.Session()
    s.mount('https://', HTTPAdapter(max_retries=3))
    s.mount('http://',  HTTPAdapter(max_retries=3))
    s.headers['User-Agent'] = UA
    return s

def _train(text, node, session):
    try:
        r = session.post(f'http://{node}/media/train',
                         data=json.dumps({'modality': 'text', 'text': text}),
                         headers={'Content-Type': 'application/json'}, timeout=15)
        return r.status_code == 200
    except Exception as e:
        print(f'  [WARN] train: {e}', flush=True)
        return False


# -- WEB download ---------------------------------------------------------------

def _download_web_csv(session) -> dict:
    """
    Download WEB CSV and return nested dict:
      {book_num: {chapter_num: {verse_num: text}}}
    """
    print('  Downloading World English Bible CSV...', flush=True)
    r = session.get(WEB_CSV_URL, timeout=60)
    r.raise_for_status()

    bible = {}
    reader = csv.reader(io.StringIO(r.text))
    next(reader, None)   # skip header
    for row in reader:
        if len(row) < 5:
            continue
        _, b, c, v, *t_parts = row
        text = ','.join(t_parts).strip().strip('"')
        b, c, v = int(b), int(c), int(v)
        bible.setdefault(b, {}).setdefault(c, {})[v] = text

    books_found = len(bible)
    verses_total = sum(len(vv) for ch in bible.values() for vv in ch.values())
    print(f'  Loaded {books_found} books, {verses_total:,} verses', flush=True)
    return bible


def _fetch_chapter_fallback(book_name: str, chapter: int, session) -> dict:
    """Fallback: fetch one chapter from bible-api.com if CSV unavailable."""
    slug = book_name.replace(' ', '+')
    url  = f'https://bible-api.com/{slug}+{chapter}?translation=web'
    try:
        r = session.get(url, timeout=15)
        if not r.ok:
            return {}
        verses = {}
        for v in r.json().get('verses', []):
            verses[v['verse']] = v['text'].strip()
        return verses
    except Exception:
        return {}


# -- Training -------------------------------------------------------------------

def _train_book_overview(book_name: str, testament: str, node: str, session):
    overview = BOOK_OVERVIEWS.get(book_name, '')
    text = (
        f'World English Bible -- Book Overview\n\n'
        f'Book: {book_name}\n'
        f'Testament: {"Old Testament" if testament == "OT" else "New Testament"}\n'
    )
    if overview:
        text += f'Overview: {overview}\n'
    _train(text, node, session)


def _train_chapter(book_name: str, chapter: int, verses: dict,
                   node: str, session) -> bool:
    """
    Train one full chapter as a single text item with inline verse numbers.
    Format: "World English Bible -- Genesis, Chapter 1\n\n1 In the beginning..."
    """
    lines = [f'World English Bible -- {book_name}, Chapter {chapter}', '']
    for v in sorted(verses):
        lines.append(f'{v} {verses[v]}')
    text = '\n'.join(lines)
    return _train(text, node, session)


# -- Stage 29 -------------------------------------------------------------------

def build_bible_corpus(out_dir: Path, node: str,
                       books_filter: list = None) -> list:
    """Stage 29: Complete World English Bible -- all 66 books, chapter by chapter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / 'checkpoint.json'

    # Load checkpoint (set of trained "BOOKNUM:CHAP" keys)
    trained: set = set()
    if ckpt_path.exists():
        try:
            trained = set(json.loads(ckpt_path.read_text(encoding='utf-8')))
            print(f'  Resuming -- {len(trained)} chapters already trained', flush=True)
        except Exception:
            pass

    def save_ckpt():
        ckpt_path.write_text(json.dumps(sorted(trained)), encoding='utf-8')

    api_session   = _make_session()
    train_session = _make_session()
    items = []
    ok_chapters = 0
    ok_books    = 0

    # Download full WEB CSV (one request for the whole Bible)
    bible = {}
    try:
        bible = _download_web_csv(api_session)
    except Exception as e:
        print(f'  [WARN] CSV download failed ({e}); will use fallback API per chapter', flush=True)

    # Determine which books to process
    target_books = [b for b in BOOKS
                    if books_filter is None or b[1] in books_filter]

    for book_num, book_name, testament, abbrev in target_books:
        print(f'  [{book_name}]', end=' ', flush=True)

        # Book-level overview
        ov_key = f'{book_num}:overview'
        if ov_key not in trained:
            _train_book_overview(book_name, testament, node, train_session)
            trained.add(ov_key)
            ok_books += 1
            time.sleep(0.08)

        # Chapter-by-chapter
        book_data = bible.get(book_num, {})
        chapters  = sorted(book_data.keys()) if book_data else []

        if not chapters and not bible:
            # Fallback: fetch chapters 1-150 for Psalms, 1-50 for others
            max_ch = 150 if book_num == 19 else 50
            chapters = list(range(1, max_ch + 1))

        chap_count = 0
        for chap in chapters:
            key = f'{book_num}:{chap}'
            if key in trained:
                chap_count += 1
                continue

            verses = book_data.get(chap, {})
            if not verses and not bible:
                verses = _fetch_chapter_fallback(book_name, chap, api_session)
                time.sleep(0.3)

            if not verses:
                continue

            if _train_chapter(book_name, chap, verses, node, train_session):
                trained.add(key)
                ok_chapters += 1
                chap_count  += 1
                time.sleep(0.08)

            if ok_chapters % 50 == 0:
                save_ckpt()

        print(f'{chap_count} chapters', flush=True)
        items.append({
            'stage': 29,
            'book': book_name,
            'testament': testament,
            'chapters': chap_count,
            'type': 'bible_chapter',
            'modality': 'text',
            'tags': ['bible', 'web', testament.lower(), book_name.lower()],
        })

    save_ckpt()
    total_v = sum(
        len(vv)
        for bnum in bible
        if books_filter is None or BOOK_BY_NUM.get(bnum, ('',''))[1] in (books_filter or [b[1] for b in BOOKS])
        for vv in bible[bnum].values()
    )
    print(f'\n  WEB corpus: {ok_books} book overviews, {ok_chapters} chapters trained', flush=True)
    return items


# -- Entry point ----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description='Stage 29: World English Bible corpus',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join(f'  {n}: {d}' for n, d in STAGES.items()),
    )
    ap.add_argument('--stages',   default='29')
    ap.add_argument('--node',     default=DEFAULT_NODE,
                    help='Node host:port (default: %(default)s)')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR)
    ap.add_argument('--books',    default=None,
                    help='Comma-separated book names to run (default: all 66)')
    args = ap.parse_args()

    stages = {int(s.strip()) for s in args.stages.split(',')}
    books  = ([b.strip() for b in args.books.split(',')]
              if args.books else None)
    if books:
        unknown = [b for b in books if b not in BOOK_BY_NAME]
        if unknown:
            print(f'Unknown book(s): {unknown}')
            print('Valid names:', [b[1] for b in BOOKS])
            sys.exit(1)

    train_dir = Path(args.data_dir) / 'training'

    print('World English Bible Corpus Builder -- Stage 29')
    print(f'  Node   : {args.node}')
    print(f'  Books  : {books or "all 66"}')

    all_items: dict = {}

    if 29 in stages:
        print('\n[Stage 29] World English Bible')
        all_items[29] = build_bible_corpus(
            train_dir / 'stage29_bible',
            args.node,
            books_filter=books,
        )

    manifest = [item for items in all_items.values() for item in items]
    mpath = train_dir / 'stage29_manifest.json'
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'\nManifest -> {mpath}')
    print('Done.')


if __name__ == '__main__':
    main()
