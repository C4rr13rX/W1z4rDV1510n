#!/usr/bin/env python3
# coding: utf-8
"""
build_foreign_language_corpus.py -- Comprehensive multilingual training corpus

Stages:
  36 -- Project Gutenberg: native-language literary texts (20 languages)
  37 -- Project Gutenberg: English language-learning books, grammars, dictionaries
  38 -- Wikibooks: language courses for all target languages
  39 -- Built-in: scripts/alphabets, vocabulary, grammar rules, phrase pairs (20 languages)
  40 -- Wiktionary: multilingual definitions, translations, etymology

Languages: zh (Chinese/Mandarin), ru (Russian), it (Italian), fr (French), es (Spanish),
           hi (Hindi), ur (Urdu), ar (Arabic), ja (Japanese), de (German), pt (Portuguese),
           ko (Korean), tr (Turkish), pl (Polish), nl (Dutch), sw (Swahili), vi (Vietnamese),
           fa (Persian/Farsi), bn (Bengali), th (Thai)

Usage:
  python scripts/build_foreign_language_corpus.py --stages 36,37,38,39,40 --node localhost:8090
  python scripts/build_foreign_language_corpus.py --stages 39 --node localhost:8090
"""

import argparse, html, json, re, sys, time, urllib.request, urllib.error, urllib.parse
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = Path('D:/w1z4rdv1510n-data')

STAGES = {
    36: 'Project Gutenberg: native-language literary texts (20 languages)',
    37: 'Project Gutenberg: English language-learning books, grammars, dictionaries',
    38: 'Wikibooks: language courses for all target languages',
    39: 'Built-in: scripts, vocabulary, grammar, phrase pairs (20 languages)',
    40: 'Wiktionary: multilingual definitions and etymology',
}

LANGUAGES = {
    'zh': 'Chinese (Mandarin)',
    'ru': 'Russian',
    'it': 'Italian',
    'fr': 'French',
    'es': 'Spanish',
    'hi': 'Hindi',
    'ur': 'Urdu',
    'ar': 'Arabic',
    'ja': 'Japanese',
    'de': 'German',
    'pt': 'Portuguese',
    'ko': 'Korean',
    'tr': 'Turkish',
    'pl': 'Polish',
    'nl': 'Dutch',
    'sw': 'Swahili',
    'vi': 'Vietnamese',
    'fa': 'Persian (Farsi)',
    'bn': 'Bengali',
    'th': 'Thai',
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


def _gutenberg_text_url(book: dict) -> str | None:
    fmts = book.get('formats', {})
    for key in ('text/plain; charset=utf-8', 'text/plain; charset=us-ascii', 'text/plain'):
        if key in fmts:
            return fmts[key]
    return None


def _extract_gutenberg_passages(raw: str, max_chars=40000) -> list[str]:
    start = re.search(r'\*\*\* START OF (THE|THIS) PROJECT GUTENBERG', raw, re.I)
    end   = re.search(r'\*\*\* END OF (THE|THIS) PROJECT GUTENBERG',   raw, re.I)
    if start:
        raw = raw[start.end():]
    if end:
        raw = raw[:end.start()]
    raw = raw[:max_chars]
    paras = [p.strip() for p in re.split(r'\n\s*\n', raw) if len(p.strip()) > 80]
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


# -- Stage 36: Project Gutenberg native-language texts -------------------------

# (lang_code, max_books, topic_query_or_empty)
GUTENBERG_NATIVE = [
    ('zh', 20, ''),
    ('ru', 20, ''),
    ('fr', 15, ''),
    ('es', 15, ''),
    ('it', 15, ''),
    ('de', 15, ''),
    ('pt', 12, ''),
    ('ja', 12, ''),
    ('ar', 10, ''),
    ('hi', 8,  ''),
    ('nl', 8,  ''),
    ('pl', 8,  ''),
    ('ko', 6,  ''),
    ('tr', 6,  ''),
    ('fa', 6,  ''),
    ('vi', 5,  ''),
    ('bn', 5,  ''),
    ('sw', 4,  ''),
    ('ur', 4,  ''),
    ('th', 4,  ''),
]


def build_gutenberg_native_corpus(node: str, args):
    print(f'\n[Stage 36] Project Gutenberg -- native-language literary texts\n')
    trained = 0
    seen_ids: set = set()

    for lang_code, max_books, topic in GUTENBERG_NATIVE:
        lang_name = LANGUAGES.get(lang_code, lang_code)
        print(f'  {lang_name} ({lang_code}): up to {max_books} books...')
        page = 1
        fetched = 0
        while fetched < max_books:
            url = f'https://gutendex.com/books/?languages={lang_code}&page={page}'
            if topic:
                url += f'&topic={urllib.parse.quote(topic)}'
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
                if not raw or len(raw) < 300:
                    continue
                passages = _extract_gutenberg_passages(raw, max_chars=args.gutenberg_chars)
                header = (
                    f'Project Gutenberg -- {lang_name} text: "{title}" by {authors}.\n'
                    f'Language: {lang_name} ({lang_code}). Source: gutenberg.org\n\n'
                )
                for passage in passages[:args.gutenberg_passages]:
                    try:
                        _train_text(header + passage, node)
                        trained += 1
                    except Exception as e:
                        print(f'    [WARN] train failed: {e}')
                fetched += 1
                print(f'    [{fetched}/{max_books}] {title[:55]} -- {len(passages)} passages')
                time.sleep(0.6)
            if not data.get('next'):
                break
            page += 1
            time.sleep(0.4)

    print(f'  Stage 36 done -- {trained} passages from {len(seen_ids)} books')


# -- Stage 37: Project Gutenberg language-learning books -----------------------

GUTENBERG_LL_SEARCHES = [
    ('french grammar',        12),
    ('french language',       10),
    ('spanish grammar',       12),
    ('spanish language',      10),
    ('italian grammar',       10),
    ('german grammar',        10),
    ('german language',        8),
    ('portuguese grammar',     8),
    ('chinese language',      10),
    ('chinese characters',     8),
    ('japanese language',      8),
    ('russian language',       8),
    ('arabic language',        8),
    ('hindi language',         6),
    ('latin grammar',         10),
    ('greek grammar',          8),
    ('hebrew grammar',         6),
    ('turkish language',       6),
    ('swahili language',       5),
    ('persian language',       5),
    ('dutch language',         5),
    ('polish language',        5),
    ('korean language',        4),
    ('vietnamese language',    4),
    ('bengali language',       4),
    ('urdu language',          4),
    ('thai language',          4),
    ('dictionary',            15),
    ('phrase book',           10),
    ('language lessons',      10),
    ('pronunciation guide',    8),
    ('vocabulary',            10),
    ('polyglot',               6),
    ('translation',           10),
    ('linguistics',           10),
    ('comparative grammar',    8),
]


def build_gutenberg_ll_corpus(node: str, args):
    print(f'\n[Stage 37] Project Gutenberg -- language-learning books\n')
    trained = 0
    seen_ids: set = set()

    for topic, max_books in GUTENBERG_LL_SEARCHES:
        print(f'  Topic: "{topic}" (max {max_books})...')
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
                if not raw or len(raw) < 300:
                    continue
                passages = _extract_gutenberg_passages(raw, max_chars=args.gutenberg_chars)
                header = (
                    f'Project Gutenberg -- Language Learning: "{title}" by {authors}.\n'
                    f'Category: {topic}. Source: gutenberg.org\n\n'
                )
                for passage in passages[:args.gutenberg_passages]:
                    try:
                        _train_text(header + passage, node)
                        trained += 1
                    except Exception as e:
                        print(f'    [WARN] {e}')
                fetched += 1
                print(f'    [{fetched}/{max_books}] {title[:55]}')
                time.sleep(0.5)
            if not data.get('next'):
                break
            page += 1
            time.sleep(0.3)

    print(f'  Stage 37 done -- {trained} passages from {len(seen_ids)} books')


# -- Stage 38: Wikibooks language courses --------------------------------------

WIKIBOOKS_LANG_COURSES = [
    # Core language courses (en.wikibooks.org)
    'French', 'French/Lessons', 'French/Grammar', 'French/Vocabulary',
    'Spanish', 'Spanish/Lessons', 'Spanish/Grammar', 'Spanish/Vocabulary',
    'Italian', 'Italian/Grammar', 'Italian/Vocabulary',
    'German', 'German/Grammar', 'German/Vocabulary',
    'Mandarin Chinese', 'Mandarin Chinese/Grammar', 'Mandarin Chinese/Vocabulary',
    'Chinese (Mandarin)/Lessons', 'Chinese (Mandarin)/Grammar',
    'Japanese', 'Japanese/Grammar', 'Japanese/Vocabulary', 'Japanese/Kanji',
    'Russian', 'Russian/Grammar', 'Russian/Vocabulary', 'Russian/Alphabet',
    'Arabic', 'Arabic/Grammar', 'Arabic/Vocabulary',
    'Hindi', 'Hindi/Grammar', 'Hindi/Vocabulary',
    'Portuguese', 'Portuguese/Grammar', 'Portuguese/Vocabulary',
    'Korean', 'Korean/Grammar', 'Korean/Vocabulary',
    'Turkish', 'Turkish/Grammar', 'Turkish/Vocabulary',
    'Polish', 'Polish/Grammar', 'Polish/Vocabulary',
    'Dutch', 'Dutch/Grammar', 'Dutch/Vocabulary',
    'Swahili', 'Swahili/Grammar', 'Swahili/Vocabulary',
    'Vietnamese', 'Vietnamese/Grammar', 'Vietnamese/Vocabulary',
    'Persian', 'Persian/Grammar', 'Persian/Vocabulary',
    'Bengali', 'Bengali/Grammar', 'Bengali/Vocabulary',
    'Thai', 'Thai/Grammar', 'Thai/Vocabulary',
    'Urdu', 'Urdu/Grammar', 'Urdu/Vocabulary',
    # Additional useful topics
    'Latin', 'Ancient Greek', 'Hebrew', 'Ancient Egyptian',
    'Linguistics/Index', 'Linguistics',
    'Language Learning with Latin', 'Esperanto',
    'IPA (International Phonetic Alphabet)',
    'Introduction to Linguistics',
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


def build_wikibooks_lang_corpus(node: str, args):
    print(f'\n[Stage 38] Wikibooks -- language courses\n')
    trained = 0
    for title in WIKIBOOKS_LANG_COURSES:
        text = _wikibooks_extract(title)
        if not text:
            print(f'  SKIP {title}')
            time.sleep(0.2)
            continue
        chunks = [text[i:i+1400] for i in range(0, min(len(text), args.wikibooks_chars), 1400)]
        header = f'Wikibooks -- Language Course: "{title}". Free educational content.\n\n'
        count = 0
        for chunk in chunks:
            if len(chunk.strip()) < 80:
                continue
            try:
                _train_text(header + chunk.strip(), node)
                trained += 1
                count += 1
            except Exception as e:
                print(f'  [WARN] {e}')
        print(f'  {title[:55]}: {count} chunks')
        time.sleep(0.4)
    print(f'  Stage 38 done -- {trained} chunks trained')


# -- Stage 39: Built-in multilingual data --------------------------------------
# Embedded vocabulary, grammar, scripts, and phrase pairs for 20 languages.
# This stage ensures script-level competency even without network access.

BUILTIN_LANG_DATA: dict[str, list[str]] = {}

# -- Chinese / Mandarin (zh) ----------------------------------------------------
BUILTIN_LANG_DATA['zh'] = [

"""MANDARIN CHINESE -- 普通话 (Pǔtōnghuà) / 汉语 (Hànyǔ)

TONES (声调 shēngdiào) -- Mandarin has 4 tones plus a neutral tone:
  First tone  (ā ē ī ō ū): high and level       -- māo (猫) cat
  Second tone (á e í ó ú): rising                -- máo (毛) hair/fur
  Third tone  (ǎ ě ǐ ǒ ǔ): low dipping then rise -- mǎo (卯) a time period
  Fourth tone (à e ì ò ù): falling sharply       -- mào (帽) hat
  Neutral tone (no mark):  short and unstressed  -- ma (吗) question particle

Tone change rule: Two 3rd tones in a row -- the first becomes 2nd tone.
  nǐ hǎo -> ní hǎo (你好 Hello)

PINYIN -- Romanization system for Mandarin pronunciation:
  Initials: b p m f / d t n l / g k h / j q x / zh ch sh r / z c s / y w
  Finals:   a o e i u u / ai ei ao ou / an en ang eng ong / ia ie iao iou ian iang iong / ua uo uai uei uan uang / ue uan
  Special:  zh=j sound (retroflex), x=sh sound (palatal), q=ch (palatal), u=German u""",

"""MANDARIN CHINESE -- Numbers and Time (数字和时间)

Numbers (数字 shùzì):
  0 零 (líng)   1 一 (yī)    2 二 (er)    3 三 (sān)   4 四 (sì)
  5 五 (wǔ)     6 六 (liù)   7 七 (qī)    8 八 (bā)    9 九 (jiǔ)
  10 十 (shí)   11 十一 (shíyī)  20 二十 (ershí)  100 百 (bǎi)
  1,000 千 (qiān)   10,000 万 (wàn)   100,000,000 亿 (yì)

Days of the week (星期 xīngqī):
  星期一 (xīngqīyī) Monday    星期二 (xīngqīer) Tuesday
  星期三 (xīngqīsān) Wednesday  星期四 (xīngqīsì) Thursday
  星期五 (xīngqīwǔ) Friday    星期六 (xīngqīliù) Saturday
  星期天/日 (xīngqītiān/rì) Sunday

Months: 一月 January ... 十二月 December (just number + 月 yue)
Time: 现在几点？(Xiànzài jǐ diǎn?) What time is it now?
      两点半 (liǎng diǎn bàn) 2:30 / 上午 (shàngwǔ) AM / 下午 (xiàwǔ) PM""",

"""MANDARIN CHINESE -- Core Vocabulary (核心词汇)

PRONOUNS (代词 dàicí):
  我 (wǒ) I/me          你 (nǐ) you (sing.)     您 (nín) you (formal)
  他 (tā) he/him        她 (tā) she/her          它 (tā) it
  我们 (wǒmen) we/us    你们 (nǐmen) you (pl.)   他们/她们 (tāmen) they

ESSENTIAL VERBS (动词 dòngcí):
  是 (shì) to be        有 (yǒu) to have        在 (zài) to be at/located
  吃 (chī) to eat       喝 (hē) to drink        睡 (shuì) to sleep
  去 (qù) to go         来 (lái) to come        回 (huí) to return
  看 (kàn) to look/watch 说 (shuō) to speak/say  听 (tīng) to listen
  写 (xiě) to write     读/看书 (dú/kànshū) to read  学 (xue) to study
  知道 (zhīdào) to know 喜欢 (xǐhuān) to like   爱 (ài) to love
  想 (xiǎng) to think/want 要 (yào) to want/need  买 (mǎi) to buy
  工作 (gōngzuò) to work 住 (zhù) to live/reside  走 (zǒu) to walk/leave

ADJECTIVES (形容词 xíngróngcí):
  好 (hǎo) good         坏 (huài) bad            大 (dà) big
  小 (xiǎo) small       多 (duō) many/much       少 (shǎo) few/little
  新 (xīn) new          旧 (jiù) old (objects)   年轻 (niánqīng) young
  老 (lǎo) old (people) 高 (gāo) tall/high       矮 (ǎi) short
  漂亮 (piàoliang) beautiful  帅 (shuài) handsome  贵 (guì) expensive
  便宜 (piányí) cheap   快 (kuài) fast           慢 (màn) slow
  热 (re) hot           冷 (lěng) cold           累 (lei) tired
  高兴 (gāoxìng) happy  难过 (nánguò) sad        忙 (máng) busy""",

"""MANDARIN CHINESE -- Common Phrases and Grammar (常用语和语法)

GREETINGS AND SOCIAL:
  你好 (nǐ hǎo) Hello          你好吗？(nǐ hǎo ma?) How are you?
  我很好，谢谢 (wǒ hěn hǎo, xiexie) I'm fine, thank you
  不客气 (bù keqi) You're welcome    对不起 (duìbuqǐ) I'm sorry
  没关系 (mei guānxi) No problem     请 (qǐng) Please
  再见 (zàijiàn) Goodbye             早上好 (zǎoshang hǎo) Good morning
  晚上好 (wǎnshang hǎo) Good evening  晚安 (wǎn'ān) Good night
  我叫... (wǒ jiào...) My name is...        你叫什么名字？(nǐ jiào shenme míngzi?) What's your name?
  我是美国人/中国人 (wǒ shì Měiguóren/Zhōngguóren) I'm American/Chinese
  我不会说中文 (wǒ bù huì shuō Zhōngwen) I can't speak Chinese
  请说慢一点 (qǐng shuō màn yīdiǎn) Please speak more slowly
  我听不懂 (wǒ tīng bù dǒng) I don't understand

GRAMMAR PATTERNS:
  Subject + Verb + Object (SVO):  我吃饭 (wǒ chī fàn) I eat rice/food
  Negation with 不 (bù):           我不吃 (wǒ bù chī) I don't eat
  Negation of 有 with 没 (mei):    我没有钱 (wǒ meiyǒu qián) I have no money
  Question with 吗 (ma):           你是学生吗？(nǐ shì xuesheng ma?) Are you a student?
  Question with 呢 (ne):           你呢？(nǐ ne?) And you?
  Question words: 什么 (shenme) what / 谁 (shei) who / 哪里 (nǎlǐ) where
                  什么时候 (shenme shíhòu) when / 为什么 (weishenme) why / 怎么 (zěnme) how
  Measure words (量词): 一个人 (yī ge ren) one person / 一本书 (yī běn shū) one book
                        一杯水 (yī bēi shuǐ) one cup of water / 一张纸 (yī zhāng zhǐ) one sheet of paper
  Aspect particles: 了 (le) completed action / 过 (guò) past experience / 着 (zhe) ongoing state
  能愿动词 (modal verbs): 能/可以 (neng/kěyǐ) can / 会 (huì) can (learned skill) / 要 (yào) want/will / 应该 (yīnggāi) should
  Comparison: A 比 B + adj: 他比我高 (tā bǐ wǒ gāo) He is taller than me
  的 (de) adjectival particle: 漂亮的女孩 (piàoliang de nǚhái) beautiful girl""",

"""MANDARIN CHINESE -- Body, Family, Food, Places (身体、家庭、食物、地方)

FAMILY (家庭 jiātíng):
  爸爸/父亲 (bàba/fùqīn) father   妈妈/母亲 (māma/mǔqīn) mother
  哥哥 (gēgē) elder brother        弟弟 (dìdi) younger brother
  姐姐 (jiějiě) elder sister       妹妹 (meimei) younger sister
  儿子 (erzi) son                  女儿 (nǚ'er) daughter
  祖父/爷爷 (zǔfù/yeye) grandfather (paternal)  祖母/奶奶 (zǔmǔ/nǎinai) grandmother (paternal)
  外公 (wàigōng) grandfather (maternal)  外婆 (wàipó) grandmother (maternal)
  丈夫 (zhàngfu) husband           妻子 (qīzi) wife

FOOD (食物 shíwù):
  米饭 (mǐfàn) rice                面条 (miàntiáo) noodles       包子 (bāozi) steamed bun
  饺子 (jiǎozi) dumplings          豆腐 (dòufu) tofu             猪肉 (zhūròu) pork
  鸡肉 (jīròu) chicken             牛肉 (niúròu) beef            鱼 (yú) fish
  蔬菜 (shūcài) vegetables         水果 (shuǐguǒ) fruit          茶 (chá) tea
  水 (shuǐ) water                  啤酒 (píjiǔ) beer

PLACES (地方 dìfāng):
  家 (jiā) home                    学校 (xuexiào) school         医院 (yīyuàn) hospital
  餐厅 (cāntīng) restaurant        超市 (chāoshì) supermarket    银行 (yínháng) bank
  公园 (gōngyuán) park             地铁站 (dìtiězhàn) subway station
  机场 (jīchǎng) airport           酒店 (jiǔdiàn) hotel

MOST FREQUENT CHARACTERS (最常用汉字):
  的 一 是 在 不 了 有 和 人 这 中 大 为 上 个 国 我 以 要 他 时 来 用 们 生 到 作 地 于 出
  就 分 对 成 会 可 主 发 年 动 同 工 也 能 下 过 子 说 产 种 面 而 方 后 多 定 行 学 法 所""",
]

# -- Russian (ru) --------------------------------------------------------------
BUILTIN_LANG_DATA['ru'] = [

"""RUSSIAN -- Русский язык (Russkiy yazyk)

CYRILLIC ALPHABET (Кириллица):
  А а (a)   Б б (b)   В в (v)   Г г (g)   Д д (d)   Е е (ye)
  Ё ё (yo)  Ж ж (zh)  З з (z)   И и (i)   Й й (y)   К к (k)
  Л л (l)   М м (m)   Н н (n)   О о (o)   П п (p)   Р р (r)
  С с (s)   Т т (t)   У у (u)   Ф ф (f)   Х х (kh)  Ц ц (ts)
  Ч ч (ch)  Ш ш (sh)  Щ щ (shch) Ъ ъ (hard sign) Ы ы (y, back) Ь ь (soft sign)
  Э э (e)   Ю ю (yu)  Я я (ya)

PRONUNCIATION NOTES:
  Ж (zh): like 's' in 'measure'    Х (kh): like Scottish 'loch'
  Ц (ts): like 'ts' in 'cats'     Ш (sh): hard 'sh'    Щ (shch): soft 'shch'
  Ы: a central vowel, no English equivalent -- between 'i' and 'u'
  Soft sign Ь softens the preceding consonant
  Vowel reduction: unstressed О sounds like А (e.g. молоко -> malakó)
  Stress is unpredictable and must be memorized""",

"""RUSSIAN -- Grammar Overview (Грамматика)

GENDER: All nouns have grammatical gender.
  Masculine: consonant ending -- стол (stol) table, город (gorod) city
  Feminine: -а/-я ending -- книга (kniga) book, неделя (nedelya) week
  Neuter: -о/-е ending -- окно (okno) window, море (more) sea

SIX GRAMMATICAL CASES (падежи):
  Nominative (кто? что?) -- subject: Студент читает. (The student reads.)
  Accusative (кого? что?) -- direct object: Я читаю книгу. (I read a book.)
  Genitive (кого? чего?) -- possession/absence: нет книги (no book)
  Dative (кому? чему?) -- indirect object: Я дам книгу другу. (I'll give the book to a friend.)
  Instrumental (кем? чем?) -- by/with: Я пишу ручкой. (I write with a pen.)
  Prepositional (о ком? о чём?) -- after prepositions: Я думаю о книге. (I think about the book.)

VERB CONJUGATION (present tense, 1st conjugation -- читать to read):
  я читаю (I read)         ты читаешь (you read)       он/она читает (he/she reads)
  мы читаем (we read)      вы читаете (you pl. read)    они читают (they read)

ASPECT (вид): Russian verbs come in pairs -- imperfective (ongoing) / perfective (completed):
  читать/прочитать (to read/to finish reading)
  писать/написать (to write/to finish writing)
  делать/сделать (to do/to finish doing)

PERSONAL PRONOUNS:
  я (I)   ты (you, informal)   он (he)   она (she)   оно (it)
  мы (we)   вы (you formal/pl.)   они (they)""",

"""RUSSIAN -- Core Vocabulary and Phrases (Базовая лексика и фразы)

GREETINGS:
  Здравствуйте (Zdravstvuyte) Hello (formal)   Привет (Privet) Hi (informal)
  Доброе утро (Dobroye utro) Good morning      Добрый день (Dobryy den') Good day
  Добрый вечер (Dobryy vecher) Good evening    Спокойной ночи (Spokoynoy nochi) Good night
  Как дела? (Kak dela?) How are you?           Хорошо, спасибо (Khorosho, spasibo) Fine, thank you
  Пожалуйста (Pozhaluysta) Please / You're welcome
  Извините (Izvinite) Excuse me / I'm sorry    Спасибо (Spasibo) Thank you
  До свидания (Do svidaniya) Goodbye           Пока (Poka) Bye (informal)
  Меня зовут... (Menya zovut...) My name is...       Как вас зовут? (Kak vas zovut?) What's your name?
  Я не понимаю (Ya ne ponimayu) I don't understand
  Говорите медленнее (Govorite medlennee) Speak more slowly

NUMBERS: один (1) два (2) три (3) четыре (4) пять (5) шесть (6) семь (7) восемь (8) девять (9) десять (10)
  двадцать (20) тридцать (30) сто (100) тысяча (1000)

COMMON WORDS:
  да (yes) нет (no) и (and) или (or) но (but) очень (very) немного (a little)
  большой (big) маленький (small) хороший (good) плохой (bad) новый (new) старый (old)
  человек (person) мужчина (man) женщина (woman) ребёнок (child)
  дом (house) вода (water) еда (food) время (time) день (day) ночь (night)
  работа (work) деньги (money) город (city) страна (country) язык (language)""",
]

# -- Japanese (ja) -------------------------------------------------------------
BUILTIN_LANG_DATA['ja'] = [

"""JAPANESE -- 日本語 (Nihongo)

HIRAGANA (平仮名) -- phonetic syllabary for native Japanese words:
  あ(a) い(i) う(u) え(e) お(o)
  か(ka) き(ki) く(ku) け(ke) こ(ko)
  さ(sa) し(shi) す(su) せ(se) そ(so)
  た(ta) ち(chi) つ(tsu) て(te) と(to)
  な(na) に(ni) ぬ(nu) ね(ne) の(no)
  は(ha) ひ(hi) ふ(fu) へ(he) ほ(ho)
  ま(ma) み(mi) む(mu) め(me) も(mo)
  や(ya)         ゆ(yu)         よ(yo)
  ら(ra) り(ri) る(ru) れ(re) ろ(ro)
  わ(wa)                        を(wo/o)
  ん(n)
  Voiced: が(ga) ぎ(gi) ぐ(gu) げ(ge) ご(go) / ざ(za) じ(ji) ず(zu) ぜ(ze) ぞ(zo)
          だ(da) ぢ(ji) づ(zu) で(de) ど(do) / ば(ba) び(bi) ぶ(bu) べ(be) ぼ(bo)
  P-row:  ぱ(pa) ぴ(pi) ぷ(pu) ぺ(pe) ぽ(po)
  Combined: きゃ(kya) きゅ(kyu) きょ(kyo) / しゃ(sha) しゅ(shu) しょ(sho) etc.""",

"""JAPANESE -- Katakana and Kanji (片仮名・漢字)

KATAKANA (片仮名) -- phonetic syllabary for foreign loanwords:
  ア(a) イ(i) ウ(u) エ(e) オ(o)
  カ(ka) キ(ki) ク(ku) ケ(ke) コ(ko)
  サ(sa) シ(shi) ス(su) セ(se) ソ(so)
  タ(ta) チ(chi) ツ(tsu) テ(te) ト(to)
  ナ(na) ニ(ni) ヌ(nu) ネ(ne) ノ(no)
  ハ(ha) ヒ(hi) フ(fu) ヘ(he) ホ(ho)
  マ(ma) ミ(mi) ム(mu) メ(me) モ(mo)
  ヤ(ya)         ユ(yu)         ヨ(yo)
  ラ(ra) リ(ri) ル(ru) レ(re) ロ(ro)
  ワ(wa)                        ヲ(wo)
  ン(n)
  Loanword examples: コーヒー(kōhī)=coffee テレビ(terebi)=TV スマホ(sumaho)=smartphone
                     パソコン(pasokon)=PC アイスクリーム(aisu kurīmu)=ice cream

ESSENTIAL KANJI (漢字) -- 常用漢字 (jōyō kanji):
  日(nichi/hi) sun/day  月(tsuki/getsu) moon/month  山(yama/san) mountain
  川(kawa/sen) river    田(ta/den) rice field        木(ki/moku) tree
  人(hito/jin) person   子(ko/shi) child             女(onna/jo) woman  男(otoko/dan) man
  大(ō/dai) big  小(chii/shō) small  中(naka/chū) middle/inside  上(ue/jō) above  下(shita/ka) below
  一(ichi) 二(ni) 三(san) 四(shi) 五(go) 六(roku) 七(shichi) 八(hachi) 九(ku) 十(jū) 百(hyaku) 千(sen) 万(man)
  国(koku/kuni) country  語(go/kata) language  学(gaku/mana) study  生(sei/nama) life/raw
  食(shoku/ta) eat  水(sui/mizu) water  火(ka/hi) fire  土(do/tsuchi) earth/soil
  気(ki) spirit/energy  心(kokoro/shin) heart/mind  手(te/shu) hand  目(me/moku) eye  耳(mimi/ji) ear  口(kuchi/kō) mouth""",

"""JAPANESE -- Grammar and Essential Phrases (文法と基本表現)

GRAMMAR PARTICLES (助詞 joshi):
  は (wa) -- topic marker: 私は学生です (Watashi wa gakusei desu) I am a student
  が (ga) -- subject marker: 猫が好きです (Neko ga suki desu) I like cats
  を (wo) -- direct object: 本を読みます (Hon wo yomimasu) I read a book
  に (ni) -- direction/location/time: 学校に行きます (Gakkō ni ikimasu) I go to school
  で (de) -- location of action/by means of: 駅で待ちます (Eki de machimasu) I wait at the station
  の (no) -- possessive: 私の本 (Watashi no hon) my book
  と (to) -- and (nouns) / with: 友達と行く (Tomodachi to iku) go with friends
  も (mo) -- also/too: 私も行きます (Watashi mo ikimasu) I'll go too
  か (ka) -- question marker (sentence-final): これは何ですか？(Kore wa nan desu ka?) What is this?
  ね (ne) -- seeking agreement: いい天気ですね (Ii tenki desu ne) Nice weather, isn't it?

VERB FORMS (動詞):
  Plain/Dictionary form: 食べる (taberu) to eat / 行く (iku) to go / する (suru) to do
  Polite -masu form: 食べます (tabemasu) / 行きます (ikimasu) / します (shimasu)
  Negative: 食べません (tabemasen) / 行きません (ikimasen)
  Past polite: 食べました (tabemashita) / 行きました (ikimashita)
  Te-form (connects clauses): 食べて (tabete) / 行って (itte)

ESSENTIAL PHRASES:
  こんにちは (Konnichiwa) Hello (daytime)     おはようございます (Ohayō gozaimasu) Good morning
  こんばんは (Konbanwa) Good evening          おやすみなさい (Oyasumi nasai) Good night
  ありがとうございます (Arigatō gozaimasu) Thank you (formal)
  どういたしまして (Dō itashimashite) You're welcome
  すみません (Sumimasen) Excuse me / I'm sorry   はじめまして (Hajimemashite) Nice to meet you
  私は...と申します (Watashi wa... to mōshimasu) My name is... (formal)
  日本語が少し話せます (Nihongo ga sukoshi hanasemasu) I can speak a little Japanese
  わかりません (Wakarimasen) I don't understand    もう一度お願いします (Mō ichido onegaishimasu) Please repeat""",
]

# -- Arabic (ar) ---------------------------------------------------------------
BUILTIN_LANG_DATA['ar'] = [

"""ARABIC -- العربية (Al-'Arabiyya)

SCRIPT: Arabic is written RIGHT-TO-LEFT. Letters are connected and change shape based on position.
28 letters. Vowels are often omitted in everyday writing (diacritics added for learners).

THE ALPHABET (الأبجدية):
  ا (alif, a/ā)  ب (ba, b)   ت (ta, t)   ث (tha, th)  ج (jeem, j)  ح (ḥa, ḥ)
  خ (kha, kh)   د (dal, d)   ذ (dhal, dh) ر (ra, r)    ز (zay, z)   س (seen, s)
  ش (sheen, sh)  ص (ṣad, ṣ)  ض (ḍad, ḍ)  ط (ṭa, ṭ)    ظ (ẓa, ẓ)   ع ('ayn, ')
  غ (ghayn, gh)  ف (fa, f)   ق (qaf, q)   ك (kaf, k)   ل (lam, l)   م (meem, m)
  ن (noon, n)    ه (ha, h)   و (waw, w/ū) ي (ya, y/ī)

SPECIAL SOUNDS:
  ح (ḥ): deep H from throat       ع ('): voiced pharyngeal -- unique to Arabic
  غ (gh): like French R (gargled) خ (kh): like German 'ch' in 'Bach'
  ق (q): uvular K from deep throat  ص ض ط ظ: emphatic consonants (pharyngealized)

VOWELS (حركات): Short vowels shown as diacritics -- فَ (fa) فِ (fi) فُ (fu)
  Long vowels: ā (آ/ا), ī (ي), ū (و)  Tanwin (nunation): -an -in -un (indefinite nouns)

DEFINITE ARTICLE: ال (al-) -- الكتاب (al-kitāb) the book
  Sun letters: ال assimilates -- الشمس -> ash-shams (the sun), not al-shams
  Moon letters: ال stays -- القمر -> al-qamar (the moon)""",

"""ARABIC -- Core Vocabulary and Grammar (المفردات الأساسية والقواعد)

GREETINGS (التحيات):
  السلام عليكم (As-salāmu 'alaykum) Peace be upon you -- universal greeting
  وعليكم السلام (Wa-'alaykum as-salām) And upon you peace -- response
  مرحبا (Marḥaban) Hello (informal)    أهلاً (Ahlan) Welcome/Hi
  صباح الخير (Ṣabāḥ al-khayr) Good morning   صباح النور (Ṣabāḥ an-nūr) Response
  مساء الخير (Masā' al-khayr) Good evening    مساء النور (Masā' an-nūr) Response
  كيف حالك؟ (Kayfa ḥālak?) How are you?       بخير، شكراً (Bikhayrin, shukran) Fine, thank you
  شكراً (Shukran) Thank you            عفواً (Afwan) You're welcome / Excuse me
  من فضلك (Min faḍlak) Please           آسف/آسفة (Āsif/Āsifa) Sorry (m./f.)
  مع السلامة (Ma'a as-salāma) Goodbye   إلى اللقاء (Ilā al-liqā') Until we meet again

NUMBERS (الأرقام):
  ١(1) ٢(2) ٣(3) ٤(4) ٥(5) ٦(6) ٧(7) ٨(8) ٩(9) ١٠(10)
  واحد(1) اثنان(2) ثلاثة(3) أربعة(4) خمسة(5) ستة(6) سبعة(7) ثمانية(8) تسعة(9) عشرة(10)
  عشرون(20) مئة(100) ألف(1000)

GRAMMAR:
  Dual form: كتاب (book) -> كتابان (two books)
  Broken plural: Most nouns have irregular plurals -- كتاب -> كُتُب (books)
  Verb-Subject-Object (VSO) is common: ذهب الولد إلى المدرسة (The boy went to school)
  Root system: trilateral roots carry meaning -- ك-ت-ب = writing: كتب write, كتاب book, مكتبة library, كاتب writer
  Masculine/feminine: ه-suffix often marks feminine -- مدرس(teacher m.) / مدرسة(teacher f.)""",
]

# -- Hindi (hi) ----------------------------------------------------------------
BUILTIN_LANG_DATA['hi'] = [

"""HINDI -- हिन्दी (Hindī)

DEVANAGARI SCRIPT (देवनागरी):
  Vowels (स्वर): अ(a) आ(ā) इ(i) ई(ī) उ(u) ऊ(ū) ए(e) ऐ(ai) ओ(o) औ(au) अं(aṃ) अः(aḥ)
  Consonants (व्यंजन):
    क(ka) ख(kha) ग(ga) घ(gha) ङ(ṅa)
    च(ca) छ(cha) ज(ja) झ(jha) ञ(na)
    ट(ṭa) ठ(ṭha) ड(ḍa) ढ(ḍha) ण(ṇa)
    त(ta) थ(tha) द(da) ध(dha) न(na)
    प(pa) फ(pha) ब(ba) भ(bha) म(ma)
    य(ya) र(ra) ल(la) व(va) श(śa) ष(ṣa) स(sa) ह(ha)
  Vowel diacritics attach to consonants: क + ा = का(kā), क + ि = कि(ki), क + ी = की(kī)
  Virama (्) cancels inherent vowel: क् = k (no vowel)
  Anusvara (ं): nasalization -- हं(haṃ)   Visarga (ः): aspirate breath -- अः(aḥ)

WRITING DIRECTION: Left to right. Characters hang from a horizontal headline (शिरोरेखा).""",

"""HINDI -- Core Vocabulary and Grammar (मूल शब्द-भंडार और व्याकरण)

GREETINGS (अभिवादन):
  नमस्ते (Namaste) Hello/Greetings (hands folded)  नमस्कार (Namaskār) Hello (formal)
  आप कैसे हैं? (Āp kaise haiṃ?) How are you? (formal)  मैं ठीक हूँ (Maiṃ ṭhīk hūṃ) I am fine
  धन्यवाद (Dhanyavād) Thank you     कृपया (Kṛpayā) Please      माफ़ करें (Māf kareṃ) Sorry
  अलविदा (Alavida) Goodbye           हाँ (Hāṃ) Yes              नहीं (Nahīṃ) No

PRONOUNS: मैं(I) तू(you, intimate) तुम(you, familiar) आप(you, formal) वह(he/she/it) हम(we) वे(they)

POSTPOSITIONS (Hindi uses postpositions, not prepositions):
  में (meṃ) in     पर (par) on      को (ko) to/for   से (se) from/with   का/की/के (kā/kī/ke) of/possessive
  ने (ne) subject marker (with transitive verbs in past tense)

VERB AGREEMENT: Verbs agree with subject in gender and number in most tenses.
  मैं जाता हूँ (maiṃ jātā hūṃ) I go (male speaker)
  मैं जाती हूँ (maiṃ jātī hūṃ) I go (female speaker)

NUMBERS: एक(1) दो(2) तीन(3) चार(4) पाँच(5) छह(6) सात(7) आठ(8) नौ(9) दस(10)
  बीस(20) पचास(50) सौ(100) हज़ार(1000)

COMMON WORDS: पानी(water) खाना(food) घर(home) काम(work) दिन(day) रात(night)
  अच्छा(good) बुरा(bad) बड़ा(big) छोटा(small) नया(new) पुराना(old)""",
]

# -- Urdu (ur) -----------------------------------------------------------------
BUILTIN_LANG_DATA['ur'] = [

"""URDU -- اُردُو (Urdū)

SCRIPT: Urdu uses Nastaliq script, a flowing cursive form of Perso-Arabic script.
Written RIGHT-TO-LEFT. 38 letters (more than Arabic due to added South Asian sounds).
Shares most vocabulary with Hindi but written in different script and with more Persian/Arabic loanwords.

ALPHABET includes Perso-Arabic letters plus:
  ٹ (ṭ -- retroflex T)   ڈ (ḍ -- retroflex D)   ڑ (ṛ -- retroflex R)
  ں (nasal n -- noon ghunna)   ے (ye -- final/medial)   ہ/ھ (h/aspirated h)

GREETINGS (سلامتی):
  السلام علیکم (As-salāmu 'alaykum) Peace be upon you -- standard greeting
  آداب (Ādāb) Respectful greeting (secular/cultural)
  آپ کیسے ہیں؟ (Āp kaise haiṃ?) How are you? (formal)
  میں ٹھیک ہوں (Maiṃ ṭhīk hūṃ) I am fine
  شکریہ (Shukriyā) Thank you    براہ کرم (Barāh-e-karam) Please
  معاف کریں (Muāf kareṃ) Please forgive/Sorry   خدا حافظ (Khudā ḥāfiẓ) Goodbye

NUMBERS: ایک(1) دو(2) تین(3) چار(4) پانچ(5) چھ(6) سات(7) آٹھ(8) نو(9) دس(10)
  بیس(20) سو(100) ہزار(1000)

KEY VOCABULARY:
  پانی(pānī) water   کھانا(khānā) food   گھر(ghar) home   کام(kām) work
  اچھا(acchā) good   برا(burā) bad       بڑا(baṛā) big    چھوٹا(choṭā) small
  ہاں(hāṃ) yes      نہیں(nahīṃ) no       اور(aur) and     لیکن(lekin) but""",
]

# -- French (fr) ---------------------------------------------------------------
BUILTIN_LANG_DATA['fr'] = [

"""FRENCH -- Le français

GREETINGS AND PHRASES:
  Bonjour Good morning/Hello (formal)   Bonsoir Good evening   Bonne nuit Good night
  Salut Hi/Bye (informal)               Au revoir Goodbye      À bientôt See you soon
  Comment allez-vous? How are you? (formal)  Comment ça va? How's it going? (informal)
  Je vais bien, merci I'm fine, thank you    Merci Thank you    De rien You're welcome
  S'il vous plaît / S'il te plaît Please (formal/informal)
  Excusez-moi Excuse me   Je suis desole(e) I'm sorry   Je ne comprends pas I don't understand
  Parlez-vous anglais? Do you speak English?  Je parle un peu français I speak a little French
  Comment vous appelez-vous? What is your name?  Je m'appelle... My name is...

PRONUNCIATION NOTES:
  Silent consonants at end of words: grand (grɑ̃), est (ɛ), les (le)
  Nasal vowels: an/en(ɑ̃), in/ain(ɛ̃), on(ɔ̃), un(œ̃)
  Liaison: les enfants -> [lezɑ̃fɑ̃] (S links to next vowel)
  R: uvular/guttural sound (back of throat)  U: round lips for [y] sound
  É: [e] as in 'say'  È/Ê: [ɛ] as in 'set'  OU: [u] as in 'boot'

NUMBERS: un(1) deux(2) trois(3) quatre(4) cinq(5) six(6) sept(7) huit(8) neuf(9) dix(10)
  onze(11) douze(12) vingt(20) trente(30) quarante(40) cinquante(50) soixante(60)
  soixante-dix(70) quatre-vingts(80) quatre-vingt-dix(90) cent(100) mille(1000)""",

"""FRENCH -- Grammar (La grammaire française)

GENDER: All nouns are masculine or feminine.
  Masculine: le/un -- le livre (the book), un homme (a man)
  Feminine: la/une -- la femme (the woman), une maison (a house)
  Plural: les/des -- les livres, des maisons

VERB CONJUGATION -- Present tense:
  ÊTRE (to be): je suis, tu es, il/elle est, nous sommes, vous etes, ils/elles sont
  AVOIR (to have): j'ai, tu as, il/elle a, nous avons, vous avez, ils/elles ont
  ALLER (to go): je vais, tu vas, il va, nous allons, vous allez, ils vont
  Regular -ER (parler, to speak): je parle, tu parles, il parle, nous parlons, vous parlez, ils parlent
  Regular -IR (finir, to finish): je finis, tu finis, il finit, nous finissons, vous finissez, ils finissent
  Regular -RE (vendre, to sell): je vends, tu vends, il vend, nous vendons, vous vendez, ils vendent

PAST TENSES:
  Passe compose (completed actions): J'ai mange (I ate/have eaten) -- auxiliary avoir/etre + past participle
  Imparfait (ongoing/habitual past): Je mangeais (I was eating/used to eat)
  Verbs using ÊTRE in passe compose: aller, venir, partir, arriver, naître, mourir, monter, descendre, etc. + all reflexive verbs

SUBJUNCTIVE (le subjonctif) -- after expressions of doubt, wish, emotion, necessity:
  Il faut que tu sois (It is necessary that you be)
  Je veux qu'il vienne (I want him to come)

KEY VOCABULARY:
  Temps: aujourd'hui(today) demain(tomorrow) hier(yesterday) maintenant(now) toujours(always) jamais(never)
  Lieu: ici(here) là(there) où(where) dedans(inside) dehors(outside)""",
]

# -- Spanish (es) --------------------------------------------------------------
BUILTIN_LANG_DATA['es'] = [

"""SPANISH -- El espanol / El castellano

GREETINGS AND PHRASES:
  Hola Hello   Buenos días Good morning   Buenas tardes Good afternoon
  Buenas noches Good evening/night   Adiós Goodbye   Hasta luego See you later
  ¿Cómo está usted? How are you? (formal)   ¿Cómo estás? How are you? (informal)
  Muy bien, gracias Very well, thank you    Por favor Please   Gracias Thank you
  De nada You're welcome   Perdón / Disculpe Excuse me / I'm sorry   Lo siento I'm sorry
  No entiendo I don't understand   ¿Habla ingles? Do you speak English?
  Hablo un poco de espanol I speak a little Spanish
  ¿Cómo se llama usted? / ¿Cómo te llamas? What is your name?   Me llamo... My name is...

PRONUNCIATION:
  LL: [ʝ] or [ʒ] -- like 'y' in 'yes'    Ñ: [ɲ] -- like 'ny' in 'canyon'
  J/G(e,i): [x] -- like 'h' in 'hot' (guttural)   RR: trilled R
  V: pronounced like B in most dialects   H: always silent (e.g., hola = 'ola')
  C(e,i)/Z: [θ] in Spain, [s] in Latin America   QU: always [k] (que = ke)

NUMBERS: uno(1) dos(2) tres(3) cuatro(4) cinco(5) seis(6) siete(7) ocho(8) nueve(9) diez(10)
  once(11) doce(12) veinte(20) treinta(30) cien/ciento(100) mil(1000)""",

"""SPANISH -- Grammar (La gramática espanola)

GENDER AND ARTICLES:
  Masculine: el/un -- el libro (the book), un hombre (a man)
  Feminine: la/una -- la mujer (the woman), una casa (a house)
  Plurals: los/las/unos/unas

TWO VERBS "TO BE": SER vs ESTAR
  SER -- permanent/essential qualities: ser alto (to be tall), ser medico (to be a doctor), ser espanol (to be Spanish)
  ESTAR -- states/conditions/location: estar cansado (to be tired), estar en casa (to be at home)
  SER: soy, eres, es, somos, sois, son
  ESTAR: estoy, estás, está, estamos, estáis, están

REGULAR VERB CONJUGATION (present tense):
  -AR (hablar, to speak): hablo, hablas, habla, hablamos, habláis, hablan
  -ER (comer, to eat): como, comes, come, comemos, comeis, comen
  -IR (vivir, to live): vivo, vives, vive, vivimos, vivís, viven

REFLEXIVE VERBS (verbos reflexivos): levantarse (to get up), llamarse (to be named), ducharse (to shower)
  Me llamo Juan (My name is Juan) / Se llama María (Her name is María)

SUBJUNCTIVE: Used after querer que, esperar que, es importante que, etc.
  Quiero que vengas (I want you to come)
  Es importante que estudies (It's important that you study)

KEY VOCABULARY:
  Tiempo: hoy(today) manana(tomorrow) ayer(yesterday) ahora(now) siempre(always) nunca(never)
  Descripción: grande(big) pequeno(small) bueno(good) malo(bad) nuevo(new) viejo(old) bonito(pretty) feo(ugly)""",
]

# -- Italian (it) --------------------------------------------------------------
BUILTIN_LANG_DATA['it'] = [

"""ITALIAN -- L'italiano

GREETINGS AND PHRASES:
  Ciao Hello/Bye (informal)   Buongiorno Good morning/Good day   Buonasera Good evening
  Buonanotte Good night       Arrivederci Goodbye (formal)        A presto See you soon
  Come sta? How are you? (formal)   Come stai? How are you? (informal)
  Sto bene, grazie I'm fine, thank you   Per favore Please   Grazie Thank you
  Prego You're welcome / Please go ahead   Mi scusi Excuse me (formal)   Scusa Sorry (informal)
  Non capisco I don't understand   Parla inglese? Do you speak English?
  Mi chiamo... My name is...   Come si chiama? What is your name? (formal)

GRAMMAR HIGHLIGHTS:
  GENDER: All nouns masculine or feminine. -o endings often masculine, -a often feminine.
    Masculine: il/lo/l'/i/gli -- il libro (the book), lo studente (the student)
    Feminine: la/l'/le -- la casa (the house), le ragazze (the girls)
  ESSERE (to be): sono, sei, e, siamo, siete, sono
  AVERE (to have): ho, hai, ha, abbiamo, avete, hanno
  ANDARE (to go): vado, vai, va, andiamo, andate, vanno
  Regular -ARE (parlare): parlo, parli, parla, parliamo, parlate, parlano
  Regular -ERE (vendere): vendo, vendi, vende, vendiamo, vendete, vendono
  Regular -IRE (dormire): dormo, dormi, dorme, dormiamo, dormite, dormono
  Passato prossimo: ho mangiato (I ate), sono andato (I went) -- transitive uses avere, motion/state uses essere
  NUMBERS: uno(1) due(2) tre(3) quattro(4) cinque(5) sei(6) sette(7) otto(8) nove(9) dieci(10)
    venti(20) trenta(30) cento(100) mille(1000)""",
]

# -- German (de) ---------------------------------------------------------------
BUILTIN_LANG_DATA['de'] = [

"""GERMAN -- Deutsch

GREETINGS AND PHRASES:
  Hallo Hello (informal)   Guten Morgen Good morning   Guten Tag Good day
  Guten Abend Good evening  Gute Nacht Good night       Auf Wiedersehen Goodbye (formal)
  Tschuss Bye (informal)    Wie geht es Ihnen? How are you? (formal)   Wie geht's? How's it going?
  Mir geht es gut, danke I'm fine, thank you   Bitte Please / You're welcome
  Danke (schon) Thank you (very much)   Entschuldigung Excuse me / I'm sorry
  Ich verstehe nicht I don't understand   Sprechen Sie Englisch? Do you speak English?
  Ich heiße... My name is...   Wie heißen Sie? What is your name? (formal)

GRAMMAR -- THE FOUR CASES:
  Nominative (subject): Der Mann liest. (The man reads.)
  Accusative (direct object): Ich sehe den Mann. (I see the man.)
  Dative (indirect object): Ich gebe dem Mann ein Buch. (I give the man a book.)
  Genitive (possession): Das Buch des Mannes. (The man's book.)
  Definite article declension:
    Masculine: der(nom) den(acc) dem(dat) des(gen)
    Feminine:  die(nom) die(acc) der(dat) der(gen)
    Neuter:    das(nom) das(acc) dem(dat) des(gen)
    Plural:    die(nom) die(acc) den(dat) der(gen)

VERB CONJUGATION (present) -- SEIN (to be): bin, bist, ist, sind, seid, sind
  HABEN (to have): habe, hast, hat, haben, habt, haben
  Regular verb (machen, to do/make): mache, machst, macht, machen, macht, machen
  MODAL VERBS: konnen(can) mussen(must) wollen(want to) sollen(should) durfen(may) mogen(like)
  WORD ORDER: Verb-second rule -- main clause verb always second position
    Ich gehe morgen -- Morgen gehe ich (Tomorrow I go -- Tomorrow go I)
  NUMBERS: eins(1) zwei(2) drei(3) vier(4) funf(5) sechs(6) sieben(7) acht(8) neun(9) zehn(10)
    zwanzig(20) dreißig(30) hundert(100) tausend(1000)""",
]

# -- Portuguese (pt) -----------------------------------------------------------
BUILTIN_LANG_DATA['pt'] = [

"""PORTUGUESE -- Portugues (European / Brazilian)

GREETINGS:
  Olá Hello   Bom dia Good morning   Boa tarde Good afternoon   Boa noite Good evening/night
  Como está? How are you? (formal)   Como vai? How's it going?   Estou bem, obrigado(a) I'm fine, thanks
  Por favor Please   Obrigado(a) Thank you   De nada You're welcome
  Desculpe Excuse me/Sorry   Não entendo I don't understand   Adeus / Tchau Goodbye
  Como se chama? What is your name?   Chamo-me... / Me chamo... My name is...

GRAMMAR:
  GENDER: Masculine (o/um) and feminine (a/uma). Most -o nouns masc., -a nouns fem.
  SER vs ESTAR: Like Spanish -- SER for permanent traits, ESTAR for states/location
  SER: sou, es, e, somos, sois/são, são
  ESTAR: estou, estás, está, estamos, estais/estão, estão
  -AR verbs (falar, to speak): falo, falas, fala, falamos, falais/falam, falam
  -ER verbs (comer, to eat): como, comes, come, comemos, comeis/comem, comem
  Personal infinitive: unique to Portuguese -- infinitive conjugated for each person
  Nasal vowels: ã, ão, em, im -- pronounced through the nose (irmã = sister, pão = bread)
  NUMBERS: um/uma(1) dois/duas(2) tres(3) quatro(4) cinco(5) seis(6) sete(7) oito(8) nove(9) dez(10)
    vinte(20) trinta(30) cem/cento(100) mil(1000)""",
]

# -- Korean (ko) ---------------------------------------------------------------
BUILTIN_LANG_DATA['ko'] = [

"""KOREAN -- 한국어 (Hangugeo)

HANGUL (한글) -- The Korean alphabet, invented 1443 by King Sejong:
CONSONANTS (자음 jaeum):
  ㄱ(g/k) ㄴ(n) ㄷ(d/t) ㄹ(r/l) ㅁ(m) ㅂ(b/p) ㅅ(s) ㅇ(silent/ng) ㅈ(j) ㅊ(ch) ㅋ(k) ㅌ(t) ㅍ(p) ㅎ(h)
  Tense consonants: ㄲ(kk) ㄸ(tt) ㅃ(pp) ㅆ(ss) ㅉ(jj)

VOWELS (모음 moeum):
  Basic: ㅏ(a) ㅓ(eo) ㅗ(o) ㅜ(u) ㅡ(eu) ㅣ(i)
  Extended: ㅐ(ae) ㅔ(e) ㅑ(ya) ㅕ(yeo) ㅛ(yo) ㅠ(yu) ㅒ(yae) ㅖ(ye)
  Combined: ㅘ(wa) ㅙ(wae) ㅚ(oe) ㅝ(wo) ㅞ(we) ㅟ(wi) ㅢ(ui)

SYLLABLE BLOCKS: Each block = Initial consonant + vowel (+ optional final consonant/batchim)
  한 = ㅎ+ㅏ+ㄴ = h+a+n   국 = ㄱ+ㅜ+ㄱ = g+u+k   어 = ㅇ+ㅓ = eo

GREETINGS:
  안녕하세요 (Annyeonghaseyo) Hello (polite)   안녕 (Annyeong) Hi/Bye (informal)
  감사합니다 (Gamsahamnida) Thank you (formal)  고마워 (Gomawo) Thanks (informal)
  죄송합니다 (Joesonghamnida) I'm sorry (formal) 괜찮아요 (Gwaenchanayo) It's okay
  안녕히 가세요 (Annyeonghi gaseyo) Goodbye (to person leaving)
  안녕히 계세요 (Annyeonghi gyeseyo) Goodbye (staying behind)
  네 (Ne) Yes   아니요 (Aniyo) No   이름이 뭐예요? (Ireumi mwoyeyo?) What's your name?

NUMBERS: 일(1) 이(2) 삼(3) 사(4) 오(5) 육(6) 칠(7) 팔(8) 구(9) 십(10) -- Sino-Korean
         하나(1) 둘(2) 셋(3) 넷(4) 다섯(5) 여섯(6) 일곱(7) 여덟(8) 아홉(9) 열(10) -- Native Korean
GRAMMAR PARTICLES: 은/는(topic) 이/가(subject) 을/를(object) 에(at/to) 에서(at/from) 의(possessive) 와/과(and)""",
]

# -- Turkish (tr) --------------------------------------------------------------
BUILTIN_LANG_DATA['tr'] = [

"""TURKISH -- Turkçe

ALPHABET: Latin script (reformed 1928). 29 letters. Notable special characters:
  Ç(ch) Ğ(soft g -- lengthens preceding vowel) İ(dotted I) I(dotless ı) Ö(o) Ş(sh) Ü(u)

VOWEL HARMONY (Ünlu uyumu): Suffixes harmonize with the last vowel of the root.
  Front vowels: e i o u -> use e/i suffixes
  Back vowels:  a ı o u -> use a/ı suffixes
  Example: ev(house) + ler = evler(houses) / araba(car) + lar = arabalar(cars)

AGGLUTINATION: Words built by stacking suffixes.
  git(go) + me(neg.) + di(past) + m(1st sg.) = gitmedim (I did not go)
  ev(house) + im(my) + de(in) = evimde (in my house)

GREETINGS:
  Merhaba Hello   Gunaydın Good morning   İyi akşamlar Good evening   İyi geceler Good night
  Nasılsınız? How are you? (formal)   İyiyim, teşekkurler I'm fine, thank you
  Lutfen Please   Teşekkur ederim Thank you   Rica ederim You're welcome
  Özur dilerim I'm sorry   Anlamıyorum I don't understand   Hoşça kalın Goodbye
  Adınız ne? What is your name?   Benim adım... My name is...

NO VERB "TO BE" in present tense -- expressed as suffix: Ben oğrenciyim (I am a student = I student-am)
NUMBERS: bir(1) iki(2) uç(3) dort(4) beş(5) altı(6) yedi(7) sekiz(8) dokuz(9) on(10) yirmi(20) yuz(100) bin(1000)""",
]

# -- Polish (pl) ---------------------------------------------------------------
BUILTIN_LANG_DATA['pl'] = [

"""POLISH -- Język polski

ALPHABET: Latin + special characters: Ą(ą) Ć(ć) Ę(ę) Ł(ł) Ń(ń) Ó(ó) Ś(ś) Ź(ź) Ż(ż)
  ą/ę: nasal vowels   ł: like English 'w'   ć/ś/ź/ń: soft palatalized consonants
  sz=[ʃ] cz=[tʃ] rz/ż=[ʒ] szcz=[ʃtʃ]

GRAMMAR: 7 cases (like Russian but adds Vocative), 3 genders (masculine/feminine/neuter + animate/inanimate distinction for masculine)
  Cases: Nominative Genitive Dative Accusative Instrumental Locative Vocative
  Verbs agree with subject in person, number, and gender (past tense)

GREETINGS:
  Cześć Hi/Bye (informal)   Dzień dobry Good morning/day (formal)   Dobry wieczór Good evening
  Dobranoc Good night   Do widzenia Goodbye (formal)   Jak się masz? How are you? (informal)
  Dobrze, dziękuję Fine, thank you   Proszę Please/You're welcome   Dziękuję Thank you
  Przepraszam Excuse me/I'm sorry   Nie rozumiem I don't understand
  Jak masz na imię? What is your name? (informal)   Mam na imię... My name is...
  Tak Yes   Nie No

NUMBERS: jeden(1) dwa(2) trzy(3) cztery(4) pięć(5) sześć(6) siedem(7) osiem(8) dziewięć(9) dziesięć(10)
  dwadzieścia(20) sto(100) tysiąc(1000)""",
]

# -- Dutch (nl) ----------------------------------------------------------------
BUILTIN_LANG_DATA['nl'] = [

"""DUTCH -- Nederlands

GREETINGS:
  Hallo / Hoi Hello (informal)   Goedemorgen Good morning   Goedemiddag Good afternoon
  Goedenavond Good evening   Goedenacht Good night   Dag / Doei Bye
  Hoe gaat het? How are you?   Goed, dank je Fine, thank you
  Alsjeblieft Please (informal) / Hier is... Here is... (when handing something)
  Dank je (wel) Thank you (very much)   Graag gedaan You're welcome
  Sorry / Excuseer me Excuse me / Sorry   Ik begrijp het niet I don't understand
  Spreekt u Engels? Do you speak English?   Hoe heet u? / Hoe heet je? What is your name?
  Ik heet... / Mijn naam is... My name is...   Ja Yes   Nee No

GRAMMAR:
  DE and HET: Dutch has two grammatical genders -- de (common gender) and het (neuter)
    de man (the man), de vrouw (the woman), de auto (the car)
    het kind (the child), het huis (the house), het boek (the book)
  Indefinite: een (a/an) for all genders
  WORD ORDER: V2 rule -- verb second in main clauses; verb goes to end in subordinate clauses
    Ik ga morgen naar school. (I go tomorrow to school.)
    Ik weet dat hij morgen naar school gaat. (...that he tomorrow to school goes.)
  VERBS: infinitive ends in -en. Remove -en for stem: werken -> werk
  NUMBERS: een(1) twee(2) drie(3) vier(4) vijf(5) zes(6) zeven(7) acht(8) negen(9) tien(10)
    twintig(20) dertig(30) honderd(100) duizend(1000)""",
]

# -- Swahili (sw) --------------------------------------------------------------
BUILTIN_LANG_DATA['sw'] = [

"""SWAHILI -- Kiswahili

Swahili is a Bantu language -- the most widely spoken in sub-Saharan Africa.
Official language of Tanzania, Kenya, Uganda, Rwanda, DRC, and the African Union.

NOUN CLASSES (ngeli): Swahili has ~15 noun classes marked by prefixes. Adjectives, verbs, and pronouns agree.
  M-/WA- class (people): m-tu (person), wa-tu (people) -- m-zuri (good person), wa-zuri (good people)
  KI-/VI- class (things): ki-tabu (book), vi-tabu (books)
  N-/N- class (many animals/abstract): n-dege (bird), n-dege (birds -- same form)
  U- class (abstract/uncountable): u-zuri (beauty)

GREETINGS (salamu): Swahili greetings are elaborate and important culturally.
  Hujambo? / Jambo? How are you? (sing.)   Sijambo I'm fine (lit. I have no problem)
  Habari? / Habari yako? What's the news? / How are you?   Nzuri Fine/Good
  Habari za asubuhi? Good morning (How is the news of the morning?)
  Shikamoo (respectful greeting to elders)   Marahaba (elder's response to Shikamoo)
  Kwaheri Goodbye   Asante (sana) Thank you (very much)   Karibu Welcome/You're welcome
  Tafadhali Please   Samahani Excuse me / I'm sorry   Ndiyo Yes   Hapana No

VERB STRUCTURE: Subject prefix + Tense marker + Object prefix + Verb root
  ni-na-soma (I am reading): ni=I, na=present, soma=read
  a-li-soma (he/she read): a=he/she, li=past
  tu-ta-soma (we will read): tu=we, ta=future
NUMBERS: moja(1) mbili(2) tatu(3) nne(4) tano(5) sita(6) saba(7) nane(8) tisa(9) kumi(10) ishirini(20) mia(100) elfu(1000)""",
]

# -- Vietnamese (vi) -----------------------------------------------------------
BUILTIN_LANG_DATA['vi'] = [

"""VIETNAMESE -- Tiếng Việt

TONES: Vietnamese has 6 tones (Southern dialect) or 6+ (Northern dialect)
  Level (flat): a -- ma (ghost)         Rising: á -- má (mother/cheek)
  Falling: à -- mà (but)                Broken rising: ả -- mả (tomb)
  Heavy glottal: ã -- mã (horse, code)  Sharp falling: ạ -- mạ (rice seedling)
  Diacritics mark both tone AND vowel modification: â e ô ơ ư ă

WRITING: Latin-based script (chữ Quốc ngữ) developed by Portuguese missionaries, standardized 17th c.
  Modified vowels: ă(short a) â(central a) e(closed e) ô(rounded o) ơ(unrounded o) ư(unrounded u)
  Tonal diacritics: ´(sắc/rising) `(huyền/falling) ̉(hỏi/dipping) ̃(ngã/broken) ̣(nặng/heavy) -- no mark = level

GREETINGS:
  Xin chào Hello (formal)   Chào Hello/Goodbye (informal)   Chào buổi sáng Good morning
  Bạn có khỏe không? How are you?   Tôi khỏe, cảm ơn I'm fine, thank you
  Cảm ơn Thank you   Xin lỗi Excuse me / I'm sorry   Không có gì You're welcome
  Vâng / Dạ Yes (respectful)   Không No   Tạm biệt Goodbye
  Tôi ten là... My name is...   Bạn ten là gì? What is your name?

GRAMMAR:
  No verb conjugation -- tense shown by time words or particles: đã(past), đang(ongoing), sẽ(future)
  Tôi ăn (I eat/ate/will eat) / Tôi đã ăn (I ate) / Tôi đang ăn (I am eating) / Tôi sẽ ăn (I will eat)
  Measure words (classifier + noun): con meo (a/the cat), cái bàn (a/the table), quyển sách (a/the book)
  SVO word order.
NUMBERS: một(1) hai(2) ba(3) bốn(4) năm(5) sáu(6) bảy(7) tám(8) chín(9) mười(10) hai mươi(20) trăm(100) nghìn(1000)""",
]

# -- Persian/Farsi (fa) --------------------------------------------------------
BUILTIN_LANG_DATA['fa'] = [

"""PERSIAN (FARSI) -- فارسی (Fārsī)

SCRIPT: Modified Perso-Arabic script. Written RIGHT-TO-LEFT. 32 letters.
Extra letters compared to Arabic: پ(p) چ(ch) ژ(zh) گ(g)
Short vowels often not written (same as Arabic). Long vowels: ا(ā) و(ū/w) ی(ī/y)

GREETINGS (سلام و احوالپرسی):
  سلام (Salām) Hello   درود (Dorūd) Hello (formal/poetic)   صبح بخیر (Sobh bekheyr) Good morning
  عصر بخیر (Asr bekheyr) Good afternoon   شب بخیر (Shab bekheyr) Good night
  خداحافظ (Khodāhāfez) Goodbye   حال شما چطور است؟ (Hāle shomā chetour ast?) How are you? (formal)
  خوبم، ممنون (Khubam, mamnun) I'm fine, thank you   خواهش می‌کنم (Khāhesh mikonam) You're welcome/Please
  متأسفم (Mota'assefam) I'm sorry   نمی‌فهمم (Nemifahmam) I don't understand
  اسم من ... است (Esme man ... ast) My name is...

GRAMMAR:
  EZAFE construction (اضافه): noun + e/ye connects to modifier
    خانه‌ی بزرگ (khāne-ye bozorg) big house (lit. house-of big)
  VERB endings (infinitive ends in -an): raftan(to go) -> present stem: rav
    می‌روم(I go) می‌روی(you go) می‌رود(he/she goes) می‌رویم(we go) می‌روید(you pl.) می‌روند(they go)
  SOV word order: من کتاب می‌خوانم (Man ketāb mikhānam) I book am-reading
  No grammatical gender   No case endings (unlike Arabic)
NUMBERS: یک(1) دو(2) سه(3) چهار(4) پنج(5) شش(6) هفت(7) هشت(8) نه(9) ده(10) بیست(20) صد(100) هزار(1000)""",
]

# -- Bengali (bn) --------------------------------------------------------------
BUILTIN_LANG_DATA['bn'] = [

"""BENGALI -- বাংলা (Bānglā)

SCRIPT: Bengali script (বাংলা লিপি), an abugida. Left-to-right. Related to Devanagari.
Characters hang from a headline (মাত্রা mātrā).

VOWELS (স্বরবর্ণ): অ(a) আ(ā) ই(i) ঈ(ī) উ(u) ঊ(ū) এ(e) ঐ(oi) ও(o) ঔ(ou)
CONSONANTS (ব্যঞ্জনবর্ণ):
  ক(k) খ(kh) গ(g) ঘ(gh) ঙ(ṅ)  চ(c) ছ(ch) জ(j) ঝ(jh) ঞ(n)
  ট(ṭ) ঠ(ṭh) ড(ḍ) ঢ(ḍh) ণ(ṇ)  ত(t) থ(th) দ(d) ধ(dh) ন(n)
  প(p) ফ(ph) ব(b) ভ(bh) ম(m)   য(j/y) র(r) ল(l) শ(sh) ষ(ṣ) স(s) হ(h)

GREETINGS:
  নমস্কার (Namaskār) Hello (formal, Hindu tradition)   আস-সালামু আলাইকুম (As-salāmu alaykum) Hello (Muslim)
  কেমন আছেন? (Kemon āchen?) How are you? (formal)   ভালো আছি, ধন্যবাদ (Bhālo āchi, dhanyabād) I'm fine, thank you
  ধন্যবাদ / শুক্রিয়া (Dhanyabād / Shukriyā) Thank you   দয়া করে (Dayā kare) Please
  মাফ করবেন (Māf karben) Excuse me / Sorry   আমি বুঝতে পারছি না (Āmi bujhte pārchi nā) I don't understand
  আলবিদা / বিদায় (Ālbidā / Bidāy) Goodbye   হ্যাঁ (Hyān) Yes   না (Nā) No
  আমার নাম ... (Āmār nām ...) My name is...

NUMBERS: এক(1) দুই(2) তিন(3) চার(4) পাঁচ(5) ছয়(6) সাত(7) আট(8) নয়(9) দশ(10) বিশ(20) একশো(100) এক হাজার(1000)""",
]

# -- Thai (th) -----------------------------------------------------------------
BUILTIN_LANG_DATA['th'] = [

"""THAI -- ภาษาไทย (Phāsā Thai)

SCRIPT: Thai abugida, unique script descended from Old Khmer. Left-to-right. No spaces between words.
44 consonants, 15 vowel symbols (combining), 4 tone marks.

CONSONANTS (พยัญชนะ) -- 3 classes affecting tone:
  High class: ข ฉ ถ ผ ฝ ศ ษ ส ห
  Mid class: ก จ ด ต บ ป อ
  Low class: ค ง ช ซ ญ ฎ ฏ ฐ ฑ ฒ ณ ท ธ น พ ฟ ภ ม ย ร ล ว ฬ ฮ

TONES (วรรณยุกต์): Thai has 5 tones.
  Mid tone (สามัญ): no mark   Low tone (เอก): ่   Falling tone (โท): ้
  High tone (ตรี): ๊   Rising tone (จัตวา): ๋
  Tone is also determined by consonant class + vowel length -- complex system.

GREETINGS:
  สวัสดี (Sawatdee) Hello/Goodbye [men add ครับ/khráp, women add ค่ะ/khâ]
  สวัสดีครับ (Sawatdee khráp) Hello (male)   สวัสดีค่ะ (Sawatdee khâ) Hello (female)
  สบายดีไหม? (Sabāi dii mǎi?) How are you?   สบายดีครับ/ค่ะ (Sabāi dii) I'm fine
  ขอบคุณ (Khòp khun) Thank you   ยินดี (Yin dii) You're welcome   ขอโทษ (Khǒr thôt) Sorry
  ไม่เข้าใจ (Mâi khâo jai) I don't understand   ใช่ (Châi) Yes   ไม่ใช่ / ไม่ (Mâi châi / Mâi) No/Not

GRAMMAR: Tonal language, analytic (no inflection). SVO order.
  Polite particles: ครับ (khráp, male) and ค่ะ/คะ (khâ, female) added to sentences for politeness.
  No verb conjugation. Tense by time adverbs or aspect markers: แล้ว(already/past) กำลัง(currently) จะ(will/future)
NUMBERS: หนึ่ง(1) สอง(2) สาม(3) สี่(4) ห้า(5) หก(6) เจ็ด(7) แปด(8) เก้า(9) สิบ(10) ยี่สิบ(20) ร้อย(100) พัน(1000)""",
]

# -- Multilingual phrase comparison table --------------------------------------
BUILTIN_LANG_DATA['_multilingual'] = [

"""MULTILINGUAL PHRASE COMPARISON -- Hello / Thank you / How are you?

English:    Hello / Thank you / How are you?
Chinese:    你好 (Nǐ hǎo) / 谢谢 (Xiexie) / 你好吗？(Nǐ hǎo ma?)
Japanese:   こんにちは (Konnichiwa) / ありがとう (Arigatō) / お元気ですか？(Ogenki desu ka?)
Korean:     안녕하세요 (Annyeonghaseyo) / 감사합니다 (Gamsahamnida) / 어떻게 지내세요? (Eotteoke jinaeseyo?)
Russian:    Здравствуйте (Zdravstvuyte) / Спасибо (Spasibo) / Как дела? (Kak dela?)
Arabic:     السلام عليكم (As-salāmu 'alaykum) / شكراً (Shukran) / كيف حالك؟ (Kayfa ḥālak?)
Hindi:      नमस्ते (Namaste) / धन्यवाद (Dhanyavād) / आप कैसे हैं? (Āp kaise haiṃ?)
Urdu:       السلام علیکم (As-salāmu 'alaykum) / شکریہ (Shukriyā) / آپ کیسے ہیں؟
French:     Bonjour / Merci / Comment allez-vous?
Spanish:    Hola / Gracias / ¿Cómo está usted?
Italian:    Ciao / Grazie / Come sta?
German:     Hallo / Danke / Wie geht es Ihnen?
Portuguese: Olá / Obrigado(a) / Como está?
Dutch:      Hallo / Dank je / Hoe gaat het?
Polish:     Cześć / Dziękuję / Jak się masz?
Turkish:    Merhaba / Teşekkur ederim / Nasılsınız?
Swahili:    Jambo / Asante / Habari yako?
Vietnamese: Xin chào / Cảm ơn / Bạn có khỏe không?
Persian:    سلام (Salām) / ممنون (Mamnun) / حال شما چطور است؟
Bengali:    নমস্কার (Namaskār) / ধন্যবাদ (Dhanyabād) / কেমন আছেন?
Thai:       สวัสดี (Sawatdee) / ขอบคุณ (Khòp khun) / สบายดีไหม?""",

"""MULTILINGUAL NUMBER COMPARISON -- 1 through 10

English: one two three four five six seven eight nine ten
Chinese: 一(yī) 二(er) 三(sān) 四(sì) 五(wǔ) 六(liù) 七(qī) 八(bā) 九(jiǔ) 十(shí)
Japanese: 一(ichi) 二(ni) 三(san) 四(shi/yon) 五(go) 六(roku) 七(shichi/nana) 八(hachi) 九(ku/kyū) 十(jū)
Korean: 일(il) 이(i) 삼(sam) 사(sa) 오(o) 육(yuk) 칠(chil) 팔(pal) 구(gu) 십(sip) -- Sino-Korean
Russian: один два три четыре пять шесть семь восемь девять десять
Arabic: واحد اثنان ثلاثة أربعة خمسة ستة سبعة ثمانية تسعة عشرة
Hindi: एक दो तीन चार पाँच छह सात आठ नौ दस
French: un deux trois quatre cinq six sept huit neuf dix
Spanish: uno dos tres cuatro cinco seis siete ocho nueve diez
Italian: uno due tre quattro cinque sei sette otto nove dieci
German: ein zwei drei vier funf sechs sieben acht neun zehn
Portuguese: um dois tres quatro cinco seis sete oito nove dez
Turkish: bir iki uç dort beş altı yedi sekiz dokuz on
Polish: jeden dwa trzy cztery pięć sześć siedem osiem dziewięć dziesięć
Dutch: een twee drie vier vijf zes zeven acht negen tien
Swahili: moja mbili tatu nne tano sita saba nane tisa kumi
Vietnamese: một hai ba bốn năm sáu bảy tám chín mười
Persian: یک دو سه چهار پنج شش هفت هشت نه ده
Bengali: এক দুই তিন চার পাঁচ ছয় সাত আট নয় দশ
Thai: หนึ่ง สอง สาม สี่ ห้า หก เจ็ด แปด เก้า สิบ""",

"""MULTILINGUAL LANGUAGE TEACHING -- How to Teach Foreign Languages

When teaching foreign languages, an expert instructor should:

1. SCRIPT FIRST: Introduce the writing system immediately. For non-Latin scripts:
   - Chinese: Teach radicals (部首 bùshǒu) -- common components like 人(person) 水(water) 木(tree) 口(mouth) 日(sun) 月(moon)
   - Japanese: Hiragana before Katakana before Kanji. Use furigana (reading aids) initially.
   - Arabic/Persian/Urdu: Emphasize RTL direction and letter-joining rules from day one.
   - Korean: Hangul is highly systematic -- students can learn to read in 1-2 days with practice.
   - Russian/Bulgarian: Cyrillic resembles Latin enough that learners progress quickly.
   - Hindi/Bengali/Thai/Devanagari: The headline/mātrā concept anchors the writing system.

2. PHONOLOGY: Master pronunciation before vocabulary.
   - Tonal languages (Chinese/Vietnamese/Thai): Tone drilling is essential before vocabulary.
   - Aspirated vs. unaspirated stops (Hindi/Chinese): ป vs. พ, प vs. फ, b vs. p in various languages.
   - Pharyngeal consonants (Arabic): ح ع require throat constriction exercises.
   - Rolling R (Spanish/Italian/Russian): Requires physical practice.
   - Vowel harmony (Turkish/Finnish): Explain suffix rules systematically.

3. CORE VOCABULARY: First 500 words in any language cover ~70% of everyday speech.
   Focus on: pronouns, common verbs (be have go come eat drink see say), numbers, time words, question words, family, body, food, directions.

4. GRAMMAR STRUCTURES: Teach patterns, not rules. "I want [noun]", "Where is [place]?", "How much does [item] cost?"

5. CULTURAL CONTEXT: Language cannot be separated from culture.
   - Japanese: Keigo (honorific speech levels), bowing, contextual communication
   - Arabic: Classical MSA vs. dialects (Egyptian, Levantine, Gulf, Moroccan)
   - Chinese: Traditional (繁體字) vs. Simplified (简体字) characters
   - Spanish: Castilian vs. Latin American pronunciation and vocabulary differences""",
]


def build_builtin_lang_corpus(node: str, args):
    print(f'\n[Stage 39] Built-in multilingual vocabulary, grammar, and scripts\n')
    trained = 0

    for lang_code, texts in BUILTIN_LANG_DATA.items():
        if lang_code == '_multilingual':
            label = 'Multilingual comparison tables'
        else:
            label = LANGUAGES.get(lang_code, lang_code)
        print(f'  {label}: {len(texts)} text block(s)...')
        for text in texts:
            try:
                header = f'Foreign Language Training -- {label}.\nSource: W1z4rDV1510n built-in multilingual corpus.\n\n'
                _train_text(header + text.strip(), node)
                trained += 1
            except Exception as e:
                print(f'  [WARN] {lang_code}: {e}')

    print(f'  Stage 39 done -- {trained} built-in text blocks trained')


# -- Stage 40: Wiktionary multilingual definitions -----------------------------

# Common words to look up -- these have rich multilingual Wiktionary entries
WIKTIONARY_LOOKUPS = [
    # Core concepts with deep etymology
    'water', 'fire', 'earth', 'air', 'sun', 'moon', 'star', 'sky',
    'house', 'food', 'bread', 'wine', 'milk', 'salt', 'gold', 'iron',
    'mother', 'father', 'child', 'man', 'woman', 'life', 'death', 'love',
    'god', 'king', 'war', 'peace', 'truth', 'beauty', 'time', 'world',
    # Common verbs
    'be', 'have', 'do', 'say', 'go', 'come', 'see', 'know', 'think', 'give',
    'take', 'make', 'hear', 'eat', 'drink', 'sleep', 'speak', 'write', 'read',
    # Adjectives
    'good', 'bad', 'big', 'small', 'new', 'old', 'long', 'short', 'hot', 'cold',
    'black', 'white', 'red', 'green', 'blue', 'high', 'low', 'fast', 'slow',
    # Numbers
    'one', 'two', 'three', 'four', 'five', 'ten', 'hundred', 'thousand',
    # Language-specific vocabulary
    'language', 'word', 'grammar', 'alphabet', 'tone', 'vowel', 'consonant',
    'translation', 'dictionary', 'pronunciation', 'sentence', 'verb', 'noun',
]

# Words in non-Latin scripts to look up in Wiktionary
WIKTIONARY_NATIVE_LOOKUPS = [
    '你好',   # Chinese hello
    '日本語', # Japanese language
    '한국어', # Korean language
    'сказать', # Russian to say
    'مرحبا',  # Arabic hello
    'धन्यवाद', # Hindi thank you
    'آب',     # Persian water
    'ভালো',   # Bengali good
    'สวัสดี', # Thai hello
    'xin chào', # Vietnamese hello
]


def _wiktionary_extract(word: str) -> str | None:
    api = (
        'https://en.wiktionary.org/w/api.php'
        f'?action=query&titles={urllib.parse.quote(word)}'
        '&prop=extracts&explaintext=1&exsectionformat=plain&format=json'
    )
    data = _get_json(api)
    if not data:
        return None
    pages = data.get('query', {}).get('pages', {})
    for page in pages.values():
        extract = page.get('extract', '')
        if len(extract) > 100:
            return extract[:3000]
    return None


def build_wiktionary_corpus(node: str, args):
    print(f'\n[Stage 40] Wiktionary -- multilingual definitions and etymology\n')
    trained = 0

    all_words = WIKTIONARY_LOOKUPS + WIKTIONARY_NATIVE_LOOKUPS
    for word in all_words:
        text = _wiktionary_extract(word)
        if not text:
            print(f'  SKIP {word!r}')
            time.sleep(0.3)
            continue
        try:
            _train_text(
                f'Wiktionary -- "{word}": multilingual definitions, etymology, translations.\n'
                f'Source: en.wiktionary.org (CC BY-SA)\n\n{text}',
                node
            )
            trained += 1
            print(f'  {word!r}: {len(text)} chars')
        except Exception as e:
            print(f'  [WARN] {word!r}: {e}')
        time.sleep(0.4)

    print(f'  Stage 40 done -- {trained} Wiktionary entries trained')


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description='Build foreign language corpus for W1z4rD node')
    ap.add_argument('--stages',             default='36,37,38,39,40')
    ap.add_argument('--node',               default='localhost:8090')
    ap.add_argument('--gutenberg-chars',    type=int, default=50000,
                    help='Max chars to read per Gutenberg book')
    ap.add_argument('--gutenberg-passages', type=int, default=15,
                    help='Max passages per Gutenberg book')
    ap.add_argument('--wikibooks-chars',    type=int, default=20000,
                    help='Max chars per Wikibooks page')
    args = ap.parse_args()

    stages = [int(s.strip()) for s in args.stages.split(',')]
    node   = args.node

    print('=' * 70)
    print('  W1z4rD V1510n -- Foreign Language Corpus Builder')
    print('=' * 70)
    print(f'  Node:      http://{node}')
    print(f'  Stages:    {stages}')
    print(f'  Languages: {", ".join(LANGUAGES.values())}')
    print()
    for s in stages:
        print(f'  Stage {s}: {STAGES.get(s, "?")}')
    print()

    if 36 in stages:
        build_gutenberg_native_corpus(node, args)
    if 37 in stages:
        build_gutenberg_ll_corpus(node, args)
    if 38 in stages:
        build_wikibooks_lang_corpus(node, args)
    if 39 in stages:
        build_builtin_lang_corpus(node, args)
    if 40 in stages:
        build_wiktionary_corpus(node, args)

    print('\n' + '=' * 70)
    print('  Foreign language corpus build complete.')
    print('=' * 70 + '\n')


if __name__ == '__main__':
    main()
