from scripts.market_news_corpus import article_key, canonical_url, normalize_article


def test_canonical_url_removes_tracking_but_preserves_semantic_query():
    url = canonical_url("HTTPS://CoinDesk.com/story/?asset=btc&utm_source=rss#fragment")
    assert url == "https://coindesk.com/story?asset=btc"


def test_news_identity_preserves_distinct_publication_times():
    base = {"headline": "Bitcoin update", "source": "CoinDesk",
            "url": "https://coindesk.com/update", "timestamp": 1700000000,
            "tokens": ["BTC"], "sentiment": "neutral"}
    first = normalize_article(base)
    second = normalize_article({**base, "timestamp": 1700086400})
    assert first is not None and second is not None
    assert article_key(first) != article_key(second)


def test_irrelevant_generic_advisory_is_filtered():
    assert normalize_article({"headline": "Package parser security advisory",
                              "timestamp": 1700000000, "source": "GitHub"}) is None
