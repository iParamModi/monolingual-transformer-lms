#!/usr/bin/env python3
"""Crawl the manual-collection corpus for one language (Phase 1, stage 0).

This is the *collection* half of the manual pipeline. It reads a language's
source registry, discovers article URLs on each site, fetches them politely,
extracts the article text, and writes compressed JSONL shards.
scripts/combine_manual.py then merges those shards into the single combined
file that scripts/clean_manual.py consumes:

    <lang>/data/sources.yaml
        |  scrape_manual.py     (this script -- discover, fetch, extract)
        v
    <lang>/data/raw/manual/shards/shard-*.jsonl.zst
        |  combine_manual.py    (merge, dedup, per-source statistics)
        v
    <lang>/data/raw/manual/<code>_manual_combined.jsonl
        |  clean_manual.py      (normalise, language-ID, quality, dedup)
        v
    <lang>/data/processed/<code>_manual_clean.jsonl

Design notes, each of which is a viva-answerable choice:

  * Threads, not asyncio. A crawl is I/O-bound and the politeness delay
    dominates everything else, so a bounded ThreadPoolExecutor over
    requests.Session is enough. It avoids an async dependency and keeps the
    control flow readable -- there is no callback or event-loop reasoning
    needed to explain what happens to a single URL.
  * SQLite frontier. The URL queue lives on disk rather than in memory so an
    interrupted crawl resumes where it stopped. A multi-day crawl WILL be
    interrupted; losing the queue means re-discovering every sitemap.
  * One shard per N documents. A finished shard is never reopened, so a kill
    costs at most the shard in progress.
  * Text is extracted in-flight and the source HTML is discarded. Keeping a
    gzipped copy of every page would allow re-extraction without re-crawling,
    but it costs several times the size of the corpus itself, and the crawl is
    cheap enough to repeat if the extractor ever needs changing.

Everything this script writes lives under <lang>/data/raw/manual/: the shards
and the frontier database, nothing else.

Politeness: robots.txt is consulted and cached per domain, a minimum interval
is enforced per domain, and the User-Agent identifies the crawler and carries a
contact address. Sites that disallow a path are skipped, not worked around.

Budget arithmetic (Hindi, aiming at ~100M manual tokens):
    ~900 tokens/article  ->  ~110k articles needed *after* cleaning and dedup
    cleaning + dedup keeps ~65%  ->  target ~170k fetched URLs

Usage:
    python scripts/scrape_manual.py --lang hi --discover
    python scripts/scrape_manual.py --lang hi --max-docs 50000
    python scripts/scrape_manual.py --lang ne --discover --max-urls-per-site 20000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import threading
import time
import urllib.robotparser
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent

USER_AGENT = (
    "LMA-CourseProject-Crawler/1.0 "
    "(IIIT Hyderabad monsoon-2026 individual project; contact: gm.itsparam@gmail.com)"
)

# Status codes worth a second attempt: rate limiting and transient server
# faults. A 404 or 403 is a real answer and is not retried.
RETRY_STATUS = {429, 500, 502, 503, 504}

# Sites whose article text is better taken from the MediaWiki Action API than
# scraped from rendered HTML -- the API returns clean plaintext with no
# navigation chrome, which removes a whole class of extraction bugs.
MEDIAWIKI_HOSTS = (
    "wikipedia.org",
    "wikisource.org",
    "wikibooks.org",
    "wikiquote.org",
    "kavitakosh.org",
)

# Conventional locations for a sitemap index, tried in order before falling
# back to the REST API and then to link crawling.
SITEMAP_PATHS = (
    "/robots.txt",
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-index.xml",
    "/sitemap/sitemap.xml",
    "/news-sitemap.xml",
)

# HTML elements that never contain article prose. Dropping them before reading
# text is what separates the body from the navigation.
BOILERPLATE_TAGS = (
    "script", "style", "nav", "header", "footer", "aside",
    "form", "button", "noscript", "iframe", "svg",
)

# File extensions that are never an article. Sitemaps routinely list media and
# assets alongside pages, and fetching them wastes the crawl budget on payloads
# that can only be discarded after download.
NON_ARTICLE_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp", ".ico", ".avif",
    ".mp4", ".webm", ".mov", ".avi", ".mp3", ".wav", ".ogg", ".m4a",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".gz", ".rar", ".7z", ".css", ".js", ".json", ".rss",
)

# A Devanagari-bearing paragraph must clear this to count as article text.
MIN_PARAGRAPH_CHARS = 40
# Below this a "document" is a stub, a redirect notice, or an error page.
MIN_DOCUMENT_CHARS = 200

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# --------------------------------------------------------------------- config

class LanguageConfig:
    """One language's crawl settings, read from <lang>/data/sources.yaml."""

    def __init__(self, root: Path, code: str):
        folder = {"hi": "hindi", "ne": "nepali"}[code]
        self.code = code
        self.folder = folder
        self.root = root
        self.path = root / folder / "data" / "sources.yaml"
        if not self.path.exists():
            sys.exit(f"error: no source registry at {self.path}")

        try:
            import yaml
        except ImportError:
            sys.exit("error: PyYAML is required to read sources.yaml.  pip install PyYAML")
        with open(self.path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}

        self.lang_code = cfg.get("lang_code", code)
        self.rps_per_domain = float(cfg.get("rps_per_domain", 1.5))
        self.category_priority = list(cfg.get("category_priority", []))
        self.sites = [s for s in cfg.get("manual_sites", []) if isinstance(s, dict)]
        if not self.sites:
            sys.exit(f"error: {self.path} lists no manual_sites")

    @property
    def data_dir(self) -> Path:
        return self.root / self.folder / "data" / "raw" / "manual"

    @property
    def shard_dir(self) -> Path:
        return self.data_dir / "shards"

    @property
    def frontier_path(self) -> Path:
        return self.data_dir / "frontier.sqlite3"

    def priority_of(self, category: str) -> int:
        """Lower sorts first. Categories absent from the list go last.

        The crawl is dominated by a handful of enormous news archives; without
        weighting, the smaller registers (poetry, literature, sports) would be
        a rounding error in the finished corpus and the model would only ever
        learn to write news.
        """
        try:
            return self.category_priority.index(category)
        except ValueError:
            return len(self.category_priority)


# ------------------------------------------------------------------- frontier

class Frontier:
    """Resumable URL queue backed by SQLite.

    States: pending -> claimed -> done | failed | skipped. `claimed` exists so
    an interrupted run can tell "in flight when we died" from "never started"
    and requeue only the former.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS urls (
                   url      TEXT PRIMARY KEY,
                   domain   TEXT NOT NULL,
                   source   TEXT NOT NULL,
                   priority INTEGER NOT NULL DEFAULT 0,
                   state    TEXT NOT NULL DEFAULT 'pending',
                   n_chars  INTEGER DEFAULT 0
               )"""
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_state ON urls(state, priority)")
        self.conn.commit()

    def add_many(self, rows) -> int:
        """Insert (url, domain, source, priority) rows, ignoring known URLs."""
        with self.lock:
            cur = self.conn.executemany(
                "INSERT OR IGNORE INTO urls (url, domain, source, priority) VALUES (?,?,?,?)",
                rows,
            )
            self.conn.commit()
            return cur.rowcount

    def requeue_claimed(self) -> int:
        """Return URLs abandoned mid-flight by a previous run to the queue."""
        with self.lock:
            cur = self.conn.execute("UPDATE urls SET state='pending' WHERE state='claimed'")
            self.conn.commit()
            return cur.rowcount

    def claim(self, limit: int) -> list:
        """Take up to `limit` pending URLs, highest priority first."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT url, source FROM urls WHERE state='pending' "
                "ORDER BY priority ASC LIMIT ?",
                (limit,),
            ).fetchall()
            if rows:
                self.conn.executemany(
                    "UPDATE urls SET state='claimed' WHERE url=?",
                    [(r[0],) for r in rows],
                )
                self.conn.commit()
            return rows

    def mark(self, url: str, state: str, n_chars: int = 0) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE urls SET state=?, n_chars=? WHERE url=?", (state, n_chars, url)
            )
            self.conn.commit()

    def counts(self) -> dict:
        with self.lock:
            return dict(self.conn.execute("SELECT state, COUNT(*) FROM urls GROUP BY state"))

    def close(self) -> None:
        self.conn.close()


# ----------------------------------------------------------------- politeness

class Politeness:
    """Per-domain robots.txt compliance and request spacing."""

    def __init__(self, session, delay: float):
        self.session = session
        self.delay = delay
        self.robots = {}
        self.last_hit = {}
        self.lock = threading.Lock()

    def allowed(self, url: str) -> bool:
        """Consult robots.txt, fetching and caching one parser per domain."""
        domain = urlparse(url).netloc
        with self.lock:
            parser = self.robots.get(domain)
        if parser is None:
            parser = urllib.robotparser.RobotFileParser()
            try:
                r = self.session.get(f"https://{domain}/robots.txt", timeout=20)
                parser.parse(r.text.splitlines() if r.status_code == 200 else [])
            except Exception:
                # An unreachable robots.txt is not consent, but treating it as a
                # block would strand whole domains on one flaky request. The
                # rate limit below still applies.
                parser.parse([])
            with self.lock:
                self.robots[domain] = parser
        try:
            return parser.can_fetch(USER_AGENT, url)
        except Exception:
            return True

    def wait(self, domain: str) -> None:
        """Block until this domain's minimum interval has elapsed."""
        while True:
            with self.lock:
                now = time.monotonic()
                ready = self.last_hit.get(domain, 0.0) + self.delay
                if now >= ready:
                    self.last_hit[domain] = now
                    return
                sleep_for = ready - now
            time.sleep(sleep_for)


# ------------------------------------------------------------------- fetching

class Fetcher:
    """HTTP GET with retry/backoff, shared across worker threads."""

    def __init__(self, politeness: Politeness, timeout: int = 30):
        self.p = politeness
        self.timeout = timeout

    def get(self, url: str, tries: int = 3, respect_delay: bool = True):
        """Return a Response, or None once the retries are exhausted."""
        domain = urlparse(url).netloc
        for attempt in range(tries):
            if respect_delay:
                self.p.wait(domain)
            try:
                r = self.p.session.get(url, timeout=self.timeout)
            except Exception:
                time.sleep(2 ** attempt)
                continue
            if r.status_code in RETRY_STATUS:
                # Exponential backoff: a 429 answered immediately is still a 429.
                time.sleep(5 * (2 ** attempt))
                continue
            return r
        return None


# ------------------------------------------------------------------ discovery

def is_mediawiki(domain: str) -> bool:
    return any(h in domain for h in MEDIAWIKI_HOSTS)


def mediawiki_api(domain: str, scheme: str = "https") -> str:
    """Action API endpoint. Most wikis serve /w/api.php; Kavita Kosh /wiki/."""
    path = "/wiki/api.php" if "kavitakosh" in domain else "/w/api.php"
    return f"{scheme}://{domain}{path}"


def looks_like_article(url: str) -> bool:
    """False for URLs whose extension marks them as media or an asset.

    The path is checked without the query string, since '?' parameters often
    follow an extension (…/photo.jpg?w=640).
    """
    path = urlparse(url).path.lower()
    return not path.endswith(NON_ARTICLE_SUFFIXES)


class Discovery:
    """Turn a site root into a list of article URLs.

    Four strategies, tried in order of how complete and cheap they are:
      1. MediaWiki Action API -- exhaustive and paginated, so a wiki is fully
         enumerated rather than sampled.
      2. XML sitemaps -- what a news site publishes for search engines, which
         is exactly the archive listing we want.
      3. WordPress REST API -- many smaller outlets run WordPress and expose
         /wp-json even when their sitemap is missing or broken.
      4. Breadth-first link crawl -- last resort, slow and incomplete, but
         better than abandoning a domain.
    """

    def __init__(self, fetcher: Fetcher):
        self.f = fetcher

    def mediawiki(self, root: str, max_urls: int) -> list:
        parsed = urlparse(root)
        domain, scheme = parsed.netloc, parsed.scheme or "https"
        api, out, cont = mediawiki_api(domain, scheme), [], None

        while len(out) < max_urls:
            params = {
                "action": "query", "list": "allpages", "apnamespace": "0",
                "aplimit": "500", "format": "json",
            }
            if cont:
                params["apcontinue"] = cont
            self.f.p.wait(domain)
            try:
                r = self.f.p.session.get(api, params=params, timeout=self.f.timeout)
                if r.status_code != 200:
                    break
                data = r.json()
            except Exception:
                break

            pages = data.get("query", {}).get("allpages", [])
            if not pages:
                break
            for item in pages:
                title = item.get("title", "")
                if title:
                    slug = quote(title.replace(" ", "_"))
                    out.append(f"{scheme}://{domain}/wiki/{slug}")
                    if len(out) >= max_urls:
                        break
            cont = data.get("continue", {}).get("apcontinue")
            if not cont:
                break
        return out

    def _parse_sitemap(self, content: bytes):
        """Return (nested sitemap urls, page urls) from a sitemap document."""
        try:
            rootel = ET.fromstring(content)
        except ET.ParseError:
            return [], []
        nested, pages = [], []
        # Sitemap XML is namespaced; matching on the local tag name avoids
        # hard-coding a namespace URI that varies between generators.
        #
        # A <loc> inside a <sitemapindex> points at another sitemap, and one
        # inside a <urlset> points at a page. ElementTree gives no parent
        # pointer, so the URL suffix decides: sitemaps are .xml/.xml.gz,
        # articles are not.
        for el in rootel.iter():
            tag = el.tag
            namespace = tag[1:tag.index("}")] if tag.startswith("{") else ""
            if tag.rsplit("}", 1)[-1] != "loc":
                continue
            # An image or video sitemap nests <image:loc> INSIDE a <url> entry,
            # and its local tag name is also "loc". Without this check every
            # illustration on a page is queued as though it were an article.
            if "sitemap-image" in namespace or "sitemap-video" in namespace:
                continue
            loc = (el.text or "").strip()
            if not loc:
                continue
            if loc.endswith((".xml", ".xml.gz")):
                nested.append(loc)
            elif looks_like_article(loc):
                pages.append(loc)
        return nested, pages

    def _sitemaps_from_robots(self, text: str) -> list:
        return [
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.lower().startswith("sitemap:")
        ]

    def sitemaps(self, root: str, max_urls: int) -> list:
        base = root.rstrip("/")
        pending = [base + p for p in SITEMAP_PATHS]
        seen, pages = set(), []

        while pending and len(pages) < max_urls:
            url = pending.pop(0)
            if url in seen:
                continue
            seen.add(url)
            r = self.f.get(url, tries=2)
            if r is None or r.status_code != 200:
                continue
            if url.endswith("robots.txt"):
                pending.extend(s for s in self._sitemaps_from_robots(r.text) if s not in seen)
                continue
            nested, found = self._parse_sitemap(r.content)
            pending.extend(n for n in nested if n not in seen)
            pages.extend(found)
        return pages

    def wordpress(self, root: str, max_urls: int) -> list:
        parsed = urlparse(root)
        domain, scheme = parsed.netloc, parsed.scheme or "https"
        api = f"{scheme}://{domain}/wp-json/wp/v2/posts"
        out, page = [], 1

        while len(out) < max_urls and page <= 200:
            self.f.p.wait(domain)
            try:
                r = self.f.p.session.get(
                    api, params={"per_page": 100, "page": page, "_fields": "link"},
                    timeout=self.f.timeout,
                )
                if r.status_code != 200:
                    break
                items = r.json()
            except Exception:
                break
            if not isinstance(items, list) or not items:
                break
            out.extend(i["link"] for i in items if isinstance(i, dict) and i.get("link"))
            page += 1
        return out[:max_urls]

    def link_crawl(self, root: str, max_urls: int, max_depth: int = 2) -> list:
        """Breadth-first walk of same-domain links."""
        from bs4 import BeautifulSoup

        parsed = urlparse(root)
        domain, scheme = parsed.netloc, parsed.scheme or "https"
        start = f"{scheme}://{domain}"
        seen, out, level = {start}, [], [start]

        for _ in range(max_depth + 1):
            if not level or len(out) >= max_urls:
                break
            nxt = []
            for url in level:
                if len(out) >= max_urls:
                    break
                if not self.f.p.allowed(url):
                    continue
                r = self.f.get(url, tries=1)
                if r is None or r.status_code != 200:
                    continue
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    link = urljoin(url, a["href"]).split("#")[0]
                    if urlparse(link).netloc != domain or link in seen:
                        continue
                    if not looks_like_article(link):
                        continue  # same reason as in the sitemap parser
                    seen.add(link)
                    out.append(link)
                    nxt.append(link)
                    if len(out) >= max_urls:
                        break
            level = nxt
        return out

    def discover(self, root: str, max_urls: int) -> list:
        domain = urlparse(root).netloc
        if is_mediawiki(domain):
            found = self.mediawiki(root, max_urls)
            if found:
                return found

        found = self.sitemaps(root, max_urls)
        # De-duplicate while preserving discovery order.
        seen, out = set(), []
        for u in found:
            if u not in seen:
                seen.add(u)
                out.append(u)

        # A near-empty result means the sitemap was missing or unparseable, not
        # that the site is small -- fall through to the cheaper-to-parse APIs.
        if len(out) < 50:
            for extra in self.wordpress(root, min(max_urls, 20_000)):
                if extra not in seen:
                    seen.add(extra)
                    out.append(extra)
        if len(out) < 50:
            for extra in self.link_crawl(root, min(max_urls, 10_000)):
                if extra not in seen:
                    seen.add(extra)
                    out.append(extra)
        return out[:max_urls]


# ----------------------------------------------------------------- extraction

class Extractor:
    """Pull article prose out of a rendered HTML page."""

    def text_from_html(self, html: str) -> str:
        """Strip chrome, then keep paragraphs that look like body text.

        Deliberately conservative: it is better to lose a short pull-quote than
        to admit a navigation menu, because menu text repeats across every page
        of a domain and would survive into the corpus as high-frequency noise.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(list(BOILERPLATE_TAGS)):
            tag.decompose()

        node = soup.find("article") or soup.find("main") or soup.body or soup
        blocks = []
        for p in node.find_all(["p", "h1", "h2", "h3", "li"]):
            chunk = " ".join(p.get_text(" ", strip=True).split())
            if len(chunk) >= MIN_PARAGRAPH_CHARS and DEVANAGARI.search(chunk):
                blocks.append(chunk)

        # Repeated blocks within one page are almost always furniture
        # (breadcrumbs, "read more" rails) rather than prose.
        seen, kept = set(), []
        for b in blocks:
            if b not in seen:
                seen.add(b)
                kept.append(b)
        return "\n".join(kept)

    def text_from_mediawiki(self, fetcher: Fetcher, url: str) -> str:
        """Ask the Action API for plaintext instead of scraping the skin."""
        parsed = urlparse(url)
        domain, scheme = parsed.netloc, parsed.scheme or "https"
        parts = parsed.path.split("/wiki/", 1)
        if len(parts) < 2:
            return ""
        params = {
            "action": "query", "prop": "extracts", "explaintext": "1",
            "titles": unquote(parts[1].replace("_", " ")),
            "format": "json", "redirects": "1",
        }
        fetcher.p.wait(domain)
        try:
            r = fetcher.p.session.get(
                mediawiki_api(domain, scheme), params=params, timeout=fetcher.timeout
            )
            if r.status_code != 200:
                return ""
            pages = r.json().get("query", {}).get("pages", {})
        except Exception:
            return ""
        for pid, data in pages.items():
            if pid != "-1" and data.get("extract"):
                return data["extract"]
        return ""


# -------------------------------------------------------------------- writing

class ShardWriter:
    """Write documents to rotating zstd-compressed JSONL shards.

    Thread-safe: worker threads hand finished documents here. Rotating on a
    document count (rather than one big file) means an interrupted crawl loses
    only the shard in progress.
    """

    def __init__(self, out_dir: Path, shard_docs: int = 1000):
        try:
            import zstandard as zstd
        except ImportError:
            sys.exit("error: zstandard is required to write shards.  pip install zstandard")
        self.zstd = zstd
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_docs = shard_docs
        self.lock = threading.Lock()
        self.fh = None
        self.writer = None
        self.n_in_shard = 0
        # Continue numbering after whatever a previous run already wrote.
        existing = sorted(self.out_dir.glob("shard-*.jsonl.zst"))
        self.index = len(existing)

    def _open(self) -> None:
        path = self.out_dir / f"shard-{self.index:05d}.jsonl.zst"
        self.fh = open(path, "wb")
        self.writer = self.zstd.ZstdCompressor(level=10).stream_writer(self.fh)
        self.n_in_shard = 0

    def add(self, doc: dict) -> None:
        with self.lock:
            if self.writer is None:
                self._open()
            line = json.dumps(doc, ensure_ascii=False) + "\n"
            self.writer.write(line.encode("utf-8"))
            self.n_in_shard += 1
            if self.n_in_shard >= self.shard_docs:
                self._close_current()
                self.index += 1

    def _close_current(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.fh.close()
            self.writer = None
            self.fh = None

    def close(self) -> None:
        with self.lock:
            self._close_current()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def build_document(text: str, lang_code: str, source: str, url: str) -> dict:
    """Assemble one corpus record.

    The field layout matches what combine_manual.py and clean_manual.py expect;
    doc_id is a content hash so the same article fetched twice collapses to one
    record during the merge.
    """
    normalised = " ".join(text.split())
    return {
        "text": text,
        "lang": lang_code,
        "source": source,
        "source_type": "manual",
        "url": url,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "doc_id": hashlib.blake2b(normalised.encode("utf-8"), digest_size=8).hexdigest(),
        "flags": {"n_chars": len(text), "n_bytes": len(text.encode("utf-8"))},
    }


# -------------------------------------------------------------------- crawler

class ManualCrawler:
    """Drive discovery and fetching for one language."""

    def __init__(self, cfg: LanguageConfig, workers: int = 16):
        import requests

        self.cfg = cfg
        self.workers = workers
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.politeness = Politeness(self.session, delay=1.0 / cfg.rps_per_domain)
        self.fetcher = Fetcher(self.politeness)
        self.discovery = Discovery(self.fetcher)
        self.extractor = Extractor()
        self.frontier = Frontier(cfg.frontier_path)
        self.n_ok = 0
        self.n_fail = 0
        self.n_chars = 0
        self.counter_lock = threading.Lock()

    # ---- discovery -------------------------------------------------------

    def discover_all(self, max_urls_per_site: int) -> None:
        """Populate the frontier from every site in the registry."""
        sites = sorted(
            self.cfg.sites, key=lambda s: self.cfg.priority_of(s.get("category", ""))
        )
        for site in sites:
            name = site.get("name", site.get("domain", "?"))
            root = site.get("root")
            domain = site.get("domain", urlparse(root or "").netloc)
            if not root:
                print(f"[discover] {name}: no root declared, skipping")
                continue

            started = time.time()
            try:
                urls = self.discovery.discover(root, max_urls_per_site)
            except Exception as exc:
                print(f"[discover] {name}: failed ({exc.__class__.__name__})")
                continue

            priority = self.cfg.priority_of(site.get("category", ""))
            added = self.frontier.add_many(
                [(u, domain, name, priority) for u in urls]
            )
            print(
                f"[discover] {name:<22} found={len(urls):>7,} new={added:>7,} "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )

    # ---- fetching --------------------------------------------------------

    def fetch_one(self, url: str, source: str, writer: ShardWriter) -> None:
        """Fetch, extract and emit a single page."""
        if not self.politeness.allowed(url):
            self.frontier.mark(url, "skipped")
            return

        domain = urlparse(url).netloc
        text = ""

        if is_mediawiki(domain) and "/wiki/" in url:
            text = self.extractor.text_from_mediawiki(self.fetcher, url)

        if not text:
            r = self.fetcher.get(url)
            if r is None or r.status_code != 200:
                self.frontier.mark(url, "failed")
                with self.counter_lock:
                    self.n_fail += 1
                return
            text = self.extractor.text_from_html(r.text)

        if len(text) < MIN_DOCUMENT_CHARS:
            self.frontier.mark(url, "failed", len(text))
            with self.counter_lock:
                self.n_fail += 1
            return

        writer.add(build_document(text, self.cfg.lang_code, source, url))
        self.frontier.mark(url, "done", len(text))
        with self.counter_lock:
            self.n_ok += 1
            self.n_chars += len(text)

    def run(self, max_docs: int | None, batch: int = 256) -> None:
        """Drain the frontier until it empties or max_docs is reached."""
        requeued = self.frontier.requeue_claimed()
        if requeued:
            print(f"[crawl] requeued {requeued:,} URLs left in flight by a previous run")

        counts = self.frontier.counts()
        if not counts.get("pending"):
            print(
                "[crawl] frontier is empty. Run with --discover first:\n"
                f"        python scripts/scrape_manual.py --lang {self.cfg.code} --discover"
            )
            return
        print(f"[crawl] {counts.get('pending', 0):,} URLs pending, {self.workers} workers")

        started = time.time()
        with ShardWriter(self.cfg.shard_dir) as writer:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                while True:
                    claimed = self.frontier.claim(batch)
                    if not claimed:
                        break
                    list(pool.map(
                        lambda row: self.fetch_one(row[0], row[1], writer), claimed
                    ))
                    elapsed = time.time() - started
                    print(
                        f"[crawl] ok={self.n_ok:,} fail={self.n_fail:,} "
                        f"chars={self.n_chars:,} ({self.n_ok / max(elapsed, 1):.1f} docs/s)",
                        flush=True,
                    )
                    if max_docs and self.n_ok >= max_docs:
                        print(f"[crawl] reached --max-docs {max_docs:,}")
                        break

        print(
            f"\n[crawl] finished: {self.n_ok:,} documents, {self.n_chars:,} characters "
            f"in {time.time() - started:.0f}s"
        )
        print(f"[crawl] shards written to {self.cfg.shard_dir}")
        print("\nNext stage:  python scripts/combine_manual.py")

    def close(self) -> None:
        self.frontier.close()
        self.session.close()


# ------------------------------------------------------------------------ cli

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--lang", choices=["hi", "ne"], required=True,
                    help="which language's sources.yaml to crawl")
    ap.add_argument("--root", default=str(REPO_ROOT),
                    help="repo root containing hindi/ and nepali/")
    ap.add_argument("--discover", action="store_true",
                    help="expand every site's sitemap into the frontier, then stop")
    ap.add_argument("--max-urls-per-site", type=int, default=200_000,
                    help="cap on URLs discovered per site (default: 200000)")
    ap.add_argument("--max-docs", type=int, default=None,
                    help="stop after N successfully fetched documents")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent fetches across all domains (default: 16)")
    args = ap.parse_args()

    cfg = LanguageConfig(Path(args.root).resolve(), args.lang)
    crawler = ManualCrawler(cfg, workers=args.workers)
    try:
        if args.discover:
            crawler.discover_all(args.max_urls_per_site)
            pending = crawler.frontier.counts().get("pending", 0)
            print(f"\n[discover] frontier now holds {pending:,} pending URLs")
        else:
            crawler.run(args.max_docs)
    finally:
        crawler.close()


if __name__ == "__main__":
    main()

