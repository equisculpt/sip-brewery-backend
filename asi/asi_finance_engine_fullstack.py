"""
Institutional-Grade Backend Finance Search Engine (Fullstack, Self-Contained)
- Multi-source async crawling (NSE, MoneyControl, Yahoo)
- Smart anti-bot headers, retry, fallback
- Full normalization, enrichment, error logging
- In-memory full-text/entity/symbol search
- No public interface; backend Python API only
"""
import asyncio
import aiohttp
from typing import List, Dict, Any
from datetime import datetime
import re
import random
import logging

logging.basicConfig(level=logging.INFO)

# --- 1. Data Source Plugins ---

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Add more if needed
]

HEADERS_BASE = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "DNT": "1",
    "Referer": "https://www.nseindia.com/",
    "Origin": "https://www.nseindia.com",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
}

def get_random_headers():
    h = HEADERS_BASE.copy()
    h["User-Agent"] = random.choice(USER_AGENTS)
    return h

class DataSource:
    name: str
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

from asi.nse_cookie_manager import get_nse_cookies

class NSESource(DataSource):
    name = "NSE"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={query.upper()}"
        cookies = await get_nse_cookies()
        for attempt in range(2):  # Try twice: once with fresh cookies, once fallback
            try:
                jar = aiohttp.CookieJar()
                for k, v in cookies.items():
                    jar.update_cookies({k: v})
                async with aiohttp.ClientSession(cookie_jar=jar) as nse_session:
                    async with nse_session.get(url, headers=get_random_headers(), timeout=10) as resp:
                        text = await resp.text()
                        if resp.status == 200 and 'priceInfo' in text:
                            data = await resp.json()
                            price_info = data.get('priceInfo', {})
                            return [{
                                "symbol": query.upper(),
                                "price": price_info.get('lastPrice'),
                                "change": price_info.get('change'),
                                "change_percent": price_info.get('pChange'),
                                "volume": data.get('securityWiseDP', {}).get('quantityTraded'),
                                "timestamp": datetime.now(),
                                "source": "NSE",
                                "content": str(data)
                            }]
                        elif 'not found' in text.lower() or 'captcha' in text.lower() or '<html' in text.lower():
                            logging.warning(f"NSE blocked or not found for {query}: {text[:120]}")
                            if attempt == 0:
                                cookies = await get_nse_cookies()  # Refresh cookies and retry
                                continue
                            return []
                        else:
                            logging.warning(f"NSE unknown response for {query}: {text[:120]}")
                            return []
            except Exception as e:
                logging.error(f"NSE fetch error for {query}: {e}")
                if attempt == 0:
                    cookies = await get_nse_cookies()
                    continue
                return []
        return []

class MoneyControlSource(DataSource):
    name = "MoneyControl"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        # MoneyControl URL pattern
        url = f"https://www.moneycontrol.com/india/stockpricequote/{query.lower()}"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                # Extract price using regex (simple, demo)
                m = re.search(r'<div[^>]*id="Nse_Prc_tick"[^>]*>([0-9.,]+)</div>', html)
                price = m.group(1) if m else None
                if price:
                    return [{
                        "symbol": query.upper(),
                        "price": price,
                        "timestamp": datetime.now(),
                        "source": "MoneyControl",
                        "content": html[:500]
                    }]
                else:
                    logging.warning(f"MoneyControl no price for {query}")
                    return []
        except Exception as e:
            logging.error(f"MoneyControl fetch error for {query}: {e}")
            return []

class YahooFinanceSource(DataSource):
    name = "YahooFinance"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={query.upper()}.NS"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                data = await resp.json()
                result = data.get('quoteResponse', {}).get('result', [])
                if result:
                    r = result[0]
                    return [{
                        "symbol": r.get("symbol"),
                        "price": r.get("regularMarketPrice"),
                        "change": r.get("regularMarketChange"),
                        "change_percent": r.get("regularMarketChangePercent"),
                        "volume": r.get("regularMarketVolume"),
                        "timestamp": datetime.fromtimestamp(r.get("regularMarketTime", 0)),
                        "source": "YahooFinance",
                        "content": str(r)
                    }]
                else:
                    logging.warning(f"YahooFinance no result for {query}")
                    return []
        except Exception as e:
            logging.error(f"YahooFinance fetch error for {query}: {e}")
            return []

import bs4

class EconomicTimesNewsSource(DataSource):
    name = "EconomicTimesNews"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://economictimes.indiatimes.com/markets/stocks/stock-quotes/{query.lower()}"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                news_items = []
                for div in soup.find_all("div", class_="eachStory"):
                    title = div.find("a").get_text(strip=True) if div.find("a") else None
                    link = div.find("a")['href'] if div.find("a") else None
                    summary = div.find("p").get_text(strip=True) if div.find("p") else ""
                    if title and link:
                        news_items.append({
                            "symbol": query.upper(),
                            "title": title,
                            "summary": summary,
                            "url": link,
                            "timestamp": datetime.now(),
                            "source": "EconomicTimesNews",
                            "content": title + " " + summary
                        })
                return news_items
        except Exception as e:
            logging.error(f"EconomicTimesNews fetch error for {query}: {e}")
            return []

class BSEFilingsSource(DataSource):
    name = "BSEFilings"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.bseindia.com/corporates/ann.html?scrip={query.upper()}&dur=A"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                filings = []
                for row in soup.find_all("tr", class_="td_bg"):
                    cols = row.find_all("td")
                    if len(cols) >= 4:
                        date = cols[0].get_text(strip=True)
                        headline = cols[2].get_text(strip=True)
                        link = cols[2].find("a")['href'] if cols[2].find("a") else None
                        filings.append({
                            "symbol": query.upper(),
                            "headline": headline,
                            "date": date,
                            "url": link,
                            "timestamp": datetime.now(),
                            "source": "BSEFilings",
                            "content": headline
                        })
                return filings
        except Exception as e:
            logging.error(f"BSEFilings fetch error for {query}: {e}")
            return []

class ReutersNewsSource(DataSource):
    name = "ReutersNews"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.reuters.com/companies/{query.upper()}.NS/news"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                news_items = []
                for div in soup.find_all("div", class_="item-title"):
                    title = div.get_text(strip=True)
                    link = div.find_parent("a")['href'] if div.find_parent("a") else None
                    if title and link:
                        news_items.append({
                            "symbol": query.upper(),
                            "title": title,
                            "url": f"https://www.reuters.com{link}" if link.startswith("/") else link,
                            "timestamp": datetime.now(),
                            "source": "ReutersNews",
                            "content": title
                        })
                return news_items
        except Exception as e:
            logging.error(f"ReutersNews fetch error for {query}: {e}")
            return []

class YahooFinanceNewsSource(DataSource):
    name = "YahooFinanceNews"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://finance.yahoo.com/quote/{query.upper()}.NS/news"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                news_items = []
                for li in soup.find_all("li", class_="js-stream-content"):
                    title = li.find("h3")
                    link = li.find("a")['href'] if li.find("a") else None
                    if title and link:
                        news_items.append({
                            "symbol": query.upper(),
                            "title": title.get_text(strip=True),
                            "url": f"https://finance.yahoo.com{link}" if link.startswith("/") else link,
                            "timestamp": datetime.now(),
                            "source": "YahooFinanceNews",
                            "content": title.get_text(strip=True)
                        })
                return news_items
        except Exception as e:
            logging.error(f"YahooFinanceNews fetch error for {query}: {e}")
            return []

class SEBIFilingsSource(DataSource):
    name = "SEBIFilings"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=1&smid=0&ssid=0&keyword={query.upper()}"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                filings = []
                for row in soup.find_all("tr", class_="alt-row"):
                    cols = row.find_all("td")
                    if len(cols) >= 3:
                        date = cols[0].get_text(strip=True)
                        headline = cols[1].get_text(strip=True)
                        link = cols[1].find("a")['href'] if cols[1].find("a") else None
                        filings.append({
                            "symbol": query.upper(),
                            "headline": headline,
                            "date": date,
                            "url": link,
                            "timestamp": datetime.now(),
                            "source": "SEBIFilings",
                            "content": headline
                        })
                return filings
        except Exception as e:
            logging.error(f"SEBIFilings fetch error for {query}: {e}")
            return []

# --- Additional News/Filings Sources ---
class BloombergNewsSource(DataSource):
    name = "BloombergNews"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.bloomberg.com/search?query={query}"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                news_items = []
                for div in soup.find_all("div", class_="search-result-story__container"):
                    title_tag = div.find("a", class_="search-result-story__headline")
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        link = title_tag['href']
                        news_items.append({
                            "symbol": query.upper(),
                            "title": title,
                            "url": link if link.startswith("http") else f"https://www.bloomberg.com{link}",
                            "timestamp": datetime.now(),
                            "source": "BloombergNews",
                            "content": title
                        })
                return news_items
        except Exception as e:
            logging.error(f"BloombergNews fetch error for {query}: {e}")
            return []

class CNBCNewsSource(DataSource):
    name = "CNBCNews"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.cnbctv18.com/search/?term={query}"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                news_items = []
                for div in soup.find_all("div", class_="search-story-card"):
                    title_tag = div.find("a", class_="search-story-card__headline")
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        link = title_tag['href']
                        news_items.append({
                            "symbol": query.upper(),
                            "title": title,
                            "url": link if link.startswith("http") else f"https://www.cnbctv18.com{link}",
                            "timestamp": datetime.now(),
                            "source": "CNBCNews",
                            "content": title
                        })
                return news_items
        except Exception as e:
            logging.error(f"CNBCNews fetch error for {query}: {e}")
            return []

class RBICircularsSource(DataSource):
    name = "RBICirculars"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.rbi.org.in/scripts/BS_PressReleaseDisplay.aspx?prid={query}"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                circulars = []
                for li in soup.find_all("li"):
                    text = li.get_text(strip=True)
                    link = li.find("a")['href'] if li.find("a") else None
                    if text and link:
                        circulars.append({
                            "symbol": query.upper(),
                            "title": text,
                            "url": link if link.startswith("http") else f"https://www.rbi.org.in{link}",
                            "timestamp": datetime.now(),
                            "source": "RBICirculars",
                            "content": text
                        })
                return circulars
        except Exception as e:
            logging.error(f"RBICirculars fetch error for {query}: {e}")
            return []

class SEBICircularsSource(DataSource):
    name = "SEBICirculars"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.sebi.gov.in/sebiweb/home/HomeAction.do?doListing=yes&sid=3&smid=0&ssid=0&keyword={query}"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                circulars = []
                for row in soup.find_all("tr", class_="alt-row"):
                    cols = row.find_all("td")
                    if len(cols) >= 3:
                        date = cols[0].get_text(strip=True)
                        headline = cols[1].get_text(strip=True)
                        link = cols[1].find("a")['href'] if cols[1].find("a") else None
                        circulars.append({
                            "symbol": query.upper(),
                            "headline": headline,
                            "date": date,
                            "url": link,
                            "timestamp": datetime.now(),
                            "source": "SEBICirculars",
                            "content": headline
                        })
                return circulars
        except Exception as e:
            logging.error(f"SEBICirculars fetch error for {query}: {e}")
            return []

class MintNewsSource(DataSource):
    name = "MintNews"
    async def fetch(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        url = f"https://www.livemint.com/search-results/{query}"
        try:
            async with session.get(url, headers=get_random_headers(), timeout=10) as resp:
                html = await resp.text()
                soup = bs4.BeautifulSoup(html, "html.parser")
                news_items = []
                for div in soup.find_all("div", class_="listingNewText"):
                    title_tag = div.find("a")
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        link = title_tag['href']
                        news_items.append({
                            "symbol": query.upper(),
                            "title": title,
                            "url": link if link.startswith("http") else f"https://www.livemint.com{link}",
                            "timestamp": datetime.now(),
                            "source": "MintNews",
                            "content": title
                        })
                return news_items
        except Exception as e:
            logging.error(f"MintNews fetch error for {query}: {e}")
            return []

# --- Semantic Search (sentence-transformers) ---
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SemanticSearchIndex:
    def __init__(self):
        self.docs = []
        self.embeddings = None
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def add(self, doc: Dict[str, Any]):
        self.docs.append(doc)
        # Lazy: rebuild embeddings each add (ok for demo, batch for prod)
        self.embeddings = self.model.encode([d['content'] for d in self.docs], convert_to_tensor=True)

    def search(self, query: str, top_k=5):
        if not self.docs or self.embeddings is None:
            return []
        q_emb = self.model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, self.embeddings, top_k=top_k)[0]
        return [self.docs[hit['corpus_id']] for hit in hits]

# --- Analytics/LLM Endpoint Logic (for API integration) ---
from transformers import pipeline

# Summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
def summarize_text(text):
    try:
        return summarizer(text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return ""

# Sentiment
sentiment_analyzer = pipeline("sentiment-analysis")
def analyze_sentiment(text):
    try:
        return sentiment_analyzer(text)[0]
    except Exception as e:
        logging.error(f"Sentiment error: {e}")
        return {"label": "ERROR", "score": 0}

# Q&A (over docs)
qa_pipeline = pipeline("question-answering")
def answer_question(question, context):
    try:
        return qa_pipeline(question=question, context=context)["answer"]
    except Exception as e:
        logging.error(f"Q&A error: {e}")
        return ""

# Trend/Anomaly Detection (simple volume spike demo)
def detect_trend(docs, field="volume"):
    try:
        values = [d.get(field, 0) for d in docs if d.get(field) is not None]
        if not values:
            return {"trend": "no data"}
        avg = np.mean(values)
        spikes = [v for v in values if v > 2 * avg]
        return {"avg": avg, "spikes": len(spikes), "trend": "spike" if spikes else "normal"}
    except Exception as e:
        logging.error(f"Trend detection error: {e}")
        return {"trend": "error"}

# --- 2. Data Normalization & Enrichment ---

def normalize(doc: Dict[str, Any], source: str) -> Dict[str, Any]:
    return {
        "symbol": doc.get("symbol"),
        "price": doc.get("price"),
        "change": doc.get("change"),
        "change_percent": doc.get("change_percent"),
        "volume": doc.get("volume"),
        "timestamp": doc.get("timestamp", datetime.now()),
        "source": source,
        "entities": extract_entities(doc.get("content", "")),
        "content": doc.get("content", "")
    }

def extract_entities(text: str) -> List[str]:
    if not text: return []
    return re.findall(r"[A-Z]{3,10}", text)

# --- 3. Indexing and Search (In-Memory/Pluggable) ---

import os
from elasticsearch import Elasticsearch, helpers
import redis
import json

class FinanceDocIndex:
    def __init__(self):
        self.backend = os.environ.get("FINANCE_INDEX_BACKEND", "memory") # "elasticsearch", "redis", or "memory"
        self.index_name = os.environ.get("ELASTIC_INDEX", "finance_docs")
        self.docs = []
        self.redis = None
        self.es = None
        if self.backend == "elasticsearch":
            self.es = Elasticsearch(os.environ.get("ELASTIC_HOST", "http://elasticsearch:9200"))
            # Create index if not exists
            if not self.es.indices.exists(index=self.index_name):
                self.es.indices.create(index=self.index_name)
        elif self.backend == "redis":
            self.redis = redis.Redis(host=os.environ.get("REDIS_HOST", "redis"), port=int(os.environ.get("REDIS_PORT", 6379)), decode_responses=True)

    def add(self, doc: Dict[str, Any]):
        if self.backend == "elasticsearch" and self.es:
            self.es.index(index=self.index_name, document=doc)
        elif self.backend == "redis" and self.redis:
            key = f"finance_doc:{doc.get('symbol','NA')}:{doc.get('timestamp',datetime.now()).isoformat()}"
            self.redis.set(key, json.dumps(doc))
        else:
            self.docs.append(doc)

    def full_text_search(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        if self.backend == "elasticsearch" and self.es:
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content", "symbol", "title"]
                    }
                }
            }
            res = self.es.search(index=self.index_name, body=body, size=20)
            return [hit["_source"] for hit in res["hits"]["hits"]]
        elif self.backend == "redis" and self.redis:
            # Simple scan all keys (production: use RediSearch)
            results = []
            for key in self.redis.scan_iter("finance_doc:*"):
                doc = json.loads(self.redis.get(key))
                if q in doc.get("content", "").lower() or q in doc.get("symbol", "").lower():
                    results.append(doc)
            return results
        else:
            return [d for d in self.docs if q in d.get("content", "").lower() or q in d.get("symbol", "").lower()]

    def entity_search(self, entity: str) -> List[Dict[str, Any]]:
        if self.backend == "elasticsearch" and self.es:
            body = {
                "query": {"match": {"entities": entity}}
            }
            res = self.es.search(index=self.index_name, body=body, size=20)
            return [hit["_source"] for hit in res["hits"]["hits"]]
        elif self.backend == "redis" and self.redis:
            results = []
            for key in self.redis.scan_iter("finance_doc:*"):
                doc = json.loads(self.redis.get(key))
                if entity in doc.get("entities", []):
                    results.append(doc)
            return results
        else:
            return [d for d in self.docs if entity in d.get("entities", [])]

    def symbol_search(self, symbol: str) -> List[Dict[str, Any]]:
        if self.backend == "elasticsearch" and self.es:
            body = {
                "query": {"term": {"symbol": symbol}}
            }
            res = self.es.search(index=self.index_name, body=body, size=20)
            return [hit["_source"] for hit in res["hits"]["hits"]]
        elif self.backend == "redis" and self.redis:
            results = []
            for key in self.redis.scan_iter(f"finance_doc:{symbol}:*"):
                doc = json.loads(self.redis.get(key))
                if doc.get("symbol") == symbol:
                    results.append(doc)
            return results
        else:
            return [d for d in self.docs if d.get("symbol") == symbol]

# --- 4. Ranking & Relevance ---

def rank_results(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    ranked = sorted(results, key=lambda d: (query.lower() in d.get("symbol", "").lower(), d.get("timestamp", datetime.min)), reverse=True)
    return ranked

# --- 5. Search Engine Orchestrator ---

class FinanceEngineFullstack:
    def __init__(self, sources: List[DataSource]):
        self.sources = sources
        self.index = FinanceDocIndex()
        self.semantic_index = SemanticSearchIndex()

    async def crawl_and_index(self, query: str):
        async with aiohttp.ClientSession() as session:
            tasks = [src.fetch(session, query) for src in self.sources]
            all_results = await asyncio.gather(*tasks)
            for src, docs in zip(self.sources, all_results):
                for doc in docs:
                    norm = normalize(doc, src.name)
                    self.index.add(norm)
                    self.semantic_index.add(norm)
            logging.info(f"Indexed {sum(len(docs) for docs in all_results)} docs for {query}")

    def search(self, query: str, mode: str = "full_text") -> List[Dict[str, Any]]:
        if mode == "full_text":
            results = self.index.full_text_search(query)
        elif mode == "entity":
            results = self.index.entity_search(query)
        elif mode == "symbol":
            results = self.index.symbol_search(query)
        elif mode == "semantic":
            results = self.semantic_index.search(query)
        else:
            results = []
        return rank_results(results, query)

# --- 6. Example Backend-Only Usage ---

async def main():
    sources = [NSESource(), MoneyControlSource(), YahooFinanceSource(), EconomicTimesNewsSource(), BSEFilingsSource(), ReutersNewsSource(), YahooFinanceNewsSource(), SEBIFilingsSource()]
    engine = FinanceEngineFullstack(sources)
    await engine.crawl_and_index("RELIANCE")
    await engine.crawl_and_index("TCS")
    print("\nFull Text Search for 'Reliance':")
    for r in engine.search("Reliance", mode="full_text"):
        print(r)
    print("\nEntity Search for 'RELIANCE':")
    for r in engine.search("RELIANCE", mode="entity"):
        print(r)
    print("\nSymbol Search for 'TCS':")
    for r in engine.search("TCS", mode="symbol"):
        print(r)
    print("\nSemantic Search for 'India's largest private sector company':")
    for r in engine.search("India's largest private sector company", mode="semantic"):
        print(r)

if __name__ == "__main__":
    asyncio.run(main())
