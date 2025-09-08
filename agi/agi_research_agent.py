# AGI Research Agent: Ingests news, research, and market events for mutual funds.
# Requirements: requests, beautifulsoup4, pymongo, transformers, PyPDF2

import logging
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import json
from transformers import pipeline
from PyPDF2 import PdfReader
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agi_research_agent")

# MongoDB for storing research and explanations
mongo_client = MongoClient('mongodb://localhost:27017/')
research_db = mongo_client['agi_research']
explain_collection = research_db['explanations']
news_collection = research_db['news']
macro_collection = research_db['macro']
metadata_log = mongo_client['agi_cache']['metadata_audit_log']

# --- LLM-based Extraction Utility ---
def llm_extract_metadata(text):
    # Use transformers pipeline or OpenAI API for robust extraction
    # Example with transformers pipeline (replace with your LLM as needed)
    nlp = pipeline("text2text-generation", model="google/flan-t5-large")
    prompt = (
        "Extract the following metadata from the text: scheme style (equity/debt/hybrid/sector), liquidity percent, AMC name, ESG score (0-1). "
        "Return as JSON.\nText: " + text[:2000]
    )
    result = nlp(prompt, max_new_tokens=128)[0]['generated_text']
    try:
        meta = json.loads(result)
        return meta
    except Exception:
        return {}

# --- PDF Extraction Utility ---
def extract_text_from_pdf(url):
    r = requests.get(url)
    reader = PdfReader(BytesIO(r.content))
    text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    return text

# --- Trusted Source Scrapers ---
def scrape_amfi_metadata(scheme_code):
    # Example: AMFI API or web scrape
    url = f"https://www.amfiindia.com/spages/NAVAll.txt"
    r = requests.get(url)
    text = r.text
    # ...parse for scheme_code, AMC, style...
    # Placeholder: return {}
    return {}

def scrape_morningstar_metadata(scheme_code):
    url = f"https://www.morningstar.in/mutualfunds/f0gbr06y9q/{scheme_code}.html"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    # ...parse for style, AMC, ESG, liquidity...
    # Placeholder: return {}
    return {}

def scrape_etmoney_metadata(scheme_code):
    url = f"https://www.etmoney.com/mutual-funds/{scheme_code}"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    # ...parse for style, AMC, liquidity, ESG...
    # Placeholder: return {}
    return {}

# --- Main Metadata Extraction ---
def get_fund_metadata(scheme_code):
    sources = []
    # 1. Try AGI LLM-based extraction from AMC/factsheet/website
    try:
        # Try PDF factsheet (if URL known)
        # pdf_url = ...
        # text = extract_text_from_pdf(pdf_url)
        # meta = llm_extract_metadata(text)
        # if meta: sources.append(('LLM_PDF', meta))
        pass
    except Exception as e:
        logger.warning(f"PDF LLM extraction failed: {e}")
    # 2. Try trusted sources
    for fn, label in [
        (scrape_amfi_metadata, 'AMFI'),
        (scrape_morningstar_metadata, 'Morningstar'),
        (scrape_etmoney_metadata, 'ETMoney')
    ]:
        try:
            meta = fn(scheme_code)
            if meta: sources.append((label, meta))
        except Exception as e:
            logger.warning(f"{label} scrape failed: {e}")
    # 3. Fallback: ValueResearch scraping
    try:
        url = f"https://www.valueresearchonline.com/funds/newsnapshot.asp?schemecode={scheme_code}"
        r = requests.get(url)
        soup = BeautifulSoup(r.text, "html.parser")
        # ...parse for style, AMC, liquidity, ESG...
        # Placeholder: return dummy values
        meta = {'style': 'equity', 'liquidity': 0.05, 'amc': 'HDFC', 'esg_score': 0.7}
        sources.append(('ValueResearch', meta))
    except Exception as e:
        logger.warning(f"ValueResearch scrape failed: {e}")
    # 4. Merge/choose best
    final = {}
    for label, meta in sources:
        for k, v in meta.items():
            if k not in final or (v and v != 'unknown'):
                final[k] = v
    # 5. Audit log
    metadata_log.insert_one({
        'scheme_code': scheme_code,
        'sources': sources,
        'final_metadata': final,
        'timestamp': datetime.utcnow()
    })
    return final if final else {'style': 'other', 'liquidity': 0, 'amc': 'unknown', 'esg_score': 0.5}

# Example: Ingest fund news from trusted sources

def ingest_fund_news():
    # ... existing logic ...
    pass

# Real integration with NewsAPI.org (replace API_KEY with your key)
def fetch_market_news():
    # Real integration with NewsAPI.org (replace API_KEY with your key)
    logger.info("Fetching market news...")
    NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"
    url = f"https://newsapi.org/v2/top-headlines?category=business&language=en&apiKey={NEWSAPI_KEY}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            data = resp.json()
            news = [{
                "title": article["title"],
                "summary": article.get("description", ""),
                "url": article.get("url", ""),
                "timestamp": datetime.utcnow()
            } for article in data.get("articles", [])]
            if news:
                news_collection.insert_many(news)
                # Trigger major event for news ingestion
                from agi_microservice import trigger_major_event
                trigger_major_event('news')
            logger.info(f"Stored {len(news)} news items.")
        else:
            logger.error(f"NewsAPI fetch failed: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"News fetch failed: {e}")


def fetch_macro_data():
    # Real integration with FRED (Federal Reserve Economic Data)
    logger.info("Fetching macroeconomic data...")
    FRED_API_KEY = "YOUR_FRED_API_KEY"
    indicators = ["GDP", "UNRATE", "CPIAUCSL"]  # GDP, unemployment, inflation
    try:
        macro = []
        for ind in indicators:
            url = f"https://api.stlouisfed.org/fred/series/observations?series_id={ind}&api_key={FRED_API_KEY}&file_type=json"
            resp = requests.get(url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("observations"):
                    latest = data["observations"][-1]
                    macro.append({
                        "indicator": ind,
                        "value": latest["value"],
                        "date": latest["date"],
                        "timestamp": datetime.utcnow()
                    })
        if macro:
            macro_collection.insert_many(macro)
        logger.info(f"Stored {len(macro)} macro items.")
    except Exception as e:
        logger.error(f"Macro fetch failed: {e}")


def explain_recommendation(user_id, prompt, model, output):
    # Example: Generate an explanation for a model output
    explanation = {
        "user_id": user_id,
        "prompt": prompt,
        "model": model,
        "output": output,
        "rationale": f"This recommendation is based on recent market trends and your portfolio preferences.",
        "timestamp": datetime.utcnow()
    }
    try:
        explain_collection.insert_one(explanation)
        logger.info(f"Explanation stored for user {user_id}.")
    except Exception as e:
        logger.error(f"Explanation log failed: {e}")
    return explanation


def run_daily_research():
    fetch_market_news()
    fetch_macro_data()
    logger.info("Daily research agent run complete.")

if __name__ == "__main__":
    run_daily_research()
