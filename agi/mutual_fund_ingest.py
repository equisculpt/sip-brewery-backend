import logging
import requests
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mutual_fund_ingest")

# MongoDB setup
mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['mf_data']
schemes_col = db['schemes']
nav_col = db['nav_history']
manager_col = db['fund_managers']
events_col = db['scheme_events']

# --- 1. Ingest All Mutual Fund Schemes (AMFI) ---
def fetch_all_schemes():
    url = 'https://www.amfiindia.com/spages/NAVAll.txt'
    logger.info('Fetching all mutual fund schemes and NAVs from AMFI...')
    try:
        resp = requests.get(url)
        lines = resp.text.splitlines()
        scheme_data = []
        for line in lines:
            parts = line.split(';')
            if len(parts) >= 6 and parts[0].isdigit():
                scheme_data.append({
                    'scheme_code': parts[0],
                    'isin_div_payout': parts[1],
                    'isin_div_reinvest': parts[2],
                    'scheme_name': parts[3],
                    'nav': parts[4],
                    'repurchase_price': parts[5] if len(parts) > 5 else None,
                    'sale_price': parts[6] if len(parts) > 6 else None,
                    'date': parts[-1],
                    'timestamp': datetime.utcnow()
                })
        if scheme_data:
            nav_col.insert_many(scheme_data)
            logger.info(f"Inserted {len(scheme_data)} NAV records.")
        return scheme_data
    except Exception as e:
        logger.error(f"Scheme/NAV fetch failed: {e}")
        return []

# --- 2. Ingest Fund Manager Data (Value Research) ---
def fetch_fund_managers():
    import re
    from bs4 import BeautifulSoup
    logger.info('Fetching fund manager data from Value Research Online...')
    base_url = 'https://www.valueresearchonline.com/funds/fundSelector/default.asp?advanced=1'
    try:
        # Example: Scrape a few sample schemes (for full, iterate through all scheme URLs)
        # In production, build a list of all scheme URLs from VRO or AMFI mapping
        sample_scheme_urls = [
            'https://www.valueresearchonline.com/funds/newsnapshot.asp?schemecode=100820',
            'https://www.valueresearchonline.com/funds/newsnapshot.asp?schemecode=102885',
        ]
        for url in sample_scheme_urls:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            # Find fund manager section (VRO HTML structure may change)
            mgr_section = soup.find(text=re.compile('Fund Manager'))
            if mgr_section:
                parent = mgr_section.find_parent('tr')
                if parent:
                    cells = parent.find_all('td')
                    if len(cells) > 1:
                        managers = cells[1].get_text(strip=True)
                        manager_col.insert_one({
                            'scheme_url': url,
                            'managers': managers,
                            'timestamp': datetime.utcnow()
                        })
                        logger.info(f"Inserted manager data for {url}")
    except Exception as e:
        logger.error(f"Fund manager fetch failed: {e}")

# --- 3. Ingest Market Index Data (NIFTY/SENSEX) ---
def fetch_index_data():
    import yfinance as yf
    logger.info('Fetching NIFTY and SENSEX historical data...')
    indices = {
        'NIFTY50': '^NSEI',
        'SENSEX': '^BSESN',
    }
    try:
        for name, ticker in indices.items():
            data = yf.download(ticker, period='max', interval='1d')
            if not data.empty:
                records = data.reset_index().to_dict(orient='records')
                for rec in records:
                    rec['index'] = name
                    rec['timestamp'] = datetime.utcnow()
                db['index_history'].insert_many(records)
                logger.info(f"Inserted {len(records)} records for {name}")
                # Trigger major event for NAV update
                from agi_microservice import trigger_major_event
                trigger_major_event('fund', meta={"index": name})
    except Exception as e:
        logger.error(f"Index data fetch failed: {e}")

# --- 4. Event Correlation Engine ---
def correlate_events():
    logger.info('Correlating scheme events (manager change, market moves)...')
    # Example: For each scheme, find manager change dates, correlate with NAV jumps/drops, market index moves
    # 1. Find all manager change events
    # 2. For each, fetch NAV history window (before/after)
    # 3. Fetch NIFTY/SENSEX data for same window
    # 4. Store event correlations in events_col
    # TODO: Implement event correlation logic
    pass

if __name__ == "__main__":
    fetch_all_schemes()
    fetch_fund_managers()
    correlate_events()
