#!/usr/bin/env python3
"""
Universal Indian Finance Crawler
Google-like crawler for ALL major Indian financial websites
NSE, BSE, MCX, NCDEX, SEBI, RBI, AMFI, CRISIL, and more
"""

import json
import logging
import requests
import time
import random
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin, urlparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class FinancialInstrument:
    """Universal financial instrument data structure"""
    name: str
    symbol: str
    isin: Optional[str] = None
    instrument_type: str = "EQUITY"  # EQUITY, MUTUAL_FUND, BOND, COMMODITY, DERIVATIVE
    exchange: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    market_cap: Optional[float] = None
    face_value: Optional[float] = None
    listing_date: Optional[str] = None
    status: str = "ACTIVE"
    website_source: str = ""
    data_confidence: float = 1.0
    last_updated: str = ""
    additional_data: Dict = None

class UniversalIndianFinanceCrawler:
    """Universal crawler for all Indian financial websites"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Financial websites to crawl
        self.financial_websites = {
            # Stock Exchanges
            "NSE": {
                "base_url": "https://www.nseindia.com",
                "endpoints": [
                    "/api/equity-stockIndices?index=NIFTY%2050",
                    "/api/equity-stockIndices?index=NIFTY%20NEXT%2050",
                    "/api/equity-stockIndices?index=NIFTY%20500",
                    "/content/equities/EQUITY_L.csv"
                ],
                "type": "STOCK_EXCHANGE"
            },
            "BSE": {
                "base_url": "https://www.bseindia.com",
                "endpoints": [
                    "/corporates/List_Scrips.html",
                    "/markets/equity/EQReports/StockPrcHistori.html",
                    "/static/markets/ActiveScriptsTraded.html"
                ],
                "type": "STOCK_EXCHANGE"
            },
            
            # Commodity Exchanges
            "MCX": {
                "base_url": "https://www.mcxindia.com",
                "endpoints": [
                    "/market-data/bhavcopy",
                    "/market-data/top-gainers-losers",
                    "/docs/default-source/market-data/commodity-wise-turnover"
                ],
                "type": "COMMODITY_EXCHANGE"
            },
            "NCDEX": {
                "base_url": "https://www.ncdex.com",
                "endpoints": [
                    "/market/live-market",
                    "/market/market-statistics",
                    "/downloads/bhavcopy"
                ],
                "type": "COMMODITY_EXCHANGE"
            },
            
            # Mutual Funds
            "AMFI": {
                "base_url": "https://www.amfiindia.com",
                "endpoints": [
                    "/spages/NAVAll.txt",
                    "/research-information/other-data/nav-history",
                    "/modules/MasterDisp.aspx"
                ],
                "type": "MUTUAL_FUND"
            },
            
            # Regulatory Bodies
            "SEBI": {
                "base_url": "https://www.sebi.gov.in",
                "endpoints": [
                    "/sebiweb/other/OtherAction.do?doRecognisedFpi=yes",
                    "/sebiweb/other/OtherAction.do?doIssuerCompany=yes",
                    "/filings/exchange-filings"
                ],
                "type": "REGULATORY"
            },
            "RBI": {
                "base_url": "https://www.rbi.org.in",
                "endpoints": [
                    "/Scripts/Data_NBFC.aspx",
                    "/Scripts/BS_PressReleaseDisplay.aspx",
                    "/Scripts/Data_Handbook.aspx"
                ],
                "type": "REGULATORY"
            },
            
            # Rating Agencies
            "CRISIL": {
                "base_url": "https://www.crisil.com",
                "endpoints": [
                    "/en/home/our-businesses/ratings/credit-ratings/listed-entity-ratings.html",
                    "/en/home/our-businesses/research/equity-research.html"
                ],
                "type": "RATING_AGENCY"
            },
            "ICRA": {
                "base_url": "https://www.icra.in",
                "endpoints": [
                    "/Rating/ShowRatingList.aspx",
                    "/Research/SectoralUpdate.aspx"
                ],
                "type": "RATING_AGENCY"
            },
            
            # Financial Data Providers
            "BSE_STAR_MF": {
                "base_url": "https://www.bsestarmf.in",
                "endpoints": [
                    "/RptNavMaster.aspx",
                    "/RptSchemeMaster.aspx",
                    "/RptAMCMaster.aspx"
                ],
                "type": "MUTUAL_FUND_PLATFORM"
            },
            "CDSL": {
                "base_url": "https://www.cdslindia.com",
                "endpoints": [
                    "/Footer/grievances.html",
                    "/ipo/ipo.html",
                    "/investors/open-demat-account.html"
                ],
                "type": "DEPOSITORY"
            },
            "NSDL": {
                "base_url": "https://www.nsdl.co.in",
                "endpoints": [
                    "/beneficial-owner-services/find-your-pan.php",
                    "/corporates/corporate-actions.php"
                ],
                "type": "DEPOSITORY"
            },
            
            # Financial News & Data
            "MONEY_CONTROL": {
                "base_url": "https://www.moneycontrol.com",
                "endpoints": [
                    "/stocks/marketstats/indexcomp.php?optex=NSE&opttopic=indexcomp&index=9",
                    "/mutual-funds/nav/",
                    "/commodity/"
                ],
                "type": "FINANCIAL_DATA"
            },
            "ECONOMIC_TIMES": {
                "base_url": "https://economictimes.indiatimes.com",
                "endpoints": [
                    "/markets/stocks/stock-screener",
                    "/mf/",
                    "/markets/commodities"
                ],
                "type": "FINANCIAL_NEWS"
            },
            
            # Insurance
            "IRDAI": {
                "base_url": "https://www.irdai.gov.in",
                "endpoints": [
                    "/ADMINCMS/cms/NormalData_Layout.aspx?page=PageNo234",
                    "/ADMINCMS/cms/NormalData_Layout.aspx?page=PageNo3130"
                ],
                "type": "INSURANCE_REGULATOR"
            }
        }
        
        self.crawled_data = {
            "equities": [],
            "mutual_funds": [],
            "commodities": [],
            "bonds": [],
            "derivatives": [],
            "insurance": [],
            "regulatory_data": []
        }
        
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0'
        ]
    
    def rotate_user_agent(self):
        """Rotate user agent to avoid detection"""
        self.session.headers.update({
            'User-Agent': random.choice(self.user_agents)
        })
    
    def smart_delay(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """Smart delay to avoid rate limiting"""
        delay = random.uniform(min_delay, max_delay)
        time.sleep(delay)
    
    def safe_request(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        """Make safe HTTP request with error handling"""
        try:
            self.rotate_user_agent()
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None
    
    def parse_nse_data(self) -> List[FinancialInstrument]:
        """Parse NSE data"""
        logger.info("🔍 Crawling NSE data...")
        instruments = []
        
        try:
            # NSE Equity List
            csv_url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
            response = self.safe_request(csv_url)
            
            if response:
                # Parse CSV data
                from io import StringIO
                df = pd.read_csv(StringIO(response.text))
                
                for _, row in df.iterrows():
                    instrument = FinancialInstrument(
                        name=row.get('NAME OF COMPANY', ''),
                        symbol=row.get('SYMBOL', ''),
                        isin=row.get('ISIN NUMBER', ''),
                        instrument_type="EQUITY",
                        exchange="NSE",
                        face_value=row.get('FACE VALUE', None),
                        listing_date=row.get('DATE OF LISTING', ''),
                        website_source="NSE_OFFICIAL",
                        data_confidence=0.95,
                        last_updated=datetime.now().isoformat()
                    )
                    instruments.append(instrument)
                
                logger.info(f"✅ NSE: Found {len(instruments)} equities")
        
        except Exception as e:
            logger.error(f"❌ NSE crawling failed: {e}")
        
        return instruments
    
    def parse_bse_data(self) -> List[FinancialInstrument]:
        """Parse BSE data"""
        logger.info("🔍 Crawling BSE data...")
        instruments = []
        
        try:
            # BSE Scrip List
            bse_url = "https://www.bseindia.com/corporates/List_Scrips.html"
            response = self.safe_request(bse_url)
            
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find tables with company data
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows[1:]:  # Skip header
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            instrument = FinancialInstrument(
                                name=cells[1].get_text(strip=True) if len(cells) > 1 else '',
                                symbol=cells[0].get_text(strip=True) if len(cells) > 0 else '',
                                instrument_type="EQUITY",
                                exchange="BSE",
                                sector=cells[2].get_text(strip=True) if len(cells) > 2 else 'Unknown',
                                website_source="BSE_OFFICIAL",
                                data_confidence=0.90,
                                last_updated=datetime.now().isoformat()
                            )
                            instruments.append(instrument)
                
                logger.info(f"✅ BSE: Found {len(instruments)} equities")
        
        except Exception as e:
            logger.error(f"❌ BSE crawling failed: {e}")
        
        return instruments
    
    def parse_mcx_data(self) -> List[FinancialInstrument]:
        """Parse MCX commodity data"""
        logger.info("🔍 Crawling MCX commodity data...")
        instruments = []
        
        try:
            mcx_url = "https://www.mcxindia.com/market-data/bhavcopy"
            response = self.safe_request(mcx_url)
            
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for commodity tables
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows[1:]:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            instrument = FinancialInstrument(
                                name=cells[0].get_text(strip=True) if len(cells) > 0 else '',
                                symbol=cells[1].get_text(strip=True) if len(cells) > 1 else '',
                                instrument_type="COMMODITY",
                                exchange="MCX",
                                website_source="MCX_OFFICIAL",
                                data_confidence=0.85,
                                last_updated=datetime.now().isoformat()
                            )
                            instruments.append(instrument)
                
                logger.info(f"✅ MCX: Found {len(instruments)} commodities")
        
        except Exception as e:
            logger.error(f"❌ MCX crawling failed: {e}")
        
        return instruments
    
    def parse_amfi_data(self) -> List[FinancialInstrument]:
        """Parse AMFI mutual fund data"""
        logger.info("🔍 Crawling AMFI mutual fund data...")
        instruments = []
        
        try:
            amfi_url = "https://www.amfiindia.com/spages/NAVAll.txt"
            response = self.safe_request(amfi_url)
            
            if response:
                lines = response.text.split('\n')
                current_amc = ""
                
                for line in lines:
                    line = line.strip()
                    if line and not line[0].isdigit():
                        # AMC name line
                        current_amc = line
                    elif line and ';' in line:
                        # Fund data line
                        parts = line.split(';')
                        if len(parts) >= 6:
                            instrument = FinancialInstrument(
                                name=parts[3].strip(),
                                symbol=parts[0].strip(),
                                isin=parts[1].strip(),
                                instrument_type="MUTUAL_FUND",
                                exchange="AMFI",
                                sector="Mutual Fund",
                                industry=current_amc,
                                website_source="AMFI_OFFICIAL",
                                data_confidence=0.95,
                                last_updated=datetime.now().isoformat(),
                                additional_data={
                                    "nav": parts[4].strip(),
                                    "nav_date": parts[6].strip() if len(parts) > 6 else ""
                                }
                            )
                            instruments.append(instrument)
                
                logger.info(f"✅ AMFI: Found {len(instruments)} mutual funds")
        
        except Exception as e:
            logger.error(f"❌ AMFI crawling failed: {e}")
        
        return instruments
    
    def parse_sebi_data(self) -> List[FinancialInstrument]:
        """Parse SEBI regulatory data"""
        logger.info("🔍 Crawling SEBI regulatory data...")
        instruments = []
        
        try:
            # SEBI FPI data
            sebi_url = "https://www.sebi.gov.in/sebiweb/other/OtherAction.do?doRecognisedFpi=yes"
            response = self.safe_request(sebi_url)
            
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse FPI tables
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    for row in rows[1:]:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            instrument = FinancialInstrument(
                                name=cells[0].get_text(strip=True) if len(cells) > 0 else '',
                                symbol="",
                                instrument_type="FPI",
                                exchange="SEBI",
                                sector="Foreign Portfolio Investment",
                                website_source="SEBI_OFFICIAL",
                                data_confidence=0.90,
                                last_updated=datetime.now().isoformat()
                            )
                            instruments.append(instrument)
                
                logger.info(f"✅ SEBI: Found {len(instruments)} FPI entities")
        
        except Exception as e:
            logger.error(f"❌ SEBI crawling failed: {e}")
        
        return instruments
    
    def crawl_financial_website(self, website_name: str, website_config: Dict) -> List[FinancialInstrument]:
        """Crawl a specific financial website"""
        logger.info(f"🕷️ Crawling {website_name} ({website_config['type']})...")
        
        instruments = []
        base_url = website_config['base_url']
        
        for endpoint in website_config['endpoints']:
            try:
                full_url = urljoin(base_url, endpoint)
                response = self.safe_request(full_url)
                
                if response:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Generic table parsing
                    tables = soup.find_all('table')
                    for table in tables:
                        rows = table.find_all('tr')
                        if len(rows) > 1:  # Has header and data
                            for row in rows[1:10]:  # Limit to first 10 rows per table
                                cells = row.find_all(['td', 'th'])
                                if len(cells) >= 2:
                                    instrument = FinancialInstrument(
                                        name=cells[0].get_text(strip=True)[:100],  # Limit length
                                        symbol=cells[1].get_text(strip=True)[:20] if len(cells) > 1 else '',
                                        instrument_type=self.get_instrument_type(website_config['type']),
                                        exchange=website_name,
                                        sector=website_config['type'],
                                        website_source=f"{website_name}_CRAWLED",
                                        data_confidence=0.70,  # Lower confidence for generic crawling
                                        last_updated=datetime.now().isoformat()
                                    )
                                    instruments.append(instrument)
                
                self.smart_delay(1, 2)  # Delay between requests
                
            except Exception as e:
                logger.warning(f"Failed to crawl {full_url}: {e}")
                continue
        
        logger.info(f"✅ {website_name}: Found {len(instruments)} instruments")
        return instruments
    
    def get_instrument_type(self, website_type: str) -> str:
        """Map website type to instrument type"""
        mapping = {
            "STOCK_EXCHANGE": "EQUITY",
            "COMMODITY_EXCHANGE": "COMMODITY",
            "MUTUAL_FUND": "MUTUAL_FUND",
            "MUTUAL_FUND_PLATFORM": "MUTUAL_FUND",
            "REGULATORY": "REGULATORY",
            "RATING_AGENCY": "RATING",
            "DEPOSITORY": "DEPOSITORY",
            "FINANCIAL_DATA": "DATA",
            "FINANCIAL_NEWS": "NEWS",
            "INSURANCE_REGULATOR": "INSURANCE"
        }
        return mapping.get(website_type, "UNKNOWN")
    
    def crawl_all_websites(self) -> Dict[str, List[FinancialInstrument]]:
        """Crawl all configured financial websites"""
        logger.info("🚀 Starting Universal Indian Finance Crawling...")
        logger.info(f"   Target Websites: {len(self.financial_websites)}")
        
        all_instruments = {}
        
        # Priority crawling - most important first
        priority_websites = ["NSE", "BSE", "AMFI", "MCX", "SEBI"]
        
        # Crawl priority websites with specialized parsers
        for website in priority_websites:
            if website in self.financial_websites:
                try:
                    if website == "NSE":
                        instruments = self.parse_nse_data()
                    elif website == "BSE":
                        instruments = self.parse_bse_data()
                    elif website == "MCX":
                        instruments = self.parse_mcx_data()
                    elif website == "AMFI":
                        instruments = self.parse_amfi_data()
                    elif website == "SEBI":
                        instruments = self.parse_sebi_data()
                    else:
                        instruments = []
                    
                    all_instruments[website] = instruments
                    self.smart_delay(2, 4)  # Longer delay between major sites
                    
                except Exception as e:
                    logger.error(f"❌ Failed to crawl {website}: {e}")
                    all_instruments[website] = []
        
        # Crawl remaining websites with generic parser
        remaining_websites = [w for w in self.financial_websites.keys() if w not in priority_websites]
        
        for website_name in remaining_websites:
            try:
                website_config = self.financial_websites[website_name]
                instruments = self.crawl_financial_website(website_name, website_config)
                all_instruments[website_name] = instruments
                self.smart_delay(1, 3)
                
            except Exception as e:
                logger.error(f"❌ Failed to crawl {website_name}: {e}")
                all_instruments[website_name] = []
        
        return all_instruments
    
    def categorize_instruments(self, all_instruments: Dict[str, List[FinancialInstrument]]):
        """Categorize instruments by type"""
        for website, instruments in all_instruments.items():
            for instrument in instruments:
                instrument_type = instrument.instrument_type
                
                if instrument_type == "EQUITY":
                    self.crawled_data["equities"].append(instrument)
                elif instrument_type == "MUTUAL_FUND":
                    self.crawled_data["mutual_funds"].append(instrument)
                elif instrument_type == "COMMODITY":
                    self.crawled_data["commodities"].append(instrument)
                elif instrument_type in ["BOND", "DEBT"]:
                    self.crawled_data["bonds"].append(instrument)
                elif instrument_type in ["DERIVATIVE", "FUTURES", "OPTIONS"]:
                    self.crawled_data["derivatives"].append(instrument)
                elif instrument_type == "INSURANCE":
                    self.crawled_data["insurance"].append(instrument)
                else:
                    self.crawled_data["regulatory_data"].append(instrument)
    
    def deduplicate_instruments(self):
        """Remove duplicate instruments across categories"""
        logger.info("🔄 Deduplicating instruments...")
        
        for category, instruments in self.crawled_data.items():
            seen_instruments = set()
            unique_instruments = []
            
            for instrument in instruments:
                # Create unique key
                key = (instrument.name.lower().strip(), instrument.symbol.lower().strip(), instrument.isin)
                
                if key not in seen_instruments:
                    seen_instruments.add(key)
                    unique_instruments.append(instrument)
            
            self.crawled_data[category] = unique_instruments
            logger.info(f"   {category}: {len(instruments)} → {len(unique_instruments)} (removed {len(instruments) - len(unique_instruments)} duplicates)")
    
    def save_crawled_data(self) -> str:
        """Save crawled data to JSON file"""
        output_file = "market_data/universal_indian_finance_data.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert to dictionaries
        data_dict = {}
        total_instruments = 0
        
        for category, instruments in self.crawled_data.items():
            data_dict[category] = []
            for instrument in instruments:
                instrument_dict = asdict(instrument)
                if instrument_dict['additional_data'] is None:
                    instrument_dict['additional_data'] = {}
                data_dict[category].append(instrument_dict)
            
            total_instruments += len(instruments)
        
        # Add metadata
        metadata = {
            "total_instruments": total_instruments,
            "crawling_method": "UNIVERSAL_INDIAN_FINANCE_CRAWLER",
            "websites_crawled": list(self.financial_websites.keys()),
            "categories": {
                "equities": len(self.crawled_data["equities"]),
                "mutual_funds": len(self.crawled_data["mutual_funds"]),
                "commodities": len(self.crawled_data["commodities"]),
                "bonds": len(self.crawled_data["bonds"]),
                "derivatives": len(self.crawled_data["derivatives"]),
                "insurance": len(self.crawled_data["insurance"]),
                "regulatory_data": len(self.crawled_data["regulatory_data"])
            },
            "last_updated": datetime.now().isoformat(),
            "crawler_version": "1.0.0"
        }
        
        final_data = {
            "metadata": metadata,
            "data": data_dict
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved universal finance data to {output_file}")
        return output_file
    
    def run_universal_crawling(self) -> str:
        """Run complete universal crawling"""
        logger.info("🚀 Starting Universal Indian Finance Crawling...")
        logger.info(f"   Target: ALL major Indian financial websites")
        logger.info(f"   Websites: {len(self.financial_websites)}")
        
        # Crawl all websites
        all_instruments = self.crawl_all_websites()
        
        # Categorize instruments
        self.categorize_instruments(all_instruments)
        
        # Deduplicate
        self.deduplicate_instruments()
        
        # Save data
        output_file = self.save_crawled_data()
        
        # Display results
        total_instruments = sum(len(instruments) for instruments in self.crawled_data.values())
        
        logger.info(f"\n📊 Universal Crawling Results:")
        logger.info(f"   Total Instruments: {total_instruments:,}")
        logger.info(f"   Equities: {len(self.crawled_data['equities']):,}")
        logger.info(f"   Mutual Funds: {len(self.crawled_data['mutual_funds']):,}")
        logger.info(f"   Commodities: {len(self.crawled_data['commodities']):,}")
        logger.info(f"   Bonds: {len(self.crawled_data['bonds']):,}")
        logger.info(f"   Derivatives: {len(self.crawled_data['derivatives']):,}")
        logger.info(f"   Insurance: {len(self.crawled_data['insurance']):,}")
        logger.info(f"   Regulatory Data: {len(self.crawled_data['regulatory_data']):,}")
        
        logger.info(f"\n🌐 Websites Crawled:")
        for website, config in self.financial_websites.items():
            count = len(all_instruments.get(website, []))
            logger.info(f"   {website} ({config['type']}): {count:,} instruments")
        
        logger.info(f"\n🎉 UNIVERSAL INDIAN FINANCE CRAWLING COMPLETE!")
        logger.info(f"   ✅ Comprehensive coverage across ALL financial websites")
        logger.info(f"   ✅ {total_instruments:,} total financial instruments")
        logger.info(f"   ✅ Multi-category support (Equities, MF, Commodities, etc.)")
        logger.info(f"   📁 Data saved to: {output_file}")
        
        return output_file

def main():
    """Main function"""
    crawler = UniversalIndianFinanceCrawler()
    crawler.run_universal_crawling()

if __name__ == "__main__":
    main()
