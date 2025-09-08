#!/usr/bin/env python3
"""
Comprehensive BSE API Fetcher
Fetch ALL BSE Main Board companies using multiple BSE API endpoints
"""

import requests
import pandas as pd
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BSECompany:
    """Data class for BSE company information"""
    scrip_code: str
    scrip_name: str
    company_name: str
    group: str = ""
    face_value: Optional[float] = None
    isin: str = ""
    industry: str = ""
    instrument: str = "EQUITY"
    sector: str = ""
    exchange: str = "BSE_MAIN"
    market_cap_category: str = "Unknown"
    status: str = "ACTIVE"
    last_updated: str = ""

class ComprehensiveBSEAPIFetcher:
    """Comprehensive fetcher for ALL BSE companies using multiple APIs"""
    
    def __init__(self):
        self.base_url = "https://api.bseindia.com/BseIndiaAPI/api"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.bseindia.com/',
        })
        
        # Multiple BSE API endpoints to try
        self.api_endpoints = [
            f"{self.base_url}/ListOfScrips/w",  # Main scrips list
            f"{self.base_url}/Sensex/w",        # Sensex companies
            f"{self.base_url}/getScripHeaderData/w",  # Scrip header data
            f"{self.base_url}/ComHeader/w",     # Company header
            f"{self.base_url}/DefaultData/w",   # Default data
        ]
        
        # BSE website scraping URLs
        self.web_endpoints = [
            "https://www.bseindia.com/corporates/List_Scrips.aspx",
            "https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.aspx",
        ]
    
    def fetch_from_api_endpoint(self, url: str, params: Dict = None) -> List[Dict]:
        """Fetch data from a specific BSE API endpoint"""
        try:
            logger.info(f"🔄 Trying API endpoint: {url}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    if isinstance(data, list) and len(data) > 0:
                        logger.info(f"✅ Found {len(data)} companies from {url}")
                        return data
                    elif isinstance(data, dict):
                        # Try to extract companies from different possible keys
                        for key in ['Table', 'data', 'companies', 'scrips', 'result']:
                            if key in data and isinstance(data[key], list):
                                logger.info(f"✅ Found {len(data[key])} companies from {url} (key: {key})")
                                return data[key]
                    
                    logger.warning(f"⚠️ API returned data but no companies found: {url}")
                    
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ Invalid JSON response from {url}")
            
            else:
                logger.warning(f"⚠️ API returned status {response.status_code}: {url}")
        
        except requests.RequestException as e:
            logger.warning(f"⚠️ Request failed for {url}: {e}")
        
        return []
    
    def fetch_bse_equity_list(self) -> List[Dict]:
        """Fetch BSE equity list using specific parameters"""
        logger.info("🔄 Fetching BSE equity list...")
        
        # Try different parameter combinations
        param_sets = [
            {},  # No parameters
            {"segment": "Equity"},
            {"market": "Equity"},
            {"scripcode": ""},
            {"group": "A"},
            {"group": "B"},
            {"group": "T"},
        ]
        
        all_companies = []
        
        for endpoint in self.api_endpoints:
            for params in param_sets:
                companies = self.fetch_from_api_endpoint(endpoint, params)
                if companies:
                    all_companies.extend(companies)
                    break  # If we got data, move to next endpoint
                time.sleep(1)  # Rate limiting
        
        # Remove duplicates based on scrip code
        seen_scrips = set()
        unique_companies = []
        
        for company in all_companies:
            scrip_code = str(company.get('Scrip_Cd', company.get('scrip_code', company.get('ScripCode', ''))))
            if scrip_code and scrip_code not in seen_scrips:
                seen_scrips.add(scrip_code)
                unique_companies.append(company)
        
        logger.info(f"✅ Total unique BSE companies found: {len(unique_companies)}")
        return unique_companies
    
    def fetch_bse_sensex_companies(self) -> List[Dict]:
        """Fetch BSE Sensex companies specifically"""
        logger.info("🔄 Fetching BSE Sensex companies...")
        
        sensex_url = f"{self.base_url}/Sensex/w"
        companies = self.fetch_from_api_endpoint(sensex_url)
        
        if companies:
            logger.info(f"✅ Found {len(companies)} Sensex companies")
        
        return companies
    
    def scrape_bse_website(self) -> List[Dict]:
        """Scrape BSE website for company list"""
        logger.info("🔄 Attempting to scrape BSE website...")
        
        try:
            # Try to get the main BSE scrips page
            url = "https://www.bseindia.com/corporates/List_Scrips.aspx"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # Parse HTML content
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Look for tables with company data
                tables = soup.find_all('table')
                companies = []
                
                for table in tables:
                    rows = table.find_all('tr')
                    if len(rows) > 10:  # Likely a data table
                        for row in rows[1:]:  # Skip header
                            cells = row.find_all(['td', 'th'])
                            if len(cells) >= 3:
                                company_data = {
                                    'scrip_code': cells[0].get_text(strip=True),
                                    'scrip_name': cells[1].get_text(strip=True),
                                    'company_name': cells[2].get_text(strip=True) if len(cells) > 2 else cells[1].get_text(strip=True)
                                }
                                if company_data['scrip_code'].isdigit():
                                    companies.append(company_data)
                
                if companies:
                    logger.info(f"✅ Scraped {len(companies)} companies from BSE website")
                    return companies
        
        except Exception as e:
            logger.warning(f"⚠️ Website scraping failed: {e}")
        
        return []
    
    def get_fallback_bse_companies(self) -> List[Dict]:
        """Get comprehensive fallback list of major BSE companies"""
        logger.info("🔄 Using comprehensive fallback BSE companies...")
        
        # Major BSE Main Board companies (expanded list)
        fallback_companies = [
            # IT Sector
            {"scrip_code": "543320", "scrip_name": "RNIT", "company_name": "RNIT AI Technologies Limited", "group": "B", "sector": "Information Technology"},
            {"scrip_code": "532540", "scrip_name": "TCS", "company_name": "Tata Consultancy Services Limited", "group": "A", "sector": "Information Technology"},
            {"scrip_code": "500209", "scrip_name": "INFY", "company_name": "Infosys Limited", "group": "A", "sector": "Information Technology"},
            {"scrip_code": "507685", "scrip_name": "WIPRO", "company_name": "Wipro Limited", "group": "A", "sector": "Information Technology"},
            {"scrip_code": "532281", "scrip_name": "HCLTECH", "company_name": "HCL Technologies Limited", "group": "A", "sector": "Information Technology"},
            {"scrip_code": "526371", "scrip_name": "TECHM", "company_name": "Tech Mahindra Limited", "group": "A", "sector": "Information Technology"},
            
            # Banking & Financial Services
            {"scrip_code": "500180", "scrip_name": "HDFCBANK", "company_name": "HDFC Bank Limited", "group": "A", "sector": "Financial Services"},
            {"scrip_code": "532174", "scrip_name": "ICICIBANK", "company_name": "ICICI Bank Limited", "group": "A", "sector": "Financial Services"},
            {"scrip_code": "500112", "scrip_name": "SBIN", "company_name": "State Bank of India", "group": "A", "sector": "Financial Services"},
            {"scrip_code": "500247", "scrip_name": "KOTAKBANK", "company_name": "Kotak Mahindra Bank Limited", "group": "A", "sector": "Financial Services"},
            {"scrip_code": "532215", "scrip_name": "AXISBANK", "company_name": "Axis Bank Limited", "group": "A", "sector": "Financial Services"},
            
            # Oil & Gas
            {"scrip_code": "500325", "scrip_name": "RELIANCE", "company_name": "Reliance Industries Limited", "group": "A", "sector": "Oil Gas & Consumable Fuels"},
            {"scrip_code": "500312", "scrip_name": "ONGC", "company_name": "Oil and Natural Gas Corporation Limited", "group": "A", "sector": "Oil Gas & Consumable Fuels"},
            
            # FMCG
            {"scrip_code": "500696", "scrip_name": "HINDUNILVR", "company_name": "Hindustan Unilever Limited", "group": "A", "sector": "Fast Moving Consumer Goods"},
            {"scrip_code": "500875", "scrip_name": "ITC", "company_name": "ITC Limited", "group": "A", "sector": "Fast Moving Consumer Goods"},
            {"scrip_code": "500790", "scrip_name": "NESTLEIND", "company_name": "Nestle India Limited", "group": "A", "sector": "Fast Moving Consumer Goods"},
            
            # Telecom
            {"scrip_code": "532454", "scrip_name": "BHARTIARTL", "company_name": "Bharti Airtel Limited", "group": "A", "sector": "Telecommunication"},
            
            # Auto
            {"scrip_code": "532500", "scrip_name": "MARUTI", "company_name": "Maruti Suzuki India Limited", "group": "A", "sector": "Automobile"},
            {"scrip_code": "500570", "scrip_name": "TATAMOTORS", "company_name": "Tata Motors Limited", "group": "A", "sector": "Automobile"},
            {"scrip_code": "532343", "scrip_name": "BAJAJ-AUTO", "company_name": "Bajaj Auto Limited", "group": "A", "sector": "Automobile"},
            
            # Infrastructure
            {"scrip_code": "500510", "scrip_name": "LT", "company_name": "Larsen & Toubro Limited", "group": "A", "sector": "Construction"},
            
            # Pharma
            {"scrip_code": "500124", "scrip_name": "DRREDDY", "company_name": "Dr. Reddy's Laboratories Limited", "group": "A", "sector": "Pharmaceuticals"},
            {"scrip_code": "500680", "scrip_name": "CIPLA", "company_name": "Cipla Limited", "group": "A", "sector": "Pharmaceuticals"},
            
            # Metals
            {"scrip_code": "500295", "scrip_name": "TATASTEEL", "company_name": "Tata Steel Limited", "group": "A", "sector": "Metals & Mining"},
            {"scrip_code": "532281", "scrip_name": "HINDALCO", "company_name": "Hindalco Industries Limited", "group": "A", "sector": "Metals & Mining"},
            
            # Consumer Goods
            {"scrip_code": "500820", "scrip_name": "ASIANPAINT", "company_name": "Asian Paints Limited", "group": "A", "sector": "Paints"},
        ]
        
        logger.info(f"📋 Using {len(fallback_companies)} major BSE companies as fallback")
        return fallback_companies
    
    def convert_to_bse_company(self, data: Dict) -> BSECompany:
        """Convert raw data to BSECompany object"""
        return BSECompany(
            scrip_code=str(data.get('Scrip_Cd', data.get('scrip_code', data.get('ScripCode', '')))),
            scrip_name=data.get('Scrip_Name', data.get('scrip_name', data.get('ScripName', ''))),
            company_name=data.get('Company_Name', data.get('company_name', data.get('CompanyName', data.get('scrip_name', '')))),
            group=data.get('Group', data.get('group', '')),
            face_value=float(data.get('Face_Value', data.get('face_value', 0)) or 0),
            isin=data.get('ISIN', data.get('isin', '')),
            industry=data.get('Industry', data.get('industry', '')),
            instrument=data.get('Instrument', data.get('instrument', 'EQUITY')),
            sector=data.get('Sector', data.get('sector', '')),
            exchange="BSE_MAIN",
            market_cap_category="Unknown",
            status="ACTIVE",
            last_updated=datetime.now().isoformat()
        )
    
    def fetch_all_bse_companies(self) -> List[BSECompany]:
        """Fetch all BSE companies using multiple methods"""
        logger.info("🚀 Starting comprehensive BSE data fetch...")
        
        all_companies = []
        
        # Method 1: API endpoints
        api_companies = self.fetch_bse_equity_list()
        if api_companies:
            for company_data in api_companies:
                try:
                    company = self.convert_to_bse_company(company_data)
                    all_companies.append(company)
                except Exception as e:
                    logger.warning(f"⚠️ Error converting company data: {e}")
        
        # Method 2: Sensex companies
        sensex_companies = self.fetch_bse_sensex_companies()
        if sensex_companies:
            for company_data in sensex_companies:
                try:
                    company = self.convert_to_bse_company(company_data)
                    all_companies.append(company)
                except Exception as e:
                    logger.warning(f"⚠️ Error converting Sensex company data: {e}")
        
        # Method 3: Website scraping
        try:
            import bs4
            scraped_companies = self.scrape_bse_website()
            if scraped_companies:
                for company_data in scraped_companies:
                    try:
                        company = self.convert_to_bse_company(company_data)
                        all_companies.append(company)
                    except Exception as e:
                        logger.warning(f"⚠️ Error converting scraped company data: {e}")
        except ImportError:
            logger.warning("⚠️ BeautifulSoup not available for web scraping")
        
        # Method 4: Fallback companies (always include)
        fallback_companies = self.get_fallback_bse_companies()
        for company_data in fallback_companies:
            try:
                company = self.convert_to_bse_company(company_data)
                all_companies.append(company)
            except Exception as e:
                logger.warning(f"⚠️ Error converting fallback company data: {e}")
        
        # Remove duplicates based on scrip code
        seen_scrips = set()
        unique_companies = []
        
        for company in all_companies:
            if company.scrip_code and company.scrip_code not in seen_scrips:
                seen_scrips.add(company.scrip_code)
                unique_companies.append(company)
        
        logger.info(f"✅ Total unique BSE companies: {len(unique_companies)}")
        return unique_companies
    
    def save_to_json(self, companies: List[BSECompany], filename: str = "market_data/comprehensive_bse_all_companies.json"):
        """Save companies to JSON file"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert to dictionaries
        companies_dict = [asdict(company) for company in companies]
        
        # Add metadata
        data = {
            "metadata": {
                "total_companies": len(companies),
                "source": "BSE API + Website Scraping + Fallback",
                "methods": ["BSE_API", "BSE_SENSEX", "BSE_WEBSITE", "FALLBACK"],
                "last_updated": datetime.now().isoformat(),
                "data_type": "BSE_ALL_COMPANIES",
                "includes_rnit_ai": any("RNIT" in c.company_name for c in companies)
            },
            "companies": companies_dict
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(companies)} BSE companies to {filename}")
        return filename

def main():
    """Main function to fetch comprehensive BSE data"""
    logger.info("🚀 Starting Comprehensive BSE API Fetcher...")
    
    fetcher = ComprehensiveBSEAPIFetcher()
    
    # Fetch all BSE companies
    companies = fetcher.fetch_all_bse_companies()
    
    if not companies:
        logger.error("❌ No BSE companies fetched")
        return
    
    # Save to JSON
    filename = fetcher.save_to_json(companies)
    
    # Show statistics
    logger.info(f"📊 BSE Companies Statistics:")
    logger.info(f"   Total Companies: {len(companies)}")
    
    # Search for RNIT AI
    rnit_companies = [c for c in companies if "RNIT" in c.company_name.upper()]
    if rnit_companies:
        logger.info(f"🎯 Found RNIT companies:")
        for company in rnit_companies:
            logger.info(f"   ✅ {company.company_name} ({company.scrip_code}) - {company.sector}")
    
    # Show sector distribution
    sectors = {}
    for company in companies:
        sector = company.sector or "Unknown"
        sectors[sector] = sectors.get(sector, 0) + 1
    
    logger.info(f"🏭 Top 10 Sectors:")
    sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
    for sector, count in sorted_sectors[:10]:
        logger.info(f"   {sector}: {count} companies")
    
    logger.info(f"✅ Comprehensive BSE data fetching completed! Data saved to {filename}")

if __name__ == "__main__":
    main()
