#!/usr/bin/env python3
"""
Enhanced BSE Complete Fetcher
Try to get the complete BSE Main Board list (should be 3000+ companies)
"""

import requests
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedBSECompleteFetcher:
    """Enhanced fetcher to get complete BSE Main Board data"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.bseindia.com/',
        })
        
        # Multiple BSE data sources
        self.data_sources = [
            {
                "name": "BSE Official API",
                "url": "https://api.bseindia.com/BseIndiaAPI/api/ListOfScrips/w",
                "method": "api"
            },
            {
                "name": "BSE Market Data",
                "url": "https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w",
                "method": "api"
            },
            {
                "name": "BSE CSV Download",
                "url": "https://www.bseindia.com/download/BhavCopy/Equity/EQ_ISINCODE_",
                "method": "csv"
            },
            {
                "name": "BSE Equity List",
                "url": "https://www.bseindia.com/corporates/List_Scrips.aspx",
                "method": "web"
            }
        ]
    
    def fetch_bse_csv_data(self) -> List[Dict]:
        """Try to fetch BSE data from CSV files"""
        logger.info("🔄 Attempting to fetch BSE CSV data...")
        
        # Try different date formats for CSV files
        today = datetime.now()
        date_formats = [
            today.strftime("%d%m%y"),
            today.strftime("%d%m%Y"),
            (today.replace(day=today.day-1)).strftime("%d%m%y"),
            (today.replace(day=today.day-1)).strftime("%d%m%Y"),
        ]
        
        for date_str in date_formats:
            try:
                csv_url = f"https://www.bseindia.com/download/BhavCopy/Equity/EQ_ISINCODE_{date_str}.zip"
                logger.info(f"🔄 Trying CSV URL: {csv_url}")
                
                response = self.session.get(csv_url, timeout=30)
                if response.status_code == 200:
                    # Try to process ZIP file
                    import zipfile
                    import io
                    
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                        for file_name in zip_file.namelist():
                            if file_name.endswith('.csv'):
                                csv_content = zip_file.read(file_name).decode('utf-8')
                                df = pd.read_csv(io.StringIO(csv_content))
                                
                                companies = []
                                for _, row in df.iterrows():
                                    company = {
                                        'scrip_code': str(row.get('SC_CODE', '')),
                                        'scrip_name': str(row.get('SC_NAME', '')),
                                        'company_name': str(row.get('SC_NAME', '')),
                                        'isin': str(row.get('ISIN_NO', '')),
                                        'group': str(row.get('SC_GROUP', '')),
                                        'face_value': float(row.get('FACE_VAL', 0) or 0),
                                    }
                                    companies.append(company)
                                
                                logger.info(f"✅ Found {len(companies)} companies from CSV")
                                return companies
                
            except Exception as e:
                logger.warning(f"⚠️ CSV fetch failed for {date_str}: {e}")
        
        return []
    
    def fetch_bse_api_comprehensive(self) -> List[Dict]:
        """Comprehensive API fetch with multiple endpoints"""
        logger.info("🔄 Fetching comprehensive BSE API data...")
        
        all_companies = []
        
        # Try different API endpoints and parameters
        endpoints = [
            ("https://api.bseindia.com/BseIndiaAPI/api/ListOfScrips/w", {}),
            ("https://api.bseindia.com/BseIndiaAPI/api/getScripHeaderData/w", {}),
            ("https://api.bseindia.com/BseIndiaAPI/api/ComHeader/w", {}),
            ("https://api.bseindia.com/BseIndiaAPI/api/DefaultData/w", {}),
            ("https://api.bseindia.com/BseIndiaAPI/api/Sensex/w", {}),
        ]
        
        for url, params in endpoints:
            try:
                logger.info(f"🔄 Trying: {url}")
                response = self.session.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        if isinstance(data, list) and len(data) > 0:
                            logger.info(f"✅ Found {len(data)} companies from {url}")
                            all_companies.extend(data)
                        elif isinstance(data, dict):
                            # Try to extract from different keys
                            for key in ['Table', 'data', 'companies', 'scrips', 'result', 'Table1']:
                                if key in data and isinstance(data[key], list):
                                    logger.info(f"✅ Found {len(data[key])} companies from {url} (key: {key})")
                                    all_companies.extend(data[key])
                                    break
                    
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ Invalid JSON from {url}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.warning(f"⚠️ Error fetching {url}: {e}")
        
        return all_companies
    
    def get_comprehensive_fallback(self) -> List[Dict]:
        """Get comprehensive fallback list of BSE companies"""
        logger.info("🔄 Loading comprehensive BSE fallback data...")
        
        # Comprehensive list of major BSE companies
        companies = [
            # Technology
            {"scrip_code": "543320", "scrip_name": "RNIT", "company_name": "RNIT AI Technologies Limited", "sector": "Information Technology", "group": "B"},
            {"scrip_code": "532540", "scrip_name": "TCS", "company_name": "Tata Consultancy Services Limited", "sector": "Information Technology", "group": "A"},
            {"scrip_code": "500209", "scrip_name": "INFY", "company_name": "Infosys Limited", "sector": "Information Technology", "group": "A"},
            {"scrip_code": "507685", "scrip_name": "WIPRO", "company_name": "Wipro Limited", "sector": "Information Technology", "group": "A"},
            {"scrip_code": "532281", "scrip_name": "HCLTECH", "company_name": "HCL Technologies Limited", "sector": "Information Technology", "group": "A"},
            {"scrip_code": "526371", "scrip_name": "TECHM", "company_name": "Tech Mahindra Limited", "sector": "Information Technology", "group": "A"},
            
            # Banking & Financial Services
            {"scrip_code": "500180", "scrip_name": "HDFCBANK", "company_name": "HDFC Bank Limited", "sector": "Financial Services", "group": "A"},
            {"scrip_code": "532174", "scrip_name": "ICICIBANK", "company_name": "ICICI Bank Limited", "sector": "Financial Services", "group": "A"},
            {"scrip_code": "500112", "scrip_name": "SBIN", "company_name": "State Bank of India", "sector": "Financial Services", "group": "A"},
            {"scrip_code": "500247", "scrip_name": "KOTAKBANK", "company_name": "Kotak Mahindra Bank Limited", "sector": "Financial Services", "group": "A"},
            {"scrip_code": "532215", "scrip_name": "AXISBANK", "company_name": "Axis Bank Limited", "sector": "Financial Services", "group": "A"},
            {"scrip_code": "532978", "scrip_name": "BAJFINANCE", "company_name": "Bajaj Finance Limited", "sector": "Financial Services", "group": "A"},
            {"scrip_code": "500034", "scrip_name": "BAJAJFINSV", "company_name": "Bajaj Finserv Limited", "sector": "Financial Services", "group": "A"},
            
            # Oil & Gas
            {"scrip_code": "500325", "scrip_name": "RELIANCE", "company_name": "Reliance Industries Limited", "sector": "Oil Gas & Consumable Fuels", "group": "A"},
            {"scrip_code": "500312", "scrip_name": "ONGC", "company_name": "Oil and Natural Gas Corporation Limited", "sector": "Oil Gas & Consumable Fuels", "group": "A"},
            {"scrip_code": "500104", "scrip_name": "IOCL", "company_name": "Indian Oil Corporation Limited", "sector": "Oil Gas & Consumable Fuels", "group": "A"},
            {"scrip_code": "500547", "scrip_name": "BPCL", "company_name": "Bharat Petroleum Corporation Limited", "sector": "Oil Gas & Consumable Fuels", "group": "A"},
            
            # FMCG
            {"scrip_code": "500696", "scrip_name": "HINDUNILVR", "company_name": "Hindustan Unilever Limited", "sector": "Fast Moving Consumer Goods", "group": "A"},
            {"scrip_code": "500875", "scrip_name": "ITC", "company_name": "ITC Limited", "sector": "Fast Moving Consumer Goods", "group": "A"},
            {"scrip_code": "500790", "scrip_name": "NESTLEIND", "company_name": "Nestle India Limited", "sector": "Fast Moving Consumer Goods", "group": "A"},
            {"scrip_code": "500770", "scrip_name": "TATACONSUMR", "company_name": "Tata Consumer Products Limited", "sector": "Fast Moving Consumer Goods", "group": "A"},
            {"scrip_code": "500870", "scrip_name": "COLPAL", "company_name": "Colgate Palmolive (India) Limited", "sector": "Fast Moving Consumer Goods", "group": "A"},
            
            # Telecom
            {"scrip_code": "532454", "scrip_name": "BHARTIARTL", "company_name": "Bharti Airtel Limited", "sector": "Telecommunication", "group": "A"},
            {"scrip_code": "532321", "scrip_name": "IDEA", "company_name": "Vodafone Idea Limited", "sector": "Telecommunication", "group": "B"},
            
            # Automobile
            {"scrip_code": "532500", "scrip_name": "MARUTI", "company_name": "Maruti Suzuki India Limited", "sector": "Automobile", "group": "A"},
            {"scrip_code": "500570", "scrip_name": "TATAMOTORS", "company_name": "Tata Motors Limited", "sector": "Automobile", "group": "A"},
            {"scrip_code": "532343", "scrip_name": "BAJAJ-AUTO", "company_name": "Bajaj Auto Limited", "sector": "Automobile", "group": "A"},
            {"scrip_code": "500490", "scrip_name": "HEROMOTOCO", "company_name": "Hero MotoCorp Limited", "sector": "Automobile", "group": "A"},
            {"scrip_code": "532454", "scrip_name": "MAHINDRA", "company_name": "Mahindra & Mahindra Limited", "sector": "Automobile", "group": "A"},
            
            # Infrastructure & Construction
            {"scrip_code": "500510", "scrip_name": "LT", "company_name": "Larsen & Toubro Limited", "sector": "Construction", "group": "A"},
            {"scrip_code": "500114", "scrip_name": "ULTRACEMCO", "company_name": "UltraTech Cement Limited", "sector": "Cement", "group": "A"},
            {"scrip_code": "500387", "scrip_name": "SHREECEM", "company_name": "Shree Cement Limited", "sector": "Cement", "group": "A"},
            
            # Pharmaceuticals
            {"scrip_code": "500124", "scrip_name": "DRREDDY", "company_name": "Dr. Reddy's Laboratories Limited", "sector": "Pharmaceuticals", "group": "A"},
            {"scrip_code": "500680", "scrip_name": "CIPLA", "company_name": "Cipla Limited", "sector": "Pharmaceuticals", "group": "A"},
            {"scrip_code": "500550", "scrip_name": "SUNPHARMA", "company_name": "Sun Pharmaceutical Industries Limited", "sector": "Pharmaceuticals", "group": "A"},
            {"scrip_code": "532321", "scrip_name": "LUPIN", "company_name": "Lupin Limited", "sector": "Pharmaceuticals", "group": "A"},
            
            # Metals & Mining
            {"scrip_code": "500295", "scrip_name": "TATASTEEL", "company_name": "Tata Steel Limited", "sector": "Metals & Mining", "group": "A"},
            {"scrip_code": "500188", "scrip_name": "HINDALCO", "company_name": "Hindalco Industries Limited", "sector": "Metals & Mining", "group": "A"},
            {"scrip_code": "500440", "scrip_name": "HINDZINC", "company_name": "Hindustan Zinc Limited", "sector": "Metals & Mining", "group": "A"},
            {"scrip_code": "532454", "scrip_name": "COALINDIA", "company_name": "Coal India Limited", "sector": "Metals & Mining", "group": "A"},
            
            # Consumer Goods
            {"scrip_code": "500820", "scrip_name": "ASIANPAINT", "company_name": "Asian Paints Limited", "sector": "Paints", "group": "A"},
            {"scrip_code": "500570", "scrip_name": "BERGER", "company_name": "Berger Paints India Limited", "sector": "Paints", "group": "A"},
            
            # Power
            {"scrip_code": "532454", "scrip_name": "NTPC", "company_name": "NTPC Limited", "sector": "Power", "group": "A"},
            {"scrip_code": "500790", "scrip_name": "POWERGRID", "company_name": "Power Grid Corporation of India Limited", "sector": "Power", "group": "A"},
        ]
        
        logger.info(f"📋 Loaded {len(companies)} comprehensive fallback companies")
        return companies
    
    def normalize_company_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Normalize and clean company data"""
        logger.info("🔄 Normalizing company data...")
        
        normalized = []
        seen_scrips = set()
        
        for company in raw_data:
            # Extract scrip code from various possible fields
            scrip_code = str(company.get('Scrip_Cd', company.get('scrip_code', company.get('ScripCode', company.get('SC_CODE', '')))))
            
            # Skip if no scrip code or already seen
            if not scrip_code or scrip_code in seen_scrips:
                continue
            
            # Extract other fields
            scrip_name = company.get('Scrip_Name', company.get('scrip_name', company.get('ScripName', company.get('SC_NAME', ''))))
            company_name = company.get('Company_Name', company.get('company_name', company.get('CompanyName', scrip_name)))
            
            # Skip if no meaningful name
            if not company_name or company_name.strip() == '':
                continue
            
            normalized_company = {
                'scrip_code': scrip_code,
                'scrip_name': scrip_name,
                'company_name': company_name,
                'group': company.get('Group', company.get('group', company.get('SC_GROUP', ''))),
                'face_value': float(company.get('Face_Value', company.get('face_value', company.get('FACE_VAL', 0)) or 0)),
                'isin': company.get('ISIN', company.get('isin', company.get('ISIN_NO', ''))),
                'industry': company.get('Industry', company.get('industry', '')),
                'sector': company.get('Sector', company.get('sector', '')),
                'instrument': company.get('Instrument', company.get('instrument', 'EQUITY')),
                'exchange': 'BSE_MAIN',
                'status': 'ACTIVE',
                'last_updated': datetime.now().isoformat()
            }
            
            normalized.append(normalized_company)
            seen_scrips.add(scrip_code)
        
        logger.info(f"✅ Normalized {len(normalized)} unique companies")
        return normalized
    
    def fetch_complete_bse_data(self) -> List[Dict]:
        """Fetch complete BSE data using all available methods"""
        logger.info("🚀 Starting complete BSE data fetch...")
        
        all_raw_data = []
        
        # Method 1: API fetch
        api_data = self.fetch_bse_api_comprehensive()
        if api_data:
            all_raw_data.extend(api_data)
            logger.info(f"✅ API fetch: {len(api_data)} companies")
        
        # Method 2: CSV fetch
        csv_data = self.fetch_bse_csv_data()
        if csv_data:
            all_raw_data.extend(csv_data)
            logger.info(f"✅ CSV fetch: {len(csv_data)} companies")
        
        # Method 3: Fallback data (always include)
        fallback_data = self.get_comprehensive_fallback()
        all_raw_data.extend(fallback_data)
        logger.info(f"✅ Fallback data: {len(fallback_data)} companies")
        
        # Normalize all data
        normalized_companies = self.normalize_company_data(all_raw_data)
        
        logger.info(f"🎯 Total BSE companies: {len(normalized_companies)}")
        return normalized_companies
    
    def save_complete_data(self, companies: List[Dict], filename: str = "market_data/complete_bse_companies.json"):
        """Save complete BSE data"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        data = {
            "metadata": {
                "total_companies": len(companies),
                "source": "BSE API + CSV + Comprehensive Fallback",
                "methods": ["BSE_API_COMPREHENSIVE", "BSE_CSV", "COMPREHENSIVE_FALLBACK"],
                "last_updated": datetime.now().isoformat(),
                "data_type": "COMPLETE_BSE_COMPANIES",
                "includes_rnit_ai": any("RNIT" in c['company_name'] for c in companies),
                "major_companies_count": len([c for c in companies if c['group'] == 'A'])
            },
            "companies": companies
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(companies)} BSE companies to {filename}")
        return filename

def main():
    """Main function"""
    logger.info("🚀 Starting Enhanced BSE Complete Fetcher...")
    
    fetcher = EnhancedBSECompleteFetcher()
    
    # Fetch complete BSE data
    companies = fetcher.fetch_complete_bse_data()
    
    # Save data
    filename = fetcher.save_complete_data(companies)
    
    # Show statistics
    logger.info(f"\n📊 Complete BSE Data Statistics:")
    logger.info(f"   Total Companies: {len(companies)}")
    
    # Check for RNIT AI
    rnit_companies = [c for c in companies if "RNIT" in c['company_name'].upper()]
    logger.info(f"   RNIT AI Found: {len(rnit_companies) > 0}")
    
    # Show group distribution
    groups = {}
    for company in companies:
        group = company['group'] or "Unknown"
        groups[group] = groups.get(group, 0) + 1
    
    logger.info(f"   Group Distribution:")
    for group, count in sorted(groups.items()):
        logger.info(f"     Group {group}: {count} companies")
    
    logger.info(f"\n✅ Enhanced BSE complete fetch completed! Data saved to {filename}")

if __name__ == "__main__":
    main()
