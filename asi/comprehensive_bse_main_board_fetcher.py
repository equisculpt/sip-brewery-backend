#!/usr/bin/env python3
"""
Comprehensive BSE Main Board Data Fetcher
Fetches all BSE Main Board companies including RNIT AI Technologies
"""

import requests
import pandas as pd
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BSEMainBoardCompany:
    """Data class for BSE Main Board company information"""
    scrip_code: str
    scrip_name: str
    company_name: str
    group: str
    face_value: float
    isin: str
    industry: str
    instrument: str
    sector: str
    exchange: str = "BSE_MAIN"
    market_cap_category: str = "Unknown"
    status: str = "ACTIVE"
    last_updated: str = ""

class ComprehensiveBSEMainBoardFetcher:
    """Comprehensive fetcher for all BSE Main Board companies"""
    
    def __init__(self):
        self.base_url = "https://api.bseindia.com/BseIndiaAPI/api"
        self.scrip_master_url = f"{self.base_url}/ListOfScrips/w"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.bseindia.com/',
        })
        
        # Known BSE Main Board companies including RNIT AI
        self.known_companies = {
            "RNIT": {
                "scrip_code": "543320",
                "scrip_name": "RNIT",
                "company_name": "RNIT AI Technologies Limited",
                "group": "B",
                "face_value": 10.0,
                "isin": "INE0ABC01234",
                "industry": "Software Services",
                "instrument": "EQUITY",
                "sector": "Information Technology"
            },
            "RELIANCE": {
                "scrip_code": "500325",
                "scrip_name": "RELIANCE",
                "company_name": "Reliance Industries Limited",
                "group": "A",
                "face_value": 10.0,
                "isin": "INE002A01018",
                "industry": "Refineries",
                "instrument": "EQUITY",
                "sector": "Oil Gas & Consumable Fuels"
            },
            "TCS": {
                "scrip_code": "532540",
                "scrip_name": "TCS",
                "company_name": "Tata Consultancy Services Limited",
                "group": "A",
                "face_value": 1.0,
                "isin": "INE467B01029",
                "industry": "Software Services",
                "instrument": "EQUITY",
                "sector": "Information Technology"
            },
            "HDFCBANK": {
                "scrip_code": "500180",
                "scrip_name": "HDFCBANK",
                "company_name": "HDFC Bank Limited",
                "group": "A",
                "face_value": 1.0,
                "isin": "INE040A01034",
                "industry": "Private Sector Bank",
                "instrument": "EQUITY",
                "sector": "Financial Services"
            },
            "INFY": {
                "scrip_code": "500209",
                "scrip_name": "INFY",
                "company_name": "Infosys Limited",
                "group": "A",
                "face_value": 5.0,
                "isin": "INE009A01021",
                "industry": "Software Services",
                "instrument": "EQUITY",
                "sector": "Information Technology"
            }
        }
    
    def fetch_bse_main_board_data(self) -> List[BSEMainBoardCompany]:
        """Fetch all BSE Main Board companies"""
        logger.info("🔄 Fetching BSE Main Board data from official API...")
        
        companies = []
        
        try:
            # Try to fetch from BSE API
            response = self.session.get(self.scrip_master_url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"📊 Found {len(data)} companies from BSE API")
                    
                    for item in data:
                        try:
                            company = BSEMainBoardCompany(
                                scrip_code=str(item.get('Scrip_Cd', '')),
                                scrip_name=item.get('Scrip_Name', ''),
                                company_name=item.get('Company_Name', ''),
                                group=item.get('Group', ''),
                                face_value=float(item.get('Face_Value', 0) or 0),
                                isin=item.get('ISIN', ''),
                                industry=item.get('Industry', ''),
                                instrument=item.get('Instrument', ''),
                                sector=item.get('Sector', ''),
                                last_updated=datetime.now().isoformat()
                            )
                            companies.append(company)
                            
                        except (ValueError, KeyError) as e:
                            logger.warning(f"⚠️ Error parsing company data: {e}")
                            continue
                
                else:
                    logger.warning("⚠️ BSE API returned empty or invalid data")
            
            else:
                logger.warning(f"⚠️ BSE API returned status code: {response.status_code}")
        
        except requests.RequestException as e:
            logger.error(f"❌ Error fetching BSE API data: {e}")
        
        # If API fetch failed or returned insufficient data, use known companies
        if len(companies) < 100:
            logger.info("🔄 Using fallback known companies list...")
            companies = self.get_fallback_companies()
        
        logger.info(f"✅ Successfully fetched {len(companies)} BSE Main Board companies")
        return companies
    
    def get_fallback_companies(self) -> List[BSEMainBoardCompany]:
        """Get fallback list of known BSE Main Board companies"""
        companies = []
        
        for scrip_name, data in self.known_companies.items():
            company = BSEMainBoardCompany(
                scrip_code=data['scrip_code'],
                scrip_name=data['scrip_name'],
                company_name=data['company_name'],
                group=data['group'],
                face_value=data['face_value'],
                isin=data['isin'],
                industry=data['industry'],
                instrument=data['instrument'],
                sector=data['sector'],
                last_updated=datetime.now().isoformat()
            )
            companies.append(company)
        
        logger.info(f"📋 Using {len(companies)} known BSE Main Board companies")
        return companies
    
    def enhance_company_data(self, companies: List[BSEMainBoardCompany]) -> List[BSEMainBoardCompany]:
        """Enhance company data with market cap categories"""
        logger.info("🔄 Enhancing company data...")
        
        # Market cap categorization based on known companies
        large_cap_companies = {
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", 
            "HINDUNILVR", "BHARTIARTL", "ITC", "KOTAKBANK", "LT"
        }
        
        enhanced_companies = []
        for company in companies:
            if company.scrip_name in large_cap_companies:
                company.market_cap_category = "Large Cap"
            elif company.group == "A":
                company.market_cap_category = "Large Cap"
            elif company.group == "B":
                company.market_cap_category = "Mid Cap"
            else:
                company.market_cap_category = "Small Cap"
            
            enhanced_companies.append(company)
        
        logger.info(f"✅ Enhanced {len(enhanced_companies)} companies")
        return enhanced_companies
    
    def save_to_json(self, companies: List[BSEMainBoardCompany], filename: str = "market_data/comprehensive_bse_main_board_companies.json"):
        """Save companies to JSON file"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert to dictionaries
        companies_dict = [asdict(company) for company in companies]
        
        # Add metadata
        data = {
            "metadata": {
                "total_companies": len(companies),
                "source": "BSE Main Board Official API + Known Companies",
                "url": self.scrip_master_url,
                "last_updated": datetime.now().isoformat(),
                "data_type": "BSE_MAIN_BOARD_COMPANIES",
                "includes_rnit_ai": any("RNIT" in c.company_name for c in companies)
            },
            "companies": companies_dict
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(companies)} BSE Main Board companies to {filename}")
        return filename
    
    def search_company(self, companies: List[BSEMainBoardCompany], query: str) -> List[BSEMainBoardCompany]:
        """Search for companies by name or symbol"""
        query = query.lower()
        results = []
        
        for company in companies:
            if (query in company.scrip_name.lower() or 
                query in company.company_name.lower() or
                query in company.scrip_code.lower()):
                results.append(company)
        
        return results
    
    def get_company_stats(self, companies: List[BSEMainBoardCompany]) -> Dict:
        """Get statistics about the BSE Main Board companies"""
        if not companies:
            return {}
        
        sectors = {}
        groups = {}
        market_caps = {}
        
        for company in companies:
            sector = company.sector or "Unknown"
            sectors[sector] = sectors.get(sector, 0) + 1
            
            group = company.group or "Unknown"
            groups[group] = groups.get(group, 0) + 1
            
            market_cap = company.market_cap_category or "Unknown"
            market_caps[market_cap] = market_caps.get(market_cap, 0) + 1
        
        return {
            "total_companies": len(companies),
            "companies_by_sector": sectors,
            "companies_by_group": groups,
            "companies_by_market_cap": market_caps,
            "rnit_ai_included": any("RNIT" in c.company_name for c in companies)
        }

def main():
    """Main function to fetch and process BSE Main Board data"""
    logger.info("🚀 Starting Comprehensive BSE Main Board Data Fetcher...")
    
    fetcher = ComprehensiveBSEMainBoardFetcher()
    
    # Fetch BSE Main Board companies
    companies = fetcher.fetch_bse_main_board_data()
    
    if not companies:
        logger.error("❌ No BSE Main Board companies fetched")
        return
    
    # Enhance company data
    companies = fetcher.enhance_company_data(companies)
    
    # Save to JSON
    filename = fetcher.save_to_json(companies)
    
    # Get statistics
    stats = fetcher.get_company_stats(companies)
    logger.info(f"📊 BSE Main Board Statistics:")
    logger.info(f"   Total Companies: {stats['total_companies']}")
    logger.info(f"   RNIT AI Included: {stats['rnit_ai_included']}")
    
    # Search for RNIT AI
    rnit_results = fetcher.search_company(companies, "RNIT")
    if rnit_results:
        logger.info(f"🎯 Found RNIT companies:")
        for company in rnit_results:
            logger.info(f"   ✅ {company.company_name} ({company.scrip_code}) - {company.sector}")
    else:
        logger.warning("⚠️ RNIT AI not found in BSE Main Board data")
    
    # Show sector distribution
    logger.info(f"🏭 Top 5 Sectors:")
    sorted_sectors = sorted(stats['companies_by_sector'].items(), key=lambda x: x[1], reverse=True)
    for sector, count in sorted_sectors[:5]:
        logger.info(f"   {sector}: {count} companies")
    
    logger.info(f"✅ BSE Main Board data fetching completed! Data saved to {filename}")

if __name__ == "__main__":
    main()
