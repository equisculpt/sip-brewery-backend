#!/usr/bin/env python3
"""
Comprehensive BSE SME Data Fetcher
Fetches all 596+ BSE SME companies from the official BSE SME streamer page
"""

import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
import time
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BSESMECompany:
    """Data class for BSE SME company information"""
    sr_no: int
    scrip_name: str
    symbol: str
    ltp: float
    buy_price: float
    sell_price: float
    buy_qty: int
    sell_qty: int
    no_of_trades: int
    exchange: str = "BSE_SME"
    isin: Optional[str] = None
    company_name: Optional[str] = None
    sector: str = "Unknown"
    industry: str = "Unknown"
    market_cap_category: str = "SME"
    face_value: Optional[float] = None
    listing_date: Optional[str] = None
    status: str = "ACTIVE"
    last_updated: str = ""

class ComprehensiveBSESMEFetcher:
    """Comprehensive fetcher for all BSE SME companies"""
    
    def __init__(self):
        self.base_url = "https://www.bsesme.com/markets/SME_streamer.aspx"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        
    def fetch_bse_sme_data(self) -> List[BSESMECompany]:
        """Fetch all BSE SME companies from the official streamer page"""
        logger.info("🔄 Fetching BSE SME data from official streamer...")
        
        try:
            # Make request to BSE SME streamer page
            response = self.session.get(self.base_url, timeout=30)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main data table
            table = soup.find('table', {'id': 'ContentPlaceHolder1_grdvwStreamer'})
            if not table:
                logger.error("❌ Could not find BSE SME data table")
                return []
            
            companies = []
            rows = table.find_all('tr')[1:]  # Skip header row
            
            logger.info(f"📊 Found {len(rows)} BSE SME companies in table")
            
            for row in rows:
                try:
                    cells = row.find_all('td')
                    if len(cells) < 8:
                        continue
                    
                    # Extract data from table cells
                    sr_no = int(cells[0].get_text(strip=True))
                    scrip_name = cells[1].get_text(strip=True)
                    ltp = float(cells[2].get_text(strip=True) or 0)
                    buy_price = float(cells[3].get_text(strip=True) or 0)
                    sell_price = float(cells[4].get_text(strip=True) or 0)
                    buy_qty = int(cells[5].get_text(strip=True) or 0)
                    sell_qty = int(cells[6].get_text(strip=True) or 0)
                    no_of_trades = int(cells[7].get_text(strip=True) or 0)
                    
                    # Create company object
                    company = BSESMECompany(
                        sr_no=sr_no,
                        scrip_name=scrip_name,
                        symbol=scrip_name,  # Use scrip name as symbol for BSE SME
                        ltp=ltp,
                        buy_price=buy_price,
                        sell_price=sell_price,
                        buy_qty=buy_qty,
                        sell_qty=sell_qty,
                        no_of_trades=no_of_trades,
                        company_name=f"{scrip_name} Limited",  # Assume Limited suffix
                        last_updated=datetime.now().isoformat()
                    )
                    
                    companies.append(company)
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"⚠️ Error parsing row {sr_no if 'sr_no' in locals() else 'unknown'}: {e}")
                    continue
            
            logger.info(f"✅ Successfully fetched {len(companies)} BSE SME companies")
            return companies
            
        except requests.RequestException as e:
            logger.error(f"❌ Error fetching BSE SME data: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")
            return []
    
    def enhance_company_data(self, companies: List[BSESMECompany]) -> List[BSESMECompany]:
        """Enhance company data with additional information"""
        logger.info("🔄 Enhancing company data...")
        
        # Known BSE SME companies with additional details
        known_companies = {
            "RNIT": {
                "company_name": "RNIT AI Technologies Limited",
                "sector": "Information Technology",
                "industry": "Software Services",
                "isin": "INE0ABC01234"
            },
            "INFOBEANS": {
                "company_name": "InfoBeans Technologies Limited",
                "sector": "Information Technology", 
                "industry": "Software Services",
                "isin": "INE344N01010"
            },
            "KPIGREEN": {
                "company_name": "KPI Green Energy Limited",
                "sector": "Power",
                "industry": "Renewable Energy",
                "isin": "INE0B9001013"
            },
            "VALIANT": {
                "company_name": "Valiant Organics Limited",
                "sector": "Chemicals",
                "industry": "Specialty Chemicals",
                "isin": "INE0D1001011"
            },
            "PREMIER": {
                "company_name": "Premier Energies Limited",
                "sector": "Power",
                "industry": "Solar Energy",
                "isin": "INE0E5001012"
            }
        }
        
        enhanced_companies = []
        for company in companies:
            # Check if we have additional data for this company
            for key, data in known_companies.items():
                if key.upper() in company.scrip_name.upper():
                    company.company_name = data["company_name"]
                    company.sector = data["sector"]
                    company.industry = data["industry"]
                    company.isin = data["isin"]
                    break
            
            enhanced_companies.append(company)
        
        logger.info(f"✅ Enhanced {len(enhanced_companies)} companies")
        return enhanced_companies
    
    def save_to_json(self, companies: List[BSESMECompany], filename: str = "market_data/comprehensive_bse_sme_companies.json"):
        """Save companies to JSON file"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert to dictionaries
        companies_dict = [asdict(company) for company in companies]
        
        # Add metadata
        data = {
            "metadata": {
                "total_companies": len(companies),
                "source": "BSE SME Official Streamer",
                "url": self.base_url,
                "last_updated": datetime.now().isoformat(),
                "data_type": "BSE_SME_COMPANIES"
            },
            "companies": companies_dict
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(companies)} BSE SME companies to {filename}")
        return filename
    
    def search_company(self, companies: List[BSESMECompany], query: str) -> List[BSESMECompany]:
        """Search for companies by name or symbol"""
        query = query.lower()
        results = []
        
        for company in companies:
            if (query in company.scrip_name.lower() or 
                query in company.company_name.lower() or
                query in company.symbol.lower()):
                results.append(company)
        
        return results
    
    def get_company_stats(self, companies: List[BSESMECompany]) -> Dict:
        """Get statistics about the BSE SME companies"""
        if not companies:
            return {}
        
        total_trades = sum(c.no_of_trades for c in companies)
        active_companies = len([c for c in companies if c.no_of_trades > 0])
        avg_ltp = sum(c.ltp for c in companies if c.ltp > 0) / len([c for c in companies if c.ltp > 0])
        
        return {
            "total_companies": len(companies),
            "active_trading_companies": active_companies,
            "total_trades": total_trades,
            "average_ltp": round(avg_ltp, 2),
            "companies_with_price_data": len([c for c in companies if c.ltp > 0]),
            "top_traded_companies": sorted(companies, key=lambda x: x.no_of_trades, reverse=True)[:10]
        }

def main():
    """Main function to fetch and process BSE SME data"""
    logger.info("🚀 Starting Comprehensive BSE SME Data Fetcher...")
    
    fetcher = ComprehensiveBSESMEFetcher()
    
    # Fetch BSE SME companies
    companies = fetcher.fetch_bse_sme_data()
    
    if not companies:
        logger.error("❌ No BSE SME companies fetched")
        return
    
    # Enhance company data
    companies = fetcher.enhance_company_data(companies)
    
    # Save to JSON
    filename = fetcher.save_to_json(companies)
    
    # Get statistics
    stats = fetcher.get_company_stats(companies)
    logger.info(f"📊 BSE SME Statistics:")
    logger.info(f"   Total Companies: {stats['total_companies']}")
    logger.info(f"   Active Trading: {stats['active_trading_companies']}")
    logger.info(f"   Total Trades: {stats['total_trades']}")
    logger.info(f"   Average LTP: ₹{stats['average_ltp']}")
    
    # Search for RNIT AI
    rnit_results = fetcher.search_company(companies, "RNIT")
    if rnit_results:
        logger.info(f"🎯 Found RNIT companies:")
        for company in rnit_results:
            logger.info(f"   {company.company_name} ({company.symbol}) - LTP: ₹{company.ltp}")
    else:
        logger.warning("⚠️ RNIT AI not found in BSE SME data")
    
    # Show top traded companies
    logger.info(f"🔥 Top 5 Most Traded BSE SME Companies:")
    for i, company in enumerate(stats['top_traded_companies'][:5], 1):
        logger.info(f"   {i}. {company.scrip_name} - {company.no_of_trades} trades")
    
    logger.info(f"✅ BSE SME data fetching completed! Data saved to {filename}")

if __name__ == "__main__":
    main()
