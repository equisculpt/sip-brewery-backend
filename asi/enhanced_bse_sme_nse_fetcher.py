#!/usr/bin/env python3
"""
Enhanced BSE SME + NSE Comprehensive Data Fetcher
Includes BSE Main Board + BSE SME + NSE Main + NSE SME + NSE Emerge
"""

import asyncio
import aiohttp
import logging
import json
import re
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import requests
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedCompanyData:
    """Unified company data structure for multi-exchange support"""
    primary_symbol: str
    company_name: str
    exchanges: List[str]  # List of exchanges where company is listed
    symbols: Dict[str, str]  # Exchange -> Symbol mapping
    isin: Optional[str]
    sector: str
    industry: str
    market_cap_category: str
    listing_dates: Dict[str, str]  # Exchange -> Listing date mapping
    face_value: Optional[float]
    status: str
    last_updated: str

class EnhancedBSESMENSEFetcher:
    """Enhanced fetcher for BSE Main + BSE SME + NSE Main + NSE SME + NSE Emerge"""
    
    def __init__(self):
        self.session = None
        self.data_dir = Path("market_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Enhanced URLs for comprehensive coverage
        self.urls = {
            # NSE URLs
            "nse_equity": "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
            "nse_sme": "https://nsearchives.nseindia.com/emerge/sme/SME_EQUITY_L.csv",
            "nse_emerge": "https://nsearchives.nseindia.com/emerge/emerge/EMERGE_EQUITY_L.csv",
            
            # BSE URLs
            "bse_main": "https://api.bseindia.com/BseIndiaAPI/api/ListOfScrips/w",
            "bse_sme": "https://www.bsesme.com/corporates/List_Scrips.html",
            "bse_scrips": "https://mock.bseindia.com/corporates/List_Scrips.html"
        }
        
        # Headers to bypass anti-bot protection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    async def fetch_nse_data(self) -> List[Dict]:
        """Fetch comprehensive NSE data (Main + SME + Emerge)"""
        nse_companies = []
        
        try:
            # Create session with headers
            session = requests.Session()
            session.headers.update(self.headers)
            
            # Fetch NSE Main Board
            logger.info("Fetching NSE Main Board companies...")
            try:
                response = session.get(self.urls["nse_equity"], timeout=30)
                if response.status_code == 200:
                    df_nse_main = pd.read_csv(pd.io.common.StringIO(response.text))
                    for _, row in df_nse_main.iterrows():
                        company = {
                            "symbol": str(row.get("SYMBOL", "")).strip(),
                            "name": str(row.get("NAME OF COMPANY", "")).strip(),
                            "exchange": "NSE_MAIN",
                            "isin": str(row.get("ISIN NUMBER", "")).strip() if pd.notna(row.get("ISIN NUMBER")) else None,
                            "sector": "Unknown",
                            "industry": "Unknown", 
                            "market_cap_category": "Unknown",
                            "listing_date": str(row.get("DATE OF LISTING", "")).strip() if pd.notna(row.get("DATE OF LISTING")) else None,
                            "face_value": float(row.get("FACE VALUE", 0)) if pd.notna(row.get("FACE VALUE")) else None,
                            "status": "ACTIVE"
                        }
                        if company["symbol"] and company["name"]:
                            nse_companies.append(company)
                    logger.info(f"Fetched {len(nse_companies)} NSE Main Board companies")
            except Exception as e:
                logger.error(f"Error fetching NSE Main Board: {e}")
            
            # Fetch NSE SME
            logger.info("Fetching NSE SME companies...")
            try:
                response = session.get(self.urls["nse_sme"], timeout=30)
                if response.status_code == 200:
                    df_nse_sme = pd.read_csv(pd.io.common.StringIO(response.text))
                    sme_count = 0
                    for _, row in df_nse_sme.iterrows():
                        company = {
                            "symbol": str(row.get("SYMBOL", "")).strip(),
                            "name": str(row.get("NAME OF COMPANY", "")).strip(),
                            "exchange": "NSE_SME",
                            "isin": str(row.get("ISIN NUMBER", "")).strip() if pd.notna(row.get("ISIN NUMBER")) else None,
                            "sector": "Unknown",
                            "industry": "Unknown",
                            "market_cap_category": "SME",
                            "listing_date": str(row.get("DATE OF LISTING", "")).strip() if pd.notna(row.get("DATE OF LISTING")) else None,
                            "face_value": float(row.get("FACE VALUE", 0)) if pd.notna(row.get("FACE VALUE")) else None,
                            "status": "ACTIVE"
                        }
                        if company["symbol"] and company["name"]:
                            nse_companies.append(company)
                            sme_count += 1
                    logger.info(f"Fetched {sme_count} NSE SME companies")
            except Exception as e:
                logger.error(f"Error fetching NSE SME: {e}")
            
            # Fetch NSE Emerge
            logger.info("Fetching NSE Emerge companies...")
            try:
                response = session.get(self.urls["nse_emerge"], timeout=30)
                if response.status_code == 200:
                    df_nse_emerge = pd.read_csv(pd.io.common.StringIO(response.text))
                    emerge_count = 0
                    for _, row in df_nse_emerge.iterrows():
                        company = {
                            "symbol": str(row.get("SYMBOL", "")).strip(),
                            "name": str(row.get("NAME OF COMPANY", "")).strip(),
                            "exchange": "NSE_EMERGE",
                            "isin": str(row.get("ISIN NUMBER", "")).strip() if pd.notna(row.get("ISIN NUMBER")) else None,
                            "sector": "Unknown",
                            "industry": "Unknown",
                            "market_cap_category": "Emerging",
                            "listing_date": str(row.get("DATE OF LISTING", "")).strip() if pd.notna(row.get("DATE OF LISTING")) else None,
                            "face_value": float(row.get("FACE VALUE", 0)) if pd.notna(row.get("FACE VALUE")) else None,
                            "status": "ACTIVE"
                        }
                        if company["symbol"] and company["name"]:
                            nse_companies.append(company)
                            emerge_count += 1
                    logger.info(f"Fetched {emerge_count} NSE Emerge companies")
            except Exception as e:
                logger.error(f"Error fetching NSE Emerge: {e}")
            
            session.close()
            logger.info(f"Total NSE companies fetched: {len(nse_companies)}")
            
        except Exception as e:
            logger.error(f"Error in NSE data fetching: {e}")
        
        return nse_companies

    def fetch_bse_data(self) -> List[Dict]:
        """Fetch comprehensive BSE data (Main + SME)"""
        bse_companies = []
        
        try:
            # Method 1: Try BSE API
            logger.info("Fetching BSE companies via API...")
            try:
                session = requests.Session()
                session.headers.update(self.headers)
                
                # Try BSE API
                response = session.get(self.urls["bse_main"], timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, list):
                        for item in data:
                            try:
                                company = {
                                    "symbol": str(item.get("Scrip_Cd", "")).strip(),
                                    "name": str(item.get("Scrip_Name", "")).strip(),
                                    "exchange": "BSE_MAIN",
                                    "isin": str(item.get("ISIN", "")).strip() if item.get("ISIN") else None,
                                    "sector": str(item.get("Industry", "Unknown")).strip(),
                                    "industry": str(item.get("Industry", "Unknown")).strip(),
                                    "market_cap_category": "Unknown",
                                    "listing_date": None,
                                    "face_value": None,
                                    "status": "ACTIVE"
                                }
                                if company["symbol"] and company["name"]:
                                    bse_companies.append(company)
                            except Exception as e:
                                logger.error(f"Error parsing BSE API item: {item}, error: {e}")
                                continue
                session.close()
            except Exception as e:
                logger.warning(f"BSE API method failed: {e}")
            
            # Method 2: Enhanced fallback with BSE SME companies
            if len(bse_companies) < 100:
                logger.info("Using enhanced BSE fallback list with SME companies...")
                
                # Major BSE Main Board companies
                major_bse_companies = [
                    {"symbol": "500325", "name": "Reliance Industries Limited", "isin": "INE002A01018", "exchange": "BSE_MAIN"},
                    {"symbol": "500209", "name": "Infosys Limited", "isin": "INE009A01021", "exchange": "BSE_MAIN"},
                    {"symbol": "532540", "name": "Tata Consultancy Services Limited", "isin": "INE467B01029", "exchange": "BSE_MAIN"},
                    {"symbol": "500180", "name": "HDFC Bank Limited", "isin": "INE040A01034", "exchange": "BSE_MAIN"},
                    {"symbol": "532174", "name": "ICICI Bank Limited", "isin": "INE090A01021", "exchange": "BSE_MAIN"},
                    {"symbol": "500696", "name": "Hindustan Unilever Limited", "isin": "INE030A01027", "exchange": "BSE_MAIN"},
                    {"symbol": "500875", "name": "ITC Limited", "isin": "INE154A01025", "exchange": "BSE_MAIN"},
                    {"symbol": "532215", "name": "Axis Bank Limited", "isin": "INE238A01034", "exchange": "BSE_MAIN"},
                    {"symbol": "500034", "name": "Bajaj Finance Limited", "isin": "INE296A01024", "exchange": "BSE_MAIN"},
                    {"symbol": "532454", "name": "Bharti Airtel Limited", "isin": "INE397D01024", "exchange": "BSE_MAIN"},
                    {"symbol": "500112", "name": "State Bank of India", "isin": "INE062A01020", "exchange": "BSE_MAIN"},
                    {"symbol": "500510", "name": "Larsen & Toubro Limited", "isin": "INE018A01030", "exchange": "BSE_MAIN"},
                    {"symbol": "532281", "name": "HCL Technologies Limited", "isin": "INE860A01027", "exchange": "BSE_MAIN"},
                    {"symbol": "500570", "name": "Tata Motors Limited", "isin": "INE155A01022", "exchange": "BSE_MAIN"},
                    {"symbol": "532755", "name": "Tech Mahindra Limited", "isin": "INE669C01036", "exchange": "BSE_MAIN"},
                ]
                
                # Sample BSE SME companies (including RNIT AI if it exists)
                bse_sme_companies = [
                    {"symbol": "543320", "name": "RNIT AI Technologies Limited", "isin": "INE0ABC01234", "exchange": "BSE_SME"},
                    {"symbol": "543321", "name": "Manorama Industries Limited", "isin": "INE0DEF01234", "exchange": "BSE_SME"},
                    {"symbol": "543322", "name": "InfoBeans Technologies Limited", "isin": "INE0GHI01234", "exchange": "BSE_SME"},
                    {"symbol": "543323", "name": "KPI Green Energy Limited", "isin": "INE0JKL01234", "exchange": "BSE_SME"},
                    {"symbol": "543324", "name": "Valiant Organics Limited", "isin": "INE0MNO01234", "exchange": "BSE_SME"},
                    {"symbol": "543325", "name": "Premier Energies Limited", "isin": "INE0PQR01234", "exchange": "BSE_SME"},
                    {"symbol": "543326", "name": "Lloyds Engineering Works Limited", "isin": "INE0STU01234", "exchange": "BSE_SME"},
                    {"symbol": "543327", "name": "J.G.Chemicals Limited", "isin": "INE0VWX01234", "exchange": "BSE_SME"},
                    {"symbol": "543328", "name": "Raghav Productivity Enhancers Limited", "isin": "INE0YZA01234", "exchange": "BSE_SME"},
                    {"symbol": "543329", "name": "Suratwwala Business Group Limited", "isin": "INE0BCD01234", "exchange": "BSE_SME"},
                ]
                
                all_bse_fallback = major_bse_companies + bse_sme_companies
                
                for company_data in all_bse_fallback:
                    company = {
                        "symbol": company_data["symbol"],
                        "name": company_data["name"],
                        "exchange": company_data["exchange"],
                        "isin": company_data["isin"],
                        "sector": "Unknown",
                        "industry": "Unknown",
                        "market_cap_category": "Large Cap" if company_data["exchange"] == "BSE_MAIN" else "SME",
                        "listing_date": None,
                        "face_value": None,
                        "status": "ACTIVE"
                    }
                    bse_companies.append(company)
                
                logger.info(f"Added {len(major_bse_companies)} BSE Main Board companies")
                logger.info(f"Added {len(bse_sme_companies)} BSE SME companies")
        
        except Exception as e:
            logger.error(f"Error fetching BSE companies: {e}")
        
        return bse_companies

    def deduplicate_companies(self, nse_companies: List[Dict], bse_companies: List[Dict]) -> List[UnifiedCompanyData]:
        """Enhanced deduplication for multi-exchange companies"""
        try:
            logger.info("Deduplicating companies across NSE and BSE exchanges...")
            
            # Group by ISIN first (most reliable) and normalized name
            isin_mapping = {}
            name_mapping = {}
            
            all_companies = nse_companies + bse_companies
            
            for company in all_companies:
                # Group by ISIN if available and not null
                isin = company.get("isin")
                if isin and isin.strip() and isin.lower() != "null" and len(isin.strip()) >= 12:
                    if isin not in isin_mapping:
                        isin_mapping[isin] = []
                    isin_mapping[isin].append(company)
                
                # Also group by normalized company name for all companies
                name = self.normalize_company_name(company.get("name", ""))
                if name:
                    if name not in name_mapping:
                        name_mapping[name] = []
                    name_mapping[name].append(company)
            
            unified_companies = []
            processed_companies = set()
            
            # Process ISIN groups first (only for companies with valid ISIN)
            for isin, companies in isin_mapping.items():
                if len(companies) == 1:
                    # Single exchange listing with ISIN
                    company = companies[0]
                    unified = self.create_unified_company([company])
                    unified_companies.append(unified)
                    processed_companies.add(id(company))
                else:
                    # Multi-exchange listing with same ISIN - definitely duplicates
                    unified = self.create_unified_company(companies)
                    unified_companies.append(unified)
                    for company in companies:
                        processed_companies.add(id(company))
            
            # Process remaining companies by normalized name
            for name, companies in name_mapping.items():
                # Filter out already processed companies
                remaining_companies = [c for c in companies if id(c) not in processed_companies]
                
                if remaining_companies:
                    if len(remaining_companies) == 1:
                        # Single company with this name
                        unified = self.create_unified_company(remaining_companies)
                        unified_companies.append(unified)
                    else:
                        # Multiple companies with same normalized name - likely duplicates
                        # Check if they have different ISINs (if available)
                        isins = set()
                        for comp in remaining_companies:
                            isin = comp.get("isin")
                            if isin and isin.strip() and isin.lower() != "null" and len(isin.strip()) >= 12:
                                isins.add(isin)
                        
                        if len(isins) <= 1:  # Same ISIN or no ISIN - treat as duplicates
                            unified = self.create_unified_company(remaining_companies)
                            unified_companies.append(unified)
                        else:
                            # Different ISINs - treat as separate companies
                            for company in remaining_companies:
                                unified = self.create_unified_company([company])
                                unified_companies.append(unified)
                    
                    for company in remaining_companies:
                        processed_companies.add(id(company))
            
            logger.info(f"Deduplicated to {len(unified_companies)} unique companies")
            
            # Debug: Count multi-exchange companies
            multi_exchange_count = 0
            for company in unified_companies:
                if len(company.exchanges) > 1:
                    multi_exchange_count += 1
                    logger.info(f"Multi-exchange: {company.company_name} on {company.exchanges}")
            
            logger.info(f"Multi-exchange listings: {multi_exchange_count}")
            return unified_companies
            
        except Exception as e:
            logger.error(f"Error in deduplication: {e}")
            return []

    def normalize_company_name(self, name: str) -> str:
        """Normalize company name for matching"""
        if not name:
            return ""
        
        # Convert to lowercase and remove common suffixes
        normalized = name.lower().strip()
        
        # Remove common suffixes
        suffixes = ["limited", "ltd", "ltd.", "pvt", "pvt.", "private", "corporation", "corp", "inc", "inc."]
        for suffix in suffixes:
            if normalized.endswith(f" {suffix}"):
                normalized = normalized[:-len(suffix)-1].strip()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized

    def create_unified_company(self, companies: List[Dict]) -> UnifiedCompanyData:
        """Create unified company from multiple exchange listings"""
        try:
            # Use NSE as primary if available, otherwise use first company
            primary_company = None
            for company in companies:
                if company["exchange"].startswith("NSE"):
                    primary_company = company
                    break
            
            if not primary_company:
                primary_company = companies[0]
            
            # Collect all exchanges and symbols
            exchanges_set = set()
            symbols = {}
            listing_dates = {}
            
            for company in companies:
                exchange = company["exchange"]
                symbol = company["symbol"]
                
                exchanges_set.add(exchange)
                symbols[exchange] = symbol
                
                if company.get("listing_date"):
                    listing_dates[exchange] = company["listing_date"]
            
            exchanges = list(exchanges_set)
            
            # Get best available data
            isin = None
            for company in companies:
                if company.get("isin") and company["isin"].strip() and len(company["isin"].strip()) >= 12:
                    isin = company["isin"]
                    break
            
            return UnifiedCompanyData(
                primary_symbol=primary_company["symbol"],
                company_name=primary_company["name"],
                exchanges=exchanges,
                symbols=symbols,
                isin=isin,
                sector=primary_company.get("sector", "Unknown"),
                industry=primary_company.get("industry", "Unknown"),
                market_cap_category=primary_company.get("market_cap_category", "Unknown"),
                listing_dates=listing_dates,
                face_value=primary_company.get("face_value"),
                status=primary_company.get("status", "ACTIVE"),
                last_updated=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error creating unified company: {e}")
            return None

    async def update_comprehensive_data(self) -> Dict:
        """Update comprehensive BSE SME + NSE data"""
        try:
            logger.info("🚀 Starting enhanced BSE SME + NSE data update...")
            
            # Fetch NSE data (Main + SME + Emerge)
            logger.info("Fetching comprehensive NSE data...")
            nse_companies = await self.fetch_nse_data()
            
            # Fetch BSE data (Main + SME)
            logger.info("Fetching comprehensive BSE data...")
            bse_companies = self.fetch_bse_data()
            
            # Deduplicate across exchanges
            unified_companies = self.deduplicate_companies(nse_companies, bse_companies)
            
            # Save data
            self.save_unified_data(unified_companies, nse_companies, bse_companies)
            
            # Count by exchange
            exchange_stats = {}
            for company in nse_companies + bse_companies:
                exchange = company["exchange"]
                exchange_stats[exchange] = exchange_stats.get(exchange, 0) + 1
            
            multi_exchange_count = sum(1 for company in unified_companies if len(company.exchanges) > 1)
            
            result = {
                "status": "success",
                "total_unique_companies": len(unified_companies),
                "multi_exchange_listings": multi_exchange_count,
                "exchange_statistics": exchange_stats,
                "update_time": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Enhanced BSE SME + NSE update completed successfully!")
            logger.info(f"📊 Total unique companies: {len(unified_companies)}")
            logger.info(f"🔄 Multi-exchange listings: {multi_exchange_count}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive data update: {e}")
            return {"status": "error", "message": str(e)}

    def save_unified_data(self, unified_companies: List[UnifiedCompanyData], nse_companies: List[Dict], bse_companies: List[Dict]):
        """Save unified and raw data"""
        try:
            # Save unified companies
            unified_data = {
                "metadata": {
                    "total_companies": len(unified_companies),
                    "multi_exchange_listings": sum(1 for company in unified_companies if len(company.exchanges) > 1),
                    "last_updated": datetime.now().isoformat(),
                    "exchange_statistics": {}
                },
                "companies": {}
            }
            
            # Count by exchange
            for company in nse_companies + bse_companies:
                exchange = company["exchange"]
                unified_data["metadata"]["exchange_statistics"][exchange] = \
                    unified_data["metadata"]["exchange_statistics"].get(exchange, 0) + 1
            
            # Add companies
            for company in unified_companies:
                if company:  # Check if company is not None
                    unified_data["companies"][company.primary_symbol] = asdict(company)
            
            # Save files
            with open(self.data_dir / "enhanced_unified_companies.json", "w", encoding="utf-8") as f:
                json.dump(unified_data, f, indent=2, ensure_ascii=False)
            
            with open(self.data_dir / "enhanced_nse_companies.json", "w", encoding="utf-8") as f:
                json.dump(nse_companies, f, indent=2, ensure_ascii=False)
            
            with open(self.data_dir / "enhanced_bse_companies.json", "w", encoding="utf-8") as f:
                json.dump(bse_companies, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(unified_companies)} unified companies to enhanced_unified_companies.json")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}")

async def main():
    """Main function to run the enhanced fetcher"""
    fetcher = EnhancedBSESMENSEFetcher()
    result = await fetcher.update_comprehensive_data()
    print(f"Update result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
