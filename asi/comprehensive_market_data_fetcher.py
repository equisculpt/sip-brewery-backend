"""
Comprehensive Market Data Fetcher
Automatically fetches ALL companies from NSE Main Board, SME, Emerge and ALL mutual funds
Maintains JSON database with auto-update capabilities for ASI
"""

import asyncio
import aiohttp
import json
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompanyData:
    """Company data structure"""
    symbol: str
    name: str
    exchange: str  # NSE_MAIN, NSE_SME, NSE_EMERGE, BSE_MAIN, BSE_SME
    exchanges: List[str]  # All exchanges where company is listed
    sector: str
    industry: str
    market_cap_category: str
    listing_date: Optional[str]
    isin: Optional[str]
    face_value: Optional[float]
    status: str  # ACTIVE, DELISTED, SUSPENDED
    last_updated: str

@dataclass
class MutualFundData:
    """Mutual fund data structure"""
    scheme_code: str
    scheme_name: str
    amc_name: str
    category: str
    sub_category: str
    nav: Optional[float]
    nav_date: Optional[str]
    launch_date: Optional[str]
    status: str  # ACTIVE, CLOSED
    last_updated: str

class ComprehensiveMarketDataFetcher:
    """Fetches and maintains comprehensive market data"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.data_dir = Path("market_data")
        self.data_dir.mkdir(exist_ok=True)
        
        # Data files
        self.companies_file = self.data_dir / "all_companies.json"
        self.mutual_funds_file = self.data_dir / "all_mutual_funds.json"
        self.metadata_file = self.data_dir / "metadata.json"
        
        # NSE URLs
        self.nse_urls = {
            "main_board": "https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%20500",
            "sme": "https://www.nseindia.com/api/emerge-sme-equity",
            "emerge": "https://www.nseindia.com/api/emerge-equity",
            "all_securities": "https://www.nseindia.com/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O",
            "equity_list": "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
        }
        
        # BSE URLs
        self.bse_urls = {
            "main_board": "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w",
            "equity_list": "https://www.bseindia.com/corporates/List_Scrips.aspx",
            "sme": "https://www.bseindia.com/static/markets/equity/EQReports/ListOfScrips.aspx?expandable=3",
            "scrip_master": "https://www.bseindia.com/download/BhavCopy/Equity/BSE_EQ_BHAVCOPY.ZIP"
        }
        
        # AMFI URLs for mutual funds
        self.amfi_urls = {
            "nav_data": "https://www.amfiindia.com/spages/NAVAll.txt",
            "scheme_master": "https://www.amfiindia.com/spages/AMFISchemeCodeMaster.txt"
        }
        
        # Headers to bypass anti-bot protection
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def fetch_nse_data(self, url: str, retries: int = 3) -> Optional[Dict]:
        """Fetch data from NSE with retry logic"""
        for attempt in range(retries):
            try:
                # Add NSE session cookies
                cookies = {
                    'nsit': 'session_value',
                    'nseappid': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9'
                }
                
                async with self.session.get(url, cookies=cookies) as response:
                    if response.status == 200:
                        if 'application/json' in response.headers.get('content-type', ''):
                            return await response.json()
                        else:
                            text = await response.text()
                            return {"raw_data": text}
                    else:
                        logger.warning(f"NSE API returned status {response.status} for {url}")
                        
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    
        return None

    async def fetch_all_nse_companies(self) -> List[CompanyData]:
        """Fetch all companies from NSE Main Board, SME, and Emerge"""
        all_companies = []
        
        try:
            # Fetch NSE equity list (most comprehensive)
            logger.info("Fetching NSE equity list...")
            equity_data = await self.fetch_nse_data(self.nse_urls["equity_list"])
            
            if equity_data and "raw_data" in equity_data:
                # Parse CSV data
                lines = equity_data["raw_data"].strip().split('\n')
                headers = lines[0].split(',')
                
                for line in lines[1:]:
                    try:
                        values = line.split(',')
                        if len(values) >= 2:
                            symbol = values[0].strip().strip('"')
                            name = values[1].strip().strip('"')
                            
                            # Determine exchange based on symbol patterns
                            exchange = "NSE_MAIN"
                            if symbol.endswith("-SM"):
                                exchange = "NSE_SME"
                            elif symbol.endswith("-EM"):
                                exchange = "NSE_EMERGE"
                            
                            company = CompanyData(
                                symbol=symbol,
                                name=name,
                                exchange=exchange,
                                exchanges=[exchange],
                                sector="Unknown",
                                industry="Unknown",
                                market_cap_category="Unknown",
                                listing_date=None,
                                isin=values[2].strip().strip('"') if len(values) > 2 else None,
                                face_value=None,
                                status="ACTIVE",
                                last_updated=datetime.now().isoformat()
                            )
                            all_companies.append(company)
                            
                    except Exception as e:
                        logger.error(f"Error parsing company line: {line}, error: {e}")
                        continue
            
            # Fetch additional data from other NSE APIs
            for api_name, url in self.nse_urls.items():
                if api_name == "equity_list":
                    continue
                    
                logger.info(f"Fetching {api_name} data...")
                data = await self.fetch_nse_data(url)
                
                if data and "data" in data:
                    for item in data["data"]:
                        try:
                            symbol = item.get("symbol", "")
                            name = item.get("companyName", item.get("name", ""))
                            
                            if symbol and name:
                                # Check if company already exists
                                existing = next((c for c in all_companies if c.symbol == symbol), None)
                                if not existing:
                                    exchange = "NSE_MAIN"
                                    if api_name == "sme":
                                        exchange = "NSE_SME"
                                    elif api_name == "emerge":
                                        exchange = "NSE_EMERGE"
                                    
                                    company = CompanyData(
                                        symbol=symbol,
                                        name=name,
                                        exchange=exchange,
                                        sector=item.get("sector", "Unknown"),
                                        industry=item.get("industry", "Unknown"),
                                        market_cap_category="Unknown",
                                        listing_date=None,
                                        isin=item.get("isin"),
                                        face_value=item.get("faceValue"),
                                        status="ACTIVE",
                                        last_updated=datetime.now().isoformat()
                                    )
                                    all_companies.append(company)
                                else:
                                    # Update existing company with additional data
                                    if item.get("sector") and item["sector"] != "Unknown":
                                        existing.sector = item["sector"]
                                    if item.get("industry") and item["industry"] != "Unknown":
                                        existing.industry = item["industry"]
                                    if item.get("isin"):
                                        existing.isin = item["isin"]
                                    if item.get("faceValue"):
                                        existing.face_value = item["faceValue"]
                                        
                        except Exception as e:
                            logger.error(f"Error processing {api_name} item: {item}, error: {e}")
                            continue
                
                # Add delay between API calls
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Error fetching NSE companies: {e}")
        
        logger.info(f"Fetched {len(all_companies)} companies from NSE")
        return all_companies

    async def fetch_all_mutual_funds(self) -> List[MutualFundData]:
        """Fetch all mutual funds from AMFI"""
        all_funds = []
        
        try:
            # Fetch NAV data
            logger.info("Fetching AMFI NAV data...")
            nav_response = await self.fetch_nse_data(self.amfi_urls["nav_data"])
            
            if nav_response and "raw_data" in nav_response:
                lines = nav_response["raw_data"].strip().split('\n')
                current_amc = ""
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this is an AMC header
                    if line.isupper() and not ';' in line and len(line) > 10:
                        current_amc = line
                        continue
                    
                    # Parse fund data
                    if ';' in line:
                        try:
                            parts = line.split(';')
                            if len(parts) >= 6:
                                scheme_code = parts[0].strip()
                                scheme_name = parts[3].strip()
                                nav = float(parts[4].strip()) if parts[4].strip() else None
                                nav_date = parts[5].strip() if parts[5].strip() else None
                                
                                # Categorize fund
                                category = "Unknown"
                                sub_category = "Unknown"
                                
                                name_lower = scheme_name.lower()
                                if "equity" in name_lower or "growth" in name_lower:
                                    category = "Equity"
                                    if "large cap" in name_lower or "blue chip" in name_lower:
                                        sub_category = "Large Cap"
                                    elif "mid cap" in name_lower:
                                        sub_category = "Mid Cap"
                                    elif "small cap" in name_lower:
                                        sub_category = "Small Cap"
                                    elif "flexi cap" in name_lower or "multi cap" in name_lower:
                                        sub_category = "Flexi Cap"
                                elif "debt" in name_lower or "income" in name_lower or "bond" in name_lower:
                                    category = "Debt"
                                elif "hybrid" in name_lower or "balanced" in name_lower:
                                    category = "Hybrid"
                                elif "liquid" in name_lower:
                                    category = "Liquid"
                                elif "index" in name_lower:
                                    category = "Index"
                                elif "etf" in name_lower:
                                    category = "ETF"
                                
                                fund = MutualFundData(
                                    scheme_code=scheme_code,
                                    scheme_name=scheme_name,
                                    amc_name=current_amc,
                                    category=category,
                                    sub_category=sub_category,
                                    nav=nav,
                                    nav_date=nav_date,
                                    launch_date=None,
                                    status="ACTIVE",
                                    last_updated=datetime.now().isoformat()
                                )
                                all_funds.append(fund)
                                
                        except Exception as e:
                            logger.error(f"Error parsing fund line: {line}, error: {e}")
                            continue
            
            # Fetch scheme master for additional details
            logger.info("Fetching AMFI scheme master...")
            master_response = await self.fetch_nse_data(self.amfi_urls["scheme_master"])
            
            if master_response and "raw_data" in master_response:
                lines = master_response["raw_data"].strip().split('\n')
                
                for line in lines[1:]:  # Skip header
                    try:
                        parts = line.split(';')
                        if len(parts) >= 4:
                            scheme_code = parts[0].strip()
                            launch_date = parts[3].strip() if len(parts) > 3 else None
                            
                            # Update existing fund with launch date
                            existing_fund = next((f for f in all_funds if f.scheme_code == scheme_code), None)
                            if existing_fund and launch_date:
                                existing_fund.launch_date = launch_date
                                
                    except Exception as e:
                        logger.error(f"Error parsing master line: {line}, error: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error fetching mutual funds: {e}")
        
        logger.info(f"Fetched {len(all_funds)} mutual funds from AMFI")
        return all_funds

    def save_companies_to_json(self, companies: List[CompanyData]):
        """Save companies to JSON file"""
        try:
            companies_dict = {
                "metadata": {
                    "total_companies": len(companies),
                    "last_updated": datetime.now().isoformat(),
                    "exchanges": {
                        "NSE_MAIN": len([c for c in companies if c.exchange == "NSE_MAIN"]),
                        "NSE_SME": len([c for c in companies if c.exchange == "NSE_SME"]),
                        "NSE_EMERGE": len([c for c in companies if c.exchange == "NSE_EMERGE"])
                    }
                },
                "companies": {company.symbol: asdict(company) for company in companies}
            }
            
            with open(self.companies_file, 'w', encoding='utf-8') as f:
                json.dump(companies_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(companies)} companies to {self.companies_file}")
            
        except Exception as e:
            logger.error(f"Error saving companies to JSON: {e}")

    def save_mutual_funds_to_json(self, funds: List[MutualFundData]):
        """Save mutual funds to JSON file"""
        try:
            funds_dict = {
                "metadata": {
                    "total_funds": len(funds),
                    "last_updated": datetime.now().isoformat(),
                    "categories": {}
                },
                "mutual_funds": {fund.scheme_code: asdict(fund) for fund in funds}
            }
            
            # Count by category
            for fund in funds:
                category = fund.category
                if category not in funds_dict["metadata"]["categories"]:
                    funds_dict["metadata"]["categories"][category] = 0
                funds_dict["metadata"]["categories"][category] += 1
            
            with open(self.mutual_funds_file, 'w', encoding='utf-8') as f:
                json.dump(funds_dict, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Saved {len(funds)} mutual funds to {self.mutual_funds_file}")
            
        except Exception as e:
            logger.error(f"Error saving mutual funds to JSON: {e}")

    def load_existing_data(self) -> tuple[List[CompanyData], List[MutualFundData]]:
        """Load existing data from JSON files"""
        companies = []
        funds = []
        
        try:
            if self.companies_file.exists():
                with open(self.companies_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for symbol, company_data in data.get("companies", {}).items():
                        companies.append(CompanyData(**company_data))
                        
            if self.mutual_funds_file.exists():
                with open(self.mutual_funds_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for code, fund_data in data.get("mutual_funds", {}).items():
                        funds.append(MutualFundData(**fund_data))
                        
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            
        return companies, funds

    async def update_database(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Update the complete database"""
        logger.info("🚀 Starting comprehensive market data update...")
        
        # Load existing data
        existing_companies, existing_funds = self.load_existing_data()
        
        # Check if we need to update
        if not force_refresh and existing_companies and existing_funds:
            # Check last update time
            try:
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                    last_update = datetime.fromisoformat(metadata.get("last_full_update", "2000-01-01"))
                    if datetime.now() - last_update < timedelta(hours=24):
                        logger.info("Data is recent, skipping full update")
                        return {
                            "status": "skipped",
                            "reason": "Recent data available",
                            "companies_count": len(existing_companies),
                            "funds_count": len(existing_funds)
                        }
            except:
                pass
        
        # Fetch fresh data
        companies = await self.fetch_all_nse_companies()
        funds = await self.fetch_all_mutual_funds()
        
        # Save to JSON files
        self.save_companies_to_json(companies)
        self.save_mutual_funds_to_json(funds)
        
        # Update metadata
        metadata = {
            "last_full_update": datetime.now().isoformat(),
            "companies_count": len(companies),
            "funds_count": len(funds),
            "exchanges": {
                "NSE_MAIN": len([c for c in companies if c.exchange == "NSE_MAIN"]),
                "NSE_SME": len([c for c in companies if c.exchange == "NSE_SME"]),
                "NSE_EMERGE": len([c for c in companies if c.exchange == "NSE_EMERGE"])
            },
            "fund_categories": {}
        }
        
        # Count fund categories
        for fund in funds:
            category = fund.category
            if category not in metadata["fund_categories"]:
                metadata["fund_categories"][category] = 0
            metadata["fund_categories"][category] += 1
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"✅ Database update complete: {len(companies)} companies, {len(funds)} funds")
        
        return {
            "status": "success",
            "companies_count": len(companies),
            "funds_count": len(funds),
            "new_companies": len(companies) - len(existing_companies),
            "new_funds": len(funds) - len(existing_funds),
            "metadata": metadata
        }

async def main():
    """Main function for testing"""
    async with ComprehensiveMarketDataFetcher() as fetcher:
        result = await fetcher.update_database(force_refresh=True)
        print(f"Update result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
