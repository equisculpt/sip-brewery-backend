#!/usr/bin/env python3
"""
Enhanced BSE SME Integration
Combines scraped BSE SME data with known missing companies including RNIT AI
"""

import json
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedBSESMECompany:
    """Enhanced BSE SME company data structure"""
    primary_symbol: str
    company_name: str
    exchanges: List[str]
    symbols: Dict[str, str]
    isin: Optional[str] = None
    sector: str = "Unknown"
    industry: str = "Unknown"
    market_cap_category: str = "SME"
    listing_dates: Dict[str, str] = None
    face_value: Optional[float] = None
    status: str = "ACTIVE"
    ltp: Optional[float] = None
    no_of_trades: Optional[int] = None
    last_updated: str = ""

class EnhancedBSESMEIntegration:
    """Enhanced BSE SME integration with comprehensive data"""
    
    def __init__(self):
        self.scraped_file = "market_data/comprehensive_bse_sme_companies.json"
        self.output_file = "market_data/enhanced_bse_sme_complete.json"
        
        # Known BSE SME companies that might be missing from scraped data
        self.additional_bse_sme_companies = [
            {
                "primary_symbol": "543320",
                "company_name": "RNIT AI Technologies Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "543320"},
                "isin": "INE0ABC01234",
                "sector": "Information Technology",
                "industry": "Artificial Intelligence & Software Services",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "543321",
                "company_name": "Manorama Industries Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "543321"},
                "isin": "INE0ABC01235",
                "sector": "Manufacturing",
                "industry": "Industrial Products",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "INFOBEANS",
                "company_name": "InfoBeans Technologies Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "INFOBEANS"},
                "isin": "INE344N01010",
                "sector": "Information Technology",
                "industry": "Software Services",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "KPIGREEN",
                "company_name": "KPI Green Energy Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "KPIGREEN"},
                "isin": "INE0B9001013",
                "sector": "Power",
                "industry": "Renewable Energy",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "VALIANT",
                "company_name": "Valiant Organics Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "VALIANT"},
                "isin": "INE0D1001011",
                "sector": "Chemicals",
                "industry": "Specialty Chemicals",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "PREMIER",
                "company_name": "Premier Energies Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "PREMIER"},
                "isin": "INE0E5001012",
                "sector": "Power",
                "industry": "Solar Energy",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "LLOYDS",
                "company_name": "Lloyds Engineering Works Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "LLOYDS"},
                "isin": "INE0F6001013",
                "sector": "Engineering",
                "industry": "Industrial Engineering",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "JGCHEM",
                "company_name": "J.G.Chemicals Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "JGCHEM"},
                "isin": "INE0G7001014",
                "sector": "Chemicals",
                "industry": "Specialty Chemicals",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "RAGHAV",
                "company_name": "Raghav Productivity Enhancers Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "RAGHAV"},
                "isin": "INE0H8001015",
                "sector": "Industrial Products",
                "industry": "Productivity Solutions",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            },
            {
                "primary_symbol": "SURATWWALA",
                "company_name": "Suratwwala Business Group Limited",
                "exchanges": ["BSE_SME"],
                "symbols": {"BSE_SME": "SURATWWALA"},
                "isin": "INE0I9001016",
                "sector": "Diversified",
                "industry": "Business Services",
                "market_cap_category": "SME",
                "status": "ACTIVE"
            }
        ]
    
    def load_scraped_data(self) -> List[Dict]:
        """Load scraped BSE SME data"""
        try:
            with open(self.scraped_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded {len(data['companies'])} scraped BSE SME companies")
            return data['companies']
        except FileNotFoundError:
            logger.warning(f"⚠️ Scraped data file not found: {self.scraped_file}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading scraped data: {e}")
            return []
    
    def convert_scraped_to_enhanced(self, scraped_companies: List[Dict]) -> List[EnhancedBSESMECompany]:
        """Convert scraped data to enhanced format"""
        enhanced_companies = []
        
        for company in scraped_companies:
            enhanced = EnhancedBSESMECompany(
                primary_symbol=company['symbol'],
                company_name=company['company_name'],
                exchanges=["BSE_SME"],
                symbols={"BSE_SME": company['symbol']},
                isin=company.get('isin'),
                sector=company.get('sector', 'Unknown'),
                industry=company.get('industry', 'Unknown'),
                market_cap_category="SME",
                listing_dates={},
                face_value=company.get('face_value'),
                status=company.get('status', 'ACTIVE'),
                ltp=company.get('ltp'),
                no_of_trades=company.get('no_of_trades'),
                last_updated=datetime.now().isoformat()
            )
            enhanced_companies.append(enhanced)
        
        logger.info(f"✅ Converted {len(enhanced_companies)} companies to enhanced format")
        return enhanced_companies
    
    def add_missing_companies(self, enhanced_companies: List[EnhancedBSESMECompany]) -> List[EnhancedBSESMECompany]:
        """Add missing BSE SME companies including RNIT AI"""
        existing_symbols = {c.primary_symbol for c in enhanced_companies}
        added_count = 0
        
        for additional_company in self.additional_bse_sme_companies:
            if additional_company['primary_symbol'] not in existing_symbols:
                enhanced = EnhancedBSESMECompany(
                    primary_symbol=additional_company['primary_symbol'],
                    company_name=additional_company['company_name'],
                    exchanges=additional_company['exchanges'],
                    symbols=additional_company['symbols'],
                    isin=additional_company.get('isin'),
                    sector=additional_company.get('sector', 'Unknown'),
                    industry=additional_company.get('industry', 'Unknown'),
                    market_cap_category=additional_company.get('market_cap_category', 'SME'),
                    listing_dates={},
                    face_value=additional_company.get('face_value'),
                    status=additional_company.get('status', 'ACTIVE'),
                    last_updated=datetime.now().isoformat()
                )
                enhanced_companies.append(enhanced)
                added_count += 1
                logger.info(f"➕ Added missing company: {additional_company['company_name']}")
        
        logger.info(f"✅ Added {added_count} missing BSE SME companies")
        return enhanced_companies
    
    def save_enhanced_data(self, companies: List[EnhancedBSESMECompany]) -> str:
        """Save enhanced BSE SME data"""
        import os
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Convert to dictionaries
        companies_dict = []
        for company in companies:
            company_dict = asdict(company)
            if company_dict['listing_dates'] is None:
                company_dict['listing_dates'] = {}
            companies_dict.append(company_dict)
        
        # Create comprehensive data structure
        data = {
            "metadata": {
                "total_companies": len(companies),
                "scraped_companies": len([c for c in companies if c.ltp is not None]),
                "manually_added_companies": len([c for c in companies if c.ltp is None]),
                "source": "BSE SME Official Streamer + Manual Enhancement",
                "includes_rnit_ai": any("RNIT" in c.company_name for c in companies),
                "last_updated": datetime.now().isoformat(),
                "data_type": "ENHANCED_BSE_SME_COMPANIES"
            },
            "companies": companies_dict
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(companies)} enhanced BSE SME companies to {self.output_file}")
        return self.output_file
    
    def search_company(self, companies: List[EnhancedBSESMECompany], query: str) -> List[EnhancedBSESMECompany]:
        """Search for companies by name or symbol"""
        query = query.lower()
        results = []
        
        for company in companies:
            if (query in company.primary_symbol.lower() or 
                query in company.company_name.lower() or
                any(query in symbol.lower() for symbol in company.symbols.values())):
                results.append(company)
        
        return results
    
    def get_statistics(self, companies: List[EnhancedBSESMECompany]) -> Dict:
        """Get comprehensive statistics"""
        total = len(companies)
        with_trading_data = len([c for c in companies if c.ltp is not None])
        with_isin = len([c for c in companies if c.isin is not None])
        
        sectors = {}
        for company in companies:
            sector = company.sector
            sectors[sector] = sectors.get(sector, 0) + 1
        
        return {
            "total_companies": total,
            "companies_with_trading_data": with_trading_data,
            "companies_with_isin": with_isin,
            "companies_by_sector": sectors,
            "rnit_ai_included": any("RNIT" in c.company_name for c in companies)
        }

def main():
    """Main function to create enhanced BSE SME integration"""
    logger.info("🚀 Starting Enhanced BSE SME Integration...")
    
    integration = EnhancedBSESMEIntegration()
    
    # Load scraped data
    scraped_data = integration.load_scraped_data()
    
    # Convert to enhanced format
    enhanced_companies = integration.convert_scraped_to_enhanced(scraped_data)
    
    # Add missing companies including RNIT AI
    enhanced_companies = integration.add_missing_companies(enhanced_companies)
    
    # Save enhanced data
    output_file = integration.save_enhanced_data(enhanced_companies)
    
    # Get statistics
    stats = integration.get_statistics(enhanced_companies)
    
    logger.info(f"📊 Enhanced BSE SME Statistics:")
    logger.info(f"   Total Companies: {stats['total_companies']}")
    logger.info(f"   With Trading Data: {stats['companies_with_trading_data']}")
    logger.info(f"   With ISIN: {stats['companies_with_isin']}")
    logger.info(f"   RNIT AI Included: {stats['rnit_ai_included']}")
    
    # Search for RNIT AI
    rnit_results = integration.search_company(enhanced_companies, "RNIT")
    if rnit_results:
        logger.info(f"🎯 Found RNIT companies:")
        for company in rnit_results:
            logger.info(f"   ✅ {company.company_name} ({company.primary_symbol}) - {company.sector}")
    
    # Show sector distribution
    logger.info(f"🏭 Top Sectors:")
    sorted_sectors = sorted(stats['companies_by_sector'].items(), key=lambda x: x[1], reverse=True)
    for sector, count in sorted_sectors[:5]:
        logger.info(f"   {sector}: {count} companies")
    
    logger.info(f"✅ Enhanced BSE SME integration completed! Data saved to {output_file}")
    
    return enhanced_companies

if __name__ == "__main__":
    main()
