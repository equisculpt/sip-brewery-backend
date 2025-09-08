#!/usr/bin/env python3
"""
Corrected Comprehensive Integration
NSE (Main + SME + Emerge) + BSE Main Board (including RNIT AI) + BSE SME
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
class UnifiedCompany:
    """Unified company data structure"""
    primary_symbol: str
    company_name: str
    exchanges: List[str]
    symbols: Dict[str, str]
    isin: Optional[str] = None
    sector: str = "Unknown"
    industry: str = "Unknown"
    market_cap_category: str = "Unknown"
    listing_dates: Dict[str, str] = None
    face_value: Optional[float] = None
    status: str = "ACTIVE"
    last_updated: str = ""

class CorrectedComprehensiveIntegration:
    """Corrected comprehensive integration with proper BSE Main Board data"""
    
    def __init__(self):
        self.nse_file = "market_data/enhanced_nse_companies.json"
        self.bse_main_file = "market_data/comprehensive_bse_main_board_companies.json"
        self.bse_sme_file = "market_data/comprehensive_bse_sme_companies.json"
        self.output_file = "market_data/corrected_unified_companies.json"
    
    def load_nse_data(self) -> List[Dict]:
        """Load NSE data (Main + SME + Emerge)"""
        try:
            with open(self.nse_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Handle both array format and metadata format
            if isinstance(data, list):
                companies = data
            elif isinstance(data, dict) and 'companies' in data:
                companies = data['companies']
            else:
                companies = []
            logger.info(f"✅ Loaded {len(companies)} NSE companies")
            return companies
        except FileNotFoundError:
            logger.warning(f"⚠️ NSE data file not found: {self.nse_file}")
            return []
    
    def load_bse_main_data(self) -> List[Dict]:
        """Load BSE Main Board data including RNIT AI"""
        try:
            with open(self.bse_main_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded {len(data['companies'])} BSE Main Board companies")
            logger.info(f"   RNIT AI Included: {data['metadata']['includes_rnit_ai']}")
            return data['companies']
        except FileNotFoundError:
            logger.warning(f"⚠️ BSE Main Board data file not found: {self.bse_main_file}")
            return []
    
    def load_bse_sme_data(self) -> List[Dict]:
        """Load BSE SME data"""
        try:
            with open(self.bse_sme_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"✅ Loaded {len(data['companies'])} BSE SME companies")
            return data['companies']
        except FileNotFoundError:
            logger.warning(f"⚠️ BSE SME data file not found: {self.bse_sme_file}")
            return []
    
    def convert_nse_to_unified(self, nse_companies: List[Dict]) -> List[UnifiedCompany]:
        """Convert NSE data to unified format"""
        unified_companies = []
        
        for company in nse_companies:
            # Handle different NSE data formats
            symbol = company.get('symbol', '')
            name = company.get('name', company.get('company_name', ''))
            exchange = company.get('exchange', 'NSE_MAIN')
            
            unified = UnifiedCompany(
                primary_symbol=symbol,
                company_name=name,
                exchanges=[exchange] if isinstance(exchange, str) else company.get('exchanges', [exchange]),
                symbols={exchange: symbol} if isinstance(exchange, str) else company.get('symbols', {exchange: symbol}),
                isin=company.get('isin'),
                sector=company.get('sector', 'Unknown'),
                industry=company.get('industry', 'Unknown'),
                market_cap_category=company.get('market_cap_category', 'Unknown'),
                listing_dates=company.get('listing_dates', {}),
                face_value=company.get('face_value'),
                status=company.get('status', 'ACTIVE'),
                last_updated=datetime.now().isoformat()
            )
            unified_companies.append(unified)
        
        logger.info(f"✅ Converted {len(unified_companies)} NSE companies to unified format")
        return unified_companies
    
    def convert_bse_main_to_unified(self, bse_companies: List[Dict]) -> List[UnifiedCompany]:
        """Convert BSE Main Board data to unified format"""
        unified_companies = []
        
        for company in bse_companies:
            unified = UnifiedCompany(
                primary_symbol=company['scrip_code'],
                company_name=company['company_name'],
                exchanges=["BSE_MAIN"],
                symbols={"BSE_MAIN": company['scrip_code']},
                isin=company.get('isin'),
                sector=company.get('sector', 'Unknown'),
                industry=company.get('industry', 'Unknown'),
                market_cap_category=company.get('market_cap_category', 'Unknown'),
                listing_dates={},
                face_value=company.get('face_value'),
                status=company.get('status', 'ACTIVE'),
                last_updated=datetime.now().isoformat()
            )
            unified_companies.append(unified)
        
        logger.info(f"✅ Converted {len(unified_companies)} BSE Main Board companies to unified format")
        return unified_companies
    
    def convert_bse_sme_to_unified(self, bse_sme_companies: List[Dict]) -> List[UnifiedCompany]:
        """Convert BSE SME data to unified format"""
        unified_companies = []
        
        for company in bse_sme_companies:
            unified = UnifiedCompany(
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
                last_updated=datetime.now().isoformat()
            )
            unified_companies.append(unified)
        
        logger.info(f"✅ Converted {len(unified_companies)} BSE SME companies to unified format")
        return unified_companies
    
    def normalize_company_name(self, name: str) -> str:
        """Normalize company name for deduplication"""
        if not name:
            return ""
        
        # Remove common suffixes and normalize
        suffixes = ['LIMITED', 'LTD', 'PRIVATE', 'PVT', 'COMPANY', 'CO', 'CORPORATION', 'CORP', 'INC']
        normalized = name.upper().strip()
        
        for suffix in suffixes:
            if normalized.endswith(f' {suffix}'):
                normalized = normalized[:-len(suffix)].strip()
        
        # Remove special characters and extra spaces
        normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def deduplicate_companies(self, all_companies: List[UnifiedCompany]) -> List[UnifiedCompany]:
        """Deduplicate companies across exchanges"""
        logger.info("🔄 Deduplicating companies across exchanges...")
        
        # Group companies by ISIN first (most reliable)
        isin_groups = {}
        no_isin_companies = []
        
        for company in all_companies:
            if company.isin and company.isin.strip():
                isin = company.isin.strip()
                if isin not in isin_groups:
                    isin_groups[isin] = []
                isin_groups[isin].append(company)
            else:
                no_isin_companies.append(company)
        
        # Merge companies with same ISIN
        deduplicated = []
        for isin, companies in isin_groups.items():
            if len(companies) == 1:
                deduplicated.append(companies[0])
            else:
                # Merge multiple companies with same ISIN
                primary = companies[0]
                for other in companies[1:]:
                    primary.exchanges.extend(other.exchanges)
                    primary.symbols.update(other.symbols)
                
                # Remove duplicates from exchanges
                primary.exchanges = list(set(primary.exchanges))
                
                logger.info(f"🔗 Merged {len(companies)} companies with ISIN {isin}: {primary.company_name}")
                deduplicated.append(primary)
        
        # Handle companies without ISIN - group by normalized name
        name_groups = {}
        for company in no_isin_companies:
            normalized_name = self.normalize_company_name(company.company_name)
            if normalized_name not in name_groups:
                name_groups[normalized_name] = []
            name_groups[normalized_name].append(company)
        
        # Merge companies with same normalized name
        for normalized_name, companies in name_groups.items():
            if len(companies) == 1:
                deduplicated.append(companies[0])
            else:
                # Merge multiple companies with same name
                primary = companies[0]
                for other in companies[1:]:
                    primary.exchanges.extend(other.exchanges)
                    primary.symbols.update(other.symbols)
                
                # Remove duplicates from exchanges
                primary.exchanges = list(set(primary.exchanges))
                
                logger.info(f"🔗 Merged {len(companies)} companies by name: {primary.company_name}")
                deduplicated.append(primary)
        
        logger.info(f"✅ Deduplication complete: {len(all_companies)} → {len(deduplicated)} companies")
        return deduplicated
    
    def save_unified_data(self, companies: List[UnifiedCompany]) -> str:
        """Save unified data"""
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
                "nse_companies": len([c for c in companies if any("NSE" in ex for ex in c.exchanges)]),
                "bse_main_companies": len([c for c in companies if "BSE_MAIN" in c.exchanges]),
                "bse_sme_companies": len([c for c in companies if "BSE_SME" in c.exchanges]),
                "multi_exchange_companies": len([c for c in companies if len(c.exchanges) > 1]),
                "source": "NSE (Main+SME+Emerge) + BSE Main Board + BSE SME",
                "includes_rnit_ai": any("RNIT" in c.company_name for c in companies),
                "last_updated": datetime.now().isoformat(),
                "data_type": "CORRECTED_UNIFIED_COMPANIES"
            },
            "companies": companies_dict
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(companies)} unified companies to {self.output_file}")
        return self.output_file
    
    def search_company(self, companies: List[UnifiedCompany], query: str) -> List[UnifiedCompany]:
        """Search for companies"""
        query = query.lower()
        results = []
        
        for company in companies:
            if (query in company.primary_symbol.lower() or 
                query in company.company_name.lower() or
                any(query in symbol.lower() for symbol in company.symbols.values())):
                results.append(company)
        
        return results
    
    def get_statistics(self, companies: List[UnifiedCompany]) -> Dict:
        """Get comprehensive statistics"""
        total = len(companies)
        nse_companies = len([c for c in companies if any("NSE" in ex for ex in c.exchanges)])
        bse_main_companies = len([c for c in companies if "BSE_MAIN" in c.exchanges])
        bse_sme_companies = len([c for c in companies if "BSE_SME" in c.exchanges])
        multi_exchange = len([c for c in companies if len(c.exchanges) > 1])
        
        exchanges = {}
        for company in companies:
            for exchange in company.exchanges:
                exchanges[exchange] = exchanges.get(exchange, 0) + 1
        
        return {
            "total_companies": total,
            "nse_companies": nse_companies,
            "bse_main_companies": bse_main_companies,
            "bse_sme_companies": bse_sme_companies,
            "multi_exchange_companies": multi_exchange,
            "companies_by_exchange": exchanges,
            "rnit_ai_included": any("RNIT" in c.company_name for c in companies)
        }

def main():
    """Main function to create corrected comprehensive integration"""
    logger.info("🚀 Starting Corrected Comprehensive Integration...")
    logger.info("   NSE (Main + SME + Emerge) + BSE Main Board + BSE SME")
    print()
    
    integration = CorrectedComprehensiveIntegration()
    
    # Load all data sources
    nse_data = integration.load_nse_data()
    bse_main_data = integration.load_bse_main_data()
    bse_sme_data = integration.load_bse_sme_data()
    
    # Convert to unified format
    all_companies = []
    
    if nse_data:
        nse_unified = integration.convert_nse_to_unified(nse_data)
        all_companies.extend(nse_unified)
    
    if bse_main_data:
        bse_main_unified = integration.convert_bse_main_to_unified(bse_main_data)
        all_companies.extend(bse_main_unified)
    
    if bse_sme_data:
        bse_sme_unified = integration.convert_bse_sme_to_unified(bse_sme_data)
        all_companies.extend(bse_sme_unified)
    
    logger.info(f"📊 Total companies before deduplication: {len(all_companies)}")
    
    # Deduplicate companies
    unified_companies = integration.deduplicate_companies(all_companies)
    
    # Save unified data
    output_file = integration.save_unified_data(unified_companies)
    
    # Get statistics
    stats = integration.get_statistics(unified_companies)
    
    logger.info(f"📊 Corrected Comprehensive Integration Statistics:")
    logger.info(f"   Total Unified Companies: {stats['total_companies']}")
    logger.info(f"   NSE Companies: {stats['nse_companies']}")
    logger.info(f"   BSE Main Board Companies: {stats['bse_main_companies']}")
    logger.info(f"   BSE SME Companies: {stats['bse_sme_companies']}")
    logger.info(f"   Multi-Exchange Companies: {stats['multi_exchange_companies']}")
    logger.info(f"   RNIT AI Included: {stats['rnit_ai_included']}")
    
    # Search for RNIT AI
    rnit_results = integration.search_company(unified_companies, "RNIT")
    if rnit_results:
        logger.info(f"🎯 Found RNIT companies:")
        for company in rnit_results:
            exchanges_str = ' + '.join(company.exchanges)
            logger.info(f"   ✅ {company.company_name} ({company.primary_symbol})")
            logger.info(f"      Exchanges: {exchanges_str}")
            logger.info(f"      Sector: {company.sector}")
            logger.info(f"      ISIN: {company.isin}")
    
    # Show exchange distribution
    logger.info(f"🏢 Companies by Exchange:")
    for exchange, count in stats['companies_by_exchange'].items():
        logger.info(f"   {exchange}: {count} companies")
    
    logger.info(f"✅ Corrected comprehensive integration completed! Data saved to {output_file}")
    
    return unified_companies

if __name__ == "__main__":
    main()
