#!/usr/bin/env python3
"""
Final Comprehensive Integration
NSE (Complete) + BSE (Available via API + Major Companies) + BSE SME
Acknowledges BSE API limitations but ensures comprehensive coverage
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalComprehensiveIntegration:
    """Final comprehensive integration with realistic BSE coverage"""
    
    def __init__(self):
        self.nse_file = "market_data/enhanced_nse_companies.json"
        self.bse_file = "market_data/complete_bse_companies.json"
        self.bse_sme_file = "market_data/comprehensive_bse_sme_companies.json"
        self.output_file = "market_data/final_unified_companies.json"
    
    def load_data_file(self, filename: str, data_type: str) -> List[Dict]:
        """Load data from JSON file with error handling"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle different file formats
            if isinstance(data, list):
                companies = data
            elif isinstance(data, dict):
                companies = data.get('companies', [])
            else:
                companies = []
            
            logger.info(f"✅ Loaded {len(companies)} {data_type} companies from {filename}")
            return companies
            
        except FileNotFoundError:
            logger.warning(f"⚠️ {data_type} file not found: {filename}")
            return []
        except Exception as e:
            logger.error(f"❌ Error loading {data_type} data: {e}")
            return []
    
    def convert_to_unified_format(self, companies: List[Dict], exchange_prefix: str) -> List[Dict]:
        """Convert companies to unified format"""
        unified_companies = []
        
        for company in companies:
            # Handle different data structures
            if exchange_prefix == "NSE":
                unified = {
                    "primary_symbol": company.get('symbol', ''),
                    "company_name": company.get('name', company.get('company_name', '')),
                    "exchanges": [company.get('exchange', 'NSE_MAIN')],
                    "symbols": {company.get('exchange', 'NSE_MAIN'): company.get('symbol', '')},
                    "isin": company.get('isin'),
                    "sector": company.get('sector', 'Unknown'),
                    "industry": company.get('industry', 'Unknown'),
                    "market_cap_category": company.get('market_cap_category', 'Unknown'),
                    "face_value": company.get('face_value'),
                    "status": company.get('status', 'ACTIVE'),
                    "last_updated": datetime.now().isoformat()
                }
            
            elif exchange_prefix == "BSE_MAIN":
                unified = {
                    "primary_symbol": company.get('scrip_code', ''),
                    "company_name": company.get('company_name', ''),
                    "exchanges": ["BSE_MAIN"],
                    "symbols": {"BSE_MAIN": company.get('scrip_code', '')},
                    "isin": company.get('isin'),
                    "sector": company.get('sector', 'Unknown'),
                    "industry": company.get('industry', 'Unknown'),
                    "market_cap_category": "Large Cap" if company.get('group') == 'A' else "Mid Cap",
                    "face_value": company.get('face_value'),
                    "status": company.get('status', 'ACTIVE'),
                    "last_updated": datetime.now().isoformat()
                }
            
            elif exchange_prefix == "BSE_SME":
                unified = {
                    "primary_symbol": company.get('symbol', company.get('scrip_code', '')),
                    "company_name": company.get('company_name', ''),
                    "exchanges": ["BSE_SME"],
                    "symbols": {"BSE_SME": company.get('symbol', company.get('scrip_code', ''))},
                    "isin": company.get('isin'),
                    "sector": company.get('sector', 'Unknown'),
                    "industry": company.get('industry', 'Unknown'),
                    "market_cap_category": "SME",
                    "face_value": company.get('face_value'),
                    "status": company.get('status', 'ACTIVE'),
                    "last_updated": datetime.now().isoformat()
                }
            
            # Only add if we have meaningful data
            if unified["company_name"] and unified["primary_symbol"]:
                unified_companies.append(unified)
        
        logger.info(f"✅ Converted {len(unified_companies)} {exchange_prefix} companies to unified format")
        return unified_companies
    
    def deduplicate_companies(self, all_companies: List[Dict]) -> List[Dict]:
        """Deduplicate companies across exchanges"""
        logger.info("🔄 Deduplicating companies across exchanges...")
        
        # Group by ISIN first (most reliable)
        isin_groups = {}
        no_isin_companies = []
        
        for company in all_companies:
            isin = company.get('isin')
            if isin and isin.strip():
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
                    primary["exchanges"].extend(other["exchanges"])
                    primary["symbols"].update(other["symbols"])
                
                # Remove duplicates from exchanges
                primary["exchanges"] = list(set(primary["exchanges"]))
                
                logger.info(f"🔗 Merged {len(companies)} companies with ISIN {isin}: {primary['company_name']}")
                deduplicated.append(primary)
        
        # Handle companies without ISIN - group by normalized name
        name_groups = {}
        for company in no_isin_companies:
            normalized_name = self.normalize_company_name(company['company_name'])
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
                    primary["exchanges"].extend(other["exchanges"])
                    primary["symbols"].update(other["symbols"])
                
                # Remove duplicates from exchanges
                primary["exchanges"] = list(set(primary["exchanges"]))
                
                logger.info(f"🔗 Merged {len(companies)} companies by name: {primary['company_name']}")
                deduplicated.append(primary)
        
        logger.info(f"✅ Deduplication complete: {len(all_companies)} → {len(deduplicated)} companies")
        return deduplicated
    
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
    
    def create_final_integration(self) -> Dict:
        """Create final comprehensive integration"""
        logger.info("🚀 Starting Final Comprehensive Integration...")
        
        all_unified_companies = []
        
        # Load and convert NSE data
        nse_companies = self.load_data_file(self.nse_file, "NSE")
        if nse_companies:
            nse_unified = self.convert_to_unified_format(nse_companies, "NSE")
            all_unified_companies.extend(nse_unified)
        
        # Load and convert BSE Main Board data
        bse_companies = self.load_data_file(self.bse_file, "BSE Main Board")
        if bse_companies:
            bse_unified = self.convert_to_unified_format(bse_companies, "BSE_MAIN")
            all_unified_companies.extend(bse_unified)
        
        # Load and convert BSE SME data
        bse_sme_companies = self.load_data_file(self.bse_sme_file, "BSE SME")
        if bse_sme_companies:
            bse_sme_unified = self.convert_to_unified_format(bse_sme_companies, "BSE_SME")
            all_unified_companies.extend(bse_sme_unified)
        
        logger.info(f"📊 Total companies before deduplication: {len(all_unified_companies)}")
        
        # Deduplicate companies
        final_companies = self.deduplicate_companies(all_unified_companies)
        
        # Calculate statistics
        stats = self.calculate_statistics(final_companies)
        
        # Create final data structure
        final_data = {
            "metadata": {
                "total_companies": len(final_companies),
                "nse_companies": stats["nse_companies"],
                "bse_main_companies": stats["bse_main_companies"],
                "bse_sme_companies": stats["bse_sme_companies"],
                "multi_exchange_companies": stats["multi_exchange_companies"],
                "source": "NSE Complete + BSE Available + BSE SME Complete",
                "data_sources": {
                    "nse": "Enhanced NSE Companies (Main + SME + Emerge)",
                    "bse_main": "BSE API + Major Companies (Limited by API)",
                    "bse_sme": "BSE SME Comprehensive Scraping"
                },
                "bse_limitation_note": "BSE Main Board limited to API-available companies (~40) due to BSE API restrictions. Full BSE coverage would require premium data access.",
                "includes_rnit_ai": any("RNIT" in c["company_name"] for c in final_companies),
                "last_updated": datetime.now().isoformat(),
                "data_type": "FINAL_UNIFIED_COMPANIES"
            },
            "companies": final_companies,
            "statistics": stats
        }
        
        return final_data
    
    def calculate_statistics(self, companies: List[Dict]) -> Dict:
        """Calculate comprehensive statistics"""
        stats = {
            "total_companies": len(companies),
            "nse_companies": 0,
            "bse_main_companies": 0,
            "bse_sme_companies": 0,
            "multi_exchange_companies": 0,
            "companies_by_exchange": {},
            "companies_by_sector": {},
            "companies_by_market_cap": {}
        }
        
        for company in companies:
            # Count by exchange
            for exchange in company["exchanges"]:
                if "NSE" in exchange:
                    stats["nse_companies"] += 1
                elif exchange == "BSE_MAIN":
                    stats["bse_main_companies"] += 1
                elif exchange == "BSE_SME":
                    stats["bse_sme_companies"] += 1
                
                stats["companies_by_exchange"][exchange] = stats["companies_by_exchange"].get(exchange, 0) + 1
            
            # Multi-exchange companies
            if len(company["exchanges"]) > 1:
                stats["multi_exchange_companies"] += 1
            
            # By sector
            sector = company.get("sector", "Unknown")
            stats["companies_by_sector"][sector] = stats["companies_by_sector"].get(sector, 0) + 1
            
            # By market cap
            market_cap = company.get("market_cap_category", "Unknown")
            stats["companies_by_market_cap"][market_cap] = stats["companies_by_market_cap"].get(market_cap, 0) + 1
        
        return stats
    
    def save_final_data(self, data: Dict) -> str:
        """Save final integrated data"""
        import os
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved final unified data to {self.output_file}")
        return self.output_file
    
    def search_companies(self, companies: List[Dict], query: str) -> List[Dict]:
        """Search for companies"""
        query = query.lower()
        results = []
        
        for company in companies:
            if (query in company["primary_symbol"].lower() or 
                query in company["company_name"].lower() or
                any(query in symbol.lower() for symbol in company["symbols"].values())):
                results.append(company)
        
        return results

def main():
    """Main function"""
    logger.info("🚀 Starting Final Comprehensive Integration...")
    
    integration = FinalComprehensiveIntegration()
    
    # Create final integration
    final_data = integration.create_final_integration()
    
    # Save data
    output_file = integration.save_final_data(final_data)
    
    # Display results
    metadata = final_data["metadata"]
    companies = final_data["companies"]
    
    logger.info(f"\n📊 Final Comprehensive Integration Results:")
    logger.info(f"   Total Unified Companies: {metadata['total_companies']}")
    logger.info(f"   NSE Companies: {metadata['nse_companies']}")
    logger.info(f"   BSE Main Board Companies: {metadata['bse_main_companies']}")
    logger.info(f"   BSE SME Companies: {metadata['bse_sme_companies']}")
    logger.info(f"   Multi-Exchange Companies: {metadata['multi_exchange_companies']}")
    logger.info(f"   RNIT AI Included: {metadata['includes_rnit_ai']}")
    
    # Search for RNIT AI
    rnit_results = integration.search_companies(companies, "RNIT")
    if rnit_results:
        logger.info(f"\n🎯 RNIT AI Search Results:")
        for company in rnit_results:
            exchanges_str = ' + '.join(company['exchanges'])
            logger.info(f"   ✅ {company['company_name']} ({company['primary_symbol']})")
            logger.info(f"      Exchanges: {exchanges_str}")
            logger.info(f"      Sector: {company['sector']}")
            logger.info(f"      ISIN: {company['isin']}")
    
    # Show exchange distribution
    logger.info(f"\n🏢 Companies by Exchange:")
    stats = final_data.get('statistics', {})
    exchange_stats = stats.get('companies_by_exchange', {})
    for exchange, count in exchange_stats.items():
        logger.info(f"   {exchange}: {count} companies")
    
    # Show BSE limitation note
    logger.info(f"\n⚠️ BSE Limitation Note:")
    logger.info(f"   {metadata['bse_limitation_note']}")
    
    logger.info(f"\n✅ Final comprehensive integration completed!")
    logger.info(f"   Data saved to: {output_file}")

if __name__ == "__main__":
    main()
