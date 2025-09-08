#!/usr/bin/env python3
"""
Verify BSE Companies Data
Check the comprehensive BSE data and search for RNIT AI
"""

import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main verification function"""
    logger.info("🔍 Verifying BSE Companies Data...")
    
    try:
        with open('market_data/comprehensive_bse_all_companies.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        companies = data['companies']
        metadata = data['metadata']
        
        logger.info(f"📊 BSE Data Summary:")
        logger.info(f"   Total Companies: {metadata['total_companies']}")
        logger.info(f"   Methods Used: {metadata['methods']}")
        logger.info(f"   RNIT AI Included: {metadata['includes_rnit_ai']}")
        logger.info(f"   Last Updated: {metadata['last_updated']}")
        
        # Filter out empty companies
        valid_companies = [c for c in companies if c['company_name'] and c['scrip_code']]
        logger.info(f"   Valid Companies: {len(valid_companies)}")
        
        # Search for RNIT AI
        logger.info(f"\n🎯 Searching for RNIT AI...")
        rnit_companies = []
        for company in valid_companies:
            if ('rnit' in company['company_name'].lower() or 
                'rnit' in company['scrip_name'].lower() or
                company['scrip_code'] == '543320'):
                rnit_companies.append(company)
        
        if rnit_companies:
            logger.info(f"✅ Found {len(rnit_companies)} RNIT companies:")
            for company in rnit_companies:
                logger.info(f"   • {company['company_name']} ({company['scrip_code']})")
                logger.info(f"     Scrip Name: {company['scrip_name']}")
                logger.info(f"     Sector: {company['sector']}")
                logger.info(f"     Group: {company['group']}")
                logger.info(f"     ISIN: {company['isin']}")
        else:
            logger.warning(f"❌ RNIT AI not found in BSE data")
        
        # Show sample companies
        logger.info(f"\n📋 Sample BSE Companies (first 10 valid):")
        count = 0
        for company in valid_companies:
            if count >= 10:
                break
            if company['company_name']:
                logger.info(f"   {count+1:2d}. {company['company_name']} ({company['scrip_code']}) - {company['sector']}")
                count += 1
        
        # Show sector distribution
        sectors = {}
        for company in valid_companies:
            sector = company['sector'] or "Unknown"
            sectors[sector] = sectors.get(sector, 0) + 1
        
        logger.info(f"\n🏭 Top 10 Sectors:")
        sorted_sectors = sorted(sectors.items(), key=lambda x: x[1], reverse=True)
        for sector, count in sorted_sectors[:10]:
            logger.info(f"   {sector}: {count} companies")
        
        # Look for major companies
        logger.info(f"\n🏢 Major BSE Companies Found:")
        major_companies = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK', 'ITC', 'HINDUNILVR']
        for major in major_companies:
            found = [c for c in valid_companies if major.lower() in c['scrip_name'].lower() or major.lower() in c['company_name'].lower()]
            if found:
                for company in found:
                    logger.info(f"   ✅ {company['company_name']} ({company['scrip_code']})")
            else:
                logger.info(f"   ❌ {major} not found")
        
        logger.info(f"\n✅ BSE data verification completed!")
        
    except Exception as e:
        logger.error(f"❌ Error verifying BSE data: {e}")

if __name__ == "__main__":
    main()
