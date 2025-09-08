#!/usr/bin/env python3
"""
Test deduplication logic with sample data
"""

import re
from typing import List, Dict

def normalize_company_name(name: str) -> str:
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

def test_deduplication():
    # Sample NSE companies
    nse_companies = [
        {"symbol": "RELIANCE", "name": "Reliance Industries Limited", "exchange": "NSE_MAIN", "isin": None},
        {"symbol": "TCS", "name": "Tata Consultancy Services Limited", "exchange": "NSE_MAIN", "isin": None},
        {"symbol": "INFY", "name": "Infosys Limited", "exchange": "NSE_MAIN", "isin": None},
        {"symbol": "HDFCBANK", "name": "HDFC Bank Limited", "exchange": "NSE_MAIN", "isin": None},
        {"symbol": "WIPRO", "name": "Wipro Limited", "exchange": "NSE_MAIN", "isin": None},
    ]
    
    # Sample BSE companies (some duplicates)
    bse_companies = [
        {"symbol": "500325", "name": "Reliance Industries Limited", "exchange": "BSE_MAIN", "isin": "INE002A01018"},
        {"symbol": "532540", "name": "Tata Consultancy Services Limited", "exchange": "BSE_MAIN", "isin": "INE467B01029"},
        {"symbol": "500209", "name": "Infosys Limited", "exchange": "BSE_MAIN", "isin": "INE009A01021"},
        {"symbol": "500180", "name": "HDFC Bank Limited", "exchange": "BSE_MAIN", "isin": "INE040A01034"},
        {"symbol": "500875", "name": "ITC Limited", "exchange": "BSE_MAIN", "isin": "INE154A01025"},  # Only on BSE
    ]
    
    all_companies = nse_companies + bse_companies
    
    print(f"Total companies: {len(all_companies)}")
    print(f"NSE companies: {len(nse_companies)}")
    print(f"BSE companies: {len(bse_companies)}")
    print()
    
    # Group by normalized name
    name_mapping = {}
    
    for company in all_companies:
        name = normalize_company_name(company.get("name", ""))
        print(f"Company: {company['name']} -> Normalized: '{name}'")
        if name:
            if name not in name_mapping:
                name_mapping[name] = []
            name_mapping[name].append(company)
    
    print(f"\nUnique normalized names: {len(name_mapping)}")
    
    for name, companies in name_mapping.items():
        if len(companies) > 1:
            print(f"\nDuplicate group '{name}':")
            for company in companies:
                print(f"  - {company['symbol']} ({company['exchange']}): {company['name']}")
        else:
            print(f"\nUnique company '{name}': {companies[0]['symbol']} ({companies[0]['exchange']})")

if __name__ == "__main__":
    test_deduplication()
