#!/usr/bin/env python3
"""Check BSE SME data"""

import json

# Load BSE SME data
with open('market_data/comprehensive_bse_sme_companies.json', 'r') as f:
    data = json.load(f)

print(f"📊 BSE SME Data Summary:")
print(f"Total Companies: {data['metadata']['total_companies']}")
print(f"Source: {data['metadata']['source']}")
print()

companies = data['companies']

print("🔍 First 10 BSE SME Companies:")
for i, company in enumerate(companies[:10]):
    print(f"{i+1:2d}. {company['scrip_name']} - LTP: Rs{company['ltp']} - Trades: {company['no_of_trades']}")

print()
print("🎯 Searching for RNIT...")
rnit_companies = [c for c in companies if 'rnit' in c['scrip_name'].lower() or 'rnit' in c['company_name'].lower()]
if rnit_companies:
    for company in rnit_companies:
        print(f"✅ Found: {company['company_name']} ({company['symbol']}) - LTP: Rs{company['ltp']}")
else:
    print("❌ RNIT not found in current BSE SME data")

print()
print("🔥 Top 5 Most Traded BSE SME Companies:")
sorted_companies = sorted(companies, key=lambda x: x['no_of_trades'], reverse=True)
for i, company in enumerate(sorted_companies[:5], 1):
    print(f"{i}. {company['scrip_name']} - {company['no_of_trades']} trades - LTP: Rs{company['ltp']}")

print()
print("📈 Companies with highest LTP:")
price_sorted = sorted([c for c in companies if c['ltp'] > 0], key=lambda x: x['ltp'], reverse=True)
for i, company in enumerate(price_sorted[:5], 1):
    print(f"{i}. {company['scrip_name']} - Rs{company['ltp']}")

print()
print(f"📊 Statistics:")
print(f"   Companies with trading activity: {len([c for c in companies if c['no_of_trades'] > 0])}")
print(f"   Companies with price data: {len([c for c in companies if c['ltp'] > 0])}")
print(f"   Total trades across all companies: {sum(c['no_of_trades'] for c in companies)}")

# Check if we have the expected 596 companies
if data['metadata']['total_companies'] < 500:
    print()
    print("⚠️  WARNING: Expected ~596 BSE SME companies but only got", data['metadata']['total_companies'])
    print("   This might be due to pagination or dynamic loading on the website")
