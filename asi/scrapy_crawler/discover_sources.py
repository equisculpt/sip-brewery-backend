import requests
import json
import time

# Use SerpAPI (or similar) for Google/Bing search
SERPAPI_KEY = "YOUR_SERPAPI_KEY"  # Replace with your API key
SEARCH_TERMS = [
    "latest Indian finance news", "top Indian mutual fund blogs", "NSE BSE SEBI news", "Indian stock market analysis"
]

RESULTS_FILE = "discovered_finance_sources.json"


def discover_new_finance_sites():
    discovered = set()
    for term in SEARCH_TERMS:
        url = f"https://serpapi.com/search.json?q={term}&api_key={SERPAPI_KEY}&num=20"
        try:
            resp = requests.get(url)
            data = resp.json()
            for res in data.get("organic_results", []):
                link = res.get("link")
                if link:
                    discovered.add(link)
            time.sleep(2)  # Be polite to API
        except Exception as e:
            print(f"Error searching {term}: {e}")
    # Save discovered URLs
    with open(RESULTS_FILE, "w") as f:
        json.dump(list(discovered), f, indent=2)
    print(f"Discovered {len(discovered)} finance sources.")

if __name__ == "__main__":
    discover_new_finance_sites()
