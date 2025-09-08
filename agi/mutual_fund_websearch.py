import logging
import requests
from bs4 import BeautifulSoup
from pymongo import MongoClient
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mutual_fund_websearch")

mongo_client = MongoClient('mongodb://localhost:27017/')
db = mongo_client['mf_data']
manager_col = db['fund_managers']
scheme_col = db['schemes']

# --- 1. Websearch for Fund Manager and Scheme Data ---
def duckduckgo_search(query, num_results=5):
    """Scrape DuckDuckGo HTML results for URLs. No API key needed."""
    import urllib.parse
    search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(search_url, headers=headers, timeout=10)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.find_all("a", class_="result__a", limit=num_results):
        results.append(a["href"])
    return results


def extract_manager_from_page(url):
    try:
        resp = requests.get(url, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text(separator=' ', strip=True)
        # Heuristic: Look for 'Fund Manager' or similar patterns
        for marker in ['Fund Manager', 'Manager', 'Managed by']:
            if marker in text:
                idx = text.index(marker)
                snippet = text[idx:idx+120]
                return snippet
        # Fallback: Use headless browser for JS-heavy sites
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.options import Options
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            driver = webdriver.Chrome(options=chrome_options)
            driver.get(url)
            page_text = driver.find_element('tag name', 'body').text
            driver.quit()
            for marker in ['Fund Manager', 'Manager', 'Managed by']:
                if marker in page_text:
                    idx = page_text.index(marker)
                    snippet = page_text[idx:idx+120]
                    return snippet
        except Exception as e:
            logger.warning(f"Headless browser fallback failed for {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to extract manager from {url}: {e}")
        return None


def ingest_fund_manager_websearch(scheme_name, num_results=5):
    logger.info(f"Searching web for manager info: {scheme_name}")
    query = f"{scheme_name} mutual fund fund manager name"
    urls = duckduckgo_search(query, num_results=num_results)
    for url in urls:
        snippet = extract_manager_from_page(url)
        if snippet:
            manager_col.insert_one({
                'scheme_name': scheme_name,
                'source_url': url,
                'manager_snippet': snippet,
                'timestamp': datetime.utcnow()
            })
            logger.info(f"Inserted manager info for {scheme_name} from {url}")
            break  # Stop after first valid result
        time.sleep(1)  # Be polite to web servers

# --- 2. Websearch for Scheme Data (generic, fallback) ---
def ingest_scheme_websearch(scheme_name, num_results=5):
    logger.info(f"Searching web for scheme info: {scheme_name}")
    query = f"{scheme_name} mutual fund NAV performance history"
    urls = duckduckgo_search(query, num_results=num_results)
    for url in urls:
        # Just store the best links for human/AI review
        scheme_col.insert_one({
            'scheme_name': scheme_name,
            'source_url': url,
            'timestamp': datetime.utcnow()
        })
        logger.info(f"Inserted scheme info for {scheme_name} from {url}")
        break
        time.sleep(1)

# --- 3. Bing and Brave Search Fallbacks ---
def bing_search(query, num_results=5):
    import urllib.parse
    search_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(search_url, headers=headers, timeout=10)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for li in soup.find_all("li", class_="b_algo", limit=num_results):
        a = li.find("a")
        if a and a.get("href"):
            results.append(a["href"])
    return results

def brave_search(query, num_results=5):
    import urllib.parse
    search_url = f"https://search.brave.com/search?q={urllib.parse.quote(query)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(search_url, headers=headers, timeout=10)
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for a in soup.find_all("a", class_="result-header", limit=num_results):
        if a.get("href"):
            results.append(a["href"])
    return results

# --- 4. LLM-based Semantic Ranking Stub ---
def semantic_rank_snippets(snippets, prompt):
    """Use your local LLM (Mistral-7B or distilgpt2) to rank and select the best snippet for a given prompt."""
    if not snippets:
        return None
    try:
        from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
        import torch
        # Try to use Mistral-7B if available, else fallback to distilgpt2
        try:
            model_name = "mistralai/Mistral-7B-Instruct-v0.2"
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        except Exception:
            model_name = "distilgpt2"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto" if torch.cuda.is_available() else None)
        # Score each snippet by asking the LLM to rate its relevance
        scored = []
        for snippet in snippets:
            llm_prompt = f"Prompt: {prompt}\nSnippet: {snippet}\nQuestion: How relevant is this snippet to the prompt? Reply with a score from 1 (not relevant) to 10 (highly relevant)."
            output = llm(llm_prompt, max_length=50, do_sample=False)[0]["generated_text"]
            import re
            match = re.search(r'(\d+)', output)
            score = int(match.group(1)) if match else 1
            scored.append((score, snippet, output))
        scored.sort(reverse=True)
        best_score, best_snippet, best_llm_output = scored[0]
        # Audit log
        try:
            from pymongo import MongoClient
            mongo_client = MongoClient('mongodb://localhost:27017/')
            audit_col = mongo_client['mf_data']['llm_ranking_audit']
            audit_col.insert_one({
                'prompt': prompt,
                'snippets': snippets,
                'scores': [s[0] for s in scored],
                'llm_outputs': [s[2] for s in scored],
                'selected_snippet': best_snippet,
                'timestamp': datetime.utcnow()
            })
        except Exception as e:
            logger.warning(f"LLM ranking audit log failed: {e}")
        return best_snippet
    except Exception as e:
        logger.error(f"LLM semantic ranking failed: {e}")
        return max(snippets, key=len)

# --- 5. Full Automation for All Schemes ---
def get_all_scheme_names():
    # Query MongoDB for all unique scheme names (from schemes/NAV collection)
    return [x['scheme_name'] for x in scheme_col.distinct('scheme_name')]

if __name__ == "__main__":
    all_schemes = get_all_scheme_names()
    for scheme in all_schemes:
        snippets = []
        # Try DuckDuckGo, then Bing, then Brave
        for search_func in [duckduckgo_search, bing_search, brave_search]:
            urls = search_func(f"{scheme} mutual fund fund manager name", num_results=5)
            for url in urls:
                snippet = extract_manager_from_page(url)
                if snippet:
                    snippets.append(snippet)
                time.sleep(1)
            if snippets:
                break
        best_snippet = semantic_rank_snippets(snippets, f"{scheme} fund manager")
        if best_snippet:
            manager_col.insert_one({
                'scheme_name': scheme,
                'manager_snippet': best_snippet,
                'timestamp': datetime.utcnow()
            })
            logger.info(f"Inserted best manager snippet for {scheme}")
        # Repeat for scheme info
        scheme_snippets = []
        for search_func in [duckduckgo_search, bing_search, brave_search]:
            urls = search_func(f"{scheme} mutual fund NAV performance history", num_results=5)
            for url in urls:
                scheme_snippets.append(url)
                time.sleep(1)
            if scheme_snippets:
                break
        # Store all found URLs for review
        for url in scheme_snippets:
            scheme_col.insert_one({
                'scheme_name': scheme,
                'source_url': url,
                'timestamp': datetime.utcnow()
            })
        time.sleep(2)
