"""
NSE Cookie Manager using Playwright
- Launches a real browser to nseindia.com
- Extracts all relevant cookies for anti-bot bypass
- Provides fresh cookies for aiohttp sessions
"""
import asyncio
from playwright.async_api import async_playwright
import logging

NSE_URL = "https://www.nseindia.com/"
COOKIE_NAMES = ["bm_sv", "ak_bmsc", "nseappid", "nsit", "nseQuote", "nseQuoteSymbols"]

async def get_nse_cookies() -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()
        await page.goto(NSE_URL, wait_until="domcontentloaded")
        await asyncio.sleep(2)  # Let JS run, cookies set
        cookies = await context.cookies()
        await browser.close()
        cookie_dict = {c['name']: c['value'] for c in cookies if c['name'] in COOKIE_NAMES}
        logging.info(f"NSE cookies fetched: {cookie_dict}")
        return cookie_dict

# Example usage (for testing)
if __name__ == "__main__":
    cookies = asyncio.run(get_nse_cookies())
    print(cookies)
