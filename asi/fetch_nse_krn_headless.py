try:
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    import time
    uc_available = True
except ImportError:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time
    uc_available = False

symbol = 'KRNHEAT'
url = f'https://www.nseindia.com/get-quotes/equity?symbol={symbol}'

if uc_available:
    options = uc.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1280,800')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36')
    driver = uc.Chrome(options=options)
else:
    print("[INFO] undetected-chromedriver not installed. Using standard Selenium Chrome. For best results, install with: pip install undetected-chromedriver")
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1280,800')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36')
    driver = webdriver.Chrome(options=options)
    try:
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': '''
                Object.defineProperty(navigator, 'webdriver', {get: () => undefined})
            '''
        })
    except Exception:
        pass

driver.get(url)
time.sleep(4)  # Wait for JS to load

from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import NoSuchElementException

try:
    # Wait for cookie/popup and close if present
    for _ in range(10):
        try:
            # NSE sometimes shows a popup or cookie banner
            popup = driver.find_element(By.CSS_SELECTOR, 'button[id*="wzrk-cancel"]')
            popup.click()
            break
        except NoSuchElementException:
            time.sleep(0.5)
        except Exception:
            break

    # Wait up to 20 seconds for price element
    price = None
    for _ in range(20):
        try:
            # Try all known selectors for price
            selectors = [
                '//span[contains(@id,"lastPrice")]',
                'span[class*="last-price"]',
                '//div[contains(@id,"quoteLtp")]//span',
                '//span[contains(@class,"last-price")]'
            ]
            for sel in selectors:
                try:
                    if sel.startswith('//'):
                        elem = driver.find_element(By.XPATH, sel)
                    else:
                        elem = driver.find_element(By.CSS_SELECTOR, sel)
                    price = elem.text.replace(',', '').strip()
                    if price:
                        print(f"KRN Heat Exchanger (NSE: {symbol}) live price: {price}")
                        raise StopIteration
                except Exception:
                    continue
        except StopIteration:
            break
        time.sleep(1)
    if not price:
        print("Could not extract price. Page structure may have changed or NSE is blocking headless browsers.")
        print("Page title:", driver.title)
        html = driver.page_source
        print("First 2000 chars of HTML:\n", html[:2000])
        bot_blocked = any(x in html.lower() for x in ["are you a human", "robot", "captcha", "access denied", "forbidden", "verify"])
        if bot_blocked:
            print("HEADLESS BLOCKED: NSE is serving a bot-check or access denied page. Retrying with non-headless Chrome...")
            try:
                driver.quit()
            except Exception:
                pass
            # Retry with non-headless Chrome
            if uc_available:
                options = uc.ChromeOptions()
            else:
                from selenium.webdriver.chrome.options import Options
                options = Options()
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            options.add_argument('--window-size=1280,800')
            options.add_argument('--disable-blink-features=AutomationControlled')
            options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36')
            if uc_available:
                driver2 = uc.Chrome(options=options)
            else:
                driver2 = webdriver.Chrome(options=options)
            driver2.get(url)
            time.sleep(4)
            try:
                # Try all selectors again
                selectors = [
                    '//span[contains(@id,"lastPrice")]',
                    'span[class*="last-price"]',
                    '//div[contains(@id,"quoteLtp")]//span',
                    '//span[contains(@class,"last-price")]'
                ]
                found = False
                for sel in selectors:
                    try:
                        if sel.startswith('//'):
                            elem = driver2.find_element(By.XPATH, sel)
                        else:
                            elem = driver2.find_element(By.CSS_SELECTOR, sel)
                        price = elem.text.replace(',', '').strip()
                        if price:
                            print(f"KRN Heat Exchanger (NSE: {symbol}) live price (non-headless): {price}")
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    print("Non-headless mode also failed. GUI is required for this method. Please run this script on a machine with a visible Chrome browser.")
                    html2 = driver2.page_source
                    print("First 2000 chars of non-headless HTML:\n", html2[:2000])
            finally:
                driver2.quit()
finally:
    try:
        driver.quit()
    except Exception:
        pass

