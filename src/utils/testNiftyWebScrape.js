// testNiftyWebScrape.js
// Scrape live Nifty 50 price from NSE homepage (no API)
const axios = require('axios');
const cheerio = require('cheerio');

async function fetchNifty50FromWeb() {
  try {
    const headers = {
      'User-Agent': 'Mozilla/5.0',
      'Referer': 'https://www.nseindia.com/'
    };
    const url = 'https://www.nseindia.com/';
    const res = await axios.get(url, { headers });
    const $ = cheerio.load(res.data);
    // Try to find the Nifty 50 price (HTML changes often)
    let niftyPrice = null;
    $('div').each((i, el) => {
      const text = $(el).text();
      if (text.includes('NIFTY 50')) {
        // Look for price in sibling or nearby span/div
        const match = text.match(/NIFTY 50\s*([\d,]+\.?\d*)/);
        if (match) {
          niftyPrice = match[1];
        }
      }
    });
    if (!niftyPrice) {
      // Try common class selectors (update as needed)
      niftyPrice = $("#Nifty50 .value, .nifty50-value, .liveindex-value").first().text();
    }
    if (niftyPrice) {
      console.log(`Nifty 50 Live Price (web): ${niftyPrice}`);
      return niftyPrice;
    } else {
      console.log('Could not scrape live Nifty 50 price from web.');
      return null;
    }
  } catch (err) {
    console.error('Error scraping Nifty 50 price:', err.message);
    return null;
  }
}

fetchNifty50FromWeb();
