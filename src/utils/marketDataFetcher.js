// marketDataFetcher.js
// Utility functions to fetch market, macroeconomic, and news data from public APIs
// For production, move API keys to environment variables and handle errors and rate limits

const axios = require('axios');

const ALPHA_VANTAGE_API_KEY = process.env.ALPHA_VANTAGE_API_KEY;
const NEWS_API_KEY = process.env.NEWS_API_KEY;

module.exports = {
  /**
   * Fetch historical price data for a given symbol (stocks, FX, crypto)
   * @param {string} symbol
   * @param {string} market ('stock'|'forex'|'crypto')
   * @param {string} interval ('1min','5min','15min','60min','daily','weekly','monthly')
   */
  async fetchPriceData(symbol, market = 'stock', interval = 'daily') {
    let url;
    if (market === 'stock') {
      url = `https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=${symbol}&apikey=${ALPHA_VANTAGE_API_KEY}`;
    } else if (market === 'forex') {
      url = `https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=${symbol.split('/')[0]}&to_symbol=${symbol.split('/')[1]}&apikey=${ALPHA_VANTAGE_API_KEY}`;
    } else if (market === 'crypto') {
      url = `https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=${symbol}&market=USD&apikey=${ALPHA_VANTAGE_API_KEY}`;
    } else {
      throw new Error('Unsupported market type');
    }
    const response = await axios.get(url);
    return response.data;
  },

  /**
   * Fetch Indian macroeconomic data (e.g., GDP, inflation) using Trading Economics API
   * Docs: https://docs.tradingeconomics.com/
   * @param {string} indicator (e.g., 'GDP', 'Inflation Rate', 'Interest Rate')
   * @param {string} country (default 'India')
   */
  async fetchIndianMacroData(indicator, country = 'India') {
    const TE_API_KEY = process.env.TRADING_ECONOMICS_API_KEY;
    const url = `https://api.tradingeconomics.com/indicators/${encodeURIComponent(country)}?c=${TE_API_KEY}`;
    const response = await axios.get(url);
    // Filter for the requested indicator
    if (indicator) {
      return response.data.filter(item => item.category && item.category.toLowerCase().includes(indicator.toLowerCase()));
    }
    return response.data;
  },

  // (Legacy) Fetch macro data from FRED (US/global)
  // async fetchMacroData(seriesId) { ... }

  /**
   * Fetch macroeconomic data directly from RBI Database (scraping)
   * Example: GDP, CPI, Interest Rate, etc.
   * @param {string} indicator (e.g., 'GDP', 'CPI', 'Interest Rate')
   * @returns {Promise<Array|Object>} Scraped data or error
   */
  async fetchRbiMacroData(indicator) {
    const cheerio = require('cheerio');
    const url = 'https://dbie.rbi.org.in/DBIE/dbie.rbi?site=home';
    // NOTE: RBI site structure changes often; this is a placeholder for demo
    try {
      const response = await axios.get(url);
      const $ = cheerio.load(response.data);
      // Example: Find links or tables containing the indicator
      const results = [];
      $('a').each((i, el) => {
        const text = $(el).text();
        if (text.toLowerCase().includes(indicator.toLowerCase())) {
          results.push({ text, href: $(el).attr('href') });
        }
      });
      return results;
    } catch (err) {
      return { error: 'Failed to fetch RBI macro data', details: err.message };
    }
  },

  /**
   * Fetch macroeconomic data from MOSPI (scraping)
   * Example: GDP, IIP, CPI, etc.
   * @param {string} indicator (e.g., 'GDP', 'IIP', 'CPI')
   * @returns {Promise<Array|Object>} Scraped data or error
   */
  async fetchMospiMacroData(indicator) {
    // Placeholder: MOSPI site scraping logic not implemented
    // To implement: fetch and parse tables from https://mospi.gov.in/ as needed
    return { error: 'MOSPI macro data fetch not implemented yet', indicator };
  },

  /**
   * Fetch historical equity data from NSE (scraping)
   * Supports: price, volume, P/E, etc. (limited by NSE site structure)
   * @param {string} symbol (e.g., 'RELIANCE')
   * @param {string} series (e.g., 'EQ')
   * @param {string} from (YYYY-MM-DD)
   * @param {string} to (YYYY-MM-DD)
   * @returns {Promise<Array|Object>} Data rows or error
   *
   * Note: NSE blocks many bots; use headers and expect possible failures.
   * For full historical, download from https://www1.nseindia.com/products/content/equities/equities/eq_security.htm
   */
  async fetchNseHistoricalData(symbol, series = 'EQ', from, to) {
    const axios = require('axios');
    const qs = require('querystring');
    const url = 'https://www.nseindia.com/api/historical/cm/equity';
    try {
      const headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://www.nseindia.com/',
        'Accept': 'application/json'
      };
      const params = {
        symbol,
        series,
        from,
        to
      };
      const response = await axios.get(url + '?' + qs.stringify(params), { headers });
      return response.data.data || [];
    } catch (err) {
      return { error: 'Failed to fetch NSE historical data', details: err.message };
    }
  },

  /**
   * Fetch historical equity data from BSE (scraping)
   * @param {string} scripCode (e.g., '500325' for RELIANCE)
   * @param {string} from (DD/MM/YYYY)
   * @param {string} to (DD/MM/YYYY)
   * @returns {Promise<Array|Object>} Data rows or error
   *
   * BSE provides downloadable CSVs: https://www.bseindia.com/markets/MarketInfo/BhavCopy.aspx
   * For bulk, download zipped CSVs and parse locally.
   */
  async fetchBseHistoricalData(scripCode, from, to) {
    // For demo: return download URL for Bhavcopy. For full automation, download and parse CSV.
    const url = `https://www.bseindia.com/markets/MarketInfo/BhavCopy.aspx`;
    return { info: 'Download daily zipped CSVs from BSE Bhavcopy and parse for historical data.', url };
  },

  /**
   * Download and parse open government data CSV from data.gov.in
   * @param {string} csvUrl (direct link to CSV file)
   * @returns {Promise<Array|Object>} Parsed data rows or error
   */
  async fetchDataGovInCsv(csvUrl) {
    const axios = require('axios');
    const csv = require('csvtojson');
    try {
      const response = await axios.get(csvUrl);
      const data = await csv().fromString(response.data);
      return data;
    } catch (err) {
      return { error: 'Failed to fetch or parse data.gov.in CSV', details: err.message };
    }
  },

  /**
   * Fetch latest news headlines for a symbol or topic
   * @param {string} query (symbol or keyword)
   */
  async fetchNews(query) {
    const url = `https://newsapi.org/v2/everything?q=${encodeURIComponent(query)}&apiKey=${NEWS_API_KEY}`;
    const response = await axios.get(url);
    return response.data;
  }
};
