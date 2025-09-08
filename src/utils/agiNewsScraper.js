// agiNewsScraper.js
// Scrapes top Indian/global finance/economics news every 30 minutes, stores AGI-ready summaries
const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');
const path = require('path');

const NEWS_SITES = [
  {
    name: 'Moneycontrol',
    url: 'https://www.moneycontrol.com/news/business/',
    selector: '.clearfix .headline a',
    parse: el => ({
      headline: el.text(),
      link: el.attr('href'),
      source: 'Moneycontrol'
    })
  },
  {
    name: 'Economic Times',
    url: 'https://economictimes.indiatimes.com/news/economy',
    selector: 'div.eachStory h3 a',
    parse: el => ({
      headline: el.text(),
      link: 'https://economictimes.indiatimes.com' + el.attr('href'),
      source: 'Economic Times'
    })
  },
  {
    name: 'LiveMint',
    url: 'https://www.livemint.com/market',
    selector: 'div.listingNewText h2 a',
    parse: el => ({
      headline: el.text(),
      link: 'https://www.livemint.com' + el.attr('href'),
      source: 'LiveMint'
    })
  },
  {
    name: 'Business Standard',
    url: 'https://www.business-standard.com/category/economy-policy-news-1060101.htm',
    selector: '.listingPage .listing li a',
    parse: el => ({
      headline: el.text(),
      link: 'https://www.business-standard.com' + el.attr('href'),
      source: 'Business Standard'
    })
  },
  {
    name: 'Bloomberg Quint',
    url: 'https://www.bqprime.com/markets',
    selector: 'div.card__body a.card__headline',
    parse: el => ({
      headline: el.text(),
      link: 'https://www.bqprime.com' + el.attr('href'),
      source: 'Bloomberg Quint'
    })
  },
  {
    name: 'CNBC TV18',
    url: 'https://www.cnbctv18.com/market/',
    selector: '.listBody .listItem a',
    parse: el => ({
      headline: el.text(),
      link: el.attr('href'),
      source: 'CNBC TV18'
    })
  },
  {
    name: 'PIB India',
    url: 'https://pib.gov.in/AllRelease.aspx',
    selector: '.content-area .table-bordered td a',
    parse: el => ({
      headline: el.text(),
      link: 'https://pib.gov.in/' + el.attr('href'),
      source: 'PIB India'
    })
  },
  {
    name: 'RBI Press Releases',
    url: 'https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx',
    selector: 'table#ctl00_ContentPlaceHolder1_gvPressRelease a',
    parse: el => ({
      headline: el.text(),
      link: 'https://www.rbi.org.in' + el.attr('href'),
      source: 'RBI'
    })
  },
  // Example RSS-based: fallback to RSS if scraping fails
  {
    name: 'Reuters RSS',
    url: 'https://feeds.reuters.com/reuters/businessNews',
    rss: true,
    parseRss: item => ({
      headline: item.title,
      link: item.link,
      source: 'Reuters RSS'
    })
  },
  // Example: Placeholder for headless browser (puppeteer) for JS-heavy sites
  {
    name: 'NSE India',
    url: 'https://www.nseindia.com/market-data/news',
    headless: true,
    parseHeadless: html => {
      // In production: Use puppeteer to extract headlines
      return [];
    }
  }
];

const DATA_DIR = path.join(__dirname, 'data');
const NEWS_FILE = path.join(DATA_DIR, 'agi_news.jsonl');

function getTimestamp() {
  return new Date().toISOString();
}

function embedSummary(text) {
  // Placeholder: In production, use ML model for embedding/vectorization
  return text.slice(0, 128);
}

async function scrapeAndStoreNews() {
  if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);
  let newsItems = [];
  for (const site of NEWS_SITES) {
    try {
      const res = await axios.get(site.url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
      const $ = cheerio.load(res.data);
      $(site.selector).each((i, el) => {
        const item = site.parse($(el));
        if (item.headline && item.link) {
          newsItems.push({
            headline: item.headline.trim(),
            link: item.link,
            source: item.source,
            timestamp: getTimestamp(),
            embedding: embedSummary(item.headline)
          });
        }
      });
    } catch (err) {
      console.error(`Error scraping ${site.name}:`, err.message);
    }
  }
  // Deduplicate by headline+source
  const unique = {};
  for (const item of newsItems) {
    const key = item.source + '|' + item.headline;
    if (!unique[key]) unique[key] = item;
  }
  // Append to file
  const toWrite = Object.values(unique).map(i => JSON.stringify(i)).join('\n') + '\n';
  fs.appendFileSync(NEWS_FILE, toWrite);
  console.log(`Scraped and stored ${Object.keys(unique).length} news items at ${getTimestamp()}`);
  return Object.values(unique);
}

// For AGI: Load rolling N latest news, or filter by date/source
function loadRecentNews(n = 100) {
  if (!fs.existsSync(NEWS_FILE)) return [];
  const lines = fs.readFileSync(NEWS_FILE, 'utf-8').split('\n').filter(Boolean);
  return lines.slice(-n).map(l => JSON.parse(l));
}

// To schedule: use cron or OS scheduler in production
if (require.main === module) {
  scrapeAndStoreNews();
}

module.exports = { scrapeAndStoreNews, loadRecentNews };
