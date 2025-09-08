// featureExtractor.js
// Rolling feature extraction for AGI: news, market, macro data
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

const DATA_DIR = path.join(__dirname, 'data');
const NEWS_FILE = path.join(DATA_DIR, 'agi_news.jsonl');
const MARKET_FILE = path.join(DATA_DIR, 'market_features.jsonl');
const MACRO_FILE = path.join(DATA_DIR, 'macro_features.jsonl');

function hash(str) {
  return crypto.createHash('md5').update(str).digest('hex');
}

// --- News Feature Extraction ---
function extractNewsFeatures(newsItem) {
  // Example: simple sentiment, length, headline embedding stub
  const sentiment = newsItem.headline.match(/gain|rise|positive|up/i)
    ? 1 : newsItem.headline.match(/fall|drop|negative|down/i) ? -1 : 0;
  return {
    id: hash(newsItem.headline + newsItem.timestamp),
    headline: newsItem.headline,
    source: newsItem.source,
    timestamp: newsItem.timestamp,
    sentiment,
    embedding: newsItem.embedding,
    length: newsItem.headline.length
  };
}

function processRollingNews(n = 100) {
  if (!fs.existsSync(NEWS_FILE)) return [];
  const lines = fs.readFileSync(NEWS_FILE, 'utf-8').split('\n').filter(Boolean);
  const latest = lines.slice(-n).map(l => JSON.parse(l));
  return latest.map(extractNewsFeatures);
}

// --- Market Feature Extraction (stub) ---
function extractMarketFeatures(row) {
  // Example: extract price, pct change, volume, etc.
  const open = parseFloat(row.OPEN) || 0;
  const close = parseFloat(row.CLOSE) || 0;
  const change = close - open;
  const pctChange = open ? (change / open) * 100 : 0;
  return {
    symbol: row.SYMBOL,
    date: row.TIMESTAMP || row.DATE,
    open,
    close,
    change,
    pctChange,
    volume: parseInt(row.TTL_TRD_QNTY || row.VOLUME || '0', 10)
  };
}

function processRollingMarket(marketRows, n = 100) {
  // Accepts array of parsed Bhavcopy rows
  return marketRows.slice(-n).map(extractMarketFeatures);
}

// --- Mutual Fund Feature Extraction ---
// navHistory: [{date: 'YYYY-MM-DD', nav: number}], returns: {1D, 1M, ...}
const { getSebiSafeLabel, getSebiDisclaimer } = require('./sebiSafeLabels');

function extractMfFeatures({ schemeDetail, navHistory, returns }) {
  // Compute rolling returns, volatility, drawdown, alpha/beta (stubbed)
  const navs = navHistory.map(d => d.nav).filter(Number);
  const dates = navHistory.map(d => d.date);
  const latestNav = navs[navs.length - 1];
  const oldestNav = navs[0];
  const totalReturn = oldestNav ? ((latestNav - oldestNav) / oldestNav) * 100 : 0;
  // Volatility (stddev of daily returns)
  const dailyReturns = navs.slice(1).map((n, i) => (n - navs[i]) / navs[i]);
  const mean = dailyReturns.reduce((a, b) => a + b, 0) / (dailyReturns.length || 1);
  const variance = dailyReturns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / (dailyReturns.length || 1);
  const volatility = Math.sqrt(variance) * Math.sqrt(252) * 100; // annualized %
  // Max drawdown
  let maxDd = 0, peak = navs[0] || 0;
  for (const nav of navs) {
    if (nav > peak) peak = nav;
    const dd = peak ? (peak - nav) / peak : 0;
    if (dd > maxDd) maxDd = dd;
  }
  // Alpha/beta: stubbed (would need benchmark series)
  const alpha = null, beta = null;
  // Determine SEBI-safe signal (stub: based on return/volatility)
  let signal = 'hold';
  if (totalReturn > 15 && volatility < 10) signal = 'buy';
  else if (totalReturn < 0 || maxDd > 30) signal = 'sell';
  const safe = getSebiSafeLabel(signal);
  return {
    scheme_code: schemeDetail.scheme_code,
    scheme_name: schemeDetail.scheme_name,
    asset_category: schemeDetail.asset_category,
    risk: schemeDetail.risk,
    expense: schemeDetail.regular_expense,
    totalReturn,
    volatility,
    maxDrawdown: maxDd * 100,
    returns,
    alpha,
    beta,
    navHistory: navHistory.slice(-30), // last 30 for quick view
    lastUpdated: dates[dates.length - 1],
    sebiSafe: {
      label: safe.label,
      info: safe.info,
      disclaimer: getSebiDisclaimer()
    }
  };
}

// --- Portfolio Analytics ---
// portfolio: [{type: 'stock'|'mf', symbol|scheme_code, units, price|nav, ...}]
function extractPortfolioFeatures(portfolio, mfAnalytics, stockAnalytics) {
  const { getSebiSafeLabel, getSebiDisclaimer } = require('./sebiSafeLabels');
  let totalValue = 0;
  const allocation = {};
  // Calculate total value and allocation
  for (const holding of portfolio) {
    let value = 0;
    if (holding.type === 'mf') {
      const mf = mfAnalytics[holding.scheme_code];
      value = holding.units * (mf ? mf.navHistory[mf.navHistory.length-1]?.nav : holding.nav);
      allocation[mf ? mf.scheme_name : holding.scheme_code] = (allocation[mf ? mf.scheme_name : holding.scheme_code] || 0) + value;
    } else if (holding.type === 'stock') {
      const stock = stockAnalytics[holding.symbol];
      value = holding.units * (stock ? stock.close : holding.price);
      allocation[holding.symbol] = (allocation[holding.symbol] || 0) + value;
    }
    totalValue += value;
  }
  // Normalize allocation
  Object.keys(allocation).forEach(k => allocation[k] = (allocation[k]/totalValue)*100);
  // Risk/return: stubbed as average of components
  const avgVol = Object.values(mfAnalytics).reduce((a, mf) => a + (mf.volatility||0), 0) / (Object.keys(mfAnalytics).length||1);
  const avgRet = Object.values(mfAnalytics).reduce((a, mf) => a + (mf.totalReturn||0), 0) / (Object.keys(mfAnalytics).length||1);
  // Overlap: stocks in both direct and MF
  const mfStocks = new Set();
  Object.values(mfAnalytics).forEach(mf => (mf.stocks||[]).forEach(s => mfStocks.add(s)));
  const overlap = portfolio.filter(h => h.type==='stock' && mfStocks.has(h.symbol)).map(h => h.symbol);
  // Determine overall signal for portfolio (stub: based on avgRet/avgVol)
  let signal = 'hold';
  if (avgRet > 12 && avgVol < 10) signal = 'buy';
  else if (avgRet < 0 || avgVol > 25) signal = 'sell';
  const safe = getSebiSafeLabel(signal);
  return {
    totalValue,
    allocation,
    avgVol,
    avgRet,
    overlap,
    sebiSafe: {
      label: safe.label,
      info: safe.info,
      disclaimer: getSebiDisclaimer()
    }
  };
}

// --- Stock vs Mutual Fund Comparison ---
function compareStockVsMf(stockAnalytics, mfAnalytics) {
  const { getSebiSafeLabel, getSebiDisclaimer } = require('./sebiSafeLabels');
  // Compare risk/return/drawdown
  const stocks = Object.values(stockAnalytics);
  const mfs = Object.values(mfAnalytics);
  const avgStockVol = stocks.reduce((a,s)=>a+(s.volatility||0),0)/(stocks.length||1);
  const avgMfVol = mfs.reduce((a,m)=>a+(m.volatility||0),0)/(mfs.length||1);
  const avgStockRet = stocks.reduce((a,s)=>a+(s.totalReturn||0),0)/(stocks.length||1);
  const avgMfRet = mfs.reduce((a,m)=>a+(m.totalReturn||0),0)/(mfs.length||1);
  const avgStockDd = stocks.reduce((a,s)=>a+(s.maxDrawdown||0),0)/(stocks.length||1);
  const avgMfDd = mfs.reduce((a,m)=>a+(m.maxDrawdown||0),0)/(mfs.length||1);
  // Determine SEBI-safe label for comparison (stub: based on relative returns)
  let signal = 'hold';
  if (avgMfRet > avgStockRet + 2) signal = 'buy';
  else if (avgStockRet > avgMfRet + 2) signal = 'sell';
  const safe = getSebiSafeLabel(signal);
  return {
    avgStockVol,
    avgMfVol,
    avgStockRet,
    avgMfRet,
    avgStockDd,
    avgMfDd,
    sebiSafe: {
      label: safe.label,
      info: safe.info,
      disclaimer: getSebiDisclaimer()
    }
  };
}

// --- Macro Feature Extraction (stub) ---
function extractMacroFeatures(row) {
  // Example: extract GDP, inflation, etc. from macro row
  return {
    indicator: row.indicator || row.Indicator,
    date: row.date || row.Date,
    value: parseFloat(row.value || row.Value || '0')
  };
}

function processRollingMacro(macroRows, n = 100) {
  return macroRows.slice(-n).map(extractMacroFeatures);
}

// --- Save Features for ML/Analytics ---
function saveFeatures(features, file) {
  const toWrite = features.map(f => JSON.stringify(f)).join('\n') + '\n';
  fs.appendFileSync(file, toWrite);
}

module.exports = {
  extractNewsFeatures,
  processRollingNews,
  extractMarketFeatures,
  processRollingMarket,
  extractMfFeatures,
  extractPortfolioFeatures,
  compareStockVsMf,
  extractMacroFeatures,
  processRollingMacro,
  saveFeatures,
  NEWS_FILE,
  MARKET_FILE,
  MACRO_FILE
};
