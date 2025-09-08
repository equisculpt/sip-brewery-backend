// agiPipelineOrchestrator.js
// Orchestrates rolling feature extraction, ML prediction, and scenario alerting for AGI
const { scrapeAndStoreNews } = require('./agiNewsScraper');
const { processRollingNews, processRollingMarket, processRollingMacro, saveFeatures, NEWS_FILE, MARKET_FILE, MACRO_FILE } = require('./featureExtractor');
const pythonMlClient = require('./pythonMlClient');
const fs = require('fs');
const path = require('path');

// --- 1. Rolling Feature Extraction ---
async function runFeatureExtraction() {
  // News
  await scrapeAndStoreNews();
  const newsFeatures = processRollingNews(100);
  saveFeatures(newsFeatures, NEWS_FILE.replace('.jsonl', '_features.jsonl'));
  // Market & Macro: assume you have loaded parsed rows from Bhavcopy/macro CSVs
  let marketRows = [];
  let macroRows = [];
  try {
    if (fs.existsSync(path.join(__dirname, 'data', 'latest_bhavcopy.json'))) {
      marketRows = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'latest_bhavcopy.json'), 'utf-8'));
    }
    if (fs.existsSync(path.join(__dirname, 'data', 'latest_macro.json'))) {
      macroRows = JSON.parse(fs.readFileSync(path.join(__dirname, 'data', 'latest_macro.json'), 'utf-8'));
    }
  } catch (e) { /* ignore */ }
  const marketFeatures = processRollingMarket(marketRows, 100);
  const macroFeatures = processRollingMacro(macroRows, 100);
  saveFeatures(marketFeatures, MARKET_FILE);
  saveFeatures(macroFeatures, MACRO_FILE);
  return { newsFeatures, marketFeatures, macroFeatures };
}

// --- 2. ML Prediction & Explainability ---
async function runMlPrediction(features) {
  // Example: send features to Python ML microservice for prediction
  try {
    const prediction = await pythonMlClient.predictMarket(features.marketFeatures);
    const explain = await pythonMlClient.explainPrediction(features.marketFeatures);
    return { prediction, explain };
  } catch (err) {
    console.error('ML prediction/explain error:', err.message);
    return null;
  }
}

// --- 3. Scenario/Alert Triggers ---
function runScenarioAlerts(features) {
  // Example: alert if big negative news or >3% market drop
  const bigNegativeNews = features.newsFeatures.filter(n => n.sentiment < 0 && n.length > 20);
  const largeDrop = features.marketFeatures.find(m => m.pctChange < -3);
  if (bigNegativeNews.length > 0) {
    console.log('ALERT: Big negative news detected:', bigNegativeNews.map(n => n.headline));
  }
  if (largeDrop) {
    console.log('ALERT: Large market drop detected:', largeDrop.symbol, largeDrop.pctChange);
  }
  // Extend for macro shocks, user/fund-specific triggers, etc.
}

// --- Orchestrate Full Pipeline ---
async function runAgiPipeline() {
  const features = await runFeatureExtraction();
  const mlResults = await runMlPrediction(features);
  runScenarioAlerts(features);
  // Optionally: store predictions, update user dashboards, trigger retraining, etc.
}

if (require.main === module) {
  runAgiPipeline();
}

module.exports = { runAgiPipeline };
