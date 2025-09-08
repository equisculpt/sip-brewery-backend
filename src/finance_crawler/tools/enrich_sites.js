// Enrich each entry in sites.deduped.json with metadata fields for API endpoint, crawl frequency, tags, and auth requirements
const fs = require('fs');
const path = require('path');

const inputPath = path.join(__dirname, '../config/sites.deduped.json');
const outputPath = path.join(__dirname, '../config/sites.enriched.json');

const sites = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));

const enriched = sites.map(site => ({
  ...site,
  api_endpoint: '', // Fill with official/public API endpoint if available
  crawl_frequency: '', // e.g., 'daily', 'hourly', 'weekly', 'realtime'
  tags: [], // e.g., ['regulatory', 'news', 'broker', 'education', 'fintech']
  auth: {
    required: false, // true if API key/login is needed
    type: '', // e.g., 'api_key', 'oauth', 'basic', ''
    notes: '' // Any special instructions
  }
}));

fs.writeFileSync(outputPath, JSON.stringify(enriched, null, 2));
console.log(`Enrichment complete. Enriched file written to ${outputPath}`);
