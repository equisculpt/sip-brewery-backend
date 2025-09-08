const fetch = require('node-fetch');
const fs = require('fs');
const path = require('path');
const sites = require('./config/sites.final.json');
const indexData = require('./indexer');

// site now includes tags, crawl_frequency, api_endpoint, and auth metadata
async function crawlSite(site) {
  try {
    const parserPath = path.join(__dirname, 'parsers', `${site.name.toLowerCase().replace(/ /g, '-')}.js`);
    if (!fs.existsSync(parserPath)) {
      console.log(`No parser for ${site.name}, skipping.`);
      return;
    }
    const parser = require(parserPath);
    const res = await fetch(site.url, { headers: { 'User-Agent': 'Mozilla/5.0' } });
    const html = await res.text();
    const items = await parser(html);
    if (items && items.length) {
      await indexData(site.name.toLowerCase(), items);
      console.log(`Indexed ${items.length} items from ${site.name}`);
    }
  } catch (e) {
    console.error(`Error crawling ${site.name}:`, e.message);
  }
}

async function main() {
  for (const site of sites) {
    await crawlSite(site);
  }
}
if (require.main === module) main();
