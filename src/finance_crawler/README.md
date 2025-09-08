# Finance-Only Web Crawler & Index

## Quick Start

1. Install dependencies:
   npm install puppeteer cheerio node-fetch @elastic/elasticsearch express

2. Start services:
   docker-compose up

3. Run the crawler:
   docker-compose run finance-crawler

4. Start the search API:
   node api.js

5. Search:
   GET http://localhost:3000/search?q=IPO&index=chittorgarh

## Adding More Sites
- Add the site to `config/sites.json`.
- Add a parser in `parsers/` (e.g., `parsers/zerodha-varsity.js`).
- The crawler will auto-load and index it.
