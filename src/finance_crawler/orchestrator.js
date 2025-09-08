// Distributed crawl orchestrator with error handling and reporting
const sites = require('./config/sites.final.json');
const { crawlSite } = require('./crawler');
const os = require('os');
const cluster = require('cluster');
const fs = require('fs');
const path = require('path');

const NUM_WORKERS = Math.max(2, os.cpus().length - 1);
const LOG_PATH = path.join(__dirname, 'crawl_errors.log');

function logError(site, error) {
  const entry = `[${new Date().toISOString()}] ${site.name}: ${error.message || error}\n`;
  fs.appendFileSync(LOG_PATH, entry);
}

async function orchestrateCrawling() {
  const now = new Date();
  const sitesToCrawl = sites.filter(site => {
    // Example: crawl all hourly or due sites (expand with scheduler logic)
    return site.crawl_frequency === 'hourly';
  });
  if (cluster.isMaster) {
    console.log(`Master process orchestrating ${sitesToCrawl.length} sites with ${NUM_WORKERS} workers.`);
    for (let i = 0; i < NUM_WORKERS; i++) {
      cluster.fork();
    }
    let idx = 0;
    for (const id in cluster.workers) {
      cluster.workers[id].on('message', msg => {
        if (msg.done) {
          idx++;
          if (idx < sitesToCrawl.length) {
            cluster.workers[id].send({ site: sitesToCrawl[idx] });
          } else {
            cluster.workers[id].kill();
          }
        } else if (msg.error) {
          logError(msg.site, msg.error);
        }
      });
      cluster.workers[id].send({ site: sitesToCrawl[idx++] });
    }
  } else {
    process.on('message', async msg => {
      if (msg.site) {
        try {
          await crawlSite(msg.site);
          process.send({ done: true });
        } catch (error) {
          process.send({ error, site: msg.site });
        }
      }
    });
  }
}

if (require.main === module) {
  orchestrateCrawling();
}

module.exports = { orchestrateCrawling, logError };
