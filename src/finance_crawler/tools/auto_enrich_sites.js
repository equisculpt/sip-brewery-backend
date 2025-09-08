// Advanced enrichment: auto-populate tags and crawl_frequency using heuristics, update sites.enriched.json
const fs = require('fs');
const path = require('path');

const inputPath = path.join(__dirname, '../config/sites.enriched.json');
const outputPath = path.join(__dirname, '../config/sites.final.json');

const sites = JSON.parse(fs.readFileSync(inputPath, 'utf-8'));

// Heuristic rules for tags and crawl_frequency
function getTagsAndFrequency(site) {
  const name = (site.name || '').toLowerCase();
  const type = (site.type || '').toLowerCase();
  const url = (site.url || '').toLowerCase();
  let tags = [];
  let crawl_frequency = '';

  // Tag heuristics
  if (type.includes('regulatory') || name.includes('sebi') || name.includes('rbi') || name.includes('amfi') || name.includes('irdai') || name.includes('pfrda')) {
    tags.push('regulatory');
    crawl_frequency = 'daily';
  }
  if (type.includes('exchange') || name.includes('nse') || name.includes('bse') || name.includes('mcx') || name.includes('cdsl') || name.includes('nsdl')) {
    tags.push('exchange');
    crawl_frequency = 'hourly';
  }
  if (type.includes('broker') || type.includes('fintech') || type.includes('neobank') || type.includes('robo') || type.includes('wealthtech') || type.includes('investment platform')) {
    tags.push('platform');
    crawl_frequency = 'daily';
  }
  if (type.includes('news') || type.includes('media')) {
    tags.push('news');
    crawl_frequency = 'hourly';
  }
  if (type.includes('analytics') || type.includes('research') || type.includes('data')) {
    tags.push('analytics');
    crawl_frequency = 'daily';
  }
  if (type.includes('education') || type.includes('mooc') || type.includes('certification') || name.includes('academy') || name.includes('nism')) {
    tags.push('education');
    crawl_frequency = 'weekly';
  }
  if (type.includes('insurance')) {
    tags.push('insurance');
    crawl_frequency = 'weekly';
  }
  if (type.includes('blog')) {
    tags.push('blog');
    crawl_frequency = 'weekly';
  }
  if (type.includes('government')) {
    tags.push('government');
    crawl_frequency = 'weekly';
  }
  if (type.includes('aggregator')) {
    tags.push('aggregator');
    crawl_frequency = 'daily';
  }
  if (type.includes('amc') || type.includes('mutual fund')) {
    tags.push('amc');
    crawl_frequency = 'daily';
  }
  if (type.includes('rta')) {
    tags.push('rta');
    crawl_frequency = 'daily';
  }
  if (type.includes('bonds')) {
    tags.push('bonds');
    crawl_frequency = 'daily';
  }
  if (type.includes('private equity') || type.includes('hedge fund') || type.includes('investment bank')) {
    tags.push('institutional');
    crawl_frequency = 'weekly';
  }
  // Default fallbacks
  if (!crawl_frequency) crawl_frequency = 'weekly';
  if (tags.length === 0) tags.push('other');
  return { tags: Array.from(new Set(tags)), crawl_frequency };
}

const enriched = sites.map(site => {
  const { tags, crawl_frequency } = getTagsAndFrequency(site);
  return {
    ...site,
    tags,
    crawl_frequency
  };
});

fs.writeFileSync(outputPath, JSON.stringify(enriched, null, 2));
console.log(`Auto-enrichment complete. Final file written to ${outputPath}`);
