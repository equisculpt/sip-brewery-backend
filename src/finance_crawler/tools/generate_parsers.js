// Parser generator: create placeholder parser files for each site in sites.final.json if not present
const fs = require('fs');
const path = require('path');

const configPath = path.join(__dirname, '../config/sites.final.json');
const parsersDir = path.join(__dirname, '../parsers');

const sites = JSON.parse(fs.readFileSync(configPath, 'utf-8'));

if (!fs.existsSync(parsersDir)) fs.mkdirSync(parsersDir);

const template = (site) => `// Parser for ${site.name}\n// Tags: ${site.tags.join(', ')} | Frequency: ${site.crawl_frequency}\nconst cheerio = require('cheerio');\nmodule.exports = async function(html) {\n  // TODO: Implement parser for ${site.name}\n  // Use cheerio to extract relevant data\n  // Example: return []\n  return [];\n};\n`;

let created = 0;
for (const site of sites) {
  const fname = site.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.js';
  const fpath = path.join(parsersDir, fname);
  if (!fs.existsSync(fpath)) {
    fs.writeFileSync(fpath, template(site));
    created++;
  }
}
console.log(`Parser generation complete. ${created} new parser(s) created.`);
