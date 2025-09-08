// Deduplicate sites.json by both name and url, outputting a cleaned version
const fs = require('fs');
const path = require('path');

const sitesPath = path.join(__dirname, '../config/sites.json');
const outputPath = path.join(__dirname, '../config/sites.deduped.json');

const sites = JSON.parse(fs.readFileSync(sitesPath, 'utf-8'));

const seen = new Set();
const deduped = [];

for (const entry of sites) {
  // Use both name and url as a deduplication key (case-insensitive, trimmed)
  const name = (entry.name || '').trim().toLowerCase();
  const url = (entry.url || '').trim().toLowerCase();
  const key = name + '::' + url;
  if (!seen.has(key)) {
    seen.add(key);
    deduped.push(entry);
  }
}

fs.writeFileSync(outputPath, JSON.stringify(deduped, null, 2));
console.log(`Deduplication complete. Cleaned file written to ${outputPath}`);
