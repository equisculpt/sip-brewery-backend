// ASI-powered AutoParserAgent: generates and improves parsers using unified ASI system
const { processRequest } = require('./ASIMasterEngine');
const fs = require('fs');
const path = require('path');

/**
 * Attempts to auto-generate or repair a parser for the given site using ASI.
 * @param {object} site - Site config object from sites.final.json
 * @param {string} html - Raw HTML to analyze
 * @returns {Promise<string>} - Path to generated parser file
 */
async function autoGenerateParser(site, html) {
  const parserName = site.name.toLowerCase().replace(/[^a-z0-9]+/g, '-') + '.js';
  const parserPath = path.join(__dirname, '../finance_crawler/parsers', parserName);
  // Use ASI: ask for extraction logic based on html and site metadata
  const prompt = `Generate a Node.js Cheerio parser that extracts all relevant finance data from the following HTML. Site metadata: ${JSON.stringify(site)}. HTML sample: ${html.slice(0, 1200)}...`;
  const asiResult = await processRequest({
    type: 'parser_generation',
    prompt,
    site,
    html_sample: html.slice(0, 6000)
  });
  if (asiResult && asiResult.parser_code) {
    fs.writeFileSync(parserPath, asiResult.parser_code);
    return parserPath;
  }
  throw new Error('ASI failed to generate parser');
}

module.exports = { autoGenerateParser };
