/**
 * WebResearchAgent.js
 *
 * Handles autonomous web search, summarization, and extraction for ASI learning.
 * Uses DuckDuckGo HTML scraping for cost-free, API-free search.
 * Requires: node-fetch, cheerio
 */
const fetch = require('node-fetch');
const cheerio = require('cheerio');

class WebResearchAgent {
  /**
   * Initialize the agent (no API keys needed for DuckDuckGo)
   */
  constructor(config = {}) {
    this.config = config;
    this.searchUrl = 'https://duckduckgo.com/html/?q=';
  }

  /**
   * Perform a web search and return summarized results
   * @param {string} query
   * @returns {Promise<{summary: string, links: string[], raw: any}>}
   */
  async searchAndSummarize(query) {
    const results = await this.webSearch(query);
    const summary = this.summarizeResults(results);
    const links = results.map(r => r.link);
    return { summary, links, raw: results };
  }

  /**
   * Perform a web search using DuckDuckGo HTML and parse results
   * @param {string} query
   * @returns {Promise<Array<{title: string, link: string, snippet: string}>>}
   */
  async webSearch(query) {
    const url = `${this.searchUrl}${encodeURIComponent(query)}`;
    const response = await fetch(url, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (compatible; ASI-WebResearchBot/1.0)'
      }
    });
    if (!response.ok) throw new Error('DuckDuckGo web search failed');
    const html = await response.text();
    const $ = cheerio.load(html);
    const results = [];
    $('.result').each((i, elem) => {
      const title = $(elem).find('.result__title').text().trim();
      const link = $(elem).find('.result__url').attr('href');
      const snippet = $(elem).find('.result__snippet').text().trim();
      if (title && link) {
        results.push({ title, link, snippet });
      }
    });
    return results;
  }

  /**
   * Summarize search results (simple extractive summary for now)
   * @param {Array<{title: string, link: string, snippet: string}>} results
   * @returns {string}
   */
  summarizeResults(results) {
    if (!results.length) return 'No results found.';
    // Extract top 3 snippets
    return results.slice(0, 3).map(r => r.snippet).join(' ');
  }

  /**
   * Extract potential curriculum topics from web results
   * @param {string} query
   * @returns {Promise<string[]>} List of topics
   */
  async extractTopicsFromWeb(query) {
    const results = await this.webSearch(query);
    // Naive: extract unique words from titles (could use NLP for better extraction)
    const topics = [];
    results.forEach(r => {
      if (r.title) {
        r.title.split(/[^\w]+/).forEach(word => {
          if (word.length > 3 && !topics.includes(word)) topics.push(word);
        });
      }
    });
    return topics.slice(0, 5);
  }
}

module.exports = { WebResearchAgent };
