/**
 * 🔍 WEB SEARCH ENGINE
 * 
 * Advanced web search and crawling system for ASI data collection
 * Uses custom financial search engine as primary source with DuckDuckGo backup
 * No dependency on fixed websites like Google, Bing, or Yahoo
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.1.0 - Custom Financial Search with DuckDuckGo Backup
 */

const EventEmitter = require('events');
const axios = require('axios');
const cheerio = require('cheerio');
const logger = require('../../utils/logger');
const { WebResearchAgent } = require('../WebResearchAgent');
const { AMCSearchIntegration } = require('../../finance_crawler/amc-search-integration');

class WebSearchEngine extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      // Search engines - Custom financial search as primary, DuckDuckGo as backup
      enableCustomFinancialSearch: options.enableCustomFinancialSearch !== false,
      enableDuckDuckGoBackup: options.enableDuckDuckGoBackup !== false,
      customFinancialSearchUrl: options.customFinancialSearchUrl || 'http://localhost:3000',
      
      // Search parameters
      maxResultsPerQuery: options.maxResultsPerQuery || 50,
      searchDepth: options.searchDepth || 3,
      relevanceThreshold: options.relevanceThreshold || 0.6,
      
      // Content extraction
      enableContentExtraction: options.enableContentExtraction !== false,
      maxContentLength: options.maxContentLength || 10000,
      
      // Rate limiting
      searchDelay: options.searchDelay || 2000, // 2 seconds between searches
      maxConcurrentSearches: options.maxConcurrentSearches || 5,
      
      ...options
    };
    
    // Search engines configuration - Custom financial search + DuckDuckGo backup
    this.searchEngines = {
      customFinancial: {
        name: 'Custom Financial Search Engine',
        url: this.config.customFinancialSearchUrl + '/search',
        enabled: this.config.enableCustomFinancialSearch,
        priority: 1 // Highest priority
      },
      duckduckgo: {
        name: 'DuckDuckGo Search (Backup)',
        url: 'https://duckduckgo.com/html/?q=',
        enabled: this.config.enableDuckDuckGoBackup,
        priority: 2 // Backup priority
      }
    };
    
    // Initialize DuckDuckGo research agent
    this.duckDuckGoAgent = new WebResearchAgent({
      searchUrl: 'https://duckduckgo.com/html/?q='
    });
    
    // Initialize AMC Search Integration
    this.amcSearchIntegration = new AMCSearchIntegration({
      indexName: 'amc-data',
      enabled: this.config.enableAMCSearch !== false
    });
    
    // Financial search queries
    this.financialQueries = {
      market: [
        'stock market news today',
        'market analysis {symbol}',
        'financial earnings {symbol}',
        'stock price prediction {symbol}',
        'market trends 2024'
      ],
      earnings: [
        '{symbol} earnings report',
        '{symbol} quarterly results',
        '{symbol} financial performance',
        '{symbol} revenue growth'
      ],
      news: [
        '{symbol} latest news',
        '{symbol} company updates',
        '{symbol} merger acquisition',
        '{symbol} analyst rating'
      ],
      analysis: [
        '{symbol} technical analysis',
        '{symbol} fundamental analysis',
        '{symbol} price target',
        '{symbol} investment thesis'
      ]
    };
    
    // Search results storage
    this.searchResults = new Map();
    this.contentCache = new Map();
    this.relevanceScores = new Map();
    
    // Active searches tracking
    this.activeSearches = new Set();
    this.searchQueue = [];
    
    // System metrics
    this.metrics = {
      totalSearches: 0,
      successfulSearches: 0,
      failedSearches: 0,
      totalResults: 0,
      averageRelevance: 0,
      contentExtracted: 0
    };
  }

  async initialize() {
    try {
      logger.info('🔍 Initializing Web Search Engine...');
      
      // Validate API keys
      this.validateSearchEngines();
      
      // Initialize AMC Search Integration
      if (this.amcSearchIntegration.config.enabled) {
        await this.amcSearchIntegration.initialize();
        logger.info('🏦 AMC Search Integration initialized');
      }
      
      // Start search processor
      this.startSearchProcessor();
      
      logger.info('✅ Web Search Engine initialized successfully');
      this.emit('initialized');
      
    } catch (error) {
      logger.error('❌ Web Search Engine initialization failed:', error);
      throw error;
    }
  }

  validateSearchEngines() {
    let validEngines = 0;
    
    for (const [engine, config] of Object.entries(this.searchEngines)) {
      if (config.enabled) {
        if (engine === 'customFinancial') {
          // Test custom financial search engine connectivity
          this.testCustomFinancialSearch().then(isValid => {
            if (!isValid) {
              logger.warn(`⚠️ Custom Financial Search Engine not accessible, will use DuckDuckGo only`);
              config.enabled = false;
            } else {
              logger.info(`✅ Custom Financial Search Engine validated`);
            }
          }).catch(() => {
            logger.warn(`⚠️ Custom Financial Search Engine validation failed`);
            config.enabled = false;
          });
          validEngines++;
        } else if (engine === 'duckduckgo') {
          // DuckDuckGo doesn't require API keys
          validEngines++;
          logger.info(`✅ DuckDuckGo backup search validated`);
        }
      }
    }
    
    if (validEngines === 0) {
      logger.warn('⚠️ No valid search engines configured');
    }
    
    logger.info(`🔍 ${validEngines} search engines configured`);
  }

  async testCustomFinancialSearch() {
    try {
      const response = await axios.get(this.searchEngines.customFinancial.url, {
        params: { q: 'test', index: 'chittorgarh' },
        timeout: 5000
      });
      return response.status === 200;
    } catch (error) {
      logger.debug(`Custom financial search test failed: ${error.message}`);
      return false;
    }
  }

  startSearchProcessor() {
    // Process search queue every 2 seconds
    setInterval(async () => {
      if (this.searchQueue.length > 0 && this.activeSearches.size < this.config.maxConcurrentSearches) {
        const searchRequest = this.searchQueue.shift();
        this.processSearch(searchRequest);
      }
    }, this.config.searchDelay);
    
    logger.info('🔄 Search processor started');
  }

  async searchFinancialData(symbols, categories = ['market', 'news'], options = {}) {
    try {
      logger.info(`🔍 Starting financial data search for ${symbols.length} symbols`);
      
      const searchPromises = [];
      
      for (const symbol of symbols) {
        for (const category of categories) {
          const queries = this.generateSearchQueries(symbol, category);
          
          for (const query of queries) {
            searchPromises.push(this.queueSearch({
              query,
              symbol,
              category,
              priority: options.priority || 'medium',
              extractContent: options.extractContent !== false
            }));
          }
        }
      }
      
      const results = await Promise.allSettled(searchPromises);
      const successfulResults = results.filter(r => r.status === 'fulfilled').map(r => r.value);
      
      logger.info(`✅ Financial data search completed: ${successfulResults.length} successful searches`);
      
      return this.aggregateSearchResults(successfulResults, symbols, categories);
      
    } catch (error) {
      logger.error('❌ Financial data search failed:', error);
      throw error;
    }
  }

  generateSearchQueries(symbol, category) {
    const templates = this.financialQueries[category] || this.financialQueries.market;
    return templates.map(template => template.replace('{symbol}', symbol));
  }

  async queueSearch(searchRequest) {
    return new Promise((resolve, reject) => {
      searchRequest.id = `search_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      searchRequest.timestamp = new Date();
      searchRequest.resolve = resolve;
      searchRequest.reject = reject;
      
      // Add to queue based on priority
      if (searchRequest.priority === 'high') {
        this.searchQueue.unshift(searchRequest);
      } else {
        this.searchQueue.push(searchRequest);
      }
      
      logger.debug(`📝 Queued search: ${searchRequest.query} (${searchRequest.category})`);
    });
  }

  async processSearch(searchRequest) {
    const startTime = Date.now();
    this.activeSearches.add(searchRequest.id);
    
    try {
      this.metrics.totalSearches++;
      
      // Try search engines in priority order (Custom Financial first, then DuckDuckGo)
      let searchResults = [];
      
      // Sort engines by priority
      const sortedEngines = Object.entries(this.searchEngines)
        .filter(([, config]) => config.enabled)
        .sort(([, a], [, b]) => (a.priority || 999) - (b.priority || 999));
      
      for (const [engine, config] of sortedEngines) {
        if (searchResults.length < this.config.maxResultsPerQuery) {
          try {
            const engineResults = await this.searchWithEngine(engine, searchRequest.query, searchRequest);
            searchResults.push(...engineResults);
            
            // If we got good results from primary engine, we might not need backup
            if (engine === 'customFinancial' && searchResults.length >= 5) {
              logger.debug(`🎯 Custom financial search provided ${searchResults.length} results, skipping backup`);
              break;
            }
          } catch (error) {
            logger.warn(`⚠️ Search engine ${engine} failed:`, error.message);
            // Continue to next engine (backup)
          }
        }
      }
      
      // Filter and score results
      let relevantResults = await this.filterAndScoreResults(searchResults, searchRequest);
      
      // Enhance with AMC data if relevant
      if (this.amcSearchIntegration.config.enabled && this.isAMCRelatedQuery(searchRequest.query)) {
        relevantResults = await this.amcSearchIntegration.enhanceFinancialSearch(
          searchRequest.query, 
          relevantResults
        );
        logger.debug(`🏦 Enhanced search with AMC data`);
      }
      
      // Extract content if requested
      if (searchRequest.extractContent && relevantResults.length > 0) {
        await this.extractContentFromResults(relevantResults);
      }
      
      // Store results
      const resultKey = `${searchRequest.symbol}_${searchRequest.category}`;
      if (!this.searchResults.has(resultKey)) {
        this.searchResults.set(resultKey, []);
      }
      
      this.searchResults.get(resultKey).push(...relevantResults);
      
      this.metrics.successfulSearches++;
      this.metrics.totalResults += relevantResults.length;
      
      // Emit search completed event
      this.emit('searchCompleted', {
        searchId: searchRequest.id,
        query: searchRequest.query,
        symbol: searchRequest.symbol,
        category: searchRequest.category,
        results: relevantResults,
        processingTime: Date.now() - startTime
      });
      
      searchRequest.resolve(relevantResults);
      
      logger.debug(`✅ Search completed: ${searchRequest.id} (${relevantResults.length} results)`);
      
    } catch (error) {
      this.metrics.failedSearches++;
      logger.error(`❌ Search failed: ${searchRequest.id}`, error.message);
      
      this.emit('searchError', {
        searchId: searchRequest.id,
        query: searchRequest.query,
        error: error.message
      });
      
      searchRequest.reject(error);
      
    } finally {
      this.activeSearches.delete(searchRequest.id);
    }
  }

  async searchWithEngine(engine, query, searchRequest) {
    const config = this.searchEngines[engine];
    
    switch (engine) {
      case 'customFinancial':
        return await this.searchCustomFinancial(query, searchRequest, config);
      case 'duckduckgo':
        return await this.searchDuckDuckGo(query, searchRequest, config);
      default:
        logger.warn(`Unknown search engine: ${engine}`);
        return [];
    }
  }

  async searchCustomFinancial(query, searchRequest, config) {
    try {
      logger.debug(`🎯 Searching custom financial engine for: ${query}`);
      
      // Determine the best index based on search category
      const index = this.getFinancialSearchIndex(searchRequest.category);
      
      const response = await axios.get(config.url, {
        params: {
          q: query,
          index: index
        },
        timeout: 10000
      });
      
      const results = response.data || [];
      
      return results.map(item => ({
        title: item.name || item.title || `${item.symbol || ''} Financial Data`,
        url: item.url || item.link || `#financial-${item.symbol || 'data'}`,
        snippet: item.description || item.snippet || this.generateFinancialSnippet(item),
        source: 'Custom Financial Search',
        timestamp: new Date(),
        financialData: item, // Keep original financial data
        relevanceScore: this.calculateFinancialRelevance(item, searchRequest)
      })).slice(0, this.config.maxResultsPerQuery);
      
    } catch (error) {
      logger.error('❌ Custom financial search failed:', error.message);
      throw error; // Let it fallback to DuckDuckGo
    }
  }

  async searchDuckDuckGo(query, searchRequest, config) {
    try {
      logger.debug(`🦆 Searching DuckDuckGo backup for: ${query}`);
      
      // Use the existing WebResearchAgent for DuckDuckGo search
      const results = await this.duckDuckGoAgent.webSearch(query);
      
      return results.map(item => ({
        title: item.title,
        url: item.link,
        snippet: item.snippet,
        source: 'DuckDuckGo (Backup)',
        timestamp: new Date(),
        relevanceScore: this.calculateWebRelevance(item, searchRequest)
      })).slice(0, this.config.maxResultsPerQuery);
      
    } catch (error) {
      logger.error('❌ DuckDuckGo search failed:', error.message);
      return [];
    }
  }

  getFinancialSearchIndex(category) {
    // Map search categories to appropriate financial indices
    const indexMap = {
      'market': 'chittorgarh',
      'earnings': 'earnings',
      'news': 'financial-news',
      'analysis': 'research-reports',
      'mutual-funds': 'mutual-funds'
    };
    
    return indexMap[category] || 'chittorgarh'; // Default to main financial index
  }

  generateFinancialSnippet(item) {
    // Generate a meaningful snippet from financial data
    const parts = [];
    
    if (item.symbol) parts.push(`Symbol: ${item.symbol}`);
    if (item.price) parts.push(`Price: ${item.price}`);
    if (item.change) parts.push(`Change: ${item.change}`);
    if (item.sector) parts.push(`Sector: ${item.sector}`);
    if (item.marketCap) parts.push(`Market Cap: ${item.marketCap}`);
    
    return parts.length > 0 ? parts.join(' | ') : 'Financial data available';
  }

  calculateFinancialRelevance(item, searchRequest) {
    let score = 0.8; // Base score for financial data
    
    const symbol = searchRequest.symbol?.toLowerCase();
    const category = searchRequest.category?.toLowerCase();
    
    // Symbol matching
    if (symbol && item.symbol?.toLowerCase().includes(symbol)) {
      score += 0.2;
    }
    
    // Category relevance
    if (category === 'market' && (item.price || item.marketCap)) {
      score += 0.1;
    } else if (category === 'earnings' && (item.revenue || item.earnings)) {
      score += 0.1;
    }
    
    return Math.min(score, 1.0);
  }

  calculateWebRelevance(item, searchRequest) {
    // Use existing relevance calculation for web results
    return this.calculateRelevanceScore(item, searchRequest);
  }

  isAMCRelatedQuery(query) {
    const amcKeywords = [
      'mutual fund', 'mf', 'amc', 'asset management', 'fund house',
      'sip', 'nav', 'scheme', 'portfolio', 'investment',
      'icici', 'hdfc', 'sbi', 'axis', 'kotak', 'aditya birla',
      'uti', 'nippon', 'mirae', 'tata', 'dsp', 'franklin',
      'invesco', 'motilal oswal', 'quant', 'ppfas', 'groww',
      'zerodha', 'bandhan', 'canara robeco', 'edelweiss'
    ];
    
    const queryLower = query.toLowerCase();
    return amcKeywords.some(keyword => queryLower.includes(keyword));
  }

  async filterAndScoreResults(results, searchRequest) {
    const relevantResults = [];
    
    for (const result of results) {
      const relevanceScore = this.calculateRelevanceScore(result, searchRequest);
      
      if (relevanceScore >= this.config.relevanceThreshold) {
        result.relevanceScore = relevanceScore;
        result.symbol = searchRequest.symbol;
        result.category = searchRequest.category;
        relevantResults.push(result);
      }
    }
    
    // Sort by relevance score
    relevantResults.sort((a, b) => b.relevanceScore - a.relevanceScore);
    
    // Update average relevance
    if (relevantResults.length > 0) {
      const avgRelevance = relevantResults.reduce((sum, r) => sum + r.relevanceScore, 0) / relevantResults.length;
      this.metrics.averageRelevance = (this.metrics.averageRelevance + avgRelevance) / 2;
    }
    
    return relevantResults.slice(0, this.config.maxResultsPerQuery);
  }

  calculateRelevanceScore(result, searchRequest) {
    let score = 0.5; // Base score
    
    const text = (result.title + ' ' + result.snippet).toLowerCase();
    const symbol = searchRequest.symbol.toLowerCase();
    const category = searchRequest.category.toLowerCase();
    
    // Symbol relevance (40% weight)
    if (text.includes(symbol)) score += 0.4;
    if (text.includes(symbol.replace(/\$/g, ''))) score += 0.2; // Without $ symbol
    
    // Category relevance (30% weight)
    const categoryKeywords = {
      market: ['market', 'stock', 'trading', 'price', 'chart'],
      earnings: ['earnings', 'revenue', 'profit', 'quarterly', 'financial'],
      news: ['news', 'announcement', 'update', 'report', 'press'],
      analysis: ['analysis', 'forecast', 'prediction', 'target', 'rating']
    };
    
    const keywords = categoryKeywords[category] || [];
    const matchedKeywords = keywords.filter(keyword => text.includes(keyword));
    score += (matchedKeywords.length / keywords.length) * 0.3;
    
    // Source credibility (20% weight)
    const credibleSources = [
      'reuters', 'bloomberg', 'wsj', 'marketwatch', 'yahoo', 'cnbc',
      'sec.gov', 'investor.', 'finance.', 'nasdaq', 'nyse',
      'custom financial search', 'chittorgarh' // Add our custom sources
    ];
    
    const url = result.url?.toLowerCase() || '';
    const source = result.source?.toLowerCase() || '';
    
    if (credibleSources.some(credibleSource => 
      url.includes(credibleSource) || source.includes(credibleSource)
    )) {
      score += 0.2;
    }
    
    // Bonus for custom financial search results
    if (source.includes('custom financial search')) {
      score += 0.1; // Extra credibility for our own financial data
    }
    
    // Recency (10% weight)
    const now = new Date();
    const daysDiff = (now - result.timestamp) / (1000 * 60 * 60 * 24);
    if (daysDiff <= 1) score += 0.1;
    else if (daysDiff <= 7) score += 0.05;
    
    return Math.min(score, 1.0);
  }

  async extractContentFromResults(results) {
    const extractionPromises = results.slice(0, 10).map(result => 
      this.extractContentFromUrl(result.url).then(content => {
        result.extractedContent = content;
        return result;
      }).catch(error => {
        logger.warn(`⚠️ Content extraction failed for ${result.url}:`, error.message);
        return result;
      })
    );
    
    await Promise.allSettled(extractionPromises);
    this.metrics.contentExtracted += results.filter(r => r.extractedContent).length;
  }

  async extractContentFromUrl(url) {
    // Check cache first
    if (this.contentCache.has(url)) {
      return this.contentCache.get(url);
    }
    
    try {
      const response = await axios.get(url, {
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        },
        timeout: 10000,
        maxContentLength: this.config.maxContentLength
      });
      
      const $ = cheerio.load(response.data);
      
      // Remove unwanted elements
      $('script, style, nav, header, footer, aside, .ad, .advertisement').remove();
      
      // Extract main content
      let content = '';
      const contentSelectors = ['article', 'main', '.content', '.post', '.article-body', 'p'];
      
      for (const selector of contentSelectors) {
        const elements = $(selector);
        if (elements.length > 0) {
          content = elements.text().trim();
          break;
        }
      }
      
      // Fallback to body text
      if (!content) {
        content = $('body').text().trim();
      }
      
      // Clean and limit content
      content = content.replace(/\s+/g, ' ').substring(0, this.config.maxContentLength);
      
      // Cache the content
      this.contentCache.set(url, content);
      
      return content;
      
    } catch (error) {
      throw new Error(`Content extraction failed: ${error.message}`);
    }
  }

  aggregateSearchResults(results, symbols, categories) {
    const aggregated = {
      symbols,
      categories,
      totalResults: results.reduce((sum, r) => sum + r.length, 0),
      resultsBySymbol: {},
      resultsByCategory: {},
      topResults: [],
      timestamp: new Date()
    };
    
    // Group by symbol
    for (const symbol of symbols) {
      aggregated.resultsBySymbol[symbol] = [];
    }
    
    // Group by category
    for (const category of categories) {
      aggregated.resultsByCategory[category] = [];
    }
    
    // Process all results
    const allResults = results.flat();
    
    for (const result of allResults) {
      if (result.symbol && aggregated.resultsBySymbol[result.symbol]) {
        aggregated.resultsBySymbol[result.symbol].push(result);
      }
      
      if (result.category && aggregated.resultsByCategory[result.category]) {
        aggregated.resultsByCategory[result.category].push(result);
      }
    }
    
    // Get top results across all searches
    aggregated.topResults = allResults
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 50);
    
    return aggregated;
  }

  // Public API methods
  async searchSymbol(symbol, categories = ['market', 'news'], options = {}) {
    return await this.searchFinancialData([symbol], categories, options);
  }

  async searchMarketNews(options = {}) {
    const generalQueries = [
      'stock market news today',
      'market analysis today',
      'financial markets update',
      'market trends 2024'
    ];
    
    const searchPromises = generalQueries.map(query => 
      this.queueSearch({
        query,
        category: 'market',
        priority: options.priority || 'medium',
        extractContent: options.extractContent !== false
      })
    );
    
    const results = await Promise.allSettled(searchPromises);
    return results.filter(r => r.status === 'fulfilled').map(r => r.value).flat();
  }

  getSearchResults(symbol, category) {
    const key = `${symbol}_${category}`;
    return this.searchResults.get(key) || [];
  }

  getAllSearchResults() {
    const allResults = {};
    for (const [key, results] of this.searchResults) {
      allResults[key] = results;
    }
    return allResults;
  }

  getMetrics() {
    return {
      ...this.metrics,
      queueSize: this.searchQueue.length,
      activeSearches: this.activeSearches.size,
      cachedResults: this.searchResults.size,
      cachedContent: this.contentCache.size,
      enabledEngines: Object.values(this.searchEngines).filter(e => e.enabled).length
    };
  }

  clearCache() {
    this.searchResults.clear();
    this.contentCache.clear();
    this.relevanceScores.clear();
    logger.info('🧹 Search cache cleared');
  }
}

module.exports = { WebSearchEngine };
