/**
 * 🌐 AUTOMATED DATA CORE
 * 
 * Core data ingestion and web search system for continuous ASI data feeding
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Autonomous Data Intelligence
 */

const EventEmitter = require('events');
const axios = require('axios');
const schedule = require('node-cron');
const logger = require('../../utils/logger');

class AutomatedDataCore extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      enableMarketData: options.enableMarketData !== false,
      enableNewsData: options.enableNewsData !== false,
      enableEarningsData: options.enableEarningsData !== false,
      enableSocialSentiment: options.enableSocialSentiment !== false,
      
      // Update frequencies (in minutes)
      marketDataFrequency: options.marketDataFrequency || 1,
      newsDataFrequency: options.newsDataFrequency || 5,
      earningsDataFrequency: options.earningsDataFrequency || 60,
      sentimentDataFrequency: options.sentimentDataFrequency || 15,
      
      // Rate limiting
      maxRequestsPerMinute: options.maxRequestsPerMinute || 60,
      requestDelay: options.requestDelay || 1000,
      
      ...options
    };
    
    // Data sources configuration
    this.dataSources = {
      market: {
        apis: [
          { name: 'Yahoo Finance', url: 'https://query1.finance.yahoo.com/v8/finance/chart', key: null },
          { name: 'Alpha Vantage', url: 'https://www.alphavantage.co/query', key: process.env.ALPHA_VANTAGE_KEY },
          { name: 'IEX Cloud', url: 'https://cloud.iexapis.com/stable', key: process.env.IEX_CLOUD_KEY }
        ],
        symbols: ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ']
      },
      
      news: {
        sources: [
          { name: 'NewsAPI', url: 'https://newsapi.org/v2', key: process.env.NEWS_API_KEY },
          { name: 'RSS Feeds', urls: [
            'https://feeds.finance.yahoo.com/rss/2.0/headline',
            'https://www.marketwatch.com/rss/topstories'
          ]}
        ]
      },
      
      earnings: {
        sources: [
          { name: 'FMP', url: 'https://financialmodelingprep.com/api/v3/earning_calendar', key: process.env.FMP_KEY }
        ]
      },
      
      sentiment: {
        sources: [
          { name: 'Reddit', url: 'https://www.reddit.com/r/investing', key: null },
          { name: 'StockTwits', url: 'https://api.stocktwits.com/api/2', key: null }
        ]
      }
    };
    
    // Data storage
    this.dataCache = new Map();
    this.requestQueue = [];
    this.activeRequests = new Set();
    this.rateLimiter = new Map();
    this.scheduledJobs = new Map();
    
    // System metrics
    this.metrics = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      dataPointsCollected: 0,
      averageResponseTime: 0,
      lastUpdateTime: null,
      uptime: Date.now()
    };
    
    this.isRunning = false;
  }

  async initialize() {
    try {
      logger.info('🌐 Initializing Automated Data Core...');
      
      this.initializeRateLimiters();
      this.startRequestProcessor();
      await this.scheduleDataCollection();
      
      this.isRunning = true;
      logger.info('✅ Automated Data Core initialized successfully');
      
      this.emit('initialized');
      
    } catch (error) {
      logger.error('❌ Automated Data Core initialization failed:', error);
      throw error;
    }
  }

  initializeRateLimiters() {
    const sources = ['Yahoo Finance', 'Alpha Vantage', 'IEX Cloud', 'NewsAPI', 'FMP', 'Reddit', 'StockTwits'];
    
    for (const source of sources) {
      this.rateLimiter.set(source, {
        requests: 0,
        resetTime: Date.now() + 60000,
        maxRequests: this.config.maxRequestsPerMinute
      });
    }
    
    logger.info('⏱️ Rate limiters initialized');
  }

  startRequestProcessor() {
    setInterval(async () => {
      if (this.requestQueue.length > 0 && this.activeRequests.size < 10) {
        const request = this.requestQueue.shift();
        this.processRequest(request);
      }
    }, 1000);
    
    logger.info('🔄 Request processor started');
  }

  async scheduleDataCollection() {
    // Market data collection (every minute during market hours)
    if (this.config.enableMarketData) {
      const marketDataJob = schedule.schedule('* 9-16 * * 1-5', async () => {
        await this.collectMarketData();
      });
      this.scheduledJobs.set('marketData', marketDataJob);
    }
    
    // News data collection (every 5 minutes)
    if (this.config.enableNewsData) {
      const newsDataJob = schedule.schedule(`*/${this.config.newsDataFrequency} * * * *`, async () => {
        await this.collectNewsData();
      });
      this.scheduledJobs.set('newsData', newsDataJob);
    }
    
    // Earnings data collection (every hour)
    if (this.config.enableEarningsData) {
      const earningsDataJob = schedule.schedule(`0 */${Math.floor(this.config.earningsDataFrequency / 60)} * * *`, async () => {
        await this.collectEarningsData();
      });
      this.scheduledJobs.set('earningsData', earningsDataJob);
    }
    
    // Sentiment data collection (every 15 minutes)
    if (this.config.enableSocialSentiment) {
      const sentimentDataJob = schedule.schedule(`*/${this.config.sentimentDataFrequency} * * * *`, async () => {
        await this.collectSentimentData();
      });
      this.scheduledJobs.set('sentimentData', sentimentDataJob);
    }
    
    logger.info('📅 Data collection jobs scheduled');
  }

  async collectMarketData() {
    try {
      logger.info('📊 Collecting market data...');
      
      const symbols = this.dataSources.market.symbols;
      const apis = this.dataSources.market.apis.filter(api => api.key || api.name === 'Yahoo Finance');
      
      for (const symbol of symbols) {
        for (const api of apis) {
          if (this.canMakeRequest(api.name)) {
            this.queueRequest({
              type: 'market',
              source: api.name,
              symbol,
              url: this.buildMarketDataUrl(api, symbol),
              priority: 'high'
            });
          }
        }
      }
      
    } catch (error) {
      logger.error('❌ Market data collection failed:', error);
    }
  }

  async collectNewsData() {
    try {
      logger.info('📰 Collecting news data...');
      
      const sources = this.dataSources.news.sources;
      
      for (const source of sources) {
        if (source.key && this.canMakeRequest(source.name)) {
          this.queueRequest({
            type: 'news',
            source: source.name,
            url: this.buildNewsDataUrl(source),
            priority: 'medium'
          });
        }
        
        if (source.urls) {
          for (const url of source.urls) {
            this.queueRequest({
              type: 'news',
              source: 'RSS',
              url,
              priority: 'medium'
            });
          }
        }
      }
      
    } catch (error) {
      logger.error('❌ News data collection failed:', error);
    }
  }

  async collectEarningsData() {
    try {
      logger.info('💰 Collecting earnings data...');
      
      const sources = this.dataSources.earnings.sources.filter(source => source.key);
      
      for (const source of sources) {
        if (this.canMakeRequest(source.name)) {
          this.queueRequest({
            type: 'earnings',
            source: source.name,
            url: this.buildEarningsDataUrl(source),
            priority: 'medium'
          });
        }
      }
      
    } catch (error) {
      logger.error('❌ Earnings data collection failed:', error);
    }
  }

  async collectSentimentData() {
    try {
      logger.info('😊 Collecting sentiment data...');
      
      const sources = this.dataSources.sentiment.sources;
      const symbols = this.dataSources.market.symbols.slice(0, 5);
      
      for (const source of sources) {
        if (this.canMakeRequest(source.name)) {
          for (const symbol of symbols) {
            this.queueRequest({
              type: 'sentiment',
              source: source.name,
              symbol,
              url: this.buildSentimentDataUrl(source, symbol),
              priority: 'low'
            });
          }
        }
      }
      
    } catch (error) {
      logger.error('❌ Sentiment data collection failed:', error);
    }
  }

  queueRequest(request) {
    request.id = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    request.timestamp = new Date();
    
    if (request.priority === 'high') {
      this.requestQueue.unshift(request);
    } else {
      this.requestQueue.push(request);
    }
    
    logger.debug(`📝 Queued request: ${request.type} from ${request.source}`);
  }

  async processRequest(request) {
    const startTime = Date.now();
    this.activeRequests.add(request.id);
    
    try {
      this.metrics.totalRequests++;
      
      const response = await this.makeHttpRequest(request);
      const processedData = await this.processResponse(request, response);
      
      await this.storeData(request, processedData);
      
      this.metrics.successfulRequests++;
      this.metrics.dataPointsCollected += Array.isArray(processedData) ? processedData.length : 1;
      
      this.emit('dataReceived', {
        type: request.type,
        source: request.source,
        symbol: request.symbol,
        data: processedData,
        timestamp: new Date()
      });
      
      logger.debug(`✅ Processed request: ${request.id}`);
      
    } catch (error) {
      this.metrics.failedRequests++;
      logger.error(`❌ Request failed: ${request.id}`, error.message);
      
      this.emit('requestError', {
        request,
        error: error.message,
        timestamp: new Date()
      });
      
    } finally {
      this.activeRequests.delete(request.id);
      this.updateRateLimit(request.source);
      
      const responseTime = Date.now() - startTime;
      this.metrics.averageResponseTime = (this.metrics.averageResponseTime + responseTime) / 2;
      this.metrics.lastUpdateTime = new Date();
    }
  }

  async makeHttpRequest(request) {
    const config = {
      method: 'GET',
      url: request.url,
      timeout: 10000,
      headers: {
        'User-Agent': 'ASI-DataIngestion/1.0',
        'Accept': 'application/json'
      }
    };
    
    if (request.source === 'NewsAPI' && process.env.NEWS_API_KEY) {
      config.headers['X-API-Key'] = process.env.NEWS_API_KEY;
    }
    
    const response = await axios(config);
    return response.data;
  }

  async processResponse(request, response) {
    switch (request.type) {
      case 'market':
        return this.processMarketData(request, response);
      case 'news':
        return this.processNewsData(request, response);
      case 'earnings':
        return this.processEarningsData(request, response);
      case 'sentiment':
        return this.processSentimentData(request, response);
      default:
        return response;
    }
  }

  processMarketData(request, response) {
    if (request.source === 'Yahoo Finance') {
      const result = response.chart?.result?.[0];
      if (result) {
        const quotes = result.indicators?.quote?.[0];
        const timestamps = result.timestamp;
        
        return timestamps?.map((timestamp, index) => ({
          symbol: request.symbol,
          timestamp: new Date(timestamp * 1000),
          open: quotes?.open?.[index],
          high: quotes?.high?.[index],
          low: quotes?.low?.[index],
          close: quotes?.close?.[index],
          volume: quotes?.volume?.[index],
          source: request.source
        })) || [];
      }
    }
    
    if (request.source === 'Alpha Vantage') {
      const timeSeries = response['Time Series (1min)'] || response['Time Series (Daily)'];
      if (timeSeries) {
        return Object.entries(timeSeries).map(([timestamp, data]) => ({
          symbol: request.symbol,
          timestamp: new Date(timestamp),
          open: parseFloat(data['1. open']),
          high: parseFloat(data['2. high']),
          low: parseFloat(data['3. low']),
          close: parseFloat(data['4. close']),
          volume: parseInt(data['5. volume']),
          source: request.source
        }));
      }
    }
    
    return [];
  }

  processNewsData(request, response) {
    if (request.source === 'NewsAPI') {
      return response.articles?.map(article => ({
        title: article.title,
        description: article.description,
        content: article.content,
        url: article.url,
        publishedAt: new Date(article.publishedAt),
        source: article.source?.name,
        relevance: this.calculateNewsRelevance(article)
      })) || [];
    }
    
    return [];
  }

  processEarningsData(request, response) {
    if (Array.isArray(response)) {
      return response.map(earning => ({
        symbol: earning.symbol,
        date: new Date(earning.date),
        time: earning.time,
        epsEstimate: earning.epsEstimate,
        epsActual: earning.epsActual,
        source: request.source
      }));
    }
    
    return [];
  }

  processSentimentData(request, response) {
    return [];
  }

  async storeData(request, data) {
    if (!data || (Array.isArray(data) && data.length === 0)) {
      return;
    }
    
    const key = `${request.type}_${request.source}_${request.symbol || 'general'}`;
    
    if (!this.dataCache.has(key)) {
      this.dataCache.set(key, []);
    }
    
    const cached = this.dataCache.get(key);
    
    if (Array.isArray(data)) {
      cached.push(...data);
    } else {
      cached.push(data);
    }
    
    // Limit cache size
    if (cached.length > 1000) {
      cached.splice(0, cached.length - 1000);
    }
  }

  // URL builders
  buildMarketDataUrl(api, symbol) {
    switch (api.name) {
      case 'Yahoo Finance':
        return `${api.url}/${symbol}?interval=1m&range=1d`;
      case 'Alpha Vantage':
        return `${api.url}?function=TIME_SERIES_INTRADAY&symbol=${symbol}&interval=1min&apikey=${api.key}`;
      case 'IEX Cloud':
        return `${api.url}/stock/${symbol}/quote?token=${api.key}`;
      default:
        return api.url;
    }
  }

  buildNewsDataUrl(source) {
    switch (source.name) {
      case 'NewsAPI':
        return `${source.url}/everything?q=finance OR stocks OR market&sortBy=publishedAt&apiKey=${source.key}`;
      default:
        return source.url;
    }
  }

  buildEarningsDataUrl(source) {
    const today = new Date().toISOString().split('T')[0];
    const nextWeek = new Date(Date.now() + 7 * 24 * 60 * 60 * 1000).toISOString().split('T')[0];
    
    return `${source.url}?from=${today}&to=${nextWeek}&apikey=${source.key}`;
  }

  buildSentimentDataUrl(source, symbol) {
    switch (source.name) {
      case 'Reddit':
        return `${source.url}/search.json?q=${symbol}&sort=new&limit=100`;
      case 'StockTwits':
        return `${source.url}/streams/symbol/${symbol}.json`;
      default:
        return source.url;
    }
  }

  // Utility methods
  canMakeRequest(sourceName) {
    const limiter = this.rateLimiter.get(sourceName);
    if (!limiter) return true;
    
    if (Date.now() > limiter.resetTime) {
      limiter.requests = 0;
      limiter.resetTime = Date.now() + 60000;
    }
    
    return limiter.requests < limiter.maxRequests;
  }

  updateRateLimit(sourceName) {
    const limiter = this.rateLimiter.get(sourceName);
    if (limiter) {
      limiter.requests++;
    }
  }

  calculateNewsRelevance(article) {
    const financialKeywords = [
      'stock', 'market', 'earnings', 'revenue', 'profit', 'investment',
      'trading', 'portfolio', 'dividend', 'merger', 'acquisition'
    ];
    
    const text = (article.title + ' ' + article.description).toLowerCase();
    const matches = financialKeywords.filter(keyword => text.includes(keyword));
    
    return Math.min(matches.length / 5, 1.0);
  }

  // Getter methods
  getDataCache() {
    return this.dataCache;
  }

  getMetrics() {
    return {
      ...this.metrics,
      queueSize: this.requestQueue.length,
      activeRequests: this.activeRequests.size,
      cacheSize: this.dataCache.size,
      isRunning: this.isRunning
    };
  }

  getLatestData(type, source, symbol = null) {
    const key = `${type}_${source}_${symbol || 'general'}`;
    const data = this.dataCache.get(key);
    return data ? data.slice(-10) : []; // Return last 10 records
  }

  async shutdown() {
    try {
      logger.info('🛑 Shutting down Automated Data Core...');
      
      // Stop all scheduled jobs
      for (const [name, job] of this.scheduledJobs) {
        job.destroy();
        logger.info(`📅 Stopped ${name} job`);
      }
      
      this.isRunning = false;
      logger.info('✅ Automated Data Core shutdown completed');
      
    } catch (error) {
      logger.error('❌ Automated Data Core shutdown failed:', error);
    }
  }
}

module.exports = { AutomatedDataCore };
