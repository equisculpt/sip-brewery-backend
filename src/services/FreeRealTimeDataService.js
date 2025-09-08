/**
 * 🌐 FREE REAL-TIME DATA INTEGRATION SERVICE
 * 
 * Enterprise-grade real-time market data aggregation using free APIs
 * - Multiple free data sources with intelligent failover
 * - Real-time price streaming and portfolio updates
 * - Market event detection and notifications
 * - Data quality validation and cleansing
 * - Caching and rate limit management
 * 
 * @author Senior Data Architect (35 years experience)
 * @version 1.0.0 - Free Real-Time Data Engine
 */

const axios = require('axios');
const WebSocket = require('ws');
const EventEmitter = require('events');
const logger = require('../utils/logger');
const NodeCache = require('node-cache');

class FreeRealTimeDataService extends EventEmitter {
  constructor() {
    super();
    
    // Data sources configuration
    this.dataSources = new Map();
    this.activeConnections = new Map();
    this.dataCache = new NodeCache({ stdTTL: 300 }); // 5 minutes cache
    this.subscriptions = new Map();
    this.rateLimits = new Map();
    
    // Data quality metrics
    this.dataQuality = {
      totalRequests: 0,
      successfulRequests: 0,
      failedRequests: 0,
      averageLatency: 0,
      lastUpdate: null
    };
    
    this.initializeDataSources();
    this.startHealthMonitoring();
  }

  /**
   * Initialize free data sources
   */
  initializeDataSources() {
    // Yahoo Finance (Free, No API Key Required)
    this.dataSources.set('yahoo', {
      name: 'Yahoo Finance',
      baseUrl: 'https://query1.finance.yahoo.com/v8/finance/chart',
      rateLimit: 2000, // 2 seconds between requests
      reliability: 0.85,
      dataTypes: ['stocks', 'indices', 'forex', 'commodities'],
      active: true
    });

    // Alpha Vantage (Free tier: 5 API calls per minute, 500 per day)
    this.dataSources.set('alphavantage', {
      name: 'Alpha Vantage',
      baseUrl: 'https://www.alphavantage.co/query',
      apiKey: process.env.ALPHA_VANTAGE_API_KEY || 'demo', // Free demo key
      rateLimit: 12000, // 12 seconds between requests (5 per minute)
      reliability: 0.90,
      dataTypes: ['stocks', 'forex', 'crypto'],
      active: true
    });

    // Finnhub (Free tier: 60 API calls per minute)
    this.dataSources.set('finnhub', {
      name: 'Finnhub',
      baseUrl: 'https://finnhub.io/api/v1',
      apiKey: process.env.FINNHUB_API_KEY || 'demo',
      rateLimit: 1000, // 1 second between requests
      reliability: 0.88,
      dataTypes: ['stocks', 'forex', 'crypto'],
      active: true
    });

    // IEX Cloud (Free tier: 500,000 core data credits per month)
    this.dataSources.set('iex', {
      name: 'IEX Cloud',
      baseUrl: 'https://cloud.iexapis.com/stable',
      apiKey: process.env.IEX_API_KEY || 'pk_test',
      rateLimit: 100, // 100ms between requests
      reliability: 0.92,
      dataTypes: ['stocks', 'indices'],
      active: true
    });

    // NSE India (Free, No API Key Required)
    this.dataSources.set('nse', {
      name: 'NSE India',
      baseUrl: 'https://www.nseindia.com/api',
      rateLimit: 1000,
      reliability: 0.80,
      dataTypes: ['indian_stocks', 'indices', 'derivatives'],
      active: true
    });

    // BSE India (Free, No API Key Required)
    this.dataSources.set('bse', {
      name: 'BSE India',
      baseUrl: 'https://api.bseindia.com',
      rateLimit: 2000,
      reliability: 0.75,
      dataTypes: ['indian_stocks', 'indices'],
      active: true
    });

    // CoinGecko (Free tier: 10-50 calls/minute)
    this.dataSources.set('coingecko', {
      name: 'CoinGecko',
      baseUrl: 'https://api.coingecko.com/api/v3',
      rateLimit: 1200, // 1.2 seconds between requests
      reliability: 0.87,
      dataTypes: ['crypto'],
      active: true
    });

    logger.info('✅ Free real-time data sources initialized');
  }

  /**
   * Get real-time price data with intelligent source selection
   */
  async getRealTimePrice(symbol, exchange = 'NSE') {
    try {
      const cacheKey = `price_${symbol}_${exchange}`;
      const cachedData = this.dataCache.get(cacheKey);
      
      if (cachedData) {
        return cachedData;
      }

      // Select best data source for the symbol
      const dataSource = this.selectOptimalDataSource(symbol, exchange);
      
      let priceData = null;
      let attempts = 0;
      const maxAttempts = 3;

      while (!priceData && attempts < maxAttempts) {
        try {
          priceData = await this.fetchPriceFromSource(symbol, exchange, dataSource);
          break;
        } catch (error) {
          attempts++;
          logger.warn(`Attempt ${attempts} failed for ${dataSource.name}: ${error.message}`);
          
          if (attempts < maxAttempts) {
            // Try next best source
            const fallbackSource = this.getNextBestSource(symbol, exchange, dataSource);
            if (fallbackSource) {
              dataSource = fallbackSource;
            }
          }
        }
      }

      if (!priceData) {
        throw new Error(`Failed to fetch price data for ${symbol} after ${maxAttempts} attempts`);
      }

      // Validate and enrich data
      const enrichedData = await this.validateAndEnrichData(priceData, symbol, exchange);
      
      // Cache the data
      this.dataCache.set(cacheKey, enrichedData, 60); // 1 minute cache
      
      // Update quality metrics
      this.updateDataQualityMetrics(true);
      
      return enrichedData;

    } catch (error) {
      this.updateDataQualityMetrics(false);
      logger.error(`❌ Failed to get real-time price for ${symbol}:`, error);
      throw error;
    }
  }

  /**
   * Fetch price from specific data source
   */
  async fetchPriceFromSource(symbol, exchange, source) {
    await this.respectRateLimit(source.name);

    switch (source.name) {
      case 'Yahoo Finance':
        return await this.fetchFromYahoo(symbol, exchange);
      case 'Alpha Vantage':
        return await this.fetchFromAlphaVantage(symbol, source.apiKey);
      case 'Finnhub':
        return await this.fetchFromFinnhub(symbol, source.apiKey);
      case 'IEX Cloud':
        return await this.fetchFromIEX(symbol, source.apiKey);
      case 'NSE India':
        return await this.fetchFromNSE(symbol);
      case 'BSE India':
        return await this.fetchFromBSE(symbol);
      case 'CoinGecko':
        return await this.fetchFromCoinGecko(symbol);
      default:
        throw new Error(`Unknown data source: ${source.name}`);
    }
  }

  /**
   * Fetch from Yahoo Finance
   */
  async fetchFromYahoo(symbol, exchange) {
    const yahooSymbol = this.convertToYahooSymbol(symbol, exchange);
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${yahooSymbol}`;
    
    const response = await axios.get(url, {
      timeout: 5000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
      }
    });

    const data = response.data.chart.result[0];
    const meta = data.meta;
    const quote = data.indicators.quote[0];
    
    return {
      symbol: symbol,
      exchange: exchange,
      price: meta.regularMarketPrice,
      change: meta.regularMarketPrice - meta.previousClose,
      changePercent: ((meta.regularMarketPrice - meta.previousClose) / meta.previousClose) * 100,
      volume: quote.volume[quote.volume.length - 1],
      high: meta.regularMarketDayHigh,
      low: meta.regularMarketDayLow,
      open: quote.open[quote.open.length - 1],
      previousClose: meta.previousClose,
      timestamp: new Date(meta.regularMarketTime * 1000).toISOString(),
      source: 'Yahoo Finance'
    };
  }

  /**
   * Fetch from NSE India
   */
  async fetchFromNSE(symbol) {
    const url = `https://www.nseindia.com/api/quote-equity?symbol=${symbol}`;
    
    const response = await axios.get(url, {
      timeout: 5000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br'
      }
    });

    const data = response.data;
    
    return {
      symbol: symbol,
      exchange: 'NSE',
      price: data.priceInfo.lastPrice,
      change: data.priceInfo.change,
      changePercent: data.priceInfo.pChange,
      volume: data.priceInfo.totalTradedVolume,
      high: data.priceInfo.intraDayHighLow.max,
      low: data.priceInfo.intraDayHighLow.min,
      open: data.priceInfo.open,
      previousClose: data.priceInfo.previousClose,
      timestamp: new Date().toISOString(),
      source: 'NSE India'
    };
  }

  /**
   * Fetch from Alpha Vantage
   */
  async fetchFromAlphaVantage(symbol, apiKey) {
    const url = `https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=${symbol}&apikey=${apiKey}`;
    
    const response = await axios.get(url, { timeout: 5000 });
    const quote = response.data['Global Quote'];
    
    if (!quote) {
      throw new Error('No data returned from Alpha Vantage');
    }
    
    return {
      symbol: symbol,
      exchange: 'US',
      price: parseFloat(quote['05. price']),
      change: parseFloat(quote['09. change']),
      changePercent: parseFloat(quote['10. change percent'].replace('%', '')),
      volume: parseInt(quote['06. volume']),
      high: parseFloat(quote['03. high']),
      low: parseFloat(quote['04. low']),
      open: parseFloat(quote['02. open']),
      previousClose: parseFloat(quote['08. previous close']),
      timestamp: new Date().toISOString(),
      source: 'Alpha Vantage'
    };
  }

  /**
   * Start real-time streaming for portfolio
   */
  async startRealTimeStreaming(portfolio) {
    const streamId = `stream_${portfolio.id}_${Date.now()}`;
    
    logger.info(`🔄 Starting real-time streaming for portfolio ${portfolio.id}`);
    
    // Get unique symbols from portfolio
    const symbols = this.extractSymbolsFromPortfolio(portfolio);
    
    // Set up periodic updates for each symbol
    const updateInterval = setInterval(async () => {
      try {
        const updates = await this.getBatchPriceUpdates(symbols);
        
        // Emit real-time updates
        this.emit('priceUpdate', {
          portfolio_id: portfolio.id,
          updates: updates,
          timestamp: new Date().toISOString()
        });
        
        // Update portfolio value
        const updatedValue = await this.calculatePortfolioValue(portfolio, updates);
        
        this.emit('portfolioUpdate', {
          portfolio_id: portfolio.id,
          total_value: updatedValue.total,
          change: updatedValue.change,
          change_percent: updatedValue.changePercent,
          timestamp: new Date().toISOString()
        });
        
      } catch (error) {
        logger.error('❌ Real-time streaming error:', error);
      }
    }, 30000); // Update every 30 seconds
    
    // Store streaming session
    this.subscriptions.set(streamId, {
      portfolio_id: portfolio.id,
      symbols: symbols,
      interval: updateInterval,
      start_time: new Date()
    });
    
    return streamId;
  }

  /**
   * Get batch price updates
   */
  async getBatchPriceUpdates(symbols) {
    const updates = {};
    const batchSize = 5; // Process 5 symbols at a time to respect rate limits
    
    for (let i = 0; i < symbols.length; i += batchSize) {
      const batch = symbols.slice(i, i + batchSize);
      
      const batchPromises = batch.map(async (symbol) => {
        try {
          const priceData = await this.getRealTimePrice(symbol.symbol, symbol.exchange);
          return { symbol: symbol.symbol, data: priceData };
        } catch (error) {
          logger.warn(`Failed to get price for ${symbol.symbol}: ${error.message}`);
          return { symbol: symbol.symbol, data: null };
        }
      });
      
      const batchResults = await Promise.all(batchPromises);
      
      batchResults.forEach(result => {
        if (result.data) {
          updates[result.symbol] = result.data;
        }
      });
      
      // Small delay between batches
      if (i + batchSize < symbols.length) {
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
    
    return updates;
  }

  /**
   * Market event detection
   */
  async detectMarketEvents(priceData) {
    const events = [];
    
    // Significant price movement (>5%)
    if (Math.abs(priceData.changePercent) > 5) {
      events.push({
        type: 'significant_movement',
        symbol: priceData.symbol,
        change_percent: priceData.changePercent,
        severity: Math.abs(priceData.changePercent) > 10 ? 'high' : 'medium',
        timestamp: priceData.timestamp
      });
    }
    
    // High volume spike (>2x average)
    const avgVolume = await this.getAverageVolume(priceData.symbol);
    if (priceData.volume > avgVolume * 2) {
      events.push({
        type: 'volume_spike',
        symbol: priceData.symbol,
        current_volume: priceData.volume,
        average_volume: avgVolume,
        multiplier: priceData.volume / avgVolume,
        timestamp: priceData.timestamp
      });
    }
    
    // Circuit breaker levels
    if (Math.abs(priceData.changePercent) > 20) {
      events.push({
        type: 'circuit_breaker',
        symbol: priceData.symbol,
        change_percent: priceData.changePercent,
        severity: 'critical',
        timestamp: priceData.timestamp
      });
    }
    
    return events;
  }

  /**
   * Data quality validation
   */
  async validateAndEnrichData(priceData, symbol, exchange) {
    // Basic validation
    if (!priceData.price || priceData.price <= 0) {
      throw new Error('Invalid price data');
    }
    
    // Enrich with additional metrics
    const enrichedData = {
      ...priceData,
      quality_score: this.calculateDataQualityScore(priceData),
      market_cap: await this.getMarketCap(symbol, exchange),
      pe_ratio: await this.getPERatio(symbol, exchange),
      beta: await this.getBeta(symbol, exchange),
      volatility: await this.calculateVolatility(symbol, exchange),
      technical_indicators: await this.calculateTechnicalIndicators(symbol, exchange)
    };
    
    // Detect market events
    const events = await this.detectMarketEvents(enrichedData);
    if (events.length > 0) {
      enrichedData.market_events = events;
      
      // Emit market events
      events.forEach(event => {
        this.emit('marketEvent', event);
      });
    }
    
    return enrichedData;
  }

  /**
   * Calculate technical indicators
   */
  async calculateTechnicalIndicators(symbol, exchange) {
    try {
      // Get historical data for calculations
      const historicalData = await this.getHistoricalData(symbol, exchange, 50);
      
      return {
        sma_20: this.calculateSMA(historicalData, 20),
        sma_50: this.calculateSMA(historicalData, 50),
        rsi: this.calculateRSI(historicalData, 14),
        macd: this.calculateMACD(historicalData),
        bollinger_bands: this.calculateBollingerBands(historicalData, 20, 2)
      };
      
    } catch (error) {
      logger.warn(`Failed to calculate technical indicators for ${symbol}: ${error.message}`);
      return {};
    }
  }

  /**
   * Rate limit management
   */
  async respectRateLimit(sourceName) {
    const lastRequest = this.rateLimits.get(sourceName);
    const source = this.dataSources.get(sourceName.toLowerCase().replace(' ', ''));
    
    if (lastRequest && source) {
      const timeSinceLastRequest = Date.now() - lastRequest;
      const minInterval = source.rateLimit;
      
      if (timeSinceLastRequest < minInterval) {
        const waitTime = minInterval - timeSinceLastRequest;
        await new Promise(resolve => setTimeout(resolve, waitTime));
      }
    }
    
    this.rateLimits.set(sourceName, Date.now());
  }

  /**
   * Select optimal data source
   */
  selectOptimalDataSource(symbol, exchange) {
    const availableSources = Array.from(this.dataSources.values())
      .filter(source => source.active)
      .sort((a, b) => b.reliability - a.reliability);
    
    // Select based on symbol type and exchange
    if (exchange === 'NSE' || exchange === 'BSE') {
      const indianSources = availableSources.filter(s => 
        s.dataTypes.includes('indian_stocks') || s.dataTypes.includes('indices')
      );
      if (indianSources.length > 0) return indianSources[0];
    }
    
    return availableSources[0];
  }

  /**
   * Health monitoring
   */
  startHealthMonitoring() {
    setInterval(() => {
      this.checkDataSourceHealth();
    }, 60000); // Check every minute
  }

  async checkDataSourceHealth() {
    for (const [key, source] of this.dataSources.entries()) {
      try {
        // Simple health check - try to fetch a test symbol
        await this.fetchPriceFromSource('RELIANCE', 'NSE', source);
        source.active = true;
        source.lastHealthCheck = new Date();
      } catch (error) {
        source.active = false;
        source.lastError = error.message;
        logger.warn(`❌ Data source ${source.name} health check failed: ${error.message}`);
      }
    }
  }

  /**
   * Get service status
   */
  getServiceStatus() {
    const activeSources = Array.from(this.dataSources.values()).filter(s => s.active);
    
    return {
      status: activeSources.length > 0 ? 'healthy' : 'unhealthy',
      active_sources: activeSources.length,
      total_sources: this.dataSources.size,
      active_subscriptions: this.subscriptions.size,
      cache_size: this.dataCache.keys().length,
      data_quality: this.dataQuality,
      sources: Array.from(this.dataSources.entries()).map(([key, source]) => ({
        name: source.name,
        active: source.active,
        reliability: source.reliability,
        last_health_check: source.lastHealthCheck,
        last_error: source.lastError
      }))
    };
  }

  // Helper methods
  convertToYahooSymbol(symbol, exchange) {
    if (exchange === 'NSE') return `${symbol}.NS`;
    if (exchange === 'BSE') return `${symbol}.BO`;
    return symbol;
  }

  extractSymbolsFromPortfolio(portfolio) {
    // Mock implementation - extract symbols from portfolio holdings
    return [
      { symbol: 'RELIANCE', exchange: 'NSE' },
      { symbol: 'TCS', exchange: 'NSE' },
      { symbol: 'INFY', exchange: 'NSE' }
    ];
  }

  updateDataQualityMetrics(success) {
    this.dataQuality.totalRequests++;
    if (success) {
      this.dataQuality.successfulRequests++;
    } else {
      this.dataQuality.failedRequests++;
    }
    this.dataQuality.lastUpdate = new Date();
  }

  calculateDataQualityScore(priceData) {
    let score = 100;
    
    // Deduct points for missing data
    if (!priceData.volume) score -= 10;
    if (!priceData.high || !priceData.low) score -= 10;
    if (!priceData.open) score -= 5;
    
    // Deduct points for stale data
    const dataAge = Date.now() - new Date(priceData.timestamp).getTime();
    if (dataAge > 300000) score -= 20; // 5 minutes
    
    return Math.max(score, 0);
  }

  calculateSMA(data, period) {
    if (data.length < period) return null;
    const sum = data.slice(-period).reduce((acc, val) => acc + val.close, 0);
    return sum / period;
  }

  calculateRSI(data, period = 14) {
    if (data.length < period + 1) return null;
    
    let gains = 0;
    let losses = 0;
    
    for (let i = data.length - period; i < data.length; i++) {
      const change = data[i].close - data[i - 1].close;
      if (change > 0) gains += change;
      else losses -= change;
    }
    
    const avgGain = gains / period;
    const avgLoss = losses / period;
    const rs = avgGain / avgLoss;
    
    return 100 - (100 / (1 + rs));
  }
}

module.exports = { FreeRealTimeDataService };
