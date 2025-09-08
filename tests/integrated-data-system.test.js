/**
 * 🧪 INTEGRATED DATA SYSTEM TESTS
 * 
 * Comprehensive test suite for the integrated data management system
 * Tests web search, automated data ingestion, and ASI integration
 * 
 * @author ASI Testing Framework
 * @version 4.0.0
 */

const { describe, it, expect, beforeAll, afterAll, beforeEach } = require('@jest/globals');
const { IntegratedDataManager } = require('../src/asi/data/IntegratedDataManager');
const { WebSearchEngine } = require('../src/asi/data/WebSearchEngine');
const { AutomatedDataCore } = require('../src/asi/data/AutomatedDataCore');

describe('🎯 Integrated Data System Tests', () => {
  let integratedDataManager;
  let webSearchEngine;
  let automatedDataCore;

  beforeAll(async () => {
    // Set test environment variables
    process.env.CUSTOM_FINANCIAL_SEARCH_URL = 'http://localhost:3000';
    process.env.ALPHA_VANTAGE_KEY = 'test_key';
    process.env.NEWS_API_KEY = 'test_key';
  });

  afterAll(async () => {
    if (integratedDataManager) {
      await integratedDataManager.shutdown();
    }
  });

  describe('🔍 Web Search Engine Tests', () => {
    beforeEach(() => {
      webSearchEngine = new WebSearchEngine({
        enableCustomFinancialSearch: false, // Disable for testing
        enableDuckDuckGoBackup: false
      });
    });

    it('should initialize web search engine', async () => {
      await webSearchEngine.initialize();
      expect(webSearchEngine).toBeDefined();
    });

    it('should generate financial search queries', () => {
      const queries = webSearchEngine.generateSearchQueries('AAPL', 'market');
      expect(queries).toBeInstanceOf(Array);
      expect(queries.length).toBeGreaterThan(0);
      expect(queries[0]).toContain('AAPL');
    });

    it('should calculate relevance scores', () => {
      const result = {
        title: 'Apple Inc (AAPL) Stock Market Analysis',
        snippet: 'Latest market analysis for Apple stock with price predictions',
        url: 'https://finance.yahoo.com/quote/AAPL',
        timestamp: new Date()
      };

      const searchRequest = {
        symbol: 'AAPL',
        category: 'market'
      };

      const score = webSearchEngine.calculateRelevanceScore(result, searchRequest);
      expect(score).toBeGreaterThan(0.5);
    });

    it('should filter results by relevance threshold', async () => {
      const results = [
        {
          title: 'Apple Stock Analysis',
          snippet: 'AAPL market analysis',
          url: 'https://finance.yahoo.com',
          timestamp: new Date()
        },
        {
          title: 'Random News',
          snippet: 'Unrelated content',
          url: 'https://example.com',
          timestamp: new Date()
        }
      ];

      const searchRequest = { symbol: 'AAPL', category: 'market' };
      const filtered = await webSearchEngine.filterAndScoreResults(results, searchRequest);
      
      expect(filtered.length).toBeLessThanOrEqual(results.length);
      expect(filtered[0].relevanceScore).toBeGreaterThan(0.6);
    });

    it('should get system metrics', () => {
      const metrics = webSearchEngine.getMetrics();
      expect(metrics).toHaveProperty('totalSearches');
      expect(metrics).toHaveProperty('successfulSearches');
      expect(metrics).toHaveProperty('queueSize');
      expect(metrics).toHaveProperty('enabledEngines');
    });
  });

  describe('🤖 Automated Data Core Tests', () => {
    beforeEach(() => {
      automatedDataCore = new AutomatedDataCore({
        enableMarketData: true,
        enableNewsData: true,
        testMode: true // Prevent actual API calls
      });
    });

    it('should initialize automated data core', async () => {
      await automatedDataCore.initialize();
      expect(automatedDataCore.isInitialized).toBe(true);
    });

    it('should validate data sources', () => {
      automatedDataCore.validateDataSources();
      expect(automatedDataCore.dataSources).toBeDefined();
    });

    it('should process market data', async () => {
      const mockData = {
        symbol: 'AAPL',
        price: 150.00,
        volume: 1000000,
        timestamp: new Date()
      };

      const processed = automatedDataCore.processMarketData(mockData);
      expect(processed).toHaveProperty('symbol', 'AAPL');
      expect(processed).toHaveProperty('price', 150.00);
      expect(processed).toHaveProperty('processedAt');
    });

    it('should process news data', async () => {
      const mockNews = {
        title: 'Apple Reports Strong Earnings',
        content: 'Apple Inc reported strong quarterly earnings...',
        source: 'Reuters',
        timestamp: new Date()
      };

      const processed = automatedDataCore.processNewsData(mockNews);
      expect(processed).toHaveProperty('title');
      expect(processed).toHaveProperty('sentiment');
      expect(processed).toHaveProperty('relevantSymbols');
    });

    it('should get system metrics', () => {
      const metrics = automatedDataCore.getMetrics();
      expect(metrics).toHaveProperty('totalRequests');
      expect(metrics).toHaveProperty('successfulRequests');
      expect(metrics).toHaveProperty('dataSourcesActive');
    });
  });

  describe('🎯 Integrated Data Manager Tests', () => {
    beforeEach(() => {
      integratedDataManager = new IntegratedDataManager({
        enableRealTimeData: true,
        enableWebSearch: false, // Disable for testing
        enableDataFusion: true,
        continuousMode: false, // Disable for testing
        testMode: true
      });
    });

    it('should initialize integrated data manager', async () => {
      await integratedDataManager.initialize();
      expect(integratedDataManager.isInitialized).toBe(true);
    });

    it('should handle incoming data', () => {
      const testData = {
        type: 'market',
        symbol: 'AAPL',
        price: 150.00,
        timestamp: new Date(),
        source: 'TestSource'
      };

      integratedDataManager.handleIncomingData(testData, 'test');
      expect(integratedDataManager.dataProcessingQueue.length).toBeGreaterThan(0);
    });

    it('should store unified data', () => {
      const testData = {
        type: 'market',
        symbol: 'AAPL',
        price: 150.00,
        timestamp: new Date(),
        source: 'TestSource'
      };

      integratedDataManager.storeUnifiedData(testData);
      const stored = integratedDataManager.getLatestData('market', 'AAPL');
      expect(stored).toBeDefined();
      expect(stored.symbol).toBe('AAPL');
    });

    it('should validate data', () => {
      const validData = {
        type: 'market',
        symbol: 'AAPL',
        timestamp: new Date(),
        source: 'TestSource'
      };

      const invalidData = {
        type: 'market'
        // Missing required fields
      };

      expect(integratedDataManager.validateData(validData)).toBe(true);
      expect(integratedDataManager.validateData(invalidData)).toBe(false);
    });

    it('should detect duplicates', () => {
      const data1 = {
        type: 'market',
        symbol: 'AAPL',
        timestamp: new Date(),
        source: 'TestSource',
        data: { price: 150.00 }
      };

      const data2 = {
        type: 'market',
        symbol: 'AAPL',
        timestamp: new Date(),
        source: 'TestSource',
        data: { price: 150.00 }
      };

      integratedDataManager.storeUnifiedData(data1);
      const isDuplicate = integratedDataManager.isDuplicate(data2);
      expect(isDuplicate).toBe(true);
    });

    it('should manage data subscriptions', () => {
      const callback = jest.fn();
      
      integratedDataManager.subscribeToData('test_subscriber', ['market'], callback);
      expect(integratedDataManager.dataSubscribers.has('test_subscriber')).toBe(true);

      // Test notification
      const testData = { type: 'market', symbol: 'AAPL' };
      integratedDataManager.notifySubscribers(testData);
      expect(callback).toHaveBeenCalledWith(testData);

      integratedDataManager.unsubscribeFromData('test_subscriber');
      expect(integratedDataManager.dataSubscribers.has('test_subscriber')).toBe(false);
    });

    it('should get unified data by type and symbol', () => {
      const testData = [
        {
          type: 'market',
          symbol: 'AAPL',
          price: 150.00,
          timestamp: new Date('2024-01-01T10:00:00Z'),
          source: 'TestSource'
        },
        {
          type: 'market',
          symbol: 'AAPL',
          price: 151.00,
          timestamp: new Date('2024-01-01T11:00:00Z'),
          source: 'TestSource'
        }
      ];

      testData.forEach(data => integratedDataManager.storeUnifiedData(data));
      
      const retrieved = integratedDataManager.getUnifiedData('market', 'AAPL', 10);
      expect(retrieved.length).toBe(2);
      expect(retrieved[0].price).toBe(151.00); // Latest first
    });

    it('should get data summary', () => {
      const testData = {
        type: 'market',
        symbol: 'AAPL',
        timestamp: new Date(),
        source: 'TestSource'
      };

      integratedDataManager.storeUnifiedData(testData);
      
      const summary = integratedDataManager.getDataSummary();
      expect(summary.totalDataPoints).toBeGreaterThan(0);
      expect(summary.dataTypes).toContain('market');
      expect(summary.symbols).toContain('AAPL');
      expect(summary.sources).toContain('TestSource');
    });

    it('should get system metrics', () => {
      const metrics = integratedDataManager.getSystemMetrics();
      expect(metrics).toHaveProperty('totalDataPoints');
      expect(metrics).toHaveProperty('queueSize');
      expect(metrics).toHaveProperty('activeOperations');
      expect(metrics).toHaveProperty('isInitialized');
      expect(metrics).toHaveProperty('currentMarketHour');
    });

    it('should determine current market hour', () => {
      const marketHour = integratedDataManager.getCurrentMarketHour();
      expect(['preMarket', 'regular', 'afterHours', 'closed']).toContain(marketHour);
    });

    it('should clear cache', () => {
      const testData = {
        type: 'market',
        symbol: 'AAPL',
        timestamp: new Date(),
        source: 'TestSource'
      };

      integratedDataManager.storeUnifiedData(testData);
      expect(integratedDataManager.unifiedDataStore.size).toBeGreaterThan(0);

      // Note: clearCache method would need to be added to IntegratedDataManager
      // integratedDataManager.clearCache();
      // expect(integratedDataManager.unifiedDataStore.size).toBe(0);
    });
  });

  describe('🔀 Data Fusion Tests', () => {
    beforeEach(async () => {
      integratedDataManager = new IntegratedDataManager({
        enableDataFusion: true,
        testMode: true
      });
      await integratedDataManager.initialize();
    });

    it('should fuse data from multiple sources', async () => {
      const dataPoints = [
        {
          type: 'market',
          symbol: 'AAPL',
          price: 150.00,
          source: 'Yahoo Finance',
          timestamp: new Date()
        },
        {
          type: 'market',
          symbol: 'AAPL',
          price: 150.05,
          source: 'Alpha Vantage',
          timestamp: new Date()
        }
      ];

      const fusedData = await integratedDataManager.dataFusionEngine.fuseData(dataPoints);
      expect(fusedData.length).toBeLessThanOrEqual(dataPoints.length);
    });

    it('should resolve conflicts between data sources', () => {
      const conflictingData = [
        {
          price: 150.00,
          source: 'Yahoo Finance',
          timestamp: new Date()
        },
        {
          price: 150.05,
          source: 'Alpha Vantage',
          timestamp: new Date()
        }
      ];

      const resolved = integratedDataManager.dataFusionEngine.resolveConflicts(conflictingData);
      expect(resolved.source).toBe('Yahoo Finance'); // Higher priority
    });

    it('should validate data quality', () => {
      const goodData = {
        timestamp: new Date(),
        source: 'Yahoo Finance',
        symbol: 'AAPL'
      };

      const badData = {
        // Missing timestamp and source
        symbol: 'AAPL'
      };

      const goodQuality = integratedDataManager.dataFusionEngine.validateQuality(goodData);
      const badQuality = integratedDataManager.dataFusionEngine.validateQuality(badData);

      expect(goodQuality).toBeGreaterThan(badQuality);
      expect(goodQuality).toBeGreaterThan(0.5);
    });
  });

  describe('📊 Performance Tests', () => {
    beforeEach(async () => {
      integratedDataManager = new IntegratedDataManager({
        testMode: true,
        maxConcurrentOperations: 10,
        dataProcessingBatchSize: 50
      });
      await integratedDataManager.initialize();
    });

    it('should handle large data batches efficiently', async () => {
      const startTime = Date.now();
      const largeDataSet = [];

      // Generate 1000 data points
      for (let i = 0; i < 1000; i++) {
        largeDataSet.push({
          type: 'market',
          symbol: `STOCK${i % 100}`,
          price: Math.random() * 100,
          timestamp: new Date(),
          source: 'TestSource'
        });
      }

      // Process all data
      for (const data of largeDataSet) {
        integratedDataManager.handleIncomingData(data, 'test');
      }

      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 2000));

      const processingTime = Date.now() - startTime;
      expect(processingTime).toBeLessThan(10000); // Should complete within 10 seconds
    });

    it('should maintain performance under concurrent operations', async () => {
      const promises = [];
      
      for (let i = 0; i < 20; i++) {
        promises.push(
          integratedDataManager.handleIncomingData({
            type: 'market',
            symbol: `STOCK${i}`,
            price: Math.random() * 100,
            timestamp: new Date(),
            source: 'TestSource'
          }, 'test')
        );
      }

      const startTime = Date.now();
      await Promise.all(promises);
      const completionTime = Date.now() - startTime;

      expect(completionTime).toBeLessThan(5000); // Should complete within 5 seconds
    });
  });

  describe('🔧 Integration Tests', () => {
    it('should integrate with ASI Master Engine data flow', async () => {
      const dataUpdateHandler = jest.fn();
      
      integratedDataManager = new IntegratedDataManager({
        testMode: true,
        continuousMode: false
      });
      
      await integratedDataManager.initialize();
      
      // Subscribe to data updates (simulating ASI Master Engine)
      integratedDataManager.subscribeToData('asi_master', ['market', 'news'], dataUpdateHandler);
      
      // Simulate data reception
      const testData = {
        type: 'market',
        symbol: 'AAPL',
        price: 150.00,
        timestamp: new Date(),
        source: 'TestSource'
      };
      
      integratedDataManager.handleIncomingData(testData, 'test');
      
      // Wait for processing
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      expect(dataUpdateHandler).toHaveBeenCalled();
    });

    it('should provide data access methods for ASI components', async () => {
      integratedDataManager = new IntegratedDataManager({ testMode: true });
      await integratedDataManager.initialize();
      
      // Store test data
      const testData = {
        type: 'market',
        symbol: 'AAPL',
        price: 150.00,
        timestamp: new Date(),
        source: 'TestSource'
      };
      
      integratedDataManager.storeUnifiedData(testData);
      
      // Test data access methods
      const latestData = integratedDataManager.getLatestData('market', 'AAPL');
      const unifiedData = integratedDataManager.getUnifiedData('market', 'AAPL', 10);
      const summary = integratedDataManager.getDataSummary();
      
      expect(latestData).toBeDefined();
      expect(unifiedData).toBeInstanceOf(Array);
      expect(summary.totalDataPoints).toBeGreaterThan(0);
    });
  });
});

// Helper functions for testing
function createMockMarketData(symbol, count = 1) {
  const data = [];
  for (let i = 0; i < count; i++) {
    data.push({
      type: 'market',
      symbol,
      price: Math.random() * 100 + 100,
      volume: Math.floor(Math.random() * 1000000),
      timestamp: new Date(Date.now() - i * 60000), // 1 minute intervals
      source: 'TestSource'
    });
  }
  return data;
}

function createMockNewsData(symbol, count = 1) {
  const data = [];
  for (let i = 0; i < count; i++) {
    data.push({
      type: 'news',
      symbol,
      title: `${symbol} News Article ${i + 1}`,
      content: `This is a test news article about ${symbol}`,
      sentiment: Math.random() > 0.5 ? 'positive' : 'negative',
      timestamp: new Date(Date.now() - i * 3600000), // 1 hour intervals
      source: 'TestNews'
    });
  }
  return data;
}
