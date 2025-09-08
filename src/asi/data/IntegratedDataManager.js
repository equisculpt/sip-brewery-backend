/**
 * 🎯 INTEGRATED DATA MANAGER
 * 
 * Central hub that orchestrates all data sources and feeds ASI system continuously
 * Combines automated data ingestion, web search, and intelligent data processing
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Unified Data Intelligence
 */

const EventEmitter = require('events');
const schedule = require('node-cron');
const logger = require('../../utils/logger');

// Import data components
const { AutomatedDataCore } = require('./AutomatedDataCore');
const { WebSearchEngine } = require('./WebSearchEngine');
const { IntegratedDocumentSystem } = require('../../finance_crawler/integrated-document-system');
const { FreeSocialMediaIntegration } = require('../../finance_crawler/free-social-media-integration');

class IntegratedDataManager extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      // Data source priorities
      enableRealTimeData: options.enableRealTimeData !== false,
      enableWebSearch: options.enableWebSearch !== false,
      enableDataFusion: options.enableDataFusion !== false,
      
      // Update strategies
      continuousMode: options.continuousMode !== false,
      smartScheduling: options.smartScheduling !== false,
      adaptiveFrequency: options.adaptiveFrequency !== false,
      
      // Data quality
      enableDataValidation: options.enableDataValidation !== false,
      enableDuplicateDetection: options.enableDuplicateDetection !== false,
      enableAnomalyDetection: options.enableAnomalyDetection !== false,
      
      // Performance
      maxConcurrentOperations: options.maxConcurrentOperations || 20,
      dataProcessingBatchSize: options.dataProcessingBatchSize || 100,
      
      ...options
    };
    
    // Core components
    this.automatedDataCore = null;
    this.webSearchEngine = null;
    this.documentSystem = null;
    
    // Data storage and processing
    this.unifiedDataStore = new Map();
    this.dataProcessingQueue = [];
    this.dataFusionEngine = null;
    
    // Intelligent scheduling
    this.marketHours = {
      preMarket: { start: 4, end: 9.5 }, // 4:00 AM - 9:30 AM EST
      regular: { start: 9.5, end: 16 }, // 9:30 AM - 4:00 PM EST
      afterHours: { start: 16, end: 20 } // 4:00 PM - 8:00 PM EST
    };
    
    this.scheduledTasks = new Map();
    this.activeOperations = new Set();
    
    // System metrics
    this.metrics = {
      totalDataPoints: 0,
      dataSourcesActive: 0,
      searchQueriesExecuted: 0,
      dataFusionOperations: 0,
      averageDataLatency: 0,
      dataQualityScore: 0,
      systemUptime: Date.now()
    };
    
    // Data subscribers (ASI components that need data)
    this.dataSubscribers = new Map();
    
    this.isInitialized = false;
  }

  async initialize() {
    try {
      logger.info('🎯 Initializing Integrated Data Manager...');
      
      // Initialize core components
      await this.initializeDataComponents();
      
      // Setup data fusion engine
      await this.initializeDataFusion();
      
      // Initialize document system
      await this.documentSystem.initialize();
      
      // Setup document system event handlers
      this.setupDocumentSystemEventHandlers();
      
      // Initialize social media integration
      await this.socialMediaIntegration.initialize();
      
      // Setup social media integration event handlers
      this.setupSocialMediaEventHandlers();
      
      // Setup intelligent scheduling
      await this.setupIntelligentScheduling();
      
      // Setup data processing pipeline
      this.setupDataProcessingPipeline();
      
      // Setup event handlers
      this.setupEventHandlers();
      
      // Start continuous data flow
      if (this.config.continuousMode) {
        await this.startContinuousDataFlow();
      }
      
      this.isInitialized = true;
      logger.info('✅ Integrated Data Manager initialized successfully');
      
      this.emit('initialized');
      
    } catch (error) {
      logger.error('❌ Integrated Data Manager initialization failed:', error);
      throw error;
    }
  }

  async initializeDataComponents() {
    // Initialize Automated Data Core
    this.automatedDataCore = new AutomatedDataCore({
      enableMarketData: true,
      enableNewsData: true,
      enableEarningsData: true,
      enableSocialSentiment: true,
      ...this.config
    });
    
    await this.automatedDataCore.initialize();
    this.metrics.dataSourcesActive++;
    
    // Initialize Web Search Engine with custom financial search
    if (this.config.enableWebSearch) {
      this.webSearchEngine = new WebSearchEngine({
        enableCustomFinancialSearch: true,
        enableDuckDuckGoBackup: true,
        enableAMCSearchIntegration: true,
        ...this.config.webSearchOptions
      });
      
      await this.webSearchEngine.initialize();
    }
    
    // Initialize integrated document system
    this.documentSystem = new IntegratedDocumentSystem({
      enableMonitoring: true,
      enableAnalysis: true,
      enableAlerting: true,
      asiIntegration: true
    });

    // Initialize free social media integration
    this.socialMediaIntegration = new FreeSocialMediaIntegration({
      enableRealTimeTracking: true,
      enablePhilosophyAnalysis: true,
      enableSentimentAnalysis: true,
      enableTrendAnalysis: true,
      enableASIIntegration: true
    });
    
    logger.info('🔧 Data components initialized');
  }

  async initializeDataFusion() {
    if (!this.config.enableDataFusion) return;
    
    this.dataFusionEngine = {
      // Combine data from multiple sources
      fuseData: async (dataPoints) => {
        const fusedData = new Map();
        
        for (const dataPoint of dataPoints) {
          const key = this.generateDataKey(dataPoint);
          
          if (fusedData.has(key)) {
            // Merge with existing data
            const existing = fusedData.get(key);
            fusedData.set(key, this.mergeDataPoints(existing, dataPoint));
          } else {
            fusedData.set(key, dataPoint);
          }
        }
        
        return Array.from(fusedData.values());
      },
      
      // Resolve conflicts between data sources
      resolveConflicts: (dataPoints) => {
        // Prioritize by source reliability
        const sourcePriority = {
          'Yahoo Finance': 0.9,
          'Alpha Vantage': 0.85,
          'IEX Cloud': 0.8,
          'Google': 0.75,
          'Bing': 0.7,
          'RSS': 0.6
        };
        
        return dataPoints.sort((a, b) => 
          (sourcePriority[b.source] || 0.5) - (sourcePriority[a.source] || 0.5)
        )[0];
      },
      
      // Validate data quality
      validateQuality: (dataPoint) => {
        let qualityScore = 1.0;
        
        // Check for required fields
        if (!dataPoint.timestamp) qualityScore -= 0.3;
        if (!dataPoint.source) qualityScore -= 0.2;
        
        // Check data freshness
        const age = Date.now() - new Date(dataPoint.timestamp).getTime();
        const hoursOld = age / (1000 * 60 * 60);
        if (hoursOld > 24) qualityScore -= 0.2;
        if (hoursOld > 168) qualityScore -= 0.3; // More than a week old
        
        return Math.max(qualityScore, 0);
      }
    };
    
    logger.info('🔀 Data fusion engine initialized');
  }

  async setupIntelligentScheduling() {
    if (!this.config.smartScheduling) return;
    
    // Market data - high frequency during market hours
    this.scheduledTasks.set('marketData', schedule.schedule('* 9-16 * * 1-5', async () => {
      await this.collectMarketData();
    }));
    
    // News data - frequent updates
    this.scheduledTasks.set('newsData', schedule.schedule('*/5 * * * *', async () => {
      await this.collectNewsData();
    }));
    
    // Web search - adaptive frequency based on market volatility
    this.scheduledTasks.set('webSearch', schedule.schedule('*/10 * * * *', async () => {
      await this.performIntelligentWebSearch();
    }));
    
    // Data fusion - every 15 minutes
    this.scheduledTasks.set('dataFusion', schedule.schedule('*/15 * * * *', async () => {
      await this.performDataFusion();
    }));
    
    // Data quality check - hourly
    this.scheduledTasks.set('qualityCheck', schedule.schedule('0 * * * *', async () => {
      await this.performDataQualityCheck();
    }));
    
    logger.info('📅 Intelligent scheduling setup completed');
  }

  setupDataProcessingPipeline() {
    // Process data queue every second
    setInterval(async () => {
      if (this.dataProcessingQueue.length > 0 && this.activeOperations.size < this.config.maxConcurrentOperations) {
        const batch = this.dataProcessingQueue.splice(0, this.config.dataProcessingBatchSize);
        this.processDataBatch(batch);
      }
    }, 1000);
    
    logger.info('⚙️ Data processing pipeline setup completed');
  }

  setupEventHandlers() {
    // Handle data from automated core
    if (this.automatedDataCore) {
      this.automatedDataCore.on('dataReceived', (data) => {
        this.handleIncomingData(data, 'automated');
      });
      
      this.automatedDataCore.on('requestError', (error) => {
        logger.warn('⚠️ Automated data core error:', error.error);
      });
    }
    
    // Handle search results from web search engine
    if (this.webSearchEngine) {
      this.webSearchEngine.on('searchCompleted', (result) => {
        this.handleSearchResults(result);
      });
      
      this.webSearchEngine.on('searchError', (error) => {
        logger.warn('⚠️ Web search engine error:', error.error);
      });
    }
    
    logger.info('🔗 Event handlers setup completed');
  }

  setupDocumentSystemEventHandlers() {
    // Document processing events
    this.documentSystem.on('documentProcessed', (data) => {
      this.handleDocumentProcessed(data);
    });
    
    // Document analysis events
    this.documentSystem.on('documentAnalyzed', (data) => {
      this.handleDocumentAnalyzed(data);
    });
    
    // Document alerts
    this.documentSystem.on('documentAlert', (alert) => {
      this.handleDocumentAlert(alert);
    });
    
    // ASI data updates from documents
    this.documentSystem.on('asiDataUpdate', (update) => {
      this.handleDocumentASIUpdate(update);
    });
    
    logger.info('📄 Document system event handlers configured');
  }

  setupSocialMediaEventHandlers() {
    // Real-time sentiment updates
    this.socialMediaIntegration.on('realTimeSentiment', (data) => {
      this.handleRealTimeSentiment(data);
    });
    
    // Philosophy analysis updates
    this.socialMediaIntegration.on('philosophyUpdate', (data) => {
      this.handlePhilosophyUpdate(data);
    });
    
    // ASI updates from social media intelligence
    this.socialMediaIntegration.on('asiUpdate', (update) => {
      this.handleSocialMediaASIUpdate(update);
    });
    
    // Comprehensive insights
    this.socialMediaIntegration.on('comprehensiveInsights', (insights) => {
      this.handleComprehensiveInsights(insights);
    });
    
    // Daily reports
    this.socialMediaIntegration.on('dailyReport', (report) => {
      this.handleDailyReport(report);
    });
    
    // Weekly trends
    this.socialMediaIntegration.on('weeklyTrends', (trends) => {
      this.handleWeeklyTrends(trends);
    });
    
    // Monthly reviews
    this.socialMediaIntegration.on('monthlyReview', (review) => {
      this.handleMonthlyReview(review);
    });
    
    logger.info('📱 Social media integration event handlers configured');
  }

  setupDocumentEventHandlers() {
    // Document system events
    this.documentSystem.on('documentProcessed', (data) => {
      this.handleDocumentProcessed(data);
    });
    
    this.documentSystem.on('analysisCompleted', (data) => {
      this.handleDocumentAnalysis(data);
    });
    
    this.documentSystem.on('alert', (alert) => {
      this.handleDocumentAlert(alert);
    });
    
    this.documentSystem.on('asiUpdate', (update) => {
      this.handleDocumentASIUpdate(update);
    });
    
    this.documentSystem.on('criticalAlerts', (alerts) => {
      this.handleCriticalDocumentAlerts(alerts);
    });
    
    // Document system error handling
    this.documentSystem.on('error', (error) => {
      logger.error('Document System Error:', error);
    });
  }

  async startContinuousDataFlow() {
    logger.info('🌊 Starting continuous data flow...');
    
    // Immediate data collection
    await this.performInitialDataCollection();
    
    // Start adaptive frequency monitoring
    if (this.config.adaptiveFrequency) {
      this.startAdaptiveFrequencyMonitoring();
    }
    
    logger.info('✅ Continuous data flow started');
  }

  async performInitialDataCollection() {
    try {
      logger.info('🚀 Performing initial data collection...');
      
      const symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA'];
      
      // Collect market data
      await this.collectMarketData();
      
      // Collect news data
      await this.collectNewsData();
      
      // Perform web search for key symbols
      if (this.webSearchEngine) {
        await this.webSearchEngine.searchFinancialData(symbols, ['market', 'news']);
      }
      
      logger.info('✅ Initial data collection completed');
      
    } catch (error) {
      logger.error('❌ Initial data collection failed:', error);
    }
  }

  async collectMarketData() {
    if (!this.automatedDataCore) return;
    
    try {
      await this.automatedDataCore.collectMarketData();
      logger.debug('📊 Market data collection triggered');
    } catch (error) {
      logger.error('❌ Market data collection failed:', error);
    }
  }

  async collectNewsData() {
    if (!this.automatedDataCore) return;
    
    try {
      await this.automatedDataCore.collectNewsData();
      logger.debug('📰 News data collection triggered');
    } catch (error) {
      logger.error('❌ News data collection failed:', error);
    }
  }

  async performIntelligentWebSearch() {
    if (!this.webSearchEngine) return;
    
    try {
      // Get trending symbols or use default list
      const symbols = await this.getTrendingSymbols() || ['AAPL', 'GOOGL', 'MSFT'];
      
      // Perform targeted searches
      await this.webSearchEngine.searchFinancialData(symbols.slice(0, 3), ['market', 'news']);
      
      this.metrics.searchQueriesExecuted++;
      logger.debug('🔍 Intelligent web search completed');
      
    } catch (error) {
      logger.error('❌ Intelligent web search failed:', error);
    }
  }

  async performDataFusion() {
    if (!this.config.enableDataFusion || !this.dataFusionEngine) return;
    
    try {
      const operationId = `fusion_${Date.now()}`;
      this.activeOperations.add(operationId);
      
      // Get recent data from all sources
      const recentData = this.getRecentData(60); // Last 60 minutes
      
      if (recentData.length > 0) {
        // Fuse data from multiple sources
        const fusedData = await this.dataFusionEngine.fuseData(recentData);
        
        // Store fused data
        for (const dataPoint of fusedData) {
          this.storeUnifiedData(dataPoint);
        }
        
        this.metrics.dataFusionOperations++;
        logger.debug(`🔀 Data fusion completed: ${fusedData.length} data points processed`);
      }
      
    } catch (error) {
      logger.error('❌ Data fusion failed:', error);
    } finally {
      this.activeOperations.delete('fusion_' + Date.now());
    }
  }

  async performDataQualityCheck() {
    try {
      let totalQuality = 0;
      let dataPointsChecked = 0;
      
      for (const [key, dataPoints] of this.unifiedDataStore) {
        for (const dataPoint of dataPoints.slice(-10)) { // Check last 10 points
          const quality = this.dataFusionEngine?.validateQuality(dataPoint) || 0.5;
          totalQuality += quality;
          dataPointsChecked++;
        }
      }
      
      if (dataPointsChecked > 0) {
        this.metrics.dataQualityScore = totalQuality / dataPointsChecked;
        logger.debug(`📊 Data quality check completed: ${(this.metrics.dataQualityScore * 100).toFixed(1)}% average quality`);
      }
      
    } catch (error) {
      logger.error('❌ Data quality check failed:', error);
    }
  }

  handleIncomingData(data, source) {
    // Add to processing queue
    this.dataProcessingQueue.push({
      ...data,
      processingSource: source,
      receivedAt: new Date()
    });
    
    this.metrics.totalDataPoints++;
    
    // Emit data received event for subscribers
    this.emit('dataReceived', data);
  }

  handleSearchResults(result) {
    // Process search results
    for (const searchResult of result.results) {
      this.handleIncomingData({
        type: 'search',
        source: 'WebSearch',
        symbol: result.symbol,
        category: result.category,
        data: searchResult,
        timestamp: new Date()
      }, 'search');
    }
  }

  async processDataBatch(batch) {
    const operationId = `batch_${Date.now()}`;
    this.activeOperations.add(operationId);
    
    try {
      for (const dataItem of batch) {
        // Validate data
        if (this.config.enableDataValidation && !this.validateData(dataItem)) {
          continue;
        }
        
        // Check for duplicates
        if (this.config.enableDuplicateDetection && this.isDuplicate(dataItem)) {
          continue;
        }
        
        // Store in unified data store
        this.storeUnifiedData(dataItem);
        
        // Notify subscribers
        this.notifySubscribers(dataItem);
      }
      
    } catch (error) {
      logger.error('❌ Data batch processing failed:', error);
    } finally {
      this.activeOperations.delete(operationId);
    }
  }

  storeUnifiedData(dataItem) {
    const key = this.generateDataKey(dataItem);
    
    if (!this.unifiedDataStore.has(key)) {
      this.unifiedDataStore.set(key, []);
    }
    
    const dataPoints = this.unifiedDataStore.get(key);
    dataPoints.push(dataItem);
    
    // Limit storage size
    if (dataPoints.length > 1000) {
      dataPoints.splice(0, dataPoints.length - 1000);
    }
  }

  generateDataKey(dataItem) {
    return `${dataItem.type}_${dataItem.symbol || 'general'}_${dataItem.source}`;
  }

  validateData(dataItem) {
    // Basic validation
    if (!dataItem.timestamp || !dataItem.source) {
      return false;
    }
    
    // Type-specific validation
    if (dataItem.type === 'market' && !dataItem.symbol) {
      return false;
    }
    
    return true;
  }

  isDuplicate(dataItem) {
    const key = this.generateDataKey(dataItem);
    const existing = this.unifiedDataStore.get(key);
    
    if (!existing || existing.length === 0) {
      return false;
    }
    
    // Check last few items for duplicates
    const recent = existing.slice(-5);
    return recent.some(item => 
      Math.abs(new Date(item.timestamp) - new Date(dataItem.timestamp)) < 60000 && // Within 1 minute
      JSON.stringify(item.data) === JSON.stringify(dataItem.data)
    );
  }

  mergeDataPoints(existing, newData) {
    // Simple merge strategy - prefer newer data
    return {
      ...existing,
      ...newData,
      timestamp: newData.timestamp,
      sources: [...(existing.sources || [existing.source]), newData.source].filter(Boolean)
    };
  }

  getRecentData(minutes = 60) {
    const cutoff = new Date(Date.now() - minutes * 60 * 1000);
    const recentData = [];
    
    for (const [key, dataPoints] of this.unifiedDataStore) {
      const recent = dataPoints.filter(dp => new Date(dp.timestamp) > cutoff);
      recentData.push(...recent);
    }
    
    return recentData;
  }

  async getTrendingSymbols() {
    // Simple implementation - in production, this would analyze market data
    const defaultSymbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA'];
    return defaultSymbols.slice(0, 5);
  }

  startAdaptiveFrequencyMonitoring() {
    // Monitor market volatility and adjust data collection frequency
    setInterval(() => {
      const marketHour = this.getCurrentMarketHour();
      
      if (marketHour === 'regular') {
        // Increase frequency during regular market hours
        this.adjustDataCollectionFrequency('high');
      } else if (marketHour === 'preMarket' || marketHour === 'afterHours') {
        // Moderate frequency during extended hours
        this.adjustDataCollectionFrequency('medium');
      } else {
        // Low frequency during market closed
        this.adjustDataCollectionFrequency('low');
      }
    }, 5 * 60 * 1000); // Check every 5 minutes
  }

  getCurrentMarketHour() {
    const now = new Date();
    const hour = now.getHours() + now.getMinutes() / 60;
    const day = now.getDay();
    
    // Weekend
    if (day === 0 || day === 6) {
      return 'closed';
    }
    
    // Check market hours (EST)
    if (hour >= this.marketHours.preMarket.start && hour < this.marketHours.preMarket.end) {
      return 'preMarket';
    } else if (hour >= this.marketHours.regular.start && hour < this.marketHours.regular.end) {
      return 'regular';
    } else if (hour >= this.marketHours.afterHours.start && hour < this.marketHours.afterHours.end) {
      return 'afterHours';
    } else {
      return 'closed';
    }
  }

  adjustDataCollectionFrequency(level) {
    // This would adjust the frequency of scheduled tasks
    logger.debug(`📊 Adjusting data collection frequency to: ${level}`);
  }

  // Subscription management for ASI components
  subscribeToData(subscriberId, dataTypes, callback) {
    this.dataSubscribers.set(subscriberId, {
      dataTypes,
      callback,
      subscribed: new Date()
    });
    
    logger.info(`📡 Data subscription added: ${subscriberId} for ${dataTypes.join(', ')}`);
  }

  unsubscribeFromData(subscriberId) {
    this.dataSubscribers.delete(subscriberId);
    logger.info(`📡 Data subscription removed: ${subscriberId}`);
  }

  notifySubscribers(dataItem) {
    for (const [subscriberId, subscription] of this.dataSubscribers) {
      if (subscription.dataTypes.includes(dataItem.type)) {
        try {
          subscription.callback(dataItem);
        } catch (error) {
          logger.error(`❌ Subscriber notification failed for ${subscriberId}:`, error);
        }
      }
    }
  }

  // Public API methods
  getUnifiedData(type, symbol, limit = 100) {
    const key = `${type}_${symbol}_*`;
    const allData = [];
    
    for (const [storeKey, dataPoints] of this.unifiedDataStore) {
      if (storeKey.startsWith(`${type}_${symbol}`)) {
        allData.push(...dataPoints);
      }
    }
    
    return allData
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, limit);
  }

  getLatestData(type, symbol) {
    const data = this.getUnifiedData(type, symbol, 1);
    return data.length > 0 ? data[0] : null;
  }

  getDataSummary() {
    const summary = {
      totalDataPoints: 0,
      dataTypes: new Set(),
      symbols: new Set(),
      sources: new Set(),
      latestUpdate: null
    };
    
    for (const [key, dataPoints] of this.unifiedDataStore) {
      summary.totalDataPoints += dataPoints.length;
      
      for (const dp of dataPoints) {
        if (dp.type) summary.dataTypes.add(dp.type);
        if (dp.symbol) summary.symbols.add(dp.symbol);
        if (dp.source) summary.sources.add(dp.source);
        
        if (!summary.latestUpdate || new Date(dp.timestamp) > summary.latestUpdate) {
          summary.latestUpdate = new Date(dp.timestamp);
        }
      }
    }
    
    return {
      ...summary,
      dataTypes: Array.from(summary.dataTypes),
      symbols: Array.from(summary.symbols),
      sources: Array.from(summary.sources)
    };
  }

  // Social media integration event handlers
  async handleRealTimeSentiment(data) {
    try {
      logger.debug(`📱 Real-time sentiment update for ${data.company}: ${data.sentiment.aggregated.sentiment}`);
      
      // Create data item for unified store
      const dataItem = {
        type: 'social_sentiment',
        source: 'social_media_integration',
        timestamp: data.timestamp,
        company: data.company,
        platform: data.platform,
        sentiment: data.sentiment,
        metadata: {
          confidence: data.sentiment.aggregated.confidence,
          dataPoints: data.sentiment.dataPoints
        }
      };
      
      // Store in unified data store
      await this.storeDataItem(dataItem);
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit update event
      this.emit('dataUpdate', {
        type: 'social_sentiment',
        data: dataItem,
        priority: 'high'
      });
      
    } catch (error) {
      logger.error('❌ Real-time sentiment handling failed:', error);
    }
  }

  async handlePhilosophyUpdate(data) {
    try {
      logger.info(`🧠 Philosophy update for ${data.company}`);
      
      // Create data item for unified store
      const dataItem = {
        type: 'management_philosophy',
        source: 'social_media_integration',
        timestamp: new Date().toISOString(),
        company: data.company,
        analysis: data.analysis,
        insights: data.insights,
        metadata: {
          philosophyScore: data.analysis.philosophyScore,
          consistencyScore: data.analysis.consistencyScore,
          confidenceLevel: data.analysis.confidenceLevel
        }
      };
      
      // Store in unified data store
      await this.storeDataItem(dataItem);
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit update event
      this.emit('dataUpdate', {
        type: 'management_philosophy',
        data: dataItem,
        priority: 'high'
      });
      
    } catch (error) {
      logger.error('❌ Philosophy update handling failed:', error);
    }
  }

  async handleSocialMediaASIUpdate(update) {
    try {
      logger.info(`🔗 Social media ASI update: ${update.data.summary.companiesAnalyzed} companies`);
      
      // Process management insights
      for (const [company, insights] of Object.entries(update.data.managementInsights)) {
        const dataItem = {
          type: 'management_insights',
          source: 'social_media_integration',
          timestamp: update.timestamp,
          company: company,
          insights: insights.insights,
          philosophyAnalysis: insights.philosophyAnalysis,
          metadata: {
            dataPoints: insights.dataPoints,
            lastUpdated: insights.lastUpdated
          }
        };
        
        await this.storeDataItem(dataItem);
        this.notifySubscribers(dataItem);
      }
      
      // Process sentiment trends
      for (const [company, trends] of Object.entries(update.data.sentimentTrends)) {
        const dataItem = {
          type: 'sentiment_trends',
          source: 'social_media_integration',
          timestamp: update.timestamp,
          company: company,
          trends: trends,
          metadata: {
            currentTrend: trends.currentTrend,
            confidence: trends.confidence,
            lastUpdated: trends.lastUpdated
          }
        };
        
        await this.storeDataItem(dataItem);
        this.notifySubscribers(dataItem);
      }
      
      // Emit comprehensive update
      this.emit('dataUpdate', {
        type: 'social_media_asi_update',
        data: update,
        priority: 'high'
      });
      
    } catch (error) {
      logger.error('❌ Social media ASI update handling failed:', error);
    }
  }

  async handleComprehensiveInsights(insights) {
    try {
      logger.info(`💡 Comprehensive insights generated: ${insights.companiesAnalyzed} companies`);
      
      // Create data item for unified store
      const dataItem = {
        type: 'comprehensive_insights',
        source: 'social_media_integration',
        timestamp: insights.timestamp,
        insights: insights.insights,
        metadata: {
          companiesAnalyzed: insights.companiesAnalyzed,
          topPerformers: insights.insights.topPerformers.length,
          consistencyLeaders: insights.insights.consistencyLeaders.length
        }
      };
      
      // Store in unified data store
      await this.storeDataItem(dataItem);
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit update event
      this.emit('dataUpdate', {
        type: 'comprehensive_insights',
        data: dataItem,
        priority: 'medium'
      });
      
    } catch (error) {
      logger.error('❌ Comprehensive insights handling failed:', error);
    }
  }

  async handleDailyReport(report) {
    try {
      logger.info(`📊 Daily report generated for ${report.date}`);
      
      // Create data item for unified store
      const dataItem = {
        type: 'daily_report',
        source: 'social_media_integration',
        timestamp: new Date().toISOString(),
        date: report.date,
        stats: report.stats,
        insights: report.insights,
        trends: report.trends
      };
      
      // Store in unified data store
      await this.storeDataItem(dataItem);
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit update event
      this.emit('dataUpdate', {
        type: 'daily_report',
        data: dataItem,
        priority: 'low'
      });
      
    } catch (error) {
      logger.error('❌ Daily report handling failed:', error);
    }
  }

  async handleWeeklyTrends(trends) {
    try {
      logger.info(`📈 Weekly trends analyzed for week ${trends.week}`);
      
      // Create data item for unified store
      const dataItem = {
        type: 'weekly_trends',
        source: 'social_media_integration',
        timestamp: new Date().toISOString(),
        week: trends.week,
        sentimentTrends: trends.sentimentTrends,
        philosophyChanges: trends.philosophyChanges,
        communicationPatterns: trends.communicationPatterns
      };
      
      // Store in unified data store
      await this.storeDataItem(dataItem);
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit update event
      this.emit('dataUpdate', {
        type: 'weekly_trends',
        data: dataItem,
        priority: 'low'
      });
      
    } catch (error) {
      logger.error('❌ Weekly trends handling failed:', error);
    }
  }

  async handleMonthlyReview(review) {
    try {
      logger.info(`🔍 Monthly review completed for ${review.month}/${review.year}`);
      
      // Create data item for unified store
      const dataItem = {
        type: 'monthly_review',
        source: 'social_media_integration',
        timestamp: new Date().toISOString(),
        month: review.month,
        year: review.year,
        philosophyEvolution: review.philosophyEvolution,
        newInsights: review.newInsights,
        recommendations: review.recommendations
      };
      
      // Store in unified data store
      await this.storeDataItem(dataItem);
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit update event
      this.emit('dataUpdate', {
        type: 'monthly_review',
        data: dataItem,
        priority: 'low'
      });
      
    } catch (error) {
      logger.error('❌ Monthly review handling failed:', error);
    }
  }

  // Document system event handlers
  async handleDocumentProcessed(data) {
    try {
      logger.debug(`📄 Document processed: ${data.document.title} for ${data.amcName}`);
      
      // Create data item for unified store
      const dataItem = {
        type: 'document_processed',
        source: 'document_system',
        timestamp: new Date().toISOString(),
        amcName: data.amcName,
        document: data.document,
        isNew: data.isNew,
        hasChanged: data.hasChanged
      };
      
      await this.storeUnifiedData(dataItem);
      this.metrics.documentsProcessed++;
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit event for ASI components
      this.emit('documentProcessed', data);
      
    } catch (error) {
      logger.error('❌ Error handling document processed:', error);
    }
  }
  
  async handleDocumentAnalysis(data) {
    try {
      logger.debug(`🔬 Document analysis completed for ${data.amcName}`);
      
      // Extract financial insights from analysis
      const insights = this.extractFinancialInsights(data.analysisResult);
      
      // Create data item
      const dataItem = {
        type: 'document_analysis',
        source: 'document_analyzer',
        timestamp: new Date().toISOString(),
        amcName: data.amcName,
        analysisResult: data.analysisResult,
        insights,
        priority: data.task.priority
      };
      
      await this.storeUnifiedData(dataItem);
      this.metrics.documentsAnalyzed++;
      
      // Update AMC data if financial metrics found
      if (insights.financialMetrics && Object.keys(insights.financialMetrics).length > 0) {
        await this.updateAMCFinancialData(data.amcName, insights.financialMetrics);
      }
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit event for ASI components
      this.emit('documentAnalysis', data);
      
    } catch (error) {
      logger.error('❌ Error handling document analysis:', error);
    }
  }
  
  async handleDocumentAlert(alert) {
    try {
      logger.info(`🚨 Document alert: ${alert.type} for ${alert.amcName}`);
      
      // Create alert data item
      const dataItem = {
        type: 'document_alert',
        source: 'document_system',
        timestamp: new Date().toISOString(),
        alert,
        severity: alert.severity
      };
      
      await this.storeUnifiedData(dataItem);
      this.metrics.alertsGenerated++;
      
      // Trigger immediate data refresh for high-severity alerts
      if (alert.severity === 'high') {
        await this.triggerImmediateDataRefresh(alert.amcName);
      }
      
      // Notify subscribers
      this.notifySubscribers(dataItem);
      
      // Emit event for ASI components
      this.emit('documentAlert', alert);
      
    } catch (error) {
      logger.error('❌ Error handling document alert:', error);
    }
  }
  
  async handleDocumentASIUpdate(update) {
    try {
      logger.info(`🔗 Document ASI update: ${update.summary.totalAMCs} AMCs`);
      
      // Process AMC data from document system
      for (const [amcKey, amcData] of Object.entries(update.amcData)) {
        const dataItem = {
          type: 'amc_document_data',
          source: 'document_system',
          timestamp: new Date().toISOString(),
          amcKey,
          amcData,
          documentCount: amcData.documentCount
        };
        
        await this.storeUnifiedData(dataItem);
      }
      
      this.metrics.asiUpdatesReceived++;
      
      // Emit event for ASI components
      this.emit('documentASIUpdate', update);
      
    } catch (error) {
      logger.error('❌ Error handling document ASI update:', error);
    }
  }
  
  async handleCriticalDocumentAlerts(alerts) {
    try {
      logger.error(`🚨🚨 Critical document alerts: ${alerts.length} alerts`);
      
      // Create critical alert data item
      const dataItem = {
        type: 'critical_document_alerts',
        source: 'document_system',
        timestamp: new Date().toISOString(),
        alerts,
        count: alerts.length
      };
      
      await this.storeUnifiedData(dataItem);
      this.metrics.criticalAlertsGenerated++;
      
      // Trigger immediate analysis for all affected AMCs
      const affectedAMCs = [...new Set(alerts.map(alert => alert.amcName))];
      for (const amcName of affectedAMCs) {
        await this.triggerImmediateDataRefresh(amcName);
      }
      
      // Notify subscribers with high priority
      this.notifySubscribers({ ...dataItem, priority: 'critical' });
      
      // Emit event for ASI components
      this.emit('criticalDocumentAlerts', alerts);
      
    } catch (error) {
      logger.error('❌ Error handling critical document alerts:', error);
    }
  }
  
  extractFinancialInsights(analysisResult) {
    const insights = {
      financialMetrics: {},
      performanceData: {},
      portfolioData: {},
      trends: []
    };
    
    // Extract financial metrics
    if (analysisResult.financialMetrics) {
      for (const [metric, values] of Object.entries(analysisResult.financialMetrics)) {
        if (values && values.length > 0) {
          insights.financialMetrics[metric] = {
            value: values[values.length - 1].value,
            unit: values[values.length - 1].unit,
            extractedAt: new Date().toISOString()
          };
        }
      }
    }
    
    // Extract performance data
    if (analysisResult.performanceData) {
      for (const [period, values] of Object.entries(analysisResult.performanceData)) {
        if (values && values.length > 0) {
          insights.performanceData[period] = {
            value: values[values.length - 1].value,
            extractedAt: new Date().toISOString()
          };
        }
      }
    }
    
    // Extract portfolio data
    if (analysisResult.portfolioData) {
      insights.portfolioData = {
        ...analysisResult.portfolioData,
        extractedAt: new Date().toISOString()
      };
    }
    
    return insights;
  }
  
  async updateAMCFinancialData(amcName, financialMetrics) {
    try {
      const amcKey = amcName.toLowerCase().replace(/[^a-z0-9]/g, '_');
      
      // Update unified data store with latest AMC financial data
      const dataItem = {
        type: 'amc_financial_update',
        source: 'document_analysis',
        timestamp: new Date().toISOString(),
        amcName,
        amcKey,
        financialMetrics
      };
      
      await this.storeUnifiedData(dataItem);
      
      logger.debug(`💰 Updated financial data for ${amcName}`);
      
    } catch (error) {
      logger.error(`❌ Failed to update AMC financial data for ${amcName}:`, error);
    }
  }
  
  async triggerImmediateDataRefresh(amcName) {
    try {
      logger.info(`🔄 Triggering immediate data refresh for ${amcName}`);
      
      // Force document scan for the AMC
      if (this.documentSystem && this.documentSystem.documentMonitor) {
        await this.documentSystem.documentMonitor.forceDocumentScan(amcName);
      }
      
      // Trigger web search for recent AMC data
      if (this.webSearchEngine) {
        const searchQuery = `${amcName} latest financial data performance NAV AUM`;
        await this.webSearchEngine.performSearch(searchQuery, { priority: 'high' });
      }
      
      this.metrics.immediateRefreshesTriggered++;
      
    } catch (error) {
      logger.error(`❌ Failed to trigger immediate data refresh for ${amcName}:`, error);
    }
  }

  getSystemMetrics() {
    const documentSystemStats = this.documentSystem ? this.documentSystem.getSystemStats() : {};
    
    return {
      ...this.metrics,
      queueSize: this.dataProcessingQueue.length,
      activeOperations: this.activeOperations.size,
      unifiedDataSize: this.unifiedDataStore.size,
      subscribers: this.dataSubscribers.size,
      scheduledTasks: this.scheduledTasks.size,
      isInitialized: this.isInitialized,
      currentMarketHour: this.getCurrentMarketHour(),
      documentSystem: {
        isInitialized: documentSystemStats.isInitialized || false,
        documentsTracked: documentSystemStats.monitorStats?.totalDocuments || 0,
        analysisCompleted: documentSystemStats.analysisCompleted || 0,
        alertsGenerated: documentSystemStats.alertsGenerated || 0,
        queueSize: documentSystemStats.queueSize || 0
      }
    };
  }

  async shutdown() {
    try {
      logger.info('🛑 Shutting down Integrated Data Manager...');
      
      // Stop all scheduled tasks
      for (const [name, task] of this.scheduledTasks) {
        task.destroy();
        logger.info(`📅 Stopped ${name} task`);
      }
      
      // Shutdown components
      if (this.automatedDataCore) {
        await this.automatedDataCore.shutdown();
      }
      
      this.isInitialized = false;
      logger.info('✅ Integrated Data Manager shutdown completed');
      
    } catch (error) {
      logger.error('❌ Integrated Data Manager shutdown failed:', error);
    }
  }
}

module.exports = { IntegratedDataManager };
