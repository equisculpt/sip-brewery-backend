/**
 * 🔄 INTEGRATED AMC DOCUMENT SYSTEM
 * 
 * Orchestrates document monitoring, analysis, and ASI integration
 * Provides real-time document intelligence for financial analysis
 * 
 * @author Financial Document Intelligence Team
 * @version 1.0.0 - Document Intelligence Integration
 */

const EventEmitter = require('events');
const schedule = require('node-cron');
const logger = require('../utils/logger');
const { AMCDocumentMonitor } = require('./document-monitor');
const { AMCDocumentAnalyzer } = require('./document-analyzer');

class IntegratedDocumentSystem extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      // System settings
      enableRealTimeMonitoring: options.enableRealTimeMonitoring !== false,
      enableAutoAnalysis: options.enableAutoAnalysis !== false,
      enableASIIntegration: options.enableASIIntegration !== false,
      
      // Processing settings
      batchSize: options.batchSize || 10,
      processingDelay: options.processingDelay || 5000,
      maxConcurrentAnalysis: options.maxConcurrentAnalysis || 3,
      
      // Alert settings
      enableChangeAlerts: options.enableChangeAlerts !== false,
      enablePerformanceAlerts: options.enablePerformanceAlerts !== false,
      alertThresholds: {
        significantChange: options.significantChange || 5, // 5% change
        performanceAlert: options.performanceAlert || 10, // 10% performance change
        ...options.alertThresholds
      },
      
      // Integration settings
      asiUpdateInterval: options.asiUpdateInterval || 30 * 60 * 1000, // 30 minutes
      
      ...options
    };
    
    // Initialize components
    this.documentMonitor = new AMCDocumentMonitor({
      enableContinuousMonitoring: this.config.enableRealTimeMonitoring,
      ...options.monitorOptions
    });
    
    this.documentAnalyzer = new AMCDocumentAnalyzer({
      enablePDFAnalysis: true,
      enableExcelAnalysis: true,
      enableContentExtraction: true,
      ...options.analyzerOptions
    });
    
    // Processing queues
    this.analysisQueue = [];
    this.processingInProgress = new Set();
    
    // Data storage
    this.documentInsights = new Map();
    this.changeAlerts = [];
    this.performanceMetrics = new Map();
    
    // ASI integration
    this.asiDataBuffer = new Map();
    this.lastASIUpdate = null;
    
    // Statistics
    this.stats = {
      documentsProcessed: 0,
      analysisCompleted: 0,
      alertsGenerated: 0,
      asiUpdates: 0,
      systemUptime: Date.now(),
      lastActivity: null
    };
    
    this.isInitialized = false;
    this.setupEventHandlers();
  }

  setupEventHandlers() {
    // Document monitor events
    this.documentMonitor.on('documentProcessed', (data) => {
      this.handleDocumentProcessed(data);
    });
    
    this.documentMonitor.on('changeAnalysis', (analysis) => {
      this.handleChangeAnalysis(analysis);
    });
    
    this.documentMonitor.on('scanCompleted', (stats) => {
      this.handleScanCompleted(stats);
    });
    
    // Document analyzer events
    this.documentAnalyzer.on('documentAnalyzed', (data) => {
      this.handleDocumentAnalyzed(data);
    });
    
    // Error handling
    this.documentMonitor.on('error', (error) => {
      logger.error('❌ Document Monitor Error:', error);
    });
    
    this.documentAnalyzer.on('error', (error) => {
      logger.error('❌ Document Analyzer Error:', error);
    });
  }

  async initialize() {
    try {
      logger.info('🔄 Initializing Integrated Document System...');
      
      // Initialize components
      await this.documentMonitor.initialize();
      await this.documentAnalyzer.initialize();
      
      // Start processing loops
      this.startAnalysisProcessing();
      
      // Setup ASI integration
      if (this.config.enableASIIntegration) {
        this.startASIIntegration();
      }
      
      // Setup alert monitoring
      if (this.config.enableChangeAlerts || this.config.enablePerformanceAlerts) {
        this.startAlertMonitoring();
      }
      
      this.isInitialized = true;
      logger.info('✅ Integrated Document System initialized successfully');
      
      this.emit('systemInitialized');
      
    } catch (error) {
      logger.error('❌ Integrated Document System initialization failed:', error);
      throw error;
    }
  }

  async handleDocumentProcessed(data) {
    try {
      this.stats.documentsProcessed++;
      this.stats.lastActivity = new Date();
      
      logger.debug(`📄 Document processed: ${data.document.title} for ${data.amcName}`);
      
      // Queue for analysis if auto-analysis is enabled
      if (this.config.enableAutoAnalysis) {
        this.queueDocumentForAnalysis(data);
      }
      
      // Emit document processed event
      this.emit('documentProcessed', data);
      
    } catch (error) {
      logger.error('❌ Error handling document processed:', error);
    }
  }

  queueDocumentForAnalysis(documentData) {
    const analysisTask = {
      id: `analysis_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      documentData,
      queuedAt: new Date(),
      priority: this.calculateAnalysisPriority(documentData)
    };
    
    this.analysisQueue.push(analysisTask);
    
    // Sort queue by priority (higher priority first)
    this.analysisQueue.sort((a, b) => b.priority - a.priority);
    
    logger.debug(`📋 Queued document for analysis: ${analysisTask.id}`);
  }

  calculateAnalysisPriority(documentData) {
    let priority = 1;
    
    // Higher priority for certain document types
    if (documentData.document.type === 'performance') priority += 3;
    if (documentData.document.type === 'factsheets') priority += 2;
    if (documentData.document.type === 'portfolios') priority += 2;
    
    // Higher priority for new documents
    if (documentData.isNew) priority += 2;
    
    // Higher priority for changed documents
    if (documentData.hasChanged) priority += 3;
    
    // Higher priority for top-tier AMCs
    const amcInfo = this.documentMonitor.documentRegistry.get(
      this.documentMonitor.generateAMCKey(documentData.amcName)
    );
    
    if (amcInfo && amcInfo.rank <= 10) priority += 2;
    if (amcInfo && amcInfo.category === 'large') priority += 1;
    
    return priority;
  }

  startAnalysisProcessing() {
    // Process analysis queue every 30 seconds
    setInterval(async () => {
      await this.processAnalysisQueue();
    }, 30000);
    
    logger.info('⚙️ Analysis processing started');
  }

  async processAnalysisQueue() {
    if (this.analysisQueue.length === 0) return;
    
    const concurrentLimit = this.config.maxConcurrentAnalysis;
    const currentlyProcessing = this.processingInProgress.size;
    
    if (currentlyProcessing >= concurrentLimit) {
      logger.debug(`⏳ Analysis queue processing delayed: ${currentlyProcessing}/${concurrentLimit} slots busy`);
      return;
    }
    
    const tasksToProcess = this.analysisQueue.splice(0, concurrentLimit - currentlyProcessing);
    
    for (const task of tasksToProcess) {
      this.processAnalysisTask(task);
    }
  }

  async processAnalysisTask(task) {
    try {
      this.processingInProgress.add(task.id);
      
      logger.debug(`🔬 Starting analysis: ${task.id}`);
      
      const { documentData } = task;
      const filePath = documentData.document.filePath;
      
      if (!filePath) {
        logger.warn(`⚠️ No file path for analysis task: ${task.id}`);
        return;
      }
      
      // Perform document analysis
      const analysisResult = await this.documentAnalyzer.analyzeDocument(
        filePath,
        {
          amcName: documentData.amcName,
          documentType: documentData.document.type,
          ...documentData.document
        }
      );
      
      if (analysisResult) {
        // Store insights
        await this.storeDocumentInsights(documentData.amcName, analysisResult);
        
        // Check for alerts
        await this.checkForAlerts(documentData.amcName, analysisResult);
        
        // Buffer for ASI integration
        this.bufferForASIIntegration(documentData.amcName, analysisResult);
        
        this.stats.analysisCompleted++;
        
        logger.debug(`✅ Analysis completed: ${task.id}`);
        
        this.emit('analysisCompleted', {
          task,
          analysisResult,
          amcName: documentData.amcName
        });
      }
      
    } catch (error) {
      logger.error(`❌ Analysis task failed: ${task.id}:`, error);
    } finally {
      this.processingInProgress.delete(task.id);
      
      // Add delay between analyses
      await new Promise(resolve => setTimeout(resolve, this.config.processingDelay));
    }
  }

  async handleDocumentAnalyzed(data) {
    try {
      logger.debug(`🔬 Document analyzed: ${data.filePath}`);
      
      // Store analysis results
      await this.storeDocumentInsights(
        data.documentInfo.amcName,
        data.analysisResult
      );
      
      this.emit('documentAnalyzed', data);
      
    } catch (error) {
      logger.error('❌ Error handling document analyzed:', error);
    }
  }

  async storeDocumentInsights(amcName, analysisResult) {
    try {
      const amcKey = amcName.toLowerCase().replace(/[^a-z0-9]/g, '_');
      
      if (!this.documentInsights.has(amcKey)) {
        this.documentInsights.set(amcKey, {
          amcName,
          documents: [],
          latestMetrics: {},
          trends: {},
          lastUpdated: null
        });
      }
      
      const insights = this.documentInsights.get(amcKey);
      
      // Add document analysis
      insights.documents.push({
        analysisResult,
        analyzedAt: new Date().toISOString()
      });
      
      // Update latest metrics
      if (analysisResult.financialMetrics) {
        for (const [metric, values] of Object.entries(analysisResult.financialMetrics)) {
          insights.latestMetrics[metric] = values[values.length - 1]; // Latest value
        }
      }
      
      // Update performance metrics
      if (analysisResult.performanceData) {
        for (const [period, values] of Object.entries(analysisResult.performanceData)) {
          if (!insights.trends[period]) insights.trends[period] = [];
          insights.trends[period].push({
            value: values[values.length - 1]?.value,
            timestamp: new Date().toISOString()
          });
          
          // Keep only last 10 data points
          if (insights.trends[period].length > 10) {
            insights.trends[period] = insights.trends[period].slice(-10);
          }
        }
      }
      
      insights.lastUpdated = new Date().toISOString();
      
      logger.debug(`💾 Stored insights for ${amcName}`);
      
    } catch (error) {
      logger.error(`❌ Failed to store insights for ${amcName}:`, error);
    }
  }

  async checkForAlerts(amcName, analysisResult) {
    try {
      const alerts = [];
      
      // Check for significant changes
      if (this.config.enableChangeAlerts) {
        const changeAlerts = this.detectSignificantChanges(amcName, analysisResult);
        alerts.push(...changeAlerts);
      }
      
      // Check for performance alerts
      if (this.config.enablePerformanceAlerts) {
        const performanceAlerts = this.detectPerformanceAlerts(amcName, analysisResult);
        alerts.push(...performanceAlerts);
      }
      
      // Store and emit alerts
      for (const alert of alerts) {
        this.changeAlerts.push(alert);
        this.stats.alertsGenerated++;
        
        logger.info(`🚨 Alert generated: ${alert.type} for ${amcName}`);
        
        this.emit('alert', alert);
      }
      
      // Keep only last 100 alerts
      if (this.changeAlerts.length > 100) {
        this.changeAlerts = this.changeAlerts.slice(-100);
      }
      
    } catch (error) {
      logger.error(`❌ Alert checking failed for ${amcName}:`, error);
    }
  }

  detectSignificantChanges(amcName, analysisResult) {
    const alerts = [];
    const amcKey = amcName.toLowerCase().replace(/[^a-z0-9]/g, '_');
    const insights = this.documentInsights.get(amcKey);
    
    if (!insights || insights.documents.length < 2) return alerts;
    
    const previousDoc = insights.documents[insights.documents.length - 2];
    const currentDoc = insights.documents[insights.documents.length - 1];
    
    // Check AUM changes
    const prevAUM = this.extractMetricValue(previousDoc.analysisResult, 'aum');
    const currentAUM = this.extractMetricValue(analysisResult, 'aum');
    
    if (prevAUM && currentAUM) {
      const changePercent = ((currentAUM - prevAUM) / prevAUM) * 100;
      
      if (Math.abs(changePercent) >= this.config.alertThresholds.significantChange) {
        alerts.push({
          type: 'significant_change',
          amcName,
          metric: 'aum',
          previousValue: prevAUM,
          currentValue: currentAUM,
          changePercent,
          severity: Math.abs(changePercent) > 15 ? 'high' : 'medium',
          timestamp: new Date().toISOString()
        });
      }
    }
    
    return alerts;
  }

  detectPerformanceAlerts(amcName, analysisResult) {
    const alerts = [];
    
    if (!analysisResult.performanceData) return alerts;
    
    // Check for significant performance changes
    for (const [period, values] of Object.entries(analysisResult.performanceData)) {
      for (const value of values) {
        if (Math.abs(value.value) >= this.config.alertThresholds.performanceAlert) {
          alerts.push({
            type: 'performance_alert',
            amcName,
            period,
            value: value.value,
            severity: Math.abs(value.value) > 20 ? 'high' : 'medium',
            timestamp: new Date().toISOString()
          });
        }
      }
    }
    
    return alerts;
  }

  extractMetricValue(analysisResult, metricName) {
    if (!analysisResult.financialMetrics || !analysisResult.financialMetrics[metricName]) {
      return null;
    }
    
    const values = analysisResult.financialMetrics[metricName];
    if (values.length === 0) return null;
    
    const latestValue = values[values.length - 1];
    return parseFloat(latestValue.value);
  }

  bufferForASIIntegration(amcName, analysisResult) {
    if (!this.config.enableASIIntegration) return;
    
    const amcKey = amcName.toLowerCase().replace(/[^a-z0-9]/g, '_');
    
    if (!this.asiDataBuffer.has(amcKey)) {
      this.asiDataBuffer.set(amcKey, []);
    }
    
    const buffer = this.asiDataBuffer.get(amcKey);
    buffer.push({
      analysisResult,
      timestamp: new Date().toISOString()
    });
    
    // Keep only last 5 analysis results per AMC
    if (buffer.length > 5) {
      buffer.splice(0, buffer.length - 5);
    }
  }

  startASIIntegration() {
    // Send buffered data to ASI system periodically
    setInterval(async () => {
      await this.sendDataToASI();
    }, this.config.asiUpdateInterval);
    
    logger.info('🔗 ASI integration started');
  }

  async sendDataToASI() {
    try {
      if (this.asiDataBuffer.size === 0) return;
      
      const asiUpdate = {
        timestamp: new Date().toISOString(),
        amcData: {},
        summary: {
          totalAMCs: this.asiDataBuffer.size,
          totalDocuments: 0,
          alertsGenerated: this.changeAlerts.length
        }
      };
      
      // Prepare data for each AMC
      for (const [amcKey, buffer] of this.asiDataBuffer) {
        if (buffer.length === 0) continue;
        
        const latestAnalysis = buffer[buffer.length - 1];
        const insights = this.documentInsights.get(amcKey);
        
        asiUpdate.amcData[amcKey] = {
          latestAnalysis: latestAnalysis.analysisResult,
          insights: insights || {},
          documentCount: buffer.length,
          lastUpdated: latestAnalysis.timestamp
        };
        
        asiUpdate.summary.totalDocuments += buffer.length;
      }
      
      // Emit ASI update event
      this.emit('asiUpdate', asiUpdate);
      
      this.stats.asiUpdates++;
      this.lastASIUpdate = new Date();
      
      logger.info(`🔗 ASI update sent: ${asiUpdate.summary.totalAMCs} AMCs, ${asiUpdate.summary.totalDocuments} documents`);
      
    } catch (error) {
      logger.error('❌ ASI integration failed:', error);
    }
  }

  startAlertMonitoring() {
    // Monitor for critical alerts every 5 minutes
    setInterval(() => {
      this.checkCriticalAlerts();
    }, 5 * 60 * 1000);
    
    logger.info('🚨 Alert monitoring started');
  }

  checkCriticalAlerts() {
    const recentAlerts = this.changeAlerts.filter(alert => {
      const alertTime = new Date(alert.timestamp);
      const fiveMinutesAgo = new Date(Date.now() - 5 * 60 * 1000);
      return alertTime > fiveMinutesAgo && alert.severity === 'high';
    });
    
    if (recentAlerts.length > 0) {
      this.emit('criticalAlerts', recentAlerts);
      logger.warn(`🚨 ${recentAlerts.length} critical alerts detected`);
    }
  }

  async handleChangeAnalysis(analysis) {
    try {
      logger.info(`📊 Change analysis received: ${analysis.totalChanges} changes`);
      
      // Process recent changes for insights
      for (const change of analysis.recentChanges) {
        await this.processDocumentChange(change);
      }
      
      this.emit('changeAnalysisProcessed', analysis);
      
    } catch (error) {
      logger.error('❌ Error handling change analysis:', error);
    }
  }

  async processDocumentChange(change) {
    // Extract AMC name from change key
    const amcKey = change.key.split('_')[0];
    const amcInfo = this.documentMonitor.documentRegistry.get(amcKey);
    
    if (amcInfo) {
      logger.debug(`📈 Processing change for ${amcInfo.name}`);
      
      // Trigger priority analysis for changed documents
      // This would queue the changed document for immediate analysis
    }
  }

  async handleScanCompleted(stats) {
    logger.info(`🔍 Document scan completed: ${stats.documentsDownloaded} documents downloaded`);
    
    this.emit('scanCompleted', stats);
  }

  // Public API methods
  async forceFullAnalysis(amcName = null) {
    try {
      logger.info(`🔬 Starting forced full analysis${amcName ? ` for ${amcName}` : ''}...`);
      
      const documents = amcName ? 
        await this.documentMonitor.getAMCDocuments(amcName) :
        await this.getAllDocuments();
      
      const analysisPromises = documents.map(doc => 
        this.documentAnalyzer.analyzeDocument(doc.filePath, {
          amcName: doc.amcName || amcName,
          ...doc
        })
      );
      
      const results = await Promise.allSettled(analysisPromises);
      const successful = results.filter(r => r.status === 'fulfilled').length;
      
      logger.info(`✅ Forced analysis completed: ${successful}/${documents.length} successful`);
      
      return { successful, total: documents.length };
      
    } catch (error) {
      logger.error('❌ Forced full analysis failed:', error);
      throw error;
    }
  }

  async getAllDocuments() {
    const allDocuments = [];
    
    for (const [amcKey, amcInfo] of this.documentMonitor.documentRegistry) {
      for (const [docKey, doc] of Object.entries(amcInfo.documents)) {
        allDocuments.push({
          ...doc,
          amcName: amcInfo.name
        });
      }
    }
    
    return allDocuments;
  }

  getSystemStats() {
    return {
      ...this.stats,
      uptime: Date.now() - this.stats.systemUptime,
      monitorStats: this.documentMonitor.getMonitoringStats(),
      analyzerStats: this.documentAnalyzer.getAnalysisStats(),
      queueSize: this.analysisQueue.length,
      processingCount: this.processingInProgress.size,
      insightsCount: this.documentInsights.size,
      alertsCount: this.changeAlerts.length,
      asiBufferSize: this.asiDataBuffer.size,
      lastASIUpdate: this.lastASIUpdate,
      isInitialized: this.isInitialized
    };
  }

  async getAMCInsights(amcName) {
    const amcKey = amcName.toLowerCase().replace(/[^a-z0-9]/g, '_');
    return this.documentInsights.get(amcKey) || null;
  }

  async getRecentAlerts(hours = 24) {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    
    return this.changeAlerts.filter(alert => 
      new Date(alert.timestamp) > cutoff
    );
  }

  async generateSystemReport() {
    const report = {
      generatedAt: new Date().toISOString(),
      systemStats: this.getSystemStats(),
      amcSummary: {},
      recentActivity: {
        alerts: await this.getRecentAlerts(24),
        changes: await this.documentMonitor.getRecentChanges(24)
      }
    };
    
    // Generate AMC summary
    for (const [amcKey, insights] of this.documentInsights) {
      report.amcSummary[insights.amcName] = {
        documentsAnalyzed: insights.documents.length,
        latestMetrics: insights.latestMetrics,
        lastUpdated: insights.lastUpdated
      };
    }
    
    return report;
  }
}

module.exports = { IntegratedDocumentSystem };
