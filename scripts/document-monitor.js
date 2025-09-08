/**
 * 🔄 AMC DOCUMENT MONITORING SERVICE
 * 
 * Standalone service for continuous document monitoring
 * Runs the document monitoring system in production mode
 * 
 * @author Financial Document Intelligence Team
 * @version 1.0.0 - Production Document Monitor
 */

const logger = require('../src/utils/logger');
const { IntegratedDocumentSystem } = require('../src/finance_crawler/integrated-document-system');

class DocumentMonitoringService {
  constructor() {
    this.documentSystem = null;
    this.isRunning = false;
    this.startTime = null;
    
    // Production configuration
    this.config = {
      enableRealTimeMonitoring: true,
      enableAutoAnalysis: true,
      enableASIIntegration: true,
      enableChangeAlerts: true,
      enablePerformanceAlerts: true,
      
      // Optimized for production
      batchSize: 10,
      processingDelay: 2000,
      maxConcurrentAnalysis: 5,
      
      // Monitor settings
      monitorOptions: {
        enableContinuousMonitoring: true,
        monitoringInterval: '0 */6 * * *', // Every 6 hours
        supportedFormats: ['pdf', 'xlsx', 'xls', 'csv'],
        maxFileSize: 100 * 1024 * 1024, // 100MB
        downloadTimeout: 60000, // 60 seconds
        documentsPath: './data/amc-documents',
        metadataPath: './data/amc-metadata'
      },
      
      // Analyzer settings
      analyzerOptions: {
        enablePDFAnalysis: true,
        enableExcelAnalysis: true,
        enableContentExtraction: true,
        extractFinancialMetrics: true,
        extractPerformanceData: true,
        extractPortfolioData: true,
        outputPath: './data/analysis-results'
      },
      
      // Alert thresholds
      alertThresholds: {
        significantChange: 3, // 3% change
        performanceAlert: 8, // 8% performance change
        volumeAlert: 15 // 15% volume change
      },
      
      // ASI integration
      asiUpdateInterval: 20 * 60 * 1000 // 20 minutes
    };
    
    // Statistics tracking
    this.stats = {
      sessionsRun: 0,
      totalDocumentsProcessed: 0,
      totalAnalysisCompleted: 0,
      totalAlertsGenerated: 0,
      totalASIUpdates: 0,
      uptime: 0,
      lastActivity: null
    };
    
    this.setupSignalHandlers();
  }

  setupSignalHandlers() {
    // Graceful shutdown
    process.on('SIGINT', () => this.shutdown('SIGINT'));
    process.on('SIGTERM', () => this.shutdown('SIGTERM'));
    
    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      logger.error('❌ Uncaught Exception:', error);
      this.shutdown('UNCAUGHT_EXCEPTION');
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      logger.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
    });
  }

  async start() {
    try {
      logger.info('🚀 Starting AMC Document Monitoring Service...');
      
      this.startTime = new Date();
      this.isRunning = true;
      
      // Initialize document system
      this.documentSystem = new IntegratedDocumentSystem(this.config);
      await this.documentSystem.initialize();
      
      // Setup event handlers
      this.setupEventHandlers();
      
      // Start monitoring loops
      this.startMonitoringLoops();
      
      // Log startup success
      logger.info('✅ Document Monitoring Service started successfully');
      logger.info(`📊 Configuration: ${JSON.stringify({
        realTimeMonitoring: this.config.enableRealTimeMonitoring,
        autoAnalysis: this.config.enableAutoAnalysis,
        asiIntegration: this.config.enableASIIntegration,
        monitoringInterval: this.config.monitorOptions.monitoringInterval,
        supportedFormats: this.config.monitorOptions.supportedFormats
      }, null, 2)}`);
      
      this.stats.sessionsRun++;
      
      // Keep service running
      this.keepAlive();
      
    } catch (error) {
      logger.error('❌ Failed to start Document Monitoring Service:', error);
      process.exit(1);
    }
  }

  setupEventHandlers() {
    // Document processing events
    this.documentSystem.on('documentProcessed', (data) => {
      this.stats.totalDocumentsProcessed++;
      this.stats.lastActivity = new Date();
      
      logger.info(`📄 Document processed: ${data.document.title} (${data.amcName})`);
      
      if (data.isNew) {
        logger.info(`  ✨ New document discovered`);
      }
      
      if (data.hasChanged) {
        logger.info(`  🔄 Document changed detected`);
      }
    });
    
    this.documentSystem.on('analysisCompleted', (data) => {
      this.stats.totalAnalysisCompleted++;
      this.stats.lastActivity = new Date();
      
      logger.info(`🔬 Analysis completed: ${data.amcName} - Priority: ${data.task.priority}`);
      
      // Log extracted metrics if available
      if (data.analysisResult.financialMetrics) {
        const metricCount = Object.keys(data.analysisResult.financialMetrics).length;
        logger.debug(`  📊 Extracted ${metricCount} financial metrics`);
      }
    });
    
    // Alert events
    this.documentSystem.on('alert', (alert) => {
      this.stats.totalAlertsGenerated++;
      
      const severity = alert.severity === 'high' ? '🚨🚨' : '🚨';
      logger.warn(`${severity} Alert: ${alert.type} for ${alert.amcName}`);
      
      if (alert.type === 'significant_change') {
        logger.warn(`  📈 ${alert.metric}: ${alert.previousValue} → ${alert.currentValue} (${alert.changePercent.toFixed(2)}%)`);
      }
      
      if (alert.type === 'performance_alert') {
        logger.warn(`  📊 Performance ${alert.period}: ${alert.value}%`);
      }
    });
    
    this.documentSystem.on('criticalAlerts', (alerts) => {
      logger.error(`🚨🚨 CRITICAL ALERTS: ${alerts.length} high-severity alerts detected!`);
      
      for (const alert of alerts) {
        logger.error(`  • ${alert.amcName}: ${alert.type} - ${JSON.stringify(alert)}`);
      }
    });
    
    // ASI integration events
    this.documentSystem.on('asiUpdate', (update) => {
      this.stats.totalASIUpdates++;
      
      logger.info(`🔗 ASI Update sent: ${update.summary.totalAMCs} AMCs, ${update.summary.totalDocuments} documents`);
      
      if (update.summary.alertsGenerated > 0) {
        logger.info(`  🚨 Included ${update.summary.alertsGenerated} alerts`);
      }
    });
    
    // System events
    this.documentSystem.on('scanCompleted', (stats) => {
      logger.info(`🔍 Document scan completed:`);
      logger.info(`  📥 Downloaded: ${stats.documentsDownloaded} documents`);
      logger.info(`  🔄 Changes detected: ${stats.changesDetected}`);
      logger.info(`  ❌ Errors: ${stats.errors}`);
    });
    
    this.documentSystem.on('changeAnalysisProcessed', (analysis) => {
      logger.info(`📊 Change analysis processed: ${analysis.totalChanges} changes in last 24h`);
    });
    
    // Error handling
    this.documentSystem.on('error', (error) => {
      logger.error('❌ Document System Error:', error);
    });
  }

  startMonitoringLoops() {
    // Status reporting every 30 minutes
    setInterval(() => {
      this.reportStatus();
    }, 30 * 60 * 1000);
    
    // Statistics update every 5 minutes
    setInterval(() => {
      this.updateStatistics();
    }, 5 * 60 * 1000);
    
    // Health check every minute
    setInterval(() => {
      this.performHealthCheck();
    }, 60 * 1000);
    
    logger.info('⏰ Monitoring loops started');
  }

  reportStatus() {
    const uptime = Date.now() - this.startTime.getTime();
    const systemStats = this.documentSystem.getSystemStats();
    
    logger.info('📊 MONITORING STATUS REPORT:');
    logger.info(`  ⏱️  Uptime: ${this.formatDuration(uptime)}`);
    logger.info(`  📄 Documents Processed: ${this.stats.totalDocumentsProcessed}`);
    logger.info(`  🔬 Analysis Completed: ${this.stats.totalAnalysisCompleted}`);
    logger.info(`  🚨 Alerts Generated: ${this.stats.totalAlertsGenerated}`);
    logger.info(`  🔗 ASI Updates: ${this.stats.totalASIUpdates}`);
    logger.info(`  📋 Queue Size: ${systemStats.queueSize}`);
    logger.info(`  ⚙️  Processing: ${systemStats.processingCount}`);
    logger.info(`  💾 Insights Stored: ${systemStats.insightsCount}`);
    logger.info(`  🏃 Last Activity: ${this.stats.lastActivity ? this.stats.lastActivity.toISOString() : 'None'}`);
  }

  updateStatistics() {
    this.stats.uptime = Date.now() - this.startTime.getTime();
    
    // Update system stats from document system
    const systemStats = this.documentSystem.getSystemStats();
    
    // Log any significant changes
    if (systemStats.queueSize > 50) {
      logger.warn(`⚠️ High queue size detected: ${systemStats.queueSize} items`);
    }
    
    if (systemStats.processingCount >= this.config.maxConcurrentAnalysis) {
      logger.info(`⚙️ Processing at full capacity: ${systemStats.processingCount}/${this.config.maxConcurrentAnalysis}`);
    }
  }

  performHealthCheck() {
    try {
      const systemStats = this.documentSystem.getSystemStats();
      
      // Check if system is responsive
      if (!systemStats.isInitialized) {
        logger.error('❌ Health Check Failed: System not initialized');
        return;
      }
      
      // Check for recent activity (within last hour)
      const oneHourAgo = new Date(Date.now() - 60 * 60 * 1000);
      if (this.stats.lastActivity && this.stats.lastActivity < oneHourAgo) {
        logger.warn('⚠️ No recent activity detected in the last hour');
      }
      
      // Check memory usage
      const memUsage = process.memoryUsage();
      const memUsageMB = Math.round(memUsage.heapUsed / 1024 / 1024);
      
      if (memUsageMB > 500) { // 500MB threshold
        logger.warn(`⚠️ High memory usage: ${memUsageMB}MB`);
      }
      
      logger.debug(`💚 Health check passed - Memory: ${memUsageMB}MB`);
      
    } catch (error) {
      logger.error('❌ Health check failed:', error);
    }
  }

  keepAlive() {
    // Keep the process alive
    setInterval(() => {
      // Heartbeat - just to keep process running
      logger.debug('💓 Service heartbeat');
    }, 10 * 60 * 1000); // Every 10 minutes
  }

  async shutdown(signal) {
    logger.info(`🛑 Received ${signal}. Shutting down gracefully...`);
    
    this.isRunning = false;
    
    try {
      // Generate final report
      await this.generateShutdownReport();
      
      // Save any pending data
      if (this.documentSystem) {
        await this.documentSystem.documentMonitor.saveDocumentRegistry();
        await this.documentSystem.documentAnalyzer.saveAnalysisResults();
      }
      
      logger.info('✅ Graceful shutdown completed');
      
    } catch (error) {
      logger.error('❌ Error during shutdown:', error);
    } finally {
      process.exit(0);
    }
  }

  async generateShutdownReport() {
    const uptime = Date.now() - this.startTime.getTime();
    const systemStats = this.documentSystem ? this.documentSystem.getSystemStats() : {};
    
    const report = {
      shutdownAt: new Date().toISOString(),
      startedAt: this.startTime.toISOString(),
      uptime: this.formatDuration(uptime),
      sessionStats: this.stats,
      systemStats,
      performance: {
        documentsPerHour: Math.round(this.stats.totalDocumentsProcessed / (uptime / (1000 * 60 * 60))),
        analysisPerHour: Math.round(this.stats.totalAnalysisCompleted / (uptime / (1000 * 60 * 60))),
        alertsPerHour: Math.round(this.stats.totalAlertsGenerated / (uptime / (1000 * 60 * 60)))
      }
    };
    
    logger.info('📊 SHUTDOWN REPORT:');
    logger.info(`  ⏱️  Session Duration: ${report.uptime}`);
    logger.info(`  📄 Documents/Hour: ${report.performance.documentsPerHour}`);
    logger.info(`  🔬 Analysis/Hour: ${report.performance.analysisPerHour}`);
    logger.info(`  🚨 Alerts/Hour: ${report.performance.alertsPerHour}`);
    
    // Save report to file
    const fs = require('fs').promises;
    const reportPath = `./logs/shutdown-report-${Date.now()}.json`;
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    logger.info(`📋 Shutdown report saved to: ${reportPath}`);
  }

  formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  }
}

// Start service if called directly
if (require.main === module) {
  const service = new DocumentMonitoringService();
  service.start().catch((error) => {
    console.error('❌ Failed to start monitoring service:', error);
    process.exit(1);
  });
}

module.exports = { DocumentMonitoringService };
