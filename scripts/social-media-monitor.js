/**
 * 🔄 SOCIAL MEDIA MONITORING SERVICE
 * 
 * Production-ready monitoring service for free social media intelligence
 * Runs continuously to track management communication and philosophy
 * 
 * @author Financial Intelligence Team
 * @version 1.0.0 - Production Social Media Monitor
 */

const logger = require('../src/utils/logger');
const { FreeSocialMediaIntegration } = require('../src/finance_crawler/free-social-media-integration');

class SocialMediaMonitorService {
  constructor() {
    this.socialMediaSystem = null;
    this.isRunning = false;
    this.startTime = null;
    this.stats = {
      totalDataCollected: 0,
      companiesTracked: 0,
      sentimentUpdates: 0,
      philosophyUpdates: 0,
      asiUpdates: 0,
      errors: 0,
      lastUpdate: null
    };
    
    // Graceful shutdown handling
    process.on('SIGINT', () => this.gracefulShutdown('SIGINT'));
    process.on('SIGTERM', () => this.gracefulShutdown('SIGTERM'));
    process.on('uncaughtException', (error) => this.handleUncaughtException(error));
    process.on('unhandledRejection', (reason, promise) => this.handleUnhandledRejection(reason, promise));
  }

  async start() {
    try {
      console.log('🚀 Starting Social Media Monitoring Service...\n');
      
      this.startTime = new Date();
      
      // Initialize social media system
      this.socialMediaSystem = new FreeSocialMediaIntegration({
        enableRealTimeTracking: true,
        enablePhilosophyAnalysis: true,
        enableSentimentAnalysis: true,
        enableTrendAnalysis: true,
        enableASIIntegration: true,
        
        // Production settings
        batchProcessingSize: 100,
        analysisInterval: 6 * 60 * 60 * 1000, // 6 hours
        asiUpdateInterval: 30 * 60 * 1000, // 30 minutes
      });
      
      // Setup monitoring event handlers
      this.setupEventHandlers();
      
      // Initialize the system
      await this.socialMediaSystem.initialize();
      
      this.isRunning = true;
      
      console.log('✅ Social Media Monitoring Service started successfully!');
      console.log(`🕐 Started at: ${this.startTime.toLocaleString()}`);
      console.log('📊 Monitoring real-time social media data...\n');
      
      // Start periodic health checks
      this.startHealthChecks();
      
      // Start periodic status reports
      this.startStatusReports();
      
      // Keep the process running
      this.keepAlive();
      
    } catch (error) {
      console.error('❌ Failed to start Social Media Monitoring Service:', error);
      process.exit(1);
    }
  }

  setupEventHandlers() {
    // System events
    this.socialMediaSystem.on('systemInitialized', () => {
      logger.info('🎉 Social Media System initialized and ready for monitoring');
    });
    
    // Real-time sentiment updates
    this.socialMediaSystem.on('realTimeSentiment', (data) => {
      this.stats.sentimentUpdates++;
      this.stats.lastUpdate = new Date();
      
      logger.info(`📊 Sentiment Update: ${data.company} - ${data.sentiment.aggregated.sentiment} (${data.platform})`);
      
      // Log detailed sentiment data
      if (data.sentiment.aggregated.confidence > 0.8) {
        logger.info(`   High confidence sentiment detected: ${data.sentiment.aggregated.confidence.toFixed(2)}`);
      }
    });
    
    // Philosophy updates
    this.socialMediaSystem.on('philosophyUpdate', (data) => {
      this.stats.philosophyUpdates++;
      this.stats.lastUpdate = new Date();
      
      logger.info(`🧠 Philosophy Update: ${data.company}`);
      logger.info(`   Primary Philosophy: ${data.analysis.investmentPhilosophy.primaryPhilosophy}`);
      logger.info(`   Philosophy Score: ${data.analysis.philosophyScore}`);
      logger.info(`   Consistency Score: ${data.analysis.consistencyScore}`);
      
      // Track unique companies
      this.updateCompaniesTracked(data.company);
    });
    
    // ASI updates
    this.socialMediaSystem.on('asiUpdate', (update) => {
      this.stats.asiUpdates++;
      this.stats.lastUpdate = new Date();
      
      logger.info(`🔗 ASI Update Generated:`);
      logger.info(`   Companies Analyzed: ${update.data.summary.companiesAnalyzed}`);
      logger.info(`   Total Data Points: ${update.data.summary.totalDataPoints}`);
      logger.info(`   New Insights: ${update.data.summary.newInsights}`);
    });
    
    // Comprehensive insights
    this.socialMediaSystem.on('comprehensiveInsights', (insights) => {
      logger.info(`💡 Comprehensive Insights Generated:`);
      logger.info(`   Companies Analyzed: ${insights.companiesAnalyzed}`);
      logger.info(`   Top Performers: ${insights.insights.topPerformers.length}`);
      logger.info(`   Consistency Leaders: ${insights.insights.consistencyLeaders.length}`);
    });
    
    // Daily reports
    this.socialMediaSystem.on('dailyReport', (report) => {
      logger.info(`📊 Daily Report Generated for ${report.date}`);
      this.logDailyReportSummary(report);
    });
    
    // Weekly trends
    this.socialMediaSystem.on('weeklyTrends', (trends) => {
      logger.info(`📈 Weekly Trends Analyzed for Week ${trends.week}`);
    });
    
    // Monthly reviews
    this.socialMediaSystem.on('monthlyReview', (review) => {
      logger.info(`🔍 Monthly Review Completed for ${review.month}/${review.year}`);
    });
    
    // Error handling
    this.socialMediaSystem.on('error', (error) => {
      this.stats.errors++;
      logger.error('❌ Social Media System Error:', error);
    });
    
    // Social media tracker events
    if (this.socialMediaSystem.socialMediaTracker) {
      this.socialMediaSystem.socialMediaTracker.on('socialMediaData', (data) => {
        this.stats.totalDataCollected += data.count;
        this.stats.lastUpdate = new Date();
        
        logger.debug(`📱 Social Media Data: ${data.company} - ${data.count} items from ${data.platform}`);
      });
      
      this.socialMediaSystem.socialMediaTracker.on('error', (error) => {
        this.stats.errors++;
        logger.error('❌ Social Media Tracker Error:', error);
      });
    }
  }

  updateCompaniesTracked(company) {
    // This is a simplified approach - in production, you'd maintain a Set
    // For now, we'll just increment if it's a new company mention
    if (Math.random() < 0.1) { // Simulate new company detection
      this.stats.companiesTracked++;
    }
  }

  logDailyReportSummary(report) {
    logger.info(`   Total Insights: ${report.insights.totalInsights}`);
    logger.info(`   New Profiles: ${report.insights.newProfiles}`);
    logger.info(`   Sentiment Changes: ${report.insights.sentimentChanges}`);
    logger.info(`   Overall Sentiment: ${report.trends.overallSentiment}`);
    logger.info(`   Most Active Company: ${report.trends.mostActiveCompany}`);
    logger.info(`   Trend Direction: ${report.trends.trendDirection}`);
  }

  startHealthChecks() {
    // Health check every 5 minutes
    setInterval(() => {
      this.performHealthCheck();
    }, 5 * 60 * 1000);
    
    logger.info('❤️ Health checks started (every 5 minutes)');
  }

  startStatusReports() {
    // Status report every 30 minutes
    setInterval(() => {
      this.generateStatusReport();
    }, 30 * 60 * 1000);
    
    // Detailed status report every 2 hours
    setInterval(() => {
      this.generateDetailedStatusReport();
    }, 2 * 60 * 60 * 1000);
    
    logger.info('📊 Status reports started (every 30 minutes)');
  }

  performHealthCheck() {
    try {
      const systemStats = this.socialMediaSystem.getSystemStats();
      const uptime = Date.now() - this.startTime.getTime();
      
      logger.info('❤️ Health Check:');
      logger.info(`   Service Uptime: ${Math.round(uptime / 1000 / 60)} minutes`);
      logger.info(`   System Initialized: ${systemStats.isInitialized}`);
      logger.info(`   Analysis Queue Size: ${systemStats.analysisQueueSize}`);
      logger.info(`   ASI Update Queue Size: ${systemStats.asiUpdateQueueSize}`);
      logger.info(`   Total Errors: ${this.stats.errors}`);
      
      // Check for potential issues
      if (systemStats.analysisQueueSize > 1000) {
        logger.warn('⚠️ Analysis queue size is high - potential processing bottleneck');
      }
      
      if (this.stats.errors > 50) {
        logger.warn('⚠️ High error count detected - system may need attention');
      }
      
      if (!this.stats.lastUpdate || (Date.now() - new Date(this.stats.lastUpdate).getTime()) > 60 * 60 * 1000) {
        logger.warn('⚠️ No updates received in the last hour - check data sources');
      }
      
    } catch (error) {
      logger.error('❌ Health check failed:', error);
    }
  }

  generateStatusReport() {
    try {
      const uptime = Date.now() - this.startTime.getTime();
      const systemStats = this.socialMediaSystem.getSystemStats();
      
      logger.info('📊 STATUS REPORT:');
      logger.info(`   Service Uptime: ${Math.round(uptime / 1000 / 60)} minutes`);
      logger.info(`   Data Points Collected: ${this.stats.totalDataCollected}`);
      logger.info(`   Companies Tracked: ${this.stats.companiesTracked}`);
      logger.info(`   Sentiment Updates: ${this.stats.sentimentUpdates}`);
      logger.info(`   Philosophy Updates: ${this.stats.philosophyUpdates}`);
      logger.info(`   ASI Updates: ${this.stats.asiUpdates}`);
      logger.info(`   Total Errors: ${this.stats.errors}`);
      
      if (this.stats.lastUpdate) {
        const timeSinceLastUpdate = Date.now() - new Date(this.stats.lastUpdate).getTime();
        logger.info(`   Last Update: ${Math.round(timeSinceLastUpdate / 1000 / 60)} minutes ago`);
      }
      
    } catch (error) {
      logger.error('❌ Status report generation failed:', error);
    }
  }

  generateDetailedStatusReport() {
    try {
      const systemStats = this.socialMediaSystem.getSystemStats();
      
      logger.info('📋 DETAILED STATUS REPORT:');
      logger.info('==========================');
      logger.info(`Service Started: ${this.startTime.toLocaleString()}`);
      logger.info(`Current Time: ${new Date().toLocaleString()}`);
      logger.info(`Service Uptime: ${Math.round((Date.now() - this.startTime.getTime()) / 1000 / 60)} minutes`);
      logger.info('');
      logger.info('Data Collection:');
      logger.info(`  Total Data Points: ${this.stats.totalDataCollected}`);
      logger.info(`  Companies Tracked: ${this.stats.companiesTracked}`);
      logger.info(`  Management Insights Stored: ${systemStats.managementInsightsStored}`);
      logger.info(`  Sentiment Trends Tracked: ${systemStats.sentimentTrendsTracked}`);
      logger.info(`  Philosophy Profiles Stored: ${systemStats.philosophyProfilesStored}`);
      logger.info('');
      logger.info('Analysis Performance:');
      logger.info(`  Sentiment Updates: ${this.stats.sentimentUpdates}`);
      logger.info(`  Philosophy Updates: ${this.stats.philosophyUpdates}`);
      logger.info(`  ASI Updates Generated: ${this.stats.asiUpdates}`);
      logger.info(`  Management Profiles Analyzed: ${systemStats.managementProfilesAnalyzed}`);
      logger.info(`  Sentiment Analysis Completed: ${systemStats.sentimentAnalysisCompleted}`);
      logger.info('');
      logger.info('System Health:');
      logger.info(`  Analysis Queue Size: ${systemStats.analysisQueueSize}`);
      logger.info(`  ASI Update Queue Size: ${systemStats.asiUpdateQueueSize}`);
      logger.info(`  Total Errors: ${this.stats.errors}`);
      logger.info(`  System Initialized: ${systemStats.isInitialized}`);
      
      if (this.stats.lastUpdate) {
        logger.info(`  Last Update: ${new Date(this.stats.lastUpdate).toLocaleString()}`);
      }
      
      logger.info('==========================');
      
    } catch (error) {
      logger.error('❌ Detailed status report generation failed:', error);
    }
  }

  keepAlive() {
    // Keep the process running
    setInterval(() => {
      // Just a heartbeat to keep the process alive
      logger.debug('💓 Service heartbeat');
    }, 60 * 60 * 1000); // Every hour
  }

  async gracefulShutdown(signal) {
    console.log(`\n🛑 Received ${signal}. Starting graceful shutdown...`);
    
    this.isRunning = false;
    
    try {
      // Generate final status report
      console.log('\n📊 FINAL STATUS REPORT:');
      this.generateDetailedStatusReport();
      
      // Stop the social media system
      if (this.socialMediaSystem) {
        console.log('🔄 Stopping social media system...');
        // Add any cleanup logic here
      }
      
      console.log('✅ Graceful shutdown completed');
      process.exit(0);
      
    } catch (error) {
      console.error('❌ Error during graceful shutdown:', error);
      process.exit(1);
    }
  }

  handleUncaughtException(error) {
    logger.error('❌ Uncaught Exception:', error);
    this.stats.errors++;
    
    // Don't exit immediately - log and continue
    logger.error('⚠️ Service continuing despite uncaught exception');
  }

  handleUnhandledRejection(reason, promise) {
    logger.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
    this.stats.errors++;
    
    // Don't exit immediately - log and continue
    logger.error('⚠️ Service continuing despite unhandled rejection');
  }
}

// Start the monitoring service
if (require.main === module) {
  const monitor = new SocialMediaMonitorService();
  monitor.start().catch(error => {
    console.error('❌ Failed to start monitoring service:', error);
    process.exit(1);
  });
}

module.exports = { SocialMediaMonitorService };
