/**
 * 🎯 SUPPLY CHAIN INTELLIGENCE MAIN ENTRY POINT
 * Complete supply chain intelligence system for ASI Financial Analysis Platform
 * 
 * @author ASI Engineering Team
 * @version 1.0.0
 */

const logger = require('../utils/logger');
const SupplyChainIntelligenceSystem = require('./supply-chain-setup');

class SupplyChainMain {
  constructor() {
    this.system = null;
    this.isRunning = false;
  }

  async initialize() {
    try {
      logger.info('🚀 Starting Supply Chain Intelligence System...');
      
      // Create system instance with configuration
      this.system = new SupplyChainIntelligenceSystem({
        enableCommodityMonitoring: true,
        enableLogisticsTracking: true,
        enableManufacturingMonitoring: true,
        enableRiskEngine: true,
        enableASIIntegration: true,
        
        autoStart: true,
        enableHealthChecks: true,
        enableReporting: true,
        enableAlerting: true,
        
        // Update intervals (in milliseconds)
        healthCheckInterval: 10 * 60 * 1000, // 10 minutes
        reportGenerationInterval: 60 * 60 * 1000, // 1 hour
        alertCheckInterval: 5 * 60 * 1000, // 5 minutes
      });
      
      // Setup event listeners
      this.setupEventListeners();
      
      // Initialize the system
      await this.system.initialize();
      
      this.isRunning = true;
      
      logger.info('✅ Supply Chain Intelligence System is now operational');
      
      // Display system status
      this.displaySystemStatus();
      
      // Setup graceful shutdown
      this.setupGracefulShutdown();
      
    } catch (error) {
      logger.error('❌ Failed to initialize Supply Chain Intelligence System:', error);
      process.exit(1);
    }
  }

  setupEventListeners() {
    // System events
    this.system.on('systemStarted', (status) => {
      logger.info('🎉 Supply Chain Intelligence System started successfully');
      this.displaySystemInfo();
    });
    
    this.system.on('systemStopped', (status) => {
      logger.info('🛑 Supply Chain Intelligence System stopped');
      this.isRunning = false;
    });
    
    // Component events
    this.system.on('componentEvent', (event) => {
      logger.debug(`📊 Component Event: ${event.component} - ${event.eventType}`);
    });
    
    // Health check events
    this.system.on('healthCheckComplete', (healthCheck) => {
      if (healthCheck.overallHealth !== 'healthy') {
        logger.warn(`🏥 Health Check: ${healthCheck.overallHealth} - Issues: ${healthCheck.issues.join(', ')}`);
      } else {
        logger.debug('🏥 Health Check: System healthy');
      }
    });
    
    // Report events
    this.system.on('systemReportGenerated', (report) => {
      logger.info(`📊 System Report Generated - Components Active: ${report.systemOverview.componentsActive}/${report.systemOverview.totalComponents}`);
    });
    
    // Alert events
    this.system.on('alertTriggered', (alert) => {
      logger.warn(`🚨 ALERT [${alert.severity.toUpperCase()}]: ${alert.message}`);
      
      if (alert.severity === 'high') {
        this.handleCriticalAlert(alert);
      }
    });
    
    this.system.on('alertAcknowledged', (alert) => {
      logger.info(`✅ Alert acknowledged: ${alert.id}`);
    });
  }

  handleCriticalAlert(alert) {
    logger.error(`🚨 CRITICAL ALERT: ${alert.message}`);
    
    // For critical alerts, you might want to:
    // 1. Send notifications
    // 2. Trigger automated responses
    // 3. Log to external systems
    // 4. Create backup
    
    if (alert.type === 'component_health') {
      logger.warn('🔧 Attempting component recovery...');
      // Could implement automatic component restart logic here
    }
  }

  displaySystemStatus() {
    const status = this.system.getSystemStatus();
    
    console.log('\n' + '='.repeat(80));
    console.log('🎯 SUPPLY CHAIN INTELLIGENCE SYSTEM STATUS');
    console.log('='.repeat(80));
    console.log(`Status: ${status.status.toUpperCase()}`);
    console.log(`Start Time: ${status.startTime}`);
    console.log(`Uptime: ${this.formatUptime(status.uptime)}`);
    console.log(`Active Alerts: ${status.alertsActive.filter(a => !a.acknowledged).length}`);
    console.log('\nComponent Status:');
    
    Object.entries(status.componentsStatus).forEach(([component, componentStatus]) => {
      const statusIcon = componentStatus === 'running' ? '✅' : 
                        componentStatus === 'error' ? '❌' : 
                        componentStatus === 'stopped' ? '🛑' : '⚠️';
      console.log(`  ${statusIcon} ${component}: ${componentStatus}`);
    });
    
    console.log('\nSystem Statistics:');
    console.log(`  System Starts: ${status.stats.systemStarts}`);
    console.log(`  Health Checks: ${status.stats.healthChecks}`);
    console.log(`  Reports Generated: ${status.stats.reportsGenerated}`);
    console.log(`  Alerts Triggered: ${status.stats.alertsTriggered}`);
    console.log('='.repeat(80) + '\n');
  }

  displaySystemInfo() {
    console.log('\n' + '='.repeat(80));
    console.log('📋 SUPPLY CHAIN INTELLIGENCE SYSTEM INFORMATION');
    console.log('='.repeat(80));
    console.log('🏭 Manufacturing Output Monitor - Tracks industrial production and capacity');
    console.log('🚚 Logistics Performance Tracker - Monitors port, railway, and road transport');
    console.log('📈 Commodity Price Monitor - Real-time commodity price tracking and alerts');
    console.log('⚠️  Supply Chain Risk Engine - Risk assessment and disruption prediction');
    console.log('🔗 ASI Integration Module - Investment insights and industry analysis');
    console.log('\n📊 Key Features:');
    console.log('  • Real-time data collection and analysis');
    console.log('  • Predictive risk assessment and mitigation strategies');
    console.log('  • Investment recommendations based on supply chain intelligence');
    console.log('  • Industry-specific impact analysis for Indian equity markets');
    console.log('  • Automated alerting and reporting');
    console.log('  • Health monitoring and system diagnostics');
    console.log('\n🎯 Integration:');
    console.log('  • Seamless integration with ASI Financial Analysis Platform');
    console.log('  • Event-driven architecture for real-time updates');
    console.log('  • Modular design for easy maintenance and scaling');
    console.log('='.repeat(80) + '\n');
  }

  formatUptime(uptimeMs) {
    if (!uptimeMs) return 'N/A';
    
    const seconds = Math.floor(uptimeMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    
    if (days > 0) return `${days}d ${hours % 24}h ${minutes % 60}m`;
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  }

  setupGracefulShutdown() {
    const gracefulShutdown = async (signal) => {
      logger.info(`\n🛑 Received ${signal}. Initiating graceful shutdown...`);
      
      try {
        if (this.system && this.isRunning) {
          // Create backup before shutdown
          await this.system.createBackup();
          
          // Stop the system
          await this.system.stop();
        }
        
        logger.info('✅ Graceful shutdown completed');
        process.exit(0);
        
      } catch (error) {
        logger.error('❌ Error during shutdown:', error);
        process.exit(1);
      }
    };
    
    // Handle various shutdown signals
    process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
    process.on('SIGINT', () => gracefulShutdown('SIGINT'));
    process.on('SIGUSR2', () => gracefulShutdown('SIGUSR2')); // nodemon restart
    
    // Handle uncaught exceptions
    process.on('uncaughtException', (error) => {
      logger.error('❌ Uncaught Exception:', error);
      gracefulShutdown('uncaughtException');
    });
    
    process.on('unhandledRejection', (reason, promise) => {
      logger.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
      gracefulShutdown('unhandledRejection');
    });
  }

  // CLI Commands
  async getStatus() {
    if (!this.system) {
      console.log('❌ System not initialized');
      return;
    }
    
    this.displaySystemStatus();
  }

  async getReport() {
    if (!this.system) {
      console.log('❌ System not initialized');
      return;
    }
    
    try {
      const report = await this.system.getSystemReport();
      console.log('\n📊 SYSTEM REPORT');
      console.log('='.repeat(50));
      console.log(JSON.stringify(report, null, 2));
    } catch (error) {
      logger.error('❌ Failed to generate report:', error);
    }
  }

  async getAlerts() {
    if (!this.system) {
      console.log('❌ System not initialized');
      return;
    }
    
    const alerts = this.system.getActiveAlerts();
    
    console.log('\n🚨 ACTIVE ALERTS');
    console.log('='.repeat(50));
    
    if (alerts.length === 0) {
      console.log('✅ No active alerts');
    } else {
      alerts.forEach(alert => {
        console.log(`[${alert.severity.toUpperCase()}] ${alert.type}: ${alert.message}`);
        console.log(`  Time: ${alert.timestamp}`);
        console.log(`  ID: ${alert.id}\n`);
      });
    }
  }

  async acknowledgeAlert(alertId) {
    if (!this.system) {
      console.log('❌ System not initialized');
      return;
    }
    
    this.system.acknowledgeAlert(alertId);
    console.log(`✅ Alert ${alertId} acknowledged`);
  }

  async restart() {
    if (!this.system) {
      console.log('❌ System not initialized');
      return;
    }
    
    try {
      logger.info('🔄 Restarting system...');
      await this.system.restart();
      logger.info('✅ System restarted successfully');
    } catch (error) {
      logger.error('❌ Failed to restart system:', error);
    }
  }

  async createBackup() {
    if (!this.system) {
      console.log('❌ System not initialized');
      return;
    }
    
    try {
      const backupPath = await this.system.createBackup();
      console.log(`✅ Backup created: ${backupPath}`);
    } catch (error) {
      logger.error('❌ Failed to create backup:', error);
    }
  }
}

// CLI Interface
if (require.main === module) {
  const main = new SupplyChainMain();
  
  const command = process.argv[2];
  
  switch (command) {
    case 'start':
      main.initialize();
      break;
      
    case 'status':
      main.initialize().then(() => main.getStatus());
      break;
      
    case 'report':
      main.initialize().then(() => main.getReport());
      break;
      
    case 'alerts':
      main.initialize().then(() => main.getAlerts());
      break;
      
    case 'ack':
      const alertId = process.argv[3];
      if (!alertId) {
        console.log('❌ Please provide alert ID: node supply-chain-main.js ack <alert-id>');
        process.exit(1);
      }
      main.initialize().then(() => main.acknowledgeAlert(alertId));
      break;
      
    case 'restart':
      main.initialize().then(() => main.restart());
      break;
      
    case 'backup':
      main.initialize().then(() => main.createBackup());
      break;
      
    default:
      console.log('\n🎯 Supply Chain Intelligence System CLI');
      console.log('='.repeat(50));
      console.log('Usage: node supply-chain-main.js <command>');
      console.log('\nCommands:');
      console.log('  start     - Start the supply chain intelligence system');
      console.log('  status    - Display system status');
      console.log('  report    - Generate and display system report');
      console.log('  alerts    - Show active alerts');
      console.log('  ack <id>  - Acknowledge an alert');
      console.log('  restart   - Restart the system');
      console.log('  backup    - Create system backup');
      console.log('\nExample:');
      console.log('  node supply-chain-main.js start');
      console.log('  node supply-chain-main.js status');
      console.log('  node supply-chain-main.js ack alert-1234567890-abc123\n');
      break;
  }
}

module.exports = SupplyChainMain;
