/**
 * 🚀 SUPPLY CHAIN INTELLIGENCE SETUP & INTEGRATION
 * Complete setup, configuration, monitoring, and reporting for supply chain intelligence
 * 
 * @author ASI Engineering Team
 * @version 1.0.0
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');

// Import all supply chain components
const CommodityPriceMonitor = require('./commodity-price-monitor');
const LogisticsPerformanceTracker = require('./logistics-performance-tracker');
const ManufacturingOutputMonitor = require('./manufacturing-output-monitor');
const SupplyChainRiskEngine = require('./supply-chain-risk-engine');
const SupplyChainASIIntegration = require('./supply-chain-asi-integration');
const NASAEarthdataClient = require('./nasa-earthdata-client');
const SatelliteDataIntegration = require('./satellite-data-integration');

class SupplyChainIntelligenceSystem extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      enableCommodityMonitoring: true,
      enableLogisticsTracking: true,
      enableManufacturingMonitoring: true,
      enableRiskEngine: true,
      enableASIIntegration: true,
      enableSatelliteIntelligence: true,
      
      autoStart: true,
      enableHealthChecks: true,
      enableReporting: true,
      enableAlerting: true,
      
      healthCheckInterval: 10 * 60 * 1000, // 10 minutes
      reportGenerationInterval: 60 * 60 * 1000, // 1 hour
      alertCheckInterval: 5 * 60 * 1000, // 5 minutes
      
      dataPath: './data/supply-chain-system',
      ...options
    };
    
    // Initialize components
    this.components = {
      commodityMonitor: null,
      logisticsTracker: null,
      manufacturingMonitor: null,
      riskEngine: null,
      asiIntegration: null,
      nasaClient: null,
      satelliteIntegration: null
    };
    
    this.systemStatus = {
      status: 'stopped',
      startTime: null,
      uptime: 0,
      componentsStatus: {},
      lastHealthCheck: null,
      lastReport: null,
      alertsActive: []
    };
    
    this.stats = {
      systemStarts: 0,
      totalUptime: 0,
      healthChecks: 0,
      reportsGenerated: 0,
      alertsTriggered: 0,
      lastUpdate: null
    };
  }

  async initialize() {
    try {
      logger.info('🚀 Initializing Supply Chain Intelligence System...');
      
      await this.createDirectories();
      await this.loadSystemConfiguration();
      await this.initializeComponents();
      await this.setupEventListeners();
      
      if (this.config.autoStart) {
        await this.start();
      }
      
      logger.info('✅ Supply Chain Intelligence System initialized successfully');
      
    } catch (error) {
      logger.error('❌ Supply Chain Intelligence System initialization failed:', error);
      throw error;
    }
  }

  async createDirectories() {
    const dirs = [
      this.config.dataPath,
      path.join(this.config.dataPath, 'system'),
      path.join(this.config.dataPath, 'health-checks'),
      path.join(this.config.dataPath, 'reports'),
      path.join(this.config.dataPath, 'alerts'),
      path.join(this.config.dataPath, 'logs'),
      path.join(this.config.dataPath, 'backups')
    ];
    
    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') throw error;
      }
    }
  }

  async loadSystemConfiguration() {
    try {
      const configPath = path.join(this.config.dataPath, 'system', 'system-config.json');
      const configData = await fs.readFile(configPath, 'utf8');
      const savedConfig = JSON.parse(configData);
      
      // Merge saved configuration with defaults
      this.config = { ...this.config, ...savedConfig };
      
      logger.info('📋 System configuration loaded');
      
    } catch (error) {
      logger.debug('No existing system configuration found, using defaults');
      await this.saveSystemConfiguration();
    }
  }

  async saveSystemConfiguration() {
    try {
      const configPath = path.join(this.config.dataPath, 'system', 'system-config.json');
      await fs.writeFile(configPath, JSON.stringify(this.config, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save system configuration:', error);
    }
  }

  async initializeComponents() {
    logger.info('🔧 Initializing supply chain components...');
    
    try {
      // Initialize Commodity Price Monitor
      if (this.config.enableCommodityMonitoring) {
        this.components.commodityMonitor = new CommodityPriceMonitor();
        await this.components.commodityMonitor.initialize();
        this.systemStatus.componentsStatus.commodityMonitor = 'initialized';
        logger.info('✅ Commodity Price Monitor initialized');
      }
      
      // Initialize Logistics Performance Tracker
      if (this.config.enableLogisticsTracking) {
        this.components.logisticsTracker = new LogisticsPerformanceTracker();
        await this.components.logisticsTracker.initialize();
        this.systemStatus.componentsStatus.logisticsTracker = 'initialized';
        logger.info('✅ Logistics Performance Tracker initialized');
      }
      
      // Initialize Manufacturing Output Monitor
      if (this.config.enableManufacturingMonitoring) {
        this.components.manufacturingMonitor = new ManufacturingOutputMonitor();
        await this.components.manufacturingMonitor.initialize();
        this.systemStatus.componentsStatus.manufacturingMonitor = 'initialized';
        logger.info('✅ Manufacturing Output Monitor initialized');
      }
      
      // Initialize Supply Chain Risk Engine
      if (this.config.enableRiskEngine) {
        this.components.riskEngine = new SupplyChainRiskEngine();
        await this.components.riskEngine.initialize();
        this.systemStatus.componentsStatus.riskEngine = 'initialized';
        logger.info('✅ Supply Chain Risk Engine initialized');
      }
      
      // Initialize ASI Integration
      if (this.config.enableASIIntegration) {
        this.components.asiIntegration = new SupplyChainASIIntegration();
        await this.components.asiIntegration.initialize();
        this.systemStatus.componentsStatus.asiIntegration = 'initialized';
        logger.info('✅ ASI Integration initialized');
      }
      
      // Initialize NASA Earthdata Client
      if (this.config.enableSatelliteIntelligence) {
        this.components.nasaClient = new NASAEarthdataClient();
        await this.components.nasaClient.initialize();
        this.systemStatus.componentsStatus.nasaClient = 'initialized';
        logger.info('🛰️ NASA Earthdata Client initialized');
      }
      
      // Initialize Satellite Data Integration
      if (this.config.enableSatelliteIntelligence) {
        this.components.satelliteIntegration = new SatelliteDataIntegration();
        await this.components.satelliteIntegration.initialize();
        this.systemStatus.componentsStatus.satelliteIntegration = 'initialized';
        logger.info('📡 Satellite Data Integration initialized');
      }
      
      logger.info('🎯 All supply chain components initialized successfully');
      
    } catch (error) {
      logger.error('❌ Component initialization failed:', error);
      throw error;
    }
  }

  setupEventListeners() {
    logger.info('📡 Setting up event listeners...');
    
    // Commodity Monitor Events
    if (this.components.commodityMonitor) {
      this.components.commodityMonitor.on('priceUpdate', (data) => {
        this.handleComponentEvent('commodityMonitor', 'priceUpdate', data);
      });
      
      this.components.commodityMonitor.on('volatilityAlert', (data) => {
        this.handleComponentEvent('commodityMonitor', 'volatilityAlert', data);
        this.triggerAlert('commodity_volatility', data);
      });
    }
    
    // Logistics Tracker Events
    if (this.components.logisticsTracker) {
      this.components.logisticsTracker.on('performanceUpdate', (data) => {
        this.handleComponentEvent('logisticsTracker', 'performanceUpdate', data);
      });
      
      this.components.logisticsTracker.on('efficiencyAlert', (data) => {
        this.handleComponentEvent('logisticsTracker', 'efficiencyAlert', data);
        this.triggerAlert('logistics_efficiency', data);
      });
    }
    
    // Manufacturing Monitor Events
    if (this.components.manufacturingMonitor) {
      this.components.manufacturingMonitor.on('productionUpdate', (data) => {
        this.handleComponentEvent('manufacturingMonitor', 'productionUpdate', data);
      });
      
      this.components.manufacturingMonitor.on('capacityAlert', (data) => {
        this.handleComponentEvent('manufacturingMonitor', 'capacityAlert', data);
        this.triggerAlert('manufacturing_capacity', data);
      });
    }
    
    // Risk Engine Events
    if (this.components.riskEngine) {
      this.components.riskEngine.on('riskAssessmentUpdate', (data) => {
        this.handleComponentEvent('riskEngine', 'riskAssessmentUpdate', data);
      });
      
      this.components.riskEngine.on('riskAlert', (data) => {
        this.handleComponentEvent('riskEngine', 'riskAlert', data);
        this.triggerAlert('supply_chain_risk', data);
      });
    }
    
    // ASI Integration Events
    if (this.components.asiIntegration) {
      this.components.asiIntegration.on('asiDataStreamUpdate', (data) => {
        this.handleComponentEvent('asiIntegration', 'asiDataStreamUpdate', data);
      });
      
      this.components.asiIntegration.on('supplyChainSignal', (data) => {
        this.handleComponentEvent('asiIntegration', 'supplyChainSignal', data);
        
        // Forward supply chain signals to ASI Integration
        if (data.actionRequired) {
          this.triggerAlert('supply_chain_signal', data);
        }
      });
    }
    
    // NASA Satellite Intelligence Events
    if (this.components.satelliteIntegration) {
      this.components.satelliteIntegration.on('satelliteDataUpdate', (data) => {
        this.handleComponentEvent('satelliteIntegration', 'satelliteDataUpdate', data);
        
        // Forward satellite insights to ASI Integration
        if (this.components.asiIntegration) {
          this.components.asiIntegration.processSatelliteIntelligence(data);
        }
      });
      
      this.components.satelliteIntegration.on('environmentalAlert', (data) => {
        this.handleComponentEvent('satelliteIntegration', 'environmentalAlert', data);
        this.triggerAlert('environmental_risk', data);
      });
      
      this.components.satelliteIntegration.on('portActivityUpdate', (data) => {
        this.handleComponentEvent('satelliteIntegration', 'portActivityUpdate', data);
        
        // Correlate with logistics data
        if (this.components.logisticsTracker) {
          this.components.logisticsTracker.correlateSatelliteData(data);
        }
      });
      
      this.components.satelliteIntegration.on('industrialActivityUpdate', (data) => {
        this.handleComponentEvent('satelliteIntegration', 'industrialActivityUpdate', data);
        
        // Correlate with manufacturing data
        if (this.components.manufacturingMonitor) {
          this.components.manufacturingMonitor.correlateSatelliteData(data);
        }
      });
    }
    
    logger.info('✅ Event listeners configured');
  }

  handleComponentEvent(component, eventType, data) {
    logger.debug(`📊 ${component} event: ${eventType}`);
    
    // Update component status
    this.systemStatus.componentsStatus[component] = 'active';
    
    // Forward events to ASI Integration if applicable
    if (this.components.asiIntegration && component !== 'asiIntegration') {
      this.components.asiIntegration.handleSupplyChainEvent(eventType, data);
    }
    
    // Emit system-level event
    this.emit('componentEvent', {
      component,
      eventType,
      data,
      timestamp: new Date().toISOString()
    });
  }

  async start() {
    try {
      logger.info('🚀 Starting Supply Chain Intelligence System...');
      
      this.systemStatus.status = 'starting';
      this.systemStatus.startTime = new Date().toISOString();
      
      // Start all components
      await this.startComponents();
      
      // Start system processes
      this.startSystemProcesses();
      
      this.systemStatus.status = 'running';
      this.stats.systemStarts++;
      this.stats.lastUpdate = new Date().toISOString();
      
      await this.saveSystemStatus();
      
      logger.info('✅ Supply Chain Intelligence System started successfully');
      this.emit('systemStarted', this.systemStatus);
      
    } catch (error) {
      this.systemStatus.status = 'error';
      logger.error('❌ System start failed:', error);
      throw error;
    }
  }

  async startComponents() {
    logger.info('▶️ Starting supply chain components...');
    
    const startPromises = [];
    
    Object.entries(this.components).forEach(([name, component]) => {
      if (component && typeof component.start === 'function') {
        startPromises.push(
          component.start().then(() => {
            this.systemStatus.componentsStatus[name] = 'running';
            logger.info(`✅ ${name} started`);
          }).catch(error => {
            this.systemStatus.componentsStatus[name] = 'error';
            logger.error(`❌ ${name} start failed:`, error);
          })
        );
      }
    });
    
    await Promise.all(startPromises);
  }

  startSystemProcesses() {
    logger.info('⚙️ Starting system processes...');
    
    // Health check process
    if (this.config.enableHealthChecks) {
      this.performHealthCheck();
      setInterval(() => this.performHealthCheck(), this.config.healthCheckInterval);
    }
    
    // Report generation process
    if (this.config.enableReporting) {
      this.generateSystemReport();
      setInterval(() => this.generateSystemReport(), this.config.reportGenerationInterval);
    }
    
    // Alert monitoring process
    if (this.config.enableAlerting) {
      this.checkAlerts();
      setInterval(() => this.checkAlerts(), this.config.alertCheckInterval);
    }
    
    // Uptime tracking
    setInterval(() => {
      if (this.systemStatus.status === 'running') {
        this.systemStatus.uptime = Date.now() - new Date(this.systemStatus.startTime).getTime();
        this.stats.totalUptime += 1000; // Add 1 second
      }
    }, 1000);
  }

  async performHealthCheck() {
    try {
      logger.debug('🏥 Performing system health check...');
      
      const healthCheck = {
        timestamp: new Date().toISOString(),
        systemStatus: this.systemStatus.status,
        uptime: this.systemStatus.uptime,
        components: {},
        overallHealth: 'healthy',
        issues: []
      };
      
      // Check each component
      for (const [name, component] of Object.entries(this.components)) {
        if (component) {
          try {
            const componentHealth = await this.checkComponentHealth(name, component);
            healthCheck.components[name] = componentHealth;
            
            if (componentHealth.status !== 'healthy') {
              healthCheck.issues.push(`${name}: ${componentHealth.issue}`);
              healthCheck.overallHealth = 'degraded';
            }
          } catch (error) {
            healthCheck.components[name] = {
              status: 'error',
              issue: error.message
            };
            healthCheck.issues.push(`${name}: Health check failed`);
            healthCheck.overallHealth = 'unhealthy';
          }
        }
      }
      
      this.systemStatus.lastHealthCheck = healthCheck.timestamp;
      this.stats.healthChecks++;
      
      // Save health check
      await this.saveHealthCheck(healthCheck);
      
      // Trigger alerts if unhealthy
      if (healthCheck.overallHealth !== 'healthy') {
        this.triggerAlert('system_health', healthCheck);
      }
      
      this.emit('healthCheckComplete', healthCheck);
      
    } catch (error) {
      logger.error('❌ Health check failed:', error);
    }
  }

  async checkComponentHealth(name, component) {
    // Basic health check - can be extended with component-specific checks
    const health = {
      status: 'healthy',
      lastActivity: null,
      issue: null
    };
    
    // Check if component has stats method
    if (typeof component.getStats === 'function') {
      try {
        const stats = component.getStats();
        health.lastActivity = stats.lastUpdate;
        
        // Check if component has been active recently (within last hour)
        if (stats.lastUpdate) {
          const lastUpdate = new Date(stats.lastUpdate);
          const hourAgo = new Date(Date.now() - 60 * 60 * 1000);
          
          if (lastUpdate < hourAgo) {
            health.status = 'stale';
            health.issue = 'No recent activity detected';
          }
        }
      } catch (error) {
        health.status = 'error';
        health.issue = 'Failed to get component stats';
      }
    }
    
    return health;
  }

  async saveHealthCheck(healthCheck) {
    try {
      const filename = `health-check-${Date.now()}.json`;
      const filePath = path.join(this.config.dataPath, 'health-checks', filename);
      await fs.writeFile(filePath, JSON.stringify(healthCheck, null, 2));
      
      // Keep only last 100 health checks
      await this.cleanupOldFiles('health-checks', 100);
      
    } catch (error) {
      logger.error('❌ Failed to save health check:', error);
    }
  }

  async generateSystemReport() {
    try {
      logger.debug('📊 Generating system report...');
      
      const report = {
        timestamp: new Date().toISOString(),
        systemOverview: {
          status: this.systemStatus.status,
          uptime: this.systemStatus.uptime,
          startTime: this.systemStatus.startTime,
          componentsActive: Object.values(this.systemStatus.componentsStatus).filter(s => s === 'running').length,
          totalComponents: Object.keys(this.components).length
        },
        statistics: this.stats,
        componentStats: {},
        recentAlerts: this.systemStatus.alertsActive.slice(-10),
        performance: await this.getPerformanceMetrics()
      };
      
      // Collect stats from each component
      for (const [name, component] of Object.entries(this.components)) {
        if (component && typeof component.getStats === 'function') {
          try {
            report.componentStats[name] = component.getStats();
          } catch (error) {
            report.componentStats[name] = { error: 'Failed to get stats' };
          }
        }
      }
      
      this.systemStatus.lastReport = report.timestamp;
      this.stats.reportsGenerated++;
      
      // Save report
      await this.saveSystemReport(report);
      
      this.emit('systemReportGenerated', report);
      
      return report;
      
    } catch (error) {
      logger.error('❌ System report generation failed:', error);
    }
  }

  async getPerformanceMetrics() {
    return {
      memoryUsage: process.memoryUsage(),
      cpuUsage: process.cpuUsage(),
      uptime: process.uptime(),
      nodeVersion: process.version
    };
  }

  async saveSystemReport(report) {
    try {
      const filename = `system-report-${Date.now()}.json`;
      const filePath = path.join(this.config.dataPath, 'reports', filename);
      await fs.writeFile(filePath, JSON.stringify(report, null, 2));
      
      // Keep only last 50 reports
      await this.cleanupOldFiles('reports', 50);
      
    } catch (error) {
      logger.error('❌ Failed to save system report:', error);
    }
  }

  async checkAlerts() {
    try {
      // Clean up old alerts (older than 24 hours)
      const dayAgo = Date.now() - 24 * 60 * 60 * 1000;
      this.systemStatus.alertsActive = this.systemStatus.alertsActive.filter(alert => 
        new Date(alert.timestamp).getTime() > dayAgo
      );
      
      // Check for system-level alerts
      await this.checkSystemAlerts();
      
    } catch (error) {
      logger.error('❌ Alert check failed:', error);
    }
  }

  async checkSystemAlerts() {
    // Check system uptime
    if (this.systemStatus.uptime > 7 * 24 * 60 * 60 * 1000) { // 7 days
      this.triggerAlert('system_uptime', {
        message: 'System has been running for over 7 days',
        uptime: this.systemStatus.uptime,
        recommendation: 'Consider restarting for maintenance'
      });
    }
    
    // Check component health
    const unhealthyComponents = Object.entries(this.systemStatus.componentsStatus)
      .filter(([name, status]) => status === 'error')
      .map(([name]) => name);
    
    if (unhealthyComponents.length > 0) {
      this.triggerAlert('component_health', {
        message: 'One or more components are unhealthy',
        unhealthyComponents,
        recommendation: 'Check component logs and restart if necessary'
      });
    }
  }

  triggerAlert(type, data) {
    const alert = {
      id: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      type,
      severity: this.determineAlertSeverity(type, data),
      message: this.generateAlertMessage(type, data),
      data,
      timestamp: new Date().toISOString(),
      acknowledged: false
    };
    
    this.systemStatus.alertsActive.push(alert);
    this.stats.alertsTriggered++;
    
    logger.warn(`🚨 Alert triggered: ${alert.type} - ${alert.message}`);
    
    // Save alert
    this.saveAlert(alert);
    
    this.emit('alertTriggered', alert);
  }

  determineAlertSeverity(type, data) {
    const severityMap = {
      commodity_volatility: 'medium',
      logistics_efficiency: 'medium',
      manufacturing_capacity: 'high',
      supply_chain_risk: 'high',
      supply_chain_signal: 'medium',
      system_health: 'high',
      system_uptime: 'low',
      component_health: 'high'
    };
    
    return severityMap[type] || 'medium';
  }

  generateAlertMessage(type, data) {
    const messageMap = {
      commodity_volatility: `High volatility detected in ${data.commodity || 'commodity'} prices`,
      logistics_efficiency: `Low efficiency detected in ${data.location || 'logistics'} operations`,
      manufacturing_capacity: `Manufacturing capacity alert in ${data.sector || 'sector'}`,
      supply_chain_risk: `High supply chain risk detected in ${data.industry || 'industry'}`,
      supply_chain_signal: `Supply chain signal requires attention: ${data.type}`,
      system_health: `System health degraded: ${data.issues?.join(', ') || 'Multiple issues'}`,
      system_uptime: 'System uptime alert',
      component_health: `Component health issues: ${data.unhealthyComponents?.join(', ') || 'Multiple components'}`
    };
    
    return messageMap[type] || `Alert triggered: ${type}`;
  }

  async saveAlert(alert) {
    try {
      const filename = `alert-${alert.id}.json`;
      const filePath = path.join(this.config.dataPath, 'alerts', filename);
      await fs.writeFile(filePath, JSON.stringify(alert, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save alert:', error);
    }
  }

  async cleanupOldFiles(directory, keepCount) {
    try {
      const dirPath = path.join(this.config.dataPath, directory);
      const files = await fs.readdir(dirPath);
      
      if (files.length > keepCount) {
        // Sort by creation time (filename contains timestamp)
        const sortedFiles = files.sort().reverse();
        const filesToDelete = sortedFiles.slice(keepCount);
        
        for (const file of filesToDelete) {
          await fs.unlink(path.join(dirPath, file));
        }
        
        logger.debug(`🧹 Cleaned up ${filesToDelete.length} old files from ${directory}`);
      }
    } catch (error) {
      logger.error(`❌ Failed to cleanup old files in ${directory}:`, error);
    }
  }

  async saveSystemStatus() {
    try {
      const statusPath = path.join(this.config.dataPath, 'system', 'system-status.json');
      await fs.writeFile(statusPath, JSON.stringify(this.systemStatus, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save system status:', error);
    }
  }

  async stop() {
    try {
      logger.info('🛑 Stopping Supply Chain Intelligence System...');
      
      this.systemStatus.status = 'stopping';
      
      // Stop all components
      const stopPromises = [];
      
      Object.entries(this.components).forEach(([name, component]) => {
        if (component && typeof component.stop === 'function') {
          stopPromises.push(
            component.stop().then(() => {
              this.systemStatus.componentsStatus[name] = 'stopped';
              logger.info(`🛑 ${name} stopped`);
            }).catch(error => {
              logger.error(`❌ ${name} stop failed:`, error);
            })
          );
        }
      });
      
      await Promise.all(stopPromises);
      
      this.systemStatus.status = 'stopped';
      this.systemStatus.startTime = null;
      this.systemStatus.uptime = 0;
      
      await this.saveSystemStatus();
      
      logger.info('✅ Supply Chain Intelligence System stopped');
      this.emit('systemStopped', this.systemStatus);
      
    } catch (error) {
      logger.error('❌ System stop failed:', error);
      throw error;
    }
  }

  async restart() {
    logger.info('🔄 Restarting Supply Chain Intelligence System...');
    
    await this.stop();
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2 seconds
    await this.start();
  }

  getSystemStatus() {
    return {
      ...this.systemStatus,
      stats: this.stats
    };
  }

  getSystemStats() {
    return this.stats;
  }

  async getSystemReport() {
    return await this.generateSystemReport();
  }

  getActiveAlerts() {
    return this.systemStatus.alertsActive.filter(alert => !alert.acknowledged);
  }

  acknowledgeAlert(alertId) {
    const alert = this.systemStatus.alertsActive.find(a => a.id === alertId);
    if (alert) {
      alert.acknowledged = true;
      alert.acknowledgedAt = new Date().toISOString();
      logger.info(`✅ Alert acknowledged: ${alertId}`);
      this.emit('alertAcknowledged', alert);
    }
  }

  async createBackup() {
    try {
      logger.info('💾 Creating system backup...');
      
      const backupData = {
        timestamp: new Date().toISOString(),
        systemStatus: this.systemStatus,
        stats: this.stats,
        config: this.config,
        componentData: {}
      };
      
      // Collect data from each component
      for (const [name, component] of Object.entries(this.components)) {
        if (component && typeof component.getStats === 'function') {
          try {
            backupData.componentData[name] = component.getStats();
          } catch (error) {
            backupData.componentData[name] = { error: 'Failed to backup component data' };
          }
        }
      }
      
      const backupFilename = `system-backup-${Date.now()}.json`;
      const backupPath = path.join(this.config.dataPath, 'backups', backupFilename);
      await fs.writeFile(backupPath, JSON.stringify(backupData, null, 2));
      
      // Keep only last 10 backups
      await this.cleanupOldFiles('backups', 10);
      
      logger.info(`✅ System backup created: ${backupFilename}`);
      return backupPath;
      
    } catch (error) {
      logger.error('❌ System backup failed:', error);
      throw error;
    }
  }
}

module.exports = SupplyChainIntelligenceSystem;
