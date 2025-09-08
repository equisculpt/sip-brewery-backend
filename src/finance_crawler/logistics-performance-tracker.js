/**
 * 🚚 LOGISTICS PERFORMANCE TRACKER
 * Real-time monitoring of port efficiency, railway performance, and road transport
 * 
 * @author ASI Engineering Team
 * @version 1.0.0
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');

class LogisticsPerformanceTracker extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      enablePortTracking: true,
      enableRailwayTracking: true,
      enableRoadTransportTracking: true,
      
      portUpdateInterval: 6 * 60 * 60 * 1000, // 6 hours
      railwayUpdateInterval: 12 * 60 * 60 * 1000, // 12 hours
      roadTransportUpdateInterval: 8 * 60 * 60 * 1000, // 8 hours
      
      dataPath: './data/logistics',
      ...options
    };
    
    // Major Indian ports
    this.ports = {
      mumbai: { name: 'Mumbai Port', capacity: 70000000, specialization: ['containers', 'bulk_cargo'] },
      chennai: { name: 'Chennai Port', capacity: 45000000, specialization: ['containers', 'automobiles'] },
      kolkata: { name: 'Kolkata Port', capacity: 35000000, specialization: ['bulk_cargo', 'coal'] },
      kandla: { name: 'Kandla Port', capacity: 30000000, specialization: ['petroleum', 'chemicals'] },
      paradip: { name: 'Paradip Port', capacity: 25000000, specialization: ['iron_ore', 'coal'] },
      visakhapatnam: { name: 'Visakhapatnam Port', capacity: 40000000, specialization: ['iron_ore', 'containers'] }
    };
    
    // Railway freight corridors
    this.railwayRoutes = {
      'mumbai_delhi': { name: 'Mumbai-Delhi Freight Corridor', distance: 1483, capacity: 5000 },
      'chennai_bangalore': { name: 'Chennai-Bangalore Route', distance: 362, capacity: 3000 },
      'kolkata_mumbai': { name: 'Kolkata-Mumbai Route', distance: 1968, capacity: 4000 },
      'delhi_kolkata': { name: 'Delhi-Kolkata Route', distance: 1472, capacity: 3500 }
    };
    
    // Road transport corridors
    this.roadCorridors = {
      'golden_quadrilateral': { name: 'Golden Quadrilateral', totalLength: 5846, tollCost: 0.05, averageSpeed: 60 },
      'north_south_corridor': { name: 'North-South Corridor', totalLength: 4076, tollCost: 0.04, averageSpeed: 55 },
      'east_west_corridor': { name: 'East-West Corridor', totalLength: 3640, tollCost: 0.045, averageSpeed: 58 }
    };
    
    this.portMetrics = new Map();
    this.railwayMetrics = new Map();
    this.roadTransportMetrics = new Map();
    
    this.stats = {
      portsTracked: 0,
      railwayRoutesTracked: 0,
      roadCorridorsTracked: 0,
      performanceUpdates: 0,
      lastUpdate: null
    };
  }

  async initialize() {
    try {
      logger.info('🚚 Initializing Logistics Performance Tracker...');
      
      await this.createDirectories();
      await this.loadExistingData();
      await this.initializeTracking();
      this.startPerformanceMonitoring();
      
      this.stats.portsTracked = Object.keys(this.ports).length;
      this.stats.railwayRoutesTracked = Object.keys(this.railwayRoutes).length;
      this.stats.roadCorridorsTracked = Object.keys(this.roadCorridors).length;
      
      logger.info(`✅ Logistics Tracker initialized - ${this.stats.portsTracked} ports, ${this.stats.railwayRoutesTracked} railway routes, ${this.stats.roadCorridorsTracked} road corridors`);
      
    } catch (error) {
      logger.error('❌ Logistics Tracker initialization failed:', error);
      throw error;
    }
  }

  async createDirectories() {
    const dirs = [
      this.config.dataPath,
      path.join(this.config.dataPath, 'ports'),
      path.join(this.config.dataPath, 'railways'),
      path.join(this.config.dataPath, 'roads'),
      path.join(this.config.dataPath, 'reports')
    ];
    
    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') throw error;
      }
    }
  }

  async loadExistingData() {
    try {
      const dataFiles = [
        { file: 'ports/port-performance.json', map: this.portMetrics },
        { file: 'railways/railway-performance.json', map: this.railwayMetrics },
        { file: 'roads/road-performance.json', map: this.roadTransportMetrics }
      ];
      
      for (const { file, map } of dataFiles) {
        try {
          const filePath = path.join(this.config.dataPath, file);
          const data = await fs.readFile(filePath, 'utf8');
          const parsedData = JSON.parse(data);
          
          for (const [key, value] of Object.entries(parsedData)) {
            map.set(key, value);
          }
        } catch (error) {
          logger.debug(`No existing data found for ${file}`);
        }
      }
      
    } catch (error) {
      logger.debug('No existing logistics data found, starting fresh');
    }
  }

  async initializeTracking() {
    // Initialize port tracking
    for (const [portKey, port] of Object.entries(this.ports)) {
      if (!this.portMetrics.has(portKey)) {
        this.portMetrics.set(portKey, {
          ...port,
          currentThroughput: 0,
          efficiency: 75,
          congestionLevel: 30,
          averageWaitTime: 4.5,
          operationalStatus: 'normal',
          lastUpdated: new Date().toISOString()
        });
      }
    }
    
    // Initialize railway tracking
    for (const [routeKey, route] of Object.entries(this.railwayRoutes)) {
      if (!this.railwayMetrics.has(routeKey)) {
        this.railwayMetrics.set(routeKey, {
          ...route,
          onTimePerformance: 80,
          averageDelay: 2.5,
          freightVolume: 0,
          capacityUtilization: 65,
          operationalStatus: 'normal',
          lastUpdated: new Date().toISOString()
        });
      }
    }
    
    // Initialize road transport tracking
    for (const [corridorKey, corridor] of Object.entries(this.roadCorridors)) {
      if (!this.roadTransportMetrics.has(corridorKey)) {
        this.roadTransportMetrics.set(corridorKey, {
          ...corridor,
          currentTraffic: 0,
          averageSpeed: corridor.averageSpeed,
          congestionIndex: 0.3,
          accidentRate: 0.02,
          roadCondition: 'good',
          lastUpdated: new Date().toISOString()
        });
      }
    }
  }

  startPerformanceMonitoring() {
    logger.info('📊 Starting logistics performance monitoring...');
    
    if (this.config.enablePortTracking) {
      this.updatePortPerformance();
      setInterval(() => this.updatePortPerformance(), this.config.portUpdateInterval);
    }
    
    if (this.config.enableRailwayTracking) {
      this.updateRailwayPerformance();
      setInterval(() => this.updateRailwayPerformance(), this.config.railwayUpdateInterval);
    }
    
    if (this.config.enableRoadTransportTracking) {
      this.updateRoadTransportPerformance();
      setInterval(() => this.updateRoadTransportPerformance(), this.config.roadTransportUpdateInterval);
    }
  }

  async updatePortPerformance() {
    try {
      const performanceUpdates = [];
      
      for (const [portKey, portData] of this.portMetrics) {
        const newMetrics = this.collectPortMetrics(portKey, portData);
        
        Object.assign(portData, newMetrics, {
          lastUpdated: new Date().toISOString()
        });
        
        const performanceIndicators = this.calculatePortPerformance(portData);
        portData.performanceIndicators = performanceIndicators;
        
        performanceUpdates.push({
          port: portKey,
          metrics: newMetrics,
          performance: performanceIndicators,
          timestamp: portData.lastUpdated
        });
      }
      
      if (performanceUpdates.length > 0) {
        this.stats.performanceUpdates += performanceUpdates.length;
        this.stats.lastUpdate = new Date().toISOString();
        
        await this.savePortData();
        
        this.emit('portPerformanceUpdate', {
          updates: performanceUpdates,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Port performance update failed:', error);
    }
  }

  collectPortMetrics(portKey, portData) {
    const baseEfficiency = 0.7 + Math.random() * 0.25;
    const congestionFactor = Math.random();
    const seasonalFactor = 1 + 0.1 * Math.sin(Date.now() / (1000 * 60 * 60 * 24 * 30));
    
    const currentThroughput = Math.round(portData.capacity * (0.6 + Math.random() * 0.3) * seasonalFactor);
    const efficiency = Math.round(baseEfficiency * 100);
    const congestionLevel = Math.round(congestionFactor * 100);
    const averageWaitTime = 2 + (congestionFactor * 8);
    
    let operationalStatus = 'normal';
    if (congestionLevel > 85) operationalStatus = 'congested';
    else if (congestionLevel > 70) operationalStatus = 'busy';
    else if (efficiency < 60) operationalStatus = 'inefficient';
    
    return {
      currentThroughput,
      efficiency,
      congestionLevel,
      averageWaitTime: Math.round(averageWaitTime * 10) / 10,
      operationalStatus
    };
  }

  calculatePortPerformance(portData) {
    const capacityUtilization = (portData.currentThroughput / portData.capacity) * 100;
    const efficiencyScore = this.calculatePortEfficiencyScore(portData);
    const performanceRating = this.getPortPerformanceRating(efficiencyScore);
    
    return {
      capacityUtilization: Math.round(capacityUtilization),
      efficiencyScore: Math.round(efficiencyScore),
      performanceRating,
      costPerTEU: this.calculatePortCostPerTEU(portData)
    };
  }

  calculatePortEfficiencyScore(portData) {
    const weights = { throughput: 0.3, waitTime: 0.25, congestion: 0.25, operational: 0.2 };
    
    const throughputScore = (portData.currentThroughput / portData.capacity) * 100;
    const waitTimeScore = Math.max(0, 100 - (portData.averageWaitTime * 10));
    const congestionScore = Math.max(0, 100 - portData.congestionLevel);
    const operationalScore = portData.operationalStatus === 'normal' ? 100 : 
                           portData.operationalStatus === 'busy' ? 70 : 40;
    
    return (throughputScore * weights.throughput) +
           (waitTimeScore * weights.waitTime) +
           (congestionScore * weights.congestion) +
           (operationalScore * weights.operational);
  }

  getPortPerformanceRating(score) {
    if (score >= 85) return 'excellent';
    if (score >= 70) return 'good';
    if (score >= 55) return 'average';
    if (score >= 40) return 'poor';
    return 'critical';
  }

  calculatePortCostPerTEU(portData) {
    const baseCost = 150;
    const congestionPenalty = (portData.congestionLevel / 100) * 50;
    const efficiencyBonus = ((portData.efficiency - 70) / 30) * 20;
    
    return Math.round(baseCost + congestionPenalty - efficiencyBonus);
  }

  async updateRailwayPerformance() {
    try {
      const performanceUpdates = [];
      
      for (const [routeKey, routeData] of this.railwayMetrics) {
        const newMetrics = this.collectRailwayMetrics(routeKey, routeData);
        
        Object.assign(routeData, newMetrics, {
          lastUpdated: new Date().toISOString()
        });
        
        const performanceIndicators = this.calculateRailwayPerformance(routeData);
        routeData.performanceIndicators = performanceIndicators;
        
        performanceUpdates.push({
          route: routeKey,
          metrics: newMetrics,
          performance: performanceIndicators,
          timestamp: routeData.lastUpdated
        });
      }
      
      if (performanceUpdates.length > 0) {
        this.stats.performanceUpdates += performanceUpdates.length;
        
        await this.saveRailwayData();
        
        this.emit('railwayPerformanceUpdate', {
          updates: performanceUpdates,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Railway performance update failed:', error);
    }
  }

  collectRailwayMetrics(routeKey, routeData) {
    const basePerformance = 0.75 + Math.random() * 0.2;
    const weatherFactor = 0.9 + Math.random() * 0.2;
    const maintenanceFactor = Math.random() > 0.95 ? 0.7 : 1.0;
    
    const onTimePerformance = Math.round(basePerformance * weatherFactor * maintenanceFactor * 100);
    const averageDelay = Math.random() * 4;
    const freightVolume = Math.round(routeData.capacity * (0.6 + Math.random() * 0.3));
    const capacityUtilization = Math.round((freightVolume / routeData.capacity) * 100);
    
    let operationalStatus = 'normal';
    if (maintenanceFactor < 1.0) operationalStatus = 'maintenance';
    else if (onTimePerformance < 70) operationalStatus = 'disrupted';
    else if (averageDelay > 2) operationalStatus = 'delayed';
    
    return {
      onTimePerformance,
      averageDelay: Math.round(averageDelay * 10) / 10,
      freightVolume,
      capacityUtilization,
      operationalStatus
    };
  }

  calculateRailwayPerformance(routeData) {
    const performanceScore = this.calculateRailwayPerformanceScore(routeData);
    const reliabilityRating = this.getRailwayReliabilityRating(performanceScore);
    
    return {
      performanceScore: Math.round(performanceScore),
      reliabilityRating,
      costPerTonKm: this.calculateRailwayCostPerTonKm(routeData)
    };
  }

  calculateRailwayPerformanceScore(routeData) {
    const weights = { onTime: 0.4, delay: 0.3, capacity: 0.2, operational: 0.1 };
    
    const onTimeScore = routeData.onTimePerformance;
    const delayScore = Math.max(0, 100 - (routeData.averageDelay * 25));
    const capacityScore = routeData.capacityUtilization;
    const operationalScore = routeData.operationalStatus === 'normal' ? 100 : 
                           routeData.operationalStatus === 'delayed' ? 60 : 30;
    
    return (onTimeScore * weights.onTime) +
           (delayScore * weights.delay) +
           (capacityScore * weights.capacity) +
           (operationalScore * weights.operational);
  }

  getRailwayReliabilityRating(score) {
    if (score >= 80) return 'excellent';
    if (score >= 65) return 'good';
    if (score >= 50) return 'average';
    if (score >= 35) return 'poor';
    return 'critical';
  }

  calculateRailwayCostPerTonKm(routeData) {
    const baseCost = 2.5;
    const delayPenalty = routeData.averageDelay * 0.2;
    const efficiencyBonus = ((routeData.onTimePerformance - 70) / 30) * 0.5;
    
    return Math.round((baseCost + delayPenalty - efficiencyBonus) * 100) / 100;
  }

  async updateRoadTransportPerformance() {
    try {
      const performanceUpdates = [];
      
      for (const [corridorKey, corridorData] of this.roadTransportMetrics) {
        const newMetrics = this.collectRoadTransportMetrics(corridorKey, corridorData);
        
        Object.assign(corridorData, newMetrics, {
          lastUpdated: new Date().toISOString()
        });
        
        const performanceIndicators = this.calculateRoadTransportPerformance(corridorData);
        corridorData.performanceIndicators = performanceIndicators;
        
        performanceUpdates.push({
          corridor: corridorKey,
          metrics: newMetrics,
          performance: performanceIndicators,
          timestamp: corridorData.lastUpdated
        });
      }
      
      if (performanceUpdates.length > 0) {
        this.stats.performanceUpdates += performanceUpdates.length;
        
        await this.saveRoadTransportData();
        
        this.emit('roadTransportPerformanceUpdate', {
          updates: performanceUpdates,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Road transport performance update failed:', error);
    }
  }

  collectRoadTransportMetrics(corridorKey, corridorData) {
    const trafficFactor = 0.7 + Math.random() * 0.6;
    const weatherFactor = 0.9 + Math.random() * 0.2;
    const timeOfDay = new Date().getHours();
    const rushHourFactor = (timeOfDay >= 7 && timeOfDay <= 10) || (timeOfDay >= 17 && timeOfDay <= 20) ? 0.7 : 1.0;
    
    const currentTraffic = Math.round(trafficFactor * 1000);
    const averageSpeed = Math.round(corridorData.averageSpeed * weatherFactor * rushHourFactor);
    const congestionIndex = Math.min(1.0, trafficFactor * rushHourFactor);
    const accidentRate = 0.01 + Math.random() * 0.02;
    
    let roadCondition = 'good';
    if (weatherFactor < 0.95) roadCondition = 'poor';
    else if (congestionIndex > 0.8) roadCondition = 'congested';
    
    return {
      currentTraffic,
      averageSpeed,
      congestionIndex: Math.round(congestionIndex * 100) / 100,
      accidentRate: Math.round(accidentRate * 1000) / 1000,
      roadCondition
    };
  }

  calculateRoadTransportPerformance(corridorData) {
    const efficiencyScore = this.calculateRoadEfficiencyScore(corridorData);
    const safetyRating = this.getRoadSafetyRating(corridorData.accidentRate);
    
    return {
      efficiencyScore: Math.round(efficiencyScore),
      safetyRating,
      costPerTonKm: this.calculateRoadCostPerTonKm(corridorData)
    };
  }

  calculateRoadEfficiencyScore(corridorData) {
    const speedWeight = 0.4;
    const congestionWeight = 0.3;
    const conditionWeight = 0.3;
    
    const speedScore = (corridorData.averageSpeed / 80) * 100;
    const congestionScore = Math.max(0, 100 - (corridorData.congestionIndex * 100));
    const conditionScore = corridorData.roadCondition === 'good' ? 100 : 
                          corridorData.roadCondition === 'congested' ? 60 : 40;
    
    return (speedScore * speedWeight) +
           (congestionScore * congestionWeight) +
           (conditionScore * conditionWeight);
  }

  getRoadSafetyRating(accidentRate) {
    if (accidentRate <= 0.01) return 'excellent';
    if (accidentRate <= 0.02) return 'good';
    if (accidentRate <= 0.03) return 'average';
    if (accidentRate <= 0.04) return 'poor';
    return 'critical';
  }

  calculateRoadCostPerTonKm(corridorData) {
    const baseCost = 3.0;
    const congestionPenalty = corridorData.congestionIndex * 1.0;
    const speedBonus = ((corridorData.averageSpeed - 40) / 40) * 0.5;
    
    return Math.round((baseCost + congestionPenalty - speedBonus) * 100) / 100;
  }

  async savePortData() {
    try {
      const filePath = path.join(this.config.dataPath, 'ports', 'port-performance.json');
      const data = Object.fromEntries(this.portMetrics);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save port data:', error);
    }
  }

  async saveRailwayData() {
    try {
      const filePath = path.join(this.config.dataPath, 'railways', 'railway-performance.json');
      const data = Object.fromEntries(this.railwayMetrics);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save railway data:', error);
    }
  }

  async saveRoadTransportData() {
    try {
      const filePath = path.join(this.config.dataPath, 'roads', 'road-performance.json');
      const data = Object.fromEntries(this.roadTransportMetrics);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save road transport data:', error);
    }
  }

  getStats() {
    return {
      ...this.stats,
      currentMetrics: {
        ports: this.portMetrics.size,
        railwayRoutes: this.railwayMetrics.size,
        roadCorridors: this.roadTransportMetrics.size
      }
    };
  }

  async generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: this.getStats(),
      portPerformance: Object.fromEntries(this.portMetrics),
      railwayPerformance: Object.fromEntries(this.railwayMetrics),
      roadTransportPerformance: Object.fromEntries(this.roadTransportMetrics)
    };
    
    const reportPath = path.join(this.config.dataPath, 'reports', `logistics-report-${Date.now()}.json`);
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    return report;
  }
}

module.exports = LogisticsPerformanceTracker;
