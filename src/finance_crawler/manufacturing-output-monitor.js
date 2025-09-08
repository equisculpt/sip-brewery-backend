/**
 * 🏭 MANUFACTURING OUTPUT MONITOR
 * Industrial production tracking, capacity utilization analysis, and supply chain health assessment
 * 
 * @author ASI Engineering Team
 * @version 1.0.0
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');

class ManufacturingOutputMonitor extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      enableProductionTracking: true,
      enableCapacityAnalysis: true,
      enableSupplyChainHealth: true,
      
      productionUpdateInterval: 4 * 60 * 60 * 1000, // 4 hours
      capacityAnalysisInterval: 6 * 60 * 60 * 1000, // 6 hours
      healthAssessmentInterval: 8 * 60 * 60 * 1000, // 8 hours
      
      dataPath: './data/manufacturing',
      ...options
    };
    
    // Manufacturing sectors with detailed tracking
    this.manufacturingSectors = {
      automotive: {
        name: 'Automotive Manufacturing',
        keyCompanies: ['TATAMOTORS', 'MARUTI', 'BAJAJ-AUTO', 'MAHINDRA', 'HEROMOTOCO'],
        productionCapacity: 25000000,
        keyInputs: ['steel', 'aluminum', 'rubber', 'electronics'],
        seasonality: { q1: 0.9, q2: 1.1, q3: 0.95, q4: 1.05 },
        exportDependency: 0.15
      },
      pharmaceuticals: {
        name: 'Pharmaceutical Manufacturing',
        keyCompanies: ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'AUROPHARMA'],
        productionCapacity: 180000000,
        keyInputs: ['api', 'excipients', 'packaging_materials', 'solvents'],
        seasonality: { q1: 1.0, q2: 1.0, q3: 1.0, q4: 1.0 },
        exportDependency: 0.65
      },
      textiles: {
        name: 'Textile Manufacturing',
        keyCompanies: ['WELSPUNIND', 'TRIDENT', 'VARDHMAN', 'ARVIND', 'RAYMOND'],
        productionCapacity: 45000000,
        keyInputs: ['cotton', 'polyester', 'dyes', 'chemicals'],
        seasonality: { q1: 1.1, q2: 0.9, q3: 1.0, q4: 1.0 },
        exportDependency: 0.45
      },
      chemicals: {
        name: 'Chemical Manufacturing',
        keyCompanies: ['RELIANCE', 'UPL', 'PIDILITIND', 'AARTI', 'BALRAMCHIN'],
        productionCapacity: 95000000,
        keyInputs: ['crude_oil', 'natural_gas', 'coal', 'minerals'],
        seasonality: { q1: 0.95, q2: 1.05, q3: 1.0, q4: 1.0 },
        exportDependency: 0.35
      },
      steel: {
        name: 'Steel Manufacturing',
        keyCompanies: ['TATASTEEL', 'JSWSTEEL', 'SAIL', 'JINDALSTEL', 'NMDC'],
        productionCapacity: 120000000,
        keyInputs: ['iron_ore', 'coking_coal', 'limestone', 'scrap'],
        seasonality: { q1: 1.0, q2: 1.05, q3: 0.95, q4: 1.0 },
        exportDependency: 0.25
      },
      electronics: {
        name: 'Electronics Manufacturing',
        keyCompanies: ['DIXON', 'AMBER', 'VOLTAS', 'BLUESTARCO', 'WHIRLPOOL'],
        productionCapacity: 350000000,
        keyInputs: ['semiconductors', 'pcb', 'metals', 'plastics'],
        seasonality: { q1: 0.9, q2: 0.95, q3: 1.1, q4: 1.05 },
        exportDependency: 0.20
      },
      fmcg: {
        name: 'FMCG Manufacturing',
        keyCompanies: ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
        productionCapacity: 85000000,
        keyInputs: ['agricultural_products', 'packaging', 'chemicals', 'energy'],
        seasonality: { q1: 1.0, q2: 1.1, q3: 0.95, q4: 0.95 },
        exportDependency: 0.10
      }
    };
    
    this.productionMetrics = new Map();
    this.capacityMetrics = new Map();
    this.supplyChainHealth = new Map();
    
    this.stats = {
      sectorsTracked: 0,
      companiesMonitored: 0,
      productionUpdates: 0,
      capacityAnalyses: 0,
      healthAssessments: 0,
      lastUpdate: null
    };
  }

  async initialize() {
    try {
      logger.info('🏭 Initializing Manufacturing Output Monitor...');
      
      await this.createDirectories();
      await this.loadExistingData();
      await this.initializeManufacturingTracking();
      this.startProductionMonitoring();
      
      this.stats.sectorsTracked = Object.keys(this.manufacturingSectors).length;
      this.stats.companiesMonitored = Object.values(this.manufacturingSectors)
        .reduce((total, sector) => total + sector.keyCompanies.length, 0);
      
      logger.info(`✅ Manufacturing Monitor initialized - tracking ${this.stats.sectorsTracked} sectors, ${this.stats.companiesMonitored} companies`);
      
    } catch (error) {
      logger.error('❌ Manufacturing Monitor initialization failed:', error);
      throw error;
    }
  }

  async createDirectories() {
    const dirs = [
      this.config.dataPath,
      path.join(this.config.dataPath, 'production'),
      path.join(this.config.dataPath, 'capacity'),
      path.join(this.config.dataPath, 'health'),
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
        { file: 'production/production-metrics.json', map: this.productionMetrics },
        { file: 'capacity/capacity-metrics.json', map: this.capacityMetrics },
        { file: 'health/supply-chain-health.json', map: this.supplyChainHealth }
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
      logger.debug('No existing manufacturing data found, starting fresh');
    }
  }

  async initializeManufacturingTracking() {
    for (const [sectorKey, sector] of Object.entries(this.manufacturingSectors)) {
      if (!this.productionMetrics.has(sectorKey)) {
        this.productionMetrics.set(sectorKey, {
          ...sector,
          currentProduction: 0,
          capacityUtilization: 70,
          efficiency: 75,
          qualityScore: 85,
          operationalStatus: 'normal',
          lastUpdated: new Date().toISOString()
        });
      }
      
      if (!this.capacityMetrics.has(sectorKey)) {
        this.capacityMetrics.set(sectorKey, {
          totalCapacity: sector.productionCapacity,
          availableCapacity: sector.productionCapacity * 0.85,
          utilizationRate: 70,
          expansionPlans: [],
          bottlenecks: [],
          lastUpdated: new Date().toISOString()
        });
      }
      
      if (!this.supplyChainHealth.has(sectorKey)) {
        this.supplyChainHealth.set(sectorKey, {
          overallHealth: 75,
          inputAvailability: 80,
          supplierReliability: 75,
          logisticsEfficiency: 70,
          inventoryLevels: 65,
          riskFactors: [],
          lastUpdated: new Date().toISOString()
        });
      }
    }
  }

  startProductionMonitoring() {
    logger.info('📊 Starting manufacturing production monitoring...');
    
    if (this.config.enableProductionTracking) {
      this.updateProductionMetrics();
      setInterval(() => this.updateProductionMetrics(), this.config.productionUpdateInterval);
    }
    
    if (this.config.enableCapacityAnalysis) {
      this.analyzeCapacityUtilization();
      setInterval(() => this.analyzeCapacityUtilization(), this.config.capacityAnalysisInterval);
    }
    
    if (this.config.enableSupplyChainHealth) {
      this.assessSupplyChainHealth();
      setInterval(() => this.assessSupplyChainHealth(), this.config.healthAssessmentInterval);
    }
  }

  async updateProductionMetrics() {
    try {
      const productionUpdates = [];
      
      for (const [sectorKey, productionData] of this.productionMetrics) {
        const newMetrics = this.collectProductionData(sectorKey, productionData);
        
        Object.assign(productionData, newMetrics, {
          lastUpdated: new Date().toISOString()
        });
        
        const performanceAnalysis = this.analyzeProductionPerformance(productionData);
        productionData.performanceAnalysis = performanceAnalysis;
        
        productionUpdates.push({
          sector: sectorKey,
          metrics: newMetrics,
          performance: performanceAnalysis,
          timestamp: productionData.lastUpdated
        });
      }
      
      if (productionUpdates.length > 0) {
        this.stats.productionUpdates += productionUpdates.length;
        this.stats.lastUpdate = new Date().toISOString();
        
        await this.saveProductionData();
        
        this.emit('productionMetricsUpdate', {
          updates: productionUpdates,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Production metrics update failed:', error);
    }
  }

  collectProductionData(sectorKey, productionData) {
    const sector = this.manufacturingSectors[sectorKey];
    
    // Simulate realistic production data with seasonality
    const currentQuarter = Math.floor((new Date().getMonth() + 3) / 3);
    const seasonalFactor = sector.seasonality[`q${currentQuarter}`] || 1.0;
    
    const baseProduction = sector.productionCapacity * (0.65 + Math.random() * 0.25);
    const currentProduction = Math.round(baseProduction * seasonalFactor);
    
    const capacityUtilization = Math.round((currentProduction / sector.productionCapacity) * 100);
    const efficiency = Math.round(70 + Math.random() * 25);
    const qualityScore = Math.round(80 + Math.random() * 15);
    
    let operationalStatus = 'normal';
    if (capacityUtilization > 90) operationalStatus = 'high_demand';
    else if (capacityUtilization < 50) operationalStatus = 'low_demand';
    else if (efficiency < 70) operationalStatus = 'inefficient';
    else if (qualityScore < 85) operationalStatus = 'quality_issues';
    
    const energyConsumption = Math.round(currentProduction * (0.8 + Math.random() * 0.4));
    const wasteGeneration = Math.round(currentProduction * (0.05 + Math.random() * 0.05));
    const employeeProductivity = Math.round(80 + Math.random() * 20);
    
    return {
      currentProduction,
      capacityUtilization,
      efficiency,
      qualityScore,
      operationalStatus,
      energyConsumption,
      wasteGeneration,
      employeeProductivity,
      productionTrend: this.calculateProductionTrend(currentProduction, productionData.currentProduction),
      costPerUnit: this.calculateProductionCost(sectorKey, currentProduction)
    };
  }

  calculateProductionTrend(current, previous) {
    if (!previous || previous === 0) return 'stable';
    
    const change = ((current - previous) / previous) * 100;
    
    if (change > 5) return 'increasing';
    if (change < -5) return 'decreasing';
    return 'stable';
  }

  calculateProductionCost(sectorKey, production) {
    const baseCosts = {
      automotive: 250000,
      pharmaceuticals: 15,
      textiles: 120,
      chemicals: 45000,
      steel: 35000,
      electronics: 8500,
      fmcg: 25
    };
    
    const sector = this.manufacturingSectors[sectorKey];
    const baseCost = baseCosts[sectorKey] || 1000;
    const volumeDiscount = Math.max(0.8, 1 - (production / sector.productionCapacity) * 0.2);
    
    return Math.round(baseCost * volumeDiscount);
  }

  analyzeProductionPerformance(productionData) {
    const performanceScore = this.calculateProductionPerformanceScore(productionData);
    const performanceRating = this.getProductionPerformanceRating(performanceScore);
    const insights = this.generateProductionInsights(productionData);
    
    return {
      performanceScore: Math.round(performanceScore),
      performanceRating,
      insights,
      efficiencyMetrics: this.calculateEfficiencyMetrics(productionData),
      benchmarkComparison: this.benchmarkProductionPerformance(productionData)
    };
  }

  calculateProductionPerformanceScore(productionData) {
    const weights = { capacity: 0.25, efficiency: 0.25, quality: 0.25, productivity: 0.25 };
    
    return (productionData.capacityUtilization * weights.capacity) +
           (productionData.efficiency * weights.efficiency) +
           (productionData.qualityScore * weights.quality) +
           (productionData.employeeProductivity * weights.productivity);
  }

  getProductionPerformanceRating(score) {
    if (score >= 85) return 'excellent';
    if (score >= 75) return 'good';
    if (score >= 65) return 'average';
    if (score >= 55) return 'poor';
    return 'critical';
  }

  generateProductionInsights(productionData) {
    const insights = [];
    
    if (productionData.capacityUtilization > 85) {
      insights.push({
        type: 'capacity_alert',
        message: 'High capacity utilization - consider expansion planning',
        priority: 'high'
      });
    }
    
    if (productionData.efficiency < 70) {
      insights.push({
        type: 'efficiency_concern',
        message: 'Below-average production efficiency detected',
        priority: 'medium'
      });
    }
    
    if (productionData.qualityScore < 85) {
      insights.push({
        type: 'quality_issue',
        message: 'Quality metrics below target threshold',
        priority: 'high'
      });
    }
    
    return insights;
  }

  calculateEfficiencyMetrics(productionData) {
    return {
      overallEfficiency: productionData.efficiency,
      capacityEfficiency: Math.min(100, productionData.capacityUtilization * 1.2),
      qualityEfficiency: productionData.qualityScore,
      energyEfficiency: Math.max(0, 100 - (productionData.energyConsumption / productionData.currentProduction) * 10),
      wasteEfficiency: Math.max(0, 100 - productionData.wasteGeneration * 10),
      laborEfficiency: productionData.employeeProductivity
    };
  }

  benchmarkProductionPerformance(productionData) {
    const industryBenchmarks = {
      capacityUtilization: 75,
      efficiency: 80,
      qualityScore: 90,
      employeeProductivity: 85
    };
    
    return {
      capacityVsBenchmark: productionData.capacityUtilization - industryBenchmarks.capacityUtilization,
      efficiencyVsBenchmark: productionData.efficiency - industryBenchmarks.efficiency,
      qualityVsBenchmark: productionData.qualityScore - industryBenchmarks.qualityScore,
      productivityVsBenchmark: productionData.employeeProductivity - industryBenchmarks.employeeProductivity
    };
  }

  async analyzeCapacityUtilization() {
    try {
      const capacityAnalyses = [];
      
      for (const [sectorKey, capacityData] of this.capacityMetrics) {
        const analysis = this.performCapacityAnalysis(sectorKey, capacityData);
        
        Object.assign(capacityData, analysis, {
          lastUpdated: new Date().toISOString()
        });
        
        capacityAnalyses.push({
          sector: sectorKey,
          analysis,
          timestamp: capacityData.lastUpdated
        });
      }
      
      if (capacityAnalyses.length > 0) {
        this.stats.capacityAnalyses += capacityAnalyses.length;
        
        await this.saveCapacityData();
        
        this.emit('capacityAnalysisUpdate', {
          analyses: capacityAnalyses,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Capacity analysis failed:', error);
    }
  }

  performCapacityAnalysis(sectorKey, capacityData) {
    const productionData = this.productionMetrics.get(sectorKey);
    const sector = this.manufacturingSectors[sectorKey];
    
    const currentUtilization = productionData ? 
      (productionData.currentProduction / sector.productionCapacity) * 100 : 70;
    
    const availableCapacity = sector.productionCapacity - (productionData?.currentProduction || 0);
    const capacityStress = this.assessCapacityStress(currentUtilization);
    const bottlenecks = this.identifyBottlenecks(sectorKey, currentUtilization);
    
    return {
      utilizationRate: Math.round(currentUtilization),
      availableCapacity: Math.round(availableCapacity),
      capacityStress,
      bottlenecks,
      expansionRecommendations: this.generateExpansionRecommendations(sectorKey, currentUtilization)
    };
  }

  assessCapacityStress(utilization) {
    if (utilization > 90) return 'critical';
    if (utilization > 80) return 'high';
    if (utilization > 70) return 'medium';
    if (utilization > 50) return 'low';
    return 'minimal';
  }

  identifyBottlenecks(sectorKey, utilization) {
    const bottlenecks = [];
    
    if (utilization > 85) {
      bottlenecks.push({
        type: 'capacity_constraint',
        description: 'Production capacity nearing maximum',
        impact: 'high',
        recommendation: 'Consider capacity expansion or efficiency improvements'
      });
    }
    
    const sectorBottlenecks = {
      automotive: ['semiconductor_shortage', 'skilled_labor', 'supply_chain_disruption'],
      pharmaceuticals: ['regulatory_approval', 'api_availability', 'quality_compliance'],
      textiles: ['raw_material_cost', 'export_demand', 'power_availability'],
      chemicals: ['feedstock_availability', 'environmental_compliance', 'safety_regulations'],
      steel: ['iron_ore_supply', 'coking_coal_cost', 'environmental_norms'],
      electronics: ['component_shortage', 'technology_obsolescence', 'skilled_workforce'],
      fmcg: ['distribution_network', 'rural_penetration', 'brand_competition']
    };
    
    const potentialBottlenecks = sectorBottlenecks[sectorKey] || [];
    potentialBottlenecks.forEach(bottleneck => {
      bottlenecks.push({
        type: 'sector_specific',
        description: bottleneck.replace(/_/g, ' '),
        impact: 'medium',
        recommendation: `Address ${bottleneck.replace(/_/g, ' ')} challenges`
      });
    });
    
    return bottlenecks;
  }

  generateExpansionRecommendations(sectorKey, utilization) {
    const recommendations = [];
    
    if (utilization > 80) {
      recommendations.push({
        type: 'capacity_expansion',
        priority: 'high',
        timeframe: '12-18 months',
        description: 'Add new production lines or facilities',
        estimatedCost: this.estimateExpansionCost(sectorKey),
        expectedROI: '15-25%'
      });
    }
    
    if (utilization > 70) {
      recommendations.push({
        type: 'efficiency_improvement',
        priority: 'medium',
        timeframe: '6-12 months',
        description: 'Optimize existing processes and equipment',
        estimatedCost: this.estimateEfficiencyImprovementCost(sectorKey),
        expectedROI: '20-30%'
      });
    }
    
    return recommendations;
  }

  estimateExpansionCost(sectorKey) {
    const expansionCosts = {
      automotive: '₹500-1000 Cr',
      pharmaceuticals: '₹200-500 Cr',
      textiles: '₹100-300 Cr',
      chemicals: '₹300-800 Cr',
      steel: '₹1000-2000 Cr',
      electronics: '₹150-400 Cr',
      fmcg: '₹100-250 Cr'
    };
    
    return expansionCosts[sectorKey] || '₹200-500 Cr';
  }

  estimateEfficiencyImprovementCost(sectorKey) {
    const improvementCosts = {
      automotive: '₹50-150 Cr',
      pharmaceuticals: '₹25-75 Cr',
      textiles: '₹15-50 Cr',
      chemicals: '₹40-120 Cr',
      steel: '₹100-300 Cr',
      electronics: '₹20-60 Cr',
      fmcg: '₹15-40 Cr'
    };
    
    return improvementCosts[sectorKey] || '₹25-75 Cr';
  }

  async assessSupplyChainHealth() {
    try {
      const healthAssessments = [];
      
      for (const [sectorKey, healthData] of this.supplyChainHealth) {
        const assessment = this.performSupplyChainHealthAssessment(sectorKey, healthData);
        
        Object.assign(healthData, assessment, {
          lastUpdated: new Date().toISOString()
        });
        
        healthAssessments.push({
          sector: sectorKey,
          assessment,
          timestamp: healthData.lastUpdated
        });
      }
      
      if (healthAssessments.length > 0) {
        this.stats.healthAssessments += healthAssessments.length;
        
        await this.saveSupplyChainHealthData();
        
        this.emit('supplyChainHealthUpdate', {
          assessments: healthAssessments,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Supply chain health assessment failed:', error);
    }
  }

  performSupplyChainHealthAssessment(sectorKey, healthData) {
    const inputAvailability = Math.round(75 + Math.random() * 20);
    const supplierReliability = Math.round(70 + Math.random() * 25);
    const logisticsEfficiency = Math.round(65 + Math.random() * 30);
    const inventoryLevels = Math.round(60 + Math.random() * 35);
    
    const overallHealth = Math.round(
      (inputAvailability * 0.3) +
      (supplierReliability * 0.3) +
      (logisticsEfficiency * 0.25) +
      (inventoryLevels * 0.15)
    );
    
    const riskFactors = this.identifySupplyChainRisks(sectorKey, {
      inputAvailability,
      supplierReliability,
      logisticsEfficiency,
      inventoryLevels
    });
    
    return {
      overallHealth,
      inputAvailability,
      supplierReliability,
      logisticsEfficiency,
      inventoryLevels,
      riskFactors,
      recommendations: this.generateSupplyChainRecommendations(sectorKey, {
        inputAvailability,
        supplierReliability,
        logisticsEfficiency,
        inventoryLevels
      })
    };
  }

  identifySupplyChainRisks(sectorKey, metrics) {
    const risks = [];
    
    if (metrics.inputAvailability < 70) {
      risks.push({
        type: 'input_shortage',
        severity: 'high',
        description: 'Critical input materials facing availability constraints'
      });
    }
    
    if (metrics.supplierReliability < 75) {
      risks.push({
        type: 'supplier_reliability',
        severity: 'medium',
        description: 'Key suppliers showing reliability issues'
      });
    }
    
    if (metrics.logisticsEfficiency < 70) {
      risks.push({
        type: 'logistics_disruption',
        severity: 'medium',
        description: 'Transportation and logistics inefficiencies'
      });
    }
    
    return risks;
  }

  generateSupplyChainRecommendations(sectorKey, metrics) {
    const recommendations = [];
    
    if (metrics.inputAvailability < 80) {
      recommendations.push({
        category: 'Input Security',
        priority: 'high',
        action: 'Diversify supplier base and increase strategic inventory',
        timeline: '3-6 months'
      });
    }
    
    if (metrics.supplierReliability < 80) {
      recommendations.push({
        category: 'Supplier Management',
        priority: 'medium',
        action: 'Implement supplier development programs and performance monitoring',
        timeline: '6-12 months'
      });
    }
    
    return recommendations;
  }

  async saveProductionData() {
    try {
      const filePath = path.join(this.config.dataPath, 'production', 'production-metrics.json');
      const data = Object.fromEntries(this.productionMetrics);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save production data:', error);
    }
  }

  async saveCapacityData() {
    try {
      const filePath = path.join(this.config.dataPath, 'capacity', 'capacity-metrics.json');
      const data = Object.fromEntries(this.capacityMetrics);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save capacity data:', error);
    }
  }

  async saveSupplyChainHealthData() {
    try {
      const filePath = path.join(this.config.dataPath, 'health', 'supply-chain-health.json');
      const data = Object.fromEntries(this.supplyChainHealth);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save supply chain health data:', error);
    }
  }

  getStats() {
    return {
      ...this.stats,
      currentMetrics: {
        sectors: this.productionMetrics.size,
        capacityAnalyses: this.capacityMetrics.size,
        healthAssessments: this.supplyChainHealth.size
      }
    };
  }

  async generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: this.getStats(),
      productionMetrics: Object.fromEntries(this.productionMetrics),
      capacityMetrics: Object.fromEntries(this.capacityMetrics),
      supplyChainHealth: Object.fromEntries(this.supplyChainHealth)
    };
    
    const reportPath = path.join(this.config.dataPath, 'reports', `manufacturing-report-${Date.now()}.json`);
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    return report;
  }
}

module.exports = ManufacturingOutputMonitor;
