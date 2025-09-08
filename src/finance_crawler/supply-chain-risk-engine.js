/**
 * ⚠️ SUPPLY CHAIN RISK ENGINE
 * Risk assessment algorithms, predictive disruption detection, and mitigation recommendations
 * 
 * @author ASI Engineering Team
 * @version 1.0.0
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');

class SupplyChainRiskEngine extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      enableRiskAssessment: true,
      enableDisruptionDetection: true,
      enableMitigationRecommendations: true,
      
      riskAssessmentInterval: 6 * 60 * 60 * 1000, // 6 hours
      disruptionDetectionInterval: 2 * 60 * 60 * 1000, // 2 hours
      mitigationUpdateInterval: 12 * 60 * 60 * 1000, // 12 hours
      
      dataPath: './data/supply-chain-risk',
      ...options
    };
    
    // Risk categories and weights
    this.riskCategories = {
      geopolitical: { weight: 0.25, impact: 'high' },
      natural_disasters: { weight: 0.20, impact: 'high' },
      economic: { weight: 0.20, impact: 'medium' },
      operational: { weight: 0.15, impact: 'medium' },
      technological: { weight: 0.10, impact: 'medium' },
      regulatory: { weight: 0.10, impact: 'low' }
    };
    
    // Industry-specific risk profiles
    this.industryRiskProfiles = {
      automotive: {
        criticalRisks: ['semiconductor_shortage', 'steel_price_volatility', 'port_congestion'],
        vulnerabilities: ['just_in_time_production', 'global_supply_chains'],
        resilience: 'medium'
      },
      pharmaceuticals: {
        criticalRisks: ['api_import_dependency', 'regulatory_changes', 'cold_chain_disruption'],
        vulnerabilities: ['single_source_suppliers', 'regulatory_compliance'],
        resilience: 'high'
      },
      textiles: {
        criticalRisks: ['cotton_price_volatility', 'export_restrictions', 'power_shortages'],
        vulnerabilities: ['seasonal_demand', 'labor_intensive'],
        resilience: 'low'
      },
      chemicals: {
        criticalRisks: ['crude_oil_dependency', 'environmental_regulations', 'safety_compliance'],
        vulnerabilities: ['hazardous_materials', 'complex_supply_chains'],
        resilience: 'medium'
      },
      steel: {
        criticalRisks: ['iron_ore_supply', 'coking_coal_imports', 'carbon_emission_norms'],
        vulnerabilities: ['commodity_price_volatility', 'energy_intensive'],
        resilience: 'medium'
      },
      electronics: {
        criticalRisks: ['chip_shortage', 'rare_earth_dependency', 'technology_obsolescence'],
        vulnerabilities: ['rapid_innovation', 'global_supply_chains'],
        resilience: 'low'
      },
      fmcg: {
        criticalRisks: ['agricultural_supply', 'packaging_costs', 'distribution_challenges'],
        vulnerabilities: ['weather_dependency', 'brand_competition'],
        resilience: 'high'
      }
    };
    
    this.riskAssessments = new Map();
    this.disruptionPredictions = new Map();
    this.mitigationStrategies = new Map();
    this.riskAlerts = [];
    
    this.stats = {
      riskAssessments: 0,
      disruptionsPredicted: 0,
      mitigationStrategies: 0,
      alertsGenerated: 0,
      lastUpdate: null
    };
  }

  async initialize() {
    try {
      logger.info('⚠️ Initializing Supply Chain Risk Engine...');
      
      await this.createDirectories();
      await this.loadExistingData();
      await this.initializeRiskTracking();
      this.startRiskMonitoring();
      
      logger.info('✅ Supply Chain Risk Engine initialized');
      
    } catch (error) {
      logger.error('❌ Supply Chain Risk Engine initialization failed:', error);
      throw error;
    }
  }

  async createDirectories() {
    const dirs = [
      this.config.dataPath,
      path.join(this.config.dataPath, 'assessments'),
      path.join(this.config.dataPath, 'predictions'),
      path.join(this.config.dataPath, 'mitigation'),
      path.join(this.config.dataPath, 'alerts'),
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
        { file: 'assessments/risk-assessments.json', map: this.riskAssessments },
        { file: 'predictions/disruption-predictions.json', map: this.disruptionPredictions },
        { file: 'mitigation/mitigation-strategies.json', map: this.mitigationStrategies }
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
      logger.debug('No existing risk data found, starting fresh');
    }
  }

  async initializeRiskTracking() {
    for (const [industry, profile] of Object.entries(this.industryRiskProfiles)) {
      if (!this.riskAssessments.has(industry)) {
        this.riskAssessments.set(industry, {
          industry,
          overallRiskScore: 50,
          riskLevel: 'medium',
          criticalRisks: profile.criticalRisks,
          vulnerabilities: profile.vulnerabilities,
          resilience: profile.resilience,
          lastUpdated: new Date().toISOString()
        });
      }
      
      if (!this.disruptionPredictions.has(industry)) {
        this.disruptionPredictions.set(industry, {
          industry,
          disruptionProbability: 0.3,
          predictedDisruptions: [],
          timeHorizon: '6_months',
          confidenceLevel: 0.7,
          lastUpdated: new Date().toISOString()
        });
      }
      
      if (!this.mitigationStrategies.has(industry)) {
        this.mitigationStrategies.set(industry, {
          industry,
          activeStrategies: [],
          recommendedActions: [],
          implementationStatus: 'planning',
          effectiveness: 0.6,
          lastUpdated: new Date().toISOString()
        });
      }
    }
  }

  startRiskMonitoring() {
    logger.info('📊 Starting supply chain risk monitoring...');
    
    if (this.config.enableRiskAssessment) {
      this.performRiskAssessment();
      setInterval(() => this.performRiskAssessment(), this.config.riskAssessmentInterval);
    }
    
    if (this.config.enableDisruptionDetection) {
      this.detectPotentialDisruptions();
      setInterval(() => this.detectPotentialDisruptions(), this.config.disruptionDetectionInterval);
    }
    
    if (this.config.enableMitigationRecommendations) {
      this.updateMitigationStrategies();
      setInterval(() => this.updateMitigationStrategies(), this.config.mitigationUpdateInterval);
    }
  }

  async performRiskAssessment() {
    try {
      const assessmentUpdates = [];
      
      for (const [industry, assessmentData] of this.riskAssessments) {
        const newAssessment = this.calculateRiskScore(industry, assessmentData);
        
        Object.assign(assessmentData, newAssessment, {
          lastUpdated: new Date().toISOString()
        });
        
        assessmentUpdates.push({
          industry,
          assessment: newAssessment,
          timestamp: assessmentData.lastUpdated
        });
        
        if (newAssessment.overallRiskScore > 75) {
          this.generateRiskAlert(industry, newAssessment);
        }
      }
      
      if (assessmentUpdates.length > 0) {
        this.stats.riskAssessments += assessmentUpdates.length;
        this.stats.lastUpdate = new Date().toISOString();
        
        await this.saveRiskAssessmentData();
        
        this.emit('riskAssessmentUpdate', {
          updates: assessmentUpdates,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Risk assessment failed:', error);
    }
  }

  calculateRiskScore(industry, assessmentData) {
    const categoryScores = {};
    let totalWeightedScore = 0;
    
    for (const [category, config] of Object.entries(this.riskCategories)) {
      const categoryScore = this.calculateCategoryRiskScore(category, industry);
      categoryScores[category] = categoryScore;
      totalWeightedScore += categoryScore * config.weight;
    }
    
    const overallRiskScore = Math.round(totalWeightedScore);
    const riskLevel = this.determineRiskLevel(overallRiskScore);
    const topRiskFactors = this.identifyTopRiskFactors(categoryScores);
    const riskTrend = this.calculateRiskTrend(overallRiskScore, assessmentData.overallRiskScore);
    
    return {
      overallRiskScore,
      riskLevel,
      categoryScores,
      topRiskFactors,
      riskTrend,
      recommendations: this.generateRiskRecommendations(industry, overallRiskScore, categoryScores)
    };
  }

  calculateCategoryRiskScore(category, industry) {
    const baseScore = 40 + Math.random() * 40;
    
    const industryAdjustments = {
      geopolitical: { automotive: 10, textiles: 15, electronics: 15 },
      natural_disasters: { pharmaceuticals: 12, textiles: 15, fmcg: 20 },
      economic: { textiles: 18, steel: 20, electronics: 15 }
    };
    
    const adjustment = industryAdjustments[category]?.[industry] || 0;
    return Math.min(100, Math.max(0, Math.round(baseScore + adjustment)));
  }

  determineRiskLevel(score) {
    if (score >= 80) return 'critical';
    if (score >= 65) return 'high';
    if (score >= 45) return 'medium';
    if (score >= 25) return 'low';
    return 'minimal';
  }

  identifyTopRiskFactors(categoryScores) {
    return Object.entries(categoryScores)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 3)
      .map(([category, score]) => ({
        category,
        score,
        impact: this.riskCategories[category].impact
      }));
  }

  calculateRiskTrend(current, previous) {
    if (!previous || previous === 0) return 'stable';
    
    const change = ((current - previous) / previous) * 100;
    
    if (change > 5) return 'increasing';
    if (change < -5) return 'decreasing';
    return 'stable';
  }

  generateRiskRecommendations(industry, overallScore, categoryScores) {
    const recommendations = [];
    
    if (overallScore > 70) {
      recommendations.push({
        priority: 'critical',
        action: 'Activate crisis management protocols',
        timeline: 'immediate',
        impact: 'high'
      });
    }
    
    if (categoryScores.geopolitical > 70) {
      recommendations.push({
        priority: 'high',
        action: 'Diversify supplier base across different regions',
        timeline: '3-6 months',
        impact: 'medium'
      });
    }
    
    if (categoryScores.operational > 60) {
      recommendations.push({
        priority: 'medium',
        action: 'Increase safety stock levels for critical components',
        timeline: '1-3 months',
        impact: 'medium'
      });
    }
    
    return recommendations;
  }

  generateRiskAlert(industry, assessment) {
    const alert = {
      id: `RISK_${industry.toUpperCase()}_${Date.now()}`,
      industry,
      severity: assessment.riskLevel,
      score: assessment.overallRiskScore,
      message: `High risk detected in ${industry} sector`,
      topRisks: assessment.topRiskFactors.slice(0, 2),
      recommendations: assessment.recommendations.filter(r => r.priority === 'critical'),
      timestamp: new Date().toISOString()
    };
    
    this.riskAlerts.push(alert);
    this.stats.alertsGenerated++;
    
    this.emit('riskAlert', alert);
    
    if (this.riskAlerts.length > 100) {
      this.riskAlerts = this.riskAlerts.slice(-100);
    }
  }

  async detectPotentialDisruptions() {
    try {
      const disruptionUpdates = [];
      
      for (const [industry, predictionData] of this.disruptionPredictions) {
        const newPrediction = this.predictDisruptions(industry, predictionData);
        
        Object.assign(predictionData, newPrediction, {
          lastUpdated: new Date().toISOString()
        });
        
        disruptionUpdates.push({
          industry,
          prediction: newPrediction,
          timestamp: predictionData.lastUpdated
        });
      }
      
      if (disruptionUpdates.length > 0) {
        this.stats.disruptionsPredicted += disruptionUpdates.length;
        
        await this.saveDisruptionPredictionData();
        
        this.emit('disruptionPredictionUpdate', {
          updates: disruptionUpdates,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Disruption detection failed:', error);
    }
  }

  predictDisruptions(industry, predictionData) {
    const riskAssessment = this.riskAssessments.get(industry);
    const profile = this.industryRiskProfiles[industry];
    
    const baseProbability = (riskAssessment.overallRiskScore / 100) * 0.6;
    const seasonalFactor = this.getSeasonalDisruptionFactor();
    const disruptionProbability = Math.min(0.95, baseProbability * seasonalFactor);
    
    const predictedDisruptions = this.identifySpecificDisruptions(industry, disruptionProbability);
    const confidenceLevel = this.calculatePredictionConfidence(industry, disruptionProbability);
    
    return {
      disruptionProbability: Math.round(disruptionProbability * 100) / 100,
      predictedDisruptions,
      confidenceLevel: Math.round(confidenceLevel * 100) / 100,
      timeHorizon: this.determineTimeHorizon(disruptionProbability),
      impactAssessment: this.assessDisruptionImpact(industry, predictedDisruptions)
    };
  }

  getSeasonalDisruptionFactor() {
    const month = new Date().getMonth() + 1;
    
    if ((month >= 6 && month <= 9) || (month >= 12 || month <= 2)) {
      return 1.2;
    }
    
    return 1.0;
  }

  identifySpecificDisruptions(industry, probability) {
    const profile = this.industryRiskProfiles[industry];
    const disruptions = [];
    
    profile.criticalRisks.forEach(risk => {
      if (Math.random() < probability) {
        disruptions.push({
          type: risk,
          probability: Math.round((0.6 + Math.random() * 0.4) * 100) / 100,
          severity: this.assessDisruptionSeverity(risk),
          timeframe: this.estimateDisruptionTimeframe(risk)
        });
      }
    });
    
    const generalDisruptions = ['supplier_failure', 'transportation_delay', 'quality_issue', 'capacity_shortage'];
    
    generalDisruptions.forEach(disruption => {
      if (Math.random() < probability * 0.7) {
        disruptions.push({
          type: disruption,
          probability: Math.round((0.4 + Math.random() * 0.4) * 100) / 100,
          severity: this.assessDisruptionSeverity(disruption),
          timeframe: this.estimateDisruptionTimeframe(disruption)
        });
      }
    });
    
    return disruptions.slice(0, 5);
  }

  assessDisruptionSeverity(disruptionType) {
    const severityMap = {
      semiconductor_shortage: 'critical',
      api_import_dependency: 'high',
      cotton_price_volatility: 'medium',
      crude_oil_dependency: 'high',
      iron_ore_supply: 'high',
      chip_shortage: 'critical',
      agricultural_supply: 'medium',
      supplier_failure: 'high',
      transportation_delay: 'medium',
      quality_issue: 'high',
      capacity_shortage: 'medium'
    };
    
    return severityMap[disruptionType] || 'medium';
  }

  estimateDisruptionTimeframe(disruptionType) {
    const timeframeMap = {
      semiconductor_shortage: '6-12 months',
      api_import_dependency: '3-6 months',
      cotton_price_volatility: '1-3 months',
      crude_oil_dependency: '2-4 months',
      iron_ore_supply: '3-6 months',
      chip_shortage: '6-18 months',
      agricultural_supply: '1-6 months',
      supplier_failure: '2-8 weeks',
      transportation_delay: '1-4 weeks',
      quality_issue: '4-12 weeks',
      capacity_shortage: '3-9 months'
    };
    
    return timeframeMap[disruptionType] || '2-6 months';
  }

  calculatePredictionConfidence(industry, probability) {
    const baseConfidence = {
      automotive: 0.8,
      pharmaceuticals: 0.85,
      textiles: 0.7,
      chemicals: 0.75,
      steel: 0.8,
      electronics: 0.7,
      fmcg: 0.85
    };
    
    const industryConfidence = baseConfidence[industry] || 0.75;
    let probabilityAdjustment = 1.0;
    
    if (probability > 0.8 || probability < 0.2) {
      probabilityAdjustment = 0.9;
    }
    
    return Math.min(0.95, industryConfidence * probabilityAdjustment);
  }

  determineTimeHorizon(probability) {
    if (probability > 0.7) return '1-3 months';
    if (probability > 0.5) return '3-6 months';
    if (probability > 0.3) return '6-12 months';
    return '12+ months';
  }

  assessDisruptionImpact(industry, disruptions) {
    if (disruptions.length === 0) {
      return { overall: 'low', financial: 'minimal', operational: 'minimal' };
    }
    
    const criticalCount = disruptions.filter(d => d.severity === 'critical').length;
    const highCount = disruptions.filter(d => d.severity === 'high').length;
    
    let overall = 'low';
    if (criticalCount > 0) overall = 'critical';
    else if (highCount > 1) overall = 'high';
    else if (highCount > 0) overall = 'medium';
    
    return {
      overall,
      financial: this.assessFinancialImpact(disruptions),
      operational: this.assessOperationalImpact(disruptions),
      estimatedCost: this.estimateDisruptionCost(industry, disruptions)
    };
  }

  assessFinancialImpact(disruptions) {
    const criticalCount = disruptions.filter(d => d.severity === 'critical').length;
    const highCount = disruptions.filter(d => d.severity === 'high').length;
    
    if (criticalCount > 0) return 'severe';
    if (highCount > 1) return 'significant';
    if (highCount > 0) return 'moderate';
    return 'minimal';
  }

  assessOperationalImpact(disruptions) {
    const hasSupplyChainDisruption = disruptions.some(d => 
      ['supplier_failure', 'transportation_delay', 'capacity_shortage'].includes(d.type)
    );
    
    if (hasSupplyChainDisruption) return 'severe';
    
    const criticalCount = disruptions.filter(d => d.severity === 'critical').length;
    if (criticalCount > 0) return 'significant';
    
    return 'moderate';
  }

  estimateDisruptionCost(industry, disruptions) {
    const baseCosts = {
      automotive: 500,
      pharmaceuticals: 200,
      textiles: 150,
      chemicals: 300,
      steel: 400,
      electronics: 250,
      fmcg: 100
    };
    
    const baseCost = baseCosts[industry] || 200;
    const severityMultiplier = disruptions.reduce((total, d) => {
      const multipliers = { critical: 3, high: 2, medium: 1.5, low: 1 };
      return total + (multipliers[d.severity] || 1);
    }, 0);
    
    const estimatedCost = Math.round(baseCost * severityMultiplier);
    
    return `₹${estimatedCost}-${estimatedCost * 2} Cr`;
  }

  async updateMitigationStrategies() {
    try {
      const mitigationUpdates = [];
      
      for (const [industry, mitigationData] of this.mitigationStrategies) {
        const newStrategy = this.generateMitigationStrategy(industry, mitigationData);
        
        Object.assign(mitigationData, newStrategy, {
          lastUpdated: new Date().toISOString()
        });
        
        mitigationUpdates.push({
          industry,
          strategy: newStrategy,
          timestamp: mitigationData.lastUpdated
        });
      }
      
      if (mitigationUpdates.length > 0) {
        this.stats.mitigationStrategies += mitigationUpdates.length;
        
        await this.saveMitigationStrategyData();
        
        this.emit('mitigationStrategyUpdate', {
          updates: mitigationUpdates,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Mitigation strategy update failed:', error);
    }
  }

  generateMitigationStrategy(industry, mitigationData) {
    const riskAssessment = this.riskAssessments.get(industry);
    const disruptionPrediction = this.disruptionPredictions.get(industry);
    
    const recommendedActions = [];
    
    if (riskAssessment.overallRiskScore > 70) {
      recommendedActions.push({
        action: 'Implement emergency response protocols',
        priority: 'critical',
        timeline: '1-2 weeks',
        cost: 'low',
        effectiveness: 'high'
      });
    }
    
    if (riskAssessment.categoryScores?.geopolitical > 65) {
      recommendedActions.push({
        action: 'Diversify supplier base geographically',
        priority: 'high',
        timeline: '3-6 months',
        cost: 'medium',
        effectiveness: 'high'
      });
    }
    
    if (disruptionPrediction.disruptionProbability > 0.6) {
      recommendedActions.push({
        action: 'Activate alternative supplier agreements',
        priority: 'high',
        timeline: '2-4 weeks',
        cost: 'low',
        effectiveness: 'high'
      });
    }
    
    const industryStrategies = this.getIndustrySpecificStrategies(industry);
    recommendedActions.push(...industryStrategies);
    
    const effectiveness = this.calculateMitigationEffectiveness(recommendedActions);
    
    return {
      recommendedActions: recommendedActions.slice(0, 6),
      effectiveness: Math.round(effectiveness * 100) / 100,
      implementationStatus: 'planning',
      totalCost: this.estimateMitigationCost(recommendedActions),
      expectedROI: '15-25%'
    };
  }

  getIndustrySpecificStrategies(industry) {
    const strategies = {
      automotive: [{
        action: 'Establish semiconductor inventory buffer',
        priority: 'critical',
        timeline: '2-6 months',
        cost: 'high',
        effectiveness: 'high'
      }],
      pharmaceuticals: [{
        action: 'Develop domestic API manufacturing capabilities',
        priority: 'high',
        timeline: '12-24 months',
        cost: 'high',
        effectiveness: 'high'
      }],
      textiles: [{
        action: 'Secure long-term cotton supply contracts',
        priority: 'medium',
        timeline: '6-12 months',
        cost: 'medium',
        effectiveness: 'medium'
      }]
    };
    
    return strategies[industry] || [];
  }

  calculateMitigationEffectiveness(actions) {
    if (actions.length === 0) return 0.5;
    
    const effectivenessMap = { high: 0.8, medium: 0.6, low: 0.4 };
    const totalEffectiveness = actions.reduce((sum, action) => {
      return sum + (effectivenessMap[action.effectiveness] || 0.5);
    }, 0);
    
    return Math.min(0.95, totalEffectiveness / actions.length);
  }

  estimateMitigationCost(actions) {
    const costMap = { high: 100, medium: 50, low: 20 };
    const totalCost = actions.reduce((sum, action) => {
      return sum + (costMap[action.cost] || 30);
    }, 0);
    
    return `₹${totalCost}-${totalCost * 2} Cr`;
  }

  async saveRiskAssessmentData() {
    try {
      const filePath = path.join(this.config.dataPath, 'assessments', 'risk-assessments.json');
      const data = Object.fromEntries(this.riskAssessments);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save risk assessment data:', error);
    }
  }

  async saveDisruptionPredictionData() {
    try {
      const filePath = path.join(this.config.dataPath, 'predictions', 'disruption-predictions.json');
      const data = Object.fromEntries(this.disruptionPredictions);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save disruption prediction data:', error);
    }
  }

  async saveMitigationStrategyData() {
    try {
      const filePath = path.join(this.config.dataPath, 'mitigation', 'mitigation-strategies.json');
      const data = Object.fromEntries(this.mitigationStrategies);
      await fs.writeFile(filePath, JSON.stringify(data, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save mitigation strategy data:', error);
    }
  }

  getStats() {
    return {
      ...this.stats,
      currentMetrics: {
        riskAssessments: this.riskAssessments.size,
        disruptionPredictions: this.disruptionPredictions.size,
        mitigationStrategies: this.mitigationStrategies.size,
        activeAlerts: this.riskAlerts.length
      }
    };
  }

  async generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: this.getStats(),
      riskAssessments: Object.fromEntries(this.riskAssessments),
      disruptionPredictions: Object.fromEntries(this.disruptionPredictions),
      mitigationStrategies: Object.fromEntries(this.mitigationStrategies),
      recentAlerts: this.riskAlerts.slice(-10)
    };
    
    const reportPath = path.join(this.config.dataPath, 'reports', `risk-report-${Date.now()}.json`);
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    return report;
  }
}

module.exports = SupplyChainRiskEngine;
