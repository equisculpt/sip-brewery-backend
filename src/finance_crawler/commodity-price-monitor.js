/**
 * 📦 COMMODITY PRICE MONITOR
 * 
 * Real-time commodity price tracking and volatility analysis
 * Monitors key commodities affecting Indian supply chains
 * 
 * @author Financial Intelligence Team
 * @version 1.0.0 - Commodity Intelligence
 */

const EventEmitter = require('events');
const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');

class CommodityPriceMonitor extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      // Monitoring settings
      enableRealTimeTracking: true,
      enableVolatilityAnalysis: true,
      enableImpactAssessment: true,
      enablePriceAlerts: true,
      
      // Data sources
      enableMCXData: true,
      enableNCDEXData: true,
      enableGlobalPrices: true,
      enableGovernmentData: true,
      
      // Update intervals
      priceUpdateInterval: 15 * 60 * 1000, // 15 minutes
      volatilityAnalysisInterval: 60 * 60 * 1000, // 1 hour
      impactAssessmentInterval: 4 * 60 * 60 * 1000, // 4 hours
      
      // Storage
      dataPath: './data/commodities',
      
      ...options
    };
    
    // Commodity definitions
    this.commodities = {
      energy: {
        crude_oil: {
          name: 'Crude Oil',
          unit: 'per barrel',
          basePrice: 6500,
          volatilityThreshold: 0.15,
          impactWeight: 1.5,
          exchanges: ['MCX', 'NYMEX'],
          industries: ['automotive', 'chemicals', 'transportation', 'aviation']
        },
        natural_gas: {
          name: 'Natural Gas',
          unit: 'per MMBtu',
          basePrice: 250,
          volatilityThreshold: 0.20,
          impactWeight: 1.2,
          exchanges: ['MCX', 'NYMEX'],
          industries: ['power', 'fertilizers', 'petrochemicals']
        },
        coal: {
          name: 'Coal',
          unit: 'per tonne',
          basePrice: 3200,
          volatilityThreshold: 0.10,
          impactWeight: 1.4,
          exchanges: ['MCX'],
          industries: ['power', 'steel', 'cement']
        }
      },
      
      metals: {
        steel: {
          name: 'Steel',
          unit: 'per tonne',
          basePrice: 55000,
          volatilityThreshold: 0.12,
          impactWeight: 1.3,
          exchanges: ['MCX', 'LME'],
          industries: ['automotive', 'construction', 'infrastructure']
        },
        aluminum: {
          name: 'Aluminum',
          unit: 'per kg',
          basePrice: 180,
          volatilityThreshold: 0.18,
          impactWeight: 1.2,
          exchanges: ['MCX', 'LME'],
          industries: ['automotive', 'aerospace', 'packaging']
        },
        copper: {
          name: 'Copper',
          unit: 'per kg',
          basePrice: 720,
          volatilityThreshold: 0.20,
          impactWeight: 1.1,
          exchanges: ['MCX', 'LME'],
          industries: ['electronics', 'construction', 'automotive']
        },
        zinc: {
          name: 'Zinc',
          unit: 'per kg',
          basePrice: 220,
          volatilityThreshold: 0.16,
          impactWeight: 1.0,
          exchanges: ['MCX', 'LME'],
          industries: ['galvanizing', 'automotive', 'construction']
        }
      },
      
      agriculture: {
        cotton: {
          name: 'Cotton',
          unit: 'per quintal',
          basePrice: 52000,
          volatilityThreshold: 0.25,
          impactWeight: 1.0,
          exchanges: ['MCX', 'NCDEX'],
          industries: ['textiles', 'apparel']
        },
        wheat: {
          name: 'Wheat',
          unit: 'per quintal',
          basePrice: 2200,
          volatilityThreshold: 0.15,
          impactWeight: 0.9,
          exchanges: ['NCDEX'],
          industries: ['food_processing', 'fmcg']
        },
        rice: {
          name: 'Rice',
          unit: 'per quintal',
          basePrice: 3500,
          volatilityThreshold: 0.12,
          impactWeight: 0.8,
          exchanges: ['NCDEX'],
          industries: ['food_processing', 'fmcg']
        },
        sugar: {
          name: 'Sugar',
          unit: 'per quintal',
          basePrice: 3800,
          volatilityThreshold: 0.18,
          impactWeight: 0.9,
          exchanges: ['NCDEX'],
          industries: ['food_processing', 'beverages']
        }
      },
      
      chemicals: {
        polyethylene: {
          name: 'Polyethylene',
          unit: 'per kg',
          basePrice: 95,
          volatilityThreshold: 0.14,
          impactWeight: 1.1,
          exchanges: ['MCX'],
          industries: ['packaging', 'petrochemicals', 'plastics']
        },
        benzene: {
          name: 'Benzene',
          unit: 'per litre',
          basePrice: 85,
          volatilityThreshold: 0.16,
          impactWeight: 1.0,
          exchanges: ['MCX'],
          industries: ['chemicals', 'pharmaceuticals', 'plastics']
        }
      }
    };
    
    // Price data storage
    this.priceData = new Map();
    this.volatilityData = new Map();
    this.impactAnalysis = new Map();
    this.priceAlerts = new Map();
    
    // Statistics
    this.stats = {
      totalCommodities: 0,
      priceUpdates: 0,
      volatilityAnalyses: 0,
      impactAssessments: 0,
      alertsGenerated: 0,
      lastUpdate: null
    };
  }

  async initialize() {
    try {
      logger.info('📦 Initializing Commodity Price Monitor...');
      
      // Create directories
      await this.createDirectories();
      
      // Load existing data
      await this.loadExistingData();
      
      // Initialize commodity tracking
      await this.initializeCommodityTracking();
      
      // Start monitoring
      this.startPriceMonitoring();
      
      // Count total commodities
      this.stats.totalCommodities = this.getTotalCommodityCount();
      
      logger.info(`✅ Commodity Price Monitor initialized - tracking ${this.stats.totalCommodities} commodities`);
      
    } catch (error) {
      logger.error('❌ Commodity Price Monitor initialization failed:', error);
      throw error;
    }
  }

  async createDirectories() {
    const dirs = [
      this.config.dataPath,
      path.join(this.config.dataPath, 'prices'),
      path.join(this.config.dataPath, 'volatility'),
      path.join(this.config.dataPath, 'impact-analysis'),
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
      // Load existing price data
      const priceFile = path.join(this.config.dataPath, 'prices', 'commodity-prices.json');
      const priceData = await fs.readFile(priceFile, 'utf8');
      const prices = JSON.parse(priceData);
      
      for (const [commodity, data] of Object.entries(prices)) {
        this.priceData.set(commodity, data);
      }
      
      logger.info(`📚 Loaded existing price data for ${this.priceData.size} commodities`);
      
    } catch (error) {
      logger.debug('No existing commodity price data found, starting fresh');
    }
  }

  async initializeCommodityTracking() {
    // Initialize tracking for all commodities
    for (const [category, commodities] of Object.entries(this.commodities)) {
      for (const [commodityKey, commodity] of Object.entries(commodities)) {
        if (!this.priceData.has(commodityKey)) {
          this.priceData.set(commodityKey, {
            ...commodity,
            currentPrice: commodity.basePrice,
            priceHistory: [],
            lastUpdated: new Date().toISOString()
          });
        }
        
        // Initialize volatility tracking
        this.volatilityData.set(commodityKey, {
          dailyVolatility: 0,
          weeklyVolatility: 0,
          monthlyVolatility: 0,
          volatilityTrend: 'stable',
          lastCalculated: new Date().toISOString()
        });
        
        // Initialize impact analysis
        this.impactAnalysis.set(commodityKey, {
          supplyChainImpact: 'low',
          industryImpact: {},
          priceShockRisk: 'low',
          lastAssessed: new Date().toISOString()
        });
      }
    }
  }

  startPriceMonitoring() {
    logger.info('📊 Starting commodity price monitoring...');
    
    // Start price updates
    this.updateCommodityPrices();
    setInterval(() => {
      this.updateCommodityPrices();
    }, this.config.priceUpdateInterval);
    
    // Start volatility analysis
    this.analyzeVolatility();
    setInterval(() => {
      this.analyzeVolatility();
    }, this.config.volatilityAnalysisInterval);
    
    // Start impact assessment
    this.assessImpact();
    setInterval(() => {
      this.assessImpact();
    }, this.config.impactAssessmentInterval);
  }

  async updateCommodityPrices() {
    try {
      logger.debug('📊 Updating commodity prices...');
      
      const priceUpdates = [];
      
      for (const [commodityKey, commodityData] of this.priceData) {
        try {
          // Simulate price data collection (in production, this would fetch real data)
          const newPrice = await this.fetchCommodityPrice(commodityKey, commodityData);
          
          if (newPrice && newPrice !== commodityData.currentPrice) {
            const priceChange = ((newPrice - commodityData.currentPrice) / commodityData.currentPrice) * 100;
            
            // Update price data
            commodityData.priceHistory.push({
              price: commodityData.currentPrice,
              timestamp: commodityData.lastUpdated
            });
            
            // Keep only last 100 price points
            if (commodityData.priceHistory.length > 100) {
              commodityData.priceHistory = commodityData.priceHistory.slice(-100);
            }
            
            commodityData.currentPrice = newPrice;
            commodityData.priceChange = priceChange;
            commodityData.lastUpdated = new Date().toISOString();
            
            priceUpdates.push({
              commodity: commodityKey,
              oldPrice: commodityData.priceHistory[commodityData.priceHistory.length - 1]?.price,
              newPrice: newPrice,
              priceChange: priceChange,
              timestamp: commodityData.lastUpdated
            });
            
            // Check for price alerts
            await this.checkPriceAlerts(commodityKey, commodityData, priceChange);
          }
          
        } catch (error) {
          logger.error(`❌ Failed to update price for ${commodityKey}:`, error);
        }
      }
      
      if (priceUpdates.length > 0) {
        this.stats.priceUpdates += priceUpdates.length;
        this.stats.lastUpdate = new Date().toISOString();
        
        // Save updated prices
        await this.savePriceData();
        
        // Emit price update event
        this.emit('priceUpdate', {
          updates: priceUpdates,
          totalUpdates: priceUpdates.length,
          timestamp: new Date().toISOString()
        });
      }
      
    } catch (error) {
      logger.error('❌ Commodity price update failed:', error);
    }
  }

  async fetchCommodityPrice(commodityKey, commodityData) {
    // Simulate price fetching with realistic volatility
    const volatility = commodityData.volatilityThreshold || 0.15;
    const priceChange = (Math.random() - 0.5) * 2 * volatility;
    const newPrice = commodityData.currentPrice * (1 + priceChange);
    
    // Add some market-specific factors
    let marketFactor = 1;
    
    // Simulate market conditions
    const marketCondition = Math.random();
    if (marketCondition > 0.9) {
      marketFactor = 1.05; // Bull market
    } else if (marketCondition < 0.1) {
      marketFactor = 0.95; // Bear market
    }
    
    return Math.round(newPrice * marketFactor);
  }

  async checkPriceAlerts(commodityKey, commodityData, priceChange) {
    const alerts = [];
    
    // High volatility alert
    if (Math.abs(priceChange) > (commodityData.volatilityThreshold * 100 * 0.8)) {
      alerts.push({
        type: 'high_volatility',
        severity: Math.abs(priceChange) > (commodityData.volatilityThreshold * 100) ? 'high' : 'medium',
        message: `High price volatility detected in ${commodityData.name}`,
        priceChange: priceChange,
        threshold: commodityData.volatilityThreshold * 100
      });
    }
    
    // Price shock alert
    if (Math.abs(priceChange) > 10) {
      alerts.push({
        type: 'price_shock',
        severity: 'critical',
        message: `Price shock detected in ${commodityData.name}`,
        priceChange: priceChange,
        impact: 'Supply chain disruption likely'
      });
    }
    
    // Trend reversal alert
    if (commodityData.priceHistory.length >= 5) {
      const recentTrend = this.calculateRecentTrend(commodityData.priceHistory.slice(-5));
      const currentDirection = priceChange > 0 ? 'up' : 'down';
      
      if (recentTrend !== 'stable' && recentTrend !== currentDirection) {
        alerts.push({
          type: 'trend_reversal',
          severity: 'medium',
          message: `Trend reversal detected in ${commodityData.name}`,
          previousTrend: recentTrend,
          currentDirection: currentDirection
        });
      }
    }
    
    if (alerts.length > 0) {
      for (const alert of alerts) {
        const alertKey = `${commodityKey}_${alert.type}_${Date.now()}`;
        this.priceAlerts.set(alertKey, {
          commodity: commodityKey,
          commodityName: commodityData.name,
          ...alert,
          timestamp: new Date().toISOString()
        });
      }
      
      this.stats.alertsGenerated += alerts.length;
      
      // Emit alert event
      this.emit('priceAlert', {
        commodity: commodityKey,
        commodityName: commodityData.name,
        alerts: alerts,
        timestamp: new Date().toISOString()
      });
    }
  }

  calculateRecentTrend(priceHistory) {
    if (priceHistory.length < 3) return 'stable';
    
    let upCount = 0;
    let downCount = 0;
    
    for (let i = 1; i < priceHistory.length; i++) {
      if (priceHistory[i].price > priceHistory[i-1].price) {
        upCount++;
      } else if (priceHistory[i].price < priceHistory[i-1].price) {
        downCount++;
      }
    }
    
    if (upCount > downCount) return 'up';
    if (downCount > upCount) return 'down';
    return 'stable';
  }

  async analyzeVolatility() {
    try {
      logger.debug('📈 Analyzing commodity volatility...');
      
      for (const [commodityKey, commodityData] of this.priceData) {
        if (commodityData.priceHistory.length >= 10) {
          const volatilityMetrics = this.calculateVolatilityMetrics(commodityData.priceHistory);
          
          this.volatilityData.set(commodityKey, {
            ...volatilityMetrics,
            lastCalculated: new Date().toISOString()
          });
        }
      }
      
      this.stats.volatilityAnalyses++;
      
      // Emit volatility analysis event
      this.emit('volatilityAnalysis', {
        commoditiesAnalyzed: this.volatilityData.size,
        timestamp: new Date().toISOString()
      });
      
    } catch (error) {
      logger.error('❌ Volatility analysis failed:', error);
    }
  }

  calculateVolatilityMetrics(priceHistory) {
    // Calculate daily volatility (last 7 days)
    const dailyReturns = this.calculateReturns(priceHistory.slice(-7));
    const dailyVolatility = this.calculateStandardDeviation(dailyReturns);
    
    // Calculate weekly volatility (last 30 days)
    const weeklyReturns = this.calculateReturns(priceHistory.slice(-30));
    const weeklyVolatility = this.calculateStandardDeviation(weeklyReturns);
    
    // Calculate monthly volatility (all available data)
    const monthlyReturns = this.calculateReturns(priceHistory);
    const monthlyVolatility = this.calculateStandardDeviation(monthlyReturns);
    
    // Determine volatility trend
    const volatilityTrend = this.determineVolatilityTrend(dailyVolatility, weeklyVolatility, monthlyVolatility);
    
    return {
      dailyVolatility: Math.round(dailyVolatility * 10000) / 100, // As percentage
      weeklyVolatility: Math.round(weeklyVolatility * 10000) / 100,
      monthlyVolatility: Math.round(monthlyVolatility * 10000) / 100,
      volatilityTrend,
      volatilityLevel: this.getVolatilityLevel(dailyVolatility)
    };
  }

  calculateReturns(priceHistory) {
    const returns = [];
    
    for (let i = 1; i < priceHistory.length; i++) {
      const currentPrice = priceHistory[i].price;
      const previousPrice = priceHistory[i-1].price;
      const returnValue = (currentPrice - previousPrice) / previousPrice;
      returns.push(returnValue);
    }
    
    return returns;
  }

  calculateStandardDeviation(returns) {
    if (returns.length === 0) return 0;
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const squaredDifferences = returns.map(ret => Math.pow(ret - mean, 2));
    const variance = squaredDifferences.reduce((sum, diff) => sum + diff, 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  determineVolatilityTrend(daily, weekly, monthly) {
    if (daily > weekly && weekly > monthly) return 'increasing';
    if (daily < weekly && weekly < monthly) return 'decreasing';
    return 'stable';
  }

  getVolatilityLevel(volatility) {
    if (volatility > 0.05) return 'high';
    if (volatility > 0.03) return 'medium';
    return 'low';
  }

  async assessImpact() {
    try {
      logger.debug('🎯 Assessing commodity impact...');
      
      for (const [commodityKey, commodityData] of this.priceData) {
        const impactAssessment = await this.calculateImpactAssessment(commodityKey, commodityData);
        
        this.impactAnalysis.set(commodityKey, {
          ...impactAssessment,
          lastAssessed: new Date().toISOString()
        });
      }
      
      this.stats.impactAssessments++;
      
      // Emit impact assessment event
      this.emit('impactAssessment', {
        commoditiesAssessed: this.impactAnalysis.size,
        timestamp: new Date().toISOString()
      });
      
    } catch (error) {
      logger.error('❌ Impact assessment failed:', error);
    }
  }

  async calculateImpactAssessment(commodityKey, commodityData) {
    const volatilityData = this.volatilityData.get(commodityKey);
    const priceChange = commodityData.priceChange || 0;
    
    // Calculate supply chain impact
    const supplyChainImpact = this.calculateSupplyChainImpact(commodityData, volatilityData, priceChange);
    
    // Calculate industry-specific impact
    const industryImpact = this.calculateIndustryImpact(commodityData, priceChange);
    
    // Calculate price shock risk
    const priceShockRisk = this.calculatePriceShockRisk(volatilityData, priceChange);
    
    return {
      supplyChainImpact,
      industryImpact,
      priceShockRisk,
      overallImpact: this.calculateOverallImpact(supplyChainImpact, priceShockRisk),
      riskFactors: this.identifyRiskFactors(commodityData, volatilityData, priceChange)
    };
  }

  calculateSupplyChainImpact(commodityData, volatilityData, priceChange) {
    const impactWeight = commodityData.impactWeight || 1.0;
    const volatilityLevel = volatilityData?.volatilityLevel || 'low';
    const absChange = Math.abs(priceChange);
    
    let impactScore = 0;
    
    // Price change impact
    if (absChange > 15) impactScore += 40;
    else if (absChange > 10) impactScore += 30;
    else if (absChange > 5) impactScore += 20;
    else if (absChange > 2) impactScore += 10;
    
    // Volatility impact
    if (volatilityLevel === 'high') impactScore += 30;
    else if (volatilityLevel === 'medium') impactScore += 15;
    
    // Commodity importance weight
    impactScore *= impactWeight;
    
    if (impactScore >= 80) return 'critical';
    if (impactScore >= 60) return 'high';
    if (impactScore >= 40) return 'medium';
    return 'low';
  }

  calculateIndustryImpact(commodityData, priceChange) {
    const industryImpact = {};
    
    if (commodityData.industries) {
      for (const industry of commodityData.industries) {
        let impact = 'low';
        
        const absChange = Math.abs(priceChange);
        if (absChange > 10) impact = 'high';
        else if (absChange > 5) impact = 'medium';
        
        industryImpact[industry] = {
          impact,
          priceChange,
          description: this.getIndustryImpactDescription(industry, priceChange)
        };
      }
    }
    
    return industryImpact;
  }

  getIndustryImpactDescription(industry, priceChange) {
    const direction = priceChange > 0 ? 'increase' : 'decrease';
    const magnitude = Math.abs(priceChange) > 10 ? 'significant' : Math.abs(priceChange) > 5 ? 'moderate' : 'minor';
    
    return `${magnitude} ${direction} likely to affect ${industry} sector costs and margins`;
  }

  calculatePriceShockRisk(volatilityData, priceChange) {
    const volatilityLevel = volatilityData?.volatilityLevel || 'low';
    const absChange = Math.abs(priceChange);
    
    if ((volatilityLevel === 'high' && absChange > 8) || absChange > 15) {
      return 'high';
    } else if ((volatilityLevel === 'medium' && absChange > 5) || absChange > 10) {
      return 'medium';
    }
    return 'low';
  }

  calculateOverallImpact(supplyChainImpact, priceShockRisk) {
    const impactLevels = { low: 1, medium: 2, high: 3, critical: 4 };
    
    const supplyScore = impactLevels[supplyChainImpact] || 1;
    const riskScore = impactLevels[priceShockRisk] || 1;
    
    const averageScore = (supplyScore + riskScore) / 2;
    
    if (averageScore >= 3.5) return 'critical';
    if (averageScore >= 2.5) return 'high';
    if (averageScore >= 1.5) return 'medium';
    return 'low';
  }

  identifyRiskFactors(commodityData, volatilityData, priceChange) {
    const riskFactors = [];
    
    if (Math.abs(priceChange) > 10) {
      riskFactors.push({
        factor: 'high_price_volatility',
        severity: 'high',
        description: `Price changed by ${priceChange.toFixed(2)}% in recent period`
      });
    }
    
    if (volatilityData?.volatilityLevel === 'high') {
      riskFactors.push({
        factor: 'elevated_volatility',
        severity: 'medium',
        description: 'Commodity showing elevated volatility patterns'
      });
    }
    
    if (volatilityData?.volatilityTrend === 'increasing') {
      riskFactors.push({
        factor: 'increasing_volatility_trend',
        severity: 'medium',
        description: 'Volatility trend is increasing over time'
      });
    }
    
    return riskFactors;
  }

  async savePriceData() {
    try {
      const priceFile = path.join(this.config.dataPath, 'prices', 'commodity-prices.json');
      const priceData = Object.fromEntries(this.priceData);
      await fs.writeFile(priceFile, JSON.stringify(priceData, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save price data:', error);
    }
  }

  getTotalCommodityCount() {
    let count = 0;
    for (const category of Object.values(this.commodities)) {
      count += Object.keys(category).length;
    }
    return count;
  }

  // Public API methods
  getCommodityPrice(commodityKey) {
    return this.priceData.get(commodityKey);
  }

  getCommodityVolatility(commodityKey) {
    return this.volatilityData.get(commodityKey);
  }

  getCommodityImpact(commodityKey) {
    return this.impactAnalysis.get(commodityKey);
  }

  getAllCommodityData() {
    return {
      prices: Object.fromEntries(this.priceData),
      volatility: Object.fromEntries(this.volatilityData),
      impact: Object.fromEntries(this.impactAnalysis),
      alerts: Object.fromEntries(this.priceAlerts),
      stats: this.getSystemStats()
    };
  }

  getSystemStats() {
    return {
      ...this.stats,
      activeCommodities: this.priceData.size,
      activeAlerts: this.priceAlerts.size,
      uptime: Date.now() - (this.stats.startTime || Date.now())
    };
  }

  getHighImpactCommodities() {
    const highImpact = [];
    
    for (const [commodityKey, impact] of this.impactAnalysis) {
      if (impact.overallImpact === 'high' || impact.overallImpact === 'critical') {
        const commodityData = this.priceData.get(commodityKey);
        highImpact.push({
          commodity: commodityKey,
          name: commodityData?.name,
          impact: impact.overallImpact,
          supplyChainImpact: impact.supplyChainImpact,
          priceShockRisk: impact.priceShockRisk,
          currentPrice: commodityData?.currentPrice,
          priceChange: commodityData?.priceChange
        });
      }
    }
    
    return highImpact.sort((a, b) => {
      const impactOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return impactOrder[b.impact] - impactOrder[a.impact];
    });
  }

  getRecentAlerts(hours = 24) {
    const cutoffTime = Date.now() - (hours * 60 * 60 * 1000);
    const recentAlerts = [];
    
    for (const [alertKey, alert] of this.priceAlerts) {
      const alertTime = new Date(alert.timestamp).getTime();
      if (alertTime > cutoffTime) {
        recentAlerts.push(alert);
      }
    }
    
    return recentAlerts.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }
}

module.exports = { CommodityPriceMonitor };
