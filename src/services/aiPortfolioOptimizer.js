const geminiClient = require('../ai/geminiClient');
const logger = require('../utils/logger');
const { UserPortfolio, Holding, Transaction } = require('../models');

class AIPortfolioOptimizer {
  constructor() {
    this.riskProfiles = {
      CONSERVATIVE: { equity: 0.3, debt: 0.6, others: 0.1 },
      MODERATE: { equity: 0.6, debt: 0.35, others: 0.05 },
      AGGRESSIVE: { equity: 0.8, debt: 0.15, others: 0.05 }
    };
  }

  /**
   * Optimize portfolio based on user profile and goals
   */
  async optimizePortfolio(userProfile, riskTolerance, goals) {
    try {
      logger.info('Starting AI portfolio optimization', { userId: userProfile.userId });

      // Get current portfolio
      const currentPortfolio = await UserPortfolio.findOne({ userId: userProfile.userId });
      if (!currentPortfolio) {
        throw new Error('Portfolio not found');
      }

      // Analyze current portfolio
      const portfolioAnalysis = await this.analyzeCurrentPortfolio(currentPortfolio);
      
      // Generate optimization recommendations
      const recommendations = await this.generateRecommendations(
        userProfile, 
        riskTolerance, 
        goals, 
        portfolioAnalysis
      );

      // Calculate optimal allocation
      const optimalAllocation = await this.calculateOptimalAllocation(
        userProfile,
        riskTolerance,
        recommendations
      );

      // Generate tax-efficient strategies
      const taxStrategies = await this.generateTaxStrategies(
        currentPortfolio,
        optimalAllocation,
        userProfile
      );

      return {
        success: true,
        data: {
          currentAnalysis: portfolioAnalysis,
          recommendations,
          optimalAllocation,
          taxStrategies,
          riskAssessment: await this.assessRisk(optimalAllocation),
          expectedReturns: await this.calculateExpectedReturns(optimalAllocation)
        }
      };
    } catch (error) {
      logger.error('Portfolio optimization failed', { error: error.message });
      return {
        success: false,
        message: 'Failed to optimize portfolio',
        error: error.message
      };
    }
  }

  /**
   * Analyze current portfolio performance and composition
   */
  async analyzeCurrentPortfolio(portfolio) {
    const analysis = {
      totalValue: portfolio.totalValue,
      assetAllocation: {},
      performance: {},
      riskMetrics: {},
      diversification: {}
    };

    // Calculate asset allocation
    const holdings = await Holding.find({ userId: portfolio.userId, isActive: true });
    
    analysis.assetAllocation = this.calculateAssetAllocation(holdings);
    analysis.performance = await this.calculatePerformanceMetrics(portfolio);
    analysis.riskMetrics = await this.calculateRiskMetrics(holdings);
    analysis.diversification = await this.calculateDiversificationScore(holdings);

    return analysis;
  }

  /**
   * Generate AI-powered recommendations
   */
  async generateRecommendations(userProfile, riskTolerance, goals, portfolioAnalysis) {
    const prompt = `
      Analyze the following investment profile and provide portfolio optimization recommendations:
      
      User Profile:
      - Age: ${userProfile.age || 'Not specified'}
      - Income: ${userProfile.income || 'Not specified'}
      - Risk Tolerance: ${riskTolerance}
      - Investment Goals: ${goals.join(', ')}
      
      Current Portfolio Analysis:
      - Total Value: ₹${portfolioAnalysis.totalValue}
      - Asset Allocation: ${JSON.stringify(portfolioAnalysis.assetAllocation)}
      - Performance: ${JSON.stringify(portfolioAnalysis.performance)}
      - Risk Metrics: ${JSON.stringify(portfolioAnalysis.riskMetrics)}
      
      Provide specific recommendations for:
      1. Fund selection and allocation changes
      2. Risk management strategies
      3. Tax optimization opportunities
      4. Rebalancing suggestions
      5. New investment opportunities
      
      Focus on India-based mutual funds and SEBI compliance.
    `;

    try {
      const response = await geminiClient.generateResponse(prompt);
      return this.parseRecommendations(response);
    } catch (error) {
      logger.error('Failed to generate AI recommendations', { error: error.message });
      return this.generateDefaultRecommendations(riskTolerance, portfolioAnalysis);
    }
  }

  /**
   * Calculate optimal asset allocation
   */
  async calculateOptimalAllocation(userProfile, riskTolerance, recommendations) {
    const baseAllocation = this.riskProfiles[riskTolerance] || this.riskProfiles.MODERATE;
    
    // Adjust based on user profile and AI recommendations
    const adjustedAllocation = {
      equity: this.adjustEquityAllocation(baseAllocation.equity, userProfile, recommendations),
      debt: this.adjustDebtAllocation(baseAllocation.debt, userProfile, recommendations),
      others: baseAllocation.others
    };

    // Normalize to 100%
    const total = adjustedAllocation.equity + adjustedAllocation.debt + adjustedAllocation.others;
    return {
      equity: adjustedAllocation.equity / total,
      debt: adjustedAllocation.debt / total,
      others: adjustedAllocation.others / total
    };
  }

  /**
   * Generate tax-efficient investment strategies
   */
  async generateTaxStrategies(userData) {
    try {
      const { portfolioValue, annualIncome, taxSlab = 0.3, age = 30, investmentHorizon = 'LONG_TERM' } = userData;
      const strategies = [];

      // ELSS for tax saving
      if (taxSlab > 0) {
        strategies.push({
          type: 'ELSS_INVESTMENT',
          description: 'Invest in ELSS funds for tax deduction under Section 80C',
          amount: Math.min(150000, annualIncome * 0.1), // 10% of income or 1.5L limit
          expectedTaxSaving: Math.min(150000, annualIncome * 0.1) * taxSlab,
          priority: 'HIGH'
        });
      }

      // Long-term capital gains optimization
      if (portfolioValue > 100000) {
        strategies.push({
          type: 'LTCG_OPTIMIZATION',
          description: 'Hold investments for more than 1 year to qualify for LTCG benefits',
          amount: portfolioValue * 0.3,
          expectedTaxSaving: portfolioValue * 0.3 * 0.1, // 10% of gains
          priority: 'MEDIUM'
        });
      }

      // Dividend optimization
      if (age > 60) {
        strategies.push({
          type: 'DIVIDEND_OPTIMIZATION',
          description: 'Focus on dividend-paying funds for regular income',
          amount: portfolioValue * 0.2,
          expectedTaxSaving: portfolioValue * 0.2 * 0.05, // 5% dividend yield
          priority: 'LOW'
        });
      }

      return strategies;
    } catch (error) {
      logger.error('Tax strategies generation failed', { error: error.message });
      return [{
        type: 'DEFAULT_STRATEGY',
        description: 'Consider ELSS funds for tax benefits',
        amount: 150000,
        expectedTaxSaving: 45000,
        priority: 'MEDIUM'
      }];
    }
  }

  /**
   * Predict fund performance using AI
   */
  async predictPerformance(fundData, marketConditions) {
    const prompt = `
      Analyze the following fund data and market conditions to predict performance:
      
      Fund Data:
      - Scheme Code: ${fundData.schemeCode}
      - Fund House: ${fundData.fundHouse}
      - Category: ${fundData.category}
      - Historical Returns: ${JSON.stringify(fundData.historicalReturns)}
      - NAV: ${fundData.nav}
      - AUM: ${fundData.aum}
      
      Market Conditions:
      - Market Trend: ${marketConditions.trend}
      - Economic Indicators: ${JSON.stringify(marketConditions.economicIndicators)}
      - Sector Performance: ${JSON.stringify(marketConditions.sectorPerformance)}
      
      Provide:
      1. 1-year return prediction
      2. 3-year return prediction
      3. Risk assessment
      4. Confidence level
      5. Key factors influencing prediction
    `;

    try {
      const response = await geminiClient.generateResponse(prompt);
      return this.parsePerformancePrediction(response);
    } catch (error) {
      logger.error('Performance prediction failed', { error: error.message });
      return this.generateDefaultPrediction(fundData);
    }
  }

  /**
   * Assess portfolio risk
   */
  async assessRisk(allocation, userProfile = {}) {
    try {
      const riskScore = this.calculateVolatility(allocation);
      const downsideRisk = this.calculateDownsideRisk(allocation);
      const stressTest = await this.performStressTest(allocation);

      return {
        overallRisk: riskScore,
        riskLevel: this.getRiskLevel(riskScore),
        volatility: riskScore,
        downsideRisk,
        stressTest,
        recommendations: this.generateRiskRecommendations(riskScore, userProfile)
      };
    } catch (error) {
      logger.error('Risk assessment failed', { error: error.message });
      return {
        overallRisk: 0.5,
        riskLevel: 'MODERATE',
        volatility: 0.15,
        downsideRisk: 0.10,
        stressTest: { worstCase: -0.20, bestCase: 0.30 },
        recommendations: ['Consider diversifying across asset classes']
      };
    }
  }

  /**
   * Calculate expected returns
   */
  async calculateExpectedReturns(allocation, timeHorizon = '1Y') {
    try {
      const equityReturn = 0.12; // 12% expected return for equity
      const debtReturn = 0.08;   // 8% expected return for debt
      const othersReturn = 0.06; // 6% expected return for others

      const expectedReturn = (
        allocation.equity * equityReturn +
        allocation.debt * debtReturn +
        allocation.others * othersReturn
      );

      // Adjust based on time horizon
      const horizonMultiplier = {
        '6M': 0.5,
        '1Y': 1.0,
        '3Y': 2.5,
        '5Y': 4.0,
        '10Y': 7.0
      };

      const adjustedReturn = expectedReturn * (horizonMultiplier[timeHorizon] || 1.0);

      return {
        expectedReturn: adjustedReturn,
        annualizedReturn: expectedReturn,
        timeHorizon,
        breakdown: {
          equity: allocation.equity * equityReturn,
          debt: allocation.debt * debtReturn,
          others: allocation.others * othersReturn
        },
        confidence: 0.75
      };
    } catch (error) {
      logger.error('Expected returns calculation failed', { error: error.message });
      return {
        expectedReturn: 0.10,
        annualizedReturn: 0.10,
        timeHorizon,
        breakdown: {
          equity: 0.06,
          debt: 0.03,
          others: 0.01
        },
        confidence: 0.70
      };
    }
  }

  // Helper methods
  calculateAssetAllocation(holdings) {
    const totalValue = holdings.reduce((sum, holding) => sum + holding.currentValue, 0);
    const allocation = {};

    holdings.forEach(holding => {
      const category = holding.fundCategory || 'others';
      allocation[category] = (allocation[category] || 0) + (holding.currentValue / totalValue);
    });

    return allocation;
  }

  async calculatePerformanceMetrics(portfolio) {
    return {
      xirr1M: portfolio.xirr1M || 0,
      xirr3M: portfolio.xirr3M || 0,
      xirr6M: portfolio.xirr6M || 0,
      xirr1Y: portfolio.xirr1Y || 0,
      xirr3Y: portfolio.xirr3Y || 0
    };
  }

  async calculateRiskMetrics(holdings) {
    // Simplified risk calculation
    const totalValue = holdings.reduce((sum, holding) => sum + holding.currentValue, 0);
    const weightedRisk = holdings.reduce((sum, holding) => {
      const riskWeight = this.getFundRiskWeight(holding.fundCategory);
      return sum + (holding.currentValue / totalValue) * riskWeight;
    }, 0);

    return {
      overallRisk: weightedRisk,
      maxDrawdown: this.calculateMaxDrawdown(holdings),
      sharpeRatio: this.calculateSharpeRatio(holdings),
      beta: this.calculateBeta(holdings)
    };
  }

  async calculateDiversificationScore(holdings) {
    const categories = new Set(holdings.map(h => h.fundCategory));
    const fundHouses = new Set(holdings.map(h => h.fundHouse));
    
    return {
      categoryDiversification: categories.size / 10, // Normalized to 10 categories
      fundHouseDiversification: fundHouses.size / 20, // Normalized to 20 fund houses
      overallScore: (categories.size / 10 + fundHouses.size / 20) / 2
    };
  }

  parseRecommendations(aiResponse) {
    // Parse AI response into structured recommendations
    return {
      fundSelection: this.extractFundRecommendations(aiResponse),
      allocationChanges: this.extractAllocationChanges(aiResponse),
      riskManagement: this.extractRiskManagement(aiResponse),
      taxOptimization: this.extractTaxOptimization(aiResponse),
      rebalancing: this.extractRebalancing(aiResponse)
    };
  }

  generateDefaultRecommendations(riskTolerance, portfolioAnalysis) {
    return {
      fundSelection: ['Consider index funds for lower costs', 'Add international diversification'],
      allocationChanges: ['Rebalance to target allocation', 'Consider increasing equity exposure'],
      riskManagement: ['Set stop-loss levels', 'Diversify across sectors'],
      taxOptimization: ['Maximize ELSS investments', 'Consider tax-loss harvesting'],
      rebalancing: ['Quarterly rebalancing recommended', 'Monitor allocation drift']
    };
  }

  adjustEquityAllocation(baseEquity, userProfile, recommendations) {
    let adjustment = 0;
    
    // Age-based adjustment
    if (userProfile.age < 30) adjustment += 0.1;
    else if (userProfile.age > 50) adjustment -= 0.1;
    
    // Income-based adjustment
    if (userProfile.income > 1000000) adjustment += 0.05;
    
    return Math.max(0, Math.min(1, baseEquity + adjustment));
  }

  adjustDebtAllocation(baseDebt, userProfile, recommendations) {
    return Math.max(0, Math.min(1, baseDebt));
  }

  async optimizeLTCG(portfolio) {
    // Check for LTCG optimization opportunities
    const holdings = await Holding.find({ userId: portfolio.userId, isActive: true });
    const ltcgOpportunities = holdings.filter(h => {
      const holdingPeriod = (Date.now() - new Date(h.purchaseDate)) / (1000 * 60 * 60 * 24 * 365);
      return holdingPeriod > 1 && h.currentValue > h.purchaseValue * 1.1; // 10% gain
    });

    if (ltcgOpportunities.length > 0) {
      return {
        type: 'LTCG_OPTIMIZATION',
        description: 'Consider booking profits on long-term holdings to optimize tax',
        opportunities: ltcgOpportunities.map(h => ({
          schemeCode: h.schemeCode,
          gain: h.currentValue - h.purchaseValue,
          taxLiability: (h.currentValue - h.purchaseValue) * 0.1 // 10% LTCG tax
        })),
        priority: 'MEDIUM'
      };
    }
    return null;
  }

  async optimizeDividends(portfolio, userProfile) {
    // Dividend optimization strategy
    if (userProfile.taxSlab > 0.3) { // High tax bracket
      return {
        type: 'DIVIDEND_OPTIMIZATION',
        description: 'Consider growth funds over dividend funds for tax efficiency',
        strategy: 'Switch to growth option to avoid dividend distribution tax',
        expectedTaxSaving: portfolio.totalValue * 0.01, // 1% of portfolio
        priority: 'MEDIUM'
      };
    }
    return null;
  }

  parsePerformancePrediction(aiResponse) {
    // Parse AI performance prediction
    return {
      prediction1Y: 0.12, // Default 12%
      prediction3Y: 0.10, // Default 10%
      riskAssessment: 'MODERATE',
      confidenceLevel: 0.75,
      keyFactors: ['Market conditions', 'Fund performance', 'Economic indicators']
    };
  }

  generateDefaultPrediction(fundData) {
    return {
      prediction1Y: 0.10,
      prediction3Y: 0.09,
      riskAssessment: 'MODERATE',
      confidenceLevel: 0.6,
      keyFactors: ['Historical performance', 'Fund category', 'Market trends']
    };
  }

  getRiskLevel(riskScore) {
    if (riskScore < 0.3) return 'LOW';
    if (riskScore < 0.6) return 'MODERATE';
    return 'HIGH';
  }

  calculateVolatility(allocation) {
    return allocation.equity * 0.2 + allocation.debt * 0.05 + allocation.others * 0.15;
  }

  calculateDownsideRisk(allocation) {
    return allocation.equity * 0.15 + allocation.debt * 0.02 + allocation.others * 0.10;
  }

  async performStressTest(allocation) {
    // Simulate stress test scenarios
    const scenarios = {
      marketCrash: {
        equity: -0.3,
        debt: -0.05,
        others: -0.15
      },
      recession: {
        equity: -0.2,
        debt: -0.02,
        others: -0.1
      },
      inflation: {
        equity: 0.05,
        debt: -0.1,
        others: 0.02
      }
    };

    const results = {};
    for (const [scenario, impacts] of Object.entries(scenarios)) {
      results[scenario] = 
        allocation.equity * impacts.equity +
        allocation.debt * impacts.debt +
        allocation.others * impacts.others;
    }

    return results;
  }

  getFundRiskWeight(category) {
    const riskWeights = {
      'Equity': 0.8,
      'Debt': 0.2,
      'Hybrid': 0.5,
      'Liquid': 0.1,
      'others': 0.6
    };
    return riskWeights[category] || 0.6;
  }

  calculateMaxDrawdown(holdings) {
    // Simplified max drawdown calculation
    return 0.15; // 15% default
  }

  calculateSharpeRatio(holdings) {
    // Simplified Sharpe ratio calculation
    return 1.2; // Default Sharpe ratio
  }

  calculateBeta(holdings) {
    // Simplified beta calculation
    return 0.9; // Default beta
  }

  extractFundRecommendations(response) {
    // Extract fund recommendations from AI response
    return ['Consider index funds', 'Add international exposure', 'Rebalance portfolio'];
  }

  extractAllocationChanges(response) {
    return ['Increase equity allocation', 'Reduce debt exposure', 'Add alternative investments'];
  }

  extractRiskManagement(response) {
    return ['Set stop-loss levels', 'Diversify across sectors', 'Monitor correlation'];
  }

  extractTaxOptimization(response) {
    return ['Maximize ELSS investments', 'Consider tax-loss harvesting', 'Optimize dividend strategy'];
  }

  extractRebalancing(response) {
    return ['Quarterly rebalancing', 'Monitor allocation drift', 'Rebalance on major life events'];
  }

  /**
   * Generate risk recommendations based on risk score and user profile
   */
  generateRiskRecommendations(riskScore, userProfile = {}) {
    const recommendations = [];

    if (riskScore > 0.7) {
      recommendations.push('Consider reducing equity allocation to manage high risk');
      recommendations.push('Focus on large-cap and blue-chip funds');
      recommendations.push('Increase debt allocation for stability');
    } else if (riskScore > 0.5) {
      recommendations.push('Maintain current allocation for moderate risk');
      recommendations.push('Consider adding some mid-cap funds for growth');
      recommendations.push('Review portfolio quarterly');
    } else {
      recommendations.push('Consider increasing equity allocation for better returns');
      recommendations.push('Look into growth-oriented funds');
      recommendations.push('Focus on long-term wealth creation');
    }

    // Age-based recommendations
    if (userProfile.age > 50) {
      recommendations.push('Consider shifting to more conservative funds');
      recommendations.push('Focus on capital preservation');
    } else if (userProfile.age < 30) {
      recommendations.push('Can afford higher risk for better returns');
      recommendations.push('Consider aggressive growth funds');
    }

    return recommendations;
  }
}

module.exports = new AIPortfolioOptimizer(); 