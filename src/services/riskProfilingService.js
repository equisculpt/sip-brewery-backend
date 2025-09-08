const logger = require('../utils/logger');
const portfolioAnalyticsService = require('./portfolioAnalyticsService');

class RiskProfilingService {
  constructor() {
    this.riskFactors = {
      age: { weight: 0.15, maxScore: 25 },
      income: { weight: 0.12, maxScore: 20 },
      investmentExperience: { weight: 0.10, maxScore: 15 },
      investmentHorizon: { weight: 0.20, maxScore: 25 },
      financialGoals: { weight: 0.18, maxScore: 25 },
      liquidityNeeds: { weight: 0.10, maxScore: 15 },
      debtLevel: { weight: 0.08, maxScore: 10 },
      marketKnowledge: { weight: 0.07, maxScore: 10 }
    };

    this.riskProfiles = {
      conservative: {
        score: { min: 0, max: 30 },
        description: 'Low risk tolerance, prefers stable returns',
        equityAllocation: { min: 0, max: 30 },
        debtAllocation: { min: 60, max: 100 },
        goldAllocation: { min: 0, max: 20 }
      },
      moderate: {
        score: { min: 31, max: 60 },
        description: 'Balanced approach, moderate risk tolerance',
        equityAllocation: { min: 30, max: 60 },
        debtAllocation: { min: 30, max: 60 },
        goldAllocation: { min: 10, max: 20 }
      },
      aggressive: {
        score: { min: 61, max: 100 },
        description: 'High risk tolerance, seeks maximum returns',
        equityAllocation: { min: 60, max: 100 },
        debtAllocation: { min: 0, max: 30 },
        goldAllocation: { min: 0, max: 15 }
      }
    };
  }

  /**
   * Get comprehensive risk profile for a user
   */
  async getComprehensiveRiskProfile({ userId, includePortfolioRisk = true, includeMarketRisk = true }) {
    try {
      logger.info(`Getting comprehensive risk profile for user: ${userId}`);

      const riskProfile = {
        personal: {},
        portfolio: {},
        market: {},
        recommendations: {},
        summary: {}
      };

      // Get personal risk assessment
      riskProfile.personal = await this.getPersonalRiskAssessment(userId);

      // Get portfolio risk assessment
      if (includePortfolioRisk) {
        riskProfile.portfolio = await this.getPortfolioRiskAssessment(userId);
      }

      // Get market risk assessment
      if (includeMarketRisk) {
        riskProfile.market = await this.getMarketRiskAssessment();
      }

      // Generate recommendations
      riskProfile.recommendations = this.generateRiskRecommendations(riskProfile);

      // Calculate overall summary
      riskProfile.summary = this.calculateOverallRiskSummary(riskProfile);

      return riskProfile;
    } catch (error) {
      logger.error('Error getting comprehensive risk profile:', error);
      throw error;
    }
  }

  /**
   * Get personal risk assessment
   */
  async getPersonalRiskAssessment(userId) {
    try {
      // This would typically fetch user data from database
      // For now, using mock data
      const userData = await this.getUserRiskData(userId);

      const assessment = {
        factors: {},
        scores: {},
        totalScore: 0,
        riskProfile: '',
        details: {}
      };

      // Calculate age factor
      assessment.factors.age = this.calculateAgeRiskScore(userData.age);
      assessment.scores.age = assessment.factors.age.score;

      // Calculate income factor
      assessment.factors.income = this.calculateIncomeRiskScore(userData.income);
      assessment.scores.income = assessment.factors.income.score;

      // Calculate investment experience factor
      assessment.factors.investmentExperience = this.calculateExperienceRiskScore(userData.investmentExperience);
      assessment.scores.investmentExperience = assessment.factors.investmentExperience.score;

      // Calculate investment horizon factor
      assessment.factors.investmentHorizon = this.calculateHorizonRiskScore(userData.investmentHorizon);
      assessment.scores.investmentHorizon = assessment.factors.investmentHorizon.score;

      // Calculate financial goals factor
      assessment.factors.financialGoals = this.calculateGoalsRiskScore(userData.financialGoals);
      assessment.scores.financialGoals = assessment.factors.financialGoals.score;

      // Calculate liquidity needs factor
      assessment.factors.liquidityNeeds = this.calculateLiquidityRiskScore(userData.liquidityNeeds);
      assessment.scores.liquidityNeeds = assessment.factors.liquidityNeeds.score;

      // Calculate debt level factor
      assessment.factors.debtLevel = this.calculateDebtRiskScore(userData.debtLevel);
      assessment.scores.debtLevel = assessment.factors.debtLevel.score;

      // Calculate market knowledge factor
      assessment.factors.marketKnowledge = this.calculateKnowledgeRiskScore(userData.marketKnowledge);
      assessment.scores.marketKnowledge = assessment.factors.marketKnowledge.score;

      // Calculate total weighted score
      assessment.totalScore = this.calculateTotalRiskScore(assessment.scores);

      // Determine risk profile
      assessment.riskProfile = this.determineRiskProfile(assessment.totalScore);

      // Get allocation recommendations
      assessment.details = this.getRiskProfileDetails(assessment.riskProfile);

      return assessment;
    } catch (error) {
      logger.error('Error getting personal risk assessment:', error);
      throw error;
    }
  }

  /**
   * Get portfolio risk assessment
   */
  async getPortfolioRiskAssessment(userId) {
    try {
      const portfolioData = await portfolioAnalyticsService.getPortfolioSummary(userId);
      
      const assessment = {
        currentAllocation: {},
        riskMetrics: {},
        concentrationRisk: {},
        diversificationScore: 0,
        recommendations: {}
      };

      // Calculate current allocation
      assessment.currentAllocation = this.calculateCurrentAllocation(portfolioData.holdings);

      // Calculate risk metrics
      assessment.riskMetrics = await this.calculatePortfolioRiskMetrics(portfolioData);

      // Calculate concentration risk
      assessment.concentrationRisk = this.calculateConcentrationRisk(portfolioData.holdings);

      // Calculate diversification score
      assessment.diversificationScore = this.calculateDiversificationScore(portfolioData.holdings);

      // Generate portfolio-specific recommendations
      assessment.recommendations = this.generatePortfolioRiskRecommendations(assessment);

      return assessment;
    } catch (error) {
      logger.error('Error getting portfolio risk assessment:', error);
      throw error;
    }
  }

  /**
   * Get market risk assessment
   */
  async getMarketRiskAssessment() {
    try {
      const assessment = {
        marketConditions: {},
        volatilityIndex: 0,
        sentiment: '',
        riskLevel: '',
        recommendations: {}
      };

      // Get current market conditions
      assessment.marketConditions = await this.getCurrentMarketConditions();

      // Calculate volatility index
      assessment.volatilityIndex = this.calculateVolatilityIndex(assessment.marketConditions);

      // Determine market sentiment
      assessment.sentiment = this.determineMarketSentiment(assessment.marketConditions);

      // Determine market risk level
      assessment.riskLevel = this.determineMarketRiskLevel(assessment.volatilityIndex, assessment.sentiment);

      // Generate market-specific recommendations
      assessment.recommendations = this.generateMarketRiskRecommendations(assessment);

      return assessment;
    } catch (error) {
      logger.error('Error getting market risk assessment:', error);
      throw error;
    }
  }

  /**
   * Calculate age risk score
   */
  calculateAgeRiskScore(age) {
    let score = 0;
    let description = '';

    if (age < 25) {
      score = 25;
      description = 'Young age allows for higher risk tolerance';
    } else if (age < 35) {
      score = 20;
      description = 'Young professional with good risk capacity';
    } else if (age < 45) {
      score = 15;
      description = 'Mid-career with moderate risk tolerance';
    } else if (age < 55) {
      score = 10;
      description = 'Approaching retirement, lower risk tolerance';
    } else {
      score = 5;
      description = 'Retirement age, very low risk tolerance';
    }

    return {
      score,
      description,
      factor: 'age',
      weight: this.riskFactors.age.weight
    };
  }

  /**
   * Calculate income risk score
   */
  calculateIncomeRiskScore(income) {
    let score = 0;
    let description = '';

    if (income > 2000000) {
      score = 20;
      description = 'High income provides strong risk capacity';
    } else if (income > 1000000) {
      score = 15;
      description = 'Good income with moderate risk capacity';
    } else if (income > 500000) {
      score = 10;
      description = 'Average income with limited risk capacity';
    } else if (income > 250000) {
      score = 5;
      description = 'Low income with very limited risk capacity';
    } else {
      score = 0;
      description = 'Very low income, minimal risk capacity';
    }

    return {
      score,
      description,
      factor: 'income',
      weight: this.riskFactors.income.weight
    };
  }

  /**
   * Calculate investment experience risk score
   */
  calculateExperienceRiskScore(experience) {
    let score = 0;
    let description = '';

    if (experience > 10) {
      score = 15;
      description = 'Extensive investment experience';
    } else if (experience > 5) {
      score = 12;
      description = 'Good investment experience';
    } else if (experience > 2) {
      score = 8;
      description = 'Moderate investment experience';
    } else if (experience > 0) {
      score = 4;
      description = 'Limited investment experience';
    } else {
      score = 0;
      description = 'No investment experience';
    }

    return {
      score,
      description,
      factor: 'investmentExperience',
      weight: this.riskFactors.investmentExperience.weight
    };
  }

  /**
   * Calculate investment horizon risk score
   */
  calculateHorizonRiskScore(horizon) {
    let score = 0;
    let description = '';

    if (horizon > 10) {
      score = 25;
      description = 'Long-term investment horizon';
    } else if (horizon > 5) {
      score = 20;
      description = 'Medium-term investment horizon';
    } else if (horizon > 3) {
      score = 15;
      description = 'Short-medium term horizon';
    } else if (horizon > 1) {
      score = 10;
      description = 'Short-term investment horizon';
    } else {
      score = 5;
      description = 'Very short-term horizon';
    }

    return {
      score,
      description,
      factor: 'investmentHorizon',
      weight: this.riskFactors.investmentHorizon.weight
    };
  }

  /**
   * Calculate financial goals risk score
   */
  calculateGoalsRiskScore(goals) {
    let score = 0;
    let description = '';

    const goalTypes = {
      wealthCreation: 8,
      retirement: 6,
      childrenEducation: 5,
      housePurchase: 4,
      emergencyFund: 2
    };

    goals.forEach(goal => {
      score += goalTypes[goal] || 0;
    });

    if (score > 20) {
      description = 'Multiple long-term financial goals';
    } else if (score > 15) {
      description = 'Several important financial goals';
    } else if (score > 10) {
      description = 'Moderate financial goals';
    } else if (score > 5) {
      description = 'Limited financial goals';
    } else {
      description = 'No specific financial goals';
    }

    return {
      score: Math.min(score, 25),
      description,
      factor: 'financialGoals',
      weight: this.riskFactors.financialGoals.weight
    };
  }

  /**
   * Calculate liquidity needs risk score
   */
  calculateLiquidityRiskScore(liquidityNeeds) {
    let score = 0;
    let description = '';

    if (liquidityNeeds === 'high') {
      score = 5;
      description = 'High liquidity needs, lower risk tolerance';
    } else if (liquidityNeeds === 'medium') {
      score = 10;
      description = 'Moderate liquidity needs';
    } else if (liquidityNeeds === 'low') {
      score = 15;
      description = 'Low liquidity needs, higher risk tolerance';
    } else {
      score = 10;
      description = 'Standard liquidity needs';
    }

    return {
      score,
      description,
      factor: 'liquidityNeeds',
      weight: this.riskFactors.liquidityNeeds.weight
    };
  }

  /**
   * Calculate debt level risk score
   */
  calculateDebtRiskScore(debtLevel) {
    let score = 0;
    let description = '';

    if (debtLevel < 0.2) {
      score = 10;
      description = 'Low debt level, high risk capacity';
    } else if (debtLevel < 0.4) {
      score = 7;
      description = 'Moderate debt level';
    } else if (debtLevel < 0.6) {
      score = 4;
      description = 'High debt level, limited risk capacity';
    } else {
      score = 0;
      description = 'Very high debt level, minimal risk capacity';
    }

    return {
      score,
      description,
      factor: 'debtLevel',
      weight: this.riskFactors.debtLevel.weight
    };
  }

  /**
   * Calculate market knowledge risk score
   */
  calculateKnowledgeRiskScore(knowledge) {
    let score = 0;
    let description = '';

    if (knowledge === 'expert') {
      score = 10;
      description = 'Expert market knowledge';
    } else if (knowledge === 'advanced') {
      score = 8;
      description = 'Advanced market knowledge';
    } else if (knowledge === 'intermediate') {
      score = 6;
      description = 'Intermediate market knowledge';
    } else if (knowledge === 'basic') {
      score = 3;
      description = 'Basic market knowledge';
    } else {
      score = 0;
      description = 'No market knowledge';
    }

    return {
      score,
      description,
      factor: 'marketKnowledge',
      weight: this.riskFactors.marketKnowledge.weight
    };
  }

  /**
   * Calculate total weighted risk score
   */
  calculateTotalRiskScore(scores) {
    let totalScore = 0;
    
    Object.keys(this.riskFactors).forEach(factor => {
      const weight = this.riskFactors[factor].weight;
      const score = scores[factor] || 0;
      totalScore += score * weight;
    });

    return Math.round(totalScore);
  }

  /**
   * Determine risk profile based on score
   */
  determineRiskProfile(score) {
    if (score <= 30) {
      return 'conservative';
    } else if (score <= 60) {
      return 'moderate';
    } else {
      return 'aggressive';
    }
  }

  /**
   * Get risk profile details
   */
  getRiskProfileDetails(riskProfile) {
    const profile = this.riskProfiles[riskProfile];
    
    return {
      name: riskProfile,
      description: profile.description,
      recommendedAllocation: {
        equity: {
          min: profile.equityAllocation.min,
          max: profile.equityAllocation.max,
          recommended: Math.round((profile.equityAllocation.min + profile.equityAllocation.max) / 2)
        },
        debt: {
          min: profile.debtAllocation.min,
          max: profile.debtAllocation.max,
          recommended: Math.round((profile.debtAllocation.min + profile.debtAllocation.max) / 2)
        },
        gold: {
          min: profile.goldAllocation.min,
          max: profile.goldAllocation.max,
          recommended: Math.round((profile.goldAllocation.min + profile.goldAllocation.max) / 2)
        }
      },
      characteristics: this.getRiskProfileCharacteristics(riskProfile)
    };
  }

  /**
   * Get risk profile characteristics
   */
  getRiskProfileCharacteristics(riskProfile) {
    const characteristics = {
      conservative: {
        volatility: 'Low',
        expectedReturn: '6-8%',
        suitableFor: ['Retirees', 'Risk-averse investors', 'Short-term goals'],
        investmentTypes: ['Debt funds', 'Liquid funds', 'Bank FDs'],
        timeHorizon: '1-3 years'
      },
      moderate: {
        volatility: 'Medium',
        expectedReturn: '10-12%',
        suitableFor: ['Working professionals', 'Balanced approach', 'Medium-term goals'],
        investmentTypes: ['Balanced funds', 'Large-cap equity', 'Corporate bonds'],
        timeHorizon: '3-7 years'
      },
      aggressive: {
        volatility: 'High',
        expectedReturn: '14-16%',
        suitableFor: ['Young investors', 'High risk tolerance', 'Long-term goals'],
        investmentTypes: ['Equity funds', 'Small-cap funds', 'International funds'],
        timeHorizon: '7+ years'
      }
    };

    return characteristics[riskProfile];
  }

  /**
   * Calculate current portfolio allocation
   */
  calculateCurrentAllocation(holdings) {
    const allocation = {
      equity: 0,
      debt: 0,
      gold: 0,
      others: 0
    };

    holdings.forEach(holding => {
      const category = holding.category?.toLowerCase() || 'others';
      const value = holding.currentValue || 0;

      if (category.includes('equity') || category.includes('growth')) {
        allocation.equity += value;
      } else if (category.includes('debt') || category.includes('income')) {
        allocation.debt += value;
      } else if (category.includes('gold')) {
        allocation.gold += value;
      } else {
        allocation.others += value;
      }
    });

    const total = Object.values(allocation).reduce((sum, val) => sum + val, 0);
    
    if (total > 0) {
      Object.keys(allocation).forEach(key => {
        allocation[key] = Math.round((allocation[key] / total) * 100);
      });
    }

    return allocation;
  }

  /**
   * Calculate portfolio risk metrics
   */
  async calculatePortfolioRiskMetrics(portfolioData) {
    try {
      const metrics = await portfolioAnalyticsService.calculateRiskMetrics(portfolioData);
      
      return {
        volatility: metrics.volatility,
        sharpeRatio: metrics.sharpeRatio,
        maxDrawdown: metrics.maxDrawdown,
        beta: metrics.beta,
        alpha: metrics.alpha,
        informationRatio: metrics.informationRatio,
        riskLevel: metrics.riskLevel
      };
    } catch (error) {
      logger.error('Error calculating portfolio risk metrics:', error);
      return {};
    }
  }

  /**
   * Calculate concentration risk
   */
  calculateConcentrationRisk(holdings) {
    const concentration = {
      topFund: 0,
      top3Funds: 0,
      top5Funds: 0,
      fundHouseConcentration: {},
      categoryConcentration: {}
    };

    if (holdings.length === 0) {
      return concentration;
    }

    const totalValue = holdings.reduce((sum, h) => sum + (h.currentValue || 0), 0);
    
    if (totalValue === 0) {
      return concentration;
    }

    // Sort by value
    const sortedHoldings = holdings.sort((a, b) => (b.currentValue || 0) - (a.currentValue || 0));

    // Calculate top fund concentration
    concentration.topFund = Math.round((sortedHoldings[0].currentValue / totalValue) * 100);

    // Calculate top 3 funds concentration
    const top3Value = sortedHoldings.slice(0, 3).reduce((sum, h) => sum + (h.currentValue || 0), 0);
    concentration.top3Funds = Math.round((top3Value / totalValue) * 100);

    // Calculate top 5 funds concentration
    const top5Value = sortedHoldings.slice(0, 5).reduce((sum, h) => sum + (h.currentValue || 0), 0);
    concentration.top5Funds = Math.round((top5Value / totalValue) * 100);

    // Calculate fund house concentration
    const fundHouseMap = {};
    holdings.forEach(holding => {
      const fundHouse = holding.fundHouse || 'Unknown';
      fundHouseMap[fundHouse] = (fundHouseMap[fundHouse] || 0) + (holding.currentValue || 0);
    });

    Object.keys(fundHouseMap).forEach(fundHouse => {
      concentration.fundHouseConcentration[fundHouse] = Math.round((fundHouseMap[fundHouse] / totalValue) * 100);
    });

    // Calculate category concentration
    const categoryMap = {};
    holdings.forEach(holding => {
      const category = holding.category || 'Unknown';
      categoryMap[category] = (categoryMap[category] || 0) + (holding.currentValue || 0);
    });

    Object.keys(categoryMap).forEach(category => {
      concentration.categoryConcentration[category] = Math.round((categoryMap[category] / totalValue) * 100);
    });

    return concentration;
  }

  /**
   * Calculate diversification score
   */
  calculateDiversificationScore(holdings) {
    if (holdings.length === 0) {
      return 0;
    }

    const totalValue = holdings.reduce((sum, h) => sum + (h.currentValue || 0), 0);
    
    if (totalValue === 0) {
      return 0;
    }

    // Calculate Herfindahl-Hirschman Index (HHI)
    let hhi = 0;
    holdings.forEach(holding => {
      const share = (holding.currentValue || 0) / totalValue;
      hhi += share * share;
    });

    // Convert HHI to diversification score (0-100)
    // Lower HHI = higher diversification
    const diversificationScore = Math.round((1 - hhi) * 100);

    return Math.max(0, Math.min(100, diversificationScore));
  }

  /**
   * Get current market conditions
   */
  async getCurrentMarketConditions() {
    // This would typically fetch from market data service
    // For now, returning mock data
    return {
      nifty50: {
        current: 18500,
        change: 150,
        changePercent: 0.82,
        volatility: 15.2
      },
      sensex: {
        current: 62000,
        change: 450,
        changePercent: 0.73,
        volatility: 14.8
      },
      vix: {
        current: 18.5,
        change: -2.1,
        changePercent: -10.2
      },
      marketSentiment: 'neutral',
      economicIndicators: {
        gdpGrowth: 6.5,
        inflation: 6.2,
        interestRate: 6.5
      }
    };
  }

  /**
   * Calculate volatility index
   */
  calculateVolatilityIndex(marketConditions) {
    const vix = marketConditions.vix?.current || 20;
    
    if (vix < 15) {
      return 'low';
    } else if (vix < 25) {
      return 'medium';
    } else {
      return 'high';
    }
  }

  /**
   * Determine market sentiment
   */
  determineMarketSentiment(marketConditions) {
    const niftyChange = marketConditions.nifty50?.changePercent || 0;
    const sensexChange = marketConditions.sensex?.changePercent || 0;
    const vixChange = marketConditions.vix?.changePercent || 0;

    const averageChange = (niftyChange + sensexChange) / 2;

    if (averageChange > 1 && vixChange < -5) {
      return 'bullish';
    } else if (averageChange < -1 && vixChange > 5) {
      return 'bearish';
    } else {
      return 'neutral';
    }
  }

  /**
   * Determine market risk level
   */
  determineMarketRiskLevel(volatilityIndex, sentiment) {
    if (volatilityIndex === 'high' || sentiment === 'bearish') {
      return 'high';
    } else if (volatilityIndex === 'medium' || sentiment === 'neutral') {
      return 'medium';
    } else {
      return 'low';
    }
  }

  /**
   * Generate risk recommendations
   */
  generateRiskRecommendations(riskProfile) {
    const recommendations = {
      personal: [],
      portfolio: [],
      market: [],
      general: []
    };

    // Personal recommendations
    if (riskProfile.personal.riskProfile === 'conservative') {
      recommendations.personal.push({
        type: 'info',
        message: 'Your conservative risk profile suggests focusing on capital preservation.',
        action: 'Consider debt funds and fixed deposits'
      });
    } else if (riskProfile.personal.riskProfile === 'aggressive') {
      recommendations.personal.push({
        type: 'warning',
        message: 'Your aggressive risk profile may expose you to higher volatility.',
        action: 'Ensure adequate emergency fund and insurance'
      });
    }

    // Portfolio recommendations
    if (riskProfile.portfolio.diversificationScore < 50) {
      recommendations.portfolio.push({
        type: 'warning',
        message: 'Low portfolio diversification detected.',
        action: 'Consider adding more funds across different categories'
      });
    }

    if (riskProfile.portfolio.concentrationRisk.topFund > 30) {
      recommendations.portfolio.push({
        type: 'warning',
        message: 'High concentration in single fund detected.',
        action: 'Consider reducing exposure to top fund'
      });
    }

    // Market recommendations
    if (riskProfile.market.riskLevel === 'high') {
      recommendations.market.push({
        type: 'warning',
        message: 'High market volatility detected.',
        action: 'Consider defensive positioning or systematic investment'
      });
    }

    // General recommendations
    recommendations.general.push({
      type: 'info',
      message: 'Regular portfolio review recommended.',
      action: 'Review portfolio quarterly and rebalance if needed'
    });

    return recommendations;
  }

  /**
   * Generate portfolio risk recommendations
   */
  generatePortfolioRiskRecommendations(assessment) {
    const recommendations = [];

    if (assessment.diversificationScore < 50) {
      recommendations.push({
        type: 'warning',
        message: 'Portfolio lacks diversification',
        action: 'Add funds from different categories and fund houses'
      });
    }

    if (assessment.concentrationRisk.topFund > 30) {
      recommendations.push({
        type: 'warning',
        message: 'High concentration in single fund',
        action: 'Reduce exposure to top fund and distribute across multiple funds'
      });
    }

    if (assessment.riskMetrics.volatility > 20) {
      recommendations.push({
        type: 'warning',
        message: 'High portfolio volatility',
        action: 'Consider adding debt funds to reduce volatility'
      });
    }

    return recommendations;
  }

  /**
   * Generate market risk recommendations
   */
  generateMarketRiskRecommendations(assessment) {
    const recommendations = [];

    if (assessment.riskLevel === 'high') {
      recommendations.push({
        type: 'warning',
        message: 'High market risk environment',
        action: 'Consider defensive positioning and systematic investment'
      });
    }

    if (assessment.sentiment === 'bearish') {
      recommendations.push({
        type: 'info',
        message: 'Bearish market sentiment',
        action: 'Good time for systematic investment and value buying'
      });
    }

    return recommendations;
  }

  /**
   * Calculate overall risk summary
   */
  calculateOverallRiskSummary(riskProfile) {
    const summary = {
      overallRiskLevel: '',
      riskScore: 0,
      recommendations: [],
      nextReviewDate: new Date(Date.now() + 90 * 24 * 60 * 60 * 1000) // 90 days
    };

    // Calculate overall risk level
    const personalScore = riskProfile.personal.totalScore || 0;
    const portfolioRisk = riskProfile.portfolio.riskMetrics?.riskLevel || 'medium';
    const marketRisk = riskProfile.market.riskLevel || 'medium';

    // Weighted average approach
    let overallScore = personalScore * 0.5;
    
    if (portfolioRisk === 'high') overallScore += 25;
    else if (portfolioRisk === 'medium') overallScore += 15;
    else overallScore += 5;

    if (marketRisk === 'high') overallScore += 25;
    else if (marketRisk === 'medium') overallScore += 15;
    else overallScore += 5;

    summary.riskScore = Math.round(overallScore);

    if (overallScore <= 30) {
      summary.overallRiskLevel = 'low';
    } else if (overallScore <= 60) {
      summary.overallRiskLevel = 'medium';
    } else {
      summary.overallRiskLevel = 'high';
    }

    return summary;
  }

  /**
   * Get user risk data (mock implementation)
   */
  async getUserRiskData(userId) {
    // This would typically fetch from database
    return {
      age: 35,
      income: 1500000,
      investmentExperience: 5,
      investmentHorizon: 8,
      financialGoals: ['wealthCreation', 'retirement', 'childrenEducation'],
      liquidityNeeds: 'medium',
      debtLevel: 0.3,
      marketKnowledge: 'intermediate'
    };
  }
}

module.exports = new RiskProfilingService(); 