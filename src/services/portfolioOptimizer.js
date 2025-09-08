const logger = require('../utils/logger');
const { User, UserPortfolio, Holding, Fund } = require('../models');
const aiPortfolioOptimizer = require('./aiPortfolioOptimizer');

class PortfolioOptimizer {
  constructor() {
    this.riskProfiles = {
      CONSERVATIVE: { equity: 0.2, debt: 0.6, liquid: 0.2 },
      MODERATE: { equity: 0.6, debt: 0.35, liquid: 0.05 },
      AGGRESSIVE: { equity: 0.8, debt: 0.15, liquid: 0.05 }
    };
    
    this.fundCategories = {
      EQUITY: ['Large Cap', 'Mid Cap', 'Small Cap', 'Multi Cap', 'Sectoral'],
      DEBT: ['Liquid', 'Ultra Short Term', 'Short Term', 'Medium Term', 'Long Term'],
      HYBRID: ['Conservative Hybrid', 'Balanced Hybrid', 'Aggressive Hybrid'],
      SOLUTION: ['Retirement', 'Children', 'Tax Saving']
    };
  }

  /**
   * Optimize portfolio using AI and ML algorithms
   */
  async optimizePortfolio(userProfile, riskTolerance, goals, currentPortfolio = null) {
    try {
      logger.info('Starting portfolio optimization', { userId: userProfile.userId, riskTolerance });

      // Get user's current portfolio if available
      const currentHoldings = currentPortfolio ? await this.getCurrentHoldings(userProfile.userId) : [];

      // Get fund universe based on user profile
      const fundUniverse = await this.getFundUniverse(userProfile, riskTolerance);

      // Calculate optimal allocation using AI
      const optimalAllocation = await this.calculateOptimalAllocation(
        userProfile, 
        riskTolerance, 
        goals, 
        currentHoldings
      );

      // Generate fund recommendations
      const fundRecommendations = await this.generateFundRecommendations(
        optimalAllocation,
        fundUniverse,
        currentHoldings
      );

      // Generate tax-efficient strategies
      const taxStrategies = await this.generateTaxStrategies(userProfile, currentHoldings);

      // Generate Smart SIP/STP recommendations
      const smartRecommendations = await this.generateSmartRecommendations(
        userProfile,
        currentHoldings,
        optimalAllocation
      );

      // Calculate rebalancing triggers
      const rebalancingTriggers = await this.calculateRebalancingTriggers(
        currentHoldings,
        optimalAllocation
      );

      const result = {
        success: true,
        data: {
          userProfile,
          riskTolerance,
          goals,
          optimalAllocation,
          fundRecommendations,
          taxStrategies,
          smartRecommendations,
          rebalancingTriggers,
          currentAnalysis: currentHoldings.length > 0 ? await this.analyzeCurrentPortfolio(currentHoldings) : null
        }
      };

      logger.info('Portfolio optimization completed', { 
        userId: userProfile.userId, 
        recommendationsCount: fundRecommendations.length 
      });

      return result;
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
   * Calculate optimal asset allocation using AI
   */
  async calculateOptimalAllocation(userProfile, riskTolerance, goals, currentHoldings) {
    try {
      // Base allocation from risk profile
      let baseAllocation = this.riskProfiles[riskTolerance] || this.riskProfiles.MODERATE;

      // Adjust based on user profile
      const ageAdjustment = this.calculateAgeAdjustment(userProfile.age);
      const incomeAdjustment = this.calculateIncomeAdjustment(userProfile.income);
      const horizonAdjustment = this.calculateHorizonAdjustment(userProfile.investmentHorizon);

      // Apply adjustments
      const adjustedAllocation = {
        equity: Math.min(0.9, Math.max(0.1, baseAllocation.equity + ageAdjustment + incomeAdjustment + horizonAdjustment)),
        debt: Math.min(0.8, Math.max(0.1, baseAllocation.debt - ageAdjustment - incomeAdjustment - horizonAdjustment)),
        liquid: Math.min(0.3, Math.max(0.05, baseAllocation.liquid))
      };

      // Normalize to 100%
      const total = adjustedAllocation.equity + adjustedAllocation.debt + adjustedAllocation.liquid;
      adjustedAllocation.equity /= total;
      adjustedAllocation.debt /= total;
      adjustedAllocation.liquid /= total;

      // Add category breakdowns
      const categoryBreakdown = await this.getCategoryBreakdown(adjustedAllocation, goals);

      return {
        ...adjustedAllocation,
        categoryBreakdown,
        riskScore: this.calculateRiskScore(adjustedAllocation),
        expectedReturn: this.calculateExpectedReturn(adjustedAllocation),
        volatility: this.calculateVolatility(adjustedAllocation)
      };
    } catch (error) {
      logger.error('Error calculating optimal allocation', { error: error.message });
      throw error;
    }
  }

  /**
   * Generate fund recommendations based on optimal allocation
   */
  async generateFundRecommendations(optimalAllocation, fundUniverse, currentHoldings) {
    try {
      const recommendations = [];

      // For each category in optimal allocation
      for (const [category, targetAllocation] of Object.entries(optimalAllocation.categoryBreakdown)) {
        const categoryFunds = fundUniverse.filter(fund => fund.category === category);
        
        // Score funds based on multiple criteria
        const scoredFunds = categoryFunds.map(fund => ({
          ...fund,
          score: this.calculateFundScore(fund, currentHoldings)
        }));

        // Sort by score and take top recommendations
        const topFunds = scoredFunds
          .sort((a, b) => b.score - a.score)
          .slice(0, 3); // Top 3 funds per category

        recommendations.push({
          category,
          targetAllocation,
          recommendedFunds: topFunds,
          reasoning: this.generateReasoning(category, topFunds)
        });
      }

      return recommendations;
    } catch (error) {
      logger.error('Error generating fund recommendations', { error: error.message });
      throw error;
    }
  }

  /**
   * Generate tax-efficient strategies
   */
  async generateTaxStrategies(userProfile, currentHoldings) {
    try {
      const strategies = [];

      // ELSS recommendation for tax saving
      if (userProfile.taxSlab > 0.1) {
        strategies.push({
          type: 'ELSS_INVESTMENT',
          description: 'Invest in ELSS funds for tax deduction under Section 80C',
          amount: Math.min(150000, userProfile.income * 0.1),
          expectedTaxSaving: Math.min(150000, userProfile.income * 0.1) * userProfile.taxSlab,
          priority: 'HIGH',
          funds: await this.getELSSFunds()
        });
      }

      // LTCG harvesting opportunities
      const ltcgOpportunities = await this.findLTCGOpportunities(currentHoldings);
      if (ltcgOpportunities.length > 0) {
        strategies.push({
          type: 'LTCG_HARVESTING',
          description: 'Consider harvesting long-term capital gains',
          opportunities: ltcgOpportunities,
          priority: 'MEDIUM'
        });
      }

      // Dividend optimization
      strategies.push({
        type: 'DIVIDEND_OPTIMIZATION',
        description: 'Consider growth funds over dividend funds for better tax efficiency',
        strategy: 'Switch to growth option where available',
        expectedTaxSaving: userProfile.income * 0.01, // Rough estimate
        priority: 'LOW'
      });

      return strategies;
    } catch (error) {
      logger.error('Error generating tax strategies', { error: error.message });
      throw error;
    }
  }

  /**
   * Generate Smart SIP/STP recommendations
   */
  async generateSmartRecommendations(userProfile, currentHoldings, optimalAllocation) {
    try {
      const recommendations = [];

      // SIP optimization
      const sipRecommendations = await this.optimizeSIP(userProfile, currentHoldings, optimalAllocation);
      recommendations.push(...sipRecommendations);

      // STP recommendations for existing investments
      const stpRecommendations = await this.optimizeSTP(currentHoldings, optimalAllocation);
      recommendations.push(...stpRecommendations);

      // SWP recommendations for retirees
      if (userProfile.age > 55) {
        const swpRecommendations = await this.optimizeSWP(userProfile, currentHoldings);
        recommendations.push(...swpRecommendations);
      }

      return recommendations;
    } catch (error) {
      logger.error('Error generating smart recommendations', { error: error.message });
      throw error;
    }
  }

  /**
   * Calculate rebalancing triggers
   */
  async calculateRebalancingTriggers(currentHoldings, optimalAllocation) {
    try {
      const triggers = [];

      if (currentHoldings.length === 0) {
        return triggers;
      }

      // Calculate current allocation
      const currentAllocation = this.calculateCurrentAllocation(currentHoldings);

      // Check for significant deviations
      for (const [category, targetAllocation] of Object.entries(optimalAllocation.categoryBreakdown)) {
        const currentAlloc = currentAllocation[category] || 0;
        const deviation = Math.abs(currentAlloc - targetAllocation);

        if (deviation > 0.05) { // 5% deviation threshold
          triggers.push({
            category,
            currentAllocation: currentAlloc,
            targetAllocation,
            deviation,
            action: currentAlloc > targetAllocation ? 'REDUCE' : 'INCREASE',
            priority: deviation > 0.1 ? 'HIGH' : 'MEDIUM'
          });
        }
      }

      return triggers;
    } catch (error) {
      logger.error('Error calculating rebalancing triggers', { error: error.message });
      throw error;
    }
  }

  // Helper methods
  calculateAgeAdjustment(age) {
    if (age < 30) return 0.1; // More aggressive for young investors
    if (age < 50) return 0;
    if (age < 60) return -0.1; // More conservative for older investors
    return -0.2; // Very conservative for retirees
  }

  calculateIncomeAdjustment(income) {
    if (income > 2000000) return 0.05; // Higher income can take more risk
    if (income > 1000000) return 0;
    return -0.05; // Lower income should be more conservative
  }

  calculateHorizonAdjustment(horizon) {
    switch (horizon) {
      case 'SHORT_TERM': return -0.15;
      case 'MEDIUM_TERM': return 0;
      case 'LONG_TERM': return 0.1;
      default: return 0;
    }
  }

  async getCategoryBreakdown(allocation, goals) {
    const breakdown = {
      'Large Cap': allocation.equity * 0.4,
      'Mid Cap': allocation.equity * 0.3,
      'Small Cap': allocation.equity * 0.2,
      'Sectoral': allocation.equity * 0.1,
      'Liquid': allocation.liquid,
      'Ultra Short Term': allocation.debt * 0.3,
      'Short Term': allocation.debt * 0.4,
      'Medium Term': allocation.debt * 0.3
    };

    // Adjust for goals
    if (goals.includes('TAX_SAVING')) {
      breakdown['ELSS'] = Math.min(0.15, allocation.equity * 0.2);
    }

    return breakdown;
  }

  calculateRiskScore(allocation) {
    return allocation.equity * 0.8 + allocation.debt * 0.2 + allocation.liquid * 0.1;
  }

  calculateExpectedReturn(allocation) {
    return allocation.equity * 0.12 + allocation.debt * 0.08 + allocation.liquid * 0.06;
  }

  calculateVolatility(allocation) {
    return allocation.equity * 0.18 + allocation.debt * 0.05 + allocation.liquid * 0.02;
  }

  calculateFundScore(fund, currentHoldings) {
    let score = 0;

    // Historical performance (40% weight)
    score += (fund.returns1Y || 0) * 0.4;

    // Risk-adjusted returns (30% weight)
    const sharpeRatio = fund.sharpeRatio || 1;
    score += sharpeRatio * 0.3;

    // Fund size and stability (20% weight)
    const aumScore = Math.min(1, (fund.aum || 0) / 1000000000); // Normalize to 1B
    score += aumScore * 0.2;

    // Expense ratio (10% weight)
    const expenseScore = Math.max(0, 1 - (fund.expenseRatio || 0.02) * 50);
    score += expenseScore * 0.1;

    return score;
  }

  generateReasoning(category, funds) {
    const topFund = funds[0];
    return `Recommended ${category} funds based on strong historical performance (${(topFund.returns1Y * 100).toFixed(1)}% 1Y return), good risk-adjusted returns, and stable fund management.`;
  }

  async getCurrentHoldings(userId) {
    try {
      const holdings = await Holding.find({ userId, isActive: true });
      return holdings;
    } catch (error) {
      logger.error('Error fetching current holdings', { error: error.message });
      return [];
    }
  }

  async getFundUniverse(userProfile, riskTolerance) {
    try {
      // In a real implementation, this would fetch from a fund database
      // For now, return mock data
      return [
        {
          schemeCode: '123456',
          fundName: 'Axis Bluechip Fund',
          category: 'Large Cap',
          returns1Y: 0.15,
          sharpeRatio: 1.2,
          aum: 5000000000,
          expenseRatio: 0.018
        },
        // Add more mock funds...
      ];
    } catch (error) {
      logger.error('Error fetching fund universe', { error: error.message });
      return [];
    }
  }

  async analyzeCurrentPortfolio(holdings) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const totalInvested = holdings.reduce((sum, h) => sum + h.purchaseValue, 0);
      
      return {
        totalValue,
        totalInvested,
        currentGain: totalValue - totalInvested,
        gainPercentage: ((totalValue - totalInvested) / totalInvested) * 100,
        assetAllocation: this.calculateCurrentAllocation(holdings)
      };
    } catch (error) {
      logger.error('Error analyzing current portfolio', { error: error.message });
      return null;
    }
  }

  calculateCurrentAllocation(holdings) {
    const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
    const allocation = {};

    holdings.forEach(holding => {
      const category = holding.fundCategory || 'others';
      allocation[category] = (allocation[category] || 0) + (holding.currentValue / totalValue);
    });

    return allocation;
  }

  async getELSSFunds() {
    // Mock ELSS funds
    return [
      { schemeCode: 'ELSS001', fundName: 'Axis ELSS Tax Saver Fund' },
      { schemeCode: 'ELSS002', fundName: 'HDFC Tax Saver Fund' }
    ];
  }

  async findLTCGOpportunities(holdings) {
    const opportunities = [];
    const now = new Date();

    holdings.forEach(holding => {
      const holdingPeriod = (now - new Date(holding.purchaseDate)) / (1000 * 60 * 60 * 24 * 365);
      
      if (holdingPeriod >= 0.9 && holdingPeriod < 1.1) { // Near 1 year
        const gain = holding.currentValue - holding.purchaseValue;
        if (gain > 0) {
          opportunities.push({
            fundName: holding.fundName,
            gain,
            gainPercentage: (gain / holding.purchaseValue) * 100,
            daysToLTCG: Math.max(0, 365 - holdingPeriod * 365)
          });
        }
      }
    });

    return opportunities;
  }

  async optimizeSIP(userProfile, currentHoldings, optimalAllocation) {
    const recommendations = [];

    // Calculate optimal SIP amount based on goals
    const monthlyIncome = userProfile.income / 12;
    const recommendedSIP = Math.min(monthlyIncome * 0.3, 50000); // 30% of monthly income, max 50k

    recommendations.push({
      type: 'SIP_OPTIMIZATION',
      description: 'Optimize SIP amount and frequency',
      currentSIP: userProfile.currentSIP || 0,
      recommendedSIP,
      reasoning: `Based on your income and goals, we recommend a monthly SIP of ₹${recommendedSIP.toLocaleString()}`
    });

    return recommendations;
  }

  async optimizeSTP(currentHoldings, optimalAllocation) {
    const recommendations = [];

    // Find funds that need rebalancing
    const currentAllocation = this.calculateCurrentAllocation(currentHoldings);
    
    for (const [category, targetAllocation] of Object.entries(optimalAllocation.categoryBreakdown)) {
      const currentAlloc = currentAllocation[category] || 0;
      const deviation = targetAllocation - currentAlloc;

      if (deviation > 0.05) {
        recommendations.push({
          type: 'STP_RECOMMENDATION',
          description: `Consider STP to increase ${category} allocation`,
          fromFund: 'Liquid Fund',
          toFund: `${category} Fund`,
          amount: deviation * 100000, // Example amount
          reasoning: `Your ${category} allocation is ${(deviation * 100).toFixed(1)}% below target`
        });
      }
    }

    return recommendations;
  }

  async optimizeSWP(userProfile, currentHoldings) {
    const recommendations = [];

    // Calculate monthly withdrawal amount
    const totalValue = currentHoldings.reduce((sum, h) => sum + h.currentValue, 0);
    const monthlyWithdrawal = totalValue * 0.04 / 12; // 4% rule

    recommendations.push({
      type: 'SWP_RECOMMENDATION',
      description: 'Set up Systematic Withdrawal Plan for retirement income',
      recommendedWithdrawal: monthlyWithdrawal,
      reasoning: `Based on your portfolio value, we recommend a monthly SWP of ₹${monthlyWithdrawal.toLocaleString()}`
    });

    return recommendations;
  }
}

module.exports = new PortfolioOptimizer(); 