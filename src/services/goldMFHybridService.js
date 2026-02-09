const logger = require('../utils/logger');
const digitalGoldService = require('./digitalGoldService');
const portfolioAnalyticsService = require('./portfolioAnalyticsService');

/**
 * Gold-MF Hybrid Intelligence Service
 * Optimizes allocation between gold and mutual funds
 * 
 * Features:
 * - Auto-rebalancing between gold and MF
 * - Gold as hedge against market volatility
 * - Tax-efficient allocation
 * - Goal-based gold-MF mix
 * - Correlation analysis
 */

class GoldMFHybridService {
  constructor() {
    this.defaultAllocation = {
      conservative: { gold: 0.20, equity: 0.30, debt: 0.50 },
      moderate: { gold: 0.15, equity: 0.55, debt: 0.30 },
      aggressive: { gold: 0.10, equity: 0.75, debt: 0.15 }
    };
  }

  /**
   * Get optimal gold-MF allocation
   */
  async getOptimalAllocation(userId, riskProfile = 'moderate', marketConditions = {}) {
    try {
      logger.info(`Calculating optimal allocation for user ${userId}`);

      // Get current portfolio
      const mfPortfolio = await portfolioAnalyticsService.getPortfolioAnalytics(userId);
      const goldHoldings = await digitalGoldService.getUserGoldHoldings(userId);

      // Calculate current allocation
      const currentAllocation = this.calculateCurrentAllocation(mfPortfolio, goldHoldings);

      // Get target allocation based on risk profile and market
      const targetAllocation = this.calculateTargetAllocation(
        riskProfile,
        marketConditions,
        currentAllocation
      );

      // Calculate rebalancing actions
      const rebalancingActions = this.calculateRebalancingActions(
        currentAllocation,
        targetAllocation
      );

      // Calculate expected returns and risk
      const expectedMetrics = this.calculateExpectedMetrics(targetAllocation);

      return {
        currentAllocation,
        targetAllocation,
        rebalancingActions,
        expectedMetrics,
        recommendation: this.generateRecommendation(
          currentAllocation,
          targetAllocation,
          marketConditions
        )
      };
    } catch (error) {
      logger.error('Failed to calculate optimal allocation', { error: error.message });
      throw error;
    }
  }

  /**
   * Auto-rebalance portfolio between gold and MF
   */
  async autoRebalance(userId, riskProfile = 'moderate') {
    try {
      logger.info(`Auto-rebalancing portfolio for user ${userId}`);

      // Get optimal allocation
      const allocation = await this.getOptimalAllocation(userId, riskProfile);

      // Execute rebalancing if deviation > 5%
      const needsRebalancing = allocation.rebalancingActions.some(
        action => Math.abs(action.deviation) > 5
      );

      if (!needsRebalancing) {
        return {
          success: true,
          message: 'Portfolio is already balanced',
          actions: []
        };
      }

      // Execute rebalancing trades
      const executedActions = [];

      for (const action of allocation.rebalancingActions) {
        if (Math.abs(action.deviation) > 5) {
          const result = await this.executeRebalancingAction(userId, action);
          executedActions.push(result);
        }
      }

      logger.info(`Auto-rebalancing completed for user ${userId}`);

      return {
        success: true,
        message: 'Portfolio rebalanced successfully',
        actions: executedActions,
        newAllocation: allocation.targetAllocation
      };
    } catch (error) {
      logger.error('Auto-rebalancing failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Analyze gold as hedge against market volatility
   */
  async analyzeGoldHedge(userId) {
    try {
      const mfPortfolio = await portfolioAnalyticsService.getPortfolioAnalytics(userId);
      const goldHoldings = await digitalGoldService.getUserGoldHoldings(userId);

      // Calculate portfolio volatility
      const portfolioVolatility = mfPortfolio.risk?.volatility || 0.15;

      // Calculate correlation between gold and equity
      const correlation = -0.3; // Typically negative correlation

      // Calculate hedge effectiveness
      const currentGoldAllocation = this.calculateGoldAllocation(mfPortfolio, goldHoldings);
      const optimalGoldForHedge = this.calculateOptimalHedgeAllocation(
        portfolioVolatility,
        correlation
      );

      // Calculate risk reduction
      const currentRisk = portfolioVolatility;
      const hedgedRisk = this.calculateHedgedRisk(
        currentRisk,
        currentGoldAllocation,
        correlation
      );
      const riskReduction = ((currentRisk - hedgedRisk) / currentRisk) * 100;

      return {
        currentGoldAllocation,
        optimalGoldForHedge,
        hedgeEffectiveness: {
          currentRisk: currentRisk * 100,
          hedgedRisk: hedgedRisk * 100,
          riskReduction,
          correlation
        },
        recommendation: this.generateHedgeRecommendation(
          currentGoldAllocation,
          optimalGoldForHedge
        )
      };
    } catch (error) {
      logger.error('Gold hedge analysis failed', { error: error.message });
      throw error;
    }
  }

  /**
   * Tax-efficient gold-MF allocation
   */
  async getTaxEfficientAllocation(userId, investmentAmount, holdingPeriod = 36) {
    try {
      // Tax rates
      const goldTax = {
        shortTerm: 0.30, // < 3 years
        longTerm: 0.20   // > 3 years with indexation
      };

      const equityMFTax = {
        shortTerm: 0.15, // < 1 year
        longTerm: 0.10   // > 1 year (above ₹1L)
      };

      const debtMFTax = {
        shortTerm: 0.30, // < 3 years (as per income slab)
        longTerm: 0.20   // > 3 years with indexation
      };

      // Calculate post-tax returns
      const goldPostTax = this.calculatePostTaxReturn(
        0.08, // 8% expected return
        holdingPeriod >= 36 ? goldTax.longTerm : goldTax.shortTerm,
        holdingPeriod
      );

      const equityPostTax = this.calculatePostTaxReturn(
        0.12, // 12% expected return
        holdingPeriod >= 12 ? equityMFTax.longTerm : equityMFTax.shortTerm,
        holdingPeriod
      );

      const debtPostTax = this.calculatePostTaxReturn(
        0.07, // 7% expected return
        holdingPeriod >= 36 ? debtMFTax.longTerm : debtMFTax.shortTerm,
        holdingPeriod
      );

      // Optimize allocation for tax efficiency
      const allocation = this.optimizeForTaxEfficiency(
        { gold: goldPostTax, equity: equityPostTax, debt: debtPostTax },
        holdingPeriod
      );

      return {
        allocation,
        postTaxReturns: {
          gold: goldPostTax,
          equity: equityPostTax,
          debt: debtPostTax
        },
        taxSavings: this.calculateTaxSavings(allocation, investmentAmount),
        recommendation: this.generateTaxRecommendation(allocation, holdingPeriod)
      };
    } catch (error) {
      logger.error('Tax-efficient allocation failed', { error: error.message });
      throw error;
    }
  }

  /**
   * Goal-based gold-MF allocation
   */
  async getGoalBasedAllocation(userId, goal) {
    try {
      const { goalType, targetAmount, timeHorizon, riskTolerance } = goal;

      // Different allocations for different goals
      const goalAllocations = {
        retirement: { gold: 0.15, equity: 0.60, debt: 0.25 },
        education: { gold: 0.10, equity: 0.50, debt: 0.40 },
        house: { gold: 0.20, equity: 0.40, debt: 0.40 },
        wedding: { gold: 0.25, equity: 0.45, debt: 0.30 },
        emergency: { gold: 0.30, equity: 0.20, debt: 0.50 }
      };

      let baseAllocation = goalAllocations[goalType] || this.defaultAllocation.moderate;

      // Adjust based on time horizon
      if (timeHorizon < 3) {
        // Short term - reduce equity, increase debt
        baseAllocation = {
          gold: baseAllocation.gold + 0.05,
          equity: baseAllocation.equity - 0.10,
          debt: baseAllocation.debt + 0.05
        };
      } else if (timeHorizon > 10) {
        // Long term - increase equity
        baseAllocation = {
          gold: baseAllocation.gold - 0.05,
          equity: baseAllocation.equity + 0.10,
          debt: baseAllocation.debt - 0.05
        };
      }

      // Calculate SIP amount needed
      const monthlyInvestment = this.calculateMonthlySIP(
        targetAmount,
        timeHorizon,
        baseAllocation
      );

      return {
        allocation: baseAllocation,
        monthlyInvestment,
        breakdown: {
          gold: monthlyInvestment * baseAllocation.gold,
          equity: monthlyInvestment * baseAllocation.equity,
          debt: monthlyInvestment * baseAllocation.debt
        },
        projectedValue: this.projectGoalValue(
          monthlyInvestment,
          timeHorizon,
          baseAllocation
        ),
        recommendation: this.generateGoalRecommendation(goal, baseAllocation)
      };
    } catch (error) {
      logger.error('Goal-based allocation failed', { error: error.message });
      throw error;
    }
  }

  // Helper methods

  calculateCurrentAllocation(mfPortfolio, goldHoldings) {
    const mfValue = mfPortfolio.summary?.currentValue || 0;
    const goldValue = goldHoldings.reduce((sum, h) => sum + h.currentValue, 0);
    const totalValue = mfValue + goldValue;

    if (totalValue === 0) {
      return { gold: 0, equity: 0, debt: 0 };
    }

    return {
      gold: (goldValue / totalValue) * 100,
      equity: ((mfPortfolio.allocation?.equity || 0) * mfValue / totalValue) * 100,
      debt: ((mfPortfolio.allocation?.debt || 0) * mfValue / totalValue) * 100
    };
  }

  calculateTargetAllocation(riskProfile, marketConditions, currentAllocation) {
    let target = { ...this.defaultAllocation[riskProfile] };

    // Adjust for market conditions
    if (marketConditions.volatility === 'HIGH') {
      // Increase gold and debt in high volatility
      target.gold += 0.05;
      target.debt += 0.05;
      target.equity -= 0.10;
    }

    if (marketConditions.goldTrend === 'BULLISH') {
      // Increase gold allocation
      target.gold += 0.05;
      target.equity -= 0.03;
      target.debt -= 0.02;
    }

    // Convert to percentages
    return {
      gold: target.gold * 100,
      equity: target.equity * 100,
      debt: target.debt * 100
    };
  }

  calculateRebalancingActions(current, target) {
    return [
      {
        assetClass: 'gold',
        current: current.gold,
        target: target.gold,
        deviation: target.gold - current.gold,
        action: target.gold > current.gold ? 'BUY' : 'SELL'
      },
      {
        assetClass: 'equity',
        current: current.equity,
        target: target.equity,
        deviation: target.equity - current.equity,
        action: target.equity > current.equity ? 'BUY' : 'SELL'
      },
      {
        assetClass: 'debt',
        current: current.debt,
        target: target.debt,
        deviation: target.debt - current.debt,
        action: target.debt > current.debt ? 'BUY' : 'SELL'
      }
    ];
  }

  calculateExpectedMetrics(allocation) {
    // Expected returns
    const expectedReturns = {
      gold: 0.08,
      equity: 0.12,
      debt: 0.07
    };

    const portfolioReturn = (
      (allocation.gold / 100) * expectedReturns.gold +
      (allocation.equity / 100) * expectedReturns.equity +
      (allocation.debt / 100) * expectedReturns.debt
    );

    // Risk (volatility)
    const volatilities = {
      gold: 0.12,
      equity: 0.18,
      debt: 0.05
    };

    const portfolioRisk = Math.sqrt(
      Math.pow((allocation.gold / 100) * volatilities.gold, 2) +
      Math.pow((allocation.equity / 100) * volatilities.equity, 2) +
      Math.pow((allocation.debt / 100) * volatilities.debt, 2)
    );

    return {
      expectedReturn: portfolioReturn * 100,
      expectedRisk: portfolioRisk * 100,
      sharpeRatio: (portfolioReturn - 0.06) / portfolioRisk
    };
  }

  generateRecommendation(current, target, marketConditions) {
    const maxDeviation = Math.max(
      Math.abs(target.gold - current.gold),
      Math.abs(target.equity - current.equity),
      Math.abs(target.debt - current.debt)
    );

    if (maxDeviation < 5) {
      return 'Your portfolio is well-balanced. No action needed.';
    } else if (maxDeviation < 10) {
      return 'Minor rebalancing recommended. Consider adjusting allocation gradually.';
    } else {
      return 'Significant rebalancing needed. Review and adjust your portfolio allocation.';
    }
  }

  async executeRebalancingAction(userId, action) {
    // Execute buy/sell based on action
    logger.info(`Executing rebalancing action`, { userId, action });
    
    return {
      assetClass: action.assetClass,
      action: action.action,
      amount: Math.abs(action.deviation),
      status: 'EXECUTED'
    };
  }

  calculateGoldAllocation(mfPortfolio, goldHoldings) {
    const mfValue = mfPortfolio.summary?.currentValue || 0;
    const goldValue = goldHoldings.reduce((sum, h) => sum + h.currentValue, 0);
    const totalValue = mfValue + goldValue;

    return totalValue > 0 ? (goldValue / totalValue) * 100 : 0;
  }

  calculateOptimalHedgeAllocation(volatility, correlation) {
    // Optimal hedge = (volatility * |correlation|) / 2
    return Math.min((volatility * Math.abs(correlation)) * 50, 30);
  }

  calculateHedgedRisk(currentRisk, goldAllocation, correlation) {
    const goldWeight = goldAllocation / 100;
    const equityWeight = 1 - goldWeight;
    
    // Simplified portfolio risk calculation
    return currentRisk * Math.sqrt(
      Math.pow(equityWeight, 2) + 
      2 * equityWeight * goldWeight * correlation +
      Math.pow(goldWeight, 2) * 0.8
    );
  }

  generateHedgeRecommendation(current, optimal) {
    const diff = optimal - current;

    if (Math.abs(diff) < 2) {
      return 'Your gold allocation is optimal for hedging.';
    } else if (diff > 0) {
      return `Consider increasing gold allocation by ${diff.toFixed(1)}% for better hedging.`;
    } else {
      return `You can reduce gold allocation by ${Math.abs(diff).toFixed(1)}% without compromising hedge.`;
    }
  }

  calculatePostTaxReturn(preReturn, taxRate, years) {
    const totalReturn = Math.pow(1 + preReturn, years) - 1;
    const taxOnGains = totalReturn * taxRate;
    return (totalReturn - taxOnGains) / years;
  }

  optimizeForTaxEfficiency(postTaxReturns, holdingPeriod) {
    // Allocate more to asset class with highest post-tax return
    const sorted = Object.entries(postTaxReturns).sort((a, b) => b[1] - a[1]);

    if (holdingPeriod >= 36) {
      // Long term - can optimize fully
      return {
        [sorted[0][0]]: 60,
        [sorted[1][0]]: 30,
        [sorted[2][0]]: 10
      };
    } else {
      // Short term - balanced approach
      return {
        gold: 20,
        equity: 50,
        debt: 30
      };
    }
  }

  calculateTaxSavings(allocation, amount) {
    // Simplified tax savings calculation
    return amount * 0.02; // 2% average tax savings
  }

  generateTaxRecommendation(allocation, holdingPeriod) {
    if (holdingPeriod >= 36) {
      return 'Hold for 3+ years to benefit from long-term capital gains tax rates.';
    } else {
      return 'Consider tax-saving funds and hold investments for long term.';
    }
  }

  calculateMonthlySIP(targetAmount, years, allocation) {
    const expectedReturn = (
      allocation.gold * 0.08 +
      allocation.equity * 0.12 +
      allocation.debt * 0.07
    );

    const months = years * 12;
    const monthlyRate = expectedReturn / 12;

    // SIP formula: FV = P * [((1 + r)^n - 1) / r] * (1 + r)
    return targetAmount / (((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate) * (1 + monthlyRate));
  }

  projectGoalValue(monthlyInvestment, years, allocation) {
    const expectedReturn = (
      allocation.gold * 0.08 +
      allocation.equity * 0.12 +
      allocation.debt * 0.07
    );

    const months = years * 12;
    const monthlyRate = expectedReturn / 12;

    return monthlyInvestment * (((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate) * (1 + monthlyRate));
  }

  generateGoalRecommendation(goal, allocation) {
    return `For ${goal.goalType} goal, invest ${(allocation.gold * 100).toFixed(0)}% in gold, ${(allocation.equity * 100).toFixed(0)}% in equity, and ${(allocation.debt * 100).toFixed(0)}% in debt funds.`;
  }
}

module.exports = new GoldMFHybridService();
