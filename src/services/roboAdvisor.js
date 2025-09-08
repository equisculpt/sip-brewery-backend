const logger = require('../utils/logger');
const { User, UserPortfolio, Holding, Goal, Transaction, SIP } = require('../models');
const portfolioOptimizer = require('./portfolioOptimizer');
const predictiveEngine = require('./predictiveEngine');
const marketAnalyticsEngine = require('./marketAnalyticsEngine');

class RoboAdvisor {
  constructor() {
    this.reviewFrequencies = {
      DAILY: 'daily',
      WEEKLY: 'weekly',
      MONTHLY: 'monthly',
      QUARTERLY: 'quarterly'
    };

    this.reviewTypes = {
      PORTFOLIO_REVIEW: 'portfolio_review',
      REBALANCING: 'rebalancing',
      TAX_HARVESTING: 'tax_harvesting',
      GOAL_TRACKING: 'goal_tracking',
      RISK_ASSESSMENT: 'risk_assessment',
      MARKET_OPPORTUNITY: 'market_opportunity'
    };

    this.recommendationTypes = {
      SWITCH: 'switch',
      REBALANCE: 'rebalance',
      TAX_HARVEST: 'tax_harvest',
      SIP_ADJUST: 'sip_adjust',
      GOAL_ADJUST: 'goal_adjust',
      RISK_REDUCE: 'risk_reduce'
    };

    this.thresholds = {
      REBALANCING_DEVIATION: 0.05, // 5%
      TAX_HARVESTING_DAYS: 30, // Days before 1 year
      SIP_DEVIATION: 0.1, // 10%
      GOAL_DEVIATION: 0.15, // 15%
      RISK_THRESHOLD: 0.7 // 70%
    };
  }

  /**
   * Perform automatic portfolio review (monthly)
   */
  async performPortfolioReview(userId, reviewType = 'monthly') {
    try {
      logger.info('Starting portfolio review', { userId, reviewType });

      // Get user and portfolio data
      const user = await User.findById(userId);
      const portfolio = await UserPortfolio.findOne({ userId });
      const holdings = await Holding.find({ userId, isActive: true });
      const goals = await Goal.find({ userId, isActive: true });

      if (!user || !portfolio) {
        throw new Error('User or portfolio not found');
      }

      // Perform comprehensive review
      const reviewResults = {
        portfolioHealth: await this.assessPortfolioHealth(holdings, portfolio),
        rebalancingNeeds: await this.checkRebalancingNeeds(holdings, user),
        taxOpportunities: await this.checkTaxHarvestingOpportunities(holdings),
        goalProgress: await this.assessGoalProgress(goals, portfolio),
        riskAssessment: await this.assessRiskLevel(holdings, user),
        marketOpportunities: await this.identifyMarketOpportunities(holdings),
        recommendations: [],
        alerts: [],
        reviewDate: new Date().toISOString()
      };

      // Generate recommendations
      reviewResults.recommendations = await this.generateRecommendations(reviewResults, user);

      // Generate alerts
      reviewResults.alerts = await this.generateAlerts(reviewResults, user);

      // Store review results
      await this.storeReviewResults(userId, reviewResults);

      logger.info('Portfolio review completed', { 
        userId, 
        recommendationsCount: reviewResults.recommendations.length,
        alertsCount: reviewResults.alerts.length 
      });

      return {
        success: true,
        data: reviewResults
      };
    } catch (error) {
      logger.error('Portfolio review failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to perform portfolio review',
        error: error.message
      };
    }
  }

  /**
   * Offer switch recommendations with explanation
   */
  async generateSwitchRecommendations(userId, fundName = null) {
    try {
      logger.info('Generating switch recommendations', { userId, fundName });

      const user = await User.findById(userId);
      const holdings = await Holding.find({ userId, isActive: true });

      if (!user) {
        throw new Error('User not found');
      }

      const switchRecommendations = [];

      // Analyze each holding for potential switches
      for (const holding of holdings) {
        if (fundName && holding.fundName !== fundName) {
          continue; // Skip if specific fund requested
        }

        const switchAnalysis = await this.analyzeFundForSwitch(holding, user);
        
        if (switchAnalysis.shouldSwitch) {
          switchRecommendations.push({
            currentFund: {
              name: holding.fundName,
              category: holding.fundCategory,
              returns1Y: holding.returns1Y,
              currentValue: holding.currentValue,
              units: holding.units
            },
            recommendedFund: switchAnalysis.recommendedFund,
            reason: switchAnalysis.reason,
            expectedBenefit: switchAnalysis.expectedBenefit,
            switchType: switchAnalysis.switchType,
            priority: switchAnalysis.priority
          });
        }
      }

      // Sort by priority
      switchRecommendations.sort((a, b) => {
        const priorityOrder = { HIGH: 3, MEDIUM: 2, LOW: 1 };
        return priorityOrder[b.priority] - priorityOrder[a.priority];
      });

      return {
        success: true,
        data: {
          recommendations: switchRecommendations,
          totalHoldings: holdings.length,
          recommendedSwitches: switchRecommendations.length
        }
      };
    } catch (error) {
      logger.error('Switch recommendations generation failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to generate switch recommendations',
        error: error.message
      };
    }
  }

  /**
   * Tax harvesting checker: show funds with 11-month holding nearing 1-year LTCG
   */
  async checkTaxHarvestingOpportunities(holdings) {
    try {
      logger.info('Checking tax harvesting opportunities');

      const opportunities = [];
      const now = new Date();

      for (const holding of holdings) {
        const holdingPeriod = (now - new Date(holding.purchaseDate)) / (1000 * 60 * 60 * 24);
        const daysToLTCG = 365 - holdingPeriod;

        // Check if fund is approaching 1-year holding period
        if (daysToLTCG <= this.thresholds.TAX_HARVESTING_DAYS && daysToLTCG > 0) {
          const gain = holding.currentValue - holding.purchaseValue;
          
          if (gain > 0) {
            const gainPercentage = (gain / holding.purchaseValue) * 100;
            
            opportunities.push({
              fundName: holding.fundName,
              fundCategory: holding.fundCategory,
              purchaseDate: holding.purchaseDate,
              holdingPeriod: Math.floor(holdingPeriod),
              daysToLTCG: Math.ceil(daysToLTCG),
              currentValue: holding.currentValue,
              purchaseValue: holding.purchaseValue,
              gain: gain,
              gainPercentage: gainPercentage,
              taxImplication: this.calculateTaxImplication(gain, holdingPeriod),
              recommendation: this.generateTaxHarvestingRecommendation(gain, daysToLTCG),
              priority: gainPercentage > 20 ? 'HIGH' : gainPercentage > 10 ? 'MEDIUM' : 'LOW'
            });
          }
        }
      }

      // Sort by priority and days to LTCG
      opportunities.sort((a, b) => {
        if (a.priority !== b.priority) {
          const priorityOrder = { HIGH: 3, MEDIUM: 2, LOW: 1 };
          return priorityOrder[b.priority] - priorityOrder[a.priority];
        }
        return a.daysToLTCG - b.daysToLTCG;
      });

      return opportunities;
    } catch (error) {
      logger.error('Tax harvesting opportunities check failed', { error: error.message });
      return [];
    }
  }

  /**
   * Smart STP/SWP planner for retirees
   */
  async generateSTPSWPPlan(userId) {
    try {
      logger.info('Generating STP/SWP plan', { userId });

      const user = await User.findById(userId);
      const holdings = await Holding.find({ userId, isActive: true });
      const portfolio = await UserPortfolio.findOne({ userId });

      if (!user || !portfolio) {
        throw new Error('User or portfolio not found');
      }

      const plans = {
        stp: [],
        swp: [],
        recommendations: []
      };

      // Generate STP plans for retirees
      if (user.age >= 55) {
        plans.stp = await this.generateSTPPlans(holdings, user, portfolio);
      }

      // Generate SWP plans for retirees
      if (user.age >= 60) {
        plans.swp = await this.generateSWPPlans(holdings, user, portfolio);
      }

      // Generate recommendations
      plans.recommendations = this.generateSTPSWPRecommendations(plans, user);

      return {
        success: true,
        data: plans
      };
    } catch (error) {
      logger.error('STP/SWP plan generation failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to generate STP/SWP plan',
        error: error.message
      };
    }
  }

  /**
   * Alerts on deviation from SIP goals
   */
  async checkSIPGoalDeviations(userId) {
    try {
      logger.info('Checking SIP goal deviations', { userId });

      const user = await User.findById(userId);
      const goals = await Goal.find({ userId, isActive: true });
      const activeSIPs = await SIP.find({ userId, status: 'ACTIVE' });

      if (!user) {
        throw new Error('User not found');
      }

      const deviations = [];

      // Check each goal for SIP deviations
      for (const goal of goals) {
        const sipDeviation = await this.calculateSIPDeviation(goal, activeSIPs, user);
        
        if (Math.abs(sipDeviation.percentage) > this.thresholds.SIP_DEVIATION * 100) {
          deviations.push({
            goalName: goal.name,
            goalTarget: goal.targetAmount,
            currentAmount: goal.currentAmount,
            requiredSIP: sipDeviation.requiredSIP,
            currentSIP: sipDeviation.currentSIP,
            deviation: sipDeviation.percentage,
            monthsRemaining: sipDeviation.monthsRemaining,
            recommendation: sipDeviation.recommendation,
            priority: Math.abs(sipDeviation.percentage) > 20 ? 'HIGH' : 'MEDIUM'
          });
        }
      }

      // Check overall portfolio SIP deviation
      const portfolioDeviation = await this.calculatePortfolioSIPDeviation(activeSIPs, user);
      if (Math.abs(portfolioDeviation.percentage) > this.thresholds.SIP_DEVIATION * 100) {
        deviations.push({
          goalName: 'Portfolio',
          goalTarget: portfolioDeviation.targetAmount,
          currentAmount: portfolioDeviation.currentAmount,
          requiredSIP: portfolioDeviation.requiredSIP,
          currentSIP: portfolioDeviation.currentSIP,
          deviation: portfolioDeviation.percentage,
          monthsRemaining: portfolioDeviation.monthsRemaining,
          recommendation: portfolioDeviation.recommendation,
          priority: Math.abs(portfolioDeviation.percentage) > 20 ? 'HIGH' : 'MEDIUM'
        });
      }

      return {
        success: true,
        data: {
          deviations,
          totalGoals: goals.length,
          activeSIPs: activeSIPs.length,
          deviationThreshold: this.thresholds.SIP_DEVIATION * 100
        }
      };
    } catch (error) {
      logger.error('SIP goal deviation check failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to check SIP goal deviations',
        error: error.message
      };
    }
  }

  /**
   * Expose API for robo-advisory suggestions
   */
  async getRoboAdvisorySuggestions(userId, suggestionType = 'all') {
    try {
      logger.info('Getting robo-advisory suggestions', { userId, suggestionType });

      const suggestions = {};

      if (suggestionType === 'all' || suggestionType === 'portfolio') {
        const review = await this.performPortfolioReview(userId);
        if (review.success) {
          suggestions.portfolio = review.data;
        }
      }

      if (suggestionType === 'all' || suggestionType === 'switches') {
        const switches = await this.generateSwitchRecommendations(userId);
        if (switches.success) {
          suggestions.switches = switches.data;
        }
      }

      if (suggestionType === 'all' || suggestionType === 'tax') {
        const holdings = await Holding.find({ userId, isActive: true });
        suggestions.taxHarvesting = await this.checkTaxHarvestingOpportunities(holdings);
      }

      if (suggestionType === 'all' || suggestionType === 'stpswp') {
        const stpswp = await this.generateSTPSWPPlan(userId);
        if (stpswp.success) {
          suggestions.stpswp = stpswp.data;
        }
      }

      if (suggestionType === 'all' || suggestionType === 'sip') {
        const sip = await this.checkSIPGoalDeviations(userId);
        if (sip.success) {
          suggestions.sip = sip.data;
        }
      }

      return {
        success: true,
        data: suggestions,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Robo-advisory suggestions failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get robo-advisory suggestions',
        error: error.message
      };
    }
  }

  // Helper methods for portfolio review
  async assessPortfolioHealth(holdings, portfolio) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const totalInvested = holdings.reduce((sum, h) => sum + h.purchaseValue, 0);
      const currentGain = totalValue - totalInvested;
      const gainPercentage = totalInvested > 0 ? (currentGain / totalInvested) * 100 : 0;

      // Calculate portfolio metrics
      const metrics = {
        totalValue,
        totalInvested,
        currentGain,
        gainPercentage,
        holdingsCount: holdings.length,
        diversification: this.calculateDiversification(holdings),
        concentration: this.calculateConcentration(holdings),
        performance: await this.calculatePortfolioPerformance(holdings)
      };

      // Determine health status
      let healthStatus = 'GOOD';
      if (gainPercentage < -10) healthStatus = 'POOR';
      else if (gainPercentage < 0) healthStatus = 'FAIR';
      else if (gainPercentage > 20) healthStatus = 'EXCELLENT';

      return {
        status: healthStatus,
        metrics,
        score: this.calculateHealthScore(metrics)
      };
    } catch (error) {
      logger.error('Portfolio health assessment failed', { error: error.message });
      return { status: 'UNKNOWN', metrics: {}, score: 0 };
    }
  }

  async checkRebalancingNeeds(holdings, user) {
    try {
      const needs = [];

      // Get optimal allocation for user
      const optimizationResult = await portfolioOptimizer.optimizePortfolio(
        {
          userId: user._id,
          age: user.age,
          income: user.income,
          taxSlab: user.taxSlab,
          investmentHorizon: user.investmentHorizon
        },
        'MODERATE', // Default risk tolerance
        ['WEALTH_CREATION', 'RETIREMENT'] // Default goals
      );

      if (!optimizationResult.success) {
        return needs;
      }

      const optimalAllocation = optimizationResult.data.optimalAllocation;
      const currentAllocation = this.calculateCurrentAllocation(holdings);

      // Check for deviations
      for (const [category, targetAllocation] of Object.entries(optimalAllocation.categoryBreakdown)) {
        const currentAlloc = currentAllocation[category] || 0;
        const deviation = Math.abs(currentAlloc - targetAllocation);

        if (deviation > this.thresholds.REBALANCING_DEVIATION) {
          needs.push({
            category,
            currentAllocation: currentAlloc,
            targetAllocation,
            deviation,
            action: currentAlloc > targetAllocation ? 'REDUCE' : 'INCREASE',
            priority: deviation > 0.1 ? 'HIGH' : 'MEDIUM',
            recommendation: `Rebalance ${category} allocation from ${(currentAlloc * 100).toFixed(1)}% to ${(targetAllocation * 100).toFixed(1)}%`
          });
        }
      }

      return needs;
    } catch (error) {
      logger.error('Rebalancing needs check failed', { error: error.message });
      return [];
    }
  }

  async assessGoalProgress(goals, portfolio) {
    try {
      const progress = [];

      for (const goal of goals) {
        const monthsRemaining = this.calculateMonthsRemaining(goal.targetDate);
        const progressPercentage = goal.targetAmount > 0 ? (goal.currentAmount / goal.targetAmount) * 100 : 0;
        const requiredMonthlyInvestment = this.calculateRequiredMonthlyInvestment(goal, monthsRemaining);

        progress.push({
          goalName: goal.name,
          targetAmount: goal.targetAmount,
          currentAmount: goal.currentAmount,
          progressPercentage,
          monthsRemaining,
          requiredMonthlyInvestment,
          status: this.getGoalStatus(progressPercentage, monthsRemaining),
          recommendation: this.getGoalRecommendation(progressPercentage, monthsRemaining, requiredMonthlyInvestment)
        });
      }

      return progress;
    } catch (error) {
      logger.error('Goal progress assessment failed', { error: error.message });
      return [];
    }
  }

  async assessRiskLevel(holdings, user) {
    try {
      const riskMetrics = {
        overallRisk: 0,
        volatility: 0,
        concentrationRisk: 0,
        sectorRisk: 0,
        fundSpecificRisk: 0
      };

      // Calculate various risk metrics
      riskMetrics.volatility = this.calculatePortfolioVolatility(holdings);
      riskMetrics.concentrationRisk = this.calculateConcentrationRisk(holdings);
      riskMetrics.sectorRisk = await this.calculateSectorRisk(holdings);
      riskMetrics.fundSpecificRisk = this.calculateFundSpecificRisk(holdings);

      // Calculate overall risk
      riskMetrics.overallRisk = (
        riskMetrics.volatility * 0.3 +
        riskMetrics.concentrationRisk * 0.25 +
        riskMetrics.sectorRisk * 0.25 +
        riskMetrics.fundSpecificRisk * 0.2
      );

      return {
        metrics: riskMetrics,
        level: this.getRiskLevel(riskMetrics.overallRisk),
        recommendation: this.getRiskRecommendation(riskMetrics, user)
      };
    } catch (error) {
      logger.error('Risk assessment failed', { error: error.message });
      return { metrics: {}, level: 'UNKNOWN', recommendation: '' };
    }
  }

  async identifyMarketOpportunities(holdings) {
    try {
      const opportunities = [];

      // Get market analysis
      const marketAnalysis = await marketAnalyticsEngine.performComprehensiveAnalysis();
      
      if (!marketAnalysis.success) {
        return opportunities;
      }

      const analysis = marketAnalysis.data;

      // Check for sector opportunities
      if (analysis.sectorData?.sectorInsights) {
        for (const insight of analysis.sectorData.sectorInsights) {
          opportunities.push({
            type: 'SECTOR_OPPORTUNITY',
            fundName: insight.fundName,
            insight: insight.insight,
            recommendation: insight.recommendation,
            priority: 'MEDIUM'
          });
        }
      }

      // Check for market timing opportunities
      if (analysis.outlook?.equity === 'POSITIVE') {
        opportunities.push({
          type: 'MARKET_OPPORTUNITY',
          insight: 'Market conditions are favorable for equity investments',
          recommendation: 'Consider increasing equity allocation',
          priority: 'LOW'
        });
      }

      return opportunities;
    } catch (error) {
      logger.error('Market opportunities identification failed', { error: error.message });
      return [];
    }
  }

  // Helper methods for switch recommendations
  async analyzeFundForSwitch(holding, user) {
    try {
      const analysis = {
        shouldSwitch: false,
        recommendedFund: null,
        reason: '',
        expectedBenefit: 0,
        switchType: 'PERFORMANCE',
        priority: 'LOW'
      };

      // Check performance
      if (holding.returns1Y < -0.1) {
        analysis.shouldSwitch = true;
        analysis.reason = 'Poor performance over the last year';
        analysis.switchType = 'PERFORMANCE';
        analysis.priority = 'HIGH';
        analysis.expectedBenefit = Math.abs(holding.returns1Y) * 0.5; // Expected 50% improvement
      }

      // Check for better alternatives
      const alternatives = await this.findBetterAlternatives(holding, user);
      if (alternatives.length > 0) {
        const bestAlternative = alternatives[0];
        analysis.shouldSwitch = true;
        analysis.recommendedFund = bestAlternative;
        analysis.reason = `Better performing alternative available: ${bestAlternative.expectedReturn * 100}% vs ${holding.returns1Y * 100}%`;
        analysis.expectedBenefit = bestAlternative.expectedReturn - holding.returns1Y;
        analysis.priority = analysis.expectedBenefit > 0.05 ? 'HIGH' : 'MEDIUM';
      }

      return analysis;
    } catch (error) {
      logger.error('Fund switch analysis failed', { error: error.message });
      return { shouldSwitch: false, reason: 'Analysis failed' };
    }
  }

  async findBetterAlternatives(holding, user) {
    try {
      // In real implementation, query fund database for better alternatives
      const alternatives = [
        {
          fundName: 'Better Alternative Fund',
          category: holding.fundCategory,
          expectedReturn: holding.returns1Y + 0.05,
          risk: 'MODERATE',
          expenseRatio: 0.018
        }
      ];

      return alternatives.sort((a, b) => b.expectedReturn - a.expectedReturn);
    } catch (error) {
      logger.error('Better alternatives search failed', { error: error.message });
      return [];
    }
  }

  // Helper methods for tax harvesting
  calculateTaxImplication(gain, holdingPeriod) {
    try {
      if (holdingPeriod >= 1) {
        // Long-term capital gains
        return {
          type: 'LTCG',
          rate: 0.10,
          taxAmount: gain * 0.10,
          exemption: gain <= 100000 ? gain : 100000
        };
      } else {
        // Short-term capital gains
        return {
          type: 'STCG',
          rate: 0.15,
          taxAmount: gain * 0.15,
          exemption: 0
        };
      }
    } catch (error) {
      logger.error('Tax implication calculation failed', { error: error.message });
      return { type: 'UNKNOWN', rate: 0, taxAmount: 0, exemption: 0 };
    }
  }

  generateTaxHarvestingRecommendation(gain, daysToLTCG) {
    try {
      if (daysToLTCG <= 7) {
        return 'Consider harvesting gains now to benefit from LTCG tax rates';
      } else if (daysToLTCG <= 30) {
        return 'Monitor closely - approaching LTCG eligibility';
      } else {
        return 'Consider holding for LTCG benefits';
      }
    } catch (error) {
      logger.error('Tax harvesting recommendation generation failed', { error: error.message });
      return 'Consult tax advisor';
    }
  }

  // Helper methods for STP/SWP planning
  async generateSTPPlans(holdings, user, portfolio) {
    try {
      const plans = [];

      // Find equity-heavy holdings for STP to debt
      for (const holding of holdings) {
        if (holding.fundCategory === 'Equity' && holding.currentValue > 100000) {
          plans.push({
            fromFund: holding.fundName,
            toFund: 'Liquid Fund',
            amount: holding.currentValue * 0.3, // 30% of holding
            frequency: 'MONTHLY',
            duration: 12, // 12 months
            reason: 'Reduce equity exposure for retirement',
            expectedBenefit: 'Lower volatility and regular income'
          });
        }
      }

      return plans;
    } catch (error) {
      logger.error('STP plan generation failed', { error: error.message });
      return [];
    }
  }

  async generateSWPPlans(holdings, user, portfolio) {
    try {
      const plans = [];

      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const monthlyWithdrawal = totalValue * 0.04 / 12; // 4% rule

      if (monthlyWithdrawal > 10000) {
        plans.push({
          fundName: 'Portfolio SWP',
          monthlyAmount: monthlyWithdrawal,
          frequency: 'MONTHLY',
          reason: 'Regular retirement income',
          taxImplication: 'Taxed as capital gains',
          sustainability: 'Sustainable for 25+ years'
        });
      }

      return plans;
    } catch (error) {
      logger.error('SWP plan generation failed', { error: error.message });
      return [];
    }
  }

  generateSTPSWPRecommendations(plans, user) {
    try {
      const recommendations = [];

      if (plans.stp.length > 0) {
        recommendations.push({
          type: 'STP_RECOMMENDATION',
          priority: 'HIGH',
          message: `Consider ${plans.stp.length} STP plans to reduce equity exposure`,
          action: 'Review STP plans'
        });
      }

      if (plans.swp.length > 0) {
        recommendations.push({
          type: 'SWP_RECOMMENDATION',
          priority: 'MEDIUM',
          message: 'Set up SWP for regular retirement income',
          action: 'Review SWP plans'
        });
      }

      return recommendations;
    } catch (error) {
      logger.error('STP/SWP recommendations generation failed', { error: error.message });
      return [];
    }
  }

  // Helper methods for SIP deviation checking
  async calculateSIPDeviation(goal, activeSIPs, user) {
    try {
      const monthsRemaining = this.calculateMonthsRemaining(goal.targetDate);
      const requiredAmount = goal.targetAmount - goal.currentAmount;
      const requiredSIP = requiredAmount / monthsRemaining;
      
      const currentSIP = activeSIPs.reduce((sum, sip) => sum + sip.amount, 0);
      const percentage = requiredSIP > 0 ? ((currentSIP - requiredSIP) / requiredSIP) * 100 : 0;

      return {
        requiredSIP,
        currentSIP,
        percentage,
        monthsRemaining,
        recommendation: percentage > 0 ? 'SIP is sufficient' : 'Increase SIP amount'
      };
    } catch (error) {
      logger.error('SIP deviation calculation failed', { error: error.message });
      return { requiredSIP: 0, currentSIP: 0, percentage: 0, monthsRemaining: 0, recommendation: '' };
    }
  }

  async calculatePortfolioSIPDeviation(activeSIPs, user) {
    try {
      const monthlyIncome = user.income / 12;
      const recommendedSIP = monthlyIncome * 0.3; // 30% of monthly income
      const currentSIP = activeSIPs.reduce((sum, sip) => sum + sip.amount, 0);
      const percentage = recommendedSIP > 0 ? ((currentSIP - recommendedSIP) / recommendedSIP) * 100 : 0;

      return {
        requiredSIP: recommendedSIP,
        currentSIP,
        percentage,
        targetAmount: recommendedSIP * 12,
        currentAmount: currentSIP * 12,
        monthsRemaining: 12,
        recommendation: percentage > 0 ? 'SIP is sufficient' : 'Consider increasing SIP'
      };
    } catch (error) {
      logger.error('Portfolio SIP deviation calculation failed', { error: error.message });
      return { requiredSIP: 0, currentSIP: 0, percentage: 0, targetAmount: 0, currentAmount: 0, monthsRemaining: 0, recommendation: '' };
    }
  }

  // Utility methods
  calculateDiversification(holdings) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const categories = new Set(holdings.map(h => h.fundCategory));
      return categories.size / Math.max(holdings.length, 1);
    } catch (error) {
      logger.error('Diversification calculation failed', { error: error.message });
      return 0;
    }
  }

  calculateConcentration(holdings) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const maxHolding = Math.max(...holdings.map(h => h.currentValue));
      return maxHolding / totalValue;
    } catch (error) {
      logger.error('Concentration calculation failed', { error: error.message });
      return 0;
    }
  }

  async calculatePortfolioPerformance(holdings) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const totalInvested = holdings.reduce((sum, h) => sum + h.purchaseValue, 0);
      return totalInvested > 0 ? (totalValue - totalInvested) / totalInvested : 0;
    } catch (error) {
      logger.error('Portfolio performance calculation failed', { error: error.message });
      return 0;
    }
  }

  calculateHealthScore(metrics) {
    try {
      let score = 0;
      score += Math.min(100, Math.max(0, metrics.gainPercentage + 50)); // 50% weight
      score += Math.min(100, metrics.diversification * 100) * 0.3; // 30% weight
      score += Math.min(100, (1 - metrics.concentration) * 100) * 0.2; // 20% weight
      return Math.round(score);
    } catch (error) {
      logger.error('Health score calculation failed', { error: error.message });
      return 0;
    }
  }

  calculateCurrentAllocation(holdings) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const allocation = {};

      holdings.forEach(holding => {
        const category = holding.fundCategory || 'others';
        allocation[category] = (allocation[category] || 0) + (holding.currentValue / totalValue);
      });

      return allocation;
    } catch (error) {
      logger.error('Current allocation calculation failed', { error: error.message });
      return {};
    }
  }

  calculateMonthsRemaining(targetDate) {
    try {
      const now = new Date();
      const target = new Date(targetDate);
      const diffTime = target - now;
      const diffMonths = Math.ceil(diffTime / (1000 * 60 * 60 * 24 * 30));
      return Math.max(0, diffMonths);
    } catch (error) {
      logger.error('Months remaining calculation failed', { error: error.message });
      return 0;
    }
  }

  calculateRequiredMonthlyInvestment(goal, monthsRemaining) {
    try {
      const requiredAmount = goal.targetAmount - goal.currentAmount;
      return monthsRemaining > 0 ? requiredAmount / monthsRemaining : 0;
    } catch (error) {
      logger.error('Required monthly investment calculation failed', { error: error.message });
      return 0;
    }
  }

  getGoalStatus(progressPercentage, monthsRemaining) {
    try {
      if (progressPercentage >= 100) return 'COMPLETED';
      if (progressPercentage >= 80) return 'ON_TRACK';
      if (monthsRemaining <= 12) return 'AT_RISK';
      if (progressPercentage >= 60) return 'NEEDS_ATTENTION';
      return 'AT_RISK';
    } catch (error) {
      logger.error('Goal status determination failed', { error: error.message });
      return 'UNKNOWN';
    }
  }

  getGoalRecommendation(progressPercentage, monthsRemaining, requiredMonthlyInvestment) {
    try {
      if (progressPercentage >= 100) return 'Goal completed successfully!';
      if (progressPercentage >= 80) return 'Continue current investment strategy';
      if (monthsRemaining <= 12) return `Increase monthly investment to ₹${requiredMonthlyInvestment.toLocaleString()}`;
      return `Consider increasing monthly investment to ₹${requiredMonthlyInvestment.toLocaleString()}`;
    } catch (error) {
      logger.error('Goal recommendation generation failed', { error: error.message });
      return 'Review goal progress with advisor';
    }
  }

  calculatePortfolioVolatility(holdings) {
    try {
      // Simplified volatility calculation
      const returns = holdings.map(h => h.returns1Y || 0);
      const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
      const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
      return Math.sqrt(variance);
    } catch (error) {
      logger.error('Portfolio volatility calculation failed', { error: error.message });
      return 0;
    }
  }

  calculateConcentrationRisk(holdings) {
    try {
      return this.calculateConcentration(holdings);
    } catch (error) {
      logger.error('Concentration risk calculation failed', { error: error.message });
      return 0;
    }
  }

  async calculateSectorRisk(holdings) {
    try {
      // Mock sector risk calculation
      return 0.3;
    } catch (error) {
      logger.error('Sector risk calculation failed', { error: error.message });
      return 0;
    }
  }

  calculateFundSpecificRisk(holdings) {
    try {
      // Mock fund-specific risk calculation
      return 0.2;
    } catch (error) {
      logger.error('Fund-specific risk calculation failed', { error: error.message });
      return 0;
    }
  }

  getRiskLevel(riskScore) {
    try {
      if (riskScore < 0.3) return 'LOW';
      if (riskScore < 0.6) return 'MODERATE';
      return 'HIGH';
    } catch (error) {
      logger.error('Risk level determination failed', { error: error.message });
      return 'UNKNOWN';
    }
  }

  getRiskRecommendation(riskMetrics, user) {
    try {
      if (riskMetrics.overallRisk > 0.7) {
        return 'Consider reducing portfolio risk through rebalancing';
      } else if (riskMetrics.overallRisk < 0.3) {
        return 'Consider increasing equity exposure for better returns';
      }
      return 'Portfolio risk is well-balanced';
    } catch (error) {
      logger.error('Risk recommendation generation failed', { error: error.message });
      return 'Consult financial advisor';
    }
  }

  // Storage methods
  async storeReviewResults(userId, results) {
    try {
      // In real implementation, store in database
      logger.info('Review results stored', { userId, timestamp: results.reviewDate });
    } catch (error) {
      logger.error('Review results storage failed', { error: error.message });
    }
  }

  async generateRecommendations(reviewResults, user) {
    try {
      const recommendations = [];

      // Portfolio health recommendations
      if (reviewResults.portfolioHealth.status === 'POOR') {
        recommendations.push({
          type: this.recommendationTypes.RISK_REDUCE,
          priority: 'HIGH',
          message: 'Portfolio health is poor. Consider rebalancing and risk reduction.',
          action: 'Review portfolio allocation'
        });
      }

      // Rebalancing recommendations
      if (reviewResults.rebalancingNeeds.length > 0) {
        recommendations.push({
          type: this.recommendationTypes.REBALANCE,
          priority: 'MEDIUM',
          message: `${reviewResults.rebalancingNeeds.length} categories need rebalancing`,
          action: 'Review rebalancing needs'
        });
      }

      // Tax harvesting recommendations
      if (reviewResults.taxOpportunities.length > 0) {
        recommendations.push({
          type: this.recommendationTypes.TAX_HARVEST,
          priority: 'MEDIUM',
          message: `${reviewResults.taxOpportunities.length} tax harvesting opportunities available`,
          action: 'Review tax opportunities'
        });
      }

      return recommendations;
    } catch (error) {
      logger.error('Recommendations generation failed', { error: error.message });
      return [];
    }
  }

  async generateAlerts(reviewResults, user) {
    try {
      const alerts = [];

      // High-risk alerts
      if (reviewResults.riskAssessment.level === 'HIGH') {
        alerts.push({
          type: 'HIGH_RISK',
          severity: 'HIGH',
          message: 'Portfolio risk level is high',
          action: 'Consider risk reduction strategies'
        });
      }

      // Goal deviation alerts
      const atRiskGoals = reviewResults.goalProgress.filter(g => g.status === 'AT_RISK');
      if (atRiskGoals.length > 0) {
        alerts.push({
          type: 'GOAL_DEVIATION',
          severity: 'MEDIUM',
          message: `${atRiskGoals.length} goals are at risk`,
          action: 'Review goal progress'
        });
      }

      return alerts;
    } catch (error) {
      logger.error('Alerts generation failed', { error: error.message });
      return [];
    }
  }
}

module.exports = new RoboAdvisor(); 