const logger = require('../utils/logger');
const { User, UserPortfolio, Holding, Goal, Transaction, Notification } = require('../models');
const portfolioOptimizer = require('./portfolioOptimizer');
const predictiveEngine = require('./predictiveEngine');

class DashboardEngine {
  constructor() {
    this.widgetTypes = {
      PORTFOLIO_SUMMARY: 'portfolio_summary',
      GOAL_PROGRESS: 'goal_progress',
      PERFORMANCE_CHART: 'performance_chart',
      ASSET_ALLOCATION: 'asset_allocation',
      RECENT_TRANSACTIONS: 'recent_transactions',
      AI_INSIGHTS: 'ai_insights',
      TAX_REMINDERS: 'tax_reminders',
      MARKET_ALERTS: 'market_alerts',
      SIP_STATUS: 'sip_status',
      RECOMMENDATIONS: 'recommendations'
    };
  }

  /**
   * Get comprehensive dashboard data for a user
   */
  async getDashboardData(userId, includeWidgets = true) {
    try {
      logger.info('Generating dashboard data for user', { userId });

      // Get user profile and portfolio
      const user = await User.findById(userId);
      const portfolio = await UserPortfolio.findOne({ userId });
      const holdings = await Holding.find({ userId, isActive: true });
      const goals = await Goal.find({ userId, isActive: true });

      if (!user) {
        throw new Error('User not found');
      }

      // Calculate portfolio metrics
      const portfolioMetrics = await this.calculatePortfolioMetrics(holdings, portfolio);

      // Calculate goal progress
      const goalProgress = await this.calculateGoalProgress(goals, portfolioMetrics.totalValue);

      // Get AI insights
      const aiInsights = await this.generateAIInsights(user, holdings, portfolio);

      // Get alerts and notifications
      const alerts = await this.getAlerts(userId, holdings, portfolio);

      // Get tax reminders
      const taxReminders = await this.getTaxReminders(user, holdings);

      // Get market alerts
      const marketAlerts = await this.getMarketAlerts();

      // Get recent transactions
      const recentTransactions = await this.getRecentTransactions(userId);

      // Get SIP status
      const sipStatus = await this.getSIPStatus(userId);

      // Get recommendations
      const recommendations = await this.getRecommendations(user, holdings, portfolio);

      const dashboardData = {
        success: true,
        data: {
          user: {
            name: user.name,
            email: user.email,
            phone: user.phone,
            age: user.age,
            income: user.income,
            investmentHorizon: user.investmentHorizon
          },
          portfolio: portfolioMetrics,
          goals: goalProgress,
          aiInsights,
          alerts,
          taxReminders,
          marketAlerts,
          recentTransactions,
          sipStatus,
          recommendations,
          lastUpdated: new Date().toISOString()
        }
      };

      // Add widgets if requested
      if (includeWidgets) {
        dashboardData.data.widgets = await this.generateWidgets(userId, dashboardData.data);
      }

      logger.info('Dashboard data generated successfully', { userId });

      return dashboardData;
    } catch (error) {
      logger.error('Error generating dashboard data', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to generate dashboard data',
        error: error.message
      };
    }
  }

  /**
   * Get specific widget data
   */
  async getWidgetData(userId, widgetType, params = {}) {
    try {
      logger.info('Generating widget data', { userId, widgetType });

      switch (widgetType) {
        case this.widgetTypes.PORTFOLIO_SUMMARY:
          return await this.getPortfolioSummaryWidget(userId);

        case this.widgetTypes.GOAL_PROGRESS:
          return await this.getGoalProgressWidget(userId, params.goalId);

        case this.widgetTypes.PERFORMANCE_CHART:
          return await this.getPerformanceChartWidget(userId, params.period);

        case this.widgetTypes.ASSET_ALLOCATION:
          return await this.getAssetAllocationWidget(userId);

        case this.widgetTypes.RECENT_TRANSACTIONS:
          return await this.getRecentTransactionsWidget(userId, params.limit);

        case this.widgetTypes.AI_INSIGHTS:
          return await this.getAIInsightsWidget(userId);

        case this.widgetTypes.TAX_REMINDERS:
          return await this.getTaxRemindersWidget(userId);

        case this.widgetTypes.MARKET_ALERTS:
          return await this.getMarketAlertsWidget();

        case this.widgetTypes.SIP_STATUS:
          return await this.getSIPStatusWidget(userId);

        case this.widgetTypes.RECOMMENDATIONS:
          return await this.getRecommendationsWidget(userId);

        default:
          throw new Error(`Unknown widget type: ${widgetType}`);
      }
    } catch (error) {
      logger.error('Error generating widget data', { error: error.message, userId, widgetType });
      return {
        success: false,
        message: 'Failed to generate widget data',
        error: error.message
      };
    }
  }

  /**
   * Calculate comprehensive portfolio metrics
   */
  async calculatePortfolioMetrics(holdings, portfolio) {
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const totalInvested = holdings.reduce((sum, h) => sum + h.purchaseValue, 0);
      const currentGain = totalValue - totalInvested;
      const gainPercentage = totalInvested > 0 ? (currentGain / totalInvested) * 100 : 0;

      // Calculate XIRR
      const xirr = await this.calculateXIRR(holdings);

      // Calculate CAGR
      const cagr = await this.calculateCAGR(holdings);

      // Calculate asset allocation
      const assetAllocation = this.calculateAssetAllocation(holdings);

      // Calculate performance metrics
      const performance = await this.calculatePerformanceMetrics(holdings, portfolio);

      // Calculate risk metrics
      const riskMetrics = await this.calculateRiskMetrics(holdings);

      return {
        totalValue,
        totalInvested,
        currentGain,
        gainPercentage,
        xirr,
        cagr,
        assetAllocation,
        performance,
        riskMetrics,
        holdingsCount: holdings.length,
        lastUpdated: new Date().toISOString()
      };
    } catch (error) {
      logger.error('Error calculating portfolio metrics', { error: error.message });
      throw error;
    }
  }

  /**
   * Calculate goal progress
   */
  async calculateGoalProgress(goals, currentPortfolioValue) {
    try {
      const goalProgress = [];

      for (const goal of goals) {
        const progress = {
          goalId: goal._id,
          goalName: goal.name,
          targetAmount: goal.targetAmount,
          targetDate: goal.targetDate,
          currentAmount: goal.currentAmount || 0,
          portfolioContribution: Math.min(currentPortfolioValue * 0.8, goal.targetAmount - goal.currentAmount),
          progressPercentage: goal.targetAmount > 0 ? ((goal.currentAmount + Math.min(currentPortfolioValue * 0.8, goal.targetAmount - goal.currentAmount)) / goal.targetAmount) * 100 : 0,
          monthsRemaining: this.calculateMonthsRemaining(goal.targetDate),
          status: this.getGoalStatus(goal, currentPortfolioValue),
          recommendations: this.getGoalRecommendations(goal, currentPortfolioValue)
        };

        goalProgress.push(progress);
      }

      return goalProgress;
    } catch (error) {
      logger.error('Error calculating goal progress', { error: error.message });
      return [];
    }
  }

  /**
   * Generate AI insights
   */
  async generateAIInsights(user, holdings, portfolio) {
    try {
      const insights = [];

      // Portfolio optimization insights
      if (holdings.length > 0) {
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

        if (optimizationResult.success) {
          insights.push({
            type: 'PORTFOLIO_OPTIMIZATION',
            title: 'Portfolio Optimization Opportunity',
            message: 'Your portfolio could be optimized for better risk-adjusted returns',
            priority: 'MEDIUM',
            action: 'View optimization suggestions',
            data: optimizationResult.data
          });
        }
      }

      // Market timing insights
      const marketTrends = await predictiveEngine.predictMarketTrends('3M');
      if (marketTrends.success && marketTrends.data.signals.overall === 'BULLISH') {
        insights.push({
          type: 'MARKET_OPPORTUNITY',
          title: 'Market Opportunity',
          message: 'Current market conditions are favorable for equity investments',
          priority: 'LOW',
          action: 'Consider increasing equity allocation',
          data: marketTrends.data
        });
      }

      // Tax optimization insights
      const taxInsights = await this.generateTaxInsights(user, holdings);
      insights.push(...taxInsights);

      return insights;
    } catch (error) {
      logger.error('Error generating AI insights', { error: error.message });
      return [];
    }
  }

  /**
   * Get alerts and notifications
   */
  async getAlerts(userId, holdings, portfolio) {
    try {
      const alerts = [];

      // Check for underperforming funds
      for (const holding of holdings) {
        if (holding.returns1Y < -0.1) {
          alerts.push({
            type: 'UNDERPERFORMING_FUND',
            severity: 'HIGH',
            title: 'Underperforming Fund Alert',
            message: `${holding.fundName} has underperformed by ${Math.abs(holding.returns1Y * 100).toFixed(1)}% in the last year`,
            action: 'Consider switching to better performing funds',
            fundName: holding.fundName,
            returns1Y: holding.returns1Y
          });
        }
      }

      // Check for portfolio rebalancing needs
      if (holdings.length > 0) {
        const assetAllocation = this.calculateAssetAllocation(holdings);
        const rebalancingAlerts = this.checkRebalancingNeeds(assetAllocation);
        alerts.push(...rebalancingAlerts);
      }

      // Check for goal deadlines
      const goals = await Goal.find({ userId, isActive: true });
      for (const goal of goals) {
        const monthsRemaining = this.calculateMonthsRemaining(goal.targetDate);
        if (monthsRemaining <= 12) {
          alerts.push({
            type: 'GOAL_DEADLINE',
            severity: monthsRemaining <= 6 ? 'HIGH' : 'MEDIUM',
            title: 'Goal Deadline Approaching',
            message: `${goal.name} target date is ${monthsRemaining} months away`,
            action: 'Review goal progress and adjust investments',
            goalName: goal.name,
            monthsRemaining
          });
        }
      }

      return alerts;
    } catch (error) {
      logger.error('Error getting alerts', { error: error.message });
      return [];
    }
  }

  /**
   * Get tax reminders
   */
  async getTaxReminders(user, holdings) {
    try {
      const reminders = [];

      // ELSS investment reminder
      if (user.taxSlab > 0.1) {
        const currentYear = new Date().getFullYear();
        const elssInvested = await this.getELSSInvestmentAmount(user._id, currentYear);
        const maxELSS = 150000;
        
        if (elssInvested < maxELSS) {
          reminders.push({
            type: 'ELSS_INVESTMENT',
            title: 'ELSS Investment Reminder',
            message: `You can invest ₹${(maxELSS - elssInvested).toLocaleString()} more in ELSS for tax deduction`,
            amount: maxELSS - elssInvested,
            taxSaving: (maxELSS - elssInvested) * user.taxSlab,
            priority: 'HIGH'
          });
        }
      }

      // LTCG harvesting reminder
      const ltcgOpportunities = await this.findLTCGOpportunities(holdings);
      if (ltcgOpportunities.length > 0) {
        reminders.push({
          type: 'LTCG_HARVESTING',
          title: 'LTCG Harvesting Opportunity',
          message: `${ltcgOpportunities.length} funds are approaching 1-year holding period`,
          opportunities: ltcgOpportunities,
          priority: 'MEDIUM'
        });
      }

      return reminders;
    } catch (error) {
      logger.error('Error getting tax reminders', { error: error.message });
      return [];
    }
  }

  /**
   * Get market alerts
   */
  async getMarketAlerts() {
    try {
      const alerts = [];

      // Get market data
      const marketData = await this.getMarketData();
      
      if (marketData.volatility > 0.25) {
        alerts.push({
          type: 'HIGH_VOLATILITY',
          title: 'High Market Volatility',
          message: 'Markets are experiencing high volatility. Consider reducing risk exposure.',
          severity: 'MEDIUM',
          volatility: marketData.volatility
        });
      }

      if (marketData.trend === 'BEARISH') {
        alerts.push({
          type: 'BEARISH_MARKET',
          title: 'Bearish Market Conditions',
          message: 'Market conditions are bearish. Consider defensive positioning.',
          severity: 'HIGH',
          trend: marketData.trend
        });
      }

      return alerts;
    } catch (error) {
      logger.error('Error getting market alerts', { error: error.message });
      return [];
    }
  }

  /**
   * Get recent transactions
   */
  async getRecentTransactions(userId, limit = 10) {
    try {
      const transactions = await Transaction.find({ userId })
        .sort({ date: -1 })
        .limit(limit)
        .populate('fundId', 'fundName schemeCode');

      return transactions.map(t => ({
        id: t._id,
        type: t.type,
        amount: t.amount,
        units: t.units,
        nav: t.nav,
        date: t.date,
        fundName: t.fundId?.fundName || 'Unknown Fund',
        schemeCode: t.fundId?.schemeCode || 'Unknown'
      }));
    } catch (error) {
      logger.error('Error getting recent transactions', { error: error.message });
      return [];
    }
  }

  /**
   * Get SIP status
   */
  async getSIPStatus(userId) {
    try {
      // In real implementation, fetch from SIP database
      const activeSIPs = [
        {
          fundName: 'Axis Bluechip Fund',
          amount: 5000,
          frequency: 'MONTHLY',
          nextDate: '2024-02-01',
          status: 'ACTIVE'
        }
      ];

      const totalSIPAmount = activeSIPs.reduce((sum, sip) => sum + sip.amount, 0);

      return {
        activeSIPs,
        totalSIPAmount,
        sipCount: activeSIPs.length,
        nextSIPDate: activeSIPs.length > 0 ? activeSIPs[0].nextDate : null
      };
    } catch (error) {
      logger.error('Error getting SIP status', { error: error.message });
      return {
        activeSIPs: [],
        totalSIPAmount: 0,
        sipCount: 0,
        nextSIPDate: null
      };
    }
  }

  /**
   * Get recommendations
   */
  async getRecommendations(user, holdings, portfolio) {
    try {
      const recommendations = [];

      // Portfolio optimization recommendations
      if (holdings.length > 0) {
        recommendations.push({
          type: 'PORTFOLIO_OPTIMIZATION',
          title: 'Optimize Your Portfolio',
          description: 'Get AI-powered recommendations to optimize your portfolio',
          priority: 'HIGH',
          action: 'View optimization suggestions'
        });
      }

      // Goal-based recommendations
      const goals = await Goal.find({ userId, isActive: true });
      for (const goal of goals) {
        const monthsRemaining = this.calculateMonthsRemaining(goal.targetDate);
        if (monthsRemaining <= 12) {
          recommendations.push({
            type: 'GOAL_ACCELERATION',
            title: `Accelerate ${goal.name}`,
            description: `Your ${goal.name} goal is ${monthsRemaining} months away. Consider increasing investments.`,
            priority: 'MEDIUM',
            action: 'Increase SIP amount'
          });
        }
      }

      // Tax optimization recommendations
      if (user.taxSlab > 0.1) {
        recommendations.push({
          type: 'TAX_OPTIMIZATION',
          title: 'Tax Optimization',
          description: 'Optimize your investments for better tax efficiency',
          priority: 'MEDIUM',
          action: 'View tax strategies'
        });
      }

      return recommendations;
    } catch (error) {
      logger.error('Error getting recommendations', { error: error.message });
      return [];
    }
  }

  /**
   * Generate widgets
   */
  async generateWidgets(userId, dashboardData) {
    try {
      const widgets = [];

      // Portfolio Summary Widget
      widgets.push({
        type: this.widgetTypes.PORTFOLIO_SUMMARY,
        title: 'Portfolio Summary',
        data: {
          totalValue: dashboardData.portfolio.totalValue,
          totalInvested: dashboardData.portfolio.totalInvested,
          currentGain: dashboardData.portfolio.currentGain,
          gainPercentage: dashboardData.portfolio.gainPercentage,
          xirr: dashboardData.portfolio.xirr,
          cagr: dashboardData.portfolio.cagr
        },
        position: { x: 0, y: 0, w: 6, h: 2 }
      });

      // Goal Progress Widget
      if (dashboardData.goals.length > 0) {
        widgets.push({
          type: this.widgetTypes.GOAL_PROGRESS,
          title: 'Goal Progress',
          data: dashboardData.goals,
          position: { x: 6, y: 0, w: 6, h: 2 }
        });
      }

      // Asset Allocation Widget
      widgets.push({
        type: this.widgetTypes.ASSET_ALLOCATION,
        title: 'Asset Allocation',
        data: dashboardData.portfolio.assetAllocation,
        position: { x: 0, y: 2, w: 4, h: 3 }
      });

      // Performance Chart Widget
      widgets.push({
        type: this.widgetTypes.PERFORMANCE_CHART,
        title: 'Portfolio Performance',
        data: dashboardData.portfolio.performance,
        position: { x: 4, y: 2, w: 8, h: 3 }
      });

      // AI Insights Widget
      if (dashboardData.aiInsights.length > 0) {
        widgets.push({
          type: this.widgetTypes.AI_INSIGHTS,
          title: 'AI Insights',
          data: dashboardData.aiInsights,
          position: { x: 0, y: 5, w: 6, h: 2 }
        });
      }

      // Recent Transactions Widget
      if (dashboardData.recentTransactions.length > 0) {
        widgets.push({
          type: this.widgetTypes.RECENT_TRANSACTIONS,
          title: 'Recent Transactions',
          data: dashboardData.recentTransactions,
          position: { x: 6, y: 5, w: 6, h: 2 }
        });
      }

      return widgets;
    } catch (error) {
      logger.error('Error generating widgets', { error: error.message });
      return [];
    }
  }

  // Widget-specific methods
  async getPortfolioSummaryWidget(userId) {
    const portfolio = await UserPortfolio.findOne({ userId });
    const holdings = await Holding.find({ userId, isActive: true });
    const metrics = await this.calculatePortfolioMetrics(holdings, portfolio);

    return {
      success: true,
      data: {
        totalValue: metrics.totalValue,
        totalInvested: metrics.totalInvested,
        currentGain: metrics.currentGain,
        gainPercentage: metrics.gainPercentage,
        xirr: metrics.xirr,
        cagr: metrics.cagr,
        holdingsCount: metrics.holdingsCount
      }
    };
  }

  async getGoalProgressWidget(userId, goalId = null) {
    const goals = await Goal.find({ userId, isActive: true });
    const portfolio = await UserPortfolio.findOne({ userId });
    const goalProgress = await this.calculateGoalProgress(goals, portfolio?.totalValue || 0);

    if (goalId) {
      return {
        success: true,
        data: goalProgress.find(g => g.goalId.toString() === goalId) || null
      };
    }

    return {
      success: true,
      data: goalProgress
    };
  }

  async getPerformanceChartWidget(userId, period = '1Y') {
    const holdings = await Holding.find({ userId, isActive: true });
    const performance = await this.calculatePerformanceMetrics(holdings, null, period);

    return {
      success: true,
      data: {
        period,
        data: performance.chartData,
        benchmark: performance.benchmark
      }
    };
  }

  async getAssetAllocationWidget(userId) {
    const holdings = await Holding.find({ userId, isActive: true });
    const assetAllocation = this.calculateAssetAllocation(holdings);

    return {
      success: true,
      data: assetAllocation
    };
  }

  async getRecentTransactionsWidget(userId, limit = 5) {
    const transactions = await this.getRecentTransactions(userId, limit);

    return {
      success: true,
      data: transactions
    };
  }

  async getAIInsightsWidget(userId) {
    const user = await User.findById(userId);
    const holdings = await Holding.find({ userId, isActive: true });
    const portfolio = await UserPortfolio.findOne({ userId });
    const insights = await this.generateAIInsights(user, holdings, portfolio);

    return {
      success: true,
      data: insights
    };
  }

  async getTaxRemindersWidget(userId) {
    const user = await User.findById(userId);
    const holdings = await Holding.find({ userId, isActive: true });
    const reminders = await this.getTaxReminders(user, holdings);

    return {
      success: true,
      data: reminders
    };
  }

  async getMarketAlertsWidget() {
    const alerts = await this.getMarketAlerts();

    return {
      success: true,
      data: alerts
    };
  }

  async getSIPStatusWidget(userId) {
    const sipStatus = await this.getSIPStatus(userId);

    return {
      success: true,
      data: sipStatus
    };
  }

  async getRecommendationsWidget(userId) {
    const user = await User.findById(userId);
    const holdings = await Holding.find({ userId, isActive: true });
    const portfolio = await UserPortfolio.findOne({ userId });
    const recommendations = await this.getRecommendations(user, holdings, portfolio);

    return {
      success: true,
      data: recommendations
    };
  }

  // Helper methods
  async calculateXIRR(holdings) {
    // Simplified XIRR calculation
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const totalInvested = holdings.reduce((sum, h) => sum + h.purchaseValue, 0);
      
      if (totalInvested === 0) return 0;
      
      // Simple approximation
      const timeWeightedReturn = (totalValue - totalInvested) / totalInvested;
      return timeWeightedReturn * 100; // Convert to percentage
    } catch (error) {
      logger.error('Error calculating XIRR', { error: error.message });
      return 0;
    }
  }

  async calculateCAGR(holdings) {
    // Simplified CAGR calculation
    try {
      const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
      const totalInvested = holdings.reduce((sum, h) => sum + h.purchaseValue, 0);
      
      if (totalInvested === 0) return 0;
      
      // Assume average holding period of 2 years
      const years = 2;
      const cagr = Math.pow(totalValue / totalInvested, 1 / years) - 1;
      return cagr * 100; // Convert to percentage
    } catch (error) {
      logger.error('Error calculating CAGR', { error: error.message });
      return 0;
    }
  }

  calculateAssetAllocation(holdings) {
    const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
    const allocation = {};

    holdings.forEach(holding => {
      const category = holding.fundCategory || 'others';
      allocation[category] = (allocation[category] || 0) + (holding.currentValue / totalValue);
    });

    return allocation;
  }

  async calculatePerformanceMetrics(holdings, portfolio, period = '1Y') {
    // Mock performance data
    return {
      returns: {
        '1M': 0.05,
        '3M': 0.15,
        '6M': 0.25,
        '1Y': 0.35,
        '3Y': 0.45
      },
      benchmark: {
        '1M': 0.04,
        '3M': 0.12,
        '6M': 0.20,
        '1Y': 0.30,
        '3Y': 0.40
      },
      chartData: [
        { date: '2024-01-01', value: 1000000 },
        { date: '2024-02-01', value: 1050000 },
        { date: '2024-03-01', value: 1100000 },
        { date: '2024-04-01', value: 1080000 },
        { date: '2024-05-01', value: 1150000 }
      ]
    };
  }

  async calculateRiskMetrics(holdings) {
    return {
      overallRisk: 0.5,
      volatility: 0.15,
      sharpeRatio: 1.2,
      beta: 0.9,
      maxDrawdown: 0.10
    };
  }

  calculateMonthsRemaining(targetDate) {
    const now = new Date();
    const target = new Date(targetDate);
    const diffTime = target - now;
    const diffMonths = Math.ceil(diffTime / (1000 * 60 * 60 * 24 * 30));
    return Math.max(0, diffMonths);
  }

  getGoalStatus(goal, currentPortfolioValue) {
    const progress = (goal.currentAmount + Math.min(currentPortfolioValue * 0.8, goal.targetAmount - goal.currentAmount)) / goal.targetAmount;
    
    if (progress >= 1) return 'COMPLETED';
    if (progress >= 0.8) return 'ON_TRACK';
    if (progress >= 0.6) return 'NEEDS_ATTENTION';
    return 'AT_RISK';
  }

  getGoalRecommendations(goal, currentPortfolioValue) {
    const recommendations = [];
    
    if (goal.currentAmount < goal.targetAmount * 0.5) {
      recommendations.push('Increase monthly investment amount');
    }
    
    if (this.calculateMonthsRemaining(goal.targetDate) <= 12) {
      recommendations.push('Consider extending target date or increasing investment');
    }
    
    return recommendations;
  }

  async generateTaxInsights(user, holdings) {
    const insights = [];
    
    // ELSS investment insight
    if (user.taxSlab > 0.1) {
      insights.push({
        type: 'TAX_OPTIMIZATION',
        title: 'Tax Saving Opportunity',
        message: 'Consider ELSS investments for tax deduction under Section 80C',
        priority: 'MEDIUM',
        action: 'View ELSS funds'
      });
    }
    
    return insights;
  }

  checkRebalancingNeeds(assetAllocation) {
    const alerts = [];
    
    // Check for over-concentration in any category
    for (const [category, allocation] of Object.entries(assetAllocation)) {
      if (allocation > 0.4) {
        alerts.push({
          type: 'REBALANCING_NEEDED',
          severity: 'MEDIUM',
          title: 'Portfolio Rebalancing',
          message: `Your ${category} allocation is ${(allocation * 100).toFixed(1)}%. Consider rebalancing.`,
          action: 'View rebalancing suggestions'
        });
      }
    }
    
    return alerts;
  }

  async getELSSInvestmentAmount(userId, year) {
    // Mock ELSS investment amount
    return 50000;
  }

  async findLTCGOpportunities(holdings) {
    const opportunities = [];
    const now = new Date();

    holdings.forEach(holding => {
      const holdingPeriod = (now - new Date(holding.purchaseDate)) / (1000 * 60 * 60 * 24 * 365);
      
      if (holdingPeriod >= 0.9 && holdingPeriod < 1.1) {
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

  async getMarketData() {
    return {
      volatility: 0.15,
      trend: 'BULLISH'
    };
  }
}

module.exports = new DashboardEngine(); 