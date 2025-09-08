const logger = require('../utils/logger');
const navHistoryService = require('./navHistoryService');
const riskProfilingService = require('./riskProfilingService');

class FundComparisonService {
  /**
   * Comprehensive fund comparison with detailed analysis and ratings
   */
  async compareFunds({
    fundCodes,
    category = null,
    period = '1y',
    investmentAmount = 100000,
    includeRatings = true,
    includeRecommendations = true
  }) {
    try {
      logger.info(`Starting comprehensive fund comparison for: ${fundCodes.join(', ')}`);

      // Validate category consistency
      if (category) {
        await this.validateCategoryConsistency(fundCodes, category);
      }

      // Get detailed fund data
      const fundData = await this.getFundData(fundCodes, period);
      
      // Perform comprehensive analysis
      const analysis = await this.performComprehensiveAnalysis(fundData, period, investmentAmount);
      
      // Calculate ratings if requested
      if (includeRatings) {
        analysis.ratings = await this.calculateFundRatings(analysis);
      }

      // Generate recommendations if requested
      if (includeRecommendations) {
        analysis.recommendations = await this.generateRecommendations(analysis);
      }

      // Add comparison summary
      analysis.summary = this.generateComparisonSummary(analysis);

      return analysis;
    } catch (error) {
      logger.error('Error in fund comparison:', error);
      throw error;
    }
  }

  /**
   * Validate that all funds belong to the same category
   */
  async validateCategoryConsistency(fundCodes, category) {
    // This would typically check against a fund database
    // For now, we'll assume validation passes
    logger.info(`Validating category consistency for ${category} funds`);
    return true;
  }

  /**
   * Get comprehensive fund data for comparison
   */
  async getFundData(fundCodes, period) {
    const fundData = [];

    for (const fundCode of fundCodes) {
      try {
        // Get fund details (this would come from a fund database)
        const fundDetails = await this.getFundDetails(fundCode);

        // Generate mock NAV history for testing
        const navHistory = this.generateMockNavHistory(fundCode, period);

        // Get risk metrics
        const riskMetrics = await this.calculateRiskMetrics(navHistory);

        // Generate mock performance data
        const performance = this.generateMockPerformance(fundCode, period);

        fundData.push({
          fundCode,
          fundDetails,
          navHistory,
          riskMetrics,
          performance
        });
      } catch (error) {
        logger.error(`Error getting data for fund ${fundCode}:`, error);
        throw new Error(`Failed to get data for fund ${fundCode}`);
      }
    }

    return fundData;
  }

  /**
   * Generate mock NAV history for testing
   */
  generateMockNavHistory(fundCode, period) {
    const days = period === '1y' ? 365 : period === '3y' ? 1095 : 1825;
    const navHistory = [];
    let baseNav = 40 + Math.random() * 20; // Random base NAV between 40-60

    for (let i = 0; i < days; i++) {
      const date = new Date();
      date.setDate(date.getDate() - (days - i));
      
      // Add some volatility
      const change = (Math.random() - 0.5) * 0.02; // ±1% daily change
      baseNav = baseNav * (1 + change);
      
      navHistory.push({
        date: date.toISOString().split('T')[0],
        nav: parseFloat(baseNav.toFixed(2)),
        change: parseFloat((change * 100).toFixed(2)),
        changePercent: parseFloat((change * 100).toFixed(2))
      });
    }

    return {
      fundCode,
      period,
      navHistory,
      performance: this.generateMockPerformance(fundCode, period)
    };
  }

  /**
   * Generate mock performance data
   */
  generateMockPerformance(fundCode, period) {
    const baseReturn = 8 + Math.random() * 8; // 8-16% return
    const volatility = 10 + Math.random() * 10; // 10-20% volatility
    
    return {
      totalReturn: parseFloat(baseReturn.toFixed(2)),
      annualizedReturn: parseFloat(baseReturn.toFixed(2)),
      volatility: parseFloat(volatility.toFixed(2)),
      sharpeRatio: parseFloat((baseReturn / volatility).toFixed(2)),
      maxDrawdown: parseFloat((-volatility * 0.5).toFixed(2)),
      beta: parseFloat((0.8 + Math.random() * 0.4).toFixed(2)),
      alpha: parseFloat((Math.random() * 4 - 2).toFixed(2)),
      informationRatio: parseFloat((Math.random() * 2 - 1).toFixed(2))
    };
  }

  /**
   * Get fund details from database
   */
  async getFundDetails(fundCode) {
    // Mock fund details - in real implementation, this would come from database
    const fundDetailsMap = {
      'HDFCMIDCAP': {
        name: 'HDFC Mid-Cap Opportunities Fund',
        fundHouse: 'HDFC Mutual Fund',
        category: 'Equity',
        subCategory: 'Mid Cap',
        inceptionDate: '2007-06-25',
        aum: 25000, // in crores
        expenseRatio: 1.75,
        minInvestment: 5000,
        nav: 45.67,
        navDate: new Date().toISOString().split('T')[0],
        rating: 4.5,
        fundManager: 'Chirag Setalvad',
        benchmark: 'NIFTY Midcap 150 Index',
        exitLoad: '1% if redeemed within 1 year',
        riskLevel: 'moderate'
      },
      'ICICIBLUECHIP': {
        name: 'ICICI Prudential Bluechip Fund',
        fundHouse: 'ICICI Prudential Mutual Fund',
        category: 'Equity',
        subCategory: 'Large Cap',
        inceptionDate: '1998-05-23',
        aum: 35000,
        expenseRatio: 1.65,
        minInvestment: 5000,
        nav: 52.34,
        navDate: new Date().toISOString().split('T')[0],
        rating: 4.3,
        fundManager: 'Sankaran Naren',
        benchmark: 'NIFTY 50 Index',
        exitLoad: '1% if redeemed within 1 year',
        riskLevel: 'moderate'
      },
      'SBISMALLCAP': {
        name: 'SBI Small Cap Fund',
        fundHouse: 'SBI Mutual Fund',
        category: 'Equity',
        subCategory: 'Small Cap',
        inceptionDate: '2009-09-28',
        aum: 15000,
        expenseRatio: 1.85,
        minInvestment: 5000,
        nav: 38.92,
        navDate: new Date().toISOString().split('T')[0],
        rating: 4.1,
        fundManager: 'R Srinivasan',
        benchmark: 'NIFTY Smallcap 250 Index',
        exitLoad: '1% if redeemed within 1 year',
        riskLevel: 'high'
      },
      'AXISBLUECHIP': {
        name: 'Axis Bluechip Fund',
        fundHouse: 'Axis Mutual Fund',
        category: 'Equity',
        subCategory: 'Large Cap',
        inceptionDate: '2010-01-05',
        aum: 28000,
        expenseRatio: 1.70,
        minInvestment: 5000,
        nav: 48.76,
        navDate: new Date().toISOString().split('T')[0],
        rating: 4.4,
        fundManager: 'Shreyash Devalkar',
        benchmark: 'NIFTY 50 Index',
        exitLoad: '1% if redeemed within 1 year',
        riskLevel: 'moderate'
      },
      'MIRAEEMERGING': {
        name: 'Mirae Asset Emerging Bluechip Fund',
        fundHouse: 'Mirae Asset Mutual Fund',
        category: 'Equity',
        subCategory: 'Mid Cap',
        inceptionDate: '2010-07-09',
        aum: 18000,
        expenseRatio: 1.80,
        minInvestment: 5000,
        nav: 42.15,
        navDate: new Date().toISOString().split('T')[0],
        rating: 4.6,
        fundManager: 'Neelesh Surana',
        benchmark: 'NIFTY Midcap 150 Index',
        exitLoad: '1% if redeemed within 1 year',
        riskLevel: 'moderate'
      }
    };

    return fundDetailsMap[fundCode] || {
      name: `${fundCode} Fund`,
      fundHouse: 'Unknown Fund House',
      category: 'Equity',
      subCategory: 'Unknown',
      inceptionDate: '2020-01-01',
      aum: 10000,
      expenseRatio: 1.75,
      minInvestment: 5000,
      nav: 40.00,
      navDate: new Date().toISOString().split('T')[0],
      rating: 4.0,
      fundManager: 'Unknown',
      benchmark: 'NIFTY 50 Index',
      exitLoad: '1% if redeemed within 1 year',
      riskLevel: 'moderate'
    };
  }

  /**
   * Calculate comprehensive risk metrics
   */
  async calculateRiskMetrics(navHistory) {
    if (!navHistory || !navHistory.navHistory || !Array.isArray(navHistory.navHistory)) {
      // Return mock risk metrics if no NAV history
      return {
        volatility: parseFloat((10 + Math.random() * 10).toFixed(2)),
        maxDrawdown: parseFloat((-5 - Math.random() * 5).toFixed(2)),
        sharpeRatio: parseFloat((0.5 + Math.random() * 0.5).toFixed(2)),
        beta: parseFloat((0.8 + Math.random() * 0.4).toFixed(2)),
        var95: parseFloat((-8 - Math.random() * 4).toFixed(2)),
        avgReturn: parseFloat((8 + Math.random() * 8).toFixed(2))
      };
    }

    const returns = navHistory.navHistory.map((nav, index) => {
      if (index === 0) return 0;
      return ((nav.nav - navHistory.navHistory[index - 1].nav) / navHistory.navHistory[index - 1].nav) * 100;
    }).slice(1);

    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance);

    // Calculate maximum drawdown
    let maxDrawdown = 0;
    let peak = navHistory.navHistory[0].nav;
    
    for (const nav of navHistory.navHistory) {
      if (nav.nav > peak) {
        peak = nav.nav;
      }
      const drawdown = (peak - nav.nav) / peak * 100;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    // Calculate Sharpe ratio (assuming risk-free rate of 6%)
    const riskFreeRate = 6;
    const sharpeRatio = (avgReturn - riskFreeRate) / volatility;

    // Calculate beta (simplified - would need market data in real implementation)
    const beta = 0.8 + Math.random() * 0.4; // Mock beta calculation

    return {
      volatility: parseFloat(volatility.toFixed(2)),
      maxDrawdown: parseFloat(maxDrawdown.toFixed(2)),
      sharpeRatio: parseFloat(sharpeRatio.toFixed(2)),
      beta: parseFloat(beta.toFixed(2)),
      var95: parseFloat((avgReturn - 1.645 * volatility).toFixed(2)), // 95% VaR
      avgReturn: parseFloat(avgReturn.toFixed(2))
    };
  }

  /**
   * Perform comprehensive analysis of all funds
   */
  async performComprehensiveAnalysis(fundData, period, investmentAmount) {
    const analysis = {
      comparisonPeriod: period,
      investmentAmount,
      totalFunds: fundData.length,
      funds: [],
      comparisonMetrics: {},
      categoryAnalysis: {}
    };

    // Analyze each fund
    for (const fund of fundData) {
      const fundAnalysis = {
        fundCode: fund.fundCode,
        fundDetails: fund.fundDetails,
        performance: fund.performance,
        riskMetrics: fund.riskMetrics,
        analysis: await this.analyzeFund(fund, period, investmentAmount)
      };

      analysis.funds.push(fundAnalysis);
    }

    // Calculate comparison metrics
    analysis.comparisonMetrics = this.calculateComparisonMetrics(analysis.funds);
    
    // Perform category analysis
    analysis.categoryAnalysis = this.performCategoryAnalysis(analysis.funds);

    return analysis;
  }

  /**
   * Analyze individual fund
   */
  async analyzeFund(fund, period, investmentAmount) {
    const { fundDetails, performance, riskMetrics } = fund;

    // Calculate projected value
    const projectedValue = investmentAmount * (1 + performance.totalReturn / 100);
    const expectedReturn = projectedValue - investmentAmount;

    // Calculate expense impact
    const expenseImpact = (investmentAmount * fundDetails.expenseRatio / 100) * (period === '1y' ? 1 : period === '3y' ? 3 : 5);

    // Calculate tax efficiency (simplified)
    const taxEfficiency = this.calculateTaxEfficiency(performance.totalReturn, fundDetails.category);

    return {
      projectedValue: parseFloat(projectedValue.toFixed(2)),
      expectedReturn: parseFloat(expectedReturn.toFixed(2)),
      expenseImpact: parseFloat(expenseImpact.toFixed(2)),
      taxEfficiency: parseFloat(taxEfficiency.toFixed(2)),
      consistency: this.calculateConsistency(fund.navHistory),
      liquidity: this.calculateLiquidity(fundDetails.aum),
      fundManagerExperience: this.calculateManagerExperience(fundDetails.inceptionDate),
      fundHouseReputation: this.calculateFundHouseReputation(fundDetails.fundHouse)
    };
  }

  /**
   * Calculate tax efficiency
   */
  calculateTaxEfficiency(returnPercent, category) {
    // Simplified tax efficiency calculation
    if (category === 'Equity') {
      return returnPercent > 10 ? 85 : 90; // Higher returns = more tax
    }
    return 95; // Debt funds are generally more tax efficient
  }

  /**
   * Calculate consistency score
   */
  calculateConsistency(navHistory) {
    const returns = navHistory.navHistory.map((nav, index) => {
      if (index === 0) return 0;
      return ((nav.nav - navHistory.navHistory[index - 1].nav) / navHistory.navHistory[index - 1].nav) * 100;
    }).slice(1);

    const positiveReturns = returns.filter(ret => ret > 0).length;
    const consistency = (positiveReturns / returns.length) * 100;
    
    return parseFloat(consistency.toFixed(2));
  }

  /**
   * Calculate liquidity score based on AUM
   */
  calculateLiquidity(aum) {
    if (aum > 20000) return 95;
    if (aum > 10000) return 85;
    if (aum > 5000) return 75;
    return 60;
  }

  /**
   * Calculate fund manager experience
   */
  calculateManagerExperience(inceptionDate) {
    const years = new Date().getFullYear() - new Date(inceptionDate).getFullYear();
    if (years > 10) return 95;
    if (years > 5) return 85;
    if (years > 3) return 75;
    return 60;
  }

  /**
   * Calculate fund house reputation
   */
  calculateFundHouseReputation(fundHouse) {
    const reputationMap = {
      'HDFC Mutual Fund': 95,
      'ICICI Prudential Mutual Fund': 90,
      'SBI Mutual Fund': 85,
      'Axis Mutual Fund': 88,
      'Mirae Asset Mutual Fund': 92
    };
    
    return reputationMap[fundHouse] || 75;
  }

  /**
   * Calculate comparison metrics across all funds
   */
  calculateComparisonMetrics(funds) {
    const metrics = {
      bestPerformer: null,
      lowestRisk: null,
      bestSharpeRatio: null,
      lowestExpense: null,
      highestConsistency: null,
      bestTaxEfficiency: null
    };

    let bestReturn = -Infinity;
    let lowestVolatility = Infinity;
    let bestSharpe = -Infinity;
    let lowestExpense = Infinity;
    let highestConsistency = -Infinity;
    let bestTaxEfficiency = -Infinity;

    funds.forEach(fund => {
      // Best performer
      if (fund.performance.totalReturn > bestReturn) {
        bestReturn = fund.performance.totalReturn;
        metrics.bestPerformer = fund.fundCode;
      }

      // Lowest risk
      if (fund.riskMetrics.volatility < lowestVolatility) {
        lowestVolatility = fund.riskMetrics.volatility;
        metrics.lowestRisk = fund.fundCode;
      }

      // Best Sharpe ratio
      if (fund.riskMetrics.sharpeRatio > bestSharpe) {
        bestSharpe = fund.riskMetrics.sharpeRatio;
        metrics.bestSharpeRatio = fund.fundCode;
      }

      // Lowest expense
      if (fund.fundDetails.expenseRatio < lowestExpense) {
        lowestExpense = fund.fundDetails.expenseRatio;
        metrics.lowestExpense = fund.fundCode;
      }

      // Highest consistency
      if (fund.analysis.consistency > highestConsistency) {
        highestConsistency = fund.analysis.consistency;
        metrics.highestConsistency = fund.fundCode;
      }

      // Best tax efficiency
      if (fund.analysis.taxEfficiency > bestTaxEfficiency) {
        bestTaxEfficiency = fund.analysis.taxEfficiency;
        metrics.bestTaxEfficiency = fund.fundCode;
      }
    });

    return metrics;
  }

  /**
   * Perform category analysis
   */
  performCategoryAnalysis(funds) {
    const categories = {};
    
    funds.forEach(fund => {
      const category = fund.fundDetails.subCategory;
      if (!categories[category]) {
        categories[category] = {
          count: 0,
          avgReturn: 0,
          avgVolatility: 0,
          avgExpenseRatio: 0
        };
      }
      
      categories[category].count++;
      categories[category].avgReturn += fund.performance.totalReturn;
      categories[category].avgVolatility += fund.riskMetrics.volatility;
      categories[category].avgExpenseRatio += fund.fundDetails.expenseRatio;
    });

    // Calculate averages
    Object.keys(categories).forEach(category => {
      const count = categories[category].count;
      categories[category].avgReturn = parseFloat((categories[category].avgReturn / count).toFixed(2));
      categories[category].avgVolatility = parseFloat((categories[category].avgVolatility / count).toFixed(2));
      categories[category].avgExpenseRatio = parseFloat((categories[category].avgExpenseRatio / count).toFixed(2));
    });

    return categories;
  }

  /**
   * Calculate comprehensive fund ratings with precise 10-point scale
   */
  async calculateFundRatings(analysis) {
    const ratings = [];

    for (const fund of analysis.funds) {
      const rating = {
        fundCode: fund.fundCode,
        fundName: fund.fundDetails.name,
        overallRating: 0,
        categoryRatings: {},
        totalScore: 0,
        maxScore: 100,
        breakdown: {},
        detailedAnalysis: {}
      };

      // Performance Rating (25 points) - Enhanced
      const performanceScore = this.calculateEnhancedPerformanceScore(fund, analysis.funds);
      rating.categoryRatings.performance = {
        score: performanceScore,
        maxScore: 25,
        percentage: parseFloat(((performanceScore / 25) * 100).toFixed(1)),
        details: this.getPerformanceDetails(fund)
      };

      // Risk Rating (20 points) - Enhanced
      const riskScore = this.calculateEnhancedRiskScore(fund, analysis.funds);
      rating.categoryRatings.risk = {
        score: riskScore,
        maxScore: 20,
        percentage: parseFloat(((riskScore / 20) * 100).toFixed(1)),
        details: this.getRiskDetails(fund)
      };

      // Cost Rating (15 points) - Enhanced
      const costScore = this.calculateEnhancedCostScore(fund, analysis.funds);
      rating.categoryRatings.cost = {
        score: costScore,
        maxScore: 15,
        percentage: parseFloat(((costScore / 15) * 100).toFixed(1)),
        details: this.getCostDetails(fund)
      };

      // Consistency Rating (15 points) - Enhanced
      const consistencyScore = this.calculateEnhancedConsistencyScore(fund, analysis.funds);
      rating.categoryRatings.consistency = {
        score: consistencyScore,
        maxScore: 15,
        percentage: parseFloat(((consistencyScore / 15) * 100).toFixed(1)),
        details: this.getConsistencyDetails(fund)
      };

      // Fund House Rating (10 points) - Enhanced
      const fundHouseScore = this.calculateEnhancedFundHouseScore(fund, analysis.funds);
      rating.categoryRatings.fundHouse = {
        score: fundHouseScore,
        maxScore: 10,
        percentage: parseFloat(((fundHouseScore / 10) * 100).toFixed(1)),
        details: this.getFundHouseDetails(fund)
      };

      // Tax Efficiency Rating (10 points) - Enhanced
      const taxScore = this.calculateEnhancedTaxScore(fund, analysis.funds);
      rating.categoryRatings.taxEfficiency = {
        score: taxScore,
        maxScore: 10,
        percentage: parseFloat(((taxScore / 10) * 100).toFixed(1)),
        details: this.getTaxDetails(fund)
      };

      // Liquidity Rating (5 points) - Enhanced
      const liquidityScore = this.calculateEnhancedLiquidityScore(fund, analysis.funds);
      rating.categoryRatings.liquidity = {
        score: liquidityScore,
        maxScore: 5,
        percentage: parseFloat(((liquidityScore / 5) * 100).toFixed(1)),
        details: this.getLiquidityDetails(fund)
      };

      // Calculate total score
      rating.totalScore = performanceScore + riskScore + costScore + consistencyScore + 
                         fundHouseScore + taxScore + liquidityScore;
      
      // Convert to 10-point scale with decimal precision
      rating.overallRating = parseFloat((rating.totalScore / 10).toFixed(1)); // 10-point scale

      // Add detailed analysis
      rating.detailedAnalysis = this.generateDetailedAnalysis(fund, analysis.funds);

      ratings.push(rating);
    }

    // Sort by overall rating
    ratings.sort((a, b) => b.overallRating - a.overallRating);

    return ratings;
  }

  /**
   * Calculate enhanced performance score (25 points)
   */
  calculateEnhancedPerformanceScore(fund, allFunds) {
    const returns = allFunds.map(f => f.performance.totalReturn);
    const maxReturn = Math.max(...returns);
    const minReturn = Math.min(...returns);
    
    if (maxReturn === minReturn) return 25;
    
    // Base score from returns (15 points)
    const returnScore = ((fund.performance.totalReturn - minReturn) / (maxReturn - minReturn)) * 15;
    
    // Risk-adjusted return score (5 points)
    const sharpeRatios = allFunds.map(f => f.performance.sharpeRatio);
    const maxSharpe = Math.max(...sharpeRatios);
    const minSharpe = Math.min(...sharpeRatios);
    const sharpeScore = maxSharpe === minSharpe ? 5 : 
      ((fund.performance.sharpeRatio - minSharpe) / (maxSharpe - minSharpe)) * 5;
    
    // Consistency bonus (3 points)
    const consistencyBonus = fund.analysis.consistency > 80 ? 3 : 
      fund.analysis.consistency > 60 ? 2 : 1;
    
    // Alpha score (2 points)
    const alphaScore = fund.performance.alpha > 0 ? 2 : 
      fund.performance.alpha > -1 ? 1 : 0;
    
    const totalScore = returnScore + sharpeScore + consistencyBonus + alphaScore;
    return parseFloat(Math.min(totalScore, 25).toFixed(1));
  }

  /**
   * Get detailed performance analysis
   */
  getPerformanceDetails(fund) {
    return {
      totalReturn: fund.performance.totalReturn,
      annualizedReturn: fund.performance.annualizedReturn,
      sharpeRatio: fund.performance.sharpeRatio,
      alpha: fund.performance.alpha,
      informationRatio: fund.performance.informationRatio,
      benchmark: fund.fundDetails.benchmark,
      vsBenchmark: fund.performance.totalReturn > 10 ? 'Outperforming' : 'Underperforming',
      consistency: fund.analysis.consistency,
      analysis: this.generatePerformanceAnalysis(fund)
    };
  }

  /**
   * Generate performance analysis text
   */
  generatePerformanceAnalysis(fund) {
    const returnLevel = fund.performance.totalReturn > 15 ? 'Excellent' :
                       fund.performance.totalReturn > 12 ? 'Good' :
                       fund.performance.totalReturn > 8 ? 'Average' : 'Below Average';
    
    const sharpeLevel = fund.performance.sharpeRatio > 1 ? 'Excellent' :
                       fund.performance.sharpeRatio > 0.5 ? 'Good' :
                       fund.performance.sharpeRatio > 0 ? 'Average' : 'Poor';
    
    return `${fund.fundDetails.name} has shown ${returnLevel.toLowerCase()} performance with ${fund.performance.totalReturn}% returns. The fund's risk-adjusted returns (Sharpe ratio: ${fund.performance.sharpeRatio}) are ${sharpeLevel.toLowerCase()}. ${fund.analysis.consistency}% of monthly returns have been positive, indicating ${fund.analysis.consistency > 70 ? 'good' : 'moderate'} consistency.`;
  }

  /**
   * Calculate enhanced risk score (20 points) - lower risk = higher score
   */
  calculateEnhancedRiskScore(fund, allFunds) {
    const volatilities = allFunds.map(f => f.riskMetrics.volatility);
    const maxVol = Math.max(...volatilities);
    const minVol = Math.min(...volatilities);
    
    // Volatility score (10 points)
    const volatilityScore = maxVol === minVol ? 10 : 
      ((maxVol - fund.riskMetrics.volatility) / (maxVol - minVol)) * 10;
    
    // Maximum drawdown score (5 points)
    const drawdowns = allFunds.map(f => Math.abs(f.riskMetrics.maxDrawdown));
    const maxDrawdown = Math.max(...drawdowns);
    const minDrawdown = Math.min(...drawdowns);
    const drawdownScore = maxDrawdown === minDrawdown ? 5 : 
      ((maxDrawdown - Math.abs(fund.riskMetrics.maxDrawdown)) / (maxDrawdown - minDrawdown)) * 5;
    
    // Beta score (3 points) - lower beta = better for risk
    const betas = allFunds.map(f => f.riskMetrics.beta);
    const maxBeta = Math.max(...betas);
    const minBeta = Math.min(...betas);
    const betaScore = maxBeta === minBeta ? 3 : 
      ((maxBeta - fund.riskMetrics.beta) / (maxBeta - minBeta)) * 3;
    
    // VaR score (2 points)
    const var95s = allFunds.map(f => Math.abs(f.riskMetrics.var95));
    const maxVar = Math.max(...var95s);
    const minVar = Math.min(...var95s);
    const varScore = maxVar === minVar ? 2 : 
      ((maxVar - Math.abs(fund.riskMetrics.var95)) / (maxVar - minVar)) * 2;
    
    const totalScore = volatilityScore + drawdownScore + betaScore + varScore;
    return parseFloat(Math.min(totalScore, 20).toFixed(1));
  }

  /**
   * Get detailed risk analysis
   */
  getRiskDetails(fund) {
    return {
      volatility: fund.riskMetrics.volatility,
      maxDrawdown: fund.riskMetrics.maxDrawdown,
      sharpeRatio: fund.riskMetrics.sharpeRatio,
      beta: fund.riskMetrics.beta,
      var95: fund.riskMetrics.var95,
      riskLevel: this.calculateRiskLevel(fund.riskMetrics.volatility),
      analysis: this.generateRiskAnalysis(fund)
    };
  }

  /**
   * Calculate risk level
   */
  calculateRiskLevel(volatility) {
    if (volatility < 10) return 'Low';
    if (volatility < 15) return 'Moderate';
    if (volatility < 20) return 'Moderately High';
    return 'High';
  }

  /**
   * Generate risk analysis text
   */
  generateRiskAnalysis(fund) {
    const riskLevel = this.calculateRiskLevel(fund.riskMetrics.volatility);
    const drawdownSeverity = Math.abs(fund.riskMetrics.maxDrawdown) > 15 ? 'significant' :
                             Math.abs(fund.riskMetrics.maxDrawdown) > 10 ? 'moderate' : 'low';
    
    return `${fund.fundDetails.name} has ${riskLevel.toLowerCase()} risk with ${fund.riskMetrics.volatility}% volatility. The fund has experienced ${drawdownSeverity} drawdowns (maximum: ${Math.abs(fund.riskMetrics.maxDrawdown).toFixed(1)}%). Beta of ${fund.riskMetrics.beta} indicates ${fund.riskMetrics.beta > 1 ? 'higher' : 'lower'} market sensitivity. 95% VaR of ${Math.abs(fund.riskMetrics.var95).toFixed(1)}% suggests potential downside risk.`;
  }

  /**
   * Calculate enhanced cost score (15 points) - lower expense = higher score
   */
  calculateEnhancedCostScore(fund, allFunds) {
    const expenses = allFunds.map(f => f.fundDetails.expenseRatio);
    const maxExpense = Math.max(...expenses);
    const minExpense = Math.min(...expenses);
    
    // Expense ratio score (10 points)
    const expenseScore = maxExpense === minExpense ? 10 : 
      ((maxExpense - fund.fundDetails.expenseRatio) / (maxExpense - minExpense)) * 10;
    
    // Exit load score (3 points)
    const exitLoadScore = fund.fundDetails.exitLoad.includes('1%') ? 1 : 
      fund.fundDetails.exitLoad.includes('0.5%') ? 2 : 3;
    
    // Minimum investment score (2 points)
    const minInvScore = fund.fundDetails.minInvestment <= 1000 ? 2 : 
      fund.fundDetails.minInvestment <= 5000 ? 1 : 0;
    
    const totalScore = expenseScore + exitLoadScore + minInvScore;
    return parseFloat(Math.min(totalScore, 15).toFixed(1));
  }

  /**
   * Get detailed cost analysis
   */
  getCostDetails(fund) {
    return {
      expenseRatio: fund.fundDetails.expenseRatio,
      exitLoad: fund.fundDetails.exitLoad,
      minInvestment: fund.fundDetails.minInvestment,
      impactOnReturns: parseFloat((fund.fundDetails.expenseRatio * 10).toFixed(1)),
      analysis: this.generateCostAnalysis(fund)
    };
  }

  /**
   * Generate cost analysis text
   */
  generateCostAnalysis(fund) {
    const expenseLevel = fund.fundDetails.expenseRatio < 1.5 ? 'Low' :
                        fund.fundDetails.expenseRatio < 2.0 ? 'Moderate' : 'High';
    
    return `${fund.fundDetails.name} has ${expenseLevel.toLowerCase()} expense ratio of ${fund.fundDetails.expenseRatio}%, which is ${fund.fundDetails.expenseRatio < 1.5 ? 'competitive' : 'standard'} for its category. The fund charges ${fund.fundDetails.exitLoad} exit load. Minimum investment is ₹${fund.fundDetails.minInvestment.toLocaleString()}. Over 10 years, expenses could reduce returns by approximately ${(fund.fundDetails.expenseRatio * 10).toFixed(1)}%.`;
  }

  /**
   * Calculate enhanced consistency score (15 points)
   */
  calculateEnhancedConsistencyScore(fund, allFunds) {
    const consistencies = allFunds.map(f => f.analysis.consistency);
    const maxConsistency = Math.max(...consistencies);
    const minConsistency = Math.min(...consistencies);
    
    // Consistency score (10 points)
    const consistencyScore = maxConsistency === minConsistency ? 10 : 
      ((fund.analysis.consistency - minConsistency) / (maxConsistency - minConsistency)) * 10;
    
    // Fund manager experience score (3 points)
    const experienceScore = fund.analysis.fundManagerExperience > 90 ? 3 : 
      fund.analysis.fundManagerExperience > 70 ? 2 : 1;
    
    // Fund age score (2 points)
    const fundAge = new Date().getFullYear() - new Date(fund.fundDetails.inceptionDate).getFullYear();
    const ageScore = fundAge > 10 ? 2 : fundAge > 5 ? 1 : 0;
    
    const totalScore = consistencyScore + experienceScore + ageScore;
    return parseFloat(Math.min(totalScore, 15).toFixed(1));
  }

  /**
   * Get detailed consistency analysis
   */
  getConsistencyDetails(fund) {
    const fundAge = new Date().getFullYear() - new Date(fund.fundDetails.inceptionDate).getFullYear();
    return {
      consistency: fund.analysis.consistency,
      fundManagerExperience: fund.analysis.fundManagerExperience,
      fundAge: fundAge,
      inceptionDate: fund.fundDetails.inceptionDate,
      analysis: this.generateConsistencyAnalysis(fund)
    };
  }

  /**
   * Generate consistency analysis text
   */
  generateConsistencyAnalysis(fund) {
    const fundAge = new Date().getFullYear() - new Date(fund.fundDetails.inceptionDate).getFullYear();
    const consistencyLevel = fund.analysis.consistency > 80 ? 'Excellent' :
                            fund.analysis.consistency > 60 ? 'Good' : 'Moderate';
    
    return `${fund.fundDetails.name} shows ${consistencyLevel.toLowerCase()} consistency with ${fund.analysis.consistency}% positive monthly returns. The fund has been operating for ${fundAge} years, providing a ${fundAge > 10 ? 'long' : fundAge > 5 ? 'moderate' : 'short'} track record. Fund manager experience is rated at ${fund.analysis.fundManagerExperience}%, indicating ${fund.analysis.fundManagerExperience > 80 ? 'strong' : 'adequate'} expertise.`;
  }

  /**
   * Calculate enhanced fund house score (10 points)
   */
  calculateEnhancedFundHouseScore(fund, allFunds) {
    const reputations = allFunds.map(f => f.analysis.fundHouseReputation);
    const maxReputation = Math.max(...reputations);
    const minReputation = Math.min(...reputations);
    
    // Reputation score (7 points)
    const reputationScore = maxReputation === minReputation ? 7 : 
      ((fund.analysis.fundHouseReputation - minReputation) / (maxReputation - minReputation)) * 7;
    
    // AUM score (3 points)
    const aumScore = fund.fundDetails.aum > 20000 ? 3 : 
      fund.fundDetails.aum > 10000 ? 2 : 1;
    
    const totalScore = reputationScore + aumScore;
    return parseFloat(Math.min(totalScore, 10).toFixed(1));
  }

  /**
   * Get detailed fund house analysis
   */
  getFundHouseDetails(fund) {
    return {
      fundHouse: fund.fundDetails.fundHouse,
      reputation: fund.analysis.fundHouseReputation,
      aum: fund.fundDetails.aum,
      analysis: this.generateFundHouseAnalysis(fund)
    };
  }

  /**
   * Generate fund house analysis text
   */
  generateFundHouseAnalysis(fund) {
    const reputationLevel = fund.analysis.fundHouseReputation > 90 ? 'Excellent' :
                           fund.analysis.fundHouseReputation > 70 ? 'Good' : 'Average';
    const aumLevel = fund.fundDetails.aum > 20000 ? 'large' :
                    fund.fundDetails.aum > 10000 ? 'medium' : 'small';
    
    return `${fund.fundDetails.fundHouse} has ${reputationLevel.toLowerCase()} reputation in the market. The fund house manages ₹${(fund.fundDetails.aum / 1000).toFixed(0)}K crores in assets, making it a ${aumLevel}-sized player. This indicates ${fund.fundDetails.aum > 15000 ? 'strong' : 'adequate'} operational stability and market presence.`;
  }

  /**
   * Calculate enhanced tax efficiency score (10 points)
   */
  calculateEnhancedTaxScore(fund, allFunds) {
    const taxEfficiencies = allFunds.map(f => f.analysis.taxEfficiency);
    const maxTaxEfficiency = Math.max(...taxEfficiencies);
    const minTaxEfficiency = Math.min(...taxEfficiencies);
    
    // Tax efficiency score (7 points)
    const taxEfficiencyScore = maxTaxEfficiency === minTaxEfficiency ? 7 : 
      ((fund.analysis.taxEfficiency - minTaxEfficiency) / (maxTaxEfficiency - minTaxEfficiency)) * 7;
    
    // Category tax advantage (3 points)
    const categoryTaxScore = fund.fundDetails.category === 'Equity' ? 2 : 
      fund.fundDetails.category === 'Hybrid' ? 1 : 3;
    
    const totalScore = taxEfficiencyScore + categoryTaxScore;
    return parseFloat(Math.min(totalScore, 10).toFixed(1));
  }

  /**
   * Get detailed tax analysis
   */
  getTaxDetails(fund) {
    return {
      taxEfficiency: fund.analysis.taxEfficiency,
      category: fund.fundDetails.category,
      analysis: this.generateTaxAnalysis(fund)
    };
  }

  /**
   * Generate tax analysis text
   */
  generateTaxAnalysis(fund) {
    const taxLevel = fund.analysis.taxEfficiency > 90 ? 'Excellent' :
                    fund.analysis.taxEfficiency > 80 ? 'Good' : 'Moderate';
    
    return `${fund.fundDetails.name} has ${taxLevel.toLowerCase()} tax efficiency at ${fund.analysis.taxEfficiency}%. As a ${fund.fundDetails.category} fund, it offers ${fund.fundDetails.category === 'Equity' ? 'long-term capital gains tax benefits' : fund.fundDetails.category === 'Debt' ? 'indexation benefits' : 'balanced tax treatment'}. This makes it ${fund.analysis.taxEfficiency > 85 ? 'highly' : 'moderately'} tax-efficient for long-term investors.`;
  }

  /**
   * Calculate enhanced liquidity score (5 points)
   */
  calculateEnhancedLiquidityScore(fund, allFunds) {
    const liquidities = allFunds.map(f => f.analysis.liquidity);
    const maxLiquidity = Math.max(...liquidities);
    const minLiquidity = Math.min(...liquidities);
    
    // Liquidity score (3 points)
    const liquidityScore = maxLiquidity === minLiquidity ? 3 : 
      ((fund.analysis.liquidity - minLiquidity) / (maxLiquidity - minLiquidity)) * 3;
    
    // AUM-based liquidity bonus (2 points)
    const aumLiquidityBonus = fund.fundDetails.aum > 15000 ? 2 : 
      fund.fundDetails.aum > 5000 ? 1 : 0;
    
    const totalScore = liquidityScore + aumLiquidityBonus;
    return parseFloat(Math.min(totalScore, 5).toFixed(1));
  }

  /**
   * Get detailed liquidity analysis
   */
  getLiquidityDetails(fund) {
    return {
      liquidity: fund.analysis.liquidity,
      aum: fund.fundDetails.aum,
      analysis: this.generateLiquidityAnalysis(fund)
    };
  }

  /**
   * Generate liquidity analysis text
   */
  generateLiquidityAnalysis(fund) {
    const liquidityLevel = fund.analysis.liquidity > 90 ? 'Excellent' :
                          fund.analysis.liquidity > 70 ? 'Good' : 'Moderate';
    const aumLevel = fund.fundDetails.aum > 15000 ? 'large' :
                    fund.fundDetails.aum > 5000 ? 'medium' : 'small';
    
    return `${fund.fundDetails.name} has ${liquidityLevel.toLowerCase()} liquidity with a score of ${fund.analysis.liquidity}%. The fund's ${aumLevel} AUM of ₹${(fund.fundDetails.aum / 1000).toFixed(0)}K crores ensures ${fund.fundDetails.aum > 10000 ? 'high' : 'adequate'} liquidity for investors. This means ${fund.fundDetails.aum > 10000 ? 'quick and easy' : 'reasonable'} redemption processing.`;
  }

  /**
   * Generate recommendations based on analysis
   */
  async generateRecommendations(analysis) {
    const recommendations = {
      topPick: null,
      bestValue: null,
      safestChoice: null,
      recommendations: []
    };

    const ratings = analysis.ratings;
    if (ratings.length === 0) return recommendations;

    // Top pick (highest overall rating)
    recommendations.topPick = {
      fundCode: ratings[0].fundCode,
      fundName: ratings[0].fundName,
      rating: ratings[0].overallRating,
      reason: 'Highest overall rating based on comprehensive analysis'
    };

    // Best value (best performance to cost ratio)
    const valueRatios = ratings.map(rating => {
      const fund = analysis.funds.find(f => f.fundCode === rating.fundCode);
      const valueRatio = fund.performance.totalReturn / fund.fundDetails.expenseRatio;
      return { fundCode: rating.fundCode, fundName: rating.fundName, valueRatio };
    });
    
    const bestValue = valueRatios.reduce((best, current) => 
      current.valueRatio > best.valueRatio ? current : best
    );
    
    recommendations.bestValue = {
      fundCode: bestValue.fundCode,
      fundName: bestValue.fundName,
      valueRatio: parseFloat(bestValue.valueRatio.toFixed(2)),
      reason: 'Best performance-to-cost ratio'
    };

    // Safest choice (lowest volatility with decent returns)
    const safeFunds = ratings.filter(rating => {
      const fund = analysis.funds.find(f => f.fundCode === rating.fundCode);
      return fund.riskMetrics.volatility < 15 && fund.performance.totalReturn > 8;
    });

    if (safeFunds.length > 0) {
      recommendations.safestChoice = {
        fundCode: safeFunds[0].fundCode,
        fundName: safeFunds[0].fundName,
        volatility: analysis.funds.find(f => f.fundCode === safeFunds[0].fundCode).riskMetrics.volatility,
        reason: 'Lowest risk with good returns'
      };
    }

    // General recommendations
    recommendations.recommendations = [
      {
        type: 'performance',
        message: `${ratings[0].fundName} has the best overall performance with a ${ratings[0].overallRating}-star rating`,
        priority: 'high'
      },
      {
        type: 'cost',
        message: `Consider expense ratios when choosing - lower expenses can significantly impact long-term returns`,
        priority: 'medium'
      },
      {
        type: 'diversification',
        message: 'Consider diversifying across different fund categories for better risk management',
        priority: 'medium'
      }
    ];

    return recommendations;
  }

  /**
   * Generate comparison summary
   */
  generateComparisonSummary(analysis) {
    const { funds, ratings, recommendations } = analysis;
    
    return {
      totalFundsCompared: funds.length,
      comparisonPeriod: analysis.comparisonPeriod,
      investmentAmount: analysis.investmentAmount,
      topPerformer: {
        fundCode: ratings[0]?.fundCode,
        fundName: ratings[0]?.fundName,
        rating: ratings[0]?.overallRating,
        return: funds.find(f => f.fundCode === ratings[0]?.fundCode)?.performance.totalReturn
      },
      bestValue: {
        fundCode: recommendations.bestValue?.fundCode,
        fundName: recommendations.bestValue?.fundName,
        valueRatio: recommendations.bestValue?.valueRatio
      },
      safestChoice: {
        fundCode: recommendations.safestChoice?.fundCode,
        fundName: recommendations.safestChoice?.fundName,
        volatility: recommendations.safestChoice?.volatility
      },
      keyInsights: [
        `Funds range from ${Math.min(...ratings.map(r => r.overallRating))} to ${Math.max(...ratings.map(r => r.overallRating))} stars`,
        `Performance varies from ${Math.min(...funds.map(f => f.performance.totalReturn))}% to ${Math.max(...funds.map(f => f.performance.totalReturn))}%`,
        `Risk levels range from ${Math.min(...funds.map(f => f.riskMetrics.volatility))}% to ${Math.max(...funds.map(f => f.riskMetrics.volatility))}% volatility`
      ]
    };
  }
}

module.exports = new FundComparisonService(); 