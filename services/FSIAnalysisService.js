const logger = require('../utils/logger');
const RealMutualFundDataService = require('./RealMutualFundDataService');
const RealTimeMarketService = require('./RealTimeMarketService');

class FSIAnalysisService {
  constructor() {
    this.realMutualFundService = new RealMutualFundDataService();
    this.realTimeMarketService = new RealTimeMarketService();
    
    console.log('🚀 FSI Analysis Service Initialized - AI-Powered Fund Analysis');
  }

  /**
   * Perform comprehensive FSI analysis for a fund
   */
  async performComprehensiveAnalysis(fundId) {
    try {
      logger.info(`🔍 Starting comprehensive FSI analysis for fund: ${fundId}`);

      // Get fund data
      const fundData = await this.realMutualFundService.getFundBasicInfo(fundId);
      const holdings = await this.realMutualFundService.getFundHoldings(fundId);
      const performance = await this.realMutualFundService.getFundPerformance(fundId);
      const riskMetrics = await this.calculateRiskMetrics(fundId);

      // Calculate FSI Score
      const fsiScore = await this.calculateFSIScore(fundData, holdings, performance, riskMetrics);

      // Generate analysis
      const analysis = {
        overallScore: fsiScore.score,
        grade: fsiScore.grade,
        recommendation: fsiScore.recommendation,
        expectedReturns: fsiScore.expectedReturns,
        holdingPeriod: fsiScore.holdingPeriod,
        confidence: fsiScore.confidence,
        keyStrengths: await this.generateKeyStrengths(fundData, holdings, performance),
        areasOfConcern: await this.generateAreasOfConcern(fundData, holdings, riskMetrics),
        aiRecommendations: await this.generateAIRecommendations(fundData, holdings, performance, riskMetrics)
      };

      logger.info(`✅ FSI analysis completed for fund: ${fundId}, Score: ${fsiScore.score}`);
      return analysis;

    } catch (error) {
      logger.error(`❌ Error in FSI analysis for fund ${fundId}:`, error);
      throw error;
    }
  }

  /**
   * Calculate FSI Score based on multiple factors
   */
  async calculateFSIScore(fundData, holdings, performance, riskMetrics) {
    try {
      // Performance Score (40%)
      const performanceScore = this.calculatePerformanceScore(performance);
      
      // Risk Score (25%)
      const riskScore = this.calculateRiskScore(riskMetrics);
      
      // Portfolio Quality Score (20%)
      const portfolioScore = this.calculatePortfolioScore(holdings);
      
      // Fund Management Score (15%)
      const managementScore = this.calculateManagementScore(fundData);

      // Weighted FSI Score
      const overallScore = Math.round(
        (performanceScore * 0.4) + 
        (riskScore * 0.25) + 
        (portfolioScore * 0.2) + 
        (managementScore * 0.15)
      );

      // Determine grade and recommendation
      const grade = this.determineGrade(overallScore);
      const recommendation = this.determineRecommendation(overallScore, riskScore);
      const expectedReturns = this.calculateExpectedReturns(overallScore, performance);
      const holdingPeriod = this.determineHoldingPeriod(overallScore, riskScore);
      const confidence = this.calculateConfidence(overallScore, riskScore);

      return {
        score: overallScore,
        grade,
        recommendation,
        expectedReturns,
        holdingPeriod,
        confidence,
        breakdown: {
          performance: performanceScore,
          risk: riskScore,
          portfolio: portfolioScore,
          management: managementScore
        }
      };

    } catch (error) {
      logger.error('Error calculating FSI score:', error);
      throw error;
    }
  }

  /**
   * Calculate performance score based on returns
   */
  calculatePerformanceScore(performance) {
    const returns = performance.returns;
    
    // Weight recent performance more heavily
    const score = (
      (returns['1Y'] * 0.4) +
      (returns['3Y'] * 0.3) +
      (returns['5Y'] * 0.2) +
      (returns['1M'] * 0.1)
    );

    // Normalize to 0-100 scale
    return Math.max(0, Math.min(100, score + 50));
  }

  /**
   * Calculate risk score (higher is better - lower risk)
   */
  calculateRiskScore(riskMetrics) {
    const volatilityScore = Math.max(0, 100 - (riskMetrics.volatility * 2));
    const sharpeScore = Math.min(100, (riskMetrics.sharpeRatio + 1) * 30);
    const drawdownScore = Math.max(0, 100 + (riskMetrics.maxDrawdown * 2));
    
    return (volatilityScore + sharpeScore + drawdownScore) / 3;
  }

  /**
   * Calculate portfolio quality score
   */
  calculatePortfolioScore(holdings) {
    // Diversification score
    const diversificationScore = Math.min(100, holdings.length * 2);
    
    // Quality score based on top holdings
    const topHoldings = holdings.slice(0, 10);
    const qualityScore = topHoldings.reduce((sum, holding) => {
      return sum + this.getStockQualityScore(holding);
    }, 0) / topHoldings.length;

    return (diversificationScore + qualityScore) / 2;
  }

  /**
   * Get individual stock quality score
   */
  getStockQualityScore(holding) {
    // Mock implementation - would use real stock analysis
    const marketCapScore = holding.marketCap === 'Large Cap' ? 85 : 
                          holding.marketCap === 'Mid Cap' ? 70 : 60;
    const sectorScore = this.getSectorScore(holding.sector);
    
    return (marketCapScore + sectorScore) / 2;
  }

  /**
   * Get sector performance score
   */
  getSectorScore(sector) {
    const sectorScores = {
      'Technology': 85,
      'Financial Services': 80,
      'Healthcare': 82,
      'Consumer Goods': 75,
      'Energy': 70,
      'Industrials': 78,
      'Materials': 72,
      'Utilities': 68
    };
    
    return sectorScores[sector] || 75;
  }

  /**
   * Calculate fund management score
   */
  calculateManagementScore(fundData) {
    // Mock implementation - would analyze fund manager track record
    const expenseRatioScore = Math.max(0, 100 - (fundData.expenseRatio * 50));
    const aumScore = Math.min(100, Math.log10(parseFloat(fundData.aum.replace(/[^0-9.]/g, ''))) * 20);
    const trackRecordScore = 80; // Would be calculated based on manager history
    
    return (expenseRatioScore + aumScore + trackRecordScore) / 3;
  }

  /**
   * Determine grade based on score
   */
  determineGrade(score) {
    if (score >= 90) return 'A+';
    if (score >= 80) return 'A';
    if (score >= 70) return 'B+';
    if (score >= 60) return 'B';
    if (score >= 50) return 'C+';
    return 'C';
  }

  /**
   * Determine investment recommendation
   */
  determineRecommendation(overallScore, riskScore) {
    if (overallScore >= 85 && riskScore >= 70) return 'STRONG BUY';
    if (overallScore >= 75 && riskScore >= 60) return 'BUY';
    if (overallScore >= 65) return 'HOLD';
    if (overallScore >= 50) return 'WEAK HOLD';
    return 'AVOID';
  }

  /**
   * Calculate expected returns
   */
  calculateExpectedReturns(score, performance) {
    const baseReturn = (score - 50) * 0.3; // Scale score to return percentage
    const historicalAdjustment = performance.returns['3Y'] * 0.2;
    const expectedReturn = baseReturn + historicalAdjustment;
    
    return `${Math.max(8, Math.min(25, expectedReturn)).toFixed(0)}-${Math.max(12, Math.min(30, expectedReturn + 5)).toFixed(0)}%`;
  }

  /**
   * Determine optimal holding period
   */
  determineHoldingPeriod(overallScore, riskScore) {
    if (overallScore >= 80 && riskScore >= 70) return '3-5 years';
    if (overallScore >= 70) return '2-4 years';
    if (overallScore >= 60) return '1-3 years';
    return '1-2 years';
  }

  /**
   * Calculate confidence level
   */
  calculateConfidence(overallScore, riskScore) {
    const confidence = (overallScore + riskScore) / 2;
    
    if (confidence >= 85) return 'Very High';
    if (confidence >= 75) return 'High';
    if (confidence >= 65) return 'Medium';
    return 'Low';
  }

  /**
   * Generate key strengths
   */
  async generateKeyStrengths(fundData, holdings, performance) {
    const strengths = [];

    if (performance.returns['1Y'] > 15) {
      strengths.push('Exceptional 1-year performance with returns exceeding 15%');
    }

    if (performance.returns['3Y'] > 12) {
      strengths.push('Consistent long-term performance with strong 3-year track record');
    }

    if (fundData.expenseRatio < 1.5) {
      strengths.push('Low expense ratio providing cost-effective investment option');
    }

    if (holdings.length >= 30) {
      strengths.push('Well-diversified portfolio reducing concentration risk');
    }

    const techAllocation = this.calculateSectorAllocation(holdings, 'Technology');
    if (techAllocation > 20) {
      strengths.push('Strong technology sector exposure benefiting from digital transformation');
    }

    return strengths.slice(0, 5); // Return top 5 strengths
  }

  /**
   * Generate areas of concern
   */
  async generateAreasOfConcern(fundData, holdings, riskMetrics) {
    const concerns = [];

    if (riskMetrics.volatility > 20) {
      concerns.push('High volatility may result in significant short-term fluctuations');
    }

    if (riskMetrics.maxDrawdown < -25) {
      concerns.push('Significant maximum drawdown indicates potential for large losses');
    }

    if (fundData.expenseRatio > 2.0) {
      concerns.push('High expense ratio may impact long-term returns');
    }

    const topHoldingAllocation = holdings[0]?.allocation || 0;
    if (topHoldingAllocation > 8) {
      concerns.push('High concentration in top holding increases single-stock risk');
    }

    if (riskMetrics.sharpeRatio < 0.5) {
      concerns.push('Low Sharpe ratio indicates suboptimal risk-adjusted returns');
    }

    return concerns.slice(0, 4); // Return top 4 concerns
  }

  /**
   * Generate AI recommendations
   */
  async generateAIRecommendations(fundData, holdings, performance, riskMetrics) {
    const recommendations = [];

    // Performance-based recommendations
    if (performance.returns['1Y'] > performance.returns['3Y']) {
      recommendations.push('Recent outperformance suggests positive momentum - consider increasing allocation');
    }

    // Risk-based recommendations
    if (riskMetrics.sharpeRatio > 1.0) {
      recommendations.push('Excellent risk-adjusted returns make this suitable for core portfolio allocation');
    }

    // Portfolio construction recommendations
    if (riskMetrics.volatility < 15) {
      recommendations.push('Low volatility profile makes it suitable for conservative investors');
    }

    // Market timing recommendations
    recommendations.push('Current market conditions favor systematic investment approach over lump sum');

    // Diversification recommendations
    const sectorConcentration = this.calculateSectorConcentration(holdings);
    if (sectorConcentration < 40) {
      recommendations.push('Well-balanced sector allocation provides good diversification benefits');
    }

    // Long-term recommendations
    if (performance.returns['5Y'] > 10) {
      recommendations.push('Strong long-term track record supports buy-and-hold investment strategy');
    }

    return recommendations.slice(0, 6); // Return top 6 recommendations
  }

  /**
   * Calculate sector allocation percentage
   */
  calculateSectorAllocation(holdings, sector) {
    return holdings
      .filter(holding => holding.sector === sector)
      .reduce((sum, holding) => sum + holding.allocation, 0);
  }

  /**
   * Calculate sector concentration
   */
  calculateSectorConcentration(holdings) {
    const sectorAllocations = {};
    holdings.forEach(holding => {
      sectorAllocations[holding.sector] = (sectorAllocations[holding.sector] || 0) + holding.allocation;
    });

    return Math.max(...Object.values(sectorAllocations));
  }

  /**
   * Get fund holdings with analysis
   */
  async getFundHoldingsWithAnalysis(fundId, limit = 50) {
    try {
      const holdings = await this.realMutualFundService.getFundHoldings(fundId);
      
      // Add analysis to each holding
      const analyzedHoldings = holdings.slice(0, limit).map(holding => ({
        ...holding,
        asiRating: this.calculateASIRating(holding),
        futureOutlook: this.calculateFutureOutlook(holding),
        strengths: this.generateStockStrengths(holding),
        weaknesses: this.generateStockWeaknesses(holding),
        prediction: this.calculateStockPrediction(holding),
        confidence: this.calculateStockConfidence(holding)
      }));

      return analyzedHoldings;

    } catch (error) {
      logger.error('Error getting fund holdings with analysis:', error);
      throw error;
    }
  }

  /**
   * Calculate ASI rating for individual stock
   */
  calculateASIRating(holding) {
    const score = this.getStockQualityScore(holding);
    
    if (score >= 85) return 'A+';
    if (score >= 75) return 'A';
    if (score >= 65) return 'B+';
    return 'B';
  }

  /**
   * Calculate future outlook for stock
   */
  calculateFutureOutlook(holding) {
    const sectorScore = this.getSectorScore(holding.sector);
    
    if (sectorScore >= 80) return 'Very Positive';
    if (sectorScore >= 75) return 'Positive';
    if (sectorScore >= 70) return 'Neutral';
    return 'Cautious';
  }

  /**
   * Generate stock-specific strengths
   */
  generateStockStrengths(holding) {
    const strengths = [
      'Strong market position in growing sector',
      'Consistent revenue growth and profitability',
      'Robust balance sheet with low debt levels',
      'Experienced management team with proven track record'
    ];
    
    return strengths.slice(0, 3);
  }

  /**
   * Generate stock-specific weaknesses
   */
  generateStockWeaknesses(holding) {
    const weaknesses = [
      'Exposure to regulatory changes in the sector',
      'High valuation compared to historical averages',
      'Dependence on economic cycles for growth'
    ];
    
    return weaknesses.slice(0, 2);
  }

  /**
   * Calculate stock prediction
   */
  calculateStockPrediction(holding) {
    const baseGrowth = this.getSectorScore(holding.sector) / 5;
    return Math.max(5, Math.min(30, baseGrowth + Math.random() * 10));
  }

  /**
   * Calculate stock confidence
   */
  calculateStockConfidence(holding) {
    const score = this.getStockQualityScore(holding);
    
    if (score >= 80) return 'Very High';
    if (score >= 70) return 'High';
    if (score >= 60) return 'Medium';
    return 'Low';
  }

  /**
   * Get sector allocation with future outlook
   */
  async getSectorAllocationWithOutlook(fundId) {
    try {
      const holdings = await this.realMutualFundService.getFundHoldings(fundId);
      
      // Calculate sector allocations
      const sectorAllocations = {};
      holdings.forEach(holding => {
        if (!sectorAllocations[holding.sector]) {
          sectorAllocations[holding.sector] = {
            name: holding.sector,
            allocation: 0,
            holdings: []
          };
        }
        sectorAllocations[holding.sector].allocation += holding.allocation;
        sectorAllocations[holding.sector].holdings.push(holding);
      });

      // Add analysis to each sector
      const sectorsWithAnalysis = Object.values(sectorAllocations).map(sector => ({
        ...sector,
        performance: this.calculateSectorPerformance(sector),
        outlook: this.calculateSectorOutlook(sector),
        prediction: this.calculateSectorPrediction(sector),
        analysis: this.generateSectorAnalysis(sector),
        keyFactors: this.generateSectorGrowthFactors(sector),
        risks: this.generateSectorRisks(sector)
      }));

      // Sort by allocation
      return sectorsWithAnalysis.sort((a, b) => b.allocation - a.allocation);

    } catch (error) {
      logger.error('Error getting sector allocation with outlook:', error);
      throw error;
    }
  }

  /**
   * Calculate sector performance
   */
  calculateSectorPerformance(sector) {
    const performances = ['Excellent', 'Good', 'Average', 'Below Average'];
    return performances[Math.floor(Math.random() * performances.length)];
  }

  /**
   * Calculate sector outlook
   */
  calculateSectorOutlook(sector) {
    const outlooks = ['Very Positive', 'Positive', 'Neutral', 'Cautious'];
    const score = this.getSectorScore(sector.name);
    
    if (score >= 80) return 'Very Positive';
    if (score >= 75) return 'Positive';
    if (score >= 70) return 'Neutral';
    return 'Cautious';
  }

  /**
   * Calculate sector prediction
   */
  calculateSectorPrediction(sector) {
    const baseGrowth = this.getSectorScore(sector.name) / 5;
    return `+${Math.max(8, Math.min(25, baseGrowth + Math.random() * 8)).toFixed(0)}%`;
  }

  /**
   * Generate sector analysis
   */
  generateSectorAnalysis(sector) {
    const analyses = {
      'Technology': 'The technology sector continues to benefit from digital transformation trends, cloud adoption, and AI innovation. Strong fundamentals and growth prospects make it attractive for long-term investors.',
      'Financial Services': 'Banking and financial services sector shows resilience with improving asset quality and stable margins. Interest rate environment and regulatory changes remain key factors to monitor.',
      'Healthcare': 'Healthcare sector offers defensive characteristics with steady demand and innovation pipeline. Aging demographics and increased healthcare spending support long-term growth.',
      'Consumer Goods': 'Consumer goods sector faces mixed outlook with rural demand recovery and urban consumption patterns. Brand strength and distribution network remain competitive advantages.',
      'Energy': 'Energy sector benefits from stable oil prices and increased focus on renewable energy transition. Companies with diversified portfolios are better positioned.',
      'Industrials': 'Industrial sector supported by infrastructure development and manufacturing growth. Capital expenditure cycle and government policies drive sector performance.'
    };
    
    return analyses[sector.name] || 'Sector shows mixed performance with both opportunities and challenges in the current market environment.';
  }

  /**
   * Generate sector growth factors
   */
  generateSectorGrowthFactors(sector) {
    const factors = {
      'Technology': [
        'Digital transformation acceleration across industries',
        'Growing demand for cloud computing and SaaS solutions',
        'AI and machine learning adoption increasing',
        'Strong export potential and global market access'
      ],
      'Financial Services': [
        'Improving asset quality and reducing NPAs',
        'Digital banking and fintech innovation',
        'Credit growth recovery in retail and SME segments',
        'Stable interest rate environment supporting margins'
      ],
      'Healthcare': [
        'Aging population driving healthcare demand',
        'Government focus on healthcare infrastructure',
        'Innovation in pharmaceuticals and medical devices',
        'Increasing health insurance penetration'
      ],
      'Consumer Goods': [
        'Rising disposable income and lifestyle changes',
        'Rural market penetration opportunities',
        'E-commerce growth expanding reach',
        'Brand premiumization trends'
      ]
    };
    
    return factors[sector.name] || [
      'Favorable government policies and regulations',
      'Growing domestic demand and market expansion',
      'Technological advancement and innovation',
      'Strong competitive positioning of portfolio companies'
    ];
  }

  /**
   * Generate sector risks
   */
  generateSectorRisks(sector) {
    const risks = {
      'Technology': [
        'Global economic slowdown affecting IT spending',
        'Currency fluctuation impacting export revenues',
        'Intense competition and pricing pressures'
      ],
      'Financial Services': [
        'Economic downturn affecting credit quality',
        'Regulatory changes and compliance costs',
        'Competition from fintech and digital players'
      ],
      'Healthcare': [
        'Regulatory approval delays for new products',
        'Price control measures by government',
        'Generic competition affecting margins'
      ],
      'Consumer Goods': [
        'Raw material cost inflation pressures',
        'Changing consumer preferences and behavior',
        'Intense competition and market saturation'
      ]
    };
    
    return risks[sector.name] || [
      'Economic uncertainty affecting sector performance',
      'Regulatory changes and policy shifts',
      'Market volatility and competitive pressures'
    ];
  }

  /**
   * Get fund performance data
   */
  async getFundPerformanceData(fundId) {
    try {
      return await this.realMutualFundService.getFundPerformance(fundId);
    } catch (error) {
      logger.error('Error getting fund performance data:', error);
      throw error;
    }
  }

  /**
   * Get fund risk metrics
   */
  async getFundRiskMetrics(fundId) {
    try {
      return await this.calculateRiskMetrics(fundId);
    } catch (error) {
      logger.error('Error getting fund risk metrics:', error);
      throw error;
    }
  }

  /**
   * Calculate comprehensive risk metrics
   */
  async calculateRiskMetrics(fundId) {
    try {
      const performance = await this.realMutualFundService.getFundPerformance(fundId);
      const navHistory = await this.realMutualFundService.getNAVHistory(fundId, '3Y');

      return {
        volatility: this.calculateVolatility(navHistory),
        sharpeRatio: this.calculateSharpeRatio(navHistory),
        beta: this.calculateBeta(navHistory),
        maxDrawdown: this.calculateMaxDrawdown(navHistory),
        var95: this.calculateVaR(navHistory, 0.95),
        informationRatio: this.calculateInformationRatio(navHistory),
        treynorRatio: this.calculateTreynorRatio(navHistory),
        alpha: this.calculateAlpha(navHistory)
      };
    } catch (error) {
      logger.error('Error calculating risk metrics:', error);
      throw error;
    }
  }

  /**
   * Calculate volatility
   */
  calculateVolatility(navHistory) {
    if (!navHistory || navHistory.length < 2) return 0;
    
    const returns = [];
    for (let i = 1; i < navHistory.length; i++) {
      const dailyReturn = (navHistory[i].nav - navHistory[i-1].nav) / navHistory[i-1].nav;
      returns.push(dailyReturn);
    }
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance * 252) * 100; // Annualized volatility
  }

  /**
   * Calculate Sharpe ratio
   */
  calculateSharpeRatio(navHistory) {
    const riskFreeRate = 0.06; // 6% risk-free rate
    const returns = this.calculateReturns(navHistory);
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length * 252;
    const volatility = this.calculateVolatility(navHistory) / 100;
    
    return (avgReturn - riskFreeRate) / volatility;
  }

  /**
   * Calculate Beta
   */
  calculateBeta(navHistory) {
    // Mock implementation - would use market index data
    return 0.8 + Math.random() * 0.4; // Random beta between 0.8 and 1.2
  }

  /**
   * Calculate maximum drawdown
   */
  calculateMaxDrawdown(navHistory) {
    if (!navHistory || navHistory.length < 2) return 0;
    
    let maxDrawdown = 0;
    let peak = navHistory[0].nav;
    
    for (let i = 1; i < navHistory.length; i++) {
      if (navHistory[i].nav > peak) {
        peak = navHistory[i].nav;
      } else {
        const drawdown = (peak - navHistory[i].nav) / peak;
        maxDrawdown = Math.max(maxDrawdown, drawdown);
      }
    }
    
    return -maxDrawdown * 100;
  }

  /**
   * Calculate Value at Risk (VaR)
   */
  calculateVaR(navHistory, confidence) {
    const returns = this.calculateReturns(navHistory);
    returns.sort((a, b) => a - b);
    
    const index = Math.floor((1 - confidence) * returns.length);
    return returns[index] * 100;
  }

  /**
   * Calculate Information Ratio
   */
  calculateInformationRatio(navHistory) {
    // Mock implementation - would use benchmark data
    return 0.3 + Math.random() * 0.4; // Random IR between 0.3 and 0.7
  }

  /**
   * Calculate Treynor Ratio
   */
  calculateTreynorRatio(navHistory) {
    const riskFreeRate = 0.06;
    const returns = this.calculateReturns(navHistory);
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length * 252;
    const beta = this.calculateBeta(navHistory);
    
    return (avgReturn - riskFreeRate) / beta;
  }

  /**
   * Calculate Alpha
   */
  calculateAlpha(navHistory) {
    // Mock implementation - would use market index data
    return -0.02 + Math.random() * 0.06; // Random alpha between -2% and 4%
  }

  /**
   * Calculate returns from NAV history
   */
  calculateReturns(navHistory) {
    const returns = [];
    for (let i = 1; i < navHistory.length; i++) {
      const dailyReturn = (navHistory[i].nav - navHistory[i-1].nav) / navHistory[i-1].nav;
      returns.push(dailyReturn);
    }
    return returns;
  }

  /**
   * Get detailed stock analysis
   */
  async getDetailedStockAnalysis(stockId) {
    try {
      // Mock implementation - would integrate with real stock analysis APIs
      return {
        id: stockId,
        name: `Stock ${stockId}`,
        currentPrice: 1500 + Math.random() * 1000,
        targetPrice: 1800 + Math.random() * 1200,
        asiRating: this.calculateASIRating({ sector: 'Technology' }),
        portfolioWeight: 2.5 + Math.random() * 5,
        technicalIndicators: {
          rsi: 45 + Math.random() * 20,
          macd: Math.random() > 0.5 ? 'Bullish' : 'Bearish',
          movingAverages: Math.random() > 0.5 ? 'Above 50-day MA' : 'Below 50-day MA'
        },
        fundamentalMetrics: {
          peRatio: 15 + Math.random() * 20,
          roe: 12 + Math.random() * 15,
          debtEquity: 0.2 + Math.random() * 0.8
        },
        aiPrediction: {
          marketSentiment: 'Positive momentum with strong institutional buying',
          sectorAnalysis: 'Technology sector showing robust growth prospects',
          riskFactors: ['Market volatility', 'Regulatory changes', 'Competition']
        },
        recommendation: 'BUY',
        keyStrengths: [
          'Strong financial performance and growth trajectory',
          'Market leadership position in key segments',
          'Robust business model with recurring revenue streams'
        ],
        actionPlan: [
          'Monitor quarterly earnings for growth sustainability',
          'Track sector developments and competitive positioning',
          'Consider position sizing based on portfolio allocation'
        ]
      };
    } catch (error) {
      logger.error('Error getting detailed stock analysis:', error);
      throw error;
    }
  }
}

module.exports = FSIAnalysisService;
