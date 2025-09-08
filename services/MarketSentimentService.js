const logger = require('../utils/logger');
const RealTimeMarketService = require('./RealTimeMarketService');

class MarketSentimentService {
  constructor() {
    this.realTimeMarketService = new RealTimeMarketService();
    
    console.log('🚀 Market Sentiment Service Initialized - AI-Powered Market Analysis');
  }

  /**
   * Get current market sentiment analysis
   */
  async getCurrentMarketSentiment() {
    try {
      logger.info('🔍 Analyzing current market sentiment...');

      // Get real-time market data
      const marketData = await this.realTimeMarketService.getCurrentMarketData();
      
      // Analyze sentiment across different parameters
      const sentiment = {
        overallMarket: await this.analyzeOverallMarketSentiment(marketData),
        sectorPerformance: await this.analyzeSectorPerformance(marketData),
        fundCategory: await this.analyzeFundCategorySentiment(marketData),
        volatilityIndex: await this.analyzeVolatilityIndex(marketData),
        timingScore: await this.calculateTimingScore(marketData)
      };

      logger.info('✅ Market sentiment analysis completed');
      return sentiment;

    } catch (error) {
      logger.error('❌ Error analyzing market sentiment:', error);
      throw error;
    }
  }

  /**
   * Analyze overall market sentiment
   */
  async analyzeOverallMarketSentiment(marketData) {
    try {
      // Mock implementation - would use real market indicators
      const niftyChange = -0.5 + Math.random() * 3; // Random change between -0.5% and 2.5%
      const volumeRatio = 0.8 + Math.random() * 0.4; // Volume ratio
      const advanceDeclineRatio = 0.6 + Math.random() * 0.8; // A/D ratio

      // Calculate sentiment score
      let sentimentScore = 50; // Neutral baseline
      
      // Market performance impact
      if (niftyChange > 1.5) sentimentScore += 20;
      else if (niftyChange > 0.5) sentimentScore += 10;
      else if (niftyChange < -1) sentimentScore -= 15;
      else if (niftyChange < -0.5) sentimentScore -= 8;

      // Volume impact
      if (volumeRatio > 1.2) sentimentScore += 10;
      else if (volumeRatio < 0.8) sentimentScore -= 5;

      // Advance/Decline impact
      if (advanceDeclineRatio > 1.5) sentimentScore += 15;
      else if (advanceDeclineRatio < 0.8) sentimentScore -= 10;

      sentimentScore = Math.max(0, Math.min(100, sentimentScore));

      return {
        value: this.getSentimentLabel(sentimentScore),
        percentage: Math.round(sentimentScore),
        trend: niftyChange > 0 ? 'Bullish' : 'Bearish',
        confidence: this.getConfidenceLevel(sentimentScore)
      };

    } catch (error) {
      logger.error('Error analyzing overall market sentiment:', error);
      throw error;
    }
  }

  /**
   * Analyze sector performance sentiment
   */
  async analyzeSectorPerformance(marketData) {
    try {
      // Mock sector performance data
      const sectorPerformances = {
        'Technology': 1.8,
        'Banking': 0.5,
        'Pharma': -0.3,
        'Auto': 1.2,
        'FMCG': 0.8,
        'Energy': -0.8,
        'Metals': 2.1,
        'Realty': -1.2
      };

      // Calculate weighted sector sentiment
      const positiveSectors = Object.values(sectorPerformances).filter(perf => perf > 0).length;
      const totalSectors = Object.keys(sectorPerformances).length;
      const avgPerformance = Object.values(sectorPerformances).reduce((sum, perf) => sum + perf, 0) / totalSectors;

      let sectorScore = 50 + (avgPerformance * 20); // Base score + performance impact
      sectorScore += (positiveSectors / totalSectors) * 30; // Breadth impact

      sectorScore = Math.max(0, Math.min(100, sectorScore));

      return {
        value: this.getSentimentLabel(sectorScore),
        percentage: Math.round(sectorScore),
        trend: avgPerformance > 0 ? 'Positive' : 'Mixed',
        confidence: this.getConfidenceLevel(sectorScore)
      };

    } catch (error) {
      logger.error('Error analyzing sector performance:', error);
      throw error;
    }
  }

  /**
   * Analyze fund category sentiment
   */
  async analyzeFundCategorySentiment(marketData) {
    try {
      // Mock fund category flows and performance
      const equityFlows = 500 + Math.random() * 1000; // Crores
      const debtFlows = 200 + Math.random() * 500;
      const hybridFlows = 100 + Math.random() * 300;

      const totalFlows = equityFlows + debtFlows + hybridFlows;
      const equityShare = equityFlows / totalFlows;

      let categoryScore = 40; // Conservative baseline for mutual funds
      
      // Equity preference indicates risk appetite
      if (equityShare > 0.7) categoryScore += 25;
      else if (equityShare > 0.5) categoryScore += 15;
      else if (equityShare < 0.3) categoryScore -= 10;

      // Total flows indicate investor confidence
      if (totalFlows > 1000) categoryScore += 15;
      else if (totalFlows < 500) categoryScore -= 10;

      categoryScore = Math.max(0, Math.min(100, categoryScore));

      return {
        value: this.getFundCategoryLabel(categoryScore),
        percentage: Math.round(categoryScore),
        trend: equityShare > 0.5 ? 'Risk-On' : 'Risk-Off',
        confidence: this.getConfidenceLevel(categoryScore)
      };

    } catch (error) {
      logger.error('Error analyzing fund category sentiment:', error);
      throw error;
    }
  }

  /**
   * Analyze volatility index
   */
  async analyzeVolatilityIndex(marketData) {
    try {
      // Mock VIX-like volatility index
      const volatilityIndex = 12 + Math.random() * 20; // VIX between 12-32
      const historicalAvg = 18;

      let volatilityScore = 100 - ((volatilityIndex - 10) * 3); // Lower volatility = higher score
      volatilityScore = Math.max(0, Math.min(100, volatilityScore));

      let trend = 'Stable';
      if (volatilityIndex > historicalAvg + 5) trend = 'High Volatility';
      else if (volatilityIndex < historicalAvg - 3) trend = 'Low Volatility';

      return {
        value: `${volatilityIndex.toFixed(1)} VIX`,
        percentage: Math.round(volatilityScore),
        trend: trend,
        confidence: volatilityIndex < 20 ? 'High' : 'Medium'
      };

    } catch (error) {
      logger.error('Error analyzing volatility index:', error);
      throw error;
    }
  }

  /**
   * Calculate investment timing score
   */
  async calculateTimingScore(marketData) {
    try {
      let timingScore = 50; // Neutral baseline

      // Technical factors
      const technicalScore = this.calculateTechnicalScore();
      timingScore += technicalScore * 0.3;

      // Fundamental factors
      const fundamentalScore = this.calculateFundamentalScore();
      timingScore += fundamentalScore * 0.25;

      // Sentiment factors
      const sentimentScore = this.calculateSentimentScore();
      timingScore += sentimentScore * 0.25;

      // Macro factors
      const macroScore = this.calculateMacroScore();
      timingScore += macroScore * 0.2;

      timingScore = Math.max(0, Math.min(100, timingScore));

      return {
        value: `${(timingScore / 10).toFixed(1)}/10`,
        percentage: Math.round(timingScore),
        trend: timingScore > 60 ? 'Favorable' : timingScore > 40 ? 'Neutral' : 'Cautious',
        confidence: this.getConfidenceLevel(timingScore)
      };

    } catch (error) {
      logger.error('Error calculating timing score:', error);
      throw error;
    }
  }

  /**
   * Calculate technical score for timing
   */
  calculateTechnicalScore() {
    // Mock technical indicators
    const rsi = 45 + Math.random() * 20; // RSI between 45-65
    const macdSignal = Math.random() > 0.6 ? 'Bullish' : 'Bearish';
    const supportLevel = Math.random() > 0.7; // Above support

    let score = 0;
    
    if (rsi > 30 && rsi < 70) score += 10; // Neutral RSI is good for entry
    if (macdSignal === 'Bullish') score += 15;
    if (supportLevel) score += 10;

    return score;
  }

  /**
   * Calculate fundamental score for timing
   */
  calculateFundamentalScore() {
    // Mock fundamental indicators
    const peRatio = 18 + Math.random() * 10; // Market PE
    const earningsGrowth = 8 + Math.random() * 15; // Expected earnings growth
    const interestRates = 6 + Math.random() * 2; // Interest rate environment

    let score = 0;
    
    if (peRatio < 22) score += 10; // Reasonable valuation
    if (earningsGrowth > 12) score += 15; // Good earnings growth
    if (interestRates < 7) score += 8; // Favorable rates

    return score;
  }

  /**
   * Calculate sentiment score for timing
   */
  calculateSentimentScore() {
    // Mock sentiment indicators
    const fearGreedIndex = 30 + Math.random() * 40; // Fear & Greed index
    const institutionalActivity = Math.random() > 0.5 ? 'Buying' : 'Selling';
    const retailParticipation = 0.6 + Math.random() * 0.3; // Retail participation

    let score = 0;
    
    if (fearGreedIndex < 50) score += 12; // Fear creates opportunity
    if (institutionalActivity === 'Buying') score += 10;
    if (retailParticipation < 0.8) score += 8; // Not excessive retail participation

    return score;
  }

  /**
   * Calculate macro score for timing
   */
  calculateMacroScore() {
    // Mock macro indicators
    const gdpGrowth = 5.5 + Math.random() * 2; // GDP growth
    const inflation = 4 + Math.random() * 3; // Inflation rate
    const fiscalPolicy = Math.random() > 0.6 ? 'Supportive' : 'Neutral';

    let score = 0;
    
    if (gdpGrowth > 6) score += 12; // Strong growth
    if (inflation < 6) score += 10; // Controlled inflation
    if (fiscalPolicy === 'Supportive') score += 8;

    return score;
  }

  /**
   * Get sentiment label based on score
   */
  getSentimentLabel(score) {
    if (score >= 80) return 'Very Bullish';
    if (score >= 65) return 'Bullish';
    if (score >= 50) return 'Neutral';
    if (score >= 35) return 'Bearish';
    return 'Very Bearish';
  }

  /**
   * Get fund category label based on score
   */
  getFundCategoryLabel(score) {
    if (score >= 75) return 'Strong Inflows';
    if (score >= 60) return 'Positive Flows';
    if (score >= 45) return 'Stable Flows';
    if (score >= 30) return 'Weak Flows';
    return 'Outflows';
  }

  /**
   * Get confidence level based on score
   */
  getConfidenceLevel(score) {
    if (score >= 80 || score <= 20) return 'Very High';
    if (score >= 70 || score <= 30) return 'High';
    if (score >= 60 || score <= 40) return 'Medium';
    return 'Low';
  }

  /**
   * Get historical market sentiment data
   */
  async getHistoricalMarketSentiment(period = '1M') {
    try {
      // Mock historical sentiment data
      const dataPoints = this.generateHistoricalSentimentData(period);
      
      return {
        period,
        data: dataPoints,
        summary: {
          avgSentiment: dataPoints.reduce((sum, point) => sum + point.sentiment, 0) / dataPoints.length,
          volatility: this.calculateSentimentVolatility(dataPoints),
          trend: this.calculateSentimentTrend(dataPoints)
        }
      };

    } catch (error) {
      logger.error('Error getting historical market sentiment:', error);
      throw error;
    }
  }

  /**
   * Generate historical sentiment data
   */
  generateHistoricalSentimentData(period) {
    const days = period === '1M' ? 30 : period === '3M' ? 90 : 365;
    const dataPoints = [];
    
    let baseSentiment = 50 + Math.random() * 20; // Starting sentiment
    
    for (let i = 0; i < days; i++) {
      // Add some randomness and trend
      const change = (Math.random() - 0.5) * 10;
      baseSentiment = Math.max(20, Math.min(80, baseSentiment + change));
      
      dataPoints.push({
        date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
        sentiment: Math.round(baseSentiment),
        label: this.getSentimentLabel(baseSentiment)
      });
    }
    
    return dataPoints;
  }

  /**
   * Calculate sentiment volatility
   */
  calculateSentimentVolatility(dataPoints) {
    const sentiments = dataPoints.map(point => point.sentiment);
    const mean = sentiments.reduce((sum, val) => sum + val, 0) / sentiments.length;
    const variance = sentiments.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / sentiments.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Calculate sentiment trend
   */
  calculateSentimentTrend(dataPoints) {
    const firstHalf = dataPoints.slice(0, Math.floor(dataPoints.length / 2));
    const secondHalf = dataPoints.slice(Math.floor(dataPoints.length / 2));
    
    const firstAvg = firstHalf.reduce((sum, point) => sum + point.sentiment, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((sum, point) => sum + point.sentiment, 0) / secondHalf.length;
    
    if (secondAvg > firstAvg + 5) return 'Improving';
    if (secondAvg < firstAvg - 5) return 'Deteriorating';
    return 'Stable';
  }

  /**
   * Get sector-wise sentiment analysis
   */
  async getSectorSentimentAnalysis() {
    try {
      const sectors = [
        'Technology', 'Banking', 'Pharmaceuticals', 'Automobile', 
        'FMCG', 'Energy', 'Metals', 'Real Estate'
      ];

      const sectorSentiments = sectors.map(sector => ({
        sector,
        sentiment: 30 + Math.random() * 40, // Random sentiment between 30-70
        trend: Math.random() > 0.5 ? 'Positive' : 'Negative',
        keyDrivers: this.getSectorDrivers(sector),
        outlook: this.getSectorOutlook(sector)
      }));

      return sectorSentiments.sort((a, b) => b.sentiment - a.sentiment);

    } catch (error) {
      logger.error('Error getting sector sentiment analysis:', error);
      throw error;
    }
  }

  /**
   * Get sector-specific drivers
   */
  getSectorDrivers(sector) {
    const drivers = {
      'Technology': ['Digital transformation', 'Cloud adoption', 'AI innovation'],
      'Banking': ['Credit growth', 'Asset quality', 'Interest rates'],
      'Pharmaceuticals': ['New drug approvals', 'Export demand', 'Generic competition'],
      'Automobile': ['Rural demand', 'EV transition', 'Commodity prices'],
      'FMCG': ['Rural recovery', 'Urban consumption', 'Input costs'],
      'Energy': ['Oil prices', 'Renewable transition', 'Government policies'],
      'Metals': ['Infrastructure spending', 'Global demand', 'Raw material costs'],
      'Real Estate': ['Interest rates', 'Policy support', 'Demand recovery']
    };

    return drivers[sector] || ['Market dynamics', 'Regulatory environment', 'Economic factors'];
  }

  /**
   * Get sector outlook
   */
  getSectorOutlook(sector) {
    const outlooks = ['Very Positive', 'Positive', 'Neutral', 'Cautious'];
    return outlooks[Math.floor(Math.random() * outlooks.length)];
  }
}

module.exports = MarketSentimentService;
