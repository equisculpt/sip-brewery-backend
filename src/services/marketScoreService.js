const axios = require('axios');
const logger = require('../utils/logger');

class MarketScoreService {
  constructor() {
    this.indicators = {
      peRatio: null,
      rsi: null,
      breakout: null,
      sentiment: null,
      fearGreedIndex: null,
      macd: null,
      volume: null
    };
  }

  /**
   * Calculate market score from -1 (overheated) to +1 (bottomed)
   */
  async calculateMarketScore() {
    try {
      logger.info('Calculating market score...');
      
      // Fetch current market indicators
      await this.fetchMarketIndicators();
      
      // Calculate score using multiple factors
      const score = this.computeMarketScore();
      
      // Generate market reason
      const reason = this.generateMarketReason(score);
      
      logger.info(`Market score calculated: ${score}, Reason: ${reason}`);
      
      return {
        score: score,
        reason: reason,
        indicators: { ...this.indicators },
        timestamp: new Date().toISOString()
      };
      
    } catch (error) {
      logger.error('Error calculating market score:', error);
      // Return dummy data as fallback
      return this.getDummyMarketScore();
    }
  }

  /**
   * Fetch current market indicators
   */
  async fetchMarketIndicators() {
    try {
      // In a real implementation, these would be fetched from market data APIs
      // For now, we'll use dummy data that simulates real market conditions
      
      // Simulate Nifty 50 P/E ratio (historical range: 15-25)
      this.indicators.peRatio = 19.2 + (Math.random() - 0.5) * 4;
      
      // Simulate RSI (0-100, <30 oversold, >70 overbought)
      this.indicators.rsi = 30 + Math.random() * 40;
      
      // Simulate breakout indicator
      this.indicators.breakout = Math.random() > 0.7;
      
      // Simulate sentiment (BULLISH, NEUTRAL, BEARISH)
      const sentiments = ['BULLISH', 'NEUTRAL', 'BEARISH'];
      this.indicators.sentiment = sentiments[Math.floor(Math.random() * sentiments.length)];
      
      // Simulate Fear & Greed Index (0-100, 0=fear, 100=greed)
      this.indicators.fearGreedIndex = Math.floor(Math.random() * 100);
      
      // Simulate MACD
      this.indicators.macd = (Math.random() - 0.5) * 2;
      
      // Simulate volume (normalized 0-1)
      this.indicators.volume = Math.random();
      
      logger.info('Market indicators fetched successfully');
      
    } catch (error) {
      logger.error('Error fetching market indicators:', error);
      // Set default values
      this.indicators = {
        peRatio: 19.2,
        rsi: 48,
        breakout: false,
        sentiment: 'NEUTRAL',
        fearGreedIndex: 50,
        macd: 0,
        volume: 0.5
      };
    }
  }

  /**
   * Compute market score based on indicators
   */
  computeMarketScore() {
    let score = 0;
    const weights = {
      peRatio: 0.3,
      rsi: 0.25,
      sentiment: 0.2,
      fearGreedIndex: 0.15,
      breakout: 0.1
    };

    // P/E Ratio analysis (lower P/E = better buying opportunity)
    const peScore = this.analyzePERatio(this.indicators.peRatio);
    score += peScore * weights.peRatio;

    // RSI analysis (lower RSI = oversold = better buying opportunity)
    const rsiScore = this.analyzeRSI(this.indicators.rsi);
    score += rsiScore * weights.rsi;

    // Sentiment analysis
    const sentimentScore = this.analyzeSentiment(this.indicators.sentiment);
    score += sentimentScore * weights.sentiment;

    // Fear & Greed Index analysis (lower = fear = better buying opportunity)
    const fearGreedScore = this.analyzeFearGreedIndex(this.indicators.fearGreedIndex);
    score += fearGreedScore * weights.fearGreedIndex;

    // Breakout analysis
    const breakoutScore = this.analyzeBreakout(this.indicators.breakout);
    score += breakoutScore * weights.breakout;

    // Clamp score between -1 and 1
    return Math.max(-1, Math.min(1, score));
  }

  /**
   * Analyze P/E Ratio
   */
  analyzePERatio(peRatio) {
    // P/E ratio analysis:
    // < 15: Very attractive (score: 0.8 to 1.0)
    // 15-18: Attractive (score: 0.4 to 0.8)
    // 18-22: Fair value (score: -0.2 to 0.4)
    // 22-25: Expensive (score: -0.6 to -0.2)
    // > 25: Very expensive (score: -1.0 to -0.6)
    
    if (peRatio < 15) {
      return 0.8 + (15 - peRatio) * 0.04; // 0.8 to 1.0
    } else if (peRatio < 18) {
      return 0.4 + (18 - peRatio) * 0.13; // 0.4 to 0.8
    } else if (peRatio < 22) {
      return -0.2 + (22 - peRatio) * 0.15; // -0.2 to 0.4
    } else if (peRatio < 25) {
      return -0.6 + (25 - peRatio) * 0.13; // -0.6 to -0.2
    } else {
      return -1.0 + (peRatio - 25) * 0.04; // -1.0 to -0.6
    }
  }

  /**
   * Analyze RSI
   */
  analyzeRSI(rsi) {
    // RSI analysis:
    // < 30: Oversold (score: 0.6 to 1.0)
    // 30-45: Slightly oversold (score: 0.2 to 0.6)
    // 45-55: Neutral (score: -0.2 to 0.2)
    // 55-70: Slightly overbought (score: -0.6 to -0.2)
    // > 70: Overbought (score: -1.0 to -0.6)
    
    if (rsi < 30) {
      return 0.6 + (30 - rsi) * 0.04; // 0.6 to 1.0
    } else if (rsi < 45) {
      return 0.2 + (45 - rsi) * 0.027; // 0.2 to 0.6
    } else if (rsi < 55) {
      return -0.2 + (55 - rsi) * 0.04; // -0.2 to 0.2
    } else if (rsi < 70) {
      return -0.6 + (70 - rsi) * 0.027; // -0.6 to -0.2
    } else {
      return -1.0 + (rsi - 70) * 0.04; // -1.0 to -0.6
    }
  }

  /**
   * Analyze sentiment
   */
  analyzeSentiment(sentiment) {
    switch (sentiment) {
      case 'BEARISH':
        return 0.6; // Bearish sentiment = good buying opportunity
      case 'NEUTRAL':
        return 0.0; // Neutral sentiment = neutral
      case 'BULLISH':
        return -0.6; // Bullish sentiment = expensive
      default:
        return 0.0;
    }
  }

  /**
   * Analyze Fear & Greed Index
   */
  analyzeFearGreedIndex(index) {
    // Fear & Greed Index analysis:
    // 0-25: Extreme Fear (score: 0.8 to 1.0)
    // 25-45: Fear (score: 0.4 to 0.8)
    // 45-55: Neutral (score: -0.2 to 0.4)
    // 55-75: Greed (score: -0.6 to -0.2)
    // 75-100: Extreme Greed (score: -1.0 to -0.6)
    
    if (index < 25) {
      return 0.8 + (25 - index) * 0.008; // 0.8 to 1.0
    } else if (index < 45) {
      return 0.4 + (45 - index) * 0.02; // 0.4 to 0.8
    } else if (index < 55) {
      return -0.2 + (55 - index) * 0.04; // -0.2 to 0.4
    } else if (index < 75) {
      return -0.6 + (75 - index) * 0.02; // -0.6 to -0.2
    } else {
      return -1.0 + (index - 75) * 0.008; // -1.0 to -0.6
    }
  }

  /**
   * Analyze breakout
   */
  analyzeBreakout(breakout) {
    // Breakout analysis:
    // True: Market breaking out (might be expensive)
    // False: No breakout (might be good buying opportunity)
    return breakout ? -0.3 : 0.3;
  }

  /**
   * Generate market reason based on score
   */
  generateMarketReason(score) {
    const pe = this.indicators.peRatio;
    const rsi = this.indicators.rsi;
    const sentiment = this.indicators.sentiment;
    const fearGreed = this.indicators.fearGreedIndex;

    if (score > 0.5) {
      return `Market appears oversold with P/E at ${pe.toFixed(1)}, RSI at ${rsi.toFixed(1)}, and ${sentiment.toLowerCase()} sentiment. Good opportunity to increase SIP.`;
    } else if (score > 0.2) {
      return `Market showing attractive valuations with P/E at ${pe.toFixed(1)} and RSI at ${rsi.toFixed(1)}. Consider moderate SIP increase.`;
    } else if (score > -0.2) {
      return `Market at fair value with P/E at ${pe.toFixed(1)} and RSI at ${rsi.toFixed(1)}. Maintain regular SIP amount.`;
    } else if (score > -0.5) {
      return `Market showing expensive valuations with P/E at ${pe.toFixed(1)} and RSI at ${rsi.toFixed(1)}. Consider reducing SIP amount.`;
    } else {
      return `Market appears overvalued with P/E at ${pe.toFixed(1)}, RSI at ${rsi.toFixed(1)}, and ${sentiment.toLowerCase()} sentiment. Reduce SIP amount.`;
    }
  }

  /**
   * Get dummy market score for testing
   */
  getDummyMarketScore() {
    const pe = 19.2;
    const rsi = 48;
    const breakout = false;

    let score = 0;
    if (pe < 20) score += 0.4;
    if (rsi < 45) score += 0.3;
    if (!breakout) score -= 0.2;

    score = Math.max(-1, Math.min(1, score));

    return {
      score: score,
      reason: `Dummy analysis: P/E at ${pe}, RSI at ${rsi}, ${breakout ? 'breakout' : 'no breakout'}`,
      indicators: {
        peRatio: pe,
        rsi: rsi,
        breakout: breakout,
        sentiment: 'NEUTRAL',
        fearGreedIndex: 50,
        macd: 0,
        volume: 0.5
      },
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Calculate recommended SIP amount based on market score
   */
  /**
   * Calculate recommended SIP amount based on market score and rolling average targeting
   * - Non-linear regime-based scaling
   * - Rolling average SIP targeting for 110% of user base
   */
  calculateRecommendedSIP(averageSip, minSip, maxSip, marketScore, rollingAvgSIP = null) {
    // Regime-based non-linear scaling
    let recommendedAmount;
    const targetAvg = Math.round(averageSip * 1.1); // 110% of user base
    // Add regime logic
    if (marketScore >= 0.7) {
      recommendedAmount = Math.min(maxSip * 1.1, maxSip + 0.1 * averageSip, targetAvg * 1.2);
    } else if (marketScore > 0.3) {
      recommendedAmount = maxSip;
    } else if (marketScore > -0.3) {
      recommendedAmount = averageSip;
    } else if (marketScore > -0.7) {
      recommendedAmount = minSip;
    } else {
      recommendedAmount = Math.max(minSip * 0.9, minSip - 0.1 * averageSip);
    }

    // Rolling average SIP targeting
    if (rollingAvgSIP !== null && !isNaN(rollingAvgSIP)) {
      if (rollingAvgSIP < targetAvg) {
        // Bias upward unless market is very high
        if (marketScore > -0.3) {
          recommendedAmount = Math.max(recommendedAmount, averageSip + (targetAvg - rollingAvgSIP));
        }
      } else if (rollingAvgSIP > targetAvg) {
        // Bias downward unless market is very low
        if (marketScore < 0.3) {
          recommendedAmount = Math.min(recommendedAmount, averageSip - (rollingAvgSIP - targetAvg));
        }
      }
    }

    // Clamp to min/max
    recommendedAmount = Math.max(minSip, Math.min(maxSip, recommendedAmount));
    // Add small randomization to avoid predictability
    const noise = Math.round((Math.random() - 0.5) * averageSip * 0.03 / 100) * 100; // ±1.5% of avg SIP
    recommendedAmount = Math.round((recommendedAmount + noise) / 100) * 100;
    return recommendedAmount;
  }

  /**
   * Get AI-powered market analysis (placeholder for future Gemini integration)
   */
  async getAIAnalysis() {
    // This would integrate with Gemini/OpenAI for advanced analysis
    const prompt = `Given the latest Nifty 50 chart, P/E ratio of ${this.indicators.peRatio}, RSI of ${this.indicators.rsi}, MACD of ${this.indicators.macd}, and global trend data, rate the current Indian equity market from -1 (overvalued) to +1 (undervalued). Justify in 2 lines. Also suggest SIP amount from ₹16,000 to ₹24,000.`;
    
    // For now, return a placeholder response
    return {
      aiScore: this.indicators.peRatio < 20 ? 0.3 : -0.2,
      aiReason: "AI analysis: Market showing moderate valuations with some upside potential based on technical indicators.",
      aiRecommendedAmount: this.indicators.peRatio < 20 ? 22000 : 18000
    };
  }
}

module.exports = new MarketScoreService(); 