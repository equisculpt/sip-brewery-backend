/**
 * BehavioralRecommendationEngine: User behavioral analysis and nudge engine
 * Part of Financial AGI extension
 */
const logger = require('../utils/logger');

class BehavioralRecommendationEngine {
  /**
   * Generate behavioral nudges based on user actions and market events
   * @param {Object} userProfile
   * @param {Object} userActions
   * @param {Object} marketEvents
   * @return {Array} recommendations
   */
  generateNudges(userProfile, userActions, marketEvents) {
    const nudges = [];
    // Panic selling
    if (userActions.panicSelling) {
      nudges.push('Based on your profile, staying invested may yield better results long-term.');
    }
    // Underinvested
    if (userProfile.cashBalance > 0 && !userProfile.activeSIP) {
      nudges.push('You have unused cash. Consider increasing SIP for higher growth.');
    }
    // Volatile market
    if (marketEvents.volatility && marketEvents.volatility > 20) {
      nudges.push('Market swings are normal. Diversification can help reduce risk.');
    }
    // Customizable for more behavioral patterns...
    return nudges;
  }
}

module.exports = new BehavioralRecommendationEngine();
