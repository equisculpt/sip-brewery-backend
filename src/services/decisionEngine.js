/**
 * DecisionEngine: Centralized rules + AI-based dynamic allocation and scenario simulation
 * Extends AGI system for Financial AGI blueprint
 */
const portfolioOptimizer = require('./aiPortfolioOptimizer');
const predictiveEngine = require('./predictiveEngine');
const complianceEngine = require('./complianceEngine');
const riskProfilingService = require('./riskProfilingService');
const logger = require('../utils/logger');

class DecisionEngine {
  /**
   * Dynamic SIP/portfolio allocation
   * @param {Object} userProfile
   * @param {Object} marketState
   * @param {Array} assets
   * @return {Object} allocation
   */
  async getDynamicAllocation(userProfile, marketState, assets) {
    // 1. Predict returns and risk for each asset
    const predictions = await predictiveEngine.forecastAssets(assets, marketState);
    const risks = await riskProfilingService.assessAssets(assets, userProfile);
    // 2. Optimize allocation based on predictions, risk, compliance
    const allocation = portfolioOptimizer.optimize({ userProfile, predictions, risks, assets });
    // 3. Check compliance
    const compliance = complianceEngine.checkPortfolioCompliance(allocation, userProfile);
    return { allocation, compliance };
  }

  /**
   * Scenario simulation (bull/bear/sideways)
   */
  async simulateScenario(userProfile, scenario, assets) {
    // Use predictiveEngine to simulate returns under scenario
    return predictiveEngine.simulateScenario(userProfile, scenario, assets);
  }
}

module.exports = new DecisionEngine();
