const agiScenarioService = require('../services/agiScenarioService');
const logger = require('../utils/logger');

class AGIScenarioController {
  async simulateScenario(req, res, next) {
    try {
      const { scenarioType, portfolio } = req.body;
      if (!scenarioType || !portfolio) {
        return res.status(400).json({ success: false, message: 'scenarioType and portfolio are required' });
      }
      const result = await agiScenarioService.simulateScenario(scenarioType, portfolio);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Scenario simulation error', { scenarioType: req.body.scenarioType, error: err.message });
      next(err);
    }
  }
  async advancedScenarioAnalyticsEndpoint(req, res, next) {
    try {
      const { scenarioData } = req.body;
      if (!scenarioData) {
        return res.status(400).json({ success: false, message: 'scenarioData is required' });
      }
      const result = await agiScenarioService.advancedScenarioAnalytics(scenarioData);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Advanced scenario analytics error', { error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIScenarioController();
