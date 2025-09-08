const agiExplainService = require('../services/agiExplainService');
const logger = require('../utils/logger');

class AGIExplainController {
  async explain(req, res, next) {
    try {
      const { explanationRequest } = req.body;
      if (!explanationRequest) {
        return res.status(400).json({ success: false, message: 'explanationRequest is required' });
      }
      const result = await agiExplainService.explain(explanationRequest);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Explainability error', { error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIExplainController();
