const agiMacroService = require('../services/agiMacroService');
const logger = require('../utils/logger');
const { validateObjectId } = require('../middleware/validation');

class AGIMacroController {
  async analyzeMacroeconomicImpact(req, res, next) {
    try {
      const { userId } = req.params;
      if (!validateObjectId(userId)) {
        return res.status(400).json({ success: false, message: 'Invalid user ID' });
      }
      const result = await agiMacroService.analyzeImpact(userId);
      res.json({ success: true, data: result });
    } catch (err) {
      logger.error('Macro impact error', { userId: req.params.userId, error: err.message });
      next(err);
    }
  }
}

module.exports = new AGIMacroController();
