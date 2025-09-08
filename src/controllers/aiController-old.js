/**
 * 🤖 AI CONTROLLER
 * 
 * Advanced AI controller for mutual fund analysis, predictions, and recommendations
 * Integrates with continuous learning engine and real-time data
 * 
 * @author AI Founder with 100+ years team experience
 * @version 1.0.0 - Financial ASI
 */

const aiService = require('../services/aiService');
const response = require('../utils/response');
const logger = require('../utils/logger');

class AIController {
  /**
   * Analyze mutual funds using AI
   * @param {Object} req - Express request object
   * @param {Object} res - Express response object
   */
  async analyzeMutualFunds(req, res) {
    try {
      const { schemeCodes, query } = req.body;

      // Validate request body
      if (!schemeCodes || !Array.isArray(schemeCodes)) {
        return response.validationError(res, [
          { field: 'schemeCodes', message: 'Scheme codes must be an array' }
        ]);
      }

      if (!query || typeof query !== 'string') {
        return response.validationError(res, [
          { field: 'query', message: 'Query is required and must be a string' }
        ]);
      }

      if (schemeCodes.length === 0) {
        return response.validationError(res, [
          { field: 'schemeCodes', message: 'At least one scheme code is required' }
        ]);
      }

      // Validate scheme codes format
      const invalidCodes = schemeCodes.filter(code => 
        !code || typeof code !== 'string' || code.trim() === ''
      );

      if (invalidCodes.length > 0) {
        return response.validationError(res, [
          { field: 'schemeCodes', message: 'All scheme codes must be valid strings' }
        ]);
      }

      logger.info('Starting mutual fund analysis', {
        schemeCodes,
        queryLength: query.length
      });

      // Process the analysis using the upgraded function
      const result = await aiService.analyzeFundWithNAV(schemeCodes, query);

      if (!result.success) {
        logger.error('AI analysis failed', { error: result.error });
        return response.error(res, result.error, 500);
      }

      logger.info('Mutual fund analysis completed successfully', {
        fundsAnalyzed: result.funds.length,
        analysisLength: result.analysis?.length || 0
      });

      return response.success(res, 'Mutual fund analysis completed', result);
    } catch (error) {
      logger.error('Error in analyzeMutualFunds:', error);
      return response.error(res, 'Failed to analyze mutual funds', error.message);
    }
  }
}

module.exports = new AIController();
