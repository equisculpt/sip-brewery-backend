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

      return response.success(res, {
        analysis: result.analysis,
        funds: result.funds
      }, 'Mutual fund analysis completed successfully');

    } catch (error) {
      logger.error('Error in AI controller', { error: error.message });
      return response.error(res, 'Internal server error during analysis', 500, error);
    }
  }



  /**
   * Get health status of AI service
   * @param {Object} req - Express request object
   * @param {Object} res - Express response object
   */
  async getHealth(req, res) {
    try {
      const healthStatus = {
        service: 'AI Service',
        status: 'OK',
        timestamp: new Date().toISOString(),
        features: {
          mutualFundAnalysis: true,
          geminiIntegration: !!process.env.GEMINI_API_KEY,
          mfApiIntegration: true
        }
      };

      return response.success(res, healthStatus, 'AI service is healthy');
    } catch (error) {
      logger.error('Error in AI health check', { error: error.message });
      return response.error(res, 'AI service health check failed', 500);
    }
  }

  /**
   * Test mutual fund data fetching
   * @param {Object} req - Express request object
   * @param {Object} res - Express response object
   */
  async testMFDataFetch(req, res) {
    try {
      const { schemeCode } = req.params;

      if (!schemeCode) {
        return response.validationError(res, [
          { field: 'schemeCode', message: 'Scheme code is required' }
        ]);
      }

      logger.info('Testing MF data fetch', { schemeCode });

      const navData = await aiService.fetchNAVData(schemeCode);

      return response.success(res, navData, 'Mutual fund data fetched successfully');

    } catch (error) {
      logger.error('Error testing MF data fetch', { error: error.message });
      return response.error(res, 'Failed to fetch mutual fund data', 500, error);
    }
  }
}

module.exports = new AIController(); 