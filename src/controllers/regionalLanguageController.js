const regionalLanguageService = require('../services/regionalLanguageService');
const logger = require('../utils/logger');

class RegionalLanguageController {
  /**
   * Get supported languages
   * @route GET /api/regional/languages
   */
  async getSupportedLanguages(req, res) {
    try {
      logger.info('Getting supported languages');
      const result = await regionalLanguageService.getSupportedLanguages();
      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Supported languages retrieved successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in getSupportedLanguages controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
  /**
   * Set user language preference
   * @route POST /api/regional/language-preference
   */
  async setLanguagePreference(req, res) {
    try {
      const { userId } = req.user;
      const { languageCode, preferences } = req.body;
      logger.info('Setting language preference', { userId, languageCode });
      if (!languageCode) {
        return res.status(400).json({
          success: false,
          message: 'Language code is required'
        });
      }
      const result = await regionalLanguageService.setUserLanguagePreference(
        userId,
        languageCode,
        preferences
      );
      if (result.success) {
        res.status(200).json({
          success: true,
          message: 'Language preference set successfully',
          data: result.data
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in setLanguagePreference controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
}

module.exports = new RegionalLanguageController();
