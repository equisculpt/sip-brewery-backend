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
          data: result.data,
          message: 'Language preference set successfully'
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

  /**
   * Get user language preferences
   * @route GET /api/regional/language-preference
   */
  async getLanguagePreference(req, res) {
    try {
      const { userId } = req.user;

      logger.info('Getting language preference', { userId });

      const result = await regionalLanguageService.getUserLanguagePreferences(userId);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Language preferences retrieved successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in getLanguagePreference controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Translate investment terms
   * @route POST /api/regional/translate
   */
  async translateTerms(req, res) {
    try {
      const { terms, targetLanguage } = req.body;

      logger.info('Translating investment terms', { targetLanguage, termsCount: terms?.length });

      if (!terms || !Array.isArray(terms) || !targetLanguage) {
        return res.status(400).json({
          success: false,
          message: 'Terms array and target language are required'
        });
      }

      const result = await regionalLanguageService.translateInvestmentTerms(terms, targetLanguage);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Terms translated successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in translateTerms controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get cultural context
   * @route GET /api/regional/cultural-context/:languageCode
   */
  async getCulturalContext(req, res) {
    try {
      const { languageCode } = req.params;

      logger.info('Getting cultural context', { languageCode });

      if (!languageCode) {
        return res.status(400).json({
          success: false,
          message: 'Language code is required'
        });
      }

      const result = await regionalLanguageService.getCulturalContext(languageCode);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Cultural context retrieved successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in getCulturalContext controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Process voice command
   * @route POST /api/regional/voice-command
   */
  async processVoiceCommand(req, res) {
    try {
      const { userId } = req.user;
      const { audioData, languageCode } = req.body;

      logger.info('Processing voice command', { userId, languageCode });

      if (!audioData || !languageCode) {
        return res.status(400).json({
          success: false,
          message: 'Audio data and language code are required'
        });
      }

      const result = await regionalLanguageService.processVoiceCommand(
        audioData,
        languageCode,
        userId
      );

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Voice command processed successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in processVoiceCommand controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Generate localized content
   * @route POST /api/regional/localized-content
   */
  async generateLocalizedContent(req, res) {
    try {
      const { contentType, languageCode, context } = req.body;

      logger.info('Generating localized content', { contentType, languageCode });

      if (!contentType || !languageCode) {
        return res.status(400).json({
          success: false,
          message: 'Content type and language code are required'
        });
      }

      const result = await regionalLanguageService.generateLocalizedContent(
        contentType,
        languageCode,
        context
      );

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Localized content generated successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in generateLocalizedContent controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Get regional investment preferences
   * @route GET /api/regional/investment-preferences/:languageCode
   */
  async getRegionalInvestmentPreferences(req, res) {
    try {
      const { languageCode } = req.params;

      logger.info('Getting regional investment preferences', { languageCode });

      if (!languageCode) {
        return res.status(400).json({
          success: false,
          message: 'Language code is required'
        });
      }

      const result = await regionalLanguageService.getRegionalInvestmentPreferences(languageCode);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Regional investment preferences retrieved successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in getRegionalInvestmentPreferences controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }

  /**
   * Create voice commands
   * @route POST /api/regional/voice-commands
   */
  async createVoiceCommands(req, res) {
    try {
      const { languageCode } = req.body;

      logger.info('Creating voice commands', { languageCode });

      if (!languageCode) {
        return res.status(400).json({
          success: false,
          message: 'Language code is required'
        });
      }

      const result = await regionalLanguageService.createVoiceCommands(languageCode);

      if (result.success) {
        res.status(200).json({
          success: true,
          data: result.data,
          message: 'Voice commands created successfully'
        });
      } else {
        res.status(400).json({
          success: false,
          message: result.message,
          error: result.error
        });
      }
    } catch (error) {
      logger.error('Error in createVoiceCommands controller', { error: error.message });
      res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: error.message
      });
    }
  }
}

module.exports = new RegionalLanguageController(); 