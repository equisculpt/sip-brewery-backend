const voiceBot = require('../services/voiceBot');
const logger = require('../utils/logger');
const { successResponse, errorResponse } = require('../utils/response');
const { User } = require('../models');

class VoiceBotController {
  /**
   * Analyze voice input and extract investment intent
   */
  async analyzeVoice(req, res) {
    try {
      const { userId } = req.user;
      const { audioData, language = 'en' } = req.body;

      // Validate input
      if (!audioData) {
        return errorResponse(res, 'Audio data is required', null, 400);
      }

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Analyze voice using voice bot service
      const result = await voiceBot.analyzeVoice(userId, audioData, language);

      if (!result.success) {
        return errorResponse(res, result.message, result.error, 500);
      }

      logger.info('Voice analysis completed', { userId, language, confidence: result.data.confidence });

      return successResponse(res, 'Voice analysis completed successfully', result.data);
    } catch (error) {
      logger.error('Voice analysis controller error', { error: error.message });
      return errorResponse(res, 'Failed to analyze voice input', { error: error.message });
    }
  }

  /**
   * Process voice command for investment actions
   */
  async processVoiceCommand(req, res) {
    try {
      const { userId } = req.user;
      const { command, language = 'en' } = req.body;

      // Validate input
      if (!command) {
        return errorResponse(res, 'Voice command is required', null, 400);
      }

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Process voice command
      const result = await voiceBot.processVoiceCommand(userId, command, language);

      if (!result.success) {
        return errorResponse(res, result.message, result.error, 500);
      }

      logger.info('Voice command processed', { userId, command, action: result.data.action });

      return successResponse(res, 'Voice command processed successfully', result.data);
    } catch (error) {
      logger.error('Voice command processing controller error', { error: error.message });
      return errorResponse(res, 'Failed to process voice command', { error: error.message });
    }
  }

  /**
   * Handle Hindi voice input specifically
   */
  async handleHindiVoice(req, res) {
    try {
      const { userId } = req.user;
      const { audioData } = req.body;

      // Validate input
      if (!audioData) {
        return errorResponse(res, 'Audio data is required', null, 400);
      }

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Process Hindi voice
      const result = await voiceBot.handleHindiVoice(userId, audioData);

      if (!result.success) {
        return errorResponse(res, result.message, result.error, 500);
      }

      logger.info('Hindi voice processed', { userId, transcription: result.data.transcription });

      return successResponse(res, 'Hindi voice processed successfully', result.data);
    } catch (error) {
      logger.error('Hindi voice processing controller error', { error: error.message });
      return errorResponse(res, 'Failed to process Hindi voice', { error: error.message });
    }
  }

  /**
   * Multi-language voice support
   */
  async handleMultiLanguageVoice(req, res) {
    try {
      const { userId } = req.user;
      const { audioData, preferredLanguage = 'en' } = req.body;

      // Validate input
      if (!audioData) {
        return errorResponse(res, 'Audio data is required', null, 400);
      }

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Process multi-language voice
      const result = await voiceBot.handleMultiLanguageVoice(userId, audioData, preferredLanguage);

      if (!result.success) {
        return errorResponse(res, result.message, result.error, 500);
      }

      logger.info('Multi-language voice processed', { 
        userId, 
        detectedLanguage: result.data.detectedLanguage,
        processedLanguage: result.data.processedLanguage 
      });

      return successResponse(res, 'Multi-language voice processed successfully', result.data);
    } catch (error) {
      logger.error('Multi-language voice processing controller error', { error: error.message });
      return errorResponse(res, 'Failed to process multi-language voice', { error: error.message });
    }
  }

  /**
   * Get supported languages
   */
  async getSupportedLanguages(req, res) {
    try {
      const languages = voiceBot.supportedLanguages.map(lang => ({
        code: lang,
        name: this.getLanguageName(lang),
        nativeName: this.getNativeLanguageName(lang)
      }));

      return successResponse(res, 'Supported languages retrieved successfully', { languages });
    } catch (error) {
      logger.error('Get supported languages controller error', { error: error.message });
      return errorResponse(res, 'Failed to get supported languages', { error: error.message });
    }
  }

  /**
   * Get voice processing configuration
   */
  async getVoiceConfig(req, res) {
    try {
      const config = {
        ...voiceBot.voiceProcessingConfig,
        supportedLanguages: voiceBot.supportedLanguages,
        actionIntents: Object.keys(voiceBot.actionIntents)
      };

      return successResponse(res, 'Voice configuration retrieved successfully', config);
    } catch (error) {
      logger.error('Get voice config controller error', { error: error.message });
      return errorResponse(res, 'Failed to get voice configuration', { error: error.message });
    }
  }

  /**
   * Test voice processing with sample data
   */
  async testVoiceProcessing(req, res) {
    try {
      const { userId } = req.user;
      const { testCommand, language = 'en' } = req.body;

      // Validate input
      if (!testCommand) {
        return errorResponse(res, 'Test command is required', null, 400);
      }

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Process test command
      const result = await voiceBot.processVoiceCommand(userId, testCommand, language);

      if (!result.success) {
        return errorResponse(res, result.message, result.error, 500);
      }

      logger.info('Voice processing test completed', { userId, testCommand, result: result.data.action });

      return successResponse(res, 'Voice processing test completed successfully', result.data);
    } catch (error) {
      logger.error('Voice processing test controller error', { error: error.message });
      return errorResponse(res, 'Failed to test voice processing', { error: error.message });
    }
  }

  /**
   * Get voice interaction history
   */
  async getVoiceHistory(req, res) {
    try {
      const { userId } = req.user;
      const { limit = 10, offset = 0 } = req.query;

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // In real implementation, fetch from voice interaction database
      const mockHistory = [
        {
          id: '1',
          timestamp: new Date().toISOString(),
          command: 'Check my portfolio status',
          action: 'PORTFOLIO',
          language: 'en',
          confidence: 0.9,
          response: 'Your portfolio is worth ₹1,000,000'
        },
        {
          id: '2',
          timestamp: new Date(Date.now() - 86400000).toISOString(),
          command: 'मेरा पोर्टफोलियो दिखाएं',
          action: 'PORTFOLIO',
          language: 'hi',
          confidence: 0.8,
          response: 'आपका पोर्टफोलियो ₹10,00,000 का है'
        }
      ];

      const history = mockHistory.slice(offset, offset + parseInt(limit));

      return successResponse(res, 'Voice history retrieved successfully', {
        history,
        total: mockHistory.length,
        limit: parseInt(limit),
        offset: parseInt(offset)
      });
    } catch (error) {
      logger.error('Get voice history controller error', { error: error.message });
      return errorResponse(res, 'Failed to get voice history', { error: error.message });
    }
  }

  /**
   * Get voice analytics
   */
  async getVoiceAnalytics(req, res) {
    try {
      const { userId } = req.user;
      const { period = '30d' } = req.query;

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Mock analytics data
      const analytics = {
        totalInteractions: 45,
        successfulInteractions: 42,
        successRate: 93.3,
        averageConfidence: 0.85,
        mostUsedLanguage: 'en',
        mostUsedAction: 'PORTFOLIO',
        languageBreakdown: {
          en: 30,
          hi: 12,
          ta: 3
        },
        actionBreakdown: {
          PORTFOLIO: 15,
          BUY: 8,
          SELL: 5,
          SIP: 7,
          RECOMMENDATION: 5,
          MARKET: 3,
          HELP: 2
        },
        timeOfDayUsage: {
          morning: 12,
          afternoon: 18,
          evening: 10,
          night: 5
        }
      };

      return successResponse(res, 'Voice analytics retrieved successfully', analytics);
    } catch (error) {
      logger.error('Get voice analytics controller error', { error: error.message });
      return errorResponse(res, 'Failed to get voice analytics', { error: error.message });
    }
  }

  /**
   * Update voice preferences
   */
  async updateVoicePreferences(req, res) {
    try {
      const { userId } = req.user;
      const { preferredLanguage, voiceSpeed, voiceGender, autoLanguageDetection } = req.body;

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // Validate language
      if (preferredLanguage && !voiceBot.supportedLanguages.includes(preferredLanguage)) {
        return errorResponse(res, 'Unsupported language', null, 400);
      }

      // In real implementation, update user preferences in database
      const preferences = {
        preferredLanguage: preferredLanguage || 'en',
        voiceSpeed: voiceSpeed || 1.0,
        voiceGender: voiceGender || 'neutral',
        autoLanguageDetection: autoLanguageDetection !== undefined ? autoLanguageDetection : true
      };

      logger.info('Voice preferences updated', { userId, preferences });

      return successResponse(res, 'Voice preferences updated successfully', preferences);
    } catch (error) {
      logger.error('Update voice preferences controller error', { error: error.message });
      return errorResponse(res, 'Failed to update voice preferences', { error: error.message });
    }
  }

  /**
   * Get voice preferences
   */
  async getVoicePreferences(req, res) {
    try {
      const { userId } = req.user;

      // Validate user
      const user = await User.findById(userId);
      if (!user) {
        return errorResponse(res, 'User not found', null, 404);
      }

      // In real implementation, fetch from user preferences
      const preferences = {
        preferredLanguage: 'en',
        voiceSpeed: 1.0,
        voiceGender: 'neutral',
        autoLanguageDetection: true
      };

      return successResponse(res, 'Voice preferences retrieved successfully', preferences);
    } catch (error) {
      logger.error('Get voice preferences controller error', { error: error.message });
      return errorResponse(res, 'Failed to get voice preferences', { error: error.message });
    }
  }

  // Helper methods
  getLanguageName(code) {
    const languageNames = {
      en: 'English',
      hi: 'Hindi',
      ta: 'Tamil',
      bn: 'Bengali',
      te: 'Telugu',
      mr: 'Marathi',
      gu: 'Gujarati',
      kn: 'Kannada',
      ml: 'Malayalam',
      pa: 'Punjabi'
    };
    return languageNames[code] || code;
  }

  getNativeLanguageName(code) {
    const nativeNames = {
      en: 'English',
      hi: 'हिंदी',
      ta: 'தமிழ்',
      bn: 'বাংলা',
      te: 'తెలుగు',
      mr: 'मराठी',
      gu: 'ગુજરાતી',
      kn: 'ಕನ್ನಡ',
      ml: 'മലയാളം',
      pa: 'ਪੰਜਾਬੀ'
    };
    return nativeNames[code] || code;
  }
}

module.exports = new VoiceBotController(); 