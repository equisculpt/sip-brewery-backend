const logger = require('../utils/logger');
const { User, UserPortfolio, Holding } = require('../models');
const dashboardEngine = require('./dashboardEngine');
const portfolioOptimizer = require('./portfolioOptimizer');
const predictiveEngine = require('./predictiveEngine');

class VoiceBot {
  constructor() {
    this.supportedLanguages = ['en', 'hi', 'ta', 'bn', 'te', 'mr', 'gu', 'kn', 'ml', 'pa'];
    this.actionIntents = {
      BUY: 'buy',
      SELL: 'sell',
      INQUIRY: 'inquiry',
      PORTFOLIO: 'portfolio',
      RECOMMENDATION: 'recommendation',
      SIP: 'sip',
      GOAL: 'goal',
      TAX: 'tax',
      MARKET: 'market',
      HELP: 'help'
    };

    this.voiceProcessingConfig = {
      maxDuration: 30, // seconds
      supportedFormats: ['wav', 'mp3', 'm4a'],
      qualityThreshold: 0.7,
      languageDetection: true
    };
  }

  /**
   * Analyze voice input and extract investment intent
   */
  async analyzeVoice(userId, audioData, language = 'en') {
    try {
      logger.info('Starting voice analysis', { userId, language });

      // Validate audio data
      const validationResult = await this.validateAudioData(audioData);
      if (!validationResult.valid) {
        return {
          success: false,
          message: 'Invalid audio data',
          error: validationResult.error
        };
      }

      // Convert speech to text
      const transcription = await this.speechToText(audioData, language);
      if (!transcription.success) {
        return {
          success: false,
          message: 'Failed to transcribe speech',
          error: transcription.error
        };
      }

      // Analyze intent from transcribed text
      const intentAnalysis = await this.analyzeIntent(transcription.text, language);

      // Generate response based on intent
      const response = await this.generateResponse(userId, intentAnalysis, language);

      // Log the interaction
      await this.logVoiceInteraction(userId, {
        audioData: audioData.metadata,
        transcription: transcription.text,
        intent: intentAnalysis,
        response: response.data,
        language,
        timestamp: new Date()
      });

      return {
        success: true,
        data: {
          transcription: transcription.text,
          intent: intentAnalysis,
          response: response.data,
          confidence: intentAnalysis.confidence,
          language: language
        }
      };
    } catch (error) {
      logger.error('Voice analysis failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to analyze voice input',
        error: error.message
      };
    }
  }

  /**
   * Process voice command for investment actions
   */
  async processVoiceCommand(userId, command, language = 'en') {
    try {
      logger.info('Processing voice command', { userId, command, language });

      const intent = await this.analyzeIntent(command, language);
      
      switch (intent.action) {
        case this.actionIntents.BUY:
          return await this.processBuyCommand(userId, intent, language);

        case this.actionIntents.SELL:
          return await this.processSellCommand(userId, intent, language);

        case this.actionIntents.PORTFOLIO:
          return await this.processPortfolioCommand(userId, intent, language);

        case this.actionIntents.RECOMMENDATION:
          return await this.processRecommendationCommand(userId, intent, language);

        case this.actionIntents.SIP:
          return await this.processSIPCommand(userId, intent, language);

        case this.actionIntents.GOAL:
          return await this.processGoalCommand(userId, intent, language);

        case this.actionIntents.TAX:
          return await this.processTaxCommand(userId, intent, language);

        case this.actionIntents.MARKET:
          return await this.processMarketCommand(userId, intent, language);

        case this.actionIntents.HELP:
          return await this.processHelpCommand(userId, intent, language);

        default:
          return await this.processInquiryCommand(userId, intent, language);
      }
    } catch (error) {
      logger.error('Voice command processing failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to process voice command',
        error: error.message
      };
    }
  }

  /**
   * Handle Hindi voice input specifically
   */
  async handleHindiVoice(userId, audioData) {
    try {
      logger.info('Processing Hindi voice input', { userId });

      // Use Hindi-specific language model
      const transcription = await this.speechToText(audioData, 'hi');
      
      if (!transcription.success) {
        return {
          success: false,
          message: 'हिंदी में बोलने को समझ नहीं पाया। कृपया दोबारा बोलें।',
          error: transcription.error
        };
      }

      // Analyze Hindi intent
      const intentAnalysis = await this.analyzeHindiIntent(transcription.text);

      // Generate Hindi response
      const response = await this.generateHindiResponse(userId, intentAnalysis);

      return {
        success: true,
        data: {
          transcription: transcription.text,
          intent: intentAnalysis,
          response: response.data,
          language: 'hi'
        }
      };
    } catch (error) {
      logger.error('Hindi voice processing failed', { error: error.message, userId });
      return {
        success: false,
        message: 'हिंदी वॉइस प्रोसेसिंग में त्रुटि हुई',
        error: error.message
      };
    }
  }

  /**
   * Multi-language voice support
   */
  async handleMultiLanguageVoice(userId, audioData, preferredLanguage = 'en') {
    try {
      logger.info('Processing multi-language voice input', { userId, preferredLanguage });

      // Detect language automatically
      const detectedLanguage = await this.detectLanguage(audioData);
      const language = this.supportedLanguages.includes(detectedLanguage) ? detectedLanguage : preferredLanguage;

      // Process with detected language
      const result = await this.analyzeVoice(userId, audioData, language);

      // Add language information to response
      if (result.success) {
        result.data.detectedLanguage = detectedLanguage;
        result.data.processedLanguage = language;
      }

      return result;
    } catch (error) {
      logger.error('Multi-language voice processing failed', { error: error.message, userId });
      return {
        success: false,
        message: 'Multi-language voice processing failed',
        error: error.message
      };
    }
  }

  // Helper methods for voice processing
  async validateAudioData(audioData) {
    try {
      // Check audio format
      if (!this.voiceProcessingConfig.supportedFormats.includes(audioData.format)) {
        return {
          valid: false,
          error: `Unsupported audio format. Supported formats: ${this.voiceProcessingConfig.supportedFormats.join(', ')}`
        };
      }

      // Check audio duration
      if (audioData.duration > this.voiceProcessingConfig.maxDuration) {
        return {
          valid: false,
          error: `Audio too long. Maximum duration: ${this.voiceProcessingConfig.maxDuration} seconds`
        };
      }

      // Check audio quality
      if (audioData.quality < this.voiceProcessingConfig.qualityThreshold) {
        return {
          valid: false,
          error: 'Audio quality too low. Please speak clearly and reduce background noise.'
        };
      }

      return { valid: true };
    } catch (error) {
      logger.error('Audio validation failed', { error: error.message });
      return {
        valid: false,
        error: 'Audio validation failed'
      };
    }
  }

  async speechToText(audioData, language) {
    try {
      // In real implementation, integrate with speech-to-text service
      // For now, return mock transcription
      const mockTranscriptions = {
        en: {
          'buy fund': 'I want to buy mutual fund',
          'portfolio status': 'What is my portfolio status?',
          'sip amount': 'How much should I invest in SIP?',
          'tax saving': 'Show me tax saving options'
        },
        hi: {
          'fund khareedna': 'मैं म्यूचुअल फंड खरीदना चाहता हूं',
          'portfolio status': 'मेरे पोर्टफोलियो की स्थिति क्या है?',
          'sip amount': 'SIP में कितना निवेश करूं?',
          'tax saving': 'टैक्स बचत के विकल्प दिखाएं'
        }
      };

      // Simulate transcription based on audio content
      const transcription = mockTranscriptions[language]?.[audioData.content] || 'Hello, how can I help you?';

      return {
        success: true,
        text: transcription,
        confidence: 0.9
      };
    } catch (error) {
      logger.error('Speech to text failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async analyzeIntent(text, language) {
    try {
      const intent = {
        action: this.actionIntents.INQUIRY,
        confidence: 0.7,
        entities: {},
        sentiment: 'neutral'
      };

      const lowerText = text.toLowerCase();

      // Analyze buy intent
      if (this.containsBuyKeywords(lowerText, language)) {
        intent.action = this.actionIntents.BUY;
        intent.confidence = 0.9;
        intent.entities = this.extractFundEntities(text, language);
      }

      // Analyze sell intent
      else if (this.containsSellKeywords(lowerText, language)) {
        intent.action = this.actionIntents.SELL;
        intent.confidence = 0.9;
        intent.entities = this.extractFundEntities(text, language);
      }

      // Analyze portfolio intent
      else if (this.containsPortfolioKeywords(lowerText, language)) {
        intent.action = this.actionIntents.PORTFOLIO;
        intent.confidence = 0.8;
      }

      // Analyze SIP intent
      else if (this.containsSIPKeywords(lowerText, language)) {
        intent.action = this.actionIntents.SIP;
        intent.confidence = 0.8;
        intent.entities = this.extractAmountEntities(text, language);
      }

      // Analyze goal intent
      else if (this.containsGoalKeywords(lowerText, language)) {
        intent.action = this.actionIntents.GOAL;
        intent.confidence = 0.8;
      }

      // Analyze tax intent
      else if (this.containsTaxKeywords(lowerText, language)) {
        intent.action = this.actionIntents.TAX;
        intent.confidence = 0.8;
      }

      // Analyze market intent
      else if (this.containsMarketKeywords(lowerText, language)) {
        intent.action = this.actionIntents.MARKET;
        intent.confidence = 0.8;
      }

      // Analyze help intent
      else if (this.containsHelpKeywords(lowerText, language)) {
        intent.action = this.actionIntents.HELP;
        intent.confidence = 0.9;
      }

      // Analyze sentiment
      intent.sentiment = this.analyzeSentiment(text, language);

      return intent;
    } catch (error) {
      logger.error('Intent analysis failed', { error: error.message });
      return {
        action: this.actionIntents.INQUIRY,
        confidence: 0.5,
        entities: {},
        sentiment: 'neutral'
      };
    }
  }

  async generateResponse(userId, intent, language) {
    try {
      const response = await this.processVoiceCommand(userId, 'dummy', language);
      
      // Add voice-specific formatting
      if (response.success) {
        response.data.voiceOptimized = true;
        response.data.audioResponse = await this.generateAudioResponse(response.data.message, language);
      }

      return response;
    } catch (error) {
      logger.error('Response generation failed', { error: error.message });
      return {
        success: false,
        message: 'Failed to generate response',
        error: error.message
      };
    }
  }

  // Command processing methods
  async processBuyCommand(userId, intent, language) {
    try {
      const fundName = intent.entities.fundName;
      const amount = intent.entities.amount;

      if (!fundName) {
        return {
          success: false,
          message: this.getLocalizedMessage('Please specify which fund you want to buy', language)
        };
      }

      // Get fund recommendations
      const recommendations = await this.getFundRecommendations(userId, fundName);

      return {
        success: true,
        data: {
          action: 'BUY',
          fundName,
          amount,
          recommendations,
          message: this.getLocalizedMessage(`I found ${recommendations.length} funds matching "${fundName}". Would you like me to show you the details?`, language)
        }
      };
    } catch (error) {
      logger.error('Buy command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to process buy command', language)
      };
    }
  }

  async processSellCommand(userId, intent, language) {
    try {
      const fundName = intent.entities.fundName;
      const amount = intent.entities.amount;

      if (!fundName) {
        return {
          success: false,
          message: this.getLocalizedMessage('Please specify which fund you want to sell', language)
        };
      }

      // Get user's holdings
      const holdings = await Holding.find({ userId, isActive: true });
      const holding = holdings.find(h => h.fundName.toLowerCase().includes(fundName.toLowerCase()));

      if (!holding) {
        return {
          success: false,
          message: this.getLocalizedMessage(`You don't have any holdings in ${fundName}`, language)
        };
      }

      return {
        success: true,
        data: {
          action: 'SELL',
          fundName: holding.fundName,
          currentValue: holding.currentValue,
          units: holding.units,
          message: this.getLocalizedMessage(`You have ${holding.units} units of ${holding.fundName} worth ₹${holding.currentValue.toLocaleString()}. How much would you like to sell?`, language)
        }
      };
    } catch (error) {
      logger.error('Sell command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to process sell command', language)
      };
    }
  }

  async processPortfolioCommand(userId, intent, language) {
    try {
      const dashboardData = await dashboardEngine.getDashboardData(userId, false);
      
      if (!dashboardData.success) {
        return {
          success: false,
          message: this.getLocalizedMessage('Failed to get portfolio data', language)
        };
      }

      const portfolio = dashboardData.data.portfolio;
      const message = this.getLocalizedMessage(
        `Your portfolio is worth ₹${portfolio.totalValue.toLocaleString()}. You've invested ₹${portfolio.totalInvested.toLocaleString()} and gained ₹${portfolio.currentGain.toLocaleString()} (${portfolio.gainPercentage.toFixed(1)}%). Your XIRR is ${portfolio.xirr.toFixed(1)}%.`,
        language
      );

      return {
        success: true,
        data: {
          action: 'PORTFOLIO',
          portfolio,
          message
        }
      };
    } catch (error) {
      logger.error('Portfolio command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to get portfolio information', language)
      };
    }
  }

  async processRecommendationCommand(userId, intent, language) {
    try {
      const user = await User.findById(userId);
      const holdings = await Holding.find({ userId, isActive: true });
      const portfolio = await UserPortfolio.findOne({ userId });

      const optimizationResult = await portfolioOptimizer.optimizePortfolio(
        {
          userId: user._id,
          age: user.age,
          income: user.income,
          taxSlab: user.taxSlab,
          investmentHorizon: user.investmentHorizon
        },
        'MODERATE',
        ['WEALTH_CREATION', 'RETIREMENT']
      );

      if (!optimizationResult.success) {
        return {
          success: false,
          message: this.getLocalizedMessage('Failed to generate recommendations', language)
        };
      }

      const recommendations = optimizationResult.data.fundRecommendations;
      const message = this.getLocalizedMessage(
        `I have ${recommendations.length} fund recommendations for you. The top recommendation is ${recommendations[0]?.recommendedFunds[0]?.fundName || 'a diversified fund'}. Would you like to see all recommendations?`,
        language
      );

      return {
        success: true,
        data: {
          action: 'RECOMMENDATION',
          recommendations,
          message
        }
      };
    } catch (error) {
      logger.error('Recommendation command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to generate recommendations', language)
      };
    }
  }

  async processSIPCommand(userId, intent, language) {
    try {
      const amount = intent.entities.amount;
      const user = await User.findById(userId);

      if (!amount) {
        // Calculate recommended SIP amount
        const monthlyIncome = user.income / 12;
        const recommendedSIP = Math.min(monthlyIncome * 0.3, 50000);

        return {
          success: true,
          data: {
            action: 'SIP',
            recommendedAmount: recommendedSIP,
            message: this.getLocalizedMessage(
              `Based on your income, you may consider a monthly SIP of ₹${recommendedSIP.toLocaleString()}. This represents 30% of your monthly income. This is for informational purposes only. Would you like to explore SIP options?`,
              language
            )
          }
        };
      }

      return {
        success: true,
        data: {
          action: 'SIP',
          amount,
          message: this.getLocalizedMessage(
            `Great! I'll help you set up a monthly SIP of ₹${amount.toLocaleString()}. Which fund would you like to invest in?`,
            language
          )
        }
      };
    } catch (error) {
      logger.error('SIP command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to process SIP command', language)
      };
    }
  }

  async processGoalCommand(userId, intent, language) {
    try {
      const goals = await this.getUserGoals(userId);
      
      if (goals.length === 0) {
        return {
          success: true,
          data: {
            action: 'GOAL',
            message: this.getLocalizedMessage(
              'You haven\'t set any financial goals yet. Would you like me to help you create one? Common goals include retirement, children\'s education, or buying a house.',
              language
            )
          }
        };
      }

      const message = this.getLocalizedMessage(
        `You have ${goals.length} financial goals. Your main goal is ${goals[0].name} with a target of ₹${goals[0].targetAmount.toLocaleString()}. Would you like to see progress on all your goals?`,
        language
      );

      return {
        success: true,
        data: {
          action: 'GOAL',
          goals,
          message
        }
      };
    } catch (error) {
      logger.error('Goal command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to get goal information', language)
      };
    }
  }

  async processTaxCommand(userId, intent, language) {
    try {
      const user = await User.findById(userId);
      const holdings = await Holding.find({ userId, isActive: true });

      const taxReminders = await this.getTaxReminders(user, holdings);
      
      if (taxReminders.length === 0) {
        return {
          success: true,
          data: {
            action: 'TAX',
            message: this.getLocalizedMessage(
              'Great! Your tax planning looks good. You\'re maximizing your tax savings through ELSS investments and other tax-efficient strategies.',
              language
            )
          }
        };
      }

      const message = this.getLocalizedMessage(
        `I found ${taxReminders.length} tax optimization opportunities. The most important one is ${taxReminders[0].title}. Would you like to see all tax-saving options?`,
        language
      );

      return {
        success: true,
        data: {
          action: 'TAX',
          taxReminders,
          message
        }
      };
    } catch (error) {
      logger.error('Tax command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to get tax information', language)
      };
    }
  }

  async processMarketCommand(userId, intent, language) {
    try {
      const marketTrends = await predictiveEngine.predictMarketTrends('3M');
      
      if (!marketTrends.success) {
        return {
          success: false,
          message: this.getLocalizedMessage('Failed to get market information', language)
        };
      }

      const trend = marketTrends.data.trends;
      const sentiment = marketTrends.data.sentiment;
      
      const message = this.getLocalizedMessage(
        `Current market sentiment is ${sentiment.level.toLowerCase()}. Based on historical patterns, Nifty has shown ${trend.nifty > 0 ? 'positive' : 'negative'} trends. Past performance does not guarantee future results. Would you like detailed market information?`,
        language
      );

      return {
        success: true,
        data: {
          action: 'MARKET',
          marketData: marketTrends.data,
          message
        }
      };
    } catch (error) {
      logger.error('Market command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to get market information', language)
      };
    }
  }

  async processHelpCommand(userId, intent, language) {
    try {
      const helpMessage = this.getLocalizedMessage(
        'I can help you with: 1) Check portfolio status, 2) Buy or sell funds, 3) Set up SIP, 4) Get investment recommendations, 5) Check market trends, 6) Tax planning, 7) Goal tracking. What would you like to do?',
        language
      );

      return {
        success: true,
        data: {
          action: 'HELP',
          message: helpMessage
        }
      };
    } catch (error) {
      logger.error('Help command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to provide help', language)
      };
    }
  }

  async processInquiryCommand(userId, intent, language) {
    try {
      return {
        success: true,
        data: {
          action: 'INQUIRY',
          message: this.getLocalizedMessage(
            'I\'m here to help with your investments. You can ask me about your portfolio, buy/sell funds, set up SIP, get recommendations, check market trends, or plan your taxes. What would you like to know?',
            language
          )
        }
      };
    } catch (error) {
      logger.error('Inquiry command processing failed', { error: error.message });
      return {
        success: false,
        message: this.getLocalizedMessage('Failed to process inquiry', language)
      };
    }
  }

  // Hindi-specific methods
  async analyzeHindiIntent(text) {
    try {
      const intent = {
        action: this.actionIntents.INQUIRY,
        confidence: 0.7,
        entities: {},
        sentiment: 'neutral'
      };

      const lowerText = text.toLowerCase();

      // Hindi keywords for different actions
      if (this.containsHindiBuyKeywords(lowerText)) {
        intent.action = this.actionIntents.BUY;
        intent.confidence = 0.9;
      } else if (this.containsHindiSellKeywords(lowerText)) {
        intent.action = this.actionIntents.SELL;
        intent.confidence = 0.9;
      } else if (this.containsHindiPortfolioKeywords(lowerText)) {
        intent.action = this.actionIntents.PORTFOLIO;
        intent.confidence = 0.8;
      }

      return intent;
    } catch (error) {
      logger.error('Hindi intent analysis failed', { error: error.message });
      return {
        action: this.actionIntents.INQUIRY,
        confidence: 0.5,
        entities: {},
        sentiment: 'neutral'
      };
    }
  }

  async generateHindiResponse(userId, intent) {
    try {
      const response = await this.processVoiceCommand(userId, 'dummy', 'hi');
      
      // Add Hindi-specific formatting
      if (response.success) {
        response.data.voiceOptimized = true;
        response.data.audioResponse = await this.generateAudioResponse(response.data.message, 'hi');
      }

      return response;
    } catch (error) {
      logger.error('Hindi response generation failed', { error: error.message });
      return {
        success: false,
        message: 'हिंदी प्रतिक्रिया उत्पन्न करने में त्रुटि हुई',
        error: error.message
      };
    }
  }

  // Utility methods
  containsBuyKeywords(text, language) {
    const buyKeywords = {
      en: ['buy', 'purchase', 'invest', 'start', 'begin'],
      hi: ['खरीद', 'खरीदना', 'निवेश', 'शुरू', 'शुरू करना']
    };
    
    const keywords = buyKeywords[language] || buyKeywords.en;
    return keywords.some(keyword => text.includes(keyword));
  }

  containsSellKeywords(text, language) {
    const sellKeywords = {
      en: ['sell', 'exit', 'withdraw', 'redeem'],
      hi: ['बेच', 'बेचना', 'निकाल', 'निकालना']
    };
    
    const keywords = sellKeywords[language] || sellKeywords.en;
    return keywords.some(keyword => text.includes(keyword));
  }

  containsPortfolioKeywords(text, language) {
    const portfolioKeywords = {
      en: ['portfolio', 'status', 'value', 'worth', 'balance'],
      hi: ['पोर्टफोलियो', 'स्थिति', 'मूल्य', 'बैलेंस']
    };
    
    const keywords = portfolioKeywords[language] || portfolioKeywords.en;
    return keywords.some(keyword => text.includes(keyword));
  }

  containsSIPKeywords(text, language) {
    const sipKeywords = {
      en: ['sip', 'monthly', 'regular', 'investment'],
      hi: ['सिप', 'मासिक', 'नियमित', 'निवेश']
    };
    
    const keywords = sipKeywords[language] || sipKeywords.en;
    return keywords.some(keyword => text.includes(keyword));
  }

  containsGoalKeywords(text, language) {
    const goalKeywords = {
      en: ['goal', 'target', 'objective', 'aim'],
      hi: ['लक्ष्य', 'टारगेट', 'उद्देश्य']
    };
    
    const keywords = goalKeywords[language] || goalKeywords.en;
    return keywords.some(keyword => text.includes(keyword));
  }

  containsTaxKeywords(text, language) {
    const taxKeywords = {
      en: ['tax', 'taxation', 'saving', 'deduction'],
      hi: ['टैक्स', 'कर', 'बचत', 'छूट']
    };
    
    const keywords = taxKeywords[language] || taxKeywords.en;
    return keywords.some(keyword => text.includes(keyword));
  }

  containsMarketKeywords(text, language) {
    const marketKeywords = {
      en: ['market', 'trend', 'nifty', 'sensex', 'stock'],
      hi: ['बाजार', 'ट्रेंड', 'निफ्टी', 'सेंसेक्स']
    };
    
    const keywords = marketKeywords[language] || marketKeywords.en;
    return keywords.some(keyword => text.includes(keyword));
  }

  containsHelpKeywords(text, language) {
    const helpKeywords = {
      en: ['help', 'assist', 'support', 'guide'],
      hi: ['मदद', 'सहायता', 'सहयोग']
    };
    
    const keywords = helpKeywords[language] || helpKeywords.en;
    return keywords.some(keyword => text.includes(keyword));
  }

  // Hindi-specific keyword methods
  containsHindiBuyKeywords(text) {
    const keywords = ['खरीद', 'खरीदना', 'निवेश', 'शुरू', 'शुरू करना'];
    return keywords.some(keyword => text.includes(keyword));
  }

  containsHindiSellKeywords(text) {
    const keywords = ['बेच', 'बेचना', 'निकाल', 'निकालना'];
    return keywords.some(keyword => text.includes(keyword));
  }

  containsHindiPortfolioKeywords(text) {
    const keywords = ['पोर्टफोलियो', 'स्थिति', 'मूल्य', 'बैलेंस'];
    return keywords.some(keyword => text.includes(keyword));
  }

  extractFundEntities(text, language) {
    // Simple entity extraction
    const entities = {};
    
    // Extract fund names (simplified)
    const fundPatterns = {
      en: /(axis|hdfc|sbi|icici|kotak|mirae|nippon|tata|uti|franklin)/i,
      hi: /(एक्सिस|एचडीएफसी|एसबीआई|आईसीआईसीआई|कोटक|मिराए|निप्पॉन|टाटा|यूटीआई|फ्रैंकलिन)/i
    };
    
    const pattern = fundPatterns[language] || fundPatterns.en;
    const match = text.match(pattern);
    if (match) {
      entities.fundName = match[1];
    }

    // Extract amounts
    const amountPattern = /(\d+)\s*(thousand|lakh|lac|crore|k|l|c)/i;
    const amountMatch = text.match(amountPattern);
    if (amountMatch) {
      entities.amount = this.convertAmountToNumber(amountMatch[1], amountMatch[2]);
    }

    return entities;
  }

  extractAmountEntities(text, language) {
    const entities = {};
    
    // Extract amounts
    const amountPattern = /(\d+)\s*(thousand|lakh|lac|crore|k|l|c)/i;
    const amountMatch = text.match(amountPattern);
    if (amountMatch) {
      entities.amount = this.convertAmountToNumber(amountMatch[1], amountMatch[2]);
    }

    return entities;
  }

  convertAmountToNumber(value, unit) {
    const multipliers = {
      'thousand': 1000,
      'k': 1000,
      'lakh': 100000,
      'lac': 100000,
      'l': 100000,
      'crore': 10000000,
      'c': 10000000
    };
    
    return parseInt(value) * (multipliers[unit.toLowerCase()] || 1);
  }

  analyzeSentiment(text, language) {
    // Simple sentiment analysis
    const positiveWords = {
      en: ['good', 'great', 'excellent', 'profit', 'gain', 'up'],
      hi: ['अच्छा', 'बढ़िया', 'लाभ', 'फायदा', 'ऊपर']
    };
    
    const negativeWords = {
      en: ['bad', 'loss', 'down', 'worried', 'concerned'],
      hi: ['बुरा', 'नुकसान', 'नीचे', 'चिंतित', 'परेशान']
    };
    
    const positive = positiveWords[language] || positiveWords.en;
    const negative = negativeWords[language] || negativeWords.en;
    
    const lowerText = text.toLowerCase();
    const positiveCount = positive.filter(word => lowerText.includes(word)).length;
    const negativeCount = negative.filter(word => lowerText.includes(word)).length;
    
    if (positiveCount > negativeCount) return 'positive';
    if (negativeCount > positiveCount) return 'negative';
    return 'neutral';
  }

  async detectLanguage(audioData) {
    // In real implementation, use language detection service
    // For now, return English as default
    return 'en';
  }

  async generateAudioResponse(text, language) {
    // In real implementation, use text-to-speech service
    // For now, return mock audio data
    return {
      url: `https://api.example.com/tts?text=${encodeURIComponent(text)}&lang=${language}`,
      duration: text.length * 0.1, // Rough estimate
      format: 'mp3'
    };
  }

  async logVoiceInteraction(userId, interactionData) {
    try {
      // Log voice interaction for analytics
      logger.info('Voice interaction logged', { userId, interactionData });
    } catch (error) {
      logger.error('Failed to log voice interaction', { error: error.message });
    }
  }

  async getFundRecommendations(userId, fundName) {
    // Mock fund recommendations
    return [
      {
        fundName: 'Axis Bluechip Fund',
        category: 'Large Cap',
        returns1Y: 0.15,
        risk: 'MODERATE'
      },
      {
        fundName: 'HDFC Mid-Cap Opportunities Fund',
        category: 'Mid Cap',
        returns1Y: 0.18,
        risk: 'HIGH'
      }
    ];
  }

  async getUserGoals(userId) {
    // Mock user goals
    return [
      {
        name: 'Retirement',
        targetAmount: 10000000,
        currentAmount: 2000000,
        targetDate: '2035-01-01'
      }
    ];
  }

  async getTaxReminders(user, holdings) {
    // Mock tax reminders
    return [
      {
        type: 'ELSS_INVESTMENT',
        title: 'ELSS Investment Reminder',
        message: 'You can invest ₹100,000 more in ELSS for tax deduction'
      }
    ];
  }

  getLocalizedMessage(message, language) {
    const translations = {
      hi: {
        'Please specify which fund you want to buy': 'कृपया बताएं कि आप कौन सा फंड खरीदना चाहते हैं',
        'Failed to process buy command': 'खरीदने का आदेश प्रोसेस करने में विफल',
        'Your portfolio is worth': 'आपका पोर्टफोलियो का मूल्य है',
        'Failed to get portfolio data': 'पोर्टफोलियो डेटा प्राप्त करने में विफल',
        'I can help you with': 'मैं आपकी मदद कर सकता हूं',
        'I\'m here to help with your investments': 'मैं आपके निवेश में मदद करने के लिए यहां हूं'
      }
    };
    
    return translations[language]?.[message] || message;
  }
}

module.exports = new VoiceBot(); 