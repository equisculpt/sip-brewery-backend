const logger = require('../utils/logger');
const { User, UserPreferences, LanguageContent, RegionalSettings } = require('../models');

class RegionalLanguageService {
  constructor() {
    this.supportedLanguages = {
      HINDI: {
        code: 'hi',
        name: 'Hindi',
        nativeName: 'हिंदी',
        script: 'Devanagari',
        regions: ['North India', 'Central India'],
        voiceSupport: true,
        textSupport: true
      },
      TAMIL: {
        code: 'ta',
        name: 'Tamil',
        nativeName: 'தமிழ்',
        script: 'Tamil',
        regions: ['Tamil Nadu', 'Puducherry'],
        voiceSupport: true,
        textSupport: true
      },
      TELUGU: {
        code: 'te',
        name: 'Telugu',
        nativeName: 'తెలుగు',
        script: 'Telugu',
        regions: ['Andhra Pradesh', 'Telangana'],
        voiceSupport: true,
        textSupport: true
      },
      BENGALI: {
        code: 'bn',
        name: 'Bengali',
        nativeName: 'বাংলা',
        script: 'Bengali',
        regions: ['West Bengal', 'Tripura'],
        voiceSupport: true,
        textSupport: true
      },
      MARATHI: {
        code: 'mr',
        name: 'Marathi',
        nativeName: 'मराठी',
        script: 'Devanagari',
        regions: ['Maharashtra', 'Goa'],
        voiceSupport: true,
        textSupport: true
      },
      GUJARATI: {
        code: 'gu',
        name: 'Gujarati',
        nativeName: 'ગુજરાતી',
        script: 'Gujarati',
        regions: ['Gujarat', 'Dadra and Nagar Haveli'],
        voiceSupport: true,
        textSupport: true
      },
      KANNADA: {
        code: 'kn',
        name: 'Kannada',
        nativeName: 'ಕನ್ನಡ',
        script: 'Kannada',
        regions: ['Karnataka'],
        voiceSupport: true,
        textSupport: true
      },
      MALAYALAM: {
        code: 'ml',
        name: 'Malayalam',
        nativeName: 'മലയാളം',
        script: 'Malayalam',
        regions: ['Kerala', 'Lakshadweep'],
        voiceSupport: true,
        textSupport: true
      },
      PUNJABI: {
        code: 'pa',
        name: 'Punjabi',
        nativeName: 'ਪੰਜਾਬੀ',
        script: 'Gurmukhi',
        regions: ['Punjab', 'Haryana'],
        voiceSupport: true,
        textSupport: true
      },
      ODIA: {
        code: 'or',
        name: 'Odia',
        nativeName: 'ଓଡ଼ିଆ',
        script: 'Odia',
        regions: ['Odisha'],
        voiceSupport: true,
        textSupport: true
      }
    };

    this.investmentTerms = {
      HINDI: {
        portfolio: 'पोर्टफोलियो',
        mutualFund: 'म्यूचुअल फंड',
        sip: 'एसआईपी',
        returns: 'रिटर्न',
        investment: 'निवेश',
        risk: 'जोखिम',
        goal: 'लक्ष्य',
        market: 'बाजार',
        fund: 'फंड',
        switch: 'स्विच',
        redeem: 'रिडीम',
        nav: 'एनएवी',
        expenseRatio: 'खर्च अनुपात',
        exitLoad: 'एग्जिट लोड',
        lockIn: 'लॉक-इन',
        taxSaving: 'कर बचत',
        equity: 'इक्विटी',
        debt: 'डेट',
        hybrid: 'हाइब्रिड',
        liquid: 'लिक्विड',
        largeCap: 'लार्ज कैप',
        midCap: 'मिड कैप',
        smallCap: 'स्मॉल कैप'
      },
      TAMIL: {
        portfolio: 'போர்ட்ஃபோலியோ',
        mutualFund: 'பரஸ்பர நிதி',
        sip: 'எஸ்ஐபி',
        returns: 'திருப்பம்',
        investment: 'முதலீடு',
        risk: 'ஆபத்து',
        goal: 'இலக்கு',
        market: 'சந்தை',
        fund: 'நிதி',
        switch: 'மாற்றம்',
        redeem: 'மீட்பு',
        nav: 'என்ஏவி',
        expenseRatio: 'செலவு விகிதம்',
        exitLoad: 'வெளியேற்ற சுமை',
        lockIn: 'பூட்டு',
        taxSaving: 'வரி சேமிப்பு',
        equity: 'பங்கு',
        debt: 'கடன்',
        hybrid: 'கலப்பு',
        liquid: 'திரவ',
        largeCap: 'பெரிய மூலதனம்',
        midCap: 'நடுத்தர மூலதனம்',
        smallCap: 'சிறிய மூலதனம்'
      },
      TELUGU: {
        portfolio: 'పోర్ట్‌ఫోలియో',
        mutualFund: 'మ్యూచువల్ ఫండ్',
        sip: 'ఎస్ఐపి',
        returns: 'రిటర్న్స్',
        investment: 'పెట్టుబడి',
        risk: 'ప్రమాదం',
        goal: 'లక్ష్యం',
        market: 'మార్కెట్',
        fund: 'ఫండ్',
        switch: 'స్విచ్',
        redeem: 'రీడీమ్',
        nav: 'ఎన్ఎవి',
        expenseRatio: 'ఖర్చు నిష్పత్తి',
        exitLoad: 'ఎగ్జిట్ లోడ్',
        lockIn: 'లాక్-ఇన్',
        taxSaving: 'పన్ను పొదుపు',
        equity: 'ఈక్విటీ',
        debt: 'డెట్',
        hybrid: 'హైబ్రిడ్',
        liquid: 'లిక్విడ్',
        largeCap: 'లార్జ్ క్యాప్',
        midCap: 'మిడ్ క్యాప్',
        smallCap: 'స్మాల్ క్యాప్'
      }
    };

    this.culturalContext = {
      HINDI: {
        greetings: ['नमस्ते', 'नमस्कार', 'प्रणाम'],
        formalAddress: 'आप',
        informalAddress: 'तुम',
        respectTerms: ['जी', 'साहब', 'मैडम'],
        investmentStyle: 'conservative',
        preferredFunds: ['debt', 'hybrid', 'tax_saving'],
        riskTolerance: 'low_to_medium'
      },
      TAMIL: {
        greetings: ['வணக்கம்', 'நமஸ்காரம்'],
        formalAddress: 'நீங்கள்',
        informalAddress: 'நீ',
        respectTerms: ['சார்', 'மேடம்'],
        investmentStyle: 'balanced',
        preferredFunds: ['equity', 'hybrid', 'debt'],
        riskTolerance: 'medium'
      },
      TELUGU: {
        greetings: ['నమస్కారం', 'వందనములు'],
        formalAddress: 'మీరు',
        informalAddress: 'నువ్వు',
        respectTerms: ['గారు', 'మేడం'],
        investmentStyle: 'growth_oriented',
        preferredFunds: ['equity', 'mid_cap', 'small_cap'],
        riskTolerance: 'medium_to_high'
      }
    };
  }

  /**
   * Get supported languages
   */
  async getSupportedLanguages() {
    try {
      logger.info('Getting supported languages');

      const languages = Object.entries(this.supportedLanguages).map(([key, lang]) => ({
        code: lang.code,
        name: lang.name,
        nativeName: lang.nativeName,
        script: lang.script,
        regions: lang.regions,
        voiceSupport: lang.voiceSupport,
        textSupport: lang.textSupport
      }));

      return {
        success: true,
        data: {
          languages,
          totalLanguages: languages.length,
          voiceSupported: languages.filter(l => l.voiceSupport).length,
          textSupported: languages.filter(l => l.textSupport).length
        }
      };
    } catch (error) {
      logger.error('Failed to get supported languages', { error: error.message });
      return {
        success: false,
        message: 'Failed to get supported languages',
        error: error.message
      };
    }
  }

  /**
   * Set user language preference
   */
  async setUserLanguagePreference(userId, languageCode, preferences = {}) {
    try {
      logger.info('Setting user language preference', { userId, languageCode });

      // Validate language code
      const language = Object.values(this.supportedLanguages).find(l => l.code === languageCode);
      if (!language) {
        throw new Error(`Unsupported language: ${languageCode}`);
      }

      // Update or create user preferences
      const userPrefs = await UserPreferences.findOneAndUpdate(
        { userId },
        {
          languageCode,
          languageName: language.name,
          nativeName: language.nativeName,
          script: language.script,
          voiceEnabled: preferences.voiceEnabled !== undefined ? preferences.voiceEnabled : true,
          textEnabled: preferences.textEnabled !== undefined ? preferences.textEnabled : true,
          culturalContext: this.culturalContext[languageCode.toUpperCase()] || {},
          lastUpdated: new Date()
        },
        { upsert: true, new: true }
      );

      // Update user profile
      await User.findByIdAndUpdate(userId, {
        preferredLanguage: languageCode,
        languageUpdatedAt: new Date()
      });

      return {
        success: true,
        data: {
          userPreferences: userPrefs,
          language: language,
          message: `Language preference set to ${language.name}`
        }
      };
    } catch (error) {
      logger.error('Failed to set user language preference', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to set language preference',
        error: error.message
      };
    }
  }

  /**
   * Get user language preferences
   */
  async getUserLanguagePreferences(userId) {
    try {
      logger.info('Getting user language preferences', { userId });

      const userPrefs = await UserPreferences.findOne({ userId });
      const user = await User.findById(userId);

      if (!userPrefs) {
        return {
          success: true,
          data: {
            languageCode: 'en',
            languageName: 'English',
            nativeName: 'English',
            script: 'Latin',
            voiceEnabled: true,
            textEnabled: true,
            culturalContext: {},
            isDefault: true
          }
        };
      }

      return {
        success: true,
        data: {
          ...userPrefs.toObject(),
          isDefault: false
        }
      };
    } catch (error) {
      logger.error('Failed to get user language preferences', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get language preferences',
        error: error.message
      };
    }
  }

  /**
   * Translate investment terms
   */
  async translateInvestmentTerms(terms, targetLanguage) {
    try {
      logger.info('Translating investment terms', { targetLanguage, termsCount: terms.length });

      const languageKey = targetLanguage.toUpperCase();
      const translations = this.investmentTerms[languageKey];

      if (!translations) {
        throw new Error(`Translation not available for language: ${targetLanguage}`);
      }

      const translatedTerms = {};
      const untranslatedTerms = [];

      terms.forEach(term => {
        if (translations[term]) {
          translatedTerms[term] = translations[term];
        } else {
          untranslatedTerms.push(term);
          translatedTerms[term] = term; // Keep original if no translation
        }
      });

      return {
        success: true,
        data: {
          translatedTerms,
          untranslatedTerms,
          targetLanguage,
          translationCoverage: ((terms.length - untranslatedTerms.length) / terms.length) * 100
        }
      };
    } catch (error) {
      logger.error('Failed to translate investment terms', { error: error.message });
      return {
        success: false,
        message: 'Failed to translate investment terms',
        error: error.message
      };
    }
  }

  /**
   * Get cultural context for language
   */
  async getCulturalContext(languageCode) {
    try {
      logger.info('Getting cultural context', { languageCode });

      const languageKey = languageCode.toUpperCase();
      const context = this.culturalContext[languageKey];

      if (!context) {
        return {
          success: true,
          data: {
            languageCode,
            context: {},
            message: 'No specific cultural context available'
          }
        };
      }

      return {
        success: true,
        data: {
          languageCode,
          context,
          greetings: context.greetings,
          formalAddress: context.formalAddress,
          informalAddress: context.informalAddress,
          respectTerms: context.respectTerms,
          investmentStyle: context.investmentStyle,
          preferredFunds: context.preferredFunds,
          riskTolerance: context.riskTolerance
        }
      };
    } catch (error) {
      logger.error('Failed to get cultural context', { error: error.message });
      return {
        success: false,
        message: 'Failed to get cultural context',
        error: error.message
      };
    }
  }

  /**
   * Process voice commands in regional language
   */
  async processVoiceCommand(audioData, languageCode, userId) {
    try {
      logger.info('Processing voice command', { languageCode, userId });

      // Mock voice processing - in real implementation, use speech-to-text API
      const mockTranscription = this.getMockTranscription(languageCode);
      
      const command = await this.parseVoiceCommand(mockTranscription, languageCode);
      
      // Get user preferences for response language
      const userPrefs = await this.getUserLanguagePreferences(userId);
      const responseLanguage = userPrefs.success ? userPrefs.data.languageCode : 'en';

      const response = await this.generateVoiceResponse(command, responseLanguage);

      return {
        success: true,
        data: {
          originalCommand: mockTranscription,
          parsedCommand: command,
          response: response,
          languageCode: languageCode,
          responseLanguage: responseLanguage
        }
      };
    } catch (error) {
      logger.error('Failed to process voice command', { error: error.message });
      return {
        success: false,
        message: 'Failed to process voice command',
        error: error.message
      };
    }
  }

  /**
   * Generate localized content
   */
  async generateLocalizedContent(contentType, languageCode, context = {}) {
    try {
      logger.info('Generating localized content', { contentType, languageCode });

      const languageKey = languageCode.toUpperCase();
      const culturalContext = this.culturalContext[languageKey] || {};

      let localizedContent;

      switch (contentType) {
        case 'greeting':
          localizedContent = this.generateGreeting(languageCode, context);
          break;
        case 'investment_advice':
          localizedContent = this.generateInvestmentAdvice(languageCode, context);
          break;
        case 'market_update':
          localizedContent = this.generateMarketUpdate(languageCode, context);
          break;
        case 'portfolio_summary':
          localizedContent = this.generatePortfolioSummary(languageCode, context);
          break;
        case 'educational_content':
          localizedContent = this.generateEducationalContent(languageCode, context);
          break;
        default:
          throw new Error(`Unknown content type: ${contentType}`);
      }

      return {
        success: true,
        data: {
          contentType,
          languageCode,
          content: localizedContent,
          culturalContext: culturalContext,
          timestamp: new Date().toISOString()
        }
      };
    } catch (error) {
      logger.error('Failed to generate localized content', { error: error.message });
      return {
        success: false,
        message: 'Failed to generate localized content',
        error: error.message
      };
    }
  }

  /**
   * Get regional investment preferences
   */
  async getRegionalInvestmentPreferences(languageCode) {
    try {
      logger.info('Getting regional investment preferences', { languageCode });

      const languageKey = languageCode.toUpperCase();
      const context = this.culturalContext[languageKey];

      if (!context) {
        return {
          success: true,
          data: {
            languageCode,
            preferences: {
              investmentStyle: 'balanced',
              preferredFunds: ['equity', 'debt', 'hybrid'],
              riskTolerance: 'medium',
              investmentHorizon: 'long_term',
              preferredSectors: ['technology', 'finance', 'healthcare']
            }
          }
        };
      }

      const preferences = {
        investmentStyle: context.investmentStyle,
        preferredFunds: context.preferredFunds,
        riskTolerance: context.riskTolerance,
        investmentHorizon: 'long_term',
        preferredSectors: this.getPreferredSectors(languageCode)
      };

      return {
        success: true,
        data: {
          languageCode,
          preferences,
          culturalContext: context
        }
      };
    } catch (error) {
      logger.error('Failed to get regional investment preferences', { error: error.message });
      return {
        success: false,
        message: 'Failed to get regional investment preferences',
        error: error.message
      };
    }
  }

  /**
   * Create language-specific voice commands
   */
  async createVoiceCommands(languageCode) {
    try {
      logger.info('Creating voice commands', { languageCode });

      const commands = {
        HINDI: {
          portfolio_check: 'मेरा पोर्टफोलियो दिखाओ',
          fund_search: 'फंड खोजें',
          sip_start: 'एसआईपी शुरू करें',
          market_update: 'बाजार अपडेट',
          investment_advice: 'निवेश सलाह',
          switch_fund: 'फंड स्विच करें',
          redeem_fund: 'फंड रिडीम करें',
          tax_saving: 'कर बचत फंड',
          goal_setting: 'लक्ष्य सेट करें',
          risk_assessment: 'जोखिम मूल्यांकन'
        },
        TAMIL: {
          portfolio_check: 'எனது போர்ட்ஃபோலியோவைக் காட்டு',
          fund_search: 'நிதியைத் தேடு',
          sip_start: 'எஸ்ஐபியைத் தொடங்கு',
          market_update: 'சந்தை புதுப்பிப்பு',
          investment_advice: 'முதலீட்டு ஆலோசனை',
          switch_fund: 'நிதியை மாற்று',
          redeem_fund: 'நிதியை மீட்டெடு',
          tax_saving: 'வரி சேமிப்பு நிதி',
          goal_setting: 'இலக்கை அமை',
          risk_assessment: 'ஆபத்து மதிப்பீடு'
        },
        TELUGU: {
          portfolio_check: 'నా పోర్ట్‌ఫోలియో చూపించు',
          fund_search: 'ఫండ్ శోధించు',
          sip_start: 'ఎస్ఐపి ప్రారంభించు',
          market_update: 'మార్కెట్ అప్‌డేట్',
          investment_advice: 'పెట్టుబడి సలహా',
          switch_fund: 'ఫండ్ మార్చు',
          redeem_fund: 'ఫండ్ రీడీమ్ చేయు',
          tax_saving: 'పన్ను పొదుపు ఫండ్',
          goal_setting: 'లక్ష్యం సెట్ చేయు',
          risk_assessment: 'ప్రమాదం అంచనా'
        }
      };

      const languageKey = languageCode.toUpperCase();
      const languageCommands = commands[languageKey] || commands.HINDI;

      return {
        success: true,
        data: {
          languageCode,
          commands: languageCommands,
          totalCommands: Object.keys(languageCommands).length
        }
      };
    } catch (error) {
      logger.error('Failed to create voice commands', { error: error.message });
      return {
        success: false,
        message: 'Failed to create voice commands',
        error: error.message
      };
    }
  }

  // Helper methods
  getMockTranscription(languageCode) {
    const transcriptions = {
      HINDI: 'मेरा पोर्टफोलियो कैसा है',
      TAMIL: 'எனது போர்ட்ஃபோலியோ எப்படி உள்ளது',
      TELUGU: 'నా పోర్ట్‌ఫోలియో ఎలా ఉంది'
    };

    const languageKey = languageCode.toUpperCase();
    return transcriptions[languageKey] || transcriptions.HINDI;
  }

  async parseVoiceCommand(transcription, languageCode) {
    try {
      // Mock command parsing - in real implementation, use NLP
      const commands = {
        HINDI: {
          'मेरा पोर्टफोलियो': 'portfolio_check',
          'फंड खोजें': 'fund_search',
          'एसआईपी': 'sip_start',
          'बाजार': 'market_update'
        },
        TAMIL: {
          'போர்ட்ஃபோலியோ': 'portfolio_check',
          'நிதி': 'fund_search',
          'எஸ்ஐபி': 'sip_start',
          'சந்தை': 'market_update'
        },
        TELUGU: {
          'పోర్ట్‌ఫోలియో': 'portfolio_check',
          'ఫండ్': 'fund_search',
          'ఎస్ఐపి': 'sip_start',
          'మార్కెట్': 'market_update'
        }
      };

      const languageKey = languageCode.toUpperCase();
      const languageCommands = commands[languageKey] || commands.HINDI;

      for (const [phrase, command] of Object.entries(languageCommands)) {
        if (transcription.includes(phrase)) {
          return {
            type: command,
            confidence: 0.9,
            originalText: transcription
          };
        }
      }

      return {
        type: 'unknown',
        confidence: 0.1,
        originalText: transcription
      };
    } catch (error) {
      logger.error('Failed to parse voice command', { error: error.message });
      return {
        type: 'error',
        confidence: 0,
        originalText: transcription
      };
    }
  }

  async generateVoiceResponse(command, languageCode) {
    try {
      const responses = {
        HINDI: {
          portfolio_check: 'आपका पोर्टफोलियो अच्छा प्रदर्शन कर रहा है',
          fund_search: 'कौन सा फंड खोज रहे हैं आप?',
          sip_start: 'एसआईपी शुरू करने के लिए कौन सा फंड चुनेंगे?',
          market_update: 'आज का बाजार स्थिर है'
        },
        TAMIL: {
          portfolio_check: 'உங்கள் போர்ட்ஃபோலியோ நன்றாக செயல்படுகிறது',
          fund_search: 'எந்த நிதியைத் தேடுகிறீர்கள்?',
          sip_start: 'எஸ்ஐபி தொடங்க எந்த நிதியைத் தேர்வு செய்வீர்கள்?',
          market_update: 'இன்றைய சந்தை நிலையானது'
        },
        TELUGU: {
          portfolio_check: 'మీ పోర్ట్‌ఫోలియో బాగా పని చేస్తోంది',
          fund_search: 'ఏ ఫండ్ శోధిస్తున్నారు?',
          sip_start: 'ఎస్ఐపి ప్రారంభించడానికి ఏ ఫండ్ ఎంచుకుంటారు?',
          market_update: 'నేటి మార్కెట్ స్థిరంగా ఉంది'
        }
      };

      const languageKey = languageCode.toUpperCase();
      const languageResponses = responses[languageKey] || responses.HINDI;

      return languageResponses[command.type] || 'माफ़ करें, मैं समझ नहीं पाया';
    } catch (error) {
      logger.error('Failed to generate voice response', { error: error.message });
      return 'Sorry, I could not understand';
    }
  }

  generateGreeting(languageCode, context) {
    const greetings = {
      HINDI: {
        morning: 'सुप्रभात',
        afternoon: 'नमस्कार',
        evening: 'शुभ संध्या',
        night: 'शुभ रात्रि'
      },
      TAMIL: {
        morning: 'காலை வணக்கம்',
        afternoon: 'மதிய வணக்கம்',
        evening: 'மாலை வணக்கம்',
        night: 'இரவு வணக்கம்'
      },
      TELUGU: {
        morning: 'శుభోదయం',
        afternoon: 'నమస్కారం',
        evening: 'శుభ సాయంత్రం',
        night: 'శుభ రాత్రి'
      }
    };

    const languageKey = languageCode.toUpperCase();
    const languageGreetings = greetings[languageKey] || greetings.HINDI;

    const hour = new Date().getHours();
    let timeOfDay;

    if (hour < 12) timeOfDay = 'morning';
    else if (hour < 17) timeOfDay = 'afternoon';
    else if (hour < 21) timeOfDay = 'evening';
    else timeOfDay = 'night';

    return languageGreetings[timeOfDay];
  }

  generateInvestmentAdvice(languageCode, context) {
    const advice = {
      HINDI: 'आपके लिए इक्विटी फंड में निवेश करना अच्छा रहेगा',
      TAMIL: 'உங்களுக்கு ஈக்விட்டி நிதியில் முதலீடு செய்வது நல்லது',
      TELUGU: 'మీకు ఈక్విటీ ఫండ్‌లో పెట్టుబడి పెట్టడం మంచిది'
    };

    const languageKey = languageCode.toUpperCase();
    return advice[languageKey] || advice.HINDI;
  }

  generateMarketUpdate(languageCode, context) {
    const updates = {
      HINDI: 'आज का बाजार सकारात्मक रहा है',
      TAMIL: 'இன்றைய சந்தை நேர்மறையாக இருந்தது',
      TELUGU: 'నేటి మార్కెట్ సానుకూలంగా ఉంది'
    };

    const languageKey = languageCode.toUpperCase();
    return updates[languageKey] || updates.HINDI;
  }

  generatePortfolioSummary(languageCode, context) {
    const summaries = {
      HINDI: 'आपके पोर्टफोलियो का मूल्य ₹1,00,000 है',
      TAMIL: 'உங்கள் போர்ட்ஃபோலியோ மதிப்பு ₹1,00,000',
      TELUGU: 'మీ పోర్ట్‌ఫోలియో విలువ ₹1,00,000'
    };

    const languageKey = languageCode.toUpperCase();
    return summaries[languageKey] || summaries.HINDI;
  }

  generateEducationalContent(languageCode, context) {
    const content = {
      HINDI: 'म्यूचुअल फंड में निवेश करने के लाभ',
      TAMIL: 'பரஸ்பர நிதியில் முதலீடு செய்வதன் நன்மைகள்',
      TELUGU: 'మ్యూచువల్ ఫండ్‌లో పెట్టుబడి పెట్టడం వల్ల కలిగే ప్రయోజనాలు'
    };

    const languageKey = languageCode.toUpperCase();
    return content[languageKey] || content.HINDI;
  }

  getPreferredSectors(languageCode) {
    const sectors = {
      HINDI: ['technology', 'finance', 'healthcare', 'consumer_goods'],
      TAMIL: ['technology', 'healthcare', 'finance', 'automobile'],
      TELUGU: ['technology', 'pharmaceuticals', 'finance', 'real_estate']
    };

    const languageKey = languageCode.toUpperCase();
    return sectors[languageKey] || sectors.HINDI;
  }
}

module.exports = new RegionalLanguageService(); 