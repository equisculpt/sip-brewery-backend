const logger = require('./logger');

class MessageParser {
  constructor() {
    // Enhanced intent patterns for better coverage
    this.intentPatterns = {
      // Greetings
      GREETING: [
        /^(hi|hello|hey|good morning|good afternoon|good evening)$/i,
        /^(namaste|namaskar)$/i,
        /^(start|begin|new)$/i
      ],
      
      // Enhanced Onboarding with better patterns
      ONBOARDING: [
        /^(my name is|i am|i'm)\s+(.+)$/i,
        /^(name|full name)\s*(is)?\s*(.+)$/i,
        /^(email|e-mail)\s*(is)?\s*(.+)$/i,
        /^(pan|pan number|pan card)\s*(is)?\s*(.+)$/i,
        /^(aadhaar|aadhar|uid)\s*(is)?\s*(.+)$/i,
        /^(bank|account)\s*(details?|number)\s*(is)?\s*(.+)$/i,
        /^(investment|goal|target)\s*(is)?\s*(.+)$/i,
        /^(risk|appetite)\s*(is)?\s*(.+)$/i,
        /^(amount|investment amount)\s*(is)?\s*(.+)$/i,
        /^(.+@.+\..+)$/i, // Email pattern
        /^([A-Z]{5}[0-9]{4}[A-Z])$/i, // PAN pattern
        /^(\d{12})$/i, // Aadhaar pattern
        /^(\d{10,16})$/i, // Bank account pattern
        /^(conservative|moderate|aggressive)$/i, // Risk appetite
        /^₹?(\d+(?:,\d+)*(?:\.\d+)?)$/i // Amount pattern
      ],
      
      // Portfolio
      PORTFOLIO_VIEW: [
        /^(show|view|my|check)\s+(portfolio|holdings|investments)$/i,
        /^(portfolio|holdings|investments)$/i,
        /^(how much|what is)\s+(my\s+)?(portfolio|investment)(\s+worth)?$/i,
        /^(value|worth|balance)\s+(of\s+)?(my\s+)?(portfolio|investment)$/i,
        /^(returns|performance)\s+(of\s+)?(my\s+)?(portfolio|investment)$/i
      ],
      
      // Enhanced SIP Creation
      SIP_CREATE: [
        /^(i want to|start|begin|create)\s+(sip|investment)\s+(?:of\s+)?₹?(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:in|for)\s+(.+)$/i,
        /^(invest|sip)\s+₹?(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:in|for)\s+(.+)$/i,
        /^₹?(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:in|for)\s+(.+)$/i,
        /^(sip|investment)\s+(.+)$/i,
        /^(fund|scheme)\s+(.+)$/i,
        /^(amount|monthly)\s+₹?(\d+(?:,\d+)*(?:\.\d+)?)$/i,
        /^(frequency|how often)\s+(monthly|weekly|daily)$/i
      ],
      
      // Enhanced SIP Management
      SIP_STOP: [
        /^(stop|cancel|pause)\s+(?:my\s+)?(sip|investment)\s+(?:in\s+)?(.+)$/i,
        /^(stop|cancel|pause)\s+(.+)\s+(?:sip|investment)$/i,
        /^(modify|change|update)\s+(?:my\s+)?(sip|investment)$/i,
        /^(resume|restart)\s+(?:my\s+)?(sip|investment)$/i
      ],
      
      // SIP Status
      SIP_STATUS: [
        /^(check|show|my)\s+(?:sip\s+)?(status|orders)$/i,
        /^(sip\s+)?(status|orders)$/i,
        /^(active|running)\s+(?:sip|investment)$/i
      ],
      
      // Lump Sum
      LUMP_SUM: [
        /^(lump sum|lumpsum|one time|one-time)\s+₹?(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:in|for)\s+(.+)$/i,
        /^(invest|buy)\s+₹?(\d+(?:,\d+)*(?:\.\d+)?)\s+(?:in|for)\s+(.+)$/i
      ],
      
      // Enhanced AI Analysis
      AI_ANALYSIS: [
        /^(analyse|analyze|analysis|review)\s+(.+)$/i,
        /^(tell me about|what about|info on)\s+(.+)$/i,
        /^(how is|how's)\s+(.+)\s+(?:performing|doing)$/i,
        /^(recommend|suggestion|advice)\s+(.+)$/i,
        /^(market|insights|trends)$/i,
        /^(risk|assessment)\s+(.+)$/i
      ],
      
      // Statement
      STATEMENT: [
        /^(send|get|download|generate)\s+(?:my\s+)?(statement|report|summary)$/i,
        /^(statement|report|summary)$/i,
        /^(transaction|history)$/i
      ],
      
      // Enhanced Rewards
      REWARDS: [
        /^(my|show|check)\s+(rewards|points|cashback|bonus)$/i,
        /^(rewards|points|cashback|bonus)$/i,
        /^(redeem|withdraw)\s+(rewards|points|cashback)$/i
      ],
      
      // Enhanced Referral
      REFERRAL: [
        /^(refer|referral|invite)\s+(?:a\s+)?(friend|someone)$/i,
        /^(get|share)\s+(?:my\s+)?(referral\s+)?(link|code)$/i,
        /^(referral|code|link)$/i
      ],
      
      // Leaderboard
      LEADERBOARD: [
        /^(leaderboard|top|best|leaders)$/i,
        /^(show|view)\s+(?:the\s+)?(leaderboard|top\s+performers)$/i,
        /^(rankings|scores)$/i
      ],
      
      // Copy Portfolio
      COPY_PORTFOLIO: [
        /^(copy|follow|mimic)\s+(?:portfolio\s+of\s+)?(leader|user|investor)\s*(\d+)?$/i,
        /^(copy|follow|mimic)\s+(leader|user|investor)\s*(\d+)?$/i
      ],
      
      // Enhanced Help
      HELP: [
        /^(help|support|what can you do|commands|menu)$/i,
        /^(how|what)\s+(?:can\s+)?(?:you\s+)?(?:help|do)$/i,
        /^(features|services|options)$/i
      ],
      
      // Enhanced Confirmation
      CONFIRMATION: [
        /^(yes|y|confirm|ok|okay|sure|proceed)$/i,
        /^(no|n|cancel|stop|don't|dont)$/i,
        /^(correct|right|perfect|good)$/i
      ],

      // New intents for better coverage
      FUND_RESEARCH: [
        /^(research|study|compare)\s+(.+)$/i,
        /^(fund|scheme)\s+(comparison|vs|versus)\s+(.+)$/i
      ],

      MARKET_UPDATE: [
        /^(market|news|update|trends)$/i,
        /^(nifty|sensex|indices)$/i,
        /^(today|current)\s+(market|performance)$/i
      ],

      KYC_UPDATE: [
        /^(update|change|modify)\s+(kyc|details|information)$/i,
        /^(kyc|verification)\s+(status|update)$/i
      ],

      PASSWORD_RESET: [
        /^(password|reset|forgot)\s+(password|pin)$/i,
        /^(change|update)\s+(password|pin)$/i
      ]
    };
    
    // Enhanced fund name patterns
    this.fundPatterns = [
      /(HDFC|SBI|ICICI|Axis|Kotak|Mirae|Parag Parikh|Nippon|Tata|Aditya Birla|Franklin|DSP|IDFC|Canara|Union|PGIM|Mahindra|Sundaram|L&T|Invesco|Baroda|HSBC|Edelweiss|Motilal Oswal|Quantum|IIFL|JM|Principal|BNP Paribas|UTI|Reliance|Templeton)\s+(Flexicap|Largecap|Midcap|Smallcap|Multicap|Value|Growth|Hybrid|Debt|Liquid|Gold|International|Technology|Healthcare|Banking|Infrastructure|Real Estate|Energy|Consumer|Auto|Pharma|IT|FMCG|Banking|Financial|Services|Capital|Opportunities|Advantage|Dynamic|Strategic|Premium|Elite|Prime|Select|Focus|Vision|Future|Next|Digital|Smart|Active|Passive|Index|ETF|Fund|Scheme|Plan|Portfolio|Basket|Mix|Blend|Allocation|Strategy|Approach|Style|Theme|Sector|Category|Class|Series|Option|Growth|Dividend|Payout|Reinvestment|Direct|Regular|Institutional|Retail|Wholesale|Super|Ultra|Max|Plus|Pro)/gi
    ];

    // Keywords for fallback intent detection
    this.keywordPatterns = {
      ONBOARDING: ['name', 'email', 'pan', 'aadhaar', 'bank', 'kyc', 'verify', 'register'],
      SIP_CREATE: ['sip', 'invest', 'start', 'fund', 'amount', 'monthly', 'frequency'],
      PORTFOLIO_VIEW: ['portfolio', 'holdings', 'value', 'worth', 'returns', 'performance'],
      AI_ANALYSIS: ['analyse', 'analyze', 'review', 'recommend', 'advice', 'insights'],
      REWARDS: ['rewards', 'points', 'cashback', 'bonus', 'redeem'],
      REFERRAL: ['referral', 'refer', 'code', 'link', 'invite'],
      LEADERBOARD: ['leaderboard', 'top', 'best', 'rankings'],
      STATEMENT: ['statement', 'report', 'download', 'transaction'],
      HELP: ['help', 'support', 'what', 'how', 'menu']
    };
  }

  /**
   * Enhanced parse message with context awareness
   */
  parseMessage(message, context = {}) {
    try {
      const cleanMessage = message.trim();
      
      // First, try exact pattern matching
      const exactMatch = this.findExactMatch(cleanMessage);
      if (exactMatch) {
        // Stricter ONBOARDING: Only if valid data extracted
        if (exactMatch.intent === 'ONBOARDING') {
          const data = this.extractData('ONBOARDING', exactMatch.match, cleanMessage);
          if (!this.isValidOnboardingData(data)) {
            return { intent: 'UNKNOWN', confidence: 0.3, extractedData: {} };
          }
          return { ...exactMatch, extractedData: data };
        }
        // Stricter SIP_CREATE: Require fundName
        if (exactMatch.intent === 'SIP_CREATE') {
          const data = this.extractData('SIP_CREATE', exactMatch.match, cleanMessage);
          if (!data.fundName || data.fundName.length < 3) {
            return { intent: 'UNKNOWN', confidence: 0.3, extractedData: {} };
          }
          return { ...exactMatch, extractedData: data };
        }
        // Stricter AI_ANALYSIS: Require fundName
        if (exactMatch.intent === 'AI_ANALYSIS') {
          const data = this.extractData('AI_ANALYSIS', exactMatch.match, cleanMessage);
          if (!data.fundName || data.fundName.length < 3) {
            return { intent: 'UNKNOWN', confidence: 0.3, extractedData: {} };
          }
          return { ...exactMatch, extractedData: data };
        }
        return exactMatch;
      }
      
      // If no exact match, try context-aware parsing
      if (context.lastIntent && context.pendingAction) {
        const contextMatch = this.parseWithContext(cleanMessage, context);
        if (contextMatch) {
          return contextMatch;
        }
      }
      
      // Try keyword-based fallback
      const keywordMatch = this.parseWithKeywords(cleanMessage);
      if (keywordMatch) {
        // Stricter ONBOARDING for keyword fallback
        if (keywordMatch.intent === 'ONBOARDING') {
          const data = this.extractData('ONBOARDING', [], cleanMessage);
          if (!this.isValidOnboardingData(data)) {
            return { intent: 'UNKNOWN', confidence: 0.3, extractedData: {} };
          }
          return { ...keywordMatch, extractedData: data };
        }
        return keywordMatch;
      }
      
      // Fallback to UNKNOWN
      return { intent: 'UNKNOWN', confidence: 0.2, extractedData: {} };
    } catch (err) {
      logger.error('Message parsing error:', err);
      return { intent: 'UNKNOWN', confidence: 0.1, extractedData: {} };
    }
  }

  /**
   * Find exact pattern match
   */
  findExactMatch(message) {
    for (const [intent, patterns] of Object.entries(this.intentPatterns)) {
      for (const pattern of patterns) {
        const match = message.match(pattern);
        if (match) {
          const result = {
            intent: intent,
            confidence: this.calculateConfidence(match, message),
            originalMessage: message,
            extractedData: this.extractData(intent, match, message)
          };
          
          logger.info(`Intent detected: ${intent} (confidence: ${result.confidence})`);
          return result;
        }
      }
    }
    return null;
  }

  /**
   * Parse message with context awareness
   */
  parseWithContext(message, context) {
    const { lastIntent, pendingAction } = context;
    
    // Handle multi-step flows
    if (lastIntent === 'SIP_CREATE') {
      if (pendingAction === 'WAITING_FOR_FUND') {
        const fundName = this.extractFundName(message);
        if (fundName) {
          return {
            intent: 'SIP_CREATE',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { fundName }
          };
        }
      }
      
      if (pendingAction === 'WAITING_FOR_AMOUNT') {
        const amount = this.parseAmount(message);
        if (amount) {
          return {
            intent: 'SIP_CREATE',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { amount }
          };
        }
      }
    }
    
    if (lastIntent === 'ONBOARDING') {
      if (pendingAction === 'WAITING_FOR_NAME') {
        if (message.length > 2 && !this.isEmail(message) && !this.isPAN(message) && !this.isAadhaar(message)) {
          return {
            intent: 'ONBOARDING',
            confidence: 0.8,
            originalMessage: message,
            extractedData: { name: message }
          };
        }
      }
      
      if (pendingAction === 'WAITING_FOR_EMAIL') {
        if (this.isEmail(message)) {
          return {
            intent: 'ONBOARDING',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { email: message }
          };
        }
      }
      
      if (pendingAction === 'WAITING_FOR_PAN') {
        if (this.isPAN(message)) {
          return {
            intent: 'ONBOARDING',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { pan: message }
          };
        }
      }

      if (pendingAction === 'WAITING_FOR_AADHAAR') {
        if (this.isAadhaar(message)) {
          return {
            intent: 'ONBOARDING',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { aadhaar: message }
          };
        }
      }

      if (pendingAction === 'WAITING_FOR_BANK') {
        if (this.isBankAccount(message)) {
          return {
            intent: 'ONBOARDING',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { bankAccount: message }
          };
        }
      }

      if (pendingAction === 'WAITING_FOR_RISK') {
        const riskLevel = this.extractRiskLevel(message);
        if (riskLevel) {
          return {
            intent: 'ONBOARDING',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { riskAppetite: riskLevel }
          };
        }
      }

      if (pendingAction === 'WAITING_FOR_AMOUNT') {
        const amount = this.parseAmount(message);
        if (amount) {
          return {
            intent: 'ONBOARDING',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { investmentAmount: amount }
          };
        }
      }
    }

    if (lastIntent === 'SIP_STOP') {
      if (pendingAction === 'WAITING_FOR_FUND_NAME') {
        const fundName = this.extractFundName(message);
        if (fundName) {
          return {
            intent: 'SIP_STOP',
            confidence: 0.9,
            originalMessage: message,
            extractedData: { fundName }
          };
        }
      }
    }
    
    return null;
  }

  /**
   * Extract risk level from message
   */
  extractRiskLevel(message) {
    const lowerMessage = message.toLowerCase();
    
    if (lowerMessage.includes('conservative') || lowerMessage.includes('low risk') || lowerMessage.includes('safe')) {
      return 'CONSERVATIVE';
    } else if (lowerMessage.includes('moderate') || lowerMessage.includes('medium') || lowerMessage.includes('balanced')) {
      return 'MODERATE';
    } else if (lowerMessage.includes('aggressive') || lowerMessage.includes('high risk') || lowerMessage.includes('growth')) {
      return 'AGGRESSIVE';
    }
    
    return null;
  }

  /**
   * Parse message using keyword fallback
   */
  parseWithKeywords(message) {
    const lowerMessage = message.toLowerCase();
    
    for (const [intent, keywords] of Object.entries(this.keywordPatterns)) {
      for (const keyword of keywords) {
        if (lowerMessage.includes(keyword)) {
          return {
            intent: intent,
            confidence: 0.6,
            originalMessage: message,
            extractedData: this.extractDataFromKeywords(intent, message)
          };
        }
      }
    }
    
    return null;
  }

  /**
   * Extract data from keywords with enhanced logic
   */
  extractDataFromKeywords(intent, message) {
    const data = {};
    
    switch (intent) {
      case 'SIP_CREATE':
        data.amount = this.parseAmount(message);
        data.fundName = this.extractFundName(message);
        break;
      case 'AI_ANALYSIS':
        data.fundName = this.extractFundName(message);
        break;
      case 'ONBOARDING':
        if (this.isEmail(message)) data.email = message;
        else if (this.isPAN(message)) data.pan = message;
        else if (this.isAadhaar(message)) data.aadhaar = message;
        else if (this.isBankAccount(message)) data.bankAccount = message;
        else if (this.extractRiskLevel(message)) data.riskAppetite = this.extractRiskLevel(message);
        else if (this.parseAmount(message)) data.investmentAmount = this.parseAmount(message);
        else if (message.length > 2) data.name = message;
        break;
      case 'SIP_STOP':
        data.fundName = this.extractFundName(message);
        break;
    }
    
    return data;
  }

  /**
   * Enhanced data extraction with better validation
   */
  extractData(intent, match, originalMessage) {
    const data = {};
    match = match || [];
    switch (intent) {
      case 'ONBOARDING':
        if (match[2]) {
          const value = match[2].trim();
          if (this.isEmail(value)) {
            data.email = value;
          } else if (this.isPAN(value)) {
            data.pan = value.toUpperCase();
          } else if (this.isAadhaar(value)) {
            data.aadhaar = value;
          } else if (this.isBankAccount(value)) {
            data.bankAccount = value;
          } else if (this.extractRiskLevel(value)) {
            data.riskAppetite = this.extractRiskLevel(value);
          } else if (this.parseAmount(value)) {
            data.investmentAmount = this.parseAmount(value);
          } else {
            data.name = value;
          }
        } else if (match[3]) {
          const value = match[3].trim();
          if (this.isEmail(value)) {
            data.email = value;
          } else if (this.isPAN(value)) {
            data.pan = value.toUpperCase();
          } else if (this.isAadhaar(value)) {
            data.aadhaar = value;
          } else if (this.isBankAccount(value)) {
            data.bankAccount = value;
          } else if (this.extractRiskLevel(value)) {
            data.riskAppetite = this.extractRiskLevel(value);
          } else if (this.parseAmount(value)) {
            data.investmentAmount = this.parseAmount(value);
          } else {
            data.name = value;
          }
        } else if (match[4]) {
          const value = match[4].trim();
          if (this.isEmail(value)) {
            data.email = value;
          } else if (this.isPAN(value)) {
            data.pan = value.toUpperCase();
          } else if (this.isAadhaar(value)) {
            data.aadhaar = value;
          } else if (this.isBankAccount(value)) {
            data.bankAccount = value;
          } else if (this.extractRiskLevel(value)) {
            data.riskAppetite = this.extractRiskLevel(value);
          } else if (this.parseAmount(value)) {
            data.investmentAmount = this.parseAmount(value);
          }
        }
        break;
      case 'SIP_CREATE':
        data.amount = (match[2] ? this.parseAmount(match[2]) : null) || (match[3] ? this.parseAmount(match[3]) : null);
        data.fundName = (match[3] ? this.extractFundName(match[3]) : null) || (match[4] ? this.extractFundName(match[4]) : null) || this.extractFundName(originalMessage);
        break;
      case 'SIP_STOP':
        data.fundName = (match[2] ? this.extractFundName(match[2]) : null) || this.extractFundName(originalMessage);
        break;
      case 'LUMP_SUM':
        data.amount = match[2] ? this.parseAmount(match[2]) : null;
        data.fundName = match[3] ? this.extractFundName(match[3]) : this.extractFundName(originalMessage);
        break;
      case 'AI_ANALYSIS':
        data.fundName = (match[2] ? this.extractFundName(match[2]) : null) || this.extractFundName(originalMessage);
        break;
      case 'COPY_PORTFOLIO':
        data.leaderNumber = (match[2] ? parseInt(match[2]) : null) || (match[3] ? parseInt(match[3]) : 1);
        break;
      case 'CONFIRMATION':
        data.confirmed = match[0] ? /^(yes|y|confirm|ok|okay|sure|proceed|correct|right|perfect|good)$/i.test(match[0]) : false;
        break;
    }
    return data;
  }

  /**
   * Enhanced amount parsing
   */
  parseAmount(amountStr) {
    if (!amountStr) return null;
    
    // Remove currency symbols and commas
    const cleanAmount = amountStr.replace(/[₹,]/g, '');
    const amount = parseFloat(cleanAmount);
    
    return isNaN(amount) ? null : amount;
  }

  /**
   * Enhanced fund name extraction
   */
  extractFundName(message) {
    if (!message) return null;
    
    // Try to match known fund patterns
    for (const pattern of this.fundPatterns) {
      const match = message.match(pattern);
      if (match) {
        return match[0].trim();
      }
    }
    
    // Fallback: extract any capitalized words that might be fund names
    const words = message.split(' ');
    const fundWords = words.filter(word => 
      word.length > 2 && 
      /^[A-Z][a-z]+/.test(word) && 
      !['The', 'And', 'For', 'With', 'From', 'This', 'That'].includes(word)
    );
    
    return fundWords.length > 0 ? fundWords.join(' ') : null;
  }

  /**
   * Enhanced validation methods
   */
  isEmail(text) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(text);
  }

  isPAN(text) {
    return /^[A-Z]{5}[0-9]{4}[A-Z]$/.test(text);
  }

  /**
   * Enhanced Aadhaar validation (12 digits only)
   */
  isAadhaar(text) {
    return /^\d{12}$/.test(text);
  }

  /**
   * Enhanced bank account validation (10-16 digits only)
   */
  isBankAccount(text) {
    return /^\d{10,16}$/.test(text);
  }

  /**
   * Calculate confidence score for intent detection
   */
  calculateConfidence(match, originalMessage) {
    const matchLength = match[0].length;
    const messageLength = originalMessage.length;
    const lengthRatio = matchLength / messageLength;
    
    // Higher confidence for exact matches
    if (matchLength === messageLength) {
      return 1.0;
    }
    
    // Good confidence for substantial matches
    if (lengthRatio > 0.7) {
      return 0.9;
    }
    
    // Moderate confidence for partial matches
    if (lengthRatio > 0.5) {
      return 0.7;
    }
    
    // Lower confidence for small matches
    return 0.5;
  }

  /**
   * Validate extracted data
   */
  validateData(intent, data) {
    const errors = [];
    
    switch (intent) {
      case 'SIP_CREATE':
      case 'LUMP_SUM':
        if (!data.amount || data.amount < 100) {
          errors.push('Minimum investment amount is ₹100');
        }
        if (!data.fundName) {
          errors.push('Fund name is required');
        }
        break;
        
      case 'ONBOARDING':
        if (data.email && !this.isEmail(data.email)) {
          errors.push('Invalid email format');
        }
        if (data.pan && !this.isPAN(data.pan)) {
          errors.push('Invalid PAN format');
        }
        break;
    }
    
    return errors;
  }

  /**
   * Get suggested responses based on intent
   */
  getSuggestedResponses(intent, data = {}) {
    const suggestions = {
      GREETING: ['Welcome! How can I help you today?', 'Hello! What would you like to do?'],
      ONBOARDING: ['Please share your full name', 'What is your email address?', 'Please provide your PAN number'],
      PORTFOLIO_VIEW: ['Here is your portfolio summary', 'Your current holdings are'],
      SIP_CREATE: ['Please confirm your SIP order', 'Would you like to proceed with this investment?'],
      HELP: ['I can help you with portfolio, SIPs, rewards, and fund analysis', 'Here are the available commands']
    };
    
    return suggestions[intent] || ['How can I help you?'];
  }

  /**
   * Helper: Validate ONBOARDING data
   */
  isValidOnboardingData(data) {
    if (!data) return false;
    if (data.name && data.name.length > 2) return true;
    if (data.email && this.isEmail(data.email)) return true;
    if (data.pan && this.isPAN(data.pan)) return true;
    if (data.aadhaar && this.isAadhaar(data.aadhaar)) return true;
    if (data.bankAccount && this.isBankAccount(data.bankAccount)) return true;
    return false;
  }
}

module.exports = new MessageParser(); 