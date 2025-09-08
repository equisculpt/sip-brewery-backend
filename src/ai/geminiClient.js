const { GoogleGenerativeAI } = require('@google/generative-ai');
const logger = require('../utils/logger');

class GeminiClient {
  constructor() {
    this.apiKey = process.env.GEMINI_API_KEY;
    this.genAI = null;
    this.model = null;
    
    if (this.apiKey) {
      this.genAI = new GoogleGenerativeAI(this.apiKey);
      this.model = this.genAI.getGenerativeModel({ model: 'gemini-pro' });
    } else {
      logger.warn('GEMINI_API_KEY not found. AI features will be disabled.');
    }
  }

  /**
   * Analyze a mutual fund using Gemini AI
   */
  async analyzeFund(fundName, options = {}) {
    try {
      if (!this.model) {
        return this.getFallbackAnalysis(fundName);
      }

      const prompt = this.buildFundAnalysisPrompt(fundName, options);
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      // Add SEBI disclaimer
      const disclaimer = this.getSebiDisclaimer();
      const finalResponse = `${text}\n\n${disclaimer}`;

      logger.info(`Gemini AI analysis completed for ${fundName}`);

      return {
        success: true,
        analysis: text,
        disclaimer: disclaimer,
        fullResponse: finalResponse,
        aiProvider: 'GEMINI'
      };
    } catch (error) {
      logger.error('Gemini AI analysis failed:', error);
      return this.getFallbackAnalysis(fundName);
    }
  }

  /**
   * Generate intelligent response for user queries
   */
  async generateResponse(userMessage, context = {}) {
    try {
      if (!this.model) {
        return this.getFallbackResponse(userMessage);
      }

      const prompt = this.buildResponsePrompt(userMessage, context);
      const result = await this.model.generateContent(prompt);
      const response = await result.response;
      const text = response.text();

      logger.info('Gemini AI response generated successfully');

      return {
        success: true,
        response: text,
        aiProvider: 'GEMINI'
      };
    } catch (error) {
      logger.error('Gemini AI response generation failed:', error);
      return this.getFallbackResponse(userMessage);
    }
  }

  /**
   * Build fund analysis prompt
   */
  buildFundAnalysisPrompt(fundName, options = {}) {
    const {
      timePeriod = '3 years',
      focusAreas = ['performance', 'risk', 'benchmark', 'sector exposure'],
      wordLimit = 100
    } = options;

    return `
You are a mutual fund analysis assistant for SIPBrewery. Analyze the fund "${fundName}" based on the following criteria:

Focus Areas:
- Past ${timePeriod} performance
- Risk assessment
- Benchmark comparison
- Sector exposure and diversification
- Fund manager track record

Requirements:
- Provide a concise summary in ${wordLimit} words
- Use simple, easy-to-understand language
- Focus on factual data and trends
- Avoid making specific investment recommendations
- Be objective and balanced

Format your response as a clear, structured analysis that a retail investor can easily understand.

Fund to analyze: ${fundName}
    `;
  }

  /**
   * Build response prompt for general queries
   */
  buildResponsePrompt(userMessage, context = {}) {
    const {
      userType = 'retail investor',
      tone = 'friendly',
      language = 'English'
    } = context;

    return `
You are SIPBrewery's WhatsApp investment assistant. Respond to the user's message in a ${tone} tone.

User Message: "${userMessage}"

Guidelines:
- Be helpful and informative
- Use simple, clear language
- If discussing investments, always include disclaimers
- Don't give specific investment advice
- Focus on education and information
- Keep responses concise and engaging
- Use emojis appropriately for WhatsApp

Respond as a helpful investment assistant for ${userType}s.
    `;
  }

  /**
   * Get SEBI compliance disclaimer
   */
  getSebiDisclaimer() {
    return `🔍 *Disclaimer*: This is a data-based AI insight, not investment advice. Please consult a SEBI-registered advisor before making investment decisions. Past performance doesn't guarantee future returns.`;
  }

  /**
   * Fallback analysis when AI is not available
   */
  getFallbackAnalysis(fundName) {
    const fallbackResponses = {
      'HDFC Flexicap': `HDFC Flexicap is a multi-cap equity fund that invests across market capitalizations. It has shown consistent performance over the past 3 years with moderate risk. The fund has beaten its benchmark (Nifty 500) consistently and has good sector diversification.`,
      'SBI Smallcap': `SBI Smallcap Fund focuses on small-cap companies with high growth potential. It has delivered strong returns but comes with higher volatility. The fund has outperformed its benchmark (Nifty Smallcap 250) in recent years.`,
      'Parag Parikh Flexicap': `Parag Parikh Flexicap is known for its value investing approach and international exposure. It has delivered consistent returns with lower volatility compared to peers. The fund has a unique mandate allowing up to 35% international investments.`,
      'Mirae Asset Largecap': `Mirae Asset Largecap Fund invests primarily in large-cap stocks. It has shown strong performance with good risk management. The fund has consistently beaten its benchmark (Nifty 100) and has a well-diversified portfolio.`
    };

    const response = fallbackResponses[fundName] || 
      `${fundName} is a mutual fund. For detailed analysis, please visit our website or consult a financial advisor.`;

    return {
      success: true,
      analysis: response,
      disclaimer: this.getSebiDisclaimer(),
      fullResponse: `${response}\n\n${this.getSebiDisclaimer()}`,
      aiProvider: 'FALLBACK'
    };
  }

  /**
   * Fallback response when AI is not available
   */
  getFallbackResponse(userMessage) {
    const lowerMessage = userMessage.toLowerCase();
    
    if (lowerMessage.includes('hi') || lowerMessage.includes('hello')) {
      return {
        success: true,
        response: "Hello! I'm SIPBrewery's investment assistant 🤖. How can I help you today?",
        aiProvider: 'FALLBACK'
      };
    }
    
    if (lowerMessage.includes('portfolio') || lowerMessage.includes('holdings')) {
      return {
        success: true,
        response: "To view your portfolio, please type 'My Portfolio' and I'll show you your current holdings and performance.",
        aiProvider: 'FALLBACK'
      };
    }
    
    if (lowerMessage.includes('sip') || lowerMessage.includes('invest')) {
      return {
        success: true,
        response: "To start a SIP, please specify the amount and fund name. For example: 'I want to invest ₹5000 in HDFC Flexicap'",
        aiProvider: 'FALLBACK'
      };
    }
    
    if (lowerMessage.includes('reward') || lowerMessage.includes('cashback')) {
      return {
        success: true,
        response: "To check your rewards, type 'My Rewards' and I'll show you your loyalty points, cashback, and referral bonuses.",
        aiProvider: 'FALLBACK'
      };
    }
    
    return {
      success: true,
      response: "I'm here to help with your mutual fund investments! You can ask me about your portfolio, start SIPs, check rewards, or get fund analysis. What would you like to know?",
      aiProvider: 'FALLBACK'
    };
  }

  /**
   * Check if AI is available
   */
  isAvailable() {
    return !!this.model;
  }

  /**
   * Get AI status
   */
  getStatus() {
    return {
      available: this.isAvailable(),
      provider: this.isAvailable() ? 'GEMINI' : 'FALLBACK',
      apiKeyConfigured: !!this.apiKey
    };
  }
}

module.exports = new GeminiClient(); 