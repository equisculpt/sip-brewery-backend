const axios = require('axios');
const { spawn } = require('child_process');
const logger = require('../utils/logger');

class FreedomFinanceAI {
  constructor() {
    this.ollamaModel = process.env.OLLAMA_MODEL || 'mistral';
    this.ollamaUrl = process.env.OLLAMA_URL || 'http://localhost:11434';
    this.maxTokens = 2048;
    this.temperature = 0.7;
  }

  /**
   * Initialize Ollama and load the model
   */
  async initialize() {
    try {
      // Check if Ollama is running
      await this.checkOllamaStatus();
      
      // Pull the model if not available
      await this.pullModel();
      
      logger.info(`Freedom Finance AI initialized with model: ${this.ollamaModel}`);
      return true;
    } catch (error) {
      logger.error('Failed to initialize Freedom Finance AI:', error);
      return false;
    }
  }

  /**
   * Check if Ollama service is running
   */
  async checkOllamaStatus() {
    try {
      const response = await axios.get(`${this.ollamaUrl}/api/tags`);
      return response.status === 200;
    } catch (error) {
      throw new Error('Ollama service not running. Please start Ollama first.');
    }
  }

  /**
   * Pull the specified model
   */
  async pullModel() {
    return new Promise((resolve, reject) => {
      const pull = spawn('ollama', ['pull', this.ollamaModel]);
      
      pull.stdout.on('data', (data) => {
        logger.info(`Ollama pull: ${data}`);
      });
      
      pull.stderr.on('data', (data) => {
        logger.warn(`Ollama pull warning: ${data}`);
      });
      
      pull.on('close', (code) => {
        if (code === 0) {
          logger.info(`Model ${this.ollamaModel} pulled successfully`);
          resolve();
        } else {
          reject(new Error(`Failed to pull model ${this.ollamaModel}`));
        }
      });
    });
  }

  /**
   * Generate AI response using Ollama
   */
  async generateResponse(prompt, context = {}) {
    try {
      const response = await axios.post(`${this.ollamaUrl}/api/generate`, {
        model: this.ollamaModel,
        prompt: this.buildPrompt(prompt, context),
        stream: false,
        options: {
          temperature: this.temperature,
          num_predict: this.maxTokens
        }
      });

      return {
        success: true,
        response: response.data.response,
        usage: {
          prompt_tokens: response.data.prompt_eval_count,
          completion_tokens: response.data.eval_count,
          total_tokens: response.data.prompt_eval_count + response.data.eval_count
        }
      };
    } catch (error) {
      logger.error('AI generation failed:', error);
      return {
        success: false,
        error: error.message,
        fallback: this.getFallbackResponse(prompt)
      };
    }
  }

  /**
   * Build SEBI-compliant prompt
   */
  buildPrompt(userQuery, context) {
    const { portfolioData, navData, marketData, userHistory } = context;
    
    return `You are Freedom Finance AI, an expert SEBI-compliant Indian financial analyst. 

IMPORTANT COMPLIANCE RULES:
- NEVER give direct buy/sell recommendations
- NEVER suggest specific stock picks
- ALWAYS include disclaimers
- Focus on educational insights and analysis
- Recommend consulting registered advisors

USER PORTFOLIO DATA:
${portfolioData ? JSON.stringify(portfolioData, null, 2) : 'No portfolio data provided'}

FUND NAV DATA:
${navData ? JSON.stringify(navData, null, 2) : 'No NAV data provided'}

MARKET DATA:
${marketData ? JSON.stringify(marketData, null, 2) : 'No market data provided'}

USER HISTORY:
${userHistory ? JSON.stringify(userHistory, null, 2) : 'No user history provided'}

USER QUERY: ${userQuery}

Provide a comprehensive analysis including:
1. Portfolio performance insights
2. Risk assessment
3. Diversification analysis
4. Educational recommendations
5. SEBI compliance disclaimers

Format your response in JSON:
{
  "analysis": "Detailed portfolio analysis",
  "insights": ["Insight 1", "Insight 2"],
  "recommendations": ["Educational recommendation 1", "Educational recommendation 2"],
  "risk_level": "LOW/MEDIUM/HIGH",
  "diversification_score": 0-100,
  "disclaimer": "SEBI compliance disclaimer"
}`;
  }

  /**
   * Get fallback response when AI fails
   */
  getFallbackResponse(query) {
    return {
      analysis: "I'm currently experiencing technical difficulties. Please try again later or consult a registered financial advisor for personalized advice.",
      insights: ["System temporarily unavailable"],
      recommendations: ["Please consult a registered financial advisor"],
      risk_level: "UNKNOWN",
      diversification_score: 0,
      disclaimer: "This is a system-generated response. Please consult a registered financial advisor before making investment decisions."
    };
  }

  /**
   * Analyze portfolio performance
   */
  async analyzePortfolio(portfolioData, navData) {
    const prompt = "Analyze this portfolio's performance, risk, and diversification";
    const context = { portfolioData, navData };
    
    const response = await this.generateResponse(prompt, context);
    
    if (response.success) {
      try {
        return JSON.parse(response.response);
      } catch (error) {
        return this.getFallbackResponse(prompt);
      }
    }
    
    return response.fallback;
  }

  /**
   * Generate Smart SIP recommendations
   */
  async generateSmartSIP(userPreferences, marketData) {
    const prompt = "Generate Smart SIP recommendations based on market conditions and user preferences";
    const context = { marketData, userHistory: userPreferences };
    
    const response = await this.generateResponse(prompt, context);
    
    if (response.success) {
      try {
        return JSON.parse(response.response);
      } catch (error) {
        return this.getFallbackResponse(prompt);
      }
    }
    
    return response.fallback;
  }

  /**
   * Generate fund comparison analysis
   */
  async compareFunds(fundData1, fundData2) {
    const prompt = "Compare these two mutual funds and provide educational insights";
    const context = { 
      portfolioData: { fund1: fundData1, fund2: fundData2 },
      navData: { fund1: fundData1.nav, fund2: fundData2.nav }
    };
    
    const response = await this.generateResponse(prompt, context);
    
    if (response.success) {
      try {
        return JSON.parse(response.response);
      } catch (error) {
        return this.getFallbackResponse(prompt);
      }
    }
    
    return response.fallback;
  }

  /**
   * Generate market insights
   */
  async generateMarketInsights(marketData) {
    const prompt = "Provide educational market insights and trends analysis";
    const context = { marketData };
    
    const response = await this.generateResponse(prompt, context);
    
    if (response.success) {
      try {
        return JSON.parse(response.response);
      } catch (error) {
        return this.getFallbackResponse(prompt);
      }
    }
    
    return response.fallback;
  }

  /**
   * Learn from user feedback
   */
  async learnFromFeedback(userAction, aiRecommendation, outcome) {
    const prompt = "Learn from user action and improve future recommendations";
    const context = { 
      userHistory: {
        action: userAction,
        recommendation: aiRecommendation,
        outcome: outcome,
        timestamp: new Date().toISOString()
      }
    };
    
    // Store learning data for future improvements
    logger.info('Learning from user feedback:', context.userHistory);
    
    return {
      success: true,
      message: 'Feedback recorded for AI improvement'
    };
  }
}

module.exports = FreedomFinanceAI; 