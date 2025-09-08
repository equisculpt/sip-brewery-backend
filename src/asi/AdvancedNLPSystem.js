/**
 * 🧠 ADVANCED NLP SYSTEM
 * 
 * State-of-the-art Natural Language Processing for Financial Analysis
 * Sentiment analysis, entity extraction, document intelligence, and market insights
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Advanced Financial NLP
 */

const EventEmitter = require('events');
const logger = require('../utils/logger');

class AdvancedNLPSystem extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      maxDocumentLength: options.maxDocumentLength || 10000,
      sentimentThreshold: options.sentimentThreshold || 0.7,
      entityConfidenceThreshold: options.entityConfidenceThreshold || 0.8,
      batchSize: options.batchSize || 32,
      cacheEnabled: options.cacheEnabled !== false,
      ...options
    };
    
    // NLP Models and processors
    this.models = {
      sentimentAnalyzer: null,
      entityExtractor: null,
      documentClassifier: null,
      marketInsightGenerator: null,
      riskAssessmentNLP: null
    };
    
    // Financial domain knowledge
    this.financialTerms = new Map();
    this.marketIndicators = new Map();
    this.riskKeywords = new Set();
    
    // Processing cache
    this.processingCache = new Map();
    
    // Performance metrics
    this.metrics = {
      documentsProcessed: 0,
      sentimentAccuracy: 0,
      entityExtractionAccuracy: 0,
      averageProcessingTime: 0,
      cacheHitRate: 0
    };
    
    this.isInitialized = false;
  }

  async initialize() {
    try {
      logger.info('🧠 Initializing Advanced NLP System...');
      
      // Initialize financial domain knowledge
      await this.initializeFinancialKnowledge();
      
      // Initialize NLP models
      await this.initializeNLPModels();
      
      // Initialize sentiment analysis
      await this.initializeSentimentAnalysis();
      
      // Initialize entity extraction
      await this.initializeEntityExtraction();
      
      // Initialize document classification
      await this.initializeDocumentClassification();
      
      this.isInitialized = true;
      logger.info('✅ Advanced NLP System initialized successfully');
      
    } catch (error) {
      logger.error('❌ Advanced NLP System initialization failed:', error);
      throw error;
    }
  }

  async initializeFinancialKnowledge() {
    // Financial terms with sentiment weights
    const financialTerms = {
      // Positive indicators
      'bullish': { sentiment: 0.8, category: 'market_direction', weight: 1.0 },
      'growth': { sentiment: 0.7, category: 'performance', weight: 0.9 },
      'profit': { sentiment: 0.8, category: 'financial', weight: 1.0 },
      'revenue': { sentiment: 0.6, category: 'financial', weight: 0.8 },
      'outperform': { sentiment: 0.9, category: 'performance', weight: 1.0 },
      'strong': { sentiment: 0.7, category: 'assessment', weight: 0.8 },
      'positive': { sentiment: 0.8, category: 'sentiment', weight: 0.9 },
      'upward': { sentiment: 0.7, category: 'trend', weight: 0.8 },
      'rally': { sentiment: 0.8, category: 'market_movement', weight: 0.9 },
      'surge': { sentiment: 0.9, category: 'market_movement', weight: 1.0 },
      
      // Negative indicators
      'bearish': { sentiment: -0.8, category: 'market_direction', weight: 1.0 },
      'decline': { sentiment: -0.7, category: 'performance', weight: 0.9 },
      'loss': { sentiment: -0.8, category: 'financial', weight: 1.0 },
      'deficit': { sentiment: -0.7, category: 'financial', weight: 0.8 },
      'underperform': { sentiment: -0.9, category: 'performance', weight: 1.0 },
      'weak': { sentiment: -0.7, category: 'assessment', weight: 0.8 },
      'negative': { sentiment: -0.8, category: 'sentiment', weight: 0.9 },
      'downward': { sentiment: -0.7, category: 'trend', weight: 0.8 },
      'crash': { sentiment: -0.9, category: 'market_movement', weight: 1.0 },
      'plunge': { sentiment: -0.9, category: 'market_movement', weight: 1.0 },
      
      // Neutral/Technical terms
      'volatility': { sentiment: 0.0, category: 'risk', weight: 0.7 },
      'liquidity': { sentiment: 0.1, category: 'market_structure', weight: 0.6 },
      'dividend': { sentiment: 0.3, category: 'financial', weight: 0.7 },
      'yield': { sentiment: 0.2, category: 'financial', weight: 0.6 },
      'beta': { sentiment: 0.0, category: 'risk', weight: 0.5 },
      'alpha': { sentiment: 0.4, category: 'performance', weight: 0.8 },
      'correlation': { sentiment: 0.0, category: 'statistical', weight: 0.4 }
    };
    
    for (const [term, data] of Object.entries(financialTerms)) {
      this.financialTerms.set(term.toLowerCase(), data);
    }
    
    // Market indicators
    const marketIndicators = {
      'rsi': { type: 'technical', range: [0, 100], interpretation: 'momentum' },
      'macd': { type: 'technical', range: [-Infinity, Infinity], interpretation: 'trend' },
      'pe_ratio': { type: 'fundamental', range: [0, Infinity], interpretation: 'valuation' },
      'debt_to_equity': { type: 'fundamental', range: [0, Infinity], interpretation: 'leverage' },
      'roe': { type: 'fundamental', range: [-100, 100], interpretation: 'profitability' },
      'current_ratio': { type: 'fundamental', range: [0, Infinity], interpretation: 'liquidity' }
    };
    
    for (const [indicator, data] of Object.entries(marketIndicators)) {
      this.marketIndicators.set(indicator, data);
    }
    
    // Risk keywords
    this.riskKeywords = new Set([
      'risk', 'volatility', 'uncertainty', 'loss', 'decline', 'crash',
      'bubble', 'correction', 'recession', 'inflation', 'deflation',
      'default', 'bankruptcy', 'liquidation', 'margin call', 'leverage'
    ]);
    
    logger.info(`📚 Financial knowledge initialized: ${this.financialTerms.size} terms, ${this.marketIndicators.size} indicators`);
  }

  async initializeNLPModels() {
    // Initialize sentiment analysis model (simplified implementation)
    this.models.sentimentAnalyzer = {
      analyze: async (text) => {
        return await this.analyzeSentiment(text);
      }
    };
    
    // Initialize entity extractor
    this.models.entityExtractor = {
      extract: async (text) => {
        return await this.extractEntities(text);
      }
    };
    
    // Initialize document classifier
    this.models.documentClassifier = {
      classify: async (text) => {
        return await this.classifyDocument(text);
      }
    };
    
    // Initialize market insight generator
    this.models.marketInsightGenerator = {
      generate: async (text, context) => {
        return await this.generateMarketInsights(text, context);
      }
    };
    
    // Initialize risk assessment NLP
    this.models.riskAssessmentNLP = {
      assess: async (text) => {
        return await this.assessRiskFromText(text);
      }
    };
    
    logger.info('🤖 NLP models initialized');
  }

  async initializeSentimentAnalysis() {
    // Advanced sentiment analysis with financial context
    this.sentimentAnalyzer = {
      // Lexicon-based approach with financial terms
      lexiconScore: (tokens) => {
        let score = 0;
        let weight = 0;
        
        for (const token of tokens) {
          const term = this.financialTerms.get(token.toLowerCase());
          if (term) {
            score += term.sentiment * term.weight;
            weight += term.weight;
          }
        }
        
        return weight > 0 ? score / weight : 0;
      },
      
      // Context-aware sentiment analysis
      contextualScore: (text, context = {}) => {
        const tokens = this.tokenize(text);
        let score = this.sentimentAnalyzer.lexiconScore(tokens);
        
        // Adjust based on context
        if (context.marketCondition === 'bull') {
          score *= 1.1; // Amplify positive sentiment in bull market
        } else if (context.marketCondition === 'bear') {
          score *= 0.9; // Dampen positive sentiment in bear market
        }
        
        // Adjust based on document type
        if (context.documentType === 'earnings_call') {
          score *= 1.2; // Earnings calls are more impactful
        } else if (context.documentType === 'analyst_report') {
          score *= 1.1; // Analyst reports are moderately impactful
        }
        
        return Math.max(-1, Math.min(1, score));
      }
    };
    
    logger.info('💭 Sentiment analysis initialized');
  }

  async initializeEntityExtraction() {
    // Financial entity patterns
    this.entityPatterns = {
      // Company names (simplified pattern)
      company: /\b[A-Z][a-z]+(?: [A-Z][a-z]+)*(?:\s+(?:Inc|Corp|Ltd|LLC|Co)\.?)?/g,
      
      // Stock symbols
      stock_symbol: /\b[A-Z]{2,5}\b/g,
      
      // Currency amounts
      currency: /\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d+)?\s*(?:million|billion|trillion)/gi,
      
      // Percentages
      percentage: /\d+(?:\.\d+)?%/g,
      
      // Dates
      date: /\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}|\d{1,2}\/\d{1,2}\/\d{4}|\d{4}-\d{2}-\d{2}/gi,
      
      // Financial metrics
      financial_metric: /(?:P\/E|PE|EPS|ROE|ROI|EBITDA|NAV|AUM)\s*(?:ratio|value)?/gi
    };
    
    logger.info('🏷️ Entity extraction initialized');
  }

  async initializeDocumentClassification() {
    // Document type classifiers
    this.documentClassifiers = {
      // Earnings-related documents
      earnings: {
        keywords: ['earnings', 'quarterly', 'revenue', 'profit', 'eps', 'guidance'],
        weight: 1.0
      },
      
      // Analyst reports
      analyst_report: {
        keywords: ['rating', 'target price', 'recommendation', 'upgrade', 'downgrade'],
        weight: 0.9
      },
      
      // News articles
      news: {
        keywords: ['announced', 'reported', 'according to', 'sources', 'breaking'],
        weight: 0.7
      },
      
      // Regulatory filings
      regulatory: {
        keywords: ['sec filing', '10-k', '10-q', '8-k', 'proxy', 'registration'],
        weight: 0.8
      },
      
      // Market commentary
      market_commentary: {
        keywords: ['market', 'outlook', 'forecast', 'prediction', 'trend'],
        weight: 0.6
      }
    };
    
    logger.info('📄 Document classification initialized');
  }

  async analyzeSentiment(text, context = {}) {
    try {
      const startTime = Date.now();
      
      // Check cache first
      const cacheKey = `sentiment_${this.hashText(text)}`;
      if (this.config.cacheEnabled && this.processingCache.has(cacheKey)) {
        this.metrics.cacheHitRate = (this.metrics.cacheHitRate + 1) / 2;
        return this.processingCache.get(cacheKey);
      }
      
      // Preprocess text
      const cleanText = this.preprocessText(text);
      const tokens = this.tokenize(cleanText);
      
      // Calculate sentiment scores
      const lexiconScore = this.sentimentAnalyzer.lexiconScore(tokens);
      const contextualScore = this.sentimentAnalyzer.contextualScore(cleanText, context);
      
      // Combine scores
      const finalScore = (lexiconScore * 0.6) + (contextualScore * 0.4);
      
      // Determine sentiment label
      let label, confidence;
      if (finalScore > this.config.sentimentThreshold) {
        label = 'positive';
        confidence = Math.min(0.95, Math.abs(finalScore));
      } else if (finalScore < -this.config.sentimentThreshold) {
        label = 'negative';
        confidence = Math.min(0.95, Math.abs(finalScore));
      } else {
        label = 'neutral';
        confidence = 1 - Math.abs(finalScore);
      }
      
      const result = {
        sentiment: label,
        score: finalScore,
        confidence,
        breakdown: {
          lexiconScore,
          contextualScore,
          financialTermsFound: tokens.filter(t => this.financialTerms.has(t.toLowerCase())).length
        },
        processingTime: Date.now() - startTime
      };
      
      // Cache result
      if (this.config.cacheEnabled) {
        this.processingCache.set(cacheKey, result);
      }
      
      return result;
      
    } catch (error) {
      logger.error('❌ Sentiment analysis failed:', error);
      throw error;
    }
  }

  async extractEntities(text) {
    try {
      const entities = {};
      
      for (const [entityType, pattern] of Object.entries(this.entityPatterns)) {
        const matches = text.match(pattern) || [];
        
        entities[entityType] = matches.map(match => ({
          text: match,
          confidence: this.calculateEntityConfidence(match, entityType),
          position: text.indexOf(match)
        })).filter(entity => entity.confidence >= this.config.entityConfidenceThreshold);
      }
      
      // Extract financial terms
      const tokens = this.tokenize(text.toLowerCase());
      entities.financial_terms = tokens
        .filter(token => this.financialTerms.has(token))
        .map(token => ({
          text: token,
          category: this.financialTerms.get(token).category,
          sentiment: this.financialTerms.get(token).sentiment,
          confidence: 0.9
        }));
      
      return entities;
      
    } catch (error) {
      logger.error('❌ Entity extraction failed:', error);
      throw error;
    }
  }

  async classifyDocument(text) {
    try {
      const tokens = this.tokenize(text.toLowerCase());
      const scores = {};
      
      for (const [docType, classifier] of Object.entries(this.documentClassifiers)) {
        const keywordMatches = classifier.keywords.filter(keyword => 
          tokens.some(token => token.includes(keyword.toLowerCase()))
        ).length;
        
        scores[docType] = (keywordMatches / classifier.keywords.length) * classifier.weight;
      }
      
      // Find best classification
      const bestType = Object.keys(scores).reduce((a, b) => scores[a] > scores[b] ? a : b);
      const confidence = scores[bestType];
      
      return {
        documentType: bestType,
        confidence,
        allScores: scores
      };
      
    } catch (error) {
      logger.error('❌ Document classification failed:', error);
      throw error;
    }
  }

  async generateMarketInsights(text, context = {}) {
    try {
      // Extract key information
      const sentiment = await this.analyzeSentiment(text, context);
      const entities = await this.extractEntities(text);
      const classification = await this.classifyDocument(text);
      
      // Generate insights based on analysis
      const insights = {
        marketSentiment: sentiment.sentiment,
        sentimentStrength: Math.abs(sentiment.score),
        keyEntities: {
          companies: entities.company || [],
          stockSymbols: entities.stock_symbol || [],
          financialMetrics: entities.financial_metric || [],
          currencyAmounts: entities.currency || []
        },
        documentRelevance: classification.confidence,
        riskIndicators: await this.assessRiskFromText(text),
        actionableInsights: this.generateActionableInsights(sentiment, entities, classification)
      };
      
      return insights;
      
    } catch (error) {
      logger.error('❌ Market insights generation failed:', error);
      throw error;
    }
  }

  async assessRiskFromText(text) {
    try {
      const tokens = this.tokenize(text.toLowerCase());
      const riskKeywords = tokens.filter(token => this.riskKeywords.has(token));
      
      const riskScore = Math.min(1.0, riskKeywords.length / 10); // Normalize to 0-1
      
      let riskLevel;
      if (riskScore > 0.7) {
        riskLevel = 'high';
      } else if (riskScore > 0.3) {
        riskLevel = 'medium';
      } else {
        riskLevel = 'low';
      }
      
      return {
        riskLevel,
        riskScore,
        riskKeywords,
        riskFactors: this.identifyRiskFactors(tokens)
      };
      
    } catch (error) {
      logger.error('❌ Risk assessment failed:', error);
      throw error;
    }
  }

  generateActionableInsights(sentiment, entities, classification) {
    const insights = [];
    
    // Sentiment-based insights
    if (sentiment.sentiment === 'positive' && sentiment.confidence > 0.8) {
      insights.push({
        type: 'opportunity',
        message: 'Strong positive sentiment detected - potential buying opportunity',
        confidence: sentiment.confidence
      });
    } else if (sentiment.sentiment === 'negative' && sentiment.confidence > 0.8) {
      insights.push({
        type: 'warning',
        message: 'Strong negative sentiment detected - consider risk management',
        confidence: sentiment.confidence
      });
    }
    
    // Entity-based insights
    if (entities.financial_terms && entities.financial_terms.length > 5) {
      insights.push({
        type: 'analysis',
        message: 'High financial content density - document likely contains important information',
        confidence: 0.8
      });
    }
    
    // Classification-based insights
    if (classification.documentType === 'earnings' && classification.confidence > 0.7) {
      insights.push({
        type: 'timing',
        message: 'Earnings-related content - monitor for immediate market impact',
        confidence: classification.confidence
      });
    }
    
    return insights;
  }

  identifyRiskFactors(tokens) {
    const riskFactors = [];
    
    // Market risk indicators
    const marketRiskTerms = ['volatility', 'uncertainty', 'instability'];
    if (tokens.some(token => marketRiskTerms.includes(token))) {
      riskFactors.push({ type: 'market_risk', severity: 'medium' });
    }
    
    // Credit risk indicators
    const creditRiskTerms = ['default', 'bankruptcy', 'debt'];
    if (tokens.some(token => creditRiskTerms.includes(token))) {
      riskFactors.push({ type: 'credit_risk', severity: 'high' });
    }
    
    // Liquidity risk indicators
    const liquidityRiskTerms = ['liquidity', 'cash flow', 'funding'];
    if (tokens.some(token => liquidityRiskTerms.includes(token))) {
      riskFactors.push({ type: 'liquidity_risk', severity: 'medium' });
    }
    
    return riskFactors;
  }

  preprocessText(text) {
    return text
      .toLowerCase()
      .replace(/[^\w\s\-\.%$]/g, ' ') // Keep financial symbols
      .replace(/\s+/g, ' ')
      .trim();
  }

  tokenize(text) {
    return text.split(/\s+/).filter(token => token.length > 1);
  }

  calculateEntityConfidence(match, entityType) {
    // Simple confidence calculation based on entity type and match characteristics
    switch (entityType) {
      case 'stock_symbol':
        return match.length >= 2 && match.length <= 5 ? 0.9 : 0.6;
      case 'currency':
        return 0.95; // Currency patterns are usually very reliable
      case 'percentage':
        return 0.9;
      case 'company':
        return match.length > 3 ? 0.8 : 0.6;
      default:
        return 0.7;
    }
  }

  hashText(text) {
    // Simple hash function for caching
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
  }

  async processDocument(document, context = {}) {
    try {
      this.metrics.documentsProcessed++;
      const startTime = Date.now();
      
      // Comprehensive document analysis
      const analysis = {
        sentiment: await this.analyzeSentiment(document, context),
        entities: await this.extractEntities(document),
        classification: await this.classifyDocument(document),
        insights: await this.generateMarketInsights(document, context),
        riskAssessment: await this.assessRiskFromText(document)
      };
      
      const processingTime = Date.now() - startTime;
      this.metrics.averageProcessingTime = 
        (this.metrics.averageProcessingTime + processingTime) / 2;
      
      analysis.metadata = {
        processingTime,
        documentLength: document.length,
        timestamp: new Date()
      };
      
      this.emit('documentProcessed', analysis);
      
      return analysis;
      
    } catch (error) {
      logger.error('❌ Document processing failed:', error);
      throw error;
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      cacheSize: this.processingCache.size,
      financialTermsCount: this.financialTerms.size,
      isInitialized: this.isInitialized
    };
  }

  clearCache() {
    this.processingCache.clear();
    logger.info('🧹 NLP processing cache cleared');
  }
}

module.exports = { AdvancedNLPSystem };
