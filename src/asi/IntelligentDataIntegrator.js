/**
 * 🔗 INTELLIGENT DATA INTEGRATION ENGINE
 * 
 * Seamlessly integrates crawled AMC data with ASI prediction system
 * Real-time data fusion, validation, and enrichment
 * Automated data pipeline from crawling to prediction
 * 
 * @author 35-year ASI Engineer
 * @version 2.0.0 - Production-Ready Data Integration
 */

const logger = require('../utils/logger');
const { AutomatedDataCrawler } = require('./AutomatedDataCrawler');
const { DocumentIntelligenceAnalyzer } = require('./DocumentIntelligenceAnalyzer');

class IntelligentDataIntegrator {
  constructor(options = {}) {
    this.config = {
      dataRefreshInterval: options.dataRefreshInterval || 6 * 60 * 60 * 1000, // 6 hours
      validationThreshold: options.validationThreshold || 0.8,
      batchSize: options.batchSize || 50,
      maxRetries: options.maxRetries || 3,
      ...options
    };

    // Core components
    this.dataCrawler = null;
    this.documentAnalyzer = null;
    
    // Data management
    this.integratedDatabase = new Map();
    this.schemeRegistry = new Map();
    this.dataValidationRules = new Map();
    this.enrichmentRules = new Map();
    
    // Performance tracking
    this.integrationMetrics = {
      schemesIntegrated: 0,
      documentsProcessed: 0,
      dataValidationsPassed: 0,
      enrichmentsApplied: 0,
      integrationTime: 0,
      errorCount: 0
    };
  }

  async initialize() {
    try {
      logger.info('🔗 Initializing Intelligent Data Integrator...');
      
      await this.initializeComponents();
      await this.initializeDataValidation();
      await this.initializeDataEnrichment();
      this.startAutomatedIntegration();
      
      logger.info('✅ Intelligent Data Integrator initialized successfully');
      
    } catch (error) {
      logger.error('❌ Data Integrator initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize core components
   */
  async initializeComponents() {
    logger.info('🔧 Initializing core components...');
    
    // Initialize data crawler
    this.dataCrawler = new AutomatedDataCrawler({
      maxConcurrentPages: 3,
      requestDelay: 3000,
      updateFrequency: this.config.dataRefreshInterval
    });
    await this.dataCrawler.initialize();
    
    // Initialize document analyzer
    this.documentAnalyzer = new DocumentIntelligenceAnalyzer({
      confidenceThreshold: 0.7,
      enableNLP: true
    });
    await this.documentAnalyzer.initialize();
    
    logger.info('✅ Core components initialized');
  }

  /**
   * Initialize data validation system
   */
  async initializeDataValidation() {
    logger.info('✅ Initializing data validation...');
    
    this.dataValidationRules = new Map([
      ['scheme_name', {
        required: true,
        type: 'string',
        minLength: 5,
        validator: this.validateSchemeName.bind(this)
      }],
      ['scheme_code', {
        required: true,
        type: 'string',
        pattern: /^[A-Z0-9]{3,20}$/,
        validator: this.validateSchemeCode.bind(this)
      }],
      ['nav', {
        required: true,
        type: 'number',
        min: 1,
        max: 10000,
        validator: this.validateNAV.bind(this)
      }],
      ['portfolio_holdings', {
        required: false,
        type: 'array',
        validator: this.validatePortfolioHoldings.bind(this)
      }]
    ]);
    
    logger.info('✅ Data validation initialized');
  }

  /**
   * Initialize data enrichment system
   */
  async initializeDataEnrichment() {
    logger.info('🔍 Initializing data enrichment...');
    
    this.enrichmentRules = new Map([
      ['scheme_category', {
        enricher: this.enrichSchemeCategory.bind(this),
        priority: 1,
        dependencies: ['scheme_name', 'investment_objective']
      }],
      ['risk_profile', {
        enricher: this.enrichRiskProfile.bind(this),
        priority: 2,
        dependencies: ['risk_factors', 'scheme_category']
      }],
      ['investment_suitability', {
        enricher: this.enrichInvestmentSuitability.bind(this),
        priority: 3,
        dependencies: ['risk_profile', 'minimum_investment']
      }]
    ]);
    
    logger.info('✅ Data enrichment initialized');
  }

  /**
   * Start comprehensive data integration process
   */
  async startComprehensiveIntegration() {
    try {
      logger.info('🚀 Starting comprehensive data integration...');
      const integrationStart = Date.now();
      
      // Stage 1: Data Collection
      const collectedData = await this.crawlAMCData();
      
      // Stage 2: Document Processing
      const processedDocuments = await this.processDocuments(collectedData);
      
      // Stage 3: Data Extraction
      const extractedData = await this.extractStructuredData(processedDocuments);
      
      // Stage 4: Data Validation
      const validatedData = await this.validateData(extractedData);
      
      // Stage 5: Data Enrichment
      const enrichedData = await this.enrichData(validatedData);
      
      // Stage 6: Data Integration
      const integratedData = await this.mergeData(enrichedData);
      
      this.integrationMetrics.integrationTime = Date.now() - integrationStart;
      this.integrationMetrics.schemesIntegrated = integratedData.length;
      
      logger.info(`✅ Integration completed in ${this.integrationMetrics.integrationTime}ms`);
      
      return {
        success: true,
        integratedSchemes: integratedData.length,
        integrationTime: this.integrationMetrics.integrationTime
      };
      
    } catch (error) {
      logger.error('❌ Comprehensive integration failed:', error);
      this.integrationMetrics.errorCount++;
      throw error;
    }
  }

  /**
   * Crawl AMC data
   */
  async crawlAMCData() {
    logger.info('🕷️ Crawling AMC websites...');
    
    const crawlResult = await this.dataCrawler.startComprehensiveCrawl();
    logger.info(`📊 Crawled ${crawlResult.totalSchemes} schemes`);
    
    return crawlResult;
  }

  /**
   * Process documents using AI analyzer
   */
  async processDocuments(crawlData) {
    logger.info('📄 Processing documents with AI analyzer...');
    
    const processedDocuments = [];
    const schemes = Object.values(crawlData.schemes || {});
    
    for (const scheme of schemes) {
      try {
        const schemeDocuments = [];
        
        for (const document of scheme.documents || []) {
          const analysis = await this.documentAnalyzer.analyzeDocument(document, document.type);
          schemeDocuments.push({ document, analysis });
        }
        
        processedDocuments.push({
          scheme: scheme,
          processedDocuments: schemeDocuments
        });
        
      } catch (error) {
        logger.warn(`⚠️ Failed to process documents for scheme ${scheme.schemeName}:`, error);
      }
    }
    
    this.integrationMetrics.documentsProcessed = processedDocuments.reduce(
      (total, scheme) => total + scheme.processedDocuments.length, 0
    );
    
    return processedDocuments;
  }

  /**
   * Extract structured data from processed documents
   */
  async extractStructuredData(processedDocuments) {
    logger.info('🔍 Extracting structured data...');
    
    const structuredData = [];
    
    for (const schemeData of processedDocuments) {
      try {
        const extractedScheme = await this.extractSchemeData(schemeData);
        if (extractedScheme) {
          structuredData.push(extractedScheme);
        }
      } catch (error) {
        logger.warn(`⚠️ Failed to extract data for scheme:`, error);
      }
    }
    
    return structuredData;
  }

  /**
   * Extract comprehensive scheme data
   */
  async extractSchemeData(schemeData) {
    const scheme = schemeData.scheme;
    const documents = schemeData.processedDocuments;
    
    const combinedData = {
      schemeName: scheme.schemeName,
      schemeCode: scheme.schemeCode,
      isin: scheme.isin,
      nav: parseFloat(scheme.nav) || null,
      aum: this.parseAUM(scheme.aum),
      expenseRatio: parseFloat(scheme.expenseRatio) || null,
      investmentObjective: '',
      investmentStrategy: '',
      riskFactors: [],
      portfolioHoldings: [],
      performanceData: {},
      sourceUrl: scheme.sourceUrl,
      lastUpdated: scheme.lastUpdated
    };
    
    // Extract data from documents
    for (const docData of documents) {
      const analysis = docData.analysis;
      
      if (analysis.insights) {
        combinedData.investmentObjective = analysis.insights.investmentObjective || combinedData.investmentObjective;
        combinedData.investmentStrategy = analysis.insights.investmentStrategy || combinedData.investmentStrategy;
      }
      
      if (analysis.riskAnalysis) {
        combinedData.riskFactors.push(...(analysis.riskAnalysis.identifiedRisks || []));
      }
      
      if (analysis.performanceAnalysis && analysis.performanceAnalysis.applicable) {
        Object.assign(combinedData.performanceData, analysis.performanceAnalysis.returns || {});
      }
    }
    
    combinedData.dataQuality = this.calculateSchemeDataQuality(combinedData);
    
    return combinedData;
  }

  /**
   * Validate extracted data
   */
  async validateData(extractedData) {
    logger.info('✅ Validating extracted data...');
    
    const validatedData = [];
    
    for (const scheme of extractedData) {
      try {
        const validationResult = await this.validateScheme(scheme);
        
        if (validationResult.isValid) {
          validatedData.push({ ...scheme, validation: validationResult });
          this.integrationMetrics.dataValidationsPassed++;
        } else {
          logger.warn(`⚠️ Scheme validation failed: ${scheme.schemeName}`);
        }
        
      } catch (error) {
        logger.error(`❌ Validation error for scheme ${scheme.schemeName}:`, error);
      }
    }
    
    return validatedData;
  }

  /**
   * Validate individual scheme
   */
  async validateScheme(scheme) {
    const errors = [];
    
    for (const [field, rule] of this.dataValidationRules) {
      const value = scheme[field];
      
      if (rule.required && (value === undefined || value === null || value === '')) {
        errors.push(`Required field '${field}' is missing`);
        continue;
      }
      
      if (rule.validator && value !== undefined && value !== null) {
        const validationResult = await rule.validator(value, scheme);
        if (!validationResult.isValid) {
          errors.push(validationResult.message || `Invalid ${field}`);
        }
      }
    }
    
    return {
      isValid: errors.length === 0,
      errors: errors,
      score: Math.max(0, 1 - (errors.length * 0.2))
    };
  }

  /**
   * Enrich validated data
   */
  async enrichData(validatedData) {
    logger.info('🔍 Enriching validated data...');
    
    const enrichedData = [];
    
    for (const scheme of validatedData) {
      try {
        const enrichedScheme = await this.enrichScheme(scheme);
        enrichedData.push(enrichedScheme);
        this.integrationMetrics.enrichmentsApplied++;
      } catch (error) {
        logger.warn(`⚠️ Enrichment failed for scheme ${scheme.schemeName}:`, error);
        enrichedData.push(scheme);
      }
    }
    
    return enrichedData;
  }

  /**
   * Enrich individual scheme
   */
  async enrichScheme(scheme) {
    const enrichedScheme = { ...scheme };
    
    const sortedRules = Array.from(this.enrichmentRules.entries())
      .sort((a, b) => a[1].priority - b[1].priority);
    
    for (const [enrichmentType, rule] of sortedRules) {
      try {
        const dependenciesMet = rule.dependencies.every(dep => 
          enrichedScheme[dep] !== undefined && enrichedScheme[dep] !== null
        );
        
        if (dependenciesMet) {
          const enrichmentResult = await rule.enricher(enrichedScheme);
          enrichedScheme[enrichmentType] = enrichmentResult;
        }
        
      } catch (error) {
        logger.warn(`⚠️ Enrichment '${enrichmentType}' failed:`, error);
      }
    }
    
    return enrichedScheme;
  }

  /**
   * Merge enriched data into integrated database
   */
  async mergeData(enrichedData) {
    logger.info('🔗 Merging data into integrated database...');
    
    const mergedData = [];
    
    for (const scheme of enrichedData) {
      try {
        const existingScheme = this.integratedDatabase.get(scheme.schemeCode);
        
        if (existingScheme) {
          const mergedScheme = { ...existingScheme, ...scheme, lastUpdated: new Date().toISOString() };
          this.integratedDatabase.set(scheme.schemeCode, mergedScheme);
          mergedData.push(mergedScheme);
        } else {
          this.integratedDatabase.set(scheme.schemeCode, scheme);
          mergedData.push(scheme);
        }
        
        this.schemeRegistry.set(scheme.schemeCode, {
          schemeName: scheme.schemeName,
          lastUpdated: new Date().toISOString(),
          dataQuality: scheme.dataQuality
        });
        
      } catch (error) {
        logger.error(`❌ Failed to merge scheme ${scheme.schemeName}:`, error);
      }
    }
    
    return mergedData;
  }

  /**
   * Get integrated scheme data for prediction
   */
  async getSchemeDataForPrediction(schemeCode) {
    const scheme = this.integratedDatabase.get(schemeCode);
    
    if (!scheme) {
      throw new Error(`Scheme ${schemeCode} not found in integrated database`);
    }
    
    return {
      fundName: scheme.schemeName,
      fundCode: scheme.schemeCode,
      currentNAV: scheme.nav,
      holdings: this.formatHoldingsForPrediction(scheme.portfolioHoldings),
      marketData: { prices: [], volumes: [], returns: [] },
      fundamentalData: {
        expenseRatio: scheme.expenseRatio,
        aum: scheme.aum,
        category: scheme.scheme_category,
        riskProfile: scheme.risk_profile
      },
      documentInsights: {
        investmentObjective: scheme.investmentObjective,
        investmentStrategy: scheme.investmentStrategy,
        riskFactors: scheme.riskFactors
      }
    };
  }

  /**
   * Start automated integration cycle
   */
  startAutomatedIntegration() {
    setInterval(async () => {
      try {
        logger.info('🔄 Starting automated integration cycle...');
        await this.startComprehensiveIntegration();
        logger.info('✅ Automated integration cycle completed');
      } catch (error) {
        logger.error('❌ Automated integration cycle failed:', error);
      }
    }, this.config.dataRefreshInterval);
    
    logger.info('⏰ Automated integration cycle scheduled');
  }

  // Helper methods
  parseAUM(aumString) {
    if (!aumString) return null;
    
    const match = aumString.match(/(\d+(?:,\d+)*(?:\.\d+)?)\s*(crore|cr|lakh)/i);
    if (match) {
      const value = parseFloat(match[1].replace(/,/g, ''));
      const unit = match[2].toLowerCase();
      
      if (unit.includes('crore') || unit.includes('cr')) {
        return value * 10000000;
      } else if (unit.includes('lakh')) {
        return value * 100000;
      }
    }
    
    return null;
  }

  calculateSchemeDataQuality(scheme) {
    let score = 0;
    const fields = ['schemeName', 'schemeCode', 'nav', 'investmentObjective', 'investmentStrategy'];
    
    fields.forEach(field => {
      if (scheme[field] && scheme[field] !== '') {
        score += 0.2;
      }
    });
    
    return score;
  }

  formatHoldingsForPrediction(holdings) {
    const formattedHoldings = {};
    
    holdings.forEach(holding => {
      if (holding.name && holding.percentage) {
        const symbol = holding.name.toUpperCase().replace(/\s+/g, '_');
        formattedHoldings[symbol] = holding.percentage / 100;
      }
    });
    
    return formattedHoldings;
  }

  getMetrics() {
    return {
      integration: this.integrationMetrics,
      database: {
        totalSchemes: this.integratedDatabase.size,
        registrySize: this.schemeRegistry.size
      }
    };
  }

  // Validation methods
  async validateSchemeName(value) { 
    return { isValid: value && value.length >= 5, message: 'Scheme name too short' }; 
  }
  
  async validateSchemeCode(value) { 
    return { isValid: /^[A-Z0-9]{3,20}$/.test(value), message: 'Invalid scheme code format' }; 
  }
  
  async validateNAV(value) { 
    return { isValid: value > 0 && value < 10000, message: 'NAV out of valid range' }; 
  }
  
  async validatePortfolioHoldings(value) { 
    return { isValid: Array.isArray(value), message: 'Portfolio holdings must be an array' }; 
  }

  // Enrichment methods
  async enrichSchemeCategory(scheme) {
    const name = scheme.schemeName.toLowerCase();
    if (name.includes('large cap')) return 'Large Cap';
    if (name.includes('mid cap')) return 'Mid Cap';
    if (name.includes('small cap')) return 'Small Cap';
    if (name.includes('multi cap')) return 'Multi Cap';
    if (name.includes('flexi cap')) return 'Flexi Cap';
    return 'Other';
  }
  
  async enrichRiskProfile(scheme) {
    const category = scheme.scheme_category;
    if (category === 'Large Cap') return 'Low to Moderate';
    if (category === 'Mid Cap') return 'Moderate to High';
    if (category === 'Small Cap') return 'High';
    return 'Moderate';
  }
  
  async enrichInvestmentSuitability(scheme) {
    const risk = scheme.risk_profile;
    if (risk === 'Low to Moderate') return 'Conservative to Moderate investors';
    if (risk === 'Moderate to High') return 'Moderate to Aggressive investors';
    if (risk === 'High') return 'Aggressive investors';
    return 'All types of investors';
  }
}

module.exports = { IntelligentDataIntegrator };
