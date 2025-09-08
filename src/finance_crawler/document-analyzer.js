/**
 * 🔬 AMC DOCUMENT ANALYZER
 * 
 * Advanced analysis of PDF and Excel files from AMCs
 * Extracts financial data, trends, and insights for ASI analysis
 * 
 * @author Financial Document Intelligence Team
 * @version 1.0.0 - Document Intelligence
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');
const XLSX = require('xlsx');
const pdf = require('pdf-parse');
const logger = require('../utils/logger');

class AMCDocumentAnalyzer extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      // Analysis settings
      enablePDFAnalysis: options.enablePDFAnalysis !== false,
      enableExcelAnalysis: options.enableExcelAnalysis !== false,
      enableContentExtraction: options.enableContentExtraction !== false,
      
      // Data extraction patterns
      extractFinancialMetrics: options.extractFinancialMetrics !== false,
      extractPerformanceData: options.extractPerformanceData !== false,
      extractPortfolioData: options.extractPortfolioData !== false,
      
      // Output settings
      outputPath: options.outputPath || './data/analysis-results',
      
      ...options
    };
    
    // Financial data patterns
    this.financialPatterns = {
      aum: /(?:aum|assets under management|total aum)[:\s]*[₹$]?\s*([\d,]+\.?\d*)\s*(crore|cr|billion|bn|million|mn)?/gi,
      nav: /(?:nav|net asset value)[:\s]*[₹$]?\s*([\d,]+\.?\d*)/gi,
      expense_ratio: /(?:expense ratio|ter)[:\s]*([\d.]+)%?/gi,
      returns: /(?:returns?|performance)[:\s]*(?:1y|1yr|1 year)[:\s]*([-]?[\d.]+)%/gi,
      portfolio_turnover: /(?:portfolio turnover)[:\s]*([\d.]+)%?/gi,
      sharpe_ratio: /(?:sharpe ratio)[:\s]*([\d.]+)/gi,
      alpha: /(?:alpha)[:\s]*([-]?[\d.]+)/gi,
      beta: /(?:beta)[:\s]*([\d.]+)/gi
    };
    
    // Performance metrics patterns
    this.performancePatterns = {
      ytd: /(?:ytd|year to date)[:\s]*([-]?[\d.]+)%/gi,
      '1y': /(?:1y|1yr|1 year)[:\s]*([-]?[\d.]+)%/gi,
      '3y': /(?:3y|3yr|3 year)[:\s]*([-]?[\d.]+)%/gi,
      '5y': /(?:5y|5yr|5 year)[:\s]*([-]?[\d.]+)%/gi,
      inception: /(?:since inception|inception)[:\s]*([-]?[\d.]+)%/gi
    };
    
    // Portfolio holding patterns
    this.portfolioPatterns = {
      top_holdings: /(?:top holdings|major holdings|largest holdings)/gi,
      sector_allocation: /(?:sector allocation|sectoral allocation)/gi,
      market_cap: /(?:market cap|market capitalisation)/gi,
      cash_position: /(?:cash|cash position|liquid funds)[:\s]*([\d.]+)%/gi
    };
    
    // Analysis results storage
    this.analysisResults = new Map();
    this.extractedData = new Map();
    
    // Statistics
    this.stats = {
      documentsAnalyzed: 0,
      pdfAnalyzed: 0,
      excelAnalyzed: 0,
      dataPointsExtracted: 0,
      analysisErrors: 0
    };
  }

  async initialize() {
    try {
      logger.info('🔬 Initializing AMC Document Analyzer...');
      
      // Create output directories
      await this.createOutputDirectories();
      
      // Load existing analysis results
      await this.loadAnalysisResults();
      
      logger.info('✅ AMC Document Analyzer initialized successfully');
      
    } catch (error) {
      logger.error('❌ AMC Document Analyzer initialization failed:', error);
      throw error;
    }
  }

  async createOutputDirectories() {
    const dirs = [
      this.config.outputPath,
      path.join(this.config.outputPath, 'extracted-data'),
      path.join(this.config.outputPath, 'analysis-reports'),
      path.join(this.config.outputPath, 'trends'),
      path.join(this.config.outputPath, 'insights')
    ];
    
    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') {
          throw error;
        }
      }
    }
  }

  async loadAnalysisResults() {
    try {
      const resultsPath = path.join(this.config.outputPath, 'analysis-results.json');
      const data = await fs.readFile(resultsPath, 'utf8');
      const results = JSON.parse(data);
      
      this.analysisResults = new Map(Object.entries(results.analysis || {}));
      this.extractedData = new Map(Object.entries(results.data || {}));
      
      logger.info(`📊 Loaded ${this.analysisResults.size} analysis results`);
      
    } catch (error) {
      logger.info('📊 Creating new analysis results storage');
    }
  }

  async saveAnalysisResults() {
    try {
      const resultsPath = path.join(this.config.outputPath, 'analysis-results.json');
      const results = {
        analysis: Object.fromEntries(this.analysisResults),
        data: Object.fromEntries(this.extractedData),
        stats: this.stats,
        lastUpdated: new Date().toISOString()
      };
      
      await fs.writeFile(resultsPath, JSON.stringify(results, null, 2));
      
    } catch (error) {
      logger.error('❌ Failed to save analysis results:', error);
    }
  }

  async analyzeDocument(filePath, documentInfo) {
    try {
      logger.debug(`🔍 Analyzing document: ${path.basename(filePath)}`);
      
      const fileExtension = path.extname(filePath).toLowerCase();
      let analysisResult = null;
      
      switch (fileExtension) {
        case '.pdf':
          if (this.config.enablePDFAnalysis) {
            analysisResult = await this.analyzePDF(filePath, documentInfo);
            this.stats.pdfAnalyzed++;
          }
          break;
          
        case '.xlsx':
        case '.xls':
          if (this.config.enableExcelAnalysis) {
            analysisResult = await this.analyzeExcel(filePath, documentInfo);
            this.stats.excelAnalyzed++;
          }
          break;
          
        case '.csv':
          analysisResult = await this.analyzeCSV(filePath, documentInfo);
          break;
          
        default:
          logger.warn(`⚠️ Unsupported file format: ${fileExtension}`);
          return null;
      }
      
      if (analysisResult) {
        // Store analysis result
        const documentKey = this.generateDocumentKey(filePath, documentInfo);
        this.analysisResults.set(documentKey, analysisResult);
        
        // Extract and store data points
        const extractedData = this.extractDataPoints(analysisResult);
        this.extractedData.set(documentKey, extractedData);
        
        this.stats.documentsAnalyzed++;
        this.stats.dataPointsExtracted += extractedData.dataPoints?.length || 0;
        
        // Emit analysis event
        this.emit('documentAnalyzed', {
          filePath,
          documentInfo,
          analysisResult,
          extractedData
        });
        
        return analysisResult;
      }
      
      return null;
      
    } catch (error) {
      logger.error(`❌ Document analysis failed for ${filePath}:`, error);
      this.stats.analysisErrors++;
      return null;
    }
  }

  async analyzePDF(filePath, documentInfo) {
    try {
      const dataBuffer = await fs.readFile(filePath);
      const pdfData = await pdf(dataBuffer);
      
      const analysis = {
        type: 'pdf',
        filePath,
        documentInfo,
        totalPages: pdfData.numpages,
        text: pdfData.text,
        metadata: pdfData.metadata,
        analyzedAt: new Date().toISOString()
      };
      
      // Extract financial metrics
      if (this.config.extractFinancialMetrics) {
        analysis.financialMetrics = this.extractFinancialMetrics(pdfData.text);
      }
      
      // Extract performance data
      if (this.config.extractPerformanceData) {
        analysis.performanceData = this.extractPerformanceData(pdfData.text);
      }
      
      // Extract portfolio data
      if (this.config.extractPortfolioData) {
        analysis.portfolioData = this.extractPortfolioData(pdfData.text);
      }
      
      // Analyze document structure
      analysis.structure = this.analyzePDFStructure(pdfData.text);
      
      return analysis;
      
    } catch (error) {
      throw new Error(`PDF analysis failed: ${error.message}`);
    }
  }

  async analyzeExcel(filePath, documentInfo) {
    try {
      const workbook = XLSX.readFile(filePath);
      const sheetNames = workbook.SheetNames;
      
      const analysis = {
        type: 'excel',
        filePath,
        documentInfo,
        sheetNames,
        sheets: {},
        analyzedAt: new Date().toISOString()
      };
      
      // Analyze each sheet
      for (const sheetName of sheetNames) {
        const worksheet = workbook.Sheets[sheetName];
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { header: 1 });
        
        analysis.sheets[sheetName] = {
          data: jsonData,
          rowCount: jsonData.length,
          columnCount: jsonData[0]?.length || 0,
          analysis: this.analyzeExcelSheet(jsonData, sheetName)
        };
      }
      
      // Extract consolidated financial data
      analysis.consolidatedData = this.consolidateExcelData(analysis.sheets);
      
      return analysis;
      
    } catch (error) {
      throw new Error(`Excel analysis failed: ${error.message}`);
    }
  }

  async analyzeCSV(filePath, documentInfo) {
    try {
      const csvContent = await fs.readFile(filePath, 'utf8');
      const lines = csvContent.split('\n');
      const data = lines.map(line => line.split(','));
      
      const analysis = {
        type: 'csv',
        filePath,
        documentInfo,
        rowCount: data.length,
        columnCount: data[0]?.length || 0,
        data: data.slice(0, 100), // Limit data for storage
        analyzedAt: new Date().toISOString()
      };
      
      // Analyze CSV structure and content
      analysis.structure = this.analyzeCSVStructure(data);
      analysis.financialData = this.extractCSVFinancialData(data);
      
      return analysis;
      
    } catch (error) {
      throw new Error(`CSV analysis failed: ${error.message}`);
    }
  }

  extractFinancialMetrics(text) {
    const metrics = {};
    
    for (const [metric, pattern] of Object.entries(this.financialPatterns)) {
      const matches = [...text.matchAll(pattern)];
      if (matches.length > 0) {
        metrics[metric] = matches.map(match => ({
          value: match[1],
          unit: match[2] || '',
          context: match[0]
        }));
      }
    }
    
    return metrics;
  }

  extractPerformanceData(text) {
    const performance = {};
    
    for (const [period, pattern] of Object.entries(this.performancePatterns)) {
      const matches = [...text.matchAll(pattern)];
      if (matches.length > 0) {
        performance[period] = matches.map(match => ({
          value: parseFloat(match[1]),
          context: match[0]
        }));
      }
    }
    
    return performance;
  }

  extractPortfolioData(text) {
    const portfolio = {};
    
    // Extract cash position
    const cashMatches = [...text.matchAll(this.portfolioPatterns.cash_position)];
    if (cashMatches.length > 0) {
      portfolio.cashPosition = cashMatches.map(match => ({
        percentage: parseFloat(match[1]),
        context: match[0]
      }));
    }
    
    // Extract sector information
    if (this.portfolioPatterns.sector_allocation.test(text)) {
      portfolio.hasSectorAllocation = true;
    }
    
    if (this.portfolioPatterns.top_holdings.test(text)) {
      portfolio.hasTopHoldings = true;
    }
    
    return portfolio;
  }

  analyzePDFStructure(text) {
    const structure = {
      hasTable: /\||\t/.test(text),
      hasNumbers: /\d+/.test(text),
      hasPercentages: /%/.test(text),
      hasCurrency: /[₹$]/.test(text),
      wordCount: text.split(/\s+/).length,
      lineCount: text.split('\n').length
    };
    
    // Identify document sections
    structure.sections = this.identifyDocumentSections(text);
    
    return structure;
  }

  identifyDocumentSections(text) {
    const sections = [];
    const lines = text.split('\n');
    
    const sectionPatterns = {
      'fund_overview': /fund overview|scheme overview|about the fund/i,
      'performance': /performance|returns|fund performance/i,
      'portfolio': /portfolio|holdings|top holdings/i,
      'risk_metrics': /risk|volatility|standard deviation/i,
      'expense': /expense|fees|charges/i,
      'benchmark': /benchmark|comparison/i
    };
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      
      for (const [sectionType, pattern] of Object.entries(sectionPatterns)) {
        if (pattern.test(line)) {
          sections.push({
            type: sectionType,
            line: i,
            text: line
          });
        }
      }
    }
    
    return sections;
  }

  analyzeExcelSheet(data, sheetName) {
    const analysis = {
      sheetName,
      hasHeaders: this.detectHeaders(data),
      numericColumns: this.findNumericColumns(data),
      dateColumns: this.findDateColumns(data),
      possibleMetrics: this.identifyPossibleMetrics(data)
    };
    
    return analysis;
  }

  detectHeaders(data) {
    if (data.length === 0) return false;
    
    const firstRow = data[0];
    const secondRow = data[1];
    
    if (!firstRow || !secondRow) return false;
    
    // Check if first row contains text and second row contains numbers
    const firstRowHasText = firstRow.some(cell => 
      typeof cell === 'string' && isNaN(parseFloat(cell))
    );
    const secondRowHasNumbers = secondRow.some(cell => 
      !isNaN(parseFloat(cell))
    );
    
    return firstRowHasText && secondRowHasNumbers;
  }

  findNumericColumns(data) {
    if (data.length < 2) return [];
    
    const numericColumns = [];
    const headerRow = data[0];
    
    for (let col = 0; col < headerRow.length; col++) {
      let numericCount = 0;
      let totalCount = 0;
      
      for (let row = 1; row < Math.min(data.length, 10); row++) {
        if (data[row][col] !== undefined && data[row][col] !== '') {
          totalCount++;
          if (!isNaN(parseFloat(data[row][col]))) {
            numericCount++;
          }
        }
      }
      
      if (totalCount > 0 && numericCount / totalCount > 0.7) {
        numericColumns.push({
          index: col,
          header: headerRow[col],
          numericRatio: numericCount / totalCount
        });
      }
    }
    
    return numericColumns;
  }

  findDateColumns(data) {
    if (data.length < 2) return [];
    
    const dateColumns = [];
    const headerRow = data[0];
    
    for (let col = 0; col < headerRow.length; col++) {
      let dateCount = 0;
      let totalCount = 0;
      
      for (let row = 1; row < Math.min(data.length, 10); row++) {
        if (data[row][col] !== undefined && data[row][col] !== '') {
          totalCount++;
          if (this.isDateValue(data[row][col])) {
            dateCount++;
          }
        }
      }
      
      if (totalCount > 0 && dateCount / totalCount > 0.7) {
        dateColumns.push({
          index: col,
          header: headerRow[col],
          dateRatio: dateCount / totalCount
        });
      }
    }
    
    return dateColumns;
  }

  isDateValue(value) {
    if (typeof value === 'number') {
      // Excel date serial number
      return value > 25000 && value < 100000;
    }
    
    if (typeof value === 'string') {
      const datePatterns = [
        /^\d{1,2}\/\d{1,2}\/\d{4}$/,
        /^\d{4}-\d{2}-\d{2}$/,
        /^\d{1,2}-\d{1,2}-\d{4}$/
      ];
      
      return datePatterns.some(pattern => pattern.test(value));
    }
    
    return false;
  }

  identifyPossibleMetrics(data) {
    const metrics = [];
    
    if (data.length < 2) return metrics;
    
    const headerRow = data[0];
    
    const metricPatterns = {
      nav: /nav|net asset value/i,
      aum: /aum|assets under management/i,
      returns: /return|performance/i,
      expense: /expense|ter|fee/i,
      units: /units|shares/i,
      price: /price|rate/i
    };
    
    for (let col = 0; col < headerRow.length; col++) {
      const header = String(headerRow[col]).toLowerCase();
      
      for (const [metricType, pattern] of Object.entries(metricPatterns)) {
        if (pattern.test(header)) {
          metrics.push({
            column: col,
            header: headerRow[col],
            type: metricType
          });
        }
      }
    }
    
    return metrics;
  }

  consolidateExcelData(sheets) {
    const consolidated = {
      financialMetrics: {},
      performanceData: {},
      portfolioData: {},
      timeSeriesData: []
    };
    
    for (const [sheetName, sheetData] of Object.entries(sheets)) {
      // Look for financial metrics in each sheet
      const metrics = sheetData.analysis.possibleMetrics;
      
      for (const metric of metrics) {
        if (!consolidated.financialMetrics[metric.type]) {
          consolidated.financialMetrics[metric.type] = [];
        }
        
        consolidated.financialMetrics[metric.type].push({
          sheet: sheetName,
          column: metric.column,
          header: metric.header
        });
      }
    }
    
    return consolidated;
  }

  analyzeCSVStructure(data) {
    return {
      hasHeaders: this.detectHeaders(data),
      rowCount: data.length,
      columnCount: data[0]?.length || 0,
      numericColumns: this.findNumericColumns(data),
      dateColumns: this.findDateColumns(data)
    };
  }

  extractCSVFinancialData(data) {
    const financialData = {};
    
    if (data.length < 2) return financialData;
    
    const headers = data[0];
    
    // Look for common financial data patterns
    for (let col = 0; col < headers.length; col++) {
      const header = String(headers[col]).toLowerCase();
      
      if (header.includes('nav') || header.includes('price')) {
        financialData.nav = this.extractColumnData(data, col);
      } else if (header.includes('aum') || header.includes('assets')) {
        financialData.aum = this.extractColumnData(data, col);
      } else if (header.includes('return') || header.includes('performance')) {
        financialData.returns = this.extractColumnData(data, col);
      }
    }
    
    return financialData;
  }

  extractColumnData(data, columnIndex) {
    const columnData = [];
    
    for (let row = 1; row < Math.min(data.length, 100); row++) {
      const value = data[row][columnIndex];
      if (value !== undefined && value !== '') {
        columnData.push(value);
      }
    }
    
    return columnData;
  }

  extractDataPoints(analysisResult) {
    const dataPoints = [];
    
    // Extract from financial metrics
    if (analysisResult.financialMetrics) {
      for (const [metric, values] of Object.entries(analysisResult.financialMetrics)) {
        for (const value of values) {
          dataPoints.push({
            type: 'financial_metric',
            metric,
            value: value.value,
            unit: value.unit,
            context: value.context,
            extractedAt: new Date().toISOString()
          });
        }
      }
    }
    
    // Extract from performance data
    if (analysisResult.performanceData) {
      for (const [period, values] of Object.entries(analysisResult.performanceData)) {
        for (const value of values) {
          dataPoints.push({
            type: 'performance_data',
            period,
            value: value.value,
            context: value.context,
            extractedAt: new Date().toISOString()
          });
        }
      }
    }
    
    // Extract from Excel consolidated data
    if (analysisResult.consolidatedData) {
      for (const [dataType, data] of Object.entries(analysisResult.consolidatedData)) {
        if (Array.isArray(data)) {
          for (const item of data) {
            dataPoints.push({
              type: 'excel_data',
              dataType,
              ...item,
              extractedAt: new Date().toISOString()
            });
          }
        }
      }
    }
    
    return {
      documentKey: this.generateDocumentKey(analysisResult.filePath, analysisResult.documentInfo),
      dataPoints,
      extractedAt: new Date().toISOString()
    };
  }

  generateDocumentKey(filePath, documentInfo) {
    const fileName = path.basename(filePath);
    const amcName = documentInfo?.amcName || 'unknown';
    return `${amcName}_${fileName}_${Date.now()}`;
  }

  // Public API methods
  async analyzeDocumentBatch(documents) {
    const results = [];
    
    for (const document of documents) {
      try {
        const result = await this.analyzeDocument(document.filePath, document.info);
        if (result) {
          results.push(result);
        }
      } catch (error) {
        logger.error(`❌ Batch analysis failed for ${document.filePath}:`, error);
      }
    }
    
    await this.saveAnalysisResults();
    return results;
  }

  async generateAnalysisReport(amcName = null) {
    const report = {
      generatedAt: new Date().toISOString(),
      stats: this.stats,
      summary: {},
      details: {}
    };
    
    // Filter results by AMC if specified
    const relevantResults = amcName ? 
      Array.from(this.analysisResults.entries()).filter(([key]) => 
        key.includes(amcName.toLowerCase().replace(/[^a-z0-9]/g, '_'))
      ) :
      Array.from(this.analysisResults.entries());
    
    // Generate summary
    report.summary = {
      totalDocuments: relevantResults.length,
      documentTypes: this.getDocumentTypeSummary(relevantResults),
      dataPointsExtracted: this.getTotalDataPoints(relevantResults),
      commonMetrics: this.getCommonMetrics(relevantResults)
    };
    
    // Save report
    const reportPath = path.join(
      this.config.outputPath, 
      'analysis-reports',
      `analysis-report-${amcName || 'all'}-${Date.now()}.json`
    );
    
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    return report;
  }

  getDocumentTypeSummary(results) {
    const typeCounts = {};
    
    for (const [key, result] of results) {
      const type = result.type;
      typeCounts[type] = (typeCounts[type] || 0) + 1;
    }
    
    return typeCounts;
  }

  getTotalDataPoints(results) {
    let total = 0;
    
    for (const [key] of results) {
      const extractedData = this.extractedData.get(key);
      if (extractedData) {
        total += extractedData.dataPoints?.length || 0;
      }
    }
    
    return total;
  }

  getCommonMetrics(results) {
    const metricCounts = {};
    
    for (const [key] of results) {
      const extractedData = this.extractedData.get(key);
      if (extractedData && extractedData.dataPoints) {
        for (const dataPoint of extractedData.dataPoints) {
          const metric = dataPoint.metric || dataPoint.dataType;
          if (metric) {
            metricCounts[metric] = (metricCounts[metric] || 0) + 1;
          }
        }
      }
    }
    
    return Object.entries(metricCounts)
      .sort(([,a], [,b]) => b - a)
      .slice(0, 10)
      .reduce((obj, [metric, count]) => {
        obj[metric] = count;
        return obj;
      }, {});
  }

  getAnalysisStats() {
    return {
      ...this.stats,
      totalAnalysisResults: this.analysisResults.size,
      totalExtractedDataSets: this.extractedData.size
    };
  }
}

module.exports = { AMCDocumentAnalyzer };
