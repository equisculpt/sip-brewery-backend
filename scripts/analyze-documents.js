/**
 * 🔬 AMC DOCUMENT ANALYSIS UTILITY
 * 
 * On-demand analysis of existing AMC documents
 * Batch processing and reporting capabilities
 * 
 * @author Financial Document Intelligence Team
 * @version 1.0.0 - Document Analysis Utility
 */

const fs = require('fs').promises;
const path = require('path');
const logger = require('../src/utils/logger');
const { AMCDocumentAnalyzer } = require('../src/finance_crawler/document-analyzer');
const { AMCDocumentMonitor } = require('../src/finance_crawler/document-monitor');

class DocumentAnalysisUtility {
  constructor() {
    this.analyzer = null;
    this.monitor = null;
    
    this.config = {
      // Analysis settings
      enablePDFAnalysis: true,
      enableExcelAnalysis: true,
      enableContentExtraction: true,
      extractFinancialMetrics: true,
      extractPerformanceData: true,
      extractPortfolioData: true,
      
      // Processing settings
      batchSize: 5,
      processingDelay: 1000,
      maxConcurrentAnalysis: 3,
      
      // Output settings
      outputPath: './data/analysis-results',
      generateReports: true,
      generateInsights: true,
      
      // Filter settings
      documentTypes: ['factsheets', 'portfolios', 'performance', 'financials', 'presentations'],
      amcFilter: null, // Analyze specific AMC or null for all
      dateFilter: null // Analyze documents after specific date or null for all
    };
    
    this.stats = {
      totalDocuments: 0,
      analyzedDocuments: 0,
      skippedDocuments: 0,
      failedDocuments: 0,
      extractedDataPoints: 0,
      processingTime: 0,
      startTime: null,
      endTime: null
    };
  }

  async run() {
    try {
      console.log('🔬 Starting AMC Document Analysis Utility...\n');
      
      // Parse command line arguments
      this.parseArguments();
      
      // Initialize components
      await this.initialize();
      
      // Discover documents
      const documents = await this.discoverDocuments();
      
      // Analyze documents
      await this.analyzeDocuments(documents);
      
      // Generate reports
      await this.generateReports();
      
      // Display summary
      this.displaySummary();
      
      console.log('\n✅ Document analysis completed successfully!');
      
    } catch (error) {
      console.error('❌ Document analysis failed:', error);
      process.exit(1);
    }
  }

  parseArguments() {
    const args = process.argv.slice(2);
    
    for (let i = 0; i < args.length; i++) {
      const arg = args[i];
      
      switch (arg) {
        case '--amc':
          this.config.amcFilter = args[++i];
          break;
          
        case '--type':
          this.config.documentTypes = [args[++i]];
          break;
          
        case '--types':
          this.config.documentTypes = args[++i].split(',');
          break;
          
        case '--batch-size':
          this.config.batchSize = parseInt(args[++i]);
          break;
          
        case '--no-pdf':
          this.config.enablePDFAnalysis = false;
          break;
          
        case '--no-excel':
          this.config.enableExcelAnalysis = false;
          break;
          
        case '--no-reports':
          this.config.generateReports = false;
          break;
          
        case '--output':
          this.config.outputPath = args[++i];
          break;
          
        case '--since':
          this.config.dateFilter = new Date(args[++i]);
          break;
          
        case '--help':
          this.showHelp();
          process.exit(0);
          break;
      }
    }
    
    console.log('⚙️ Configuration:');
    console.log(`  • AMC Filter: ${this.config.amcFilter || 'All AMCs'}`);
    console.log(`  • Document Types: ${this.config.documentTypes.join(', ')}`);
    console.log(`  • Batch Size: ${this.config.batchSize}`);
    console.log(`  • PDF Analysis: ${this.config.enablePDFAnalysis ? 'Enabled' : 'Disabled'}`);
    console.log(`  • Excel Analysis: ${this.config.enableExcelAnalysis ? 'Enabled' : 'Disabled'}`);
    console.log(`  • Output Path: ${this.config.outputPath}`);
    console.log(`  • Date Filter: ${this.config.dateFilter ? this.config.dateFilter.toISOString() : 'None'}\n`);
  }

  showHelp() {
    console.log(`
🔬 AMC Document Analysis Utility

Usage: node scripts/analyze-documents.js [options]

Options:
  --amc <name>           Analyze documents for specific AMC only
  --type <type>          Analyze specific document type (factsheets, portfolios, etc.)
  --types <type1,type2>  Analyze multiple document types (comma-separated)
  --batch-size <num>     Number of documents to process in parallel (default: 5)
  --no-pdf              Disable PDF analysis
  --no-excel            Disable Excel analysis
  --no-reports          Skip report generation
  --output <path>       Custom output directory
  --since <date>        Analyze documents modified since date (YYYY-MM-DD)
  --help                Show this help message

Examples:
  node scripts/analyze-documents.js
  node scripts/analyze-documents.js --amc "HDFC Asset Management"
  node scripts/analyze-documents.js --type factsheets --batch-size 10
  node scripts/analyze-documents.js --types factsheets,performance --since 2024-01-01
    `);
  }

  async initialize() {
    console.log('🔧 Initializing analysis components...');
    
    // Initialize analyzer
    this.analyzer = new AMCDocumentAnalyzer({
      enablePDFAnalysis: this.config.enablePDFAnalysis,
      enableExcelAnalysis: this.config.enableExcelAnalysis,
      enableContentExtraction: this.config.enableContentExtraction,
      extractFinancialMetrics: this.config.extractFinancialMetrics,
      extractPerformanceData: this.config.extractPerformanceData,
      extractPortfolioData: this.config.extractPortfolioData,
      outputPath: this.config.outputPath
    });
    
    await this.analyzer.initialize();
    
    // Initialize monitor (to access document registry)
    this.monitor = new AMCDocumentMonitor({
      enableContinuousMonitoring: false
    });
    
    await this.monitor.initialize();
    
    console.log('✅ Components initialized\n');
  }

  async discoverDocuments() {
    console.log('🔍 Discovering documents for analysis...');
    
    const documents = [];
    
    for (const [amcKey, amcInfo] of this.monitor.documentRegistry) {
      // Apply AMC filter
      if (this.config.amcFilter && 
          !amcInfo.name.toLowerCase().includes(this.config.amcFilter.toLowerCase())) {
        continue;
      }
      
      for (const [docKey, doc] of Object.entries(amcInfo.documents)) {
        // Apply document type filter
        if (!this.config.documentTypes.includes(doc.type)) {
          continue;
        }
        
        // Apply date filter
        if (this.config.dateFilter && 
            new Date(doc.lastUpdated) < this.config.dateFilter) {
          continue;
        }
        
        // Check if file exists
        if (doc.filePath && await this.fileExists(doc.filePath)) {
          documents.push({
            ...doc,
            amcName: amcInfo.name,
            amcKey,
            docKey
          });
        }
      }
    }
    
    this.stats.totalDocuments = documents.length;
    
    console.log(`📄 Found ${documents.length} documents for analysis:`);
    
    // Group by type for summary
    const typeGroups = {};
    documents.forEach(doc => {
      typeGroups[doc.type] = (typeGroups[doc.type] || 0) + 1;
    });
    
    for (const [type, count] of Object.entries(typeGroups)) {
      console.log(`  • ${type}: ${count} documents`);
    }
    
    console.log();
    
    return documents;
  }

  async fileExists(filePath) {
    try {
      await fs.access(filePath);
      return true;
    } catch {
      return false;
    }
  }

  async analyzeDocuments(documents) {
    console.log(`🔬 Starting analysis of ${documents.length} documents...\n`);
    
    this.stats.startTime = Date.now();
    
    // Process documents in batches
    for (let i = 0; i < documents.length; i += this.config.batchSize) {
      const batch = documents.slice(i, i + this.config.batchSize);
      
      console.log(`📦 Processing batch ${Math.floor(i / this.config.batchSize) + 1}/${Math.ceil(documents.length / this.config.batchSize)} (${batch.length} documents)...`);
      
      const batchPromises = batch.map(doc => this.analyzeDocument(doc));
      const results = await Promise.allSettled(batchPromises);
      
      // Process results
      results.forEach((result, index) => {
        const doc = batch[index];
        
        if (result.status === 'fulfilled' && result.value) {
          this.stats.analyzedDocuments++;
          
          // Count extracted data points
          if (result.value.extractedData && result.value.extractedData.dataPoints) {
            this.stats.extractedDataPoints += result.value.extractedData.dataPoints.length;
          }
          
          console.log(`  ✅ ${doc.amcName} - ${path.basename(doc.filePath)}`);
          
        } else {
          this.stats.failedDocuments++;
          console.log(`  ❌ ${doc.amcName} - ${path.basename(doc.filePath)} - ${result.reason?.message || 'Unknown error'}`);
        }
      });
      
      // Delay between batches
      if (i + this.config.batchSize < documents.length) {
        await new Promise(resolve => setTimeout(resolve, this.config.processingDelay));
      }
    }
    
    this.stats.endTime = Date.now();
    this.stats.processingTime = this.stats.endTime - this.stats.startTime;
    
    console.log('\n✅ Document analysis completed\n');
  }

  async analyzeDocument(document) {
    try {
      const analysisResult = await this.analyzer.analyzeDocument(
        document.filePath,
        {
          amcName: document.amcName,
          documentType: document.type,
          ...document
        }
      );
      
      if (analysisResult) {
        // Extract data points
        const extractedData = this.analyzer.extractDataPoints(analysisResult);
        
        return {
          document,
          analysisResult,
          extractedData
        };
      }
      
      return null;
      
    } catch (error) {
      throw new Error(`Analysis failed: ${error.message}`);
    }
  }

  async generateReports() {
    if (!this.config.generateReports) {
      console.log('📊 Skipping report generation (disabled)\n');
      return;
    }
    
    console.log('📊 Generating analysis reports...');
    
    try {
      // Generate overall analysis report
      const overallReport = await this.analyzer.generateAnalysisReport();
      console.log('  ✅ Overall analysis report generated');
      
      // Generate AMC-specific reports if filtering by AMC
      if (this.config.amcFilter) {
        const amcReport = await this.analyzer.generateAnalysisReport(this.config.amcFilter);
        console.log(`  ✅ AMC-specific report generated for ${this.config.amcFilter}`);
      }
      
      // Generate summary insights
      await this.generateSummaryInsights();
      console.log('  ✅ Summary insights generated');
      
      // Generate performance metrics
      await this.generatePerformanceMetrics();
      console.log('  ✅ Performance metrics generated');
      
    } catch (error) {
      console.log(`  ⚠️ Report generation encountered issues: ${error.message}`);
    }
    
    console.log('✅ Reports generated\n');
  }

  async generateSummaryInsights() {
    const insights = {
      generatedAt: new Date().toISOString(),
      analysisStats: this.stats,
      configuration: this.config,
      
      documentBreakdown: {
        byType: {},
        byAMC: {},
        byFileFormat: {}
      },
      
      extractionSummary: {
        totalDataPoints: this.stats.extractedDataPoints,
        averageDataPointsPerDocument: Math.round(this.stats.extractedDataPoints / this.stats.analyzedDocuments),
        successRate: Math.round((this.stats.analyzedDocuments / this.stats.totalDocuments) * 100)
      },
      
      performance: {
        processingTimeMs: this.stats.processingTime,
        documentsPerSecond: Math.round(this.stats.analyzedDocuments / (this.stats.processingTime / 1000)),
        averageTimePerDocument: Math.round(this.stats.processingTime / this.stats.analyzedDocuments)
      }
    };
    
    // Save insights
    const insightsPath = path.join(this.config.outputPath, 'insights', `analysis-insights-${Date.now()}.json`);
    await fs.writeFile(insightsPath, JSON.stringify(insights, null, 2));
  }

  async generatePerformanceMetrics() {
    const metrics = {
      generatedAt: new Date().toISOString(),
      
      processingMetrics: {
        totalDocuments: this.stats.totalDocuments,
        successfulAnalysis: this.stats.analyzedDocuments,
        failedAnalysis: this.stats.failedDocuments,
        skippedDocuments: this.stats.skippedDocuments,
        successRate: (this.stats.analyzedDocuments / this.stats.totalDocuments) * 100,
        failureRate: (this.stats.failedDocuments / this.stats.totalDocuments) * 100
      },
      
      performanceMetrics: {
        totalProcessingTime: this.stats.processingTime,
        averageTimePerDocument: this.stats.processingTime / this.stats.analyzedDocuments,
        documentsPerSecond: this.stats.analyzedDocuments / (this.stats.processingTime / 1000),
        dataPointsExtracted: this.stats.extractedDataPoints,
        dataPointsPerDocument: this.stats.extractedDataPoints / this.stats.analyzedDocuments
      },
      
      systemMetrics: {
        batchSize: this.config.batchSize,
        processingDelay: this.config.processingDelay,
        maxConcurrentAnalysis: this.config.maxConcurrentAnalysis
      }
    };
    
    // Save metrics
    const metricsPath = path.join(this.config.outputPath, 'analysis-reports', `performance-metrics-${Date.now()}.json`);
    await fs.writeFile(metricsPath, JSON.stringify(metrics, null, 2));
  }

  displaySummary() {
    console.log('📊 ANALYSIS SUMMARY:');
    console.log(`  📄 Total Documents: ${this.stats.totalDocuments}`);
    console.log(`  ✅ Successfully Analyzed: ${this.stats.analyzedDocuments}`);
    console.log(`  ❌ Failed: ${this.stats.failedDocuments}`);
    console.log(`  ⏭️  Skipped: ${this.stats.skippedDocuments}`);
    console.log(`  📊 Data Points Extracted: ${this.stats.extractedDataPoints}`);
    console.log(`  ⏱️  Processing Time: ${this.formatDuration(this.stats.processingTime)}`);
    console.log(`  🚀 Success Rate: ${Math.round((this.stats.analyzedDocuments / this.stats.totalDocuments) * 100)}%`);
    console.log(`  📈 Documents/Second: ${Math.round(this.stats.analyzedDocuments / (this.stats.processingTime / 1000))}`);
    console.log(`  📋 Average Data Points/Document: ${Math.round(this.stats.extractedDataPoints / this.stats.analyzedDocuments)}`);
    
    console.log('\n📁 Output Locations:');
    console.log(`  • Analysis Results: ${this.config.outputPath}`);
    console.log(`  • Reports: ${path.join(this.config.outputPath, 'analysis-reports')}`);
    console.log(`  • Insights: ${path.join(this.config.outputPath, 'insights')}`);
    console.log(`  • Extracted Data: ${path.join(this.config.outputPath, 'extracted-data')}`);
  }

  formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    
    if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;
  }
}

// Run utility if called directly
if (require.main === module) {
  const utility = new DocumentAnalysisUtility();
  utility.run().catch(console.error);
}

module.exports = { DocumentAnalysisUtility };
