/**
 * 📄 AMC DOCUMENT MONITOR SYSTEM
 * 
 * Continuously monitors PDF and Excel files from all 44 AMCs
 * Detects changes, extracts data, and feeds insights to ASI analysis
 * 
 * @author Financial Document Intelligence Team
 * @version 1.0.0 - Automated Document Monitoring
 */

const EventEmitter = require('events');
const axios = require('axios');
const cheerio = require('cheerio');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');
const schedule = require('node-cron');
const logger = require('../utils/logger');
const { amcDataSources } = require('./parsers/amc-data-sources');

class AMCDocumentMonitor extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      // Monitoring settings
      enableContinuousMonitoring: options.enableContinuousMonitoring !== false,
      monitoringInterval: options.monitoringInterval || '0 */6 * * *', // Every 6 hours
      
      // Document settings
      supportedFormats: options.supportedFormats || ['pdf', 'xlsx', 'xls', 'csv'],
      maxFileSize: options.maxFileSize || 50 * 1024 * 1024, // 50MB
      downloadTimeout: options.downloadTimeout || 30000,
      
      // Storage settings
      documentsPath: options.documentsPath || './data/amc-documents',
      metadataPath: options.metadataPath || './data/amc-metadata',
      
      // Analysis settings
      enableContentAnalysis: options.enableContentAnalysis !== false,
      enableChangeDetection: options.enableChangeDetection !== false,
      
      ...options
    };
    
    // Document tracking
    this.documentRegistry = new Map();
    this.documentHashes = new Map();
    this.changeHistory = new Map();
    
    // AMC document patterns
    this.documentPatterns = {
      factsheets: [
        'factsheet', 'fact-sheet', 'fund-factsheet', 'scheme-factsheet',
        'monthly-factsheet', 'quarterly-factsheet'
      ],
      portfolios: [
        'portfolio', 'holding', 'holdings', 'portfolio-disclosure',
        'monthly-portfolio', 'quarterly-portfolio'
      ],
      performance: [
        'performance', 'returns', 'nav', 'performance-report',
        'annual-report', 'semi-annual'
      ],
      financials: [
        'annual-report', 'financial-statement', 'balance-sheet',
        'income-statement', 'cash-flow'
      ],
      presentations: [
        'presentation', 'investor-presentation', 'quarterly-presentation',
        'earnings-presentation'
      ]
    };
    
    // Monitoring statistics
    this.stats = {
      totalDocumentsTracked: 0,
      documentsDownloaded: 0,
      changesDetected: 0,
      analysisCompleted: 0,
      lastMonitoringRun: null,
      errors: 0
    };
    
    this.isInitialized = false;
  }

  async initialize() {
    try {
      logger.info('📄 Initializing AMC Document Monitor...');
      
      // Create directories
      await this.createDirectories();
      
      // Load existing document registry
      await this.loadDocumentRegistry();
      
      // Setup document patterns for each AMC
      await this.setupAMCDocumentPatterns();
      
      // Start continuous monitoring if enabled
      if (this.config.enableContinuousMonitoring) {
        this.startContinuousMonitoring();
      }
      
      this.isInitialized = true;
      logger.info('✅ AMC Document Monitor initialized successfully');
      
      this.emit('initialized');
      
    } catch (error) {
      logger.error('❌ AMC Document Monitor initialization failed:', error);
      throw error;
    }
  }

  async createDirectories() {
    const dirs = [
      this.config.documentsPath,
      this.config.metadataPath,
      path.join(this.config.documentsPath, 'factsheets'),
      path.join(this.config.documentsPath, 'portfolios'),
      path.join(this.config.documentsPath, 'performance'),
      path.join(this.config.documentsPath, 'financials'),
      path.join(this.config.documentsPath, 'presentations')
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
    
    logger.info('📁 Document directories created');
  }

  async loadDocumentRegistry() {
    try {
      const registryPath = path.join(this.config.metadataPath, 'document-registry.json');
      const data = await fs.readFile(registryPath, 'utf8');
      const registry = JSON.parse(data);
      
      this.documentRegistry = new Map(Object.entries(registry.documents || {}));
      this.documentHashes = new Map(Object.entries(registry.hashes || {}));
      this.changeHistory = new Map(Object.entries(registry.changes || {}));
      
      logger.info(`📚 Loaded ${this.documentRegistry.size} documents from registry`);
      
    } catch (error) {
      logger.info('📚 Creating new document registry');
      this.documentRegistry = new Map();
      this.documentHashes = new Map();
      this.changeHistory = new Map();
    }
  }

  async saveDocumentRegistry() {
    try {
      const registryPath = path.join(this.config.metadataPath, 'document-registry.json');
      const registry = {
        documents: Object.fromEntries(this.documentRegistry),
        hashes: Object.fromEntries(this.documentHashes),
        changes: Object.fromEntries(this.changeHistory),
        lastUpdated: new Date().toISOString(),
        stats: this.stats
      };
      
      await fs.writeFile(registryPath, JSON.stringify(registry, null, 2));
      
    } catch (error) {
      logger.error('❌ Failed to save document registry:', error);
    }
  }

  async setupAMCDocumentPatterns() {
    for (const [amcName, amcData] of Object.entries(amcDataSources)) {
      const amcKey = this.generateAMCKey(amcName);
      
      if (!this.documentRegistry.has(amcKey)) {
        this.documentRegistry.set(amcKey, {
          name: amcName,
          websites: amcData.websites,
          rank: amcData.rank,
          category: amcData.category,
          documents: {},
          lastScanned: null,
          totalDocuments: 0
        });
      }
    }
    
    this.stats.totalDocumentsTracked = this.documentRegistry.size;
    logger.info(`🎯 Setup document patterns for ${this.documentRegistry.size} AMCs`);
  }

  startContinuousMonitoring() {
    // Schedule document monitoring
    schedule.schedule(this.config.monitoringInterval, async () => {
      await this.performFullDocumentScan();
    });
    
    // Daily change analysis
    schedule.schedule('0 2 * * *', async () => {
      await this.performChangeAnalysis();
    });
    
    // Weekly deep analysis
    schedule.schedule('0 3 * * 0', async () => {
      await this.performDeepAnalysis();
    });
    
    logger.info('⏰ Continuous document monitoring started');
  }

  async performFullDocumentScan() {
    try {
      logger.info('🔍 Starting full AMC document scan...');
      this.stats.lastMonitoringRun = new Date();
      
      const scanPromises = [];
      
      for (const [amcKey, amcInfo] of this.documentRegistry) {
        scanPromises.push(this.scanAMCDocuments(amcKey, amcInfo));
        
        // Limit concurrent scans
        if (scanPromises.length >= 5) {
          await Promise.allSettled(scanPromises);
          scanPromises.length = 0;
          
          // Delay between batches
          await new Promise(resolve => setTimeout(resolve, 2000));
        }
      }
      
      // Process remaining scans
      if (scanPromises.length > 0) {
        await Promise.allSettled(scanPromises);
      }
      
      // Save registry after scan
      await this.saveDocumentRegistry();
      
      logger.info('✅ Full document scan completed');
      this.emit('scanCompleted', this.stats);
      
    } catch (error) {
      logger.error('❌ Full document scan failed:', error);
      this.stats.errors++;
    }
  }

  async scanAMCDocuments(amcKey, amcInfo) {
    try {
      logger.debug(`🔍 Scanning documents for ${amcInfo.name}...`);
      
      const foundDocuments = [];
      
      for (const website of amcInfo.websites) {
        try {
          const documents = await this.discoverDocuments(website, amcInfo.name);
          foundDocuments.push(...documents);
          
          // Delay between website scans
          await new Promise(resolve => setTimeout(resolve, 1000));
          
        } catch (error) {
          logger.warn(`⚠️ Failed to scan ${website} for ${amcInfo.name}:`, error.message);
        }
      }
      
      // Process discovered documents
      for (const doc of foundDocuments) {
        await this.processDocument(amcKey, amcInfo, doc);
      }
      
      // Update AMC info
      amcInfo.lastScanned = new Date().toISOString();
      amcInfo.totalDocuments = Object.keys(amcInfo.documents).length;
      
      logger.debug(`✅ Scanned ${foundDocuments.length} documents for ${amcInfo.name}`);
      
    } catch (error) {
      logger.error(`❌ Document scan failed for ${amcInfo.name}:`, error);
      this.stats.errors++;
    }
  }

  async discoverDocuments(website, amcName) {
    try {
      const response = await axios.get(website, {
        timeout: 10000,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });
      
      const $ = cheerio.load(response.data);
      const documents = [];
      
      // Find all links to documents
      $('a[href]').each((i, elem) => {
        const href = $(elem).attr('href');
        const text = $(elem).text().trim();
        
        if (this.isDocumentLink(href, text)) {
          const fullUrl = this.resolveUrl(href, website);
          const docType = this.classifyDocument(href, text);
          
          documents.push({
            url: fullUrl,
            title: text,
            type: docType,
            discoveredAt: new Date().toISOString(),
            source: website
          });
        }
      });
      
      return documents;
      
    } catch (error) {
      throw new Error(`Document discovery failed: ${error.message}`);
    }
  }

  isDocumentLink(href, text) {
    if (!href) return false;
    
    // Check file extension
    const hasDocExtension = this.config.supportedFormats.some(ext => 
      href.toLowerCase().includes(`.${ext}`)
    );
    
    // Check text patterns
    const hasDocPattern = Object.values(this.documentPatterns).flat().some(pattern =>
      text.toLowerCase().includes(pattern) || href.toLowerCase().includes(pattern)
    );
    
    return hasDocExtension || hasDocPattern;
  }

  classifyDocument(href, text) {
    const combined = (href + ' ' + text).toLowerCase();
    
    for (const [type, patterns] of Object.entries(this.documentPatterns)) {
      if (patterns.some(pattern => combined.includes(pattern))) {
        return type;
      }
    }
    
    return 'other';
  }

  resolveUrl(href, baseUrl) {
    try {
      if (href.startsWith('http')) {
        return href;
      }
      
      const base = new URL(baseUrl);
      return new URL(href, base).toString();
      
    } catch (error) {
      return href;
    }
  }

  async processDocument(amcKey, amcInfo, document) {
    try {
      const docKey = this.generateDocumentKey(document.url);
      const existingDoc = amcInfo.documents[docKey];
      
      // Check if document is new or changed
      const isNew = !existingDoc;
      const hasChanged = await this.hasDocumentChanged(document.url, existingDoc);
      
      if (isNew || hasChanged) {
        // Download and analyze document
        const downloadResult = await this.downloadDocument(document);
        
        if (downloadResult.success) {
          // Update document info
          amcInfo.documents[docKey] = {
            ...document,
            filePath: downloadResult.filePath,
            fileSize: downloadResult.fileSize,
            fileHash: downloadResult.fileHash,
            lastUpdated: new Date().toISOString(),
            downloadCount: (existingDoc?.downloadCount || 0) + 1
          };
          
          // Track changes
          if (hasChanged) {
            this.trackDocumentChange(amcKey, docKey, existingDoc, amcInfo.documents[docKey]);
          }
          
          // Emit document event
          this.emit('documentProcessed', {
            amcName: amcInfo.name,
            document: amcInfo.documents[docKey],
            isNew,
            hasChanged
          });
          
          this.stats.documentsDownloaded++;
          if (hasChanged) this.stats.changesDetected++;
        }
      }
      
    } catch (error) {
      logger.error(`❌ Document processing failed for ${document.url}:`, error);
    }
  }

  async hasDocumentChanged(url, existingDoc) {
    if (!existingDoc) return false;
    
    try {
      // Check if file hash has changed
      const currentHash = await this.getDocumentHash(url);
      return currentHash !== existingDoc.fileHash;
      
    } catch (error) {
      return false;
    }
  }

  async getDocumentHash(url) {
    try {
      const response = await axios.head(url, { timeout: 5000 });
      const lastModified = response.headers['last-modified'];
      const contentLength = response.headers['content-length'];
      
      return crypto.createHash('md5')
        .update(`${url}_${lastModified}_${contentLength}`)
        .digest('hex');
        
    } catch (error) {
      return null;
    }
  }

  async downloadDocument(document) {
    try {
      const response = await axios.get(document.url, {
        responseType: 'arraybuffer',
        timeout: this.config.downloadTimeout,
        maxContentLength: this.config.maxFileSize,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
      });
      
      // Generate file path
      const fileName = this.generateFileName(document);
      const filePath = path.join(
        this.config.documentsPath,
        document.type,
        fileName
      );
      
      // Save file
      await fs.writeFile(filePath, response.data);
      
      // Calculate hash
      const fileHash = crypto.createHash('md5')
        .update(response.data)
        .digest('hex');
      
      return {
        success: true,
        filePath,
        fileSize: response.data.length,
        fileHash
      };
      
    } catch (error) {
      logger.error(`❌ Document download failed for ${document.url}:`, error.message);
      return { success: false, error: error.message };
    }
  }

  generateFileName(document) {
    const url = new URL(document.url);
    const pathParts = url.pathname.split('/');
    let fileName = pathParts[pathParts.length - 1];
    
    if (!fileName || !fileName.includes('.')) {
      const ext = this.getFileExtension(document.url);
      fileName = `${Date.now()}_document.${ext}`;
    }
    
    // Sanitize filename
    return fileName.replace(/[^a-zA-Z0-9.-]/g, '_');
  }

  getFileExtension(url) {
    const match = url.match(/\.([a-zA-Z0-9]+)(?:\?|$)/);
    return match ? match[1] : 'pdf';
  }

  trackDocumentChange(amcKey, docKey, oldDoc, newDoc) {
    const changeKey = `${amcKey}_${docKey}`;
    
    if (!this.changeHistory.has(changeKey)) {
      this.changeHistory.set(changeKey, []);
    }
    
    const changes = this.changeHistory.get(changeKey);
    changes.push({
      timestamp: new Date().toISOString(),
      oldHash: oldDoc?.fileHash,
      newHash: newDoc.fileHash,
      oldSize: oldDoc?.fileSize,
      newSize: newDoc.fileSize,
      changeType: oldDoc ? 'updated' : 'new'
    });
    
    // Keep only last 10 changes
    if (changes.length > 10) {
      changes.splice(0, changes.length - 10);
    }
  }

  async performChangeAnalysis() {
    try {
      logger.info('📊 Performing change analysis...');
      
      const analysis = {
        totalChanges: 0,
        changesByAMC: {},
        changesByType: {},
        recentChanges: [],
        timestamp: new Date().toISOString()
      };
      
      // Analyze changes from last 24 hours
      const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000);
      
      for (const [changeKey, changes] of this.changeHistory) {
        const recentChanges = changes.filter(change => 
          new Date(change.timestamp) > yesterday
        );
        
        if (recentChanges.length > 0) {
          analysis.totalChanges += recentChanges.length;
          analysis.recentChanges.push(...recentChanges.map(change => ({
            ...change,
            key: changeKey
          })));
        }
      }
      
      // Save analysis
      const analysisPath = path.join(this.config.metadataPath, 'change-analysis.json');
      await fs.writeFile(analysisPath, JSON.stringify(analysis, null, 2));
      
      this.stats.analysisCompleted++;
      
      // Emit analysis event
      this.emit('changeAnalysis', analysis);
      
      logger.info(`📊 Change analysis completed: ${analysis.totalChanges} changes detected`);
      
    } catch (error) {
      logger.error('❌ Change analysis failed:', error);
    }
  }

  async performDeepAnalysis() {
    try {
      logger.info('🔬 Performing deep document analysis...');
      
      // Analyze document content for insights
      const insights = {
        documentCounts: {},
        contentAnalysis: {},
        trends: {},
        timestamp: new Date().toISOString()
      };
      
      // Count documents by type and AMC
      for (const [amcKey, amcInfo] of this.documentRegistry) {
        insights.documentCounts[amcInfo.name] = {
          total: amcInfo.totalDocuments,
          byType: {}
        };
        
        for (const [docKey, doc] of Object.entries(amcInfo.documents)) {
          const type = doc.type;
          insights.documentCounts[amcInfo.name].byType[type] = 
            (insights.documentCounts[amcInfo.name].byType[type] || 0) + 1;
        }
      }
      
      // Save deep analysis
      const analysisPath = path.join(this.config.metadataPath, 'deep-analysis.json');
      await fs.writeFile(analysisPath, JSON.stringify(insights, null, 2));
      
      // Emit deep analysis event
      this.emit('deepAnalysis', insights);
      
      logger.info('🔬 Deep analysis completed');
      
    } catch (error) {
      logger.error('❌ Deep analysis failed:', error);
    }
  }

  // Helper methods
  generateAMCKey(amcName) {
    return amcName.toLowerCase().replace(/[^a-z0-9]/g, '_');
  }

  generateDocumentKey(url) {
    return crypto.createHash('md5').update(url).digest('hex');
  }

  // Public API methods
  async getAMCDocuments(amcName, documentType = null) {
    const amcKey = this.generateAMCKey(amcName);
    const amcInfo = this.documentRegistry.get(amcKey);
    
    if (!amcInfo) return [];
    
    const documents = Object.values(amcInfo.documents);
    
    if (documentType) {
      return documents.filter(doc => doc.type === documentType);
    }
    
    return documents;
  }

  async getRecentChanges(hours = 24) {
    const cutoff = new Date(Date.now() - hours * 60 * 60 * 1000);
    const recentChanges = [];
    
    for (const [changeKey, changes] of this.changeHistory) {
      const recent = changes.filter(change => 
        new Date(change.timestamp) > cutoff
      );
      
      recentChanges.push(...recent.map(change => ({
        ...change,
        key: changeKey
      })));
    }
    
    return recentChanges.sort((a, b) => 
      new Date(b.timestamp) - new Date(a.timestamp)
    );
  }

  getMonitoringStats() {
    return {
      ...this.stats,
      totalAMCs: this.documentRegistry.size,
      totalDocuments: Array.from(this.documentRegistry.values())
        .reduce((sum, amc) => sum + amc.totalDocuments, 0),
      isInitialized: this.isInitialized
    };
  }

  async forceDocumentScan(amcName = null) {
    if (amcName) {
      const amcKey = this.generateAMCKey(amcName);
      const amcInfo = this.documentRegistry.get(amcKey);
      
      if (amcInfo) {
        await this.scanAMCDocuments(amcKey, amcInfo);
        await this.saveDocumentRegistry();
      }
    } else {
      await this.performFullDocumentScan();
    }
  }
}

module.exports = { AMCDocumentMonitor };
