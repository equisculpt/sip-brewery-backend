/**
 * 🛰️ NASA EARTHDATA SEARCH CLIENT
 * Integration with NASA's Earthdata Search for supply chain intelligence
 * Based on: https://github.com/nasa/earthdata-search/
 * 
 * @author ASI Engineering Team
 * @version 1.0.0
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');
const https = require('https');
const logger = require('../utils/logger');

class NASAEarthdataClient extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      baseUrl: 'https://cmr.earthdata.nasa.gov',
      searchUrl: 'https://cmr.earthdata.nasa.gov/search',
      downloadUrl: 'https://e4ftl01.cr.usgs.gov',
      
      // Free NASA Earthdata account required
      username: process.env.EARTHDATA_USERNAME,
      token: process.env.EARTHDATA_TOKEN,
      password: process.env.EARTHDATA_PASSWORD,
      
      maxResults: 100,
      timeout: 30000,
      retryAttempts: 3,
      
      dataPath: './data/nasa-earthdata',
      ...options
    };
    
    // NASA satellite collections relevant to supply chain
    this.collections = {
      modis: {
        terra: {
          id: 'C1000000240-LPDAAC_ECS', // MOD09A1 - Surface Reflectance
          description: 'MODIS Terra Surface Reflectance 8-Day',
          resolution: '500m',
          applications: ['Vegetation monitoring', 'Agricultural assessment']
        },
        aqua: {
          id: 'C1000000241-LPDAAC_ECS', // MYD09A1 - Surface Reflectance
          description: 'MODIS Aqua Surface Reflectance 8-Day',
          resolution: '500m',
          applications: ['Ocean monitoring', 'Atmospheric analysis']
        }
      },
      
      viirs: {
        snpp: {
          id: 'C1373412048-LPDAAC_ECS', // VNP09A1 - Surface Reflectance
          description: 'VIIRS Surface Reflectance 8-Day',
          resolution: '500m',
          applications: ['Land monitoring', 'Fire detection']
        }
      },
      
      landsat: {
        landsat8: {
          id: 'C1711961296-LPCLOUD', // Landsat 8 Collection 2
          description: 'Landsat 8 OLI/TIRS Collection 2 Level-2',
          resolution: '30m',
          applications: ['Industrial monitoring', 'Urban analysis']
        },
        landsat9: {
          id: 'C2021957657-LPCLOUD', // Landsat 9 Collection 2
          description: 'Landsat 9 OLI-2/TIRS-2 Collection 2 Level-2',
          resolution: '30m',
          applications: ['High-resolution monitoring', 'Change detection']
        }
      },
      
      aster: {
        gdem: {
          id: 'C1711961296-LPDAAC_ECS', // ASTER GDEM
          description: 'ASTER Global Digital Elevation Model',
          resolution: '30m',
          applications: ['Terrain analysis', 'Infrastructure planning']
        }
      },
      
      firms: {
        modis: {
          id: 'C1214470488-FIRMS', // MODIS Active Fire
          description: 'MODIS Active Fire Detections',
          resolution: '1km',
          applications: ['Fire monitoring', 'Environmental risk']
        },
        viirs: {
          id: 'C1214470533-FIRMS', // VIIRS Active Fire
          description: 'VIIRS Active Fire Detections',
          resolution: '375m',
          applications: ['High-resolution fire detection', 'Risk assessment']
        }
      }
    };
    
    // Indian regions of interest for supply chain monitoring
    this.indiaRegions = {
      nationalBounds: {
        name: 'India',
        bbox: [68.0, 8.0, 97.0, 37.0], // [west, south, east, north]
        polygon: 'POLYGON((68 8,97 8,97 37,68 37,68 8))'
      },
      
      economicZones: {
        westernCoast: {
          name: 'Western Economic Corridor',
          bbox: [72.0, 15.0, 77.0, 24.0],
          polygon: 'POLYGON((72 15,77 15,77 24,72 24,72 15))',
          industries: ['Chemicals', 'Textiles', 'Automotive']
        },
        
        easternCoast: {
          name: 'Eastern Economic Corridor',
          bbox: [85.0, 17.0, 90.0, 25.0],
          polygon: 'POLYGON((85 17,90 17,90 25,85 25,85 17))',
          industries: ['Steel', 'Mining', 'Ports']
        },
        
        southernPeninsula: {
          name: 'Southern Technology Corridor',
          bbox: [76.0, 8.0, 80.5, 17.0],
          polygon: 'POLYGON((76 8,80.5 8,80.5 17,76 17,76 8))',
          industries: ['IT', 'Pharmaceuticals', 'Aerospace']
        },
        
        northernPlains: {
          name: 'Northern Agricultural Belt',
          bbox: [74.0, 26.0, 82.0, 32.0],
          polygon: 'POLYGON((74 26,82 26,82 32,74 32,74 26))',
          industries: ['Agriculture', 'Food Processing', 'Textiles']
        }
      },
      
      criticalInfrastructure: {
        majorPorts: [
          { name: 'Mumbai Port Trust', bbox: [72.82, 18.92, 72.87, 18.97] },
          { name: 'JNPT Mumbai', bbox: [72.93, 18.93, 72.98, 18.98] },
          { name: 'Chennai Port', bbox: [80.25, 13.05, 80.35, 13.15] },
          { name: 'Kolkata Port', bbox: [88.20, 22.55, 88.30, 22.65] },
          { name: 'Visakhapatnam Port', bbox: [83.25, 17.68, 83.35, 17.78] },
          { name: 'Kandla Port', bbox: [70.15, 22.95, 70.25, 23.05] }
        ],
        
        industrialClusters: [
          { name: 'Mumbai-Pune Industrial Belt', bbox: [72.5, 18.0, 74.0, 19.5] },
          { name: 'Chennai-Bangalore Corridor', bbox: [77.0, 12.5, 80.5, 13.5] },
          { name: 'Delhi-NCR Industrial Area', bbox: [76.5, 28.0, 77.8, 29.0] },
          { name: 'Ahmedabad-Vadodara Belt', bbox: [72.0, 22.0, 73.5, 23.5] }
        ],
        
        agriculturalZones: [
          { name: 'Punjab Wheat Belt', bbox: [74.0, 30.0, 76.5, 32.0] },
          { name: 'Haryana Agricultural Zone', bbox: [75.0, 28.0, 77.5, 30.5] },
          { name: 'Maharashtra Sugar Belt', bbox: [73.0, 17.0, 76.0, 20.0] },
          { name: 'Andhra Pradesh Rice Belt', bbox: [78.0, 14.0, 82.0, 18.0] }
        ]
      }
    };
    
    this.searchCache = new Map();
    this.downloadQueue = [];
    this.processingQueue = [];
    
    this.stats = {
      searchesPerformed: 0,
      granulesFound: 0,
      downloadsCompleted: 0,
      processingCompleted: 0,
      lastUpdate: null
    };
  }

  async initialize() {
    try {
      logger.info('🛰️ Initializing NASA Earthdata Client...');
      
      await this.createDirectories();
      await this.validateCredentials();
      await this.loadSearchCache();
      
      logger.info('✅ NASA Earthdata Client initialized');
      
    } catch (error) {
      logger.error('❌ NASA Earthdata Client initialization failed:', error);
      throw error;
    }
  }

  async createDirectories() {
    const dirs = [
      this.config.dataPath,
      path.join(this.config.dataPath, 'raw'),
      path.join(this.config.dataPath, 'processed'),
      path.join(this.config.dataPath, 'cache'),
      path.join(this.config.dataPath, 'metadata'),
      path.join(this.config.dataPath, 'analysis')
    ];
    
    for (const dir of dirs) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') throw error;
      }
    }
  }

  async validateCredentials() {
    if (!this.config.username || (!this.config.token && !this.config.password)) {
      logger.warn('⚠️ NASA Earthdata credentials not provided. Some features may be limited.');
      return false;
    }
    
    try {
      // Test authentication with a simple search
      const testSearch = await this.searchGranules({
        collection: this.collections.modis.terra.id,
        bbox: [72.8, 18.9, 72.9, 19.0],
        temporal: ['2024-01-01T00:00:00Z', '2024-01-02T23:59:59Z'],
        limit: 1
      });
      
      logger.info('✅ NASA Earthdata credentials validated');
      return true;
      
    } catch (error) {
      logger.error('❌ NASA Earthdata credential validation failed:', error);
      return false;
    }
  }

  async loadSearchCache() {
    try {
      const cachePath = path.join(this.config.dataPath, 'cache', 'search-cache.json');
      const cacheData = await fs.readFile(cachePath, 'utf8');
      const parsedCache = JSON.parse(cacheData);
      
      for (const [key, value] of Object.entries(parsedCache)) {
        this.searchCache.set(key, value);
      }
      
      logger.info(`📋 Loaded ${this.searchCache.size} cached searches`);
      
    } catch (error) {
      logger.debug('No existing search cache found, starting fresh');
    }
  }

  async saveSearchCache() {
    try {
      const cachePath = path.join(this.config.dataPath, 'cache', 'search-cache.json');
      const cacheData = Object.fromEntries(this.searchCache);
      await fs.writeFile(cachePath, JSON.stringify(cacheData, null, 2));
    } catch (error) {
      logger.error('❌ Failed to save search cache:', error);
    }
  }

  // Core search functionality
  async searchGranules(params) {
    const {
      collection,
      bbox,
      temporal,
      polygon,
      limit = this.config.maxResults,
      offset = 0
    } = params;
    
    // Generate cache key
    const cacheKey = this.generateCacheKey(params);
    
    // Check cache first
    if (this.searchCache.has(cacheKey)) {
      logger.debug('📋 Returning cached search results');
      return this.searchCache.get(cacheKey);
    }
    
    try {
      logger.debug(`🔍 Searching NASA Earthdata for collection: ${collection}`);
      
      // Build search URL
      const searchUrl = this.buildSearchUrl(params);
      
      // Perform search
      const response = await this.makeRequest(searchUrl);
      const results = await response.json();
      
      // Process results
      const processedResults = this.processSearchResults(results);
      
      // Cache results
      this.searchCache.set(cacheKey, processedResults);
      await this.saveSearchCache();
      
      this.stats.searchesPerformed++;
      this.stats.granulesFound += processedResults.granules.length;
      this.stats.lastUpdate = new Date().toISOString();
      
      this.emit('searchCompleted', {
        collection,
        granulesFound: processedResults.granules.length,
        searchParams: params
      });
      
      return processedResults;
      
    } catch (error) {
      logger.error('❌ NASA Earthdata search failed:', error);
      throw error;
    }
  }

  buildSearchUrl(params) {
    const {
      collection,
      bbox,
      temporal,
      polygon,
      limit,
      offset
    } = params;
    
    const searchParams = new URLSearchParams();
    
    // Collection ID
    searchParams.append('collection_concept_id', collection);
    
    // Spatial filter
    if (bbox) {
      searchParams.append('bounding_box', bbox.join(','));
    } else if (polygon) {
      searchParams.append('polygon', polygon);
    }
    
    // Temporal filter
    if (temporal && temporal.length === 2) {
      searchParams.append('temporal', `${temporal[0]},${temporal[1]}`);
    }
    
    // Result parameters
    searchParams.append('page_size', limit.toString());
    searchParams.append('offset', offset.toString());
    searchParams.append('pretty', 'true');
    
    return `${this.config.searchUrl}/granules.json?${searchParams.toString()}`;
  }

  async makeRequest(url, options = {}) {
    const requestOptions = {
      method: 'GET',
      timeout: this.config.timeout,
      ...options
    };
    
    // Add authentication - prefer token over password
    if (this.config.token) {
      requestOptions.headers = {
        'Authorization': `Bearer ${this.config.token}`,
        ...requestOptions.headers
      };
    } else if (this.config.username && this.config.password) {
      const auth = Buffer.from(`${this.config.username}:${this.config.password}`).toString('base64');
      requestOptions.headers = {
        'Authorization': `Basic ${auth}`,
        ...requestOptions.headers
      };
    }
    
    let lastError;
    
    for (let attempt = 1; attempt <= this.config.retryAttempts; attempt++) {
      try {
        const response = await fetch(url, requestOptions);
        
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        return response;
        
      } catch (error) {
        lastError = error;
        logger.warn(`⚠️ Request attempt ${attempt} failed: ${error.message}`);
        
        if (attempt < this.config.retryAttempts) {
          const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
          await new Promise(resolve => setTimeout(resolve, delay));
        }
      }
    }
    
    throw lastError;
  }

  processSearchResults(results) {
    const granules = results.feed?.entry || [];
    
    return {
      totalResults: results.feed?.opensearch$totalResults?.value || 0,
      granules: granules.map(granule => ({
        id: granule.id,
        title: granule.title,
        summary: granule.summary,
        timeStart: granule.time_start,
        timeEnd: granule.time_end,
        coordinates: this.extractCoordinates(granule),
        downloadUrls: this.extractDownloadUrls(granule),
        metadata: {
          producer: granule.producer_granule_id,
          collection: granule.collection_concept_id,
          size: granule.granule_size,
          format: granule.original_format
        }
      })),
      searchTime: new Date().toISOString()
    };
  }

  extractCoordinates(granule) {
    try {
      const polygons = granule.polygons;
      if (polygons && polygons.length > 0) {
        return polygons[0].map(coord => [parseFloat(coord[1]), parseFloat(coord[0])]);
      }
      return null;
    } catch (error) {
      return null;
    }
  }

  extractDownloadUrls(granule) {
    const urls = [];
    
    if (granule.links) {
      granule.links.forEach(link => {
        if (link.rel === 'http://esipfed.org/ns/fedsearch/1.1/data#') {
          urls.push({
            url: link.href,
            type: 'data',
            title: link.title || 'Data file'
          });
        }
      });
    }
    
    return urls;
  }

  generateCacheKey(params) {
    const keyData = {
      collection: params.collection,
      bbox: params.bbox,
      temporal: params.temporal,
      polygon: params.polygon,
      limit: params.limit
    };
    
    return Buffer.from(JSON.stringify(keyData)).toString('base64');
  }

  // Supply chain specific search methods
  async searchPortActivity(portName, days = 7) {
    const port = this.indiaRegions.criticalInfrastructure.majorPorts
      .find(p => p.name.toLowerCase().includes(portName.toLowerCase()));
    
    if (!port) {
      throw new Error(`Port not found: ${portName}`);
    }
    
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000);
    
    // Search for high-resolution optical data (Landsat)
    const opticalData = await this.searchGranules({
      collection: this.collections.landsat.landsat8.id,
      bbox: port.bbox,
      temporal: [startDate.toISOString(), endDate.toISOString()],
      limit: 50
    });
    
    return {
      port: port.name,
      bbox: port.bbox,
      timeRange: { start: startDate, end: endDate },
      opticalGranules: opticalData.granules,
      analysis: this.analyzePortData(opticalData.granules)
    };
  }

  async searchIndustrialActivity(region, days = 30) {
    const cluster = this.indiaRegions.criticalInfrastructure.industrialClusters
      .find(c => c.name.toLowerCase().includes(region.toLowerCase()));
    
    if (!cluster) {
      throw new Error(`Industrial cluster not found: ${region}`);
    }
    
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000);
    
    // Search for thermal and optical data
    const [thermalData, opticalData] = await Promise.all([
      this.searchGranules({
        collection: this.collections.modis.terra.id,
        bbox: cluster.bbox,
        temporal: [startDate.toISOString(), endDate.toISOString()],
        limit: 100
      }),
      this.searchGranules({
        collection: this.collections.landsat.landsat8.id,
        bbox: cluster.bbox,
        temporal: [startDate.toISOString(), endDate.toISOString()],
        limit: 50
      })
    ]);
    
    return {
      cluster: cluster.name,
      bbox: cluster.bbox,
      timeRange: { start: startDate, end: endDate },
      thermalGranules: thermalData.granules,
      opticalGranules: opticalData.granules,
      analysis: this.analyzeIndustrialData(thermalData.granules, opticalData.granules)
    };
  }

  async searchAgriculturalActivity(region, days = 60) {
    const zone = this.indiaRegions.criticalInfrastructure.agriculturalZones
      .find(z => z.name.toLowerCase().includes(region.toLowerCase()));
    
    if (!zone) {
      throw new Error(`Agricultural zone not found: ${region}`);
    }
    
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000);
    
    // Search for vegetation monitoring data
    const vegetationData = await this.searchGranules({
      collection: this.collections.modis.terra.id,
      bbox: zone.bbox,
      temporal: [startDate.toISOString(), endDate.toISOString()],
      limit: 100
    });
    
    return {
      zone: zone.name,
      bbox: zone.bbox,
      timeRange: { start: startDate, end: endDate },
      vegetationGranules: vegetationData.granules,
      analysis: this.analyzeAgriculturalData(vegetationData.granules)
    };
  }

  async searchEnvironmentalRisks(region = 'national', days = 7) {
    const regionData = region === 'national' ? 
      this.indiaRegions.nationalBounds : 
      this.indiaRegions.economicZones[region];
    
    if (!regionData) {
      throw new Error(`Region not found: ${region}`);
    }
    
    const endDate = new Date();
    const startDate = new Date(endDate.getTime() - days * 24 * 60 * 60 * 1000);
    
    // Search for fire and atmospheric data
    const [fireData, atmosphericData] = await Promise.all([
      this.searchGranules({
        collection: this.collections.firms.viirs.id,
        bbox: regionData.bbox,
        temporal: [startDate.toISOString(), endDate.toISOString()],
        limit: 200
      }),
      this.searchGranules({
        collection: this.collections.modis.aqua.id,
        bbox: regionData.bbox,
        temporal: [startDate.toISOString(), endDate.toISOString()],
        limit: 100
      })
    ]);
    
    return {
      region: regionData.name,
      bbox: regionData.bbox,
      timeRange: { start: startDate, end: endDate },
      fireGranules: fireData.granules,
      atmosphericGranules: atmosphericData.granules,
      analysis: this.analyzeEnvironmentalRisks(fireData.granules, atmosphericData.granules)
    };
  }

  // Analysis methods
  analyzePortData(granules) {
    return {
      dataAvailability: granules.length > 0 ? 'good' : 'limited',
      temporalCoverage: this.calculateTemporalCoverage(granules),
      spatialCoverage: this.calculateSpatialCoverage(granules),
      recommendedAnalysis: [
        'Ship detection using optical imagery',
        'Port infrastructure monitoring',
        'Cargo area utilization assessment',
        'Vessel traffic pattern analysis'
      ]
    };
  }

  analyzeIndustrialData(thermalGranules, opticalGranules) {
    return {
      thermalDataAvailability: thermalGranules.length > 0 ? 'good' : 'limited',
      opticalDataAvailability: opticalGranules.length > 0 ? 'good' : 'limited',
      temporalCoverage: this.calculateTemporalCoverage([...thermalGranules, ...opticalGranules]),
      recommendedAnalysis: [
        'Thermal activity monitoring for facility utilization',
        'Industrial expansion detection',
        'Environmental impact assessment',
        'Production capacity estimation'
      ]
    };
  }

  analyzeAgriculturalData(granules) {
    return {
      dataAvailability: granules.length > 0 ? 'good' : 'limited',
      temporalCoverage: this.calculateTemporalCoverage(granules),
      recommendedAnalysis: [
        'NDVI calculation for crop health',
        'Irrigation pattern monitoring',
        'Harvest timing prediction',
        'Yield estimation modeling'
      ]
    };
  }

  analyzeEnvironmentalRisks(fireGranules, atmosphericGranules) {
    return {
      fireDataAvailability: fireGranules.length > 0 ? 'good' : 'limited',
      atmosphericDataAvailability: atmosphericGranules.length > 0 ? 'good' : 'limited',
      riskLevel: fireGranules.length > 10 ? 'high' : fireGranules.length > 5 ? 'medium' : 'low',
      recommendedAnalysis: [
        'Fire risk assessment and mapping',
        'Air quality monitoring',
        'Weather pattern analysis',
        'Supply chain disruption prediction'
      ]
    };
  }

  calculateTemporalCoverage(granules) {
    if (granules.length === 0) return null;
    
    const dates = granules.map(g => new Date(g.timeStart)).sort();
    const startDate = dates[0];
    const endDate = dates[dates.length - 1];
    const daysCovered = (endDate - startDate) / (1000 * 60 * 60 * 24);
    
    return {
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
      daysCovered: Math.round(daysCovered),
      granulesCount: granules.length
    };
  }

  calculateSpatialCoverage(granules) {
    if (granules.length === 0) return null;
    
    const validGranules = granules.filter(g => g.coordinates);
    
    return {
      granulesWithCoordinates: validGranules.length,
      totalGranules: granules.length,
      coveragePercentage: Math.round((validGranules.length / granules.length) * 100)
    };
  }

  getStats() {
    return {
      ...this.stats,
      currentMetrics: {
        cachedSearches: this.searchCache.size,
        collectionsAvailable: Object.keys(this.collections).length,
        regionsConfigured: Object.keys(this.indiaRegions).length,
        queuedDownloads: this.downloadQueue.length,
        queuedProcessing: this.processingQueue.length
      }
    };
  }

  async generateReport() {
    const report = {
      timestamp: new Date().toISOString(),
      summary: this.getStats(),
      collections: this.collections,
      regions: this.indiaRegions,
      recentSearches: Array.from(this.searchCache.entries()).slice(-10)
    };
    
    const reportPath = path.join(this.config.dataPath, 'reports', `earthdata-report-${Date.now()}.json`);
    await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
    
    return report;
  }
}

module.exports = NASAEarthdataClient;
