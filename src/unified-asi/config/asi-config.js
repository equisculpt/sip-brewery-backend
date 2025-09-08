/**
 * 🚀 UNIFIED ASI CONFIGURATION
 * 
 * Complete configuration for unified Finance ASI system
 * Consolidates all AI/AGI/ASI settings and parameters
 * 
 * @author Universe-Class ASI Architect
 * @version 1.0.0 - Unified Finance ASI
 */

const path = require('path');

const ASI_CONFIG = {
  // System Information
  system: {
    name: 'Unified Finance ASI',
    version: '1.0.0',
    targetRating: 9.2,
    description: 'Complete Financial Artificial Superintelligence System',
    author: 'Universe-Class ASI Architect'
  },

  // Performance Targets
  performance: {
    responseTimeTarget: 500, // milliseconds
    accuracyTarget: 0.85, // 85%
    uptimeTarget: 0.999, // 99.9%
    throughputTarget: 1000, // requests per minute
    concurrentRequestsMax: 100
  },

  // Core ASI Engine Settings
  asiEngine: {
    complexityThresholds: {
      basic: 0.3,
      general: 0.6,
      super: 0.8,
      quantum: 0.95
    },
    qualityThreshold: 0.85,
    adaptiveLearning: true,
    maxRetries: 3,
    timeoutMs: 30000,
    enableCaching: true,
    enableMonitoring: true
  },

  // AI/ML Engine Settings
  aiEngines: {
    advancedML: {
      gpuMemoryLimit: 4096, // MB for NVIDIA 3060
      batchSize: 32,
      learningRate: 0.001,
      sequenceLength: 60, // days for LSTM
      validationSplit: 0.2,
      earlyStoppingPatience: 10,
      enableGPU: true
    },
    continuousLearning: {
      learningRate: 0.001,
      adaptationThreshold: 0.1,
      retrainingInterval: 24 * 60 * 60 * 1000, // 24 hours
      performanceThreshold: 0.8
    },
    mutualFundAnalyzer: {
      analysisDepth: 'comprehensive',
      historicalPeriod: 5, // years
      benchmarkIndex: 'NIFTY50',
      includeESG: true
    },
    realTimeData: {
      updateInterval: 1000, // 1 second
      maxSymbols: 1000,
      enableWebSocket: true,
      dataRetentionDays: 30
    },
    backtesting: {
      defaultPeriod: '5Y',
      benchmarkIndex: 'NIFTY50',
      transactionCosts: 0.001, // 0.1%
      slippageModel: 'linear'
    }
  },

  // Python ASI Integration
  pythonASI: {
    servicePort: 8001,
    healthCheckInterval: 30000, // 30 seconds
    requestTimeout: 30000, // 30 seconds
    maxRetries: 3,
    services: {
      integratedASI: {
        enabled: true,
        endpoint: '/integrated-asi'
      },
      financialLLM: {
        enabled: true,
        endpoint: '/financial-llm'
      },
      trillionFund: {
        enabled: true,
        endpoint: '/trillion-fund'
      },
      marketData: {
        enabled: true,
        endpoint: '/market-data'
      },
      predictor: {
        enabled: true,
        endpoint: '/predictor'
      }
    }
  },

  // Data Sources Configuration
  dataSources: {
    realTime: {
      primary: 'NSE',
      fallback: ['BSE', 'MoneyControl', 'Yahoo'],
      updateFrequency: 1000, // ms
      maxLatency: 5000 // ms
    },
    historical: {
      provider: 'NSE',
      maxHistory: '10Y',
      cacheExpiry: 24 * 60 * 60 * 1000 // 24 hours
    },
    alternative: {
      satellite: {
        enabled: true,
        providers: ['NASA', 'ESA', 'NOAA', 'ISRO']
      },
      sentiment: {
        enabled: true,
        sources: ['twitter', 'reddit', 'news']
      },
      economic: {
        enabled: true,
        indicators: ['google_trends', 'satellite_nightlights']
      }
    }
  },

  // Risk Management Settings
  riskManagement: {
    varConfidence: 0.95,
    cvarConfidence: 0.95,
    stressTestScenarios: ['market_crash', 'interest_rate_shock', 'currency_crisis'],
    maxPositionSize: 0.1, // 10% max position
    maxSectorExposure: 0.3, // 30% max sector
    liquidityRequirement: 0.05 // 5% cash
  },

  // Portfolio Analysis Settings
  portfolioAnalysis: {
    defaultTimeHorizon: 365, // days
    riskProfiles: ['conservative', 'moderate', 'aggressive'],
    analysisDepth: 'comprehensive',
    includeAlternatives: true,
    includePredictions: true,
    includeOptimization: true,
    benchmarks: {
      equity: 'NIFTY50',
      debt: 'NIFTY_10_YR_BENCHMARK',
      hybrid: 'CRISIL_HYBRID_FUND_INDEX'
    }
  },

  // Prediction Settings
  predictions: {
    maxHorizon: 90, // days
    confidenceLevel: 0.95,
    includeUncertainty: true,
    models: ['lstm', 'transformer', 'ensemble'],
    updateFrequency: 24 * 60 * 60 * 1000, // 24 hours
    accuracyThreshold: 0.8
  },

  // Optimization Settings
  optimization: {
    algorithms: ['quantum_inspired', 'genetic', 'gradient_descent'],
    objectives: ['max_sharpe', 'min_variance', 'max_return'],
    constraints: {
      maxWeight: 0.4, // 40% max weight
      minWeight: 0.01, // 1% min weight
      turnover: 0.5 // 50% max turnover
    },
    rebalanceFrequency: 'monthly'
  },

  // Monitoring and Logging
  monitoring: {
    healthCheckInterval: 10000, // 10 seconds
    performanceLogInterval: 60000, // 1 minute
    metricsRetentionDays: 30,
    alertThresholds: {
      responseTime: 1000, // ms
      errorRate: 0.05, // 5%
      accuracy: 0.8, // 80%
      uptime: 0.99 // 99%
    },
    enableDetailedLogging: true
  },

  // Caching Configuration
  caching: {
    redis: {
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      password: process.env.REDIS_PASSWORD || null,
      db: 0
    },
    ttl: {
      marketData: 60, // seconds
      predictions: 3600, // 1 hour
      analysis: 1800, // 30 minutes
      optimization: 7200 // 2 hours
    },
    maxMemoryCache: 1000 // items
  },

  // Security Settings
  security: {
    rateLimiting: {
      windowMs: 60000, // 1 minute
      maxRequests: 100
    },
    authentication: {
      required: false, // Set to true in production
      jwtSecret: process.env.JWT_SECRET || 'asi-secret-key'
    },
    encryption: {
      algorithm: 'aes-256-gcm',
      keyLength: 32
    }
  },

  // Database Configuration
  database: {
    primary: {
      type: 'postgresql',
      url: process.env.DATABASE_URL,
      pool: {
        min: 2,
        max: 10,
        idle: 10000
      }
    },
    cache: {
      type: 'redis',
      url: process.env.REDIS_URL
    },
    timeseries: {
      type: 'influxdb',
      url: process.env.INFLUXDB_URL
    }
  },

  // API Configuration
  api: {
    baseUrl: '/api/unified-asi',
    version: 'v1',
    documentation: {
      enabled: true,
      path: '/docs'
    },
    cors: {
      origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
      credentials: true
    },
    bodyLimit: '10mb',
    timeout: 30000
  },

  // Environment-specific Settings
  environments: {
    development: {
      logLevel: 'debug',
      enableMocks: true,
      pythonServiceRequired: false
    },
    staging: {
      logLevel: 'info',
      enableMocks: false,
      pythonServiceRequired: true
    },
    production: {
      logLevel: 'warn',
      enableMocks: false,
      pythonServiceRequired: true,
      enableSecurity: true,
      enableMonitoring: true
    }
  },

  // FINANCE-ONLY Feature Flags
  features: {
    // Core Financial Intelligence
    financialLLM: true,
    trillionFundASI: true,
    mutualFundAnalysis: true,
    
    // Portfolio & Risk Management
    quantumOptimization: true,
    advancedRiskModels: true,
    portfolioOptimization: true,
    riskAttribution: true,
    
    // Market Analysis & Prediction
    realTimePredictions: true,
    technicalAnalysis: true,
    fundamentalAnalysis: true,
    marketSentiment: true,
    
    // Financial Data & Intelligence
    alternativeData: true,
    satelliteData: true, // For economic indicators
    socialSentiment: true, // For market sentiment
    macroeconomicFactors: true,
    
    // Investment Strategy
    behavioralFinance: true,
    esgIntegration: true,
    quantitativeStrategies: true,
    backtesting: true,
    
    // Compliance & Regulation
    sebiCompliance: true,
    amfiCompliance: true,
    riskDisclosure: true
  },

  // File Paths
  paths: {
    root: path.join(__dirname, '../..'),
    asiComponents: path.join(__dirname, '../..', 'asi'),
    agiComponents: path.join(__dirname, '../..', 'agi'),
    aiComponents: path.join(__dirname, '../..', 'ai'),
    unifiedASI: path.join(__dirname, '..'),
    pythonASI: path.join(__dirname, '../python-asi'),
    models: path.join(__dirname, '../models'),
    data: path.join(__dirname, '../data'),
    logs: path.join(__dirname, '../logs')
  },

  // Service Registry
  services: {
    asiMaster: {
      enabled: true,
      priority: 1,
      healthEndpoint: '/health'
    },
    enhancedASI: {
      enabled: true,
      priority: 2,
      healthEndpoint: '/health'
    },
    portfolioAnalyzer: {
      enabled: true,
      priority: 3,
      healthEndpoint: '/health'
    },
    riskManager: {
      enabled: true,
      priority: 3,
      healthEndpoint: '/health'
    },
    predictionService: {
      enabled: true,
      priority: 4,
      healthEndpoint: '/health'
    },
    pythonBridge: {
      enabled: true,
      priority: 5,
      healthEndpoint: '/health',
      url: 'http://localhost:8001'
    }
  }
};

// Environment-specific overrides
const env = process.env.NODE_ENV || 'development';
const envConfig = ASI_CONFIG.environments[env] || {};

// Merge environment config
Object.keys(envConfig).forEach(key => {
  if (typeof envConfig[key] === 'object' && !Array.isArray(envConfig[key])) {
    ASI_CONFIG[key] = { ...ASI_CONFIG[key], ...envConfig[key] };
  } else {
    ASI_CONFIG[key] = envConfig[key];
  }
});

// Validation function
function validateConfig() {
  const errors = [];
  
  // Check required environment variables
  if (env === 'production') {
    if (!process.env.DATABASE_URL) {
      errors.push('DATABASE_URL is required in production');
    }
    if (!process.env.REDIS_URL) {
      errors.push('REDIS_URL is required in production');
    }
    if (!process.env.JWT_SECRET) {
      errors.push('JWT_SECRET is required in production');
    }
  }
  
  // Check performance targets
  if (ASI_CONFIG.performance.accuracyTarget < 0.8) {
    errors.push('Accuracy target must be at least 80% for ASI rating 9+');
  }
  
  if (ASI_CONFIG.performance.responseTimeTarget > 1000) {
    errors.push('Response time target must be under 1000ms for ASI rating 9+');
  }
  
  if (errors.length > 0) {
    throw new Error(`Configuration validation failed:\n${errors.join('\n')}`);
  }
  
  return true;
}

// Export configuration
module.exports = {
  ASI_CONFIG,
  validateConfig,
  
  // Helper functions
  getConfig: (path) => {
    return path.split('.').reduce((obj, key) => obj?.[key], ASI_CONFIG);
  },
  
  isFeatureEnabled: (feature) => {
    return ASI_CONFIG.features[feature] === true;
  },
  
  getServiceConfig: (serviceName) => {
    return ASI_CONFIG.services[serviceName];
  },
  
  getPythonServiceUrl: () => {
    return `http://localhost:${ASI_CONFIG.pythonASI.servicePort}`;
  },
  
  getEnvironment: () => {
    return env;
  },
  
  isDevelopment: () => {
    return env === 'development';
  },
  
  isProduction: () => {
    return env === 'production';
  }
};
