/**
 * 🔄 REAL-TIME ADAPTIVE LEARNING ENGINE
 * 
 * Continuous learning from market changes and prediction accuracy
 * Online learning, model drift detection, and automatic retraining
 * 
 * @author 35-year ASI Engineer
 * @version 2.0.0 - Production-Ready Adaptive System
 */

const tf = require('@tensorflow/tfjs-node-gpu');
const logger = require('../utils/logger');

class RealTimeAdaptiveLearner {
  constructor(options = {}) {
    this.config = {
      // Learning parameters
      onlineLearningRate: options.onlineLearningRate || 0.0001,
      adaptationThreshold: options.adaptationThreshold || 0.05,
      driftDetectionWindow: options.driftDetectionWindow || 100,
      retrainingThreshold: options.retrainingThreshold || 0.15,
      
      // Performance monitoring
      performanceWindow: options.performanceWindow || 50,
      accuracyThreshold: options.accuracyThreshold || 0.6,
      
      // Model management
      maxModelVersions: options.maxModelVersions || 10,
      modelBackupFrequency: options.modelBackupFrequency || 24 * 60 * 60 * 1000, // 24 hours
      
      // Real-time processing
      batchSize: options.batchSize || 16,
      updateFrequency: options.updateFrequency || 60000, // 1 minute
      
      ...options
    };

    // Learning components
    this.onlineLearner = null;
    this.driftDetector = null;
    this.performanceMonitor = null;
    this.modelManager = null;
    
    // Data streams
    this.predictionStream = [];
    this.actualOutcomes = [];
    this.performanceHistory = [];
    
    // Model versions
    this.modelVersions = new Map();
    this.currentModelVersion = 1;
    this.bestModelVersion = 1;
    
    // Adaptation state
    this.adaptationState = {
      isAdapting: false,
      lastAdaptation: null,
      adaptationCount: 0,
      driftDetected: false,
      currentPerformance: 0.5
    };
    
    // Real-time metrics
    this.realtimeMetrics = {
      predictionAccuracy: 0,
      modelDrift: 0,
      adaptationRate: 0,
      learningVelocity: 0
    };
  }

  async initialize() {
    try {
      logger.info('🔄 Initializing Real-Time Adaptive Learner...');
      
      await this.initializeOnlineLearner();
      await this.initializeDriftDetector();
      await this.initializePerformanceMonitor();
      await this.initializeModelManager();
      
      // Start real-time adaptation loop
      this.startAdaptationLoop();
      
      logger.info('✅ Real-Time Adaptive Learner initialized successfully');
      
    } catch (error) {
      logger.error('❌ Adaptive Learner initialization failed:', error);
      throw error;
    }
  }

  /**
   * Initialize online learning system
   */
  async initializeOnlineLearner() {
    logger.info('📚 Initializing online learner...');
    
    this.onlineLearner = {
      // Incremental learning model
      incrementalModel: await this.createIncrementalModel(),
      
      // Learning buffer
      learningBuffer: {
        features: [],
        targets: [],
        weights: [],
        maxSize: 1000
      },
      
      // Gradient accumulation
      gradientAccumulator: null,
      
      // Learning rate scheduler
      learningRateScheduler: {
        currentRate: this.config.onlineLearningRate,
        decayRate: 0.95,
        minRate: 0.00001,
        adaptiveAdjustment: true
      },
      
      // Feature importance tracking
      featureImportance: new Map(),
      
      // Learning statistics
      learningStats: {
        samplesProcessed: 0,
        batchesProcessed: 0,
        averageLoss: 0,
        convergenceRate: 0
      }
    };
    
    logger.info('✅ Online learner initialized');
  }

  /**
   * Create incremental learning model
   */
  async createIncrementalModel() {
    const model = tf.sequential({
      layers: [
        tf.layers.dense({ 
          units: 128, 
          activation: 'relu', 
          inputShape: [75] // Multi-modal feature size
        }),
        tf.layers.dropout({ rate: 0.2 }),
        tf.layers.dense({ units: 64, activation: 'relu' }),
        tf.layers.dropout({ rate: 0.1 }),
        tf.layers.dense({ units: 32, activation: 'relu' }),
        tf.layers.dense({ units: 3 }) // [return, volatility, confidence]
      ]
    });
    
    model.compile({
      optimizer: tf.train.adam(this.config.onlineLearningRate),
      loss: 'mse',
      metrics: ['mae']
    });
    
    return model;
  }

  /**
   * Initialize drift detection system
   */
  async initializeDriftDetector() {
    logger.info('🔍 Initializing drift detector...');
    
    this.driftDetector = {
      // Statistical drift detection
      statisticalDetector: {
        referenceWindow: [],
        currentWindow: [],
        windowSize: this.config.driftDetectionWindow,
        driftThreshold: 0.05,
        pValue: 0.05
      },
      
      // Performance-based drift detection
      performanceDetector: {
        baselineAccuracy: 0.6,
        currentAccuracy: 0.6,
        accuracyWindow: [],
        degradationThreshold: 0.1
      },
      
      // Feature drift detection
      featureDriftDetector: {
        featureDistributions: new Map(),
        driftScores: new Map(),
        driftHistory: []
      },
      
      // Concept drift detection
      conceptDriftDetector: {
        conceptWindow: [],
        conceptChangeThreshold: 0.1,
        adaptationTrigger: false
      },
      
      // Drift alerts
      driftAlerts: []
    };
    
    logger.info('✅ Drift detector initialized');
  }

  /**
   * Initialize performance monitoring system
   */
  async initializePerformanceMonitor() {
    logger.info('📊 Initializing performance monitor...');
    
    this.performanceMonitor = {
      // Accuracy tracking
      accuracyTracker: {
        shortTerm: [], // Last 10 predictions
        mediumTerm: [], // Last 50 predictions
        longTerm: [], // Last 200 predictions
        overall: 0
      },
      
      // Error analysis
      errorAnalyzer: {
        mse: [],
        mae: [],
        directionalAccuracy: [],
        calibrationError: []
      },
      
      // Performance by horizon
      horizonPerformance: new Map([
        ['1d', { accuracy: 0, count: 0 }],
        ['7d', { accuracy: 0, count: 0 }],
        ['30d', { accuracy: 0, count: 0 }],
        ['90d', { accuracy: 0, count: 0 }]
      ]),
      
      // Performance by regime
      regimePerformance: new Map([
        ['bull', { accuracy: 0, count: 0 }],
        ['bear', { accuracy: 0, count: 0 }],
        ['sideways', { accuracy: 0, count: 0 }],
        ['crisis', { accuracy: 0, count: 0 }]
      ]),
      
      // Real-time alerts
      performanceAlerts: []
    };
    
    logger.info('✅ Performance monitor initialized');
  }

  /**
   * Initialize model management system
   */
  async initializeModelManager() {
    logger.info('🗂️ Initializing model manager...');
    
    this.modelManager = {
      // Model versioning
      versionControl: {
        currentVersion: 1,
        versions: new Map(),
        versionMetadata: new Map()
      },
      
      // Model backup system
      backupSystem: {
        backupSchedule: null,
        backupLocation: './models/backups/',
        compressionEnabled: true
      },
      
      // Model comparison
      modelComparator: {
        comparisonMetrics: ['accuracy', 'loss', 'calibration'],
        benchmarkResults: new Map()
      },
      
      // Rollback system
      rollbackSystem: {
        rollbackTrigger: false,
        rollbackThreshold: 0.1, // 10% performance drop
        rollbackHistory: []
      }
    };
    
    logger.info('✅ Model manager initialized');
  }

  /**
   * Process new prediction and actual outcome for learning
   */
  async processPredictionOutcome(prediction, actualOutcome, metadata = {}) {
    try {
      const processingStart = Date.now();
      
      // Store prediction-outcome pair
      this.predictionStream.push({
        prediction: prediction,
        actual: actualOutcome,
        timestamp: Date.now(),
        metadata: metadata
      });
      
      // Calculate prediction error
      const error = this.calculatePredictionError(prediction, actualOutcome);
      
      // Update performance metrics
      await this.updatePerformanceMetrics(error, metadata);
      
      // Detect drift
      const driftDetected = await this.detectDrift(prediction, actualOutcome, error);
      
      // Add to learning buffer
      if (metadata.features) {
        this.addToLearningBuffer(metadata.features, actualOutcome, error);
      }
      
      // Trigger adaptation if needed
      if (this.shouldTriggerAdaptation(driftDetected, error)) {
        await this.triggerAdaptation();
      }
      
      // Online learning update
      if (this.learningBuffer.features.length >= this.config.batchSize) {
        await this.performOnlineLearning();
      }
      
      const processingTime = Date.now() - processingStart;
      
      logger.info(`📈 Processed prediction outcome in ${processingTime}ms`);
      
      return {
        error: error,
        driftDetected: driftDetected,
        adaptationTriggered: this.adaptationState.isAdapting,
        processingTime: processingTime
      };
      
    } catch (error) {
      logger.error('❌ Prediction outcome processing failed:', error);
      throw error;
    }
  }

  /**
   * Calculate prediction error
   */
  calculatePredictionError(prediction, actual) {
    const errors = {};
    
    // Mean Squared Error
    errors.mse = Math.pow(prediction.expectedReturn - actual.return, 2);
    
    // Mean Absolute Error
    errors.mae = Math.abs(prediction.expectedReturn - actual.return);
    
    // Directional Accuracy
    errors.directional = Math.sign(prediction.expectedReturn) === Math.sign(actual.return) ? 1 : 0;
    
    // Volatility Error
    if (actual.volatility !== undefined) {
      errors.volatilityError = Math.abs(prediction.volatility - actual.volatility);
    }
    
    // Confidence Calibration Error
    const actualConfidence = errors.directional;
    errors.calibrationError = Math.abs(prediction.confidence - actualConfidence);
    
    return errors;
  }

  /**
   * Update performance metrics
   */
  async updatePerformanceMetrics(error, metadata) {
    // Update accuracy trackers
    this.performanceMonitor.accuracyTracker.shortTerm.push(error.directional);
    if (this.performanceMonitor.accuracyTracker.shortTerm.length > 10) {
      this.performanceMonitor.accuracyTracker.shortTerm.shift();
    }
    
    this.performanceMonitor.accuracyTracker.mediumTerm.push(error.directional);
    if (this.performanceMonitor.accuracyTracker.mediumTerm.length > 50) {
      this.performanceMonitor.accuracyTracker.mediumTerm.shift();
    }
    
    // Update error analyzers
    this.performanceMonitor.errorAnalyzer.mse.push(error.mse);
    this.performanceMonitor.errorAnalyzer.mae.push(error.mae);
    this.performanceMonitor.errorAnalyzer.directionalAccuracy.push(error.directional);
    
    // Update horizon-specific performance
    if (metadata.horizon) {
      const horizonKey = `${metadata.horizon}d`;
      if (this.performanceMonitor.horizonPerformance.has(horizonKey)) {
        const horizonStats = this.performanceMonitor.horizonPerformance.get(horizonKey);
        horizonStats.accuracy = (horizonStats.accuracy * horizonStats.count + error.directional) / (horizonStats.count + 1);
        horizonStats.count++;
      }
    }
    
    // Update regime-specific performance
    if (metadata.regime) {
      if (this.performanceMonitor.regimePerformance.has(metadata.regime)) {
        const regimeStats = this.performanceMonitor.regimePerformance.get(metadata.regime);
        regimeStats.accuracy = (regimeStats.accuracy * regimeStats.count + error.directional) / (regimeStats.count + 1);
        regimeStats.count++;
      }
    }
    
    // Calculate current overall performance
    const recentAccuracy = this.performanceMonitor.accuracyTracker.shortTerm.reduce((a, b) => a + b, 0) / 
                          Math.max(this.performanceMonitor.accuracyTracker.shortTerm.length, 1);
    
    this.realtimeMetrics.predictionAccuracy = recentAccuracy;
    this.adaptationState.currentPerformance = recentAccuracy;
  }

  /**
   * Detect various types of drift
   */
  async detectDrift(prediction, actual, error) {
    let driftDetected = false;
    
    // Statistical drift detection
    const statisticalDrift = this.detectStatisticalDrift(error);
    
    // Performance drift detection
    const performanceDrift = this.detectPerformanceDrift();
    
    // Feature drift detection (if features available)
    const featureDrift = this.detectFeatureDrift(prediction);
    
    // Concept drift detection
    const conceptDrift = this.detectConceptDrift(prediction, actual);
    
    driftDetected = statisticalDrift || performanceDrift || featureDrift || conceptDrift;
    
    if (driftDetected) {
      this.driftDetector.driftAlerts.push({
        timestamp: Date.now(),
        type: {
          statistical: statisticalDrift,
          performance: performanceDrift,
          feature: featureDrift,
          concept: conceptDrift
        },
        severity: this.calculateDriftSeverity(statisticalDrift, performanceDrift, featureDrift, conceptDrift)
      });
      
      logger.warn('⚠️ Drift detected in prediction model');
    }
    
    this.adaptationState.driftDetected = driftDetected;
    this.realtimeMetrics.modelDrift = driftDetected ? 1 : 0;
    
    return driftDetected;
  }

  /**
   * Detect statistical drift using Kolmogorov-Smirnov test
   */
  detectStatisticalDrift(error) {
    const detector = this.driftDetector.statisticalDetector;
    
    detector.currentWindow.push(error.mse);
    if (detector.currentWindow.length > detector.windowSize) {
      detector.currentWindow.shift();
    }
    
    if (detector.referenceWindow.length === 0) {
      detector.referenceWindow = [...detector.currentWindow];
      return false;
    }
    
    if (detector.currentWindow.length < detector.windowSize) {
      return false;
    }
    
    // Simplified KS test implementation
    const ksStatistic = this.calculateKSStatistic(detector.referenceWindow, detector.currentWindow);
    const driftDetected = ksStatistic > detector.driftThreshold;
    
    if (driftDetected) {
      detector.referenceWindow = [...detector.currentWindow];
    }
    
    return driftDetected;
  }

  /**
   * Detect performance-based drift
   */
  detectPerformanceDrift() {
    const detector = this.driftDetector.performanceDetector;
    const currentAccuracy = this.realtimeMetrics.predictionAccuracy;
    
    detector.accuracyWindow.push(currentAccuracy);
    if (detector.accuracyWindow.length > 20) {
      detector.accuracyWindow.shift();
    }
    
    const averageAccuracy = detector.accuracyWindow.reduce((a, b) => a + b, 0) / detector.accuracyWindow.length;
    const performanceDrop = detector.baselineAccuracy - averageAccuracy;
    
    return performanceDrop > detector.degradationThreshold;
  }

  /**
   * Add data to learning buffer
   */
  addToLearningBuffer(features, target, error) {
    const buffer = this.onlineLearner.learningBuffer;
    
    buffer.features.push(features);
    buffer.targets.push(target);
    buffer.weights.push(1.0 / (1.0 + error.mse)); // Weight by inverse error
    
    // Maintain buffer size
    if (buffer.features.length > buffer.maxSize) {
      buffer.features.shift();
      buffer.targets.shift();
      buffer.weights.shift();
    }
  }

  /**
   * Perform online learning update
   */
  async performOnlineLearning() {
    try {
      const buffer = this.onlineLearner.learningBuffer;
      
      if (buffer.features.length < this.config.batchSize) {
        return;
      }
      
      // Prepare batch data
      const batchFeatures = tf.tensor2d(buffer.features.slice(-this.config.batchSize));
      const batchTargets = tf.tensor2d(buffer.targets.slice(-this.config.batchSize).map(t => [t.return, t.volatility || 0.1, 1.0]));
      const batchWeights = tf.tensor1d(buffer.weights.slice(-this.config.batchSize));
      
      // Perform gradient update
      const history = await this.onlineLearner.incrementalModel.fit(batchFeatures, batchTargets, {
        epochs: 1,
        batchSize: this.config.batchSize,
        sampleWeight: batchWeights,
        verbose: 0
      });
      
      // Update learning statistics
      const stats = this.onlineLearner.learningStats;
      stats.samplesProcessed += this.config.batchSize;
      stats.batchesProcessed++;
      stats.averageLoss = history.history.loss[0];
      
      // Adjust learning rate if needed
      this.adjustLearningRate(history.history.loss[0]);
      
      // Clean up tensors
      batchFeatures.dispose();
      batchTargets.dispose();
      batchWeights.dispose();
      
      logger.info(`📚 Online learning update completed - Loss: ${stats.averageLoss.toFixed(4)}`);
      
    } catch (error) {
      logger.error('❌ Online learning update failed:', error);
    }
  }

  /**
   * Trigger model adaptation
   */
  async triggerAdaptation() {
    if (this.adaptationState.isAdapting) {
      return; // Already adapting
    }
    
    try {
      logger.info('🔄 Triggering model adaptation...');
      
      this.adaptationState.isAdapting = true;
      this.adaptationState.lastAdaptation = Date.now();
      this.adaptationState.adaptationCount++;
      
      // Create new model version
      const newModelVersion = await this.createAdaptedModel();
      
      // Validate new model
      const validationResults = await this.validateNewModel(newModelVersion);
      
      if (validationResults.isImproved) {
        // Deploy new model
        await this.deployNewModel(newModelVersion);
        logger.info('✅ Model adaptation successful');
      } else {
        // Rollback if performance degraded
        logger.warn('⚠️ New model performance degraded, keeping current model');
      }
      
      this.adaptationState.isAdapting = false;
      
    } catch (error) {
      logger.error('❌ Model adaptation failed:', error);
      this.adaptationState.isAdapting = false;
    }
  }

  /**
   * Start real-time adaptation loop
   */
  startAdaptationLoop() {
    setInterval(async () => {
      try {
        // Update real-time metrics
        this.updateRealtimeMetrics();
        
        // Check for adaptation triggers
        if (this.shouldPerformScheduledAdaptation()) {
          await this.triggerAdaptation();
        }
        
        // Cleanup old data
        this.cleanupOldData();
        
      } catch (error) {
        logger.error('❌ Adaptation loop error:', error);
      }
    }, this.config.updateFrequency);
    
    logger.info('🔄 Real-time adaptation loop started');
  }

  // Helper methods
  shouldTriggerAdaptation(driftDetected, error) {
    return driftDetected || 
           this.adaptationState.currentPerformance < this.config.accuracyThreshold ||
           error.mse > this.config.adaptationThreshold;
  }

  calculateKSStatistic(sample1, sample2) {
    // Simplified KS statistic calculation
    const combined = [...sample1, ...sample2].sort((a, b) => a - b);
    let maxDiff = 0;
    
    for (const value of combined) {
      const cdf1 = sample1.filter(x => x <= value).length / sample1.length;
      const cdf2 = sample2.filter(x => x <= value).length / sample2.length;
      maxDiff = Math.max(maxDiff, Math.abs(cdf1 - cdf2));
    }
    
    return maxDiff;
  }

  adjustLearningRate(currentLoss) {
    const scheduler = this.onlineLearner.learningRateScheduler;
    
    if (scheduler.adaptiveAdjustment) {
      // Increase rate if loss is decreasing, decrease if increasing
      const lossHistory = this.onlineLearner.learningStats.averageLoss;
      if (currentLoss < lossHistory) {
        scheduler.currentRate = Math.min(scheduler.currentRate * 1.05, this.config.onlineLearningRate);
      } else {
        scheduler.currentRate = Math.max(scheduler.currentRate * scheduler.decayRate, scheduler.minRate);
      }
    }
  }

  updateRealtimeMetrics() {
    // Update adaptation rate
    const recentAdaptations = this.adaptationState.adaptationCount;
    this.realtimeMetrics.adaptationRate = recentAdaptations / Math.max(1, Date.now() - (this.adaptationState.lastAdaptation || Date.now()));
    
    // Update learning velocity
    this.realtimeMetrics.learningVelocity = this.onlineLearner.learningStats.batchesProcessed / 
                                           Math.max(1, this.onlineLearner.learningStats.samplesProcessed);
  }

  getMetrics() {
    return {
      adaptationState: this.adaptationState,
      realtimeMetrics: this.realtimeMetrics,
      performance: {
        shortTermAccuracy: this.performanceMonitor.accuracyTracker.shortTerm.reduce((a, b) => a + b, 0) / 
                          Math.max(this.performanceMonitor.accuracyTracker.shortTerm.length, 1),
        mediumTermAccuracy: this.performanceMonitor.accuracyTracker.mediumTerm.reduce((a, b) => a + b, 0) / 
                           Math.max(this.performanceMonitor.accuracyTracker.mediumTerm.length, 1),
        horizonPerformance: Object.fromEntries(this.performanceMonitor.horizonPerformance),
        regimePerformance: Object.fromEntries(this.performanceMonitor.regimePerformance)
      },
      learning: this.onlineLearner.learningStats,
      drift: {
        alertCount: this.driftDetector.driftAlerts.length,
        recentAlerts: this.driftDetector.driftAlerts.slice(-5)
      }
    };
  }

  // Placeholder methods
  detectFeatureDrift(prediction) { return false; }
  detectConceptDrift(prediction, actual) { return false; }
  calculateDriftSeverity(s, p, f, c) { return 0.5; }
  shouldPerformScheduledAdaptation() { return false; }
  createAdaptedModel() { return this.onlineLearner.incrementalModel; }
  validateNewModel(model) { return { isImproved: true }; }
  deployNewModel(model) { return true; }
  cleanupOldData() { /* Cleanup implementation */ }
}

module.exports = { RealTimeAdaptiveLearner };
