/**
 * 📊 PERFORMANCE METRICS AND EVALUATION DASHBOARD
 * 
 * Advanced performance tracking and evaluation system for AI models
 * Real-time metrics, model comparison, and performance visualization
 * Statistical significance testing and model degradation detection
 * 
 * @author ASI Engineers & Founders with 100+ years experience
 * @version 1.0.0 - Production Performance Metrics
 */

const tf = require('@tensorflow/tfjs-node-gpu');
const logger = require('../utils/logger');

class PerformanceMetrics {
  constructor(options = {}) {
    this.config = {
      metricsRetentionDays: options.metricsRetentionDays || 365,
      alertThresholds: {
        accuracyDrop: options.accuracyDrop || 0.05, // 5% drop
        latencyIncrease: options.latencyIncrease || 2.0, // 2x increase
        errorRateIncrease: options.errorRateIncrease || 0.10, // 10% increase
        memoryUsageIncrease: options.memoryUsageIncrease || 1.5 // 1.5x increase
      },
      performanceWindowDays: options.performanceWindowDays || 30,
      comparisonBaseline: options.comparisonBaseline || 'rolling_30d',
      ...options
    };

    // Metrics storage
    this.modelMetrics = new Map();
    this.predictionMetrics = new Map();
    this.systemMetrics = new Map();
    this.alertHistory = [];

    // Performance tracking
    this.performanceHistory = new Map();
    this.baselineMetrics = new Map();
    this.comparisonResults = new Map();

    // Real-time monitoring
    this.realTimeMetrics = {
      predictions: {
        total: 0,
        successful: 0,
        failed: 0,
        averageLatency: 0,
        averageAccuracy: 0
      },
      models: {
        totalModels: 0,
        activeModels: 0,
        trainingModels: 0,
        failedModels: 0
      },
      system: {
        cpuUsage: 0,
        memoryUsage: 0,
        gpuUsage: 0,
        diskUsage: 0
      }
    };

    // Alert system
    this.alertCallbacks = new Map();
    this.alertCooldowns = new Map();
  }

  /**
   * Initialize performance metrics system
   */
  async initialize() {
    try {
      logger.info('📊 Initializing Performance Metrics System...');

      // Initialize TensorFlow.js metrics
      await tf.ready();

      // Start real-time monitoring
      this.startRealTimeMonitoring();

      // Initialize baseline metrics
      await this.initializeBaselineMetrics();

      // Start periodic cleanup
      this.startPeriodicCleanup();

      logger.info('✅ Performance Metrics System initialized successfully');

    } catch (error) {
      logger.error('❌ Performance Metrics System initialization failed:', error);
      throw error;
    }
  }

  /**
   * Start real-time monitoring
   */
  startRealTimeMonitoring() {
    // Monitor system metrics every 30 seconds
    setInterval(() => {
      this.collectSystemMetrics();
    }, 30000);

    // Monitor model performance every 5 minutes
    setInterval(() => {
      this.evaluateModelPerformance();
    }, 300000);

    // Check for alerts every minute
    setInterval(() => {
      this.checkAlerts();
    }, 60000);

    logger.info('🔄 Real-time monitoring started');
  }

  /**
   * Record prediction metrics
   */
  recordPrediction(modelName, predictionData) {
    try {
      const {
        input,
        prediction,
        actualValue,
        confidence,
        latency,
        timestamp = new Date()
      } = predictionData;

      // Calculate accuracy metrics
      const accuracy = this.calculateAccuracy(prediction, actualValue);
      const error = this.calculateError(prediction, actualValue);

      // Create prediction record
      const predictionRecord = {
        modelName,
        timestamp,
        input,
        prediction,
        actualValue,
        confidence,
        accuracy,
        error,
        latency
      };

      // Store prediction
      if (!this.predictionMetrics.has(modelName)) {
        this.predictionMetrics.set(modelName, []);
      }
      this.predictionMetrics.get(modelName).push(predictionRecord);

      // Update real-time metrics
      this.updateRealTimePredictionMetrics(predictionRecord);

      // Check for performance degradation
      this.checkPerformanceDegradation(modelName, predictionRecord);

      return predictionRecord;

    } catch (error) {
      logger.error('❌ Failed to record prediction metrics:', error);
      return null;
    }
  }

  /**
   * Record model training metrics
   */
  recordTrainingMetrics(modelName, trainingData) {
    try {
      const {
        epoch,
        loss,
        accuracy,
        valLoss,
        valAccuracy,
        learningRate,
        batchSize,
        trainingTime,
        memoryUsage,
        timestamp = new Date()
      } = trainingData;

      // Create training record
      const trainingRecord = {
        modelName,
        timestamp,
        epoch,
        loss,
        accuracy,
        valLoss,
        valAccuracy,
        learningRate,
        batchSize,
        trainingTime,
        memoryUsage,
        overfitting: this.detectOverfitting(loss, valLoss),
        convergence: this.checkConvergence(modelName, loss)
      };

      // Store training metrics
      if (!this.modelMetrics.has(modelName)) {
        this.modelMetrics.set(modelName, {
          training: [],
          evaluation: [],
          deployment: []
        });
      }
      this.modelMetrics.get(modelName).training.push(trainingRecord);

      // Update model status
      this.updateModelStatus(modelName, 'training', trainingRecord);

      return trainingRecord;

    } catch (error) {
      logger.error('❌ Failed to record training metrics:', error);
      return null;
    }
  }

  /**
   * Evaluate model performance
   */
  async evaluateModelPerformance() {
    try {
      const evaluationResults = new Map();

      for (const [modelName, predictions] of this.predictionMetrics.entries()) {
        if (predictions.length === 0) continue;

        // Get recent predictions (last 24 hours)
        const recentPredictions = predictions.filter(p => 
          Date.now() - p.timestamp.getTime() < 24 * 60 * 60 * 1000
        );

        if (recentPredictions.length === 0) continue;

        // Calculate comprehensive metrics
        const metrics = await this.calculateComprehensiveMetrics(modelName, recentPredictions);
        
        evaluationResults.set(modelName, metrics);

        // Store in performance history
        if (!this.performanceHistory.has(modelName)) {
          this.performanceHistory.set(modelName, []);
        }
        this.performanceHistory.get(modelName).push({
          timestamp: new Date(),
          metrics
        });
      }

      return evaluationResults;

    } catch (error) {
      logger.error('❌ Model performance evaluation failed:', error);
      return new Map();
    }
  }

  /**
   * Calculate comprehensive metrics for a model
   */
  async calculateComprehensiveMetrics(modelName, predictions) {
    try {
      // Basic accuracy metrics
      const accuracies = predictions.map(p => p.accuracy).filter(a => a !== null);
      const errors = predictions.map(p => p.error).filter(e => e !== null);
      const latencies = predictions.map(p => p.latency).filter(l => l !== null);
      const confidences = predictions.map(p => p.confidence).filter(c => c !== null);

      // Statistical metrics
      const metrics = {
        // Accuracy metrics
        meanAccuracy: this.calculateMean(accuracies),
        medianAccuracy: this.calculateMedian(accuracies),
        accuracyStd: this.calculateStandardDeviation(accuracies),
        
        // Error metrics
        meanError: this.calculateMean(errors),
        meanAbsoluteError: this.calculateMean(errors.map(Math.abs)),
        rootMeanSquareError: Math.sqrt(this.calculateMean(errors.map(e => e * e))),
        
        // Performance metrics
        meanLatency: this.calculateMean(latencies),
        medianLatency: this.calculateMedian(latencies),
        p95Latency: this.calculatePercentile(latencies, 95),
        p99Latency: this.calculatePercentile(latencies, 99),
        
        // Confidence metrics
        meanConfidence: this.calculateMean(confidences),
        confidenceAccuracyCorrelation: this.calculateCorrelation(confidences, accuracies),
        
        // Quality metrics
        predictionCount: predictions.length,
        successRate: predictions.filter(p => p.accuracy !== null).length / predictions.length,
        
        // Temporal metrics
        timeRange: {
          start: Math.min(...predictions.map(p => p.timestamp.getTime())),
          end: Math.max(...predictions.map(p => p.timestamp.getTime()))
        }
      };

      // Advanced metrics
      metrics.sharpeRatio = this.calculateSharpeRatio(predictions);
      metrics.informationRatio = this.calculateInformationRatio(predictions);
      metrics.maxDrawdown = this.calculateMaxDrawdown(predictions);
      metrics.winRate = this.calculateWinRate(predictions);
      metrics.profitFactor = this.calculateProfitFactor(predictions);

      // Model-specific metrics
      if (modelName.includes('nav') || modelName.includes('price')) {
        metrics.directionalAccuracy = this.calculateDirectionalAccuracy(predictions);
        metrics.priceAccuracy = this.calculatePriceAccuracy(predictions);
      }

      if (modelName.includes('risk')) {
        metrics.riskPredictionAccuracy = this.calculateRiskPredictionAccuracy(predictions);
        metrics.volatilityForecastAccuracy = this.calculateVolatilityForecastAccuracy(predictions);
      }

      return metrics;

    } catch (error) {
      logger.error('❌ Comprehensive metrics calculation failed:', error);
      return {};
    }
  }

  /**
   * Compare model performance
   */
  compareModels(modelNames, timeframe = '30d') {
    try {
      const comparison = {
        models: modelNames,
        timeframe,
        timestamp: new Date(),
        metrics: {},
        rankings: {},
        recommendations: []
      };

      // Get performance data for each model
      const modelPerformance = new Map();
      
      for (const modelName of modelNames) {
        const performance = this.getModelPerformance(modelName, timeframe);
        if (performance) {
          modelPerformance.set(modelName, performance);
        }
      }

      if (modelPerformance.size === 0) {
        return comparison;
      }

      // Compare key metrics
      const metricsToCompare = [
        'meanAccuracy',
        'meanLatency',
        'successRate',
        'sharpeRatio',
        'maxDrawdown',
        'winRate'
      ];

      for (const metric of metricsToCompare) {
        comparison.metrics[metric] = {};
        comparison.rankings[metric] = [];

        const metricValues = [];
        
        for (const [modelName, performance] of modelPerformance.entries()) {
          const value = performance[metric] || 0;
          comparison.metrics[metric][modelName] = value;
          metricValues.push({ model: modelName, value });
        }

        // Sort by metric (higher is better for most metrics, lower for latency and drawdown)
        const ascending = ['meanLatency', 'maxDrawdown'].includes(metric);
        metricValues.sort((a, b) => ascending ? a.value - b.value : b.value - a.value);
        
        comparison.rankings[metric] = metricValues.map((item, index) => ({
          rank: index + 1,
          model: item.model,
          value: item.value
        }));
      }

      // Generate recommendations
      comparison.recommendations = this.generateModelRecommendations(comparison);

      return comparison;

    } catch (error) {
      logger.error('❌ Model comparison failed:', error);
      return null;
    }
  }

  /**
   * Generate performance dashboard data
   */
  generateDashboard() {
    try {
      const dashboard = {
        timestamp: new Date(),
        summary: {
          totalModels: this.modelMetrics.size,
          activePredictions: this.getTotalActivePredictions(),
          averageAccuracy: this.getAverageAccuracy(),
          systemHealth: this.getSystemHealth()
        },
        realTimeMetrics: { ...this.realTimeMetrics },
        topPerformingModels: this.getTopPerformingModels(5),
        recentAlerts: this.getRecentAlerts(10),
        performanceTrends: this.getPerformanceTrends(),
        systemMetrics: this.getLatestSystemMetrics(),
        modelStatuses: this.getModelStatuses()
      };

      return dashboard;

    } catch (error) {
      logger.error('❌ Dashboard generation failed:', error);
      return null;
    }
  }

  /**
   * Check for performance alerts
   */
  checkAlerts() {
    try {
      const alerts = [];

      // Check each model for performance issues
      for (const [modelName, metrics] of this.modelMetrics.entries()) {
        const recentPerformance = this.getRecentPerformance(modelName);
        const baseline = this.getBaselinePerformance(modelName);

        if (!recentPerformance || !baseline) continue;

        // Check accuracy drop
        if (baseline.meanAccuracy - recentPerformance.meanAccuracy > this.config.alertThresholds.accuracyDrop) {
          alerts.push({
            type: 'accuracy_drop',
            model: modelName,
            severity: 'high',
            message: `Model ${modelName} accuracy dropped by ${((baseline.meanAccuracy - recentPerformance.meanAccuracy) * 100).toFixed(2)}%`,
            timestamp: new Date()
          });
        }

        // Check latency increase
        if (recentPerformance.meanLatency > baseline.meanLatency * this.config.alertThresholds.latencyIncrease) {
          alerts.push({
            type: 'latency_increase',
            model: modelName,
            severity: 'medium',
            message: `Model ${modelName} latency increased by ${((recentPerformance.meanLatency / baseline.meanLatency - 1) * 100).toFixed(2)}%`,
            timestamp: new Date()
          });
        }

        // Check error rate increase
        const recentErrorRate = 1 - recentPerformance.successRate;
        const baselineErrorRate = 1 - baseline.successRate;
        
        if (recentErrorRate > baselineErrorRate + this.config.alertThresholds.errorRateIncrease) {
          alerts.push({
            type: 'error_rate_increase',
            model: modelName,
            severity: 'high',
            message: `Model ${modelName} error rate increased by ${((recentErrorRate - baselineErrorRate) * 100).toFixed(2)}%`,
            timestamp: new Date()
          });
        }
      }

      // Store alerts and trigger callbacks
      for (const alert of alerts) {
        this.alertHistory.push(alert);
        this.triggerAlertCallbacks(alert);
      }

      return alerts;

    } catch (error) {
      logger.error('❌ Alert checking failed:', error);
      return [];
    }
  }

  /**
   * Register alert callback
   */
  registerAlertCallback(alertType, callback) {
    if (!this.alertCallbacks.has(alertType)) {
      this.alertCallbacks.set(alertType, []);
    }
    this.alertCallbacks.get(alertType).push(callback);
  }

  /**
   * Trigger alert callbacks
   */
  triggerAlertCallbacks(alert) {
    try {
      const callbacks = this.alertCallbacks.get(alert.type) || [];
      const allCallbacks = this.alertCallbacks.get('all') || [];
      
      [...callbacks, ...allCallbacks].forEach(callback => {
        try {
          callback(alert);
        } catch (error) {
          logger.error('❌ Alert callback failed:', error);
        }
      });

    } catch (error) {
      logger.error('❌ Alert callback triggering failed:', error);
    }
  }

  /**
   * Helper methods for statistical calculations
   */
  calculateMean(values) {
    return values.length > 0 ? values.reduce((sum, v) => sum + v, 0) / values.length : 0;
  }

  calculateMedian(values) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  calculateStandardDeviation(values) {
    if (values.length === 0) return 0;
    const mean = this.calculateMean(values);
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  calculatePercentile(values, percentile) {
    if (values.length === 0) return 0;
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.ceil((percentile / 100) * sorted.length) - 1;
    return sorted[Math.max(0, index)];
  }

  calculateCorrelation(x, y) {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const meanX = this.calculateMean(x);
    const meanY = this.calculateMean(y);
    
    let numerator = 0;
    let denomX = 0;
    let denomY = 0;
    
    for (let i = 0; i < x.length; i++) {
      const deltaX = x[i] - meanX;
      const deltaY = y[i] - meanY;
      numerator += deltaX * deltaY;
      denomX += deltaX * deltaX;
      denomY += deltaY * deltaY;
    }
    
    return denomX === 0 || denomY === 0 ? 0 : numerator / Math.sqrt(denomX * denomY);
  }

  /**
   * Calculate accuracy based on prediction type
   */
  calculateAccuracy(prediction, actualValue) {
    if (actualValue === null || actualValue === undefined) return null;
    
    if (typeof prediction === 'number' && typeof actualValue === 'number') {
      // Numerical accuracy (percentage error)
      const error = Math.abs(prediction - actualValue) / Math.abs(actualValue);
      return Math.max(0, 1 - error);
    }
    
    if (typeof prediction === 'boolean' && typeof actualValue === 'boolean') {
      // Boolean accuracy
      return prediction === actualValue ? 1 : 0;
    }
    
    // String/categorical accuracy
    return prediction === actualValue ? 1 : 0;
  }

  /**
   * Calculate error
   */
  calculateError(prediction, actualValue) {
    if (actualValue === null || actualValue === undefined) return null;
    
    if (typeof prediction === 'number' && typeof actualValue === 'number') {
      return prediction - actualValue;
    }
    
    return prediction === actualValue ? 0 : 1;
  }

  /**
   * Get model performance for a specific timeframe
   */
  getModelPerformance(modelName, timeframe) {
    const history = this.performanceHistory.get(modelName);
    if (!history || history.length === 0) return null;

    // Parse timeframe (e.g., '30d', '7d', '1h')
    const timeframeMs = this.parseTimeframe(timeframe);
    const cutoffTime = Date.now() - timeframeMs;

    const recentHistory = history.filter(h => h.timestamp.getTime() > cutoffTime);
    if (recentHistory.length === 0) return null;

    // Return the most recent metrics
    return recentHistory[recentHistory.length - 1].metrics;
  }

  /**
   * Parse timeframe string to milliseconds
   */
  parseTimeframe(timeframe) {
    const match = timeframe.match(/^(\d+)([hdwmy])$/);
    if (!match) return 24 * 60 * 60 * 1000; // Default to 1 day

    const value = parseInt(match[1]);
    const unit = match[2];

    const multipliers = {
      'h': 60 * 60 * 1000,           // hours
      'd': 24 * 60 * 60 * 1000,     // days
      'w': 7 * 24 * 60 * 60 * 1000, // weeks
      'm': 30 * 24 * 60 * 60 * 1000, // months (approximate)
      'y': 365 * 24 * 60 * 60 * 1000 // years (approximate)
    };

    return value * (multipliers[unit] || multipliers['d']);
  }

  /**
   * Get system metrics
   */
  getMetrics() {
    return {
      totalModels: this.modelMetrics.size,
      totalPredictions: Array.from(this.predictionMetrics.values()).reduce((sum, preds) => sum + preds.length, 0),
      alertsGenerated: this.alertHistory.length,
      performanceHistorySize: Array.from(this.performanceHistory.values()).reduce((sum, hist) => sum + hist.length, 0),
      realTimeMetrics: { ...this.realTimeMetrics },
      memoryUsage: process.memoryUsage(),
      tfMemoryUsage: tf.memory()
    };
  }

  /**
   * Start periodic cleanup of old metrics
   */
  startPeriodicCleanup() {
    setInterval(() => {
      this.cleanupOldMetrics();
    }, 24 * 60 * 60 * 1000); // Daily cleanup
  }

  /**
   * Cleanup old metrics data
   */
  cleanupOldMetrics() {
    try {
      const cutoffTime = Date.now() - (this.config.metricsRetentionDays * 24 * 60 * 60 * 1000);

      // Clean prediction metrics
      for (const [modelName, predictions] of this.predictionMetrics.entries()) {
        const filtered = predictions.filter(p => p.timestamp.getTime() > cutoffTime);
        this.predictionMetrics.set(modelName, filtered);
      }

      // Clean performance history
      for (const [modelName, history] of this.performanceHistory.entries()) {
        const filtered = history.filter(h => h.timestamp.getTime() > cutoffTime);
        this.performanceHistory.set(modelName, filtered);
      }

      // Clean alert history
      this.alertHistory = this.alertHistory.filter(a => a.timestamp.getTime() > cutoffTime);

      logger.info('🧹 Metrics cleanup completed');

    } catch (error) {
      logger.error('❌ Metrics cleanup failed:', error);
    }
  }
}

module.exports = { PerformanceMetrics };
