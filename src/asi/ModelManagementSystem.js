/**
 * 🤖 MODEL MANAGEMENT SYSTEM
 * 
 * Advanced ML Model Lifecycle Management with A/B Testing and Performance Tracking
 * Handles model versioning, deployment, monitoring, and automated retraining
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Production-Grade Model Management
 */

const EventEmitter = require('events');
const fs = require('fs').promises;
const path = require('path');
const logger = require('../utils/logger');

class ModelManagementSystem extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      modelsDirectory: options.modelsDirectory || path.join(__dirname, '../../models'),
      backupDirectory: options.backupDirectory || path.join(__dirname, '../../models/backups'),
      maxModelVersions: options.maxModelVersions || 10,
      performanceThreshold: options.performanceThreshold || 0.8,
      retrainingThreshold: options.retrainingThreshold || 0.75,
      abTestTrafficSplit: options.abTestTrafficSplit || 0.1,
      ...options
    };
    
    // Model registry
    this.models = new Map();
    this.modelVersions = new Map();
    this.deploymentQueue = [];
    this.abTests = new Map();
    
    // Performance tracking
    this.modelMetrics = new Map();
    this.performanceHistory = new Map();
    
    // Model states
    this.modelStates = {
      TRAINING: 'training',
      VALIDATION: 'validation',
      TESTING: 'testing',
      DEPLOYED: 'deployed',
      DEPRECATED: 'deprecated',
      FAILED: 'failed'
    };
    
    this.isInitialized = false;
  }

  async initialize() {
    try {
      logger.info('🤖 Initializing Model Management System...');
      
      // Create directories
      await this.createDirectories();
      
      // Load existing models
      await this.loadExistingModels();
      
      // Start monitoring
      this.startPerformanceMonitoring();
      
      // Start automated retraining checks
      this.startRetrainingMonitor();
      
      this.isInitialized = true;
      logger.info('✅ Model Management System initialized successfully');
      
    } catch (error) {
      logger.error('❌ Model Management System initialization failed:', error);
      throw error;
    }
  }

  async createDirectories() {
    const directories = [
      this.config.modelsDirectory,
      this.config.backupDirectory,
      path.join(this.config.modelsDirectory, 'staging'),
      path.join(this.config.modelsDirectory, 'production'),
      path.join(this.config.modelsDirectory, 'archived')
    ];
    
    for (const dir of directories) {
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        if (error.code !== 'EEXIST') {
          throw error;
        }
      }
    }
  }

  async loadExistingModels() {
    try {
      const productionDir = path.join(this.config.modelsDirectory, 'production');
      const files = await fs.readdir(productionDir);
      
      for (const file of files) {
        if (file.endsWith('.json')) {
          const modelPath = path.join(productionDir, file);
          const modelConfig = JSON.parse(await fs.readFile(modelPath, 'utf8'));
          
          this.models.set(modelConfig.id, {
            ...modelConfig,
            state: this.modelStates.DEPLOYED,
            loadedAt: new Date()
          });
          
          logger.info(`📦 Loaded model: ${modelConfig.id} v${modelConfig.version}`);
        }
      }
      
    } catch (error) {
      logger.warn('⚠️ No existing models found or failed to load:', error.message);
    }
  }

  async registerModel(modelConfig) {
    try {
      const modelId = modelConfig.id || `model_${Date.now()}`;
      const version = modelConfig.version || '1.0.0';
      
      const model = {
        id: modelId,
        version,
        name: modelConfig.name,
        type: modelConfig.type,
        description: modelConfig.description,
        architecture: modelConfig.architecture,
        hyperparameters: modelConfig.hyperparameters,
        trainingData: modelConfig.trainingData,
        performance: modelConfig.performance || {},
        state: this.modelStates.TRAINING,
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: modelConfig.metadata || {}
      };
      
      this.models.set(modelId, model);
      
      // Track versions
      if (!this.modelVersions.has(modelId)) {
        this.modelVersions.set(modelId, []);
      }
      this.modelVersions.get(modelId).push(version);
      
      // Initialize metrics
      this.modelMetrics.set(modelId, {
        totalPredictions: 0,
        correctPredictions: 0,
        accuracy: 0,
        latency: [],
        errors: 0,
        lastUpdated: new Date()
      });
      
      logger.info(`🤖 Model registered: ${modelId} v${version}`);
      this.emit('modelRegistered', { modelId, version, model });
      
      return modelId;
      
    } catch (error) {
      logger.error('❌ Model registration failed:', error);
      throw error;
    }
  }

  async deployModel(modelId, environment = 'production') {
    try {
      const model = this.models.get(modelId);
      if (!model) {
        throw new Error(`Model not found: ${modelId}`);
      }
      
      // Validate model before deployment
      await this.validateModel(modelId);
      
      // Backup current production model if exists
      await this.backupCurrentModel(modelId, environment);
      
      // Deploy new model
      const deploymentPath = path.join(
        this.config.modelsDirectory, 
        environment, 
        `${modelId}.json`
      );
      
      await fs.writeFile(deploymentPath, JSON.stringify(model, null, 2));
      
      // Update model state
      model.state = this.modelStates.DEPLOYED;
      model.deployedAt = new Date();
      model.environment = environment;
      
      logger.info(`🚀 Model deployed: ${modelId} to ${environment}`);
      this.emit('modelDeployed', { modelId, environment, model });
      
      return true;
      
    } catch (error) {
      logger.error('❌ Model deployment failed:', error);
      throw error;
    }
  }

  async validateModel(modelId) {
    const model = this.models.get(modelId);
    if (!model) {
      throw new Error(`Model not found: ${modelId}`);
    }
    
    // Basic validation checks
    const validationChecks = [
      { check: () => model.performance.accuracy >= this.config.performanceThreshold, 
        message: 'Model accuracy below threshold' },
      { check: () => model.architecture && model.architecture.layers, 
        message: 'Invalid model architecture' },
      { check: () => model.trainingData && model.trainingData.samples > 1000, 
        message: 'Insufficient training data' }
    ];
    
    for (const { check, message } of validationChecks) {
      if (!check()) {
        throw new Error(`Model validation failed: ${message}`);
      }
    }
    
    logger.info(`✅ Model validation passed: ${modelId}`);
    return true;
  }

  async backupCurrentModel(modelId, environment) {
    try {
      const currentModelPath = path.join(
        this.config.modelsDirectory, 
        environment, 
        `${modelId}.json`
      );
      
      const backupPath = path.join(
        this.config.backupDirectory,
        `${modelId}_${Date.now()}.json`
      );
      
      try {
        await fs.copyFile(currentModelPath, backupPath);
        logger.info(`💾 Model backed up: ${modelId}`);
      } catch (error) {
        if (error.code !== 'ENOENT') {
          throw error;
        }
        // No existing model to backup
      }
      
    } catch (error) {
      logger.warn('⚠️ Model backup failed:', error.message);
    }
  }

  async abTestModels(modelAId, modelBId, trafficSplit = 0.5) {
    try {
      const modelA = this.models.get(modelAId);
      const modelB = this.models.get(modelBId);
      
      if (!modelA || !modelB) {
        throw new Error('Both models must exist for A/B testing');
      }
      
      const testId = `ab_test_${Date.now()}`;
      const abTest = {
        id: testId,
        modelA: modelAId,
        modelB: modelBId,
        trafficSplit,
        startTime: new Date(),
        endTime: null,
        metrics: {
          modelA: { requests: 0, accuracy: 0, latency: [] },
          modelB: { requests: 0, accuracy: 0, latency: [] }
        },
        status: 'running'
      };
      
      this.abTests.set(testId, abTest);
      
      logger.info(`🧪 A/B test started: ${testId} (${modelAId} vs ${modelBId})`);
      this.emit('abTestStarted', { testId, abTest });
      
      return testId;
      
    } catch (error) {
      logger.error('❌ A/B test setup failed:', error);
      throw error;
    }
  }

  async routeRequest(request, testId = null) {
    try {
      let selectedModelId;
      
      if (testId && this.abTests.has(testId)) {
        // A/B test routing
        const abTest = this.abTests.get(testId);
        const random = Math.random();
        
        selectedModelId = random < abTest.trafficSplit ? 
          abTest.modelA : abTest.modelB;
        
        // Track A/B test metrics
        const modelKey = selectedModelId === abTest.modelA ? 'modelA' : 'modelB';
        abTest.metrics[modelKey].requests++;
        
      } else {
        // Default routing to production model
        selectedModelId = this.getDefaultModel(request.type);
      }
      
      return selectedModelId;
      
    } catch (error) {
      logger.error('❌ Request routing failed:', error);
      throw error;
    }
  }

  getDefaultModel(requestType) {
    // Find the best performing model for the request type
    const candidates = Array.from(this.models.values())
      .filter(model => 
        model.state === this.modelStates.DEPLOYED && 
        model.type === requestType
      )
      .sort((a, b) => (b.performance.accuracy || 0) - (a.performance.accuracy || 0));
    
    return candidates.length > 0 ? candidates[0].id : null;
  }

  async trackModelPerformance(modelId, prediction, actual, latency) {
    try {
      const metrics = this.modelMetrics.get(modelId);
      if (!metrics) return;
      
      metrics.totalPredictions++;
      metrics.latency.push(latency);
      
      // Keep only last 1000 latency measurements
      if (metrics.latency.length > 1000) {
        metrics.latency = metrics.latency.slice(-1000);
      }
      
      if (actual !== undefined) {
        const isCorrect = this.evaluatePrediction(prediction, actual);
        if (isCorrect) {
          metrics.correctPredictions++;
        }
        
        metrics.accuracy = metrics.correctPredictions / metrics.totalPredictions;
      }
      
      metrics.lastUpdated = new Date();
      
      // Check if retraining is needed
      if (metrics.accuracy < this.config.retrainingThreshold) {
        this.emit('retrainingNeeded', { modelId, accuracy: metrics.accuracy });
      }
      
    } catch (error) {
      logger.error('❌ Performance tracking failed:', error);
    }
  }

  evaluatePrediction(prediction, actual) {
    // Simple evaluation - can be made more sophisticated
    if (typeof prediction === 'number' && typeof actual === 'number') {
      const error = Math.abs(prediction - actual) / actual;
      return error < 0.1; // Within 10% is considered correct
    }
    
    return prediction === actual;
  }

  async scheduleRetraining(modelId, trigger = 'performance') {
    try {
      const model = this.models.get(modelId);
      if (!model) {
        throw new Error(`Model not found: ${modelId}`);
      }
      
      const retrainingJob = {
        modelId,
        trigger,
        scheduledAt: new Date(),
        status: 'scheduled',
        priority: trigger === 'performance' ? 'high' : 'normal'
      };
      
      this.deploymentQueue.push(retrainingJob);
      
      logger.info(`📅 Retraining scheduled: ${modelId} (trigger: ${trigger})`);
      this.emit('retrainingScheduled', { modelId, trigger });
      
      return true;
      
    } catch (error) {
      logger.error('❌ Retraining scheduling failed:', error);
      throw error;
    }
  }

  startPerformanceMonitoring() {
    setInterval(() => {
      this.generatePerformanceReport();
    }, 300000); // Every 5 minutes
    
    logger.info('📊 Performance monitoring started');
  }

  startRetrainingMonitor() {
    setInterval(() => {
      this.checkRetrainingNeeds();
    }, 3600000); // Every hour
    
    logger.info('🔄 Retraining monitor started');
  }

  async generatePerformanceReport() {
    try {
      const report = {
        timestamp: new Date(),
        models: {},
        summary: {
          totalModels: this.models.size,
          deployedModels: 0,
          averageAccuracy: 0,
          totalPredictions: 0
        }
      };
      
      let totalAccuracy = 0;
      let deployedCount = 0;
      
      for (const [modelId, model] of this.models) {
        const metrics = this.modelMetrics.get(modelId);
        
        if (model.state === this.modelStates.DEPLOYED) {
          deployedCount++;
          totalAccuracy += metrics?.accuracy || 0;
        }
        
        report.models[modelId] = {
          state: model.state,
          version: model.version,
          accuracy: metrics?.accuracy || 0,
          totalPredictions: metrics?.totalPredictions || 0,
          averageLatency: metrics?.latency.length > 0 ? 
            metrics.latency.reduce((a, b) => a + b) / metrics.latency.length : 0,
          lastUpdated: metrics?.lastUpdated
        };
        
        report.summary.totalPredictions += metrics?.totalPredictions || 0;
      }
      
      report.summary.deployedModels = deployedCount;
      report.summary.averageAccuracy = deployedCount > 0 ? totalAccuracy / deployedCount : 0;
      
      this.emit('performanceReport', report);
      
      // Log summary
      logger.info(`📊 Performance Report: ${deployedCount} models deployed, avg accuracy: ${(report.summary.averageAccuracy * 100).toFixed(2)}%`);
      
    } catch (error) {
      logger.error('❌ Performance report generation failed:', error);
    }
  }

  async checkRetrainingNeeds() {
    try {
      for (const [modelId, model] of this.models) {
        if (model.state !== this.modelStates.DEPLOYED) continue;
        
        const metrics = this.modelMetrics.get(modelId);
        if (!metrics) continue;
        
        // Check accuracy degradation
        if (metrics.accuracy < this.config.retrainingThreshold) {
          await this.scheduleRetraining(modelId, 'accuracy_degradation');
        }
        
        // Check data drift (simplified check)
        const daysSinceLastUpdate = (Date.now() - model.updatedAt.getTime()) / (1000 * 60 * 60 * 24);
        if (daysSinceLastUpdate > 30) {
          await this.scheduleRetraining(modelId, 'data_drift');
        }
      }
      
    } catch (error) {
      logger.error('❌ Retraining needs check failed:', error);
    }
  }

  async getModelStatus(modelId) {
    const model = this.models.get(modelId);
    const metrics = this.modelMetrics.get(modelId);
    
    if (!model) {
      throw new Error(`Model not found: ${modelId}`);
    }
    
    return {
      model: {
        id: model.id,
        version: model.version,
        name: model.name,
        type: model.type,
        state: model.state,
        createdAt: model.createdAt,
        deployedAt: model.deployedAt,
        environment: model.environment
      },
      metrics: metrics || {},
      versions: this.modelVersions.get(modelId) || []
    };
  }

  async getAllModelsStatus() {
    const status = {
      summary: {
        totalModels: this.models.size,
        deployedModels: 0,
        trainingModels: 0,
        failedModels: 0
      },
      models: []
    };
    
    for (const [modelId] of this.models) {
      const modelStatus = await this.getModelStatus(modelId);
      status.models.push(modelStatus);
      
      switch (modelStatus.model.state) {
        case this.modelStates.DEPLOYED:
          status.summary.deployedModels++;
          break;
        case this.modelStates.TRAINING:
          status.summary.trainingModels++;
          break;
        case this.modelStates.FAILED:
          status.summary.failedModels++;
          break;
      }
    }
    
    return status;
  }

  async archiveModel(modelId) {
    try {
      const model = this.models.get(modelId);
      if (!model) {
        throw new Error(`Model not found: ${modelId}`);
      }
      
      // Move to archived directory
      const archivePath = path.join(
        this.config.modelsDirectory,
        'archived',
        `${modelId}_${Date.now()}.json`
      );
      
      await fs.writeFile(archivePath, JSON.stringify(model, null, 2));
      
      // Update state
      model.state = this.modelStates.DEPRECATED;
      model.archivedAt = new Date();
      
      logger.info(`📦 Model archived: ${modelId}`);
      this.emit('modelArchived', { modelId, model });
      
      return true;
      
    } catch (error) {
      logger.error('❌ Model archiving failed:', error);
      throw error;
    }
  }
}

module.exports = { ModelManagementSystem };
