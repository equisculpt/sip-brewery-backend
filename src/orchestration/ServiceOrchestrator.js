/**
 * 🎼 ENTERPRISE MICROSERVICES ORCHESTRATION
 * 
 * Advanced service orchestration with saga patterns, distributed transactions,
 * service discovery, and choreography-based workflows
 * 
 * @author Senior AI Backend Developer (35+ years)
 * @version 3.0.0
 */

const { v4: uuidv4 } = require('uuid');
const EventEmitter = require('events');
const logger = require('../utils/logger');

/**
 * Service Registry for Service Discovery
 */
class ServiceRegistry {
  constructor() {
    this.services = new Map();
    this.healthChecks = new Map();
    this.loadBalancers = new Map();
    this.serviceMetrics = new Map();
  }

  /**
   * Register a service
   */
  registerService(serviceName, serviceInfo) {
    const serviceId = uuidv4();
    const service = {
      id: serviceId,
      name: serviceName,
      host: serviceInfo.host,
      port: serviceInfo.port,
      version: serviceInfo.version || '1.0.0',
      protocol: serviceInfo.protocol || 'http',
      healthCheckUrl: serviceInfo.healthCheckUrl,
      metadata: serviceInfo.metadata || {},
      registeredAt: new Date(),
      lastHealthCheck: null,
      status: 'STARTING'
    };

    if (!this.services.has(serviceName)) {
      this.services.set(serviceName, []);
    }
    
    this.services.get(serviceName).push(service);
    
    // Initialize metrics
    this.serviceMetrics.set(serviceId, {
      requestCount: 0,
      errorCount: 0,
      averageResponseTime: 0,
      lastRequestTime: null
    });

    // Start health checking
    this.startHealthCheck(service);

    logger.info('🔧 Service registered', {
      serviceName,
      serviceId,
      host: service.host,
      port: service.port
    });

    return serviceId;
  }

  /**
   * Discover services by name
   */
  discoverServices(serviceName) {
    const services = this.services.get(serviceName) || [];
    return services.filter(service => service.status === 'HEALTHY');
  }

  /**
   * Get service by load balancing strategy
   */
  getService(serviceName, strategy = 'round-robin') {
    const healthyServices = this.discoverServices(serviceName);
    
    if (healthyServices.length === 0) {
      throw new Error(`No healthy instances of service ${serviceName} available`);
    }

    switch (strategy) {
      case 'round-robin':
        return this.roundRobinSelect(serviceName, healthyServices);
      case 'least-connections':
        return this.leastConnectionsSelect(healthyServices);
      case 'weighted':
        return this.weightedSelect(healthyServices);
      default:
        return healthyServices[0];
    }
  }

  roundRobinSelect(serviceName, services) {
    if (!this.loadBalancers.has(serviceName)) {
      this.loadBalancers.set(serviceName, { currentIndex: 0 });
    }
    
    const lb = this.loadBalancers.get(serviceName);
    const service = services[lb.currentIndex % services.length];
    lb.currentIndex++;
    
    return service;
  }

  leastConnectionsSelect(services) {
    return services.reduce((least, current) => {
      const leastMetrics = this.serviceMetrics.get(least.id);
      const currentMetrics = this.serviceMetrics.get(current.id);
      
      return currentMetrics.requestCount < leastMetrics.requestCount ? current : least;
    });
  }

  weightedSelect(services) {
    // Simple weighted selection based on inverse of error rate
    const weights = services.map(service => {
      const metrics = this.serviceMetrics.get(service.id);
      const errorRate = metrics.requestCount > 0 ? metrics.errorCount / metrics.requestCount : 0;
      return Math.max(0.1, 1 - errorRate); // Minimum weight of 0.1
    });

    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    const random = Math.random() * totalWeight;
    
    let currentWeight = 0;
    for (let i = 0; i < services.length; i++) {
      currentWeight += weights[i];
      if (random <= currentWeight) {
        return services[i];
      }
    }
    
    return services[0];
  }

  /**
   * Start health checking for a service
   */
  startHealthCheck(service) {
    const healthCheckInterval = setInterval(async () => {
      try {
        const isHealthy = await this.performHealthCheck(service);
        service.status = isHealthy ? 'HEALTHY' : 'UNHEALTHY';
        service.lastHealthCheck = new Date();
        
        if (!isHealthy) {
          logger.warn('⚠️ Service health check failed', {
            serviceName: service.name,
            serviceId: service.id
          });
        }
        
      } catch (error) {
        service.status = 'UNHEALTHY';
        service.lastHealthCheck = new Date();
        
        logger.error('❌ Health check error', {
          serviceName: service.name,
          serviceId: service.id,
          error: error.message
        });
      }
    }, 30000); // Check every 30 seconds

    this.healthChecks.set(service.id, healthCheckInterval);
  }

  async performHealthCheck(service) {
    if (!service.healthCheckUrl) {
      return true; // Assume healthy if no health check URL
    }

    // This would typically make an HTTP request to the health check endpoint
    // For now, we'll simulate it
    return Math.random() > 0.1; // 90% success rate
  }

  /**
   * Update service metrics
   */
  updateServiceMetrics(serviceId, responseTime, isError = false) {
    const metrics = this.serviceMetrics.get(serviceId);
    if (!metrics) return;

    metrics.requestCount++;
    if (isError) metrics.errorCount++;
    
    // Update average response time
    metrics.averageResponseTime = 
      (metrics.averageResponseTime * (metrics.requestCount - 1) + responseTime) / metrics.requestCount;
    
    metrics.lastRequestTime = new Date();
  }

  /**
   * Deregister service
   */
  deregisterService(serviceId) {
    for (const [serviceName, services] of this.services.entries()) {
      const index = services.findIndex(service => service.id === serviceId);
      if (index !== -1) {
        const service = services[index];
        services.splice(index, 1);
        
        // Clean up health check
        const healthCheck = this.healthChecks.get(serviceId);
        if (healthCheck) {
          clearInterval(healthCheck);
          this.healthChecks.delete(serviceId);
        }
        
        // Clean up metrics
        this.serviceMetrics.delete(serviceId);
        
        logger.info('🔧 Service deregistered', {
          serviceName: service.name,
          serviceId
        });
        
        return true;
      }
    }
    return false;
  }

  getRegistryStatus() {
    const status = {
      totalServices: 0,
      healthyServices: 0,
      unhealthyServices: 0,
      servicesByName: {}
    };

    for (const [serviceName, services] of this.services.entries()) {
      const healthy = services.filter(s => s.status === 'HEALTHY').length;
      const unhealthy = services.filter(s => s.status === 'UNHEALTHY').length;
      
      status.totalServices += services.length;
      status.healthyServices += healthy;
      status.unhealthyServices += unhealthy;
      
      status.servicesByName[serviceName] = {
        total: services.length,
        healthy,
        unhealthy,
        instances: services.map(s => ({
          id: s.id,
          host: s.host,
          port: s.port,
          status: s.status,
          lastHealthCheck: s.lastHealthCheck
        }))
      };
    }

    return status;
  }
}

/**
 * Saga Step Definition
 */
class SagaStep {
  constructor(name, action, compensation) {
    this.name = name;
    this.action = action; // Function to execute
    this.compensation = compensation; // Function to rollback
    this.status = 'PENDING'; // PENDING, COMPLETED, FAILED, COMPENSATED
    this.result = null;
    this.error = null;
    this.executedAt = null;
    this.compensatedAt = null;
  }

  async execute(context) {
    try {
      this.executedAt = new Date();
      this.result = await this.action(context);
      this.status = 'COMPLETED';
      return this.result;
    } catch (error) {
      this.error = error;
      this.status = 'FAILED';
      throw error;
    }
  }

  async compensate(context) {
    if (this.status !== 'COMPLETED') {
      return; // Nothing to compensate
    }

    try {
      this.compensatedAt = new Date();
      await this.compensation(context, this.result);
      this.status = 'COMPENSATED';
    } catch (error) {
      logger.error('❌ Compensation failed', {
        step: this.name,
        error: error.message
      });
      throw error;
    }
  }
}

/**
 * Saga Orchestrator
 */
class SagaOrchestrator extends EventEmitter {
  constructor(sagaName, steps = []) {
    super();
    this.sagaId = uuidv4();
    this.sagaName = sagaName;
    this.steps = steps;
    this.currentStepIndex = 0;
    this.status = 'PENDING'; // PENDING, RUNNING, COMPLETED, FAILED, COMPENSATING, COMPENSATED
    this.context = {};
    this.startedAt = null;
    this.completedAt = null;
    this.failedAt = null;
  }

  addStep(step) {
    if (!(step instanceof SagaStep)) {
      throw new Error('Step must be an instance of SagaStep');
    }
    this.steps.push(step);
  }

  async execute(initialContext = {}) {
    this.context = { ...initialContext, sagaId: this.sagaId };
    this.status = 'RUNNING';
    this.startedAt = new Date();

    logger.info('🎼 Starting saga execution', {
      sagaId: this.sagaId,
      sagaName: this.sagaName,
      stepCount: this.steps.length
    });

    try {
      // Execute all steps
      for (let i = 0; i < this.steps.length; i++) {
        this.currentStepIndex = i;
        const step = this.steps[i];
        
        logger.debug('⚡ Executing saga step', {
          sagaId: this.sagaId,
          stepName: step.name,
          stepIndex: i
        });

        await step.execute(this.context);
        
        this.emit('stepCompleted', {
          sagaId: this.sagaId,
          stepName: step.name,
          stepIndex: i,
          result: step.result
        });
      }

      this.status = 'COMPLETED';
      this.completedAt = new Date();
      
      logger.info('✅ Saga completed successfully', {
        sagaId: this.sagaId,
        sagaName: this.sagaName,
        duration: this.completedAt - this.startedAt
      });

      this.emit('completed', {
        sagaId: this.sagaId,
        sagaName: this.sagaName,
        context: this.context
      });

      return this.context;

    } catch (error) {
      this.status = 'FAILED';
      this.failedAt = new Date();
      
      logger.error('❌ Saga execution failed', {
        sagaId: this.sagaId,
        sagaName: this.sagaName,
        failedStep: this.steps[this.currentStepIndex].name,
        error: error.message
      });

      this.emit('failed', {
        sagaId: this.sagaId,
        sagaName: this.sagaName,
        failedStep: this.steps[this.currentStepIndex].name,
        error
      });

      // Start compensation
      await this.compensate();
      
      throw error;
    }
  }

  async compensate() {
    this.status = 'COMPENSATING';
    
    logger.info('🔄 Starting saga compensation', {
      sagaId: this.sagaId,
      sagaName: this.sagaName,
      stepsToCompensate: this.currentStepIndex
    });

    // Compensate completed steps in reverse order
    for (let i = this.currentStepIndex - 1; i >= 0; i--) {
      const step = this.steps[i];
      
      if (step.status === 'COMPLETED') {
        try {
          logger.debug('🔄 Compensating saga step', {
            sagaId: this.sagaId,
            stepName: step.name,
            stepIndex: i
          });

          await step.compensate(this.context);
          
          this.emit('stepCompensated', {
            sagaId: this.sagaId,
            stepName: step.name,
            stepIndex: i
          });

        } catch (error) {
          logger.error('❌ Step compensation failed', {
            sagaId: this.sagaId,
            stepName: step.name,
            error: error.message
          });
          // Continue with other compensations
        }
      }
    }

    this.status = 'COMPENSATED';
    
    logger.info('✅ Saga compensation completed', {
      sagaId: this.sagaId,
      sagaName: this.sagaName
    });

    this.emit('compensated', {
      sagaId: this.sagaId,
      sagaName: this.sagaName
    });
  }

  getStatus() {
    return {
      sagaId: this.sagaId,
      sagaName: this.sagaName,
      status: this.status,
      currentStepIndex: this.currentStepIndex,
      totalSteps: this.steps.length,
      startedAt: this.startedAt,
      completedAt: this.completedAt,
      failedAt: this.failedAt,
      steps: this.steps.map((step, index) => ({
        name: step.name,
        status: step.status,
        executedAt: step.executedAt,
        compensatedAt: step.compensatedAt,
        error: step.error?.message
      }))
    };
  }
}

/**
 * Service Orchestrator
 */
class ServiceOrchestrator {
  constructor(options = {}) {
    this.serviceRegistry = options.serviceRegistry || new ServiceRegistry();
    this.activeSagas = new Map();
    this.sagaHistory = new Map();
    this.eventBus = options.eventBus || null;
    
    this.metrics = {
      sagasExecuted: 0,
      sagasCompleted: 0,
      sagasFailed: 0,
      sagasCompensated: 0,
      averageExecutionTime: 0
    };
  }

  /**
   * Create and execute a saga
   */
  async executeSaga(sagaName, steps, initialContext = {}) {
    const saga = new SagaOrchestrator(sagaName, steps);
    
    // Set up event listeners
    saga.on('completed', (event) => {
      this.metrics.sagasCompleted++;
      this.updateAverageExecutionTime(saga);
      this.moveSagaToHistory(saga);
    });

    saga.on('failed', (event) => {
      this.metrics.sagasFailed++;
      this.updateAverageExecutionTime(saga);
    });

    saga.on('compensated', (event) => {
      this.metrics.sagasCompensated++;
      this.moveSagaToHistory(saga);
    });

    this.activeSagas.set(saga.sagaId, saga);
    this.metrics.sagasExecuted++;

    try {
      const result = await saga.execute(initialContext);
      return result;
    } catch (error) {
      // Saga remains in active sagas until compensated
      throw error;
    }
  }

  /**
   * Create investment saga
   */
  createInvestmentSaga(userId, investmentData) {
    const steps = [
      new SagaStep(
        'ValidateUser',
        async (context) => {
          // Validate user exists and is active
          const userService = this.serviceRegistry.getService('user-service');
          const user = await this.callService(userService, 'GET', `/users/${userId}`);
          context.user = user;
          return user;
        },
        async (context) => {
          // No compensation needed for validation
        }
      ),

      new SagaStep(
        'ValidateFund',
        async (context) => {
          // Validate fund exists and is available
          const fundService = this.serviceRegistry.getService('fund-service');
          const fund = await this.callService(fundService, 'GET', `/funds/${investmentData.fundCode}`);
          context.fund = fund;
          return fund;
        },
        async (context) => {
          // No compensation needed for validation
        }
      ),

      new SagaStep(
        'ReserveAmount',
        async (context) => {
          // Reserve amount in user's account
          const walletService = this.serviceRegistry.getService('wallet-service');
          const reservation = await this.callService(walletService, 'POST', '/reservations', {
            userId,
            amount: investmentData.amount,
            purpose: 'INVESTMENT'
          });
          context.reservation = reservation;
          return reservation;
        },
        async (context, result) => {
          // Release reservation
          const walletService = this.serviceRegistry.getService('wallet-service');
          await this.callService(walletService, 'DELETE', `/reservations/${result.id}`);
        }
      ),

      new SagaStep(
        'CreateInvestment',
        async (context) => {
          // Create investment record
          const investmentService = this.serviceRegistry.getService('investment-service');
          const investment = await this.callService(investmentService, 'POST', '/investments', {
            userId,
            fundCode: investmentData.fundCode,
            amount: investmentData.amount,
            type: investmentData.type,
            reservationId: context.reservation.id
          });
          context.investment = investment;
          return investment;
        },
        async (context, result) => {
          // Delete investment record
          const investmentService = this.serviceRegistry.getService('investment-service');
          await this.callService(investmentService, 'DELETE', `/investments/${result.id}`);
        }
      ),

      new SagaStep(
        'ProcessPayment',
        async (context) => {
          // Process payment
          const paymentService = this.serviceRegistry.getService('payment-service');
          const payment = await this.callService(paymentService, 'POST', '/payments', {
            userId,
            amount: investmentData.amount,
            investmentId: context.investment.id,
            reservationId: context.reservation.id
          });
          context.payment = payment;
          return payment;
        },
        async (context, result) => {
          // Refund payment
          const paymentService = this.serviceRegistry.getService('payment-service');
          await this.callService(paymentService, 'POST', `/payments/${result.id}/refund`);
        }
      ),

      new SagaStep(
        'UpdatePortfolio',
        async (context) => {
          // Update user's portfolio
          const portfolioService = this.serviceRegistry.getService('portfolio-service');
          const portfolio = await this.callService(portfolioService, 'PUT', `/portfolios/${userId}`, {
            investment: context.investment
          });
          context.portfolio = portfolio;
          return portfolio;
        },
        async (context, result) => {
          // Revert portfolio update
          const portfolioService = this.serviceRegistry.getService('portfolio-service');
          await this.callService(portfolioService, 'DELETE', `/portfolios/${userId}/investments/${context.investment.id}`);
        }
      ),

      new SagaStep(
        'SendNotification',
        async (context) => {
          // Send success notification
          const notificationService = this.serviceRegistry.getService('notification-service');
          const notification = await this.callService(notificationService, 'POST', '/notifications', {
            userId,
            type: 'INVESTMENT_SUCCESS',
            data: {
              fundName: context.fund.name,
              amount: investmentData.amount,
              investmentId: context.investment.id
            }
          });
          return notification;
        },
        async (context) => {
          // Send failure notification
          const notificationService = this.serviceRegistry.getService('notification-service');
          await this.callService(notificationService, 'POST', '/notifications', {
            userId,
            type: 'INVESTMENT_FAILED',
            data: {
              fundName: context.fund?.name,
              amount: investmentData.amount
            }
          });
        }
      )
    ];

    return steps;
  }

  /**
   * Call a service
   */
  async callService(service, method, path, data = null) {
    const startTime = Date.now();
    
    try {
      // This would typically make an HTTP request to the service
      // For now, we'll simulate the call
      await new Promise(resolve => setTimeout(resolve, Math.random() * 100 + 50));
      
      const responseTime = Date.now() - startTime;
      this.serviceRegistry.updateServiceMetrics(service.id, responseTime, false);
      
      // Simulate response based on method and path
      if (method === 'GET' && path.includes('/users/')) {
        return { id: path.split('/').pop(), name: 'Test User', status: 'ACTIVE' };
      } else if (method === 'GET' && path.includes('/funds/')) {
        return { code: path.split('/').pop(), name: 'Test Fund', status: 'ACTIVE' };
      } else if (method === 'POST' && path.includes('/reservations')) {
        return { id: uuidv4(), ...data, status: 'RESERVED' };
      } else if (method === 'POST' && path.includes('/investments')) {
        return { id: uuidv4(), ...data, status: 'CREATED' };
      } else if (method === 'POST' && path.includes('/payments')) {
        return { id: uuidv4(), ...data, status: 'COMPLETED' };
      } else if (method === 'PUT' && path.includes('/portfolios/')) {
        return { userId: path.split('/').pop(), updated: true };
      } else if (method === 'POST' && path.includes('/notifications')) {
        return { id: uuidv4(), ...data, sent: true };
      }
      
      return { success: true };
      
    } catch (error) {
      const responseTime = Date.now() - startTime;
      this.serviceRegistry.updateServiceMetrics(service.id, responseTime, true);
      throw error;
    }
  }

  /**
   * Get saga status
   */
  getSagaStatus(sagaId) {
    const activeSaga = this.activeSagas.get(sagaId);
    if (activeSaga) {
      return activeSaga.getStatus();
    }
    
    const historicalSaga = this.sagaHistory.get(sagaId);
    if (historicalSaga) {
      return historicalSaga;
    }
    
    return null;
  }

  /**
   * List active sagas
   */
  getActiveSagas() {
    return Array.from(this.activeSagas.values()).map(saga => saga.getStatus());
  }

  /**
   * Move saga to history
   */
  moveSagaToHistory(saga) {
    const status = saga.getStatus();
    this.sagaHistory.set(saga.sagaId, status);
    this.activeSagas.delete(saga.sagaId);
    
    // Keep only last 1000 historical sagas
    if (this.sagaHistory.size > 1000) {
      const oldestKey = this.sagaHistory.keys().next().value;
      this.sagaHistory.delete(oldestKey);
    }
  }

  /**
   * Update average execution time
   */
  updateAverageExecutionTime(saga) {
    if (!saga.completedAt && !saga.failedAt) return;
    
    const endTime = saga.completedAt || saga.failedAt;
    const executionTime = endTime - saga.startedAt;
    
    const totalSagas = this.metrics.sagasCompleted + this.metrics.sagasFailed;
    this.metrics.averageExecutionTime = 
      (this.metrics.averageExecutionTime * (totalSagas - 1) + executionTime) / totalSagas;
  }

  /**
   * Get orchestrator metrics
   */
  getMetrics() {
    return {
      ...this.metrics,
      activeSagas: this.activeSagas.size,
      historicalSagas: this.sagaHistory.size,
      serviceRegistry: this.serviceRegistry.getRegistryStatus()
    };
  }

  /**
   * Health check
   */
  async healthCheck() {
    const registryStatus = this.serviceRegistry.getRegistryStatus();
    const metrics = this.getMetrics();
    
    const healthScore = registryStatus.totalServices > 0 
      ? (registryStatus.healthyServices / registryStatus.totalServices) * 100 
      : 100;
    
    return {
      status: healthScore > 80 ? 'HEALTHY' : healthScore > 50 ? 'DEGRADED' : 'UNHEALTHY',
      healthScore: Math.round(healthScore),
      metrics,
      timestamp: new Date().toISOString()
    };
  }
}

module.exports = {
  ServiceOrchestrator,
  ServiceRegistry,
  SagaOrchestrator,
  SagaStep
};
