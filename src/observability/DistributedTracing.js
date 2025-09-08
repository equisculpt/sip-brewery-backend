/**
 * 🔍 ENTERPRISE DISTRIBUTED TRACING & OBSERVABILITY
 * 
 * OpenTelemetry-based distributed tracing with metrics, logging, and APM
 * Provides end-to-end visibility across microservices
 * 
 * @author Senior AI Backend Developer (35+ years)
 * @version 3.0.0
 */

const { NodeSDK } = require('@opentelemetry/sdk-node');
const { Resource } = require('@opentelemetry/resources');
const { SemanticResourceAttributes } = require('@opentelemetry/semantic-conventions');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');
const { PrometheusExporter } = require('@opentelemetry/exporter-prometheus');
const { getNodeAutoInstrumentations } = require('@opentelemetry/auto-instrumentations-node');
const { trace, metrics, context, SpanStatusCode, SpanKind } = require('@opentelemetry/api');
const { v4: uuidv4 } = require('uuid');
const logger = require('../utils/logger');

class EnterpriseObservability {
  constructor() {
    this.serviceName = process.env.SERVICE_NAME || 'sip-brewery-backend';
    this.serviceVersion = process.env.SERVICE_VERSION || '3.0.0';
    this.environment = process.env.NODE_ENV || 'development';
    this.jaegerEndpoint = process.env.JAEGER_ENDPOINT || 'http://localhost:14268/api/traces';
    this.prometheusPort = parseInt(process.env.PROMETHEUS_PORT) || 9090;
    
    this.tracer = null;
    this.meter = null;
    this.sdk = null;
    this.customMetrics = new Map();
    this.activeSpans = new Map();
  }

  /**
   * Initialize OpenTelemetry SDK
   */
  async initialize() {
    try {
      // Create resource
      const resource = new Resource({
        [SemanticResourceAttributes.SERVICE_NAME]: this.serviceName,
        [SemanticResourceAttributes.SERVICE_VERSION]: this.serviceVersion,
        [SemanticResourceAttributes.DEPLOYMENT_ENVIRONMENT]: this.environment,
        [SemanticResourceAttributes.SERVICE_INSTANCE_ID]: process.pid.toString()
      });

      // Configure SDK
      this.sdk = new NodeSDK({
        resource,
        traceExporter: new JaegerExporter({
          endpoint: this.jaegerEndpoint
        }),
        metricExporter: new PrometheusExporter({
          port: this.prometheusPort
        }),
        instrumentations: [
          getNodeAutoInstrumentations({
            '@opentelemetry/instrumentation-fs': {
              enabled: false // Disable file system instrumentation for performance
            }
          })
        ]
      });

      // Start SDK
      await this.sdk.start();

      // Get tracer and meter
      this.tracer = trace.getTracer(this.serviceName, this.serviceVersion);
      this.meter = metrics.getMeter(this.serviceName, this.serviceVersion);

      // Initialize custom metrics
      this.initializeCustomMetrics();

      logger.info('✅ Distributed tracing initialized', {
        serviceName: this.serviceName,
        serviceVersion: this.serviceVersion,
        environment: this.environment
      });

    } catch (error) {
      logger.error('❌ Failed to initialize distributed tracing:', error);
      throw error;
    }
  }

  /**
   * Initialize custom business metrics
   */
  initializeCustomMetrics() {
    // Portfolio metrics
    this.customMetrics.set('portfolio_operations', this.meter.createCounter('portfolio_operations_total', {
      description: 'Total portfolio operations'
    }));

    this.customMetrics.set('portfolio_value', this.meter.createHistogram('portfolio_value_distribution', {
      description: 'Portfolio value distribution',
      unit: 'INR'
    }));

    // Investment metrics
    this.customMetrics.set('investment_transactions', this.meter.createCounter('investment_transactions_total', {
      description: 'Total investment transactions'
    }));

    this.customMetrics.set('sip_amount', this.meter.createHistogram('sip_amount_distribution', {
      description: 'SIP amount distribution',
      unit: 'INR'
    }));

    // AI/ML metrics
    this.customMetrics.set('ai_predictions', this.meter.createCounter('ai_predictions_total', {
      description: 'Total AI predictions made'
    }));

    this.customMetrics.set('prediction_accuracy', this.meter.createHistogram('prediction_accuracy_score', {
      description: 'AI prediction accuracy scores'
    }));

    // Performance metrics
    this.customMetrics.set('database_operations', this.meter.createCounter('database_operations_total', {
      description: 'Total database operations'
    }));

    this.customMetrics.set('cache_operations', this.meter.createCounter('cache_operations_total', {
      description: 'Total cache operations'
    }));

    this.customMetrics.set('api_response_time', this.meter.createHistogram('api_response_time_seconds', {
      description: 'API response time distribution',
      unit: 'seconds'
    }));

    // Business metrics
    this.customMetrics.set('user_registrations', this.meter.createCounter('user_registrations_total', {
      description: 'Total user registrations'
    }));

    this.customMetrics.set('active_users', this.meter.createUpDownCounter('active_users_current', {
      description: 'Current active users'
    }));

    logger.info('📊 Custom metrics initialized', {
      metricsCount: this.customMetrics.size
    });
  }

  /**
   * Create a new span for operation tracing
   */
  startSpan(operationName, options = {}) {
    const span = this.tracer.startSpan(operationName, {
      kind: options.kind || SpanKind.INTERNAL,
      attributes: {
        'service.name': this.serviceName,
        'service.version': this.serviceVersion,
        'operation.type': options.operationType || 'business',
        'user.id': options.userId || 'anonymous',
        'correlation.id': options.correlationId || uuidv4(),
        ...options.attributes
      }
    });

    // Store span for potential cleanup
    const spanId = uuidv4();
    this.activeSpans.set(spanId, span);

    return {
      span,
      spanId,
      finish: (status = SpanStatusCode.OK, error = null) => {
        if (error) {
          span.recordException(error);
          span.setStatus({ code: SpanStatusCode.ERROR, message: error.message });
        } else {
          span.setStatus({ code: status });
        }
        span.end();
        this.activeSpans.delete(spanId);
      },
      addEvent: (name, attributes = {}) => {
        span.addEvent(name, attributes);
      },
      setAttributes: (attributes) => {
        span.setAttributes(attributes);
      }
    };
  }

  /**
   * Trace a function execution
   */
  async traceFunction(operationName, fn, options = {}) {
    const { span, finish } = this.startSpan(operationName, options);
    
    try {
      const result = await context.with(trace.setSpan(context.active(), span), fn);
      finish(SpanStatusCode.OK);
      return result;
    } catch (error) {
      finish(SpanStatusCode.ERROR, error);
      throw error;
    }
  }

  /**
   * Record custom metric
   */
  recordMetric(metricName, value, attributes = {}) {
    const metric = this.customMetrics.get(metricName);
    if (!metric) {
      logger.warn('⚠️ Unknown metric', { metricName });
      return;
    }

    try {
      if (typeof metric.add === 'function') {
        // Counter or UpDownCounter
        metric.add(value, attributes);
      } else if (typeof metric.record === 'function') {
        // Histogram
        metric.record(value, attributes);
      }

      logger.debug('📊 Metric recorded', { metricName, value, attributes });
    } catch (error) {
      logger.warn('⚠️ Failed to record metric', { metricName, error: error.message });
    }
  }

  /**
   * Trace portfolio operations
   */
  async tracePortfolioOperation(operationType, userId, portfolioId, fn) {
    return this.traceFunction(`portfolio.${operationType}`, fn, {
      operationType: 'portfolio',
      userId,
      attributes: {
        'portfolio.id': portfolioId,
        'portfolio.operation': operationType
      }
    });
  }

  /**
   * Trace investment operations
   */
  async traceInvestmentOperation(operationType, userId, amount, fundCode, fn) {
    const result = await this.traceFunction(`investment.${operationType}`, fn, {
      operationType: 'investment',
      userId,
      attributes: {
        'investment.operation': operationType,
        'investment.amount': amount,
        'investment.fund_code': fundCode
      }
    });

    // Record business metrics
    this.recordMetric('investment_transactions', 1, {
      operation: operationType,
      fund_code: fundCode
    });

    if (operationType === 'sip') {
      this.recordMetric('sip_amount', amount, {
        fund_code: fundCode
      });
    }

    return result;
  }

  /**
   * Trace AI/ML operations
   */
  async traceAIOperation(operationType, modelName, inputSize, fn) {
    const result = await this.traceFunction(`ai.${operationType}`, fn, {
      operationType: 'ai',
      attributes: {
        'ai.operation': operationType,
        'ai.model': modelName,
        'ai.input_size': inputSize
      }
    });

    // Record AI metrics
    this.recordMetric('ai_predictions', 1, {
      operation: operationType,
      model: modelName
    });

    return result;
  }

  /**
   * Trace database operations
   */
  async traceDatabaseOperation(operation, collection, query, fn) {
    const result = await this.traceFunction(`db.${operation}`, fn, {
      kind: SpanKind.CLIENT,
      operationType: 'database',
      attributes: {
        'db.operation': operation,
        'db.collection.name': collection,
        'db.system': 'mongodb'
      }
    });

    // Record database metrics
    this.recordMetric('database_operations', 1, {
      operation,
      collection
    });

    return result;
  }

  /**
   * Trace cache operations
   */
  async traceCacheOperation(operation, key, fn) {
    const result = await this.traceFunction(`cache.${operation}`, fn, {
      kind: SpanKind.CLIENT,
      operationType: 'cache',
      attributes: {
        'cache.operation': operation,
        'cache.key': key,
        'cache.system': 'redis'
      }
    });

    // Record cache metrics
    this.recordMetric('cache_operations', 1, {
      operation
    });

    return result;
  }

  /**
   * Trace HTTP requests
   */
  traceHTTPRequest(req, res, next) {
    const startTime = Date.now();
    const { span, finish } = this.startSpan(`HTTP ${req.method} ${req.route?.path || req.path}`, {
      kind: SpanKind.SERVER,
      operationType: 'http',
      userId: req.user?.id,
      correlationId: req.correlationId,
      attributes: {
        'http.method': req.method,
        'http.url': req.url,
        'http.route': req.route?.path || req.path,
        'http.user_agent': req.get('User-Agent'),
        'http.remote_addr': req.ip
      }
    });

    // Attach span to request for downstream use
    req.span = span;

    res.on('finish', () => {
      const responseTime = (Date.now() - startTime) / 1000;
      
      span.setAttributes({
        'http.status_code': res.statusCode,
        'http.response_size': res.get('Content-Length') || 0
      });

      // Record API metrics
      this.recordMetric('api_response_time', responseTime, {
        method: req.method,
        route: req.route?.path || req.path,
        status_code: res.statusCode.toString()
      });

      const status = res.statusCode >= 400 ? SpanStatusCode.ERROR : SpanStatusCode.OK;
      finish(status);
    });

    next();
  }

  /**
   * Create correlation context
   */
  createCorrelationContext(correlationId, userId = null) {
    return context.setSpan(
      context.active(),
      this.tracer.startSpan('correlation_context', {
        attributes: {
          'correlation.id': correlationId,
          'user.id': userId
        }
      })
    );
  }

  /**
   * Get current trace context
   */
  getCurrentTraceContext() {
    const span = trace.getActiveSpan();
    if (!span) return null;

    const spanContext = span.spanContext();
    return {
      traceId: spanContext.traceId,
      spanId: spanContext.spanId,
      traceFlags: spanContext.traceFlags
    };
  }

  /**
   * Record business event
   */
  recordBusinessEvent(eventName, attributes = {}) {
    const span = trace.getActiveSpan();
    if (span) {
      span.addEvent(`business.${eventName}`, {
        timestamp: Date.now(),
        ...attributes
      });
    }

    logger.info('📈 Business event recorded', { eventName, attributes });
  }

  /**
   * Get observability metrics
   */
  getMetrics() {
    return {
      activeSpans: this.activeSpans.size,
      customMetrics: this.customMetrics.size,
      serviceName: this.serviceName,
      serviceVersion: this.serviceVersion,
      environment: this.environment,
      traceContext: this.getCurrentTraceContext()
    };
  }

  /**
   * Graceful shutdown
   */
  async shutdown() {
    try {
      // Finish any active spans
      for (const [spanId, span] of this.activeSpans.entries()) {
        span.setStatus({ code: SpanStatusCode.ERROR, message: 'Service shutdown' });
        span.end();
        this.activeSpans.delete(spanId);
      }

      // Shutdown SDK
      if (this.sdk) {
        await this.sdk.shutdown();
      }

      logger.info('✅ Observability shutdown complete');
    } catch (error) {
      logger.error('❌ Observability shutdown error:', error);
    }
  }
}

/**
 * Decorator for tracing class methods
 */
function traced(operationName, options = {}) {
  return function(target, propertyName, descriptor) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function(...args) {
      const observability = global.observability || new EnterpriseObservability();
      
      return observability.traceFunction(
        operationName || `${target.constructor.name}.${propertyName}`,
        () => originalMethod.apply(this, args),
        options
      );
    };
    
    return descriptor;
  };
}

/**
 * Express middleware for automatic tracing
 */
function tracingMiddleware(observability) {
  return (req, res, next) => {
    observability.traceHTTPRequest(req, res, next);
  };
}

module.exports = { 
  EnterpriseObservability, 
  traced, 
  tracingMiddleware 
};
