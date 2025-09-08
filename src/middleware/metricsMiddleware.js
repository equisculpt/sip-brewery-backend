/**
 * 📊 PRODUCTION METRICS MIDDLEWARE
 * 
 * Comprehensive metrics collection for production monitoring
 * Prometheus-compatible metrics with business intelligence
 */

const prometheus = require('prom-client');
const logger = require('../utils/logger');

// Create a Registry to register metrics
const register = new prometheus.Registry();

// Add default metrics (CPU, Memory, etc.)
prometheus.collectDefaultMetrics({
  register,
  prefix: 'sipbrewery_',
  gcDurationBuckets: [0.001, 0.01, 0.1, 1, 2, 5]
});

// 🌐 HTTP Request Metrics
const httpRequestsTotal = new prometheus.Counter({
  name: 'sipbrewery_http_requests_total',
  help: 'Total number of HTTP requests',
  labelNames: ['method', 'route', 'status_code', 'user_type'],
  registers: [register]
});

const httpRequestDuration = new prometheus.Histogram({
  name: 'sipbrewery_http_request_duration_seconds',
  help: 'Duration of HTTP requests in seconds',
  labelNames: ['method', 'route', 'status_code'],
  buckets: [0.1, 0.3, 0.5, 0.7, 1, 3, 5, 7, 10],
  registers: [register]
});

// 🧠 ASI System Metrics
const asiRequestsTotal = new prometheus.Counter({
  name: 'sipbrewery_asi_requests_total',
  help: 'Total ASI requests processed',
  labelNames: ['type', 'capability', 'status'],
  registers: [register]
});

const asiRequestsFailedTotal = new prometheus.Counter({
  name: 'sipbrewery_asi_requests_failed_total',
  help: 'Total ASI requests that failed',
  labelNames: ['type', 'capability', 'error_type'],
  registers: [register]
});

const asiRequestDuration = new prometheus.Histogram({
  name: 'sipbrewery_asi_request_duration_seconds',
  help: 'Duration of ASI requests in seconds',
  labelNames: ['type', 'capability'],
  buckets: [0.5, 1, 2, 5, 10, 30, 60],
  registers: [register]
});

const asiSystemHealth = new prometheus.Gauge({
  name: 'sipbrewery_asi_system_health',
  help: 'ASI system health status (1=healthy, 0=unhealthy)',
  registers: [register]
});

const asiFinanceRating = new prometheus.Gauge({
  name: 'sipbrewery_asi_finance_rating',
  help: 'Current ASI finance rating (0-10)',
  registers: [register]
});

const pythonAsiBridgeHealth = new prometheus.Gauge({
  name: 'sipbrewery_python_asi_bridge_health',
  help: 'Python ASI bridge health (1=healthy, 0=unhealthy)',
  registers: [register]
});

// 💼 Business Metrics
const userRegistrationsTotal = new prometheus.Counter({
  name: 'sipbrewery_user_registrations_total',
  help: 'Total user registrations',
  labelNames: ['user_type', 'source'],
  registers: [register]
});

const portfolioAnalysisTotal = new prometheus.Counter({
  name: 'sipbrewery_portfolio_analysis_total',
  help: 'Total portfolio analyses performed',
  labelNames: ['type', 'status'],
  registers: [register]
});

const portfolioAnalysisFailuresTotal = new prometheus.Counter({
  name: 'sipbrewery_portfolio_analysis_failures_total',
  help: 'Total portfolio analysis failures',
  labelNames: ['error_type'],
  registers: [register]
});

const totalAumInr = new prometheus.Gauge({
  name: 'sipbrewery_total_aum_inr',
  help: 'Total Assets Under Management in INR',
  registers: [register]
});

const activeUsersGauge = new prometheus.Gauge({
  name: 'sipbrewery_active_users',
  help: 'Number of active users',
  labelNames: ['period'],
  registers: [register]
});

// 🔒 Security Metrics
const authFailuresTotal = new prometheus.Counter({
  name: 'sipbrewery_auth_failures_total',
  help: 'Total authentication failures',
  labelNames: ['type', 'reason'],
  registers: [register]
});

const failedLoginAttemptsTotal = new prometheus.Counter({
  name: 'sipbrewery_failed_login_attempts_total',
  help: 'Total failed login attempts',
  labelNames: ['ip', 'user_agent'],
  registers: [register]
});

const rateLimitExceededTotal = new prometheus.Counter({
  name: 'sipbrewery_rate_limit_exceeded_total',
  help: 'Total rate limit violations',
  labelNames: ['endpoint', 'ip'],
  registers: [register]
});

const suspiciousRequestsTotal = new prometheus.Counter({
  name: 'sipbrewery_suspicious_requests_total',
  help: 'Total suspicious requests detected',
  labelNames: ['type', 'severity'],
  registers: [register]
});

// 📊 Data Quality Metrics
const mutualFundDataLastUpdate = new prometheus.Gauge({
  name: 'sipbrewery_mutual_fund_data_last_update_timestamp',
  help: 'Timestamp of last mutual fund data update',
  registers: [register]
});

const marketDataLastUpdate = new prometheus.Gauge({
  name: 'sipbrewery_market_data_last_update_timestamp',
  help: 'Timestamp of last market data update',
  registers: [register]
});

const financialDataSourcesHealthy = new prometheus.Gauge({
  name: 'sipbrewery_financial_data_sources_healthy',
  help: 'Number of healthy financial data sources',
  registers: [register]
});

// 🗄️ Database Metrics
const databaseConnectionsActive = new prometheus.Gauge({
  name: 'sipbrewery_database_connections_active',
  help: 'Active database connections',
  labelNames: ['database'],
  registers: [register]
});

const databaseQueryDuration = new prometheus.Histogram({
  name: 'sipbrewery_database_query_duration_seconds',
  help: 'Database query duration in seconds',
  labelNames: ['operation', 'collection'],
  buckets: [0.01, 0.05, 0.1, 0.5, 1, 2, 5],
  registers: [register]
});

/**
 * Middleware to collect HTTP metrics
 */
function collectHttpMetrics(req, res, next) {
  const start = Date.now();
  
  // Extract route pattern (remove IDs for better grouping)
  const route = req.route?.path || req.path.replace(/\/\d+/g, '/:id');
  
  res.on('finish', () => {
    const duration = (Date.now() - start) / 1000;
    const userType = req.user?.type || 'anonymous';
    
    // Record metrics
    httpRequestsTotal
      .labels(req.method, route, res.statusCode.toString(), userType)
      .inc();
    
    httpRequestDuration
      .labels(req.method, route, res.statusCode.toString())
      .observe(duration);
    
    // Log slow requests
    if (duration > 5) {
      logger.warn('Slow request detected', {
        method: req.method,
        route,
        duration,
        statusCode: res.statusCode,
        userAgent: req.get('User-Agent'),
        ip: req.ip
      });
    }
  });
  
  next();
}

/**
 * Record ASI request metrics
 */
function recordASIMetrics(type, capability, status, duration, error = null) {
  asiRequestsTotal.labels(type, capability, status).inc();
  
  if (status === 'failed') {
    const errorType = error?.name || 'unknown';
    asiRequestsFailedTotal.labels(type, capability, errorType).inc();
  }
  
  if (duration) {
    asiRequestDuration.labels(type, capability).observe(duration / 1000);
  }
}

/**
 * Record business metrics
 */
function recordBusinessMetrics(metric, value, labels = {}) {
  switch (metric) {
    case 'user_registration':
      userRegistrationsTotal.labels(labels.userType || 'individual', labels.source || 'direct').inc();
      break;
    
    case 'portfolio_analysis':
      portfolioAnalysisTotal.labels(labels.type || 'basic', labels.status || 'success').inc();
      break;
    
    case 'portfolio_analysis_failure':
      portfolioAnalysisFailuresTotal.labels(labels.errorType || 'unknown').inc();
      break;
    
    case 'total_aum':
      totalAumInr.set(value);
      break;
    
    case 'active_users':
      activeUsersGauge.labels(labels.period || 'daily').set(value);
      break;
  }
}

/**
 * Record security metrics
 */
function recordSecurityMetrics(metric, labels = {}) {
  switch (metric) {
    case 'auth_failure':
      authFailuresTotal.labels(labels.type || 'unknown', labels.reason || 'invalid_credentials').inc();
      break;
    
    case 'failed_login':
      failedLoginAttemptsTotal.labels(labels.ip || 'unknown', labels.userAgent || 'unknown').inc();
      break;
    
    case 'rate_limit_exceeded':
      rateLimitExceededTotal.labels(labels.endpoint || 'unknown', labels.ip || 'unknown').inc();
      break;
    
    case 'suspicious_request':
      suspiciousRequestsTotal.labels(labels.type || 'unknown', labels.severity || 'medium').inc();
      break;
  }
}

/**
 * Update system health metrics
 */
function updateSystemHealth(asiHealth, asiRating, pythonBridgeHealth) {
  asiSystemHealth.set(asiHealth ? 1 : 0);
  if (asiRating !== undefined) {
    asiFinanceRating.set(asiRating);
  }
  pythonAsiBridgeHealth.set(pythonBridgeHealth ? 1 : 0);
}

/**
 * Update data quality metrics
 */
function updateDataQualityMetrics(mutualFundTimestamp, marketDataTimestamp, healthySourcesCount) {
  if (mutualFundTimestamp) {
    mutualFundDataLastUpdate.set(mutualFundTimestamp);
  }
  if (marketDataTimestamp) {
    marketDataLastUpdate.set(marketDataTimestamp);
  }
  if (healthySourcesCount !== undefined) {
    financialDataSourcesHealthy.set(healthySourcesCount);
  }
}

/**
 * Record database metrics
 */
function recordDatabaseMetrics(operation, collection, duration, activeConnections) {
  if (duration) {
    databaseQueryDuration.labels(operation, collection).observe(duration / 1000);
  }
  if (activeConnections !== undefined) {
    databaseConnectionsActive.labels('mongodb').set(activeConnections);
  }
}

/**
 * Get metrics endpoint handler
 */
function getMetrics(req, res) {
  res.set('Content-Type', register.contentType);
  res.end(register.metrics());
}

/**
 * Get business metrics endpoint handler
 */
async function getBusinessMetrics(req, res) {
  try {
    // Calculate real-time business metrics
    const businessMetrics = {
      timestamp: Date.now(),
      totalAUM: totalAumInr._getValue(),
      activeUsers: {
        daily: activeUsersGauge.labels('daily')._getValue(),
        weekly: activeUsersGauge.labels('weekly')._getValue(),
        monthly: activeUsersGauge.labels('monthly')._getValue()
      },
      registrations: {
        total: userRegistrationsTotal._getValue(),
        today: await getTodayRegistrations()
      },
      portfolioAnalyses: {
        total: portfolioAnalysisTotal._getValue(),
        successRate: await getPortfolioAnalysisSuccessRate()
      }
    };
    
    res.json(businessMetrics);
  } catch (error) {
    logger.error('Error getting business metrics:', error);
    res.status(500).json({ error: 'Failed to get business metrics' });
  }
}

/**
 * Helper functions for business metrics
 */
async function getTodayRegistrations() {
  // This would query your database for today's registrations
  // Placeholder implementation
  return 0;
}

async function getPortfolioAnalysisSuccessRate() {
  // Calculate success rate from metrics
  const total = portfolioAnalysisTotal._getValue();
  const failures = portfolioAnalysisFailuresTotal._getValue();
  return total > 0 ? ((total - failures) / total * 100).toFixed(2) : 100;
}

module.exports = {
  register,
  collectHttpMetrics,
  recordASIMetrics,
  recordBusinessMetrics,
  recordSecurityMetrics,
  updateSystemHealth,
  updateDataQualityMetrics,
  recordDatabaseMetrics,
  getMetrics,
  getBusinessMetrics,
  
  // Export individual metrics for direct access
  metrics: {
    httpRequestsTotal,
    httpRequestDuration,
    asiRequestsTotal,
    asiRequestsFailedTotal,
    asiRequestDuration,
    asiSystemHealth,
    asiFinanceRating,
    pythonAsiBridgeHealth,
    userRegistrationsTotal,
    portfolioAnalysisTotal,
    totalAumInr,
    activeUsersGauge,
    authFailuresTotal,
    failedLoginAttemptsTotal,
    rateLimitExceededTotal,
    suspiciousRequestsTotal,
    mutualFundDataLastUpdate,
    marketDataLastUpdate,
    financialDataSourcesHealthy,
    databaseConnectionsActive,
    databaseQueryDuration
  }
};
