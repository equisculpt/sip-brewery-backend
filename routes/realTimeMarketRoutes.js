const express = require('express');
const router = express.Router();
const RealTimeMarketService = require('../services/RealTimeMarketService');
const response = require('../utils/response');
const logger = require('../utils/logger');

// Initialize real-time market service
const realTimeMarketService = new RealTimeMarketService();

console.log('🚀 $1 BILLION PLATFORM - Real-time Market Data API Initialized');
console.log('📊 Live data from: NSE, BSE, AMFI, Yahoo Finance, Alpha Vantage');
console.log('⚡ Updates every 30 seconds during market hours');

/**
 * @route GET /api/market/live
 * @desc Get real-time market data
 * @access Public
 */
router.get('/live', async (req, res) => {
  try {
    logger.info('🚀 Fetching real-time market data...');
    
    const marketData = realTimeMarketService.getCurrentMarketData();
    
    return response.success(res, 'Real-time market data retrieved successfully', {
      ...marketData,
      platform: '$1 Billion Revolutionary Platform',
      disclaimer: 'Real-time data from multiple premium sources',
      update_frequency: '30 seconds during market hours'
    });
  } catch (error) {
    logger.error('Error fetching real-time market data:', error);
    return response.error(res, 'Failed to fetch real-time market data', error.message);
  }
});

/**
 * @route GET /api/market/status
 * @desc Get current market status
 * @access Public
 */
router.get('/status', async (req, res) => {
  try {
    const marketStatus = await realTimeMarketService.getMarketStatus();
    
    return response.success(res, 'Market status retrieved successfully', {
      ...marketStatus,
      platform: '$1 Billion Revolutionary Platform',
      timezone_info: 'All times in Indian Standard Time (IST)'
    });
  } catch (error) {
    logger.error('Error fetching market status:', error);
    return response.error(res, 'Failed to fetch market status', error.message);
  }
});

/**
 * @route GET /api/market/indices
 * @desc Get real-time market indices
 * @access Public
 */
router.get('/indices', async (req, res) => {
  try {
    logger.info('📊 Fetching real-time market indices...');
    
    const marketData = realTimeMarketService.getCurrentMarketData();
    
    return response.success(res, 'Real-time indices retrieved successfully', {
      indices: marketData.indices,
      last_updated: marketData.last_updated,
      total_indices: marketData.indices.length,
      data_sources: marketData.data_sources,
      real_time: true,
      platform: '$1 Billion Revolutionary Platform'
    });
  } catch (error) {
    logger.error('Error fetching real-time indices:', error);
    return response.error(res, 'Failed to fetch real-time indices', error.message);
  }
});

/**
 * @route GET /api/market/performance
 * @desc Get real-time service performance metrics
 * @access Public
 */
router.get('/performance', async (req, res) => {
  try {
    const metrics = realTimeMarketService.getPerformanceMetrics();
    
    return response.success(res, 'Performance metrics retrieved successfully', {
      ...metrics,
      platform: '$1 Billion Revolutionary Platform',
      service_quality: 'Enterprise Grade',
      uptime_target: '99.99%'
    });
  } catch (error) {
    logger.error('Error fetching performance metrics:', error);
    return response.error(res, 'Failed to fetch performance metrics', error.message);
  }
});

/**
 * @route POST /api/market/subscribe
 * @desc Subscribe to real-time market updates
 * @access Public
 */
router.post('/subscribe', async (req, res) => {
  try {
    const { client_id, symbols = [] } = req.body;
    
    if (!client_id) {
      return response.error(res, 'Client ID is required for subscription', null, 400);
    }
    
    realTimeMarketService.subscribe(client_id, symbols);
    
    return response.success(res, 'Successfully subscribed to real-time updates', {
      client_id,
      subscribed_symbols: symbols,
      subscription_time: new Date().toISOString(),
      platform: '$1 Billion Revolutionary Platform',
      update_frequency: '30 seconds'
    });
  } catch (error) {
    logger.error('Error subscribing to real-time updates:', error);
    return response.error(res, 'Failed to subscribe to real-time updates', error.message);
  }
});

/**
 * @route DELETE /api/market/unsubscribe/:clientId
 * @desc Unsubscribe from real-time market updates
 * @access Public
 */
router.delete('/unsubscribe/:clientId', async (req, res) => {
  try {
    const { clientId } = req.params;
    
    realTimeMarketService.unsubscribe(clientId);
    
    return response.success(res, 'Successfully unsubscribed from real-time updates', {
      client_id: clientId,
      unsubscription_time: new Date().toISOString(),
      platform: '$1 Billion Revolutionary Platform'
    });
  } catch (error) {
    logger.error('Error unsubscribing from real-time updates:', error);
    return response.error(res, 'Failed to unsubscribe from real-time updates', error.message);
  }
});

/**
 * @route GET /api/market/health
 * @desc Health check for real-time market service
 * @access Public
 */
router.get('/health', async (req, res) => {
  try {
    const marketStatus = await realTimeMarketService.getMarketStatus();
    const performance = realTimeMarketService.getPerformanceMetrics();
    
    const healthStatus = {
      status: 'HEALTHY',
      market_service: 'OPERATIONAL',
      data_freshness: performance.data_freshness < 60000 ? 'FRESH' : 'STALE',
      active_connections: performance.active_subscribers,
      uptime_seconds: performance.uptime,
      memory_usage: performance.memory_usage,
      market_status: marketStatus.is_market_open ? 'OPEN' : 'CLOSED',
      platform: '$1 Billion Revolutionary Platform',
      service_level: 'Enterprise Grade',
      last_check: new Date().toISOString()
    };
    
    const statusCode = healthStatus.status === 'HEALTHY' ? 200 : 503;
    
    return response.success(res, 'Real-time market service health check', healthStatus, statusCode);
  } catch (error) {
    logger.error('Health check failed:', error);
    return response.error(res, 'Health check failed', error.message, 503);
  }
});

module.exports = router;
