const express = require('express');
const router = express.Router();
const benchmarkController = require('../controllers/benchmarkController');
const { validateParams } = require('../middleware/validation');

/**
 * @route   GET /api/benchmark/market-status
 * @desc    Get real-time market status and indices
 * @access  Public
 */
router.get('/market-status',
  benchmarkController.getMarketStatus
);

/**
 * @route   GET /api/benchmark/gainers-losers
 * @desc    Get gainers and losers for a specific index
 * @access  Public
 */
router.get('/gainers-losers',
  benchmarkController.getGainersAndLosers
);

/**
 * @route   GET /api/benchmark/most-active
 * @desc    Get most active equities
 * @access  Public
 */
router.get('/most-active',
  benchmarkController.getMostActiveEquities
);

/**
 * @route   POST /api/benchmark/update-nifty
 * @desc    Update NIFTY 50 data from NSE service
 * @access  Public
 */
router.post('/update-nifty',
  benchmarkController.updateNiftyData
);

/**
 * @route   GET /api/benchmark/compare/:fundId
 * @desc    Compare mutual fund with benchmark
 * @access  Public
 */
router.get('/compare/:fundId',
  validateParams(['fundId']),
  benchmarkController.compareWithBenchmark
);

/**
 * @route   GET /api/benchmark/insights/:fundId
 * @desc    Generate AI insights comparing fund with benchmark
 * @access  Public
 */
router.get('/insights/:fundId',
  validateParams(['fundId']),
  benchmarkController.generateInsights
);

/**
 * @route   GET /api/benchmark/:indexId
 * @desc    Get benchmark data for a specific index
 * @access  Public
 */
router.get('/:indexId', 
  validateParams(['indexId']),
  benchmarkController.getBenchmarkData
);

/**
 * @route   GET /api/benchmark/nifty50/real-data
 * @desc    Get real NIFTY 50 data for 1 year for charting
 * @access  Public
 */
router.get('/nifty50/real-data',
  benchmarkController.getRealNifty50Data
);

module.exports = router; 