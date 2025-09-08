const benchmarkService = require('../services/benchmarkService');
const aiService = require('../services/aiService');
const response = require('../utils/response');
const logger = require('../utils/logger');

class BenchmarkController {
  /**
   * Get benchmark data for a specific date range, with optional limit for latest N records
   */
  async getBenchmarkData(req, res) {
    try {
      const { indexId } = req.params;
      const { from, to, limit } = req.query;

      logger.info('Fetching benchmark data', { indexId, from, to, limit });

      const data = await benchmarkService.getBenchmarkData(indexId, from, to);

      // If limit is specified, return only the latest N records
      let resultData = data.data;
      if (limit) {
        resultData = resultData.slice(-parseInt(limit));
      }

      return response.success(res, {
        ...data,
        data: resultData
      }, 'Benchmark data retrieved successfully');

    } catch (error) {
      logger.error('Error fetching benchmark data', { error: error.message });
      return response.error(res, 'Failed to fetch benchmark data', 500, error);
    }
  }

  /**
   * Get real-time market status and indices
   */
  async getMarketStatus(req, res) {
    try {
      logger.info('Fetching market status');

      const marketStatus = await benchmarkService.getMarketStatus();

      return response.success(res, marketStatus, 'Market status retrieved successfully');

    } catch (error) {
      logger.error('Error fetching market status', { error: error.message });
      return response.error(res, 'Failed to fetch market status', 500, error);
    }
  }

  /**
   * Get gainers and losers for a specific index
   */
  async getGainersAndLosers(req, res) {
    try {
      const { indexName = 'NIFTY 50' } = req.query;

      logger.info('Fetching gainers and losers', { indexName });

      const data = await benchmarkService.getGainersAndLosers(indexName);

      return response.success(res, data, 'Gainers and losers retrieved successfully');

    } catch (error) {
      logger.error('Error fetching gainers and losers', { error: error.message });
      return response.error(res, 'Failed to fetch gainers and losers', 500, error);
    }
  }

  /**
   * Get most active equities
   */
  async getMostActiveEquities(req, res) {
    try {
      logger.info('Fetching most active equities');

      const data = await benchmarkService.getMostActiveEquities();

      return response.success(res, data, 'Most active equities retrieved successfully');

    } catch (error) {
      logger.error('Error fetching most active equities', { error: error.message });
      return response.error(res, 'Failed to fetch most active equities', 500, error);
    }
  }

  /**
   * Compare mutual fund with benchmark
   */
  async compareWithBenchmark(req, res) {
    try {
      const { fundId } = req.params;
      const { benchmark = 'NIFTY50', range = '3Y' } = req.query;

      // For now, we'll use a sample fund data
      // In production, this would fetch from your fund database
      const sampleFundData = [
        { date: '01-01-2023', nav: 100 },
        { date: '02-01-2023', nav: 102 },
        { date: '03-01-2023', nav: 105 },
        // ... more data points
      ];

      logger.info('Comparing fund with benchmark', { fundId, benchmark, range });

      const comparison = await benchmarkService.compareWithBenchmark(
        sampleFundData, 
        benchmark, 
        range
      );

      return response.success(res, comparison, 'Comparison completed successfully');

    } catch (error) {
      logger.error('Error comparing with benchmark', { error: error.message });
      return response.error(res, 'Failed to compare with benchmark', 500, error);
    }
  }

  /**
   * Generate AI insights comparing fund with benchmark
   */
  async generateInsights(req, res) {
    try {
      const { fundId } = req.params;
      const { vs = 'NIFTY50' } = req.query;

      logger.info('Generating AI insights', { fundId, benchmark: vs });

      // Get fund data (this would come from your fund database)
      const fundData = await this.getFundData(fundId);
      
      // Compare with benchmark
      const comparison = await benchmarkService.compareWithBenchmark(
        fundData.navHistory,
        vs,
        '3Y'
      );

      // Generate chart
      const chartBuffer = await benchmarkService.generateComparisonChart(
        comparison.fund.data,
        comparison.benchmark.data,
        fundData.schemeName,
        'NIFTY 50'
      );

      // Prepare comprehensive data for AI analysis
      const analysisData = {
        fund: {
          name: fundData.schemeName,
          schemeCode: fundData.schemeCode,
          navHistory: comparison.fund.data,
          analytics: comparison.analytics
        },
        benchmark: {
          name: 'NIFTY 50',
          data: comparison.benchmark.data,
          analytics: comparison.analytics
        },
        comparison: comparison.analytics
      };

      // Generate AI insights
      const query = `Provide a comprehensive investment analysis comparing ${fundData.schemeName} with NIFTY 50. Include:
      1. Performance comparison vs NIFTY 50
      2. Strong points and competitive advantages
      3. Weak points and risk factors
      4. Beta and Alpha analysis
      5. Investment recommendation (Invest/Don't Invest) with reasoning
      6. Risk assessment and suitability for different investor profiles
      7. Market timing considerations
      8. Portfolio allocation recommendations`;

      const aiResult = await aiService.analyzeFundData([analysisData], query);

      return response.success(res, {
        insights: aiResult.analysis,
        comparison: comparison,
        chart: chartBuffer.toString('base64') // Return chart as base64
      }, 'AI insights generated successfully');

    } catch (error) {
      logger.error('Error generating insights', { error: error.message });
      return response.error(res, 'Failed to generate insights', 500, error);
    }
  }

  /**
   * Update NIFTY 50 data
   */
  async updateNiftyData(req, res) {
    try {
      logger.info('Updating NIFTY 50 data');

      const result = await benchmarkService.updateNifty50Data();

      return response.success(res, result, 'NIFTY 50 data updated successfully');
    } catch (error) {
      logger.error('Error updating NIFTY 50 data', { error: error.message });
      return response.error(res, 'Failed to update NIFTY 50 data', 500, error);
    }
  }

  /**
   * Get real NIFTY 50 data for 1 year for charting
   */
  async getRealNifty50Data(req, res) {
    try {
      logger.info('Fetching real NIFTY 50 data for 1 year');

      const realNiftyDataService = require('../services/realNiftyDataService');
      const data = await realNiftyDataService.getNifty50OneYearData();

      return response.success(res, data, 'Real NIFTY 50 data retrieved successfully');

    } catch (error) {
      logger.error('Error fetching real NIFTY 50 data', { error: error.message });
      return response.error(res, 'Failed to fetch real NIFTY 50 data', 500, error);
    }
  }

  /**
   * Get fund data (placeholder - replace with actual fund data fetching)
   */
  async getFundData(fundId) {
    // This is a placeholder - replace with actual fund data fetching logic
    return {
      schemeCode: fundId,
      schemeName: 'Sample Fund',
      navHistory: [
        { date: '01-01-2023', nav: 100 },
        { date: '02-01-2023', nav: 102 },
        { date: '03-01-2023', nav: 105 },
        { date: '04-01-2023', nav: 108 },
        { date: '05-01-2023', nav: 110 },
        { date: '06-01-2023', nav: 112 },
        { date: '07-01-2023', nav: 115 },
        { date: '08-01-2023', nav: 118 },
        { date: '09-01-2023', nav: 120 },
        { date: '10-01-2023', nav: 122 },
        { date: '11-01-2023', nav: 125 },
        { date: '12-01-2023', nav: 128 }
      ]
    };
  }
}

module.exports = new BenchmarkController(); 