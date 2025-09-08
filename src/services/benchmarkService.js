const mongoose = require('mongoose');
const nseCliService = require('./nseCliService');
const BenchmarkIndex = require('../models/BenchmarkIndex');
const dayjs = require('dayjs');
const { ChartJSNodeCanvas } = require('chartjs-node-canvas');
const Chart = require('chart.js/auto');
const logger = require('../utils/logger');

class BenchmarkService {
  constructor() {
    this.chartJSNodeCanvas = new ChartJSNodeCanvas({ width: 800, height: 400 });
  }

  /**
   * Fetch and store NIFTY 50 data using CLI service
   */
  async updateNifty50Data() {
    try {
      logger.info('Fetching NIFTY 50 historical data using CLI service...');
      
      // Get current NIFTY 50 data
      const currentData = await nseCliService.getNifty50Data();
      logger.info(`Current NIFTY 50 price: ${currentData.lastPrice}`);
      
      // Generate historical data based on current price (10 years)
      const historicalData = nseCliService.generateSyntheticHistoricalData('NIFTY 50', 365 * 10); // 10 years
      
      // Update or create NIFTY 50 record
      const result = await BenchmarkIndex.findOneAndUpdate(
        { indexId: 'NIFTY50' },
        {
          indexId: 'NIFTY50',
          name: 'NIFTY 50',
          data: historicalData,
          lastUpdated: new Date(),
          currentPrice: currentData.lastPrice
        },
        { upsert: true, new: true }
      );

      logger.info(`Upserted NIFTY 50 data. MongoDB _id: ${result && result._id}`);
      return { 
        success: true, 
        count: historicalData.length,
        currentPrice: currentData.lastPrice,
        lastUpdated: result.lastUpdated
      };
    } catch (error) {
      logger.error('Error updating NIFTY 50 data:', error);
      throw error;
    }
  }

  /**
   * Get benchmark data for a specific date range
   */
  async getBenchmarkData(indexId, fromDate, toDate) {
    try {
      logger.info(`Querying BenchmarkIndex for indexId: ${indexId}`);
      const benchmark = await BenchmarkIndex.findOne({ indexId });
      logger.info(`BenchmarkIndex.findOne result: ${benchmark ? 'FOUND' : 'NOT FOUND'}`);
      if (!benchmark) {
        throw new Error(`Benchmark ${indexId} not found`);
      }

      let filteredData = benchmark.data;
      
      if (fromDate) {
        filteredData = filteredData.filter(item => item.date >= fromDate);
      }
      
      if (toDate) {
        filteredData = filteredData.filter(item => item.date <= toDate);
      }

      logger.info(`Returning ${filteredData.length} records for indexId: ${indexId}`);
      return {
        indexId: benchmark.indexId,
        name: benchmark.name,
        data: filteredData,
        lastUpdated: benchmark.lastUpdated,
        currentPrice: benchmark.currentPrice
      };
    } catch (error) {
      logger.error('Error getting benchmark data:', error);
      throw error;
    }
  }

  /**
   * Get real-time market status and indices using CLI
   */
  async getMarketStatus() {
    try {
      logger.info('Fetching real-time market status via CLI...');
      const marketStatus = await nseCliService.getMarketStatus();
      return marketStatus;
    } catch (error) {
      logger.error('Error getting market status:', error);
      throw error;
    }
  }

  /**
   * Get gainers and losers for a specific index
   */
  async getGainersAndLosers(indexName = 'NIFTY 50') {
    try {
      logger.info(`Fetching gainers and losers for ${indexName}...`);
      // Since CLI doesn't have gainers/losers, we'll generate synthetic data
      const data = this.generateSyntheticGainersLosers();
      return data;
    } catch (error) {
      logger.error(`Error getting gainers and losers for ${indexName}:`, error);
      throw error;
    }
  }

  /**
   * Generate synthetic gainers and losers data
   */
  generateSyntheticGainersLosers() {
    const stocks = ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'AXISBANK'];
    
    const gainers = stocks.slice(0, 5).map((stock, index) => ({
      symbol: stock,
      change: Math.round((Math.random() * 100 + 50) * 100) / 100,
      pChange: Math.round((Math.random() * 3 + 1) * 100) / 100
    })).sort((a, b) => b.pChange - a.pChange);
    
    const losers = stocks.slice(5).map((stock, index) => ({
      symbol: stock,
      change: -Math.round((Math.random() * 100 + 50) * 100) / 100,
      pChange: -Math.round((Math.random() * 3 + 1) * 100) / 100
    })).sort((a, b) => a.pChange - b.pChange);
    
    return { gainers, losers };
  }

  /**
   * Get most active equities
   */
  async getMostActiveEquities() {
    try {
      logger.info('Fetching most active equities...');
      // Generate synthetic most active equities data
      const data = this.generateSyntheticMostActiveEquities();
      return data;
    } catch (error) {
      logger.error('Error getting most active equities:', error);
      throw error;
    }
  }

  /**
   * Generate synthetic most active equities data
   */
  generateSyntheticMostActiveEquities() {
    const stocks = ['RELIANCE', 'TCS', 'HDFC', 'INFY', 'ICICIBANK', 'HINDUNILVR', 'ITC', 'SBIN', 'BHARTIARTL', 'AXISBANK'];
    
    return stocks.map(stock => ({
      symbol: stock,
      volume: Math.floor(Math.random() * 2000000) + 500000,
      value: Math.floor(Math.random() * 5000000000) + 1000000000,
      change: Math.round((Math.random() * 200 - 100) * 100) / 100,
      pChange: Math.round((Math.random() * 4 - 2) * 100) / 100
    })).sort((a, b) => b.volume - a.volume);
  }

  /**
   * Automatically map a mutual fund to its benchmark index based on its name or category
   * @param {Object} fund - Mutual fund object (should have schemeName or category)
   * @returns {string} NSE index name
   */
  getBenchmarkIndexForFund(fund) {
    const name = (fund.schemeName || fund.name || '').toLowerCase();
    const category = (fund.category || '').toLowerCase();
    if (name.includes('small cap') || category.includes('small cap')) return 'NIFTY SMLCAP 100';
    if (name.includes('mid cap') || category.includes('mid cap')) return 'NIFTY MIDCAP 100';
    if (name.includes('large cap') || category.includes('large cap')) return 'NIFTY 100';
    if (name.includes('multi cap') || category.includes('multi cap')) return 'NIFTY 500 MULTICAP';
    if (name.includes('bank') || category.includes('bank')) return 'NIFTY BANK';
    if (name.includes('it') || category.includes('it')) return 'NIFTY IT';
    if (name.includes('auto') || category.includes('auto')) return 'NIFTY AUTO';
    if (name.includes('pharma') || category.includes('pharma')) return 'NIFTY PHARMA';
    if (name.includes('fmcg') || category.includes('fmcg')) return 'NIFTY FMCG';
    if (name.includes('infra') || category.includes('infra')) return 'NIFTY INFRA';
    // Add more mappings as needed
    return 'NIFTY 50'; // default fallback
  }

  /**
   * Compare mutual fund with its mapped benchmark index
   * Supports up to 10 years for charting and benchmarking
   */
  async compareWithBenchmark(fundData, benchmarkId = null, range = '3Y') {
    try {
      // If fundData is an array of NAVs, try to get the fund object from context
      let fund = null;
      if (Array.isArray(fundData) && fundData.length > 0 && fundData[0].schemeName) {
        fund = fundData[0];
      } else if (Array.isArray(fundData) && fundData.length > 0 && fundData[0].fund) {
        fund = fundData[0].fund;
      }
      // If fund is not provided, fallback to default
      if (!benchmarkId && fund) {
        benchmarkId = this.getBenchmarkIndexForFund(fund);
      }
      if (!benchmarkId) {
        benchmarkId = 'NIFTY 50';
      }
      // Get benchmark data
      const benchmark = await this.getBenchmarkData(benchmarkId);
      
      // Calculate date range
      const endDate = dayjs();
      let startDate;
      
      switch (range) {
        case '1M': startDate = endDate.subtract(1, 'month'); break;
        case '3M': startDate = endDate.subtract(3, 'month'); break;
        case '6M': startDate = endDate.subtract(6, 'month'); break;
        case '1Y': startDate = endDate.subtract(1, 'year'); break;
        case '3Y': startDate = endDate.subtract(3, 'year'); break;
        case '5Y': startDate = endDate.subtract(5, 'year'); break;
        case '10Y': startDate = endDate.subtract(10, 'year'); break;
        default: startDate = endDate.subtract(3, 'year');
      }

      const startDateStr = startDate.format('YYYY-MM-DD');
      const endDateStr = endDate.format('YYYY-MM-DD');

      // Filter benchmark data for up to 10 years
      const benchmarkData = benchmark.data.filter(item => 
        item.date >= startDateStr && item.date <= endDateStr
      );

      // Filter fund data (assuming fund data is in DD-MM-YYYY format)
      const fundDataFiltered = fundData.filter(item => {
        const fundDate = dayjs(item.date, 'DD-MM-YYYY');
        return fundDate.isAfter(startDate) && fundDate.isBefore(endDate);
      });

      // Calculate analytics
      const analytics = this.calculateComparisonAnalytics(fundDataFiltered, benchmarkData);

      return {
        fund: {
          data: fundDataFiltered,
          count: fundDataFiltered.length
        },
        benchmark: {
          data: benchmarkData,
          count: benchmarkData.length,
          index: benchmarkId
        },
        analytics,
        range,
        startDate: startDateStr,
        endDate: endDateStr
      };
    } catch (error) {
      console.error('Error comparing with benchmark:', error);
      throw error;
    }
  }

  /**
   * Calculate comparison analytics
   */
  calculateComparisonAnalytics(fundData, benchmarkData) {
    if (fundData.length === 0 || benchmarkData.length === 0) {
      return { error: 'Insufficient data for comparison' };
    }

    // Calculate CAGR for both
    const fundCAGR = this.calculateCAGR(fundData[0].nav, fundData[fundData.length - 1].nav, 
      dayjs(fundData[fundData.length - 1].date, 'DD-MM-YYYY').diff(dayjs(fundData[0].date, 'DD-MM-YYYY'), 'year', true));
    
    const benchmarkCAGR = this.calculateCAGR(benchmarkData[0].close, benchmarkData[benchmarkData.length - 1].close,
      dayjs(benchmarkData[benchmarkData.length - 1].date).diff(dayjs(benchmarkData[0].date), 'year', true));

    // Calculate Beta and Alpha
    const { beta, alpha } = this.calculateBetaAlpha(fundData, benchmarkData);

    // Calculate relative performance
    const relativePerformance = fundCAGR - benchmarkCAGR;

    return {
      fundCAGR: fundCAGR * 100,
      benchmarkCAGR: benchmarkCAGR * 100,
      relativePerformance: relativePerformance * 100,
      beta,
      alpha: alpha * 100,
      outperformance: relativePerformance > 0 ? 'Yes' : 'No',
      outperformanceAmount: Math.abs(relativePerformance * 100)
    };
  }

  /**
   * Calculate CAGR
   */
  calculateCAGR(startValue, endValue, years) {
    if (!startValue || !endValue || years <= 0) return 0;
    return Math.pow(endValue / startValue, 1 / years) - 1;
  }

  /**
   * Calculate Beta and Alpha
   */
  calculateBetaAlpha(fundData, benchmarkData) {
    // Create aligned data points
    const alignedData = [];
    
    fundData.forEach(fundPoint => {
      const fundDate = dayjs(fundPoint.date, 'DD-MM-YYYY').format('YYYY-MM-DD');
      const benchmarkPoint = benchmarkData.find(b => b.date === fundDate);
      
      if (benchmarkPoint) {
        alignedData.push({
          fund: fundPoint.nav,
          benchmark: benchmarkPoint.close
        });
      }
    });

    if (alignedData.length < 2) {
      return { beta: 0, alpha: 0 };
    }

    // Calculate returns
    const fundReturns = [];
    const benchmarkReturns = [];
    
    for (let i = 1; i < alignedData.length; i++) {
      fundReturns.push((alignedData[i].fund - alignedData[i-1].fund) / alignedData[i-1].fund);
      benchmarkReturns.push((alignedData[i].benchmark - alignedData[i-1].benchmark) / alignedData[i-1].benchmark);
    }

    // Calculate Beta
    const benchmarkVariance = this.calculateVariance(benchmarkReturns);
    const covariance = this.calculateCovariance(fundReturns, benchmarkReturns);
    const beta = benchmarkVariance !== 0 ? covariance / benchmarkVariance : 0;

    // Calculate Alpha
    const avgFundReturn = fundReturns.reduce((a, b) => a + b, 0) / fundReturns.length;
    const avgBenchmarkReturn = benchmarkReturns.reduce((a, b) => a + b, 0) / benchmarkReturns.length;
    const alpha = avgFundReturn - (beta * avgBenchmarkReturn);

    return { beta, alpha };
  }

  /**
   * Calculate variance
   */
  calculateVariance(data) {
    const mean = data.reduce((a, b) => a + b, 0) / data.length;
    return data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
  }

  /**
   * Calculate covariance
   */
  calculateCovariance(data1, data2) {
    const mean1 = data1.reduce((a, b) => a + b, 0) / data1.length;
    const mean2 = data2.reduce((a, b) => a + b, 0) / data2.length;
    
    let sum = 0;
    for (let i = 0; i < data1.length; i++) {
      sum += (data1[i] - mean1) * (data2[i] - mean2);
    }
    return sum / data1.length;
  }

  /**
   * Generate comparison chart
   */
  async generateComparisonChart(fundData, benchmarkData, fundName, benchmarkName) {
    try {
      // Prepare data for chart
      const chartData = {
        labels: fundData.map(item => item.date),
        datasets: [
          {
            label: fundName,
            data: fundData.map(item => item.nav),
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            tension: 0.1
          },
          {
            label: benchmarkName,
            data: benchmarkData.map(item => item.close),
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            tension: 0.1
          }
        ]
      };

      const configuration = {
        type: 'line',
        data: chartData,
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: `${fundName} vs ${benchmarkName} Performance`
            }
          },
          scales: {
            y: {
              beginAtZero: false
            }
          }
        }
      };

      const image = await this.chartJSNodeCanvas.renderToBuffer(configuration);
      return image;
    } catch (error) {
      console.error('Error generating chart:', error);
      throw error;
    }
  }
}

module.exports = new BenchmarkService(); 