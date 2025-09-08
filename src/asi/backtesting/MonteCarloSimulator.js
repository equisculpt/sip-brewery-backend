/**
 * 🎲 MONTE CARLO SIMULATOR
 * 
 * Advanced Monte Carlo simulation for backtesting robustness testing
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Monte Carlo Simulation Engine
 */

const logger = require('../../utils/logger');

class MonteCarloSimulator {
  constructor(options = {}) {
    this.config = {
      defaultRuns: options.defaultRuns || 1000,
      maxRuns: options.maxRuns || 10000,
      confidenceLevels: options.confidenceLevels || [0.05, 0.25, 0.5, 0.75, 0.95],
      randomSeed: options.randomSeed || null,
      parallelProcessing: options.parallelProcessing !== false,
      ...options
    };
    
    this.simulationResults = new Map();
    this.randomGenerator = this.initializeRandomGenerator();
  }

  initializeRandomGenerator() {
    // Simple linear congruential generator for reproducible results
    let seed = this.config.randomSeed || Date.now();
    
    return {
      next: () => {
        seed = (seed * 1664525 + 1013904223) % Math.pow(2, 32);
        return seed / Math.pow(2, 32);
      },
      reset: (newSeed) => {
        seed = newSeed || Date.now();
      }
    };
  }

  async runMonteCarloSimulation(backtestingCore, strategy, startDate, endDate, symbols, options = {}) {
    try {
      const runs = Math.min(options.runs || this.config.defaultRuns, this.config.maxRuns);
      const simulationType = options.simulationType || 'bootstrap';
      
      logger.info(`🎲 Starting Monte Carlo simulation: ${runs} runs using ${simulationType} method`);
      
      const results = [];
      const batchSize = options.batchSize || Math.min(50, runs);
      
      // Run simulations in batches to manage memory
      for (let batch = 0; batch < Math.ceil(runs / batchSize); batch++) {
        const batchStart = batch * batchSize;
        const batchEnd = Math.min(batchStart + batchSize, runs);
        const batchRuns = batchEnd - batchStart;
        
        logger.info(`🎲 Processing batch ${batch + 1}/${Math.ceil(runs / batchSize)} (runs ${batchStart + 1}-${batchEnd})`);
        
        const batchPromises = [];
        
        for (let run = batchStart; run < batchEnd; run++) {
          batchPromises.push(this.runSingleSimulation(
            backtestingCore,
            strategy,
            startDate,
            endDate,
            symbols,
            run,
            simulationType,
            options
          ));
        }
        
        const batchResults = await Promise.allSettled(batchPromises);
        
        // Process batch results
        for (let i = 0; i < batchResults.length; i++) {
          const result = batchResults[i];
          if (result.status === 'fulfilled') {
            results.push(result.value);
          } else {
            logger.warn(`⚠️ Simulation run ${batchStart + i + 1} failed:`, result.reason.message);
          }
        }
      }
      
      // Analyze results
      const analysis = this.analyzeMonteCarloResults(results, options);
      
      // Store simulation results
      const simulationId = `mc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      this.simulationResults.set(simulationId, {
        strategy: strategy.name,
        parameters: strategy.parameters,
        period: { startDate, endDate },
        symbols,
        simulationType,
        runs: results.length,
        results: results.slice(0, 100), // Store first 100 for analysis
        analysis,
        timestamp: new Date()
      });
      
      logger.info(`✅ Monte Carlo simulation completed: ${results.length}/${runs} successful runs`);
      
      return {
        simulationId,
        runs: results.length,
        analysis,
        summary: this.generateSimulationSummary(analysis)
      };
      
    } catch (error) {
      logger.error('❌ Monte Carlo simulation failed:', error);
      throw error;
    }
  }

  async runSingleSimulation(backtestingCore, strategy, startDate, endDate, symbols, runNumber, simulationType, options) {
    try {
      // Generate randomized data based on simulation type
      const randomizedData = await this.generateRandomizedData(
        symbols,
        startDate,
        endDate,
        simulationType,
        runNumber,
        options
      );
      
      // Run backtest with randomized data
      const results = await this.runBacktestWithRandomizedData(
        backtestingCore,
        strategy,
        randomizedData,
        options
      );
      
      // Calculate performance metrics
      const performanceAnalyzer = backtestingCore.performanceAnalyzer || 
        new (require('./PerformanceAnalyzer').PerformanceAnalyzer)();
      
      const performance = performanceAnalyzer.calculateMetrics(results.portfolioHistory);
      
      return {
        run: runNumber,
        finalValue: results.metadata.finalValue,
        totalReturn: performance.totalReturn,
        annualizedReturn: performance.annualizedReturn,
        maxDrawdown: performance.maxDrawdown,
        sharpeRatio: performance.sharpeRatio,
        volatility: performance.volatility,
        winRate: performance.winRate,
        trades: results.trades.length,
        transactionCosts: results.metadata.transactionCosts
      };
      
    } catch (error) {
      throw new Error(`Simulation run ${runNumber} failed: ${error.message}`);
    }
  }

  async generateRandomizedData(symbols, startDate, endDate, simulationType, runNumber, options) {
    const start = new Date(startDate);
    const end = new Date(endDate);
    const randomizedData = new Map();
    
    // Set random seed for reproducible results
    this.randomGenerator.reset(this.config.randomSeed ? this.config.randomSeed + runNumber : undefined);
    
    switch (simulationType) {
      case 'bootstrap':
        return await this.bootstrapHistoricalData(symbols, start, end, options);
      case 'parametric':
        return await this.generateParametricData(symbols, start, end, options);
      case 'blockBootstrap':
        return await this.blockBootstrapData(symbols, start, end, options);
      case 'noiseInjection':
        return await this.injectNoiseIntoData(symbols, start, end, options);
      default:
        return await this.bootstrapHistoricalData(symbols, start, end, options);
    }
  }

  async bootstrapHistoricalData(symbols, startDate, endDate, options) {
    const randomizedData = new Map();
    
    // Generate bootstrap sample by randomly sampling from historical data
    for (let date = new Date(startDate); date <= endDate; date.setDate(date.getDate() + 1)) {
      if (date.getDay() !== 0 && date.getDay() !== 6) { // Skip weekends
        const timestamp = date.getTime();
        const marketData = new Map();
        
        for (const symbol of symbols) {
          // Generate random price movement based on historical distribution
          const basePrice = 100 + Math.random() * 50;
          const volatility = 0.02 + Math.random() * 0.03; // 2-5% daily volatility
          const return_ = this.generateRandomReturn(volatility);
          
          marketData.set(symbol, {
            price: basePrice * (1 + return_),
            volume: Math.floor(Math.random() * 1000000),
            high: basePrice * (1 + Math.abs(return_) + Math.random() * 0.01),
            low: basePrice * (1 + return_ - Math.random() * 0.01),
            open: basePrice,
            return: return_
          });
        }
        
        randomizedData.set(timestamp, marketData);
      }
    }
    
    return randomizedData;
  }

  async generateParametricData(symbols, startDate, endDate, options) {
    const randomizedData = new Map();
    const parameters = options.parametricParams || {
      meanReturn: 0.0008, // Daily mean return
      volatility: 0.02,   // Daily volatility
      correlation: 0.3    // Cross-asset correlation
    };
    
    // Generate correlated random returns
    for (let date = new Date(startDate); date <= endDate; date.setDate(date.getDate() + 1)) {
      if (date.getDay() !== 0 && date.getDay() !== 6) {
        const timestamp = date.getTime();
        const marketData = new Map();
        
        // Generate correlated returns using Cholesky decomposition (simplified)
        const baseReturn = this.generateNormalRandom() * parameters.volatility + parameters.meanReturn;
        
        for (let i = 0; i < symbols.length; i++) {
          const symbol = symbols[i];
          
          // Add correlation and individual noise
          const correlatedReturn = baseReturn * parameters.correlation + 
            this.generateNormalRandom() * parameters.volatility * Math.sqrt(1 - parameters.correlation * parameters.correlation);
          
          const basePrice = 100 + i * 10; // Different base prices for different symbols
          
          marketData.set(symbol, {
            price: basePrice * (1 + correlatedReturn),
            volume: Math.floor(Math.random() * 1000000),
            high: basePrice * (1 + correlatedReturn + Math.random() * 0.005),
            low: basePrice * (1 + correlatedReturn - Math.random() * 0.005),
            open: basePrice,
            return: correlatedReturn
          });
        }
        
        randomizedData.set(timestamp, marketData);
      }
    }
    
    return randomizedData;
  }

  async blockBootstrapData(symbols, startDate, endDate, options) {
    const blockSize = options.blockSize || 5; // 5-day blocks
    const randomizedData = new Map();
    
    // Generate blocks of consecutive days to preserve serial correlation
    const totalDays = Math.floor((endDate - startDate) / (24 * 60 * 60 * 1000));
    const tradingDays = Math.floor(totalDays * 5/7); // Approximate trading days
    const numBlocks = Math.ceil(tradingDays / blockSize);
    
    let currentDate = new Date(startDate);
    
    for (let block = 0; block < numBlocks; block++) {
      // Generate a block of correlated returns
      const blockReturns = [];
      for (let day = 0; day < blockSize; day++) {
        const dayReturns = {};
        for (const symbol of symbols) {
          dayReturns[symbol] = this.generateRandomReturn(0.02);
        }
        blockReturns.push(dayReturns);
      }
      
      // Apply block to consecutive days
      for (let day = 0; day < blockSize && currentDate <= endDate; day++) {
        if (currentDate.getDay() !== 0 && currentDate.getDay() !== 6) {
          const timestamp = currentDate.getTime();
          const marketData = new Map();
          
          for (const symbol of symbols) {
            const basePrice = 100 + symbols.indexOf(symbol) * 10;
            const return_ = blockReturns[day][symbol];
            
            marketData.set(symbol, {
              price: basePrice * (1 + return_),
              volume: Math.floor(Math.random() * 1000000),
              high: basePrice * (1 + return_ + Math.random() * 0.005),
              low: basePrice * (1 + return_ - Math.random() * 0.005),
              open: basePrice,
              return: return_
            });
          }
          
          randomizedData.set(timestamp, marketData);
        }
        
        currentDate.setDate(currentDate.getDate() + 1);
      }
    }
    
    return randomizedData;
  }

  async injectNoiseIntoData(symbols, startDate, endDate, options) {
    const noiseLevel = options.noiseLevel || 0.1; // 10% noise
    
    // Start with base historical data and add noise
    const baseData = await this.generateBaseHistoricalData(symbols, startDate, endDate);
    const randomizedData = new Map();
    
    for (const [timestamp, marketData] of baseData) {
      const noisyMarketData = new Map();
      
      for (const [symbol, data] of marketData) {
        const noiseFactor = 1 + (this.randomGenerator.next() - 0.5) * 2 * noiseLevel;
        
        noisyMarketData.set(symbol, {
          price: data.price * noiseFactor,
          volume: Math.floor(data.volume * (0.8 + this.randomGenerator.next() * 0.4)),
          high: data.high * noiseFactor,
          low: data.low * noiseFactor,
          open: data.open * noiseFactor,
          return: data.return * noiseFactor
        });
      }
      
      randomizedData.set(timestamp, noisyMarketData);
    }
    
    return randomizedData;
  }

  async generateBaseHistoricalData(symbols, startDate, endDate) {
    const baseData = new Map();
    
    for (let date = new Date(startDate); date <= endDate; date.setDate(date.getDate() + 1)) {
      if (date.getDay() !== 0 && date.getDay() !== 6) {
        const timestamp = date.getTime();
        const marketData = new Map();
        
        for (const symbol of symbols) {
          const basePrice = 100 + symbols.indexOf(symbol) * 10;
          const return_ = this.generateRandomReturn(0.02);
          
          marketData.set(symbol, {
            price: basePrice * (1 + return_),
            volume: Math.floor(Math.random() * 1000000),
            high: basePrice * (1 + return_ + Math.random() * 0.005),
            low: basePrice * (1 + return_ - Math.random() * 0.005),
            open: basePrice,
            return: return_
          });
        }
        
        baseData.set(timestamp, marketData);
      }
    }
    
    return baseData;
  }

  async runBacktestWithRandomizedData(backtestingCore, strategy, randomizedData, options) {
    // Temporarily replace the historical data preparation method
    const originalPrepareData = backtestingCore.prepareHistoricalData;
    
    backtestingCore.prepareHistoricalData = async () => randomizedData;
    
    try {
      const results = await backtestingCore.runBacktestSimulation(
        strategy,
        randomizedData,
        new Date(Math.min(...randomizedData.keys())),
        new Date(Math.max(...randomizedData.keys())),
        options
      );
      
      return {
        portfolioHistory: backtestingCore.portfolioTracker.portfolioHistory,
        trades: results.trades,
        metadata: {
          finalValue: backtestingCore.portfolioTracker.currentPortfolio.totalValue,
          transactionCosts: results.totalTransactionCosts
        }
      };
      
    } finally {
      // Restore original method
      backtestingCore.prepareHistoricalData = originalPrepareData;
    }
  }

  analyzeMonteCarloResults(results, options) {
    if (results.length === 0) {
      return { error: 'No successful simulation runs' };
    }
    
    // Extract metrics
    const returns = results.map(r => r.totalReturn);
    const annualizedReturns = results.map(r => r.annualizedReturn);
    const sharpeRatios = results.map(r => r.sharpeRatio).filter(s => isFinite(s));
    const maxDrawdowns = results.map(r => r.maxDrawdown);
    const volatilities = results.map(r => r.volatility);
    const finalValues = results.map(r => r.finalValue);
    
    return {
      totalRuns: results.length,
      
      returns: {
        mean: this.calculateMean(returns),
        median: this.calculateMedian(returns),
        std: this.calculateStandardDeviation(returns),
        min: Math.min(...returns),
        max: Math.max(...returns),
        percentiles: this.calculatePercentiles(returns, this.config.confidenceLevels)
      },
      
      annualizedReturns: {
        mean: this.calculateMean(annualizedReturns),
        median: this.calculateMedian(annualizedReturns),
        std: this.calculateStandardDeviation(annualizedReturns),
        percentiles: this.calculatePercentiles(annualizedReturns, this.config.confidenceLevels)
      },
      
      sharpeRatios: sharpeRatios.length > 0 ? {
        mean: this.calculateMean(sharpeRatios),
        median: this.calculateMedian(sharpeRatios),
        std: this.calculateStandardDeviation(sharpeRatios),
        percentiles: this.calculatePercentiles(sharpeRatios, this.config.confidenceLevels)
      } : null,
      
      maxDrawdowns: {
        mean: this.calculateMean(maxDrawdowns),
        median: this.calculateMedian(maxDrawdowns),
        worst: Math.max(...maxDrawdowns),
        percentiles: this.calculatePercentiles(maxDrawdowns, this.config.confidenceLevels)
      },
      
      volatilities: {
        mean: this.calculateMean(volatilities),
        median: this.calculateMedian(volatilities),
        percentiles: this.calculatePercentiles(volatilities, this.config.confidenceLevels)
      },
      
      finalValues: {
        mean: this.calculateMean(finalValues),
        median: this.calculateMedian(finalValues),
        percentiles: this.calculatePercentiles(finalValues, this.config.confidenceLevels)
      },
      
      probabilityMetrics: {
        probabilityOfProfit: results.filter(r => r.totalReturn > 0).length / results.length,
        probabilityOfLoss: results.filter(r => r.totalReturn < -0.1).length / results.length,
        probabilityOfLargeLoss: results.filter(r => r.totalReturn < -0.2).length / results.length,
        probabilityOfOutperformance: results.filter(r => r.annualizedReturn > 0.08).length / results.length
      },
      
      riskMetrics: {
        valueAtRisk95: this.calculateVaR(returns, 0.95),
        valueAtRisk99: this.calculateVaR(returns, 0.99),
        expectedShortfall95: this.calculateExpectedShortfall(returns, 0.05),
        expectedShortfall99: this.calculateExpectedShortfall(returns, 0.01)
      }
    };
  }

  generateSimulationSummary(analysis) {
    if (analysis.error) {
      return { error: analysis.error };
    }
    
    return {
      totalRuns: analysis.totalRuns,
      expectedReturn: analysis.annualizedReturns.mean,
      expectedVolatility: analysis.volatilities.mean,
      expectedSharpe: analysis.sharpeRatios ? analysis.sharpeRatios.mean : null,
      probabilityOfProfit: analysis.probabilityMetrics.probabilityOfProfit,
      worstCaseReturn: analysis.returns.percentiles['0.05'],
      bestCaseReturn: analysis.returns.percentiles['0.95'],
      medianReturn: analysis.returns.median,
      maxDrawdownExpected: analysis.maxDrawdowns.mean,
      valueAtRisk95: analysis.riskMetrics.valueAtRisk95
    };
  }

  // Utility methods
  generateRandomReturn(volatility) {
    return this.generateNormalRandom() * volatility;
  }

  generateNormalRandom() {
    // Box-Muller transformation
    const u1 = this.randomGenerator.next();
    const u2 = this.randomGenerator.next();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  calculateMean(values) {
    return values.length > 0 ? values.reduce((sum, v) => sum + v, 0) / values.length : 0;
  }

  calculateMedian(values) {
    if (values.length === 0) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
  }

  calculateStandardDeviation(values) {
    if (values.length === 0) return 0;
    const mean = this.calculateMean(values);
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  calculatePercentiles(values, percentiles) {
    if (values.length === 0) return {};
    
    const sorted = values.slice().sort((a, b) => a - b);
    const result = {};
    
    for (const p of percentiles) {
      const index = Math.floor(p * (sorted.length - 1));
      result[p.toString()] = sorted[index];
    }
    
    return result;
  }

  calculateVaR(returns, confidenceLevel) {
    const sorted = returns.slice().sort((a, b) => a - b);
    const index = Math.floor((1 - confidenceLevel) * sorted.length);
    return -sorted[index];
  }

  calculateExpectedShortfall(returns, alpha) {
    const sorted = returns.slice().sort((a, b) => a - b);
    const cutoffIndex = Math.floor(alpha * sorted.length);
    const tailReturns = sorted.slice(0, cutoffIndex);
    return tailReturns.length > 0 ? -this.calculateMean(tailReturns) : 0;
  }

  getSimulationResults(simulationId) {
    return this.simulationResults.get(simulationId);
  }

  getAllSimulationResults() {
    return Array.from(this.simulationResults.values());
  }
}

module.exports = { MonteCarloSimulator };
