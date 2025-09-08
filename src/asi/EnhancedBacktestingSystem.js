/**
 * 📈 ENHANCED BACKTESTING SYSTEM
 * 
 * Main orchestrator for advanced backtesting with all components integrated
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Complete Backtesting Suite
 */

const EventEmitter = require('events');
const logger = require('../utils/logger');

// Import backtesting components
const { BacktestingCore } = require('./backtesting/BacktestingCore');
const { PerformanceAnalyzer } = require('./backtesting/PerformanceAnalyzer');
const { StrategyTemplates } = require('./backtesting/StrategyTemplates');
const { WalkForwardOptimizer } = require('./backtesting/WalkForwardOptimizer');
const { MonteCarloSimulator } = require('./backtesting/MonteCarloSimulator');

class EnhancedBacktestingSystem extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      initialCapital: options.initialCapital || 100000,
      transactionCosts: options.transactionCosts || 0.001,
      slippage: options.slippage || 0.0005,
      riskFreeRate: options.riskFreeRate || 0.02,
      benchmarkReturn: options.benchmarkReturn || 0.08,
      maxDrawdownLimit: options.maxDrawdownLimit || 0.2,
      enableWalkForward: options.enableWalkForward !== false,
      enableMonteCarlo: options.enableMonteCarlo !== false,
      defaultMonteCarloRuns: options.defaultMonteCarloRuns || 1000,
      ...options
    };
    
    // Core components
    this.backtestingCore = null;
    this.performanceAnalyzer = null;
    this.strategyTemplates = null;
    this.walkForwardOptimizer = null;
    this.monteCarloSimulator = null;
    
    // Results storage
    this.backtestResults = new Map();
    this.optimizationResults = new Map();
    this.simulationResults = new Map();
    
    // System metrics
    this.metrics = {
      backtestsRun: 0,
      strategiesTested: 0,
      optimizationsRun: 0,
      simulationsRun: 0,
      totalTradingDays: 0,
      averageAnnualReturn: 0,
      averageSharpeRatio: 0,
      successfulStrategies: 0
    };
    
    this.isInitialized = false;
  }

  async initialize() {
    try {
      logger.info('📈 Initializing Enhanced Backtesting System...');
      
      // Initialize core components
      this.backtestingCore = new BacktestingCore(this.config);
      await this.backtestingCore.initialize();
      
      this.performanceAnalyzer = new PerformanceAnalyzer(this.config);
      this.strategyTemplates = new StrategyTemplates();
      
      if (this.config.enableWalkForward) {
        this.walkForwardOptimizer = new WalkForwardOptimizer(this.config);
      }
      
      if (this.config.enableMonteCarlo) {
        this.monteCarloSimulator = new MonteCarloSimulator({
          defaultRuns: this.config.defaultMonteCarloRuns,
          ...this.config
        });
      }
      
      // Inject performance analyzer into backtesting core
      this.backtestingCore.performanceAnalyzer = this.performanceAnalyzer;
      
      this.isInitialized = true;
      logger.info('✅ Enhanced Backtesting System initialized successfully');
      
    } catch (error) {
      logger.error('❌ Enhanced Backtesting System initialization failed:', error);
      throw error;
    }
  }

  async runBacktest(strategyName, startDate, endDate, symbols, options = {}) {
    try {
      if (!this.isInitialized) {
        throw new Error('Backtesting system not initialized');
      }
      
      // Get strategy template
      const strategy = this.getStrategy(strategyName, options.parameters);
      
      logger.info(`🚀 Running backtest for strategy: ${strategy.name}`);
      
      // Run core backtest
      const backtestResults = await this.backtestingCore.runBacktest(
        strategy,
        startDate,
        endDate,
        symbols,
        options
      );
      
      // Calculate detailed performance metrics
      const performance = this.performanceAnalyzer.calculateMetrics(
        backtestResults.portfolioHistory,
        options.benchmarkReturns
      );
      
      // Generate performance report
      const performanceReport = this.performanceAnalyzer.generatePerformanceReport(
        backtestResults.portfolioHistory,
        options.benchmarkReturns,
        options
      );
      
      // Combine results
      const enhancedResults = {
        ...backtestResults,
        performance,
        performanceReport,
        enhancedMetrics: {
          riskAdjustedReturn: performance.annualizedReturn / Math.max(performance.volatility, 0.01),
          profitFactor: performance.profitFactor,
          recoveryFactor: performance.annualizedReturn / Math.max(performance.maxDrawdown, 0.01),
          expectancy: this.calculateExpectancy(backtestResults.trades),
          tradingFrequency: backtestResults.trades.length / performance.tradingDays * 252
        }
      };
      
      // Store results
      this.backtestResults.set(backtestResults.backtestId, enhancedResults);
      
      // Update system metrics
      this.updateSystemMetrics(enhancedResults);
      
      logger.info(`✅ Backtest completed: ${backtestResults.backtestId} - Return: ${(performance.annualizedReturn * 100).toFixed(2)}%, Sharpe: ${performance.sharpeRatio.toFixed(2)}`);
      
      this.emit('backtestCompleted', enhancedResults);
      
      return enhancedResults;
      
    } catch (error) {
      logger.error('❌ Backtest failed:', error);
      throw error;
    }
  }

  async runWalkForwardOptimization(strategyName, startDate, endDate, symbols, parameterRanges, options = {}) {
    try {
      if (!this.walkForwardOptimizer) {
        throw new Error('Walk-forward optimization not enabled');
      }
      
      const strategy = this.getStrategy(strategyName);
      
      logger.info(`🔄 Running walk-forward optimization for strategy: ${strategy.name}`);
      
      const results = await this.walkForwardOptimizer.walkForwardOptimization(
        this.backtestingCore,
        strategy,
        startDate,
        endDate,
        symbols,
        parameterRanges
      );
      
      // Store optimization results
      this.optimizationResults.set(results.optimizationId, results);
      this.metrics.optimizationsRun++;
      
      logger.info(`✅ Walk-forward optimization completed: ${results.optimizationId}`);
      
      this.emit('optimizationCompleted', results);
      
      return results;
      
    } catch (error) {
      logger.error('❌ Walk-forward optimization failed:', error);
      throw error;
    }
  }

  async runMonteCarloSimulation(strategyName, startDate, endDate, symbols, options = {}) {
    try {
      if (!this.monteCarloSimulator) {
        throw new Error('Monte Carlo simulation not enabled');
      }
      
      const strategy = this.getStrategy(strategyName, options.parameters);
      
      logger.info(`🎲 Running Monte Carlo simulation for strategy: ${strategy.name}`);
      
      const results = await this.monteCarloSimulator.runMonteCarloSimulation(
        this.backtestingCore,
        strategy,
        startDate,
        endDate,
        symbols,
        options
      );
      
      // Store simulation results
      this.simulationResults.set(results.simulationId, results);
      this.metrics.simulationsRun++;
      
      logger.info(`✅ Monte Carlo simulation completed: ${results.simulationId}`);
      
      this.emit('simulationCompleted', results);
      
      return results;
      
    } catch (error) {
      logger.error('❌ Monte Carlo simulation failed:', error);
      throw error;
    }
  }

  async runComprehensiveAnalysis(strategyName, startDate, endDate, symbols, options = {}) {
    try {
      logger.info(`🔬 Running comprehensive analysis for strategy: ${strategyName}`);
      
      const results = {
        strategy: strategyName,
        period: { startDate, endDate },
        symbols,
        timestamp: new Date()
      };
      
      // 1. Basic backtest
      results.backtest = await this.runBacktest(strategyName, startDate, endDate, symbols, options);
      
      // 2. Walk-forward optimization (if enabled and parameter ranges provided)
      if (this.walkForwardOptimizer && options.parameterRanges) {
        results.walkForwardOptimization = await this.runWalkForwardOptimization(
          strategyName,
          startDate,
          endDate,
          symbols,
          options.parameterRanges,
          options
        );
      }
      
      // 3. Monte Carlo simulation (if enabled)
      if (this.monteCarloSimulator && options.runMonteCarlo !== false) {
        results.monteCarloSimulation = await this.runMonteCarloSimulation(
          strategyName,
          startDate,
          endDate,
          symbols,
          {
            ...options,
            runs: options.monteCarloRuns || this.config.defaultMonteCarloRuns
          }
        );
      }
      
      // 4. Generate comprehensive report
      results.comprehensiveReport = this.generateComprehensiveReport(results);
      
      logger.info(`✅ Comprehensive analysis completed for strategy: ${strategyName}`);
      
      this.emit('comprehensiveAnalysisCompleted', results);
      
      return results;
      
    } catch (error) {
      logger.error('❌ Comprehensive analysis failed:', error);
      throw error;
    }
  }

  async compareStrategies(strategyConfigs, startDate, endDate, symbols, options = {}) {
    try {
      logger.info(`🏆 Comparing ${strategyConfigs.length} strategies`);
      
      const results = [];
      
      // Run backtests for all strategies
      for (const config of strategyConfigs) {
        try {
          const result = await this.runBacktest(
            config.name,
            startDate,
            endDate,
            symbols,
            { ...options, parameters: config.parameters }
          );
          
          results.push({
            strategyName: config.name,
            parameters: config.parameters,
            ...result
          });
          
        } catch (error) {
          logger.warn(`⚠️ Strategy ${config.name} failed:`, error.message);
        }
      }
      
      // Rank strategies
      const ranking = this.rankStrategies(results, options.rankingMetric || 'sharpeRatio');
      
      // Generate comparison report
      const comparisonReport = this.generateComparisonReport(results, ranking);
      
      const comparison = {
        strategies: strategyConfigs.map(c => c.name),
        period: { startDate, endDate },
        symbols,
        results,
        ranking,
        comparisonReport,
        timestamp: new Date()
      };
      
      logger.info(`✅ Strategy comparison completed. Winner: ${ranking[0]?.strategyName || 'None'}`);
      
      this.emit('strategyComparisonCompleted', comparison);
      
      return comparison;
      
    } catch (error) {
      logger.error('❌ Strategy comparison failed:', error);
      throw error;
    }
  }

  getStrategy(strategyName, parameters = {}) {
    const template = this.strategyTemplates.getTemplate(strategyName);
    
    if (!template) {
      throw new Error(`Strategy template not found: ${strategyName}`);
    }
    
    return {
      ...template,
      parameters: { ...template.parameters, ...parameters }
    };
  }

  addCustomStrategy(name, strategy) {
    this.strategyTemplates.addCustomTemplate(name, strategy);
    logger.info(`📋 Added custom strategy: ${name}`);
  }

  getAvailableStrategies() {
    return this.strategyTemplates.getAllTemplates();
  }

  calculateExpectancy(trades) {
    if (trades.length === 0) return 0;
    
    const profits = trades.filter(t => t.executedValue > 0);
    const losses = trades.filter(t => t.executedValue < 0);
    
    const avgProfit = profits.length > 0 ? 
      profits.reduce((sum, t) => sum + t.executedValue, 0) / profits.length : 0;
    const avgLoss = losses.length > 0 ? 
      Math.abs(losses.reduce((sum, t) => sum + t.executedValue, 0) / losses.length) : 0;
    
    const winRate = profits.length / trades.length;
    const lossRate = losses.length / trades.length;
    
    return (winRate * avgProfit) - (lossRate * avgLoss);
  }

  rankStrategies(results, metric = 'sharpeRatio') {
    return results
      .filter(r => r.performance && r.performance[metric] !== undefined)
      .sort((a, b) => (b.performance[metric] || -Infinity) - (a.performance[metric] || -Infinity))
      .map((result, index) => ({
        rank: index + 1,
        strategyName: result.strategyName,
        parameters: result.parameters,
        score: result.performance[metric],
        performance: result.performance
      }));
  }

  generateComparisonReport(results, ranking) {
    const validResults = results.filter(r => r.performance);
    
    if (validResults.length === 0) {
      return { error: 'No valid results to compare' };
    }
    
    return {
      summary: {
        strategiesCompared: validResults.length,
        winner: ranking[0]?.strategyName || 'None',
        winnerScore: ranking[0]?.score || 0,
        averageReturn: validResults.reduce((sum, r) => sum + (r.performance.annualizedReturn || 0), 0) / validResults.length,
        averageSharpe: validResults.reduce((sum, r) => sum + (r.performance.sharpeRatio || 0), 0) / validResults.length,
        bestReturn: Math.max(...validResults.map(r => r.performance.annualizedReturn || -Infinity)),
        worstReturn: Math.min(...validResults.map(r => r.performance.annualizedReturn || Infinity)),
        bestSharpe: Math.max(...validResults.map(r => r.performance.sharpeRatio || -Infinity)),
        worstSharpe: Math.min(...validResults.map(r => r.performance.sharpeRatio || Infinity))
      },
      ranking: ranking.slice(0, 10), // Top 10
      detailedComparison: validResults.map(r => ({
        strategy: r.strategyName,
        return: r.performance.annualizedReturn,
        volatility: r.performance.volatility,
        sharpe: r.performance.sharpeRatio,
        maxDrawdown: r.performance.maxDrawdown,
        winRate: r.performance.winRate,
        trades: r.metadata.totalTrades
      }))
    };
  }

  generateComprehensiveReport(results) {
    const report = {
      strategy: results.strategy,
      period: results.period,
      symbols: results.symbols,
      timestamp: results.timestamp
    };
    
    // Basic backtest summary
    if (results.backtest) {
      report.backtestSummary = {
        totalReturn: results.backtest.performance.totalReturn,
        annualizedReturn: results.backtest.performance.annualizedReturn,
        volatility: results.backtest.performance.volatility,
        sharpeRatio: results.backtest.performance.sharpeRatio,
        maxDrawdown: results.backtest.performance.maxDrawdown,
        winRate: results.backtest.performance.winRate,
        totalTrades: results.backtest.metadata.totalTrades
      };
    }
    
    // Walk-forward optimization summary
    if (results.walkForwardOptimization) {
      report.optimizationSummary = {
        averageReturn: results.walkForwardOptimization.aggregate.averageReturn,
        averageSharpe: results.walkForwardOptimization.aggregate.averageSharpe,
        consistency: results.walkForwardOptimization.aggregate.consistency,
        parameterStability: results.walkForwardOptimization.aggregate.parameterStability
      };
    }
    
    // Monte Carlo simulation summary
    if (results.monteCarloSimulation) {
      report.simulationSummary = {
        runs: results.monteCarloSimulation.runs,
        expectedReturn: results.monteCarloSimulation.summary.expectedReturn,
        probabilityOfProfit: results.monteCarloSimulation.summary.probabilityOfProfit,
        worstCaseReturn: results.monteCarloSimulation.summary.worstCaseReturn,
        bestCaseReturn: results.monteCarloSimulation.summary.bestCaseReturn,
        valueAtRisk95: results.monteCarloSimulation.summary.valueAtRisk95
      };
    }
    
    // Overall assessment
    report.overallAssessment = this.generateOverallAssessment(results);
    
    return report;
  }

  generateOverallAssessment(results) {
    const assessment = {
      score: 0,
      grade: 'F',
      strengths: [],
      weaknesses: [],
      recommendations: []
    };
    
    if (results.backtest && results.backtest.performance) {
      const perf = results.backtest.performance;
      
      // Calculate overall score (0-100)
      let score = 50; // Base score
      
      // Return component (30%)
      if (perf.annualizedReturn > 0.15) score += 15;
      else if (perf.annualizedReturn > 0.08) score += 10;
      else if (perf.annualizedReturn > 0.05) score += 5;
      else if (perf.annualizedReturn < 0) score -= 10;
      
      // Risk component (25%)
      if (perf.sharpeRatio > 2.0) score += 12;
      else if (perf.sharpeRatio > 1.5) score += 10;
      else if (perf.sharpeRatio > 1.0) score += 7;
      else if (perf.sharpeRatio > 0.5) score += 3;
      else if (perf.sharpeRatio < 0) score -= 10;
      
      // Drawdown component (20%)
      if (perf.maxDrawdown < 0.05) score += 10;
      else if (perf.maxDrawdown < 0.10) score += 7;
      else if (perf.maxDrawdown < 0.15) score += 5;
      else if (perf.maxDrawdown < 0.20) score += 2;
      else score -= 5;
      
      // Consistency component (15%)
      if (perf.winRate > 0.6) score += 7;
      else if (perf.winRate > 0.5) score += 5;
      else if (perf.winRate > 0.4) score += 2;
      else score -= 3;
      
      // Volatility component (10%)
      if (perf.volatility < 0.15) score += 5;
      else if (perf.volatility < 0.25) score += 3;
      else if (perf.volatility > 0.4) score -= 3;
      
      assessment.score = Math.max(0, Math.min(100, score));
      
      // Assign grade
      if (assessment.score >= 90) assessment.grade = 'A+';
      else if (assessment.score >= 85) assessment.grade = 'A';
      else if (assessment.score >= 80) assessment.grade = 'A-';
      else if (assessment.score >= 75) assessment.grade = 'B+';
      else if (assessment.score >= 70) assessment.grade = 'B';
      else if (assessment.score >= 65) assessment.grade = 'B-';
      else if (assessment.score >= 60) assessment.grade = 'C+';
      else if (assessment.score >= 55) assessment.grade = 'C';
      else if (assessment.score >= 50) assessment.grade = 'C-';
      else if (assessment.score >= 40) assessment.grade = 'D';
      else assessment.grade = 'F';
      
      // Identify strengths and weaknesses
      if (perf.annualizedReturn > 0.12) assessment.strengths.push('Strong returns');
      if (perf.sharpeRatio > 1.5) assessment.strengths.push('Excellent risk-adjusted returns');
      if (perf.maxDrawdown < 0.10) assessment.strengths.push('Low drawdown');
      if (perf.winRate > 0.55) assessment.strengths.push('High win rate');
      
      if (perf.annualizedReturn < 0.05) assessment.weaknesses.push('Low returns');
      if (perf.sharpeRatio < 0.8) assessment.weaknesses.push('Poor risk-adjusted returns');
      if (perf.maxDrawdown > 0.20) assessment.weaknesses.push('High drawdown');
      if (perf.winRate < 0.45) assessment.weaknesses.push('Low win rate');
      if (perf.volatility > 0.30) assessment.weaknesses.push('High volatility');
      
      // Generate recommendations
      if (perf.sharpeRatio < 1.0) assessment.recommendations.push('Consider risk management improvements');
      if (perf.maxDrawdown > 0.15) assessment.recommendations.push('Implement drawdown controls');
      if (perf.winRate < 0.5) assessment.recommendations.push('Review entry/exit criteria');
      if (perf.volatility > 0.25) assessment.recommendations.push('Consider position sizing optimization');
    }
    
    return assessment;
  }

  updateSystemMetrics(results) {
    this.metrics.backtestsRun++;
    this.metrics.strategiesTested++;
    
    if (results.performance) {
      this.metrics.totalTradingDays += results.performance.tradingDays || 0;
      this.metrics.averageAnnualReturn = 
        (this.metrics.averageAnnualReturn + (results.performance.annualizedReturn || 0)) / 2;
      this.metrics.averageSharpeRatio = 
        (this.metrics.averageSharpeRatio + (results.performance.sharpeRatio || 0)) / 2;
      
      if ((results.performance.sharpeRatio || 0) > 1.0) {
        this.metrics.successfulStrategies++;
      }
    }
  }

  // Getter methods
  getBacktestResults(backtestId) {
    return this.backtestResults.get(backtestId);
  }

  getOptimizationResults(optimizationId) {
    return this.optimizationResults.get(optimizationId);
  }

  getSimulationResults(simulationId) {
    return this.simulationResults.get(simulationId);
  }

  getAllBacktestResults() {
    return Array.from(this.backtestResults.values());
  }

  getSystemMetrics() {
    return {
      ...this.metrics,
      backtestsStored: this.backtestResults.size,
      optimizationsStored: this.optimizationResults.size,
      simulationsStored: this.simulationResults.size,
      isInitialized: this.isInitialized,
      componentsEnabled: {
        walkForward: !!this.walkForwardOptimizer,
        monteCarlo: !!this.monteCarloSimulator
      }
    };
  }

  async shutdown() {
    try {
      logger.info('🛑 Shutting down Enhanced Backtesting System...');
      
      if (this.backtestingCore) {
        await this.backtestingCore.shutdown();
      }
      
      this.isInitialized = false;
      logger.info('✅ Enhanced Backtesting System shutdown completed');
      
    } catch (error) {
      logger.error('❌ Enhanced Backtesting System shutdown failed:', error);
    }
  }
}

module.exports = { EnhancedBacktestingSystem };
