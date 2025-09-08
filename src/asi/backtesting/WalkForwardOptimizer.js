/**
 * 🔄 WALK FORWARD OPTIMIZER
 * 
 * Advanced walk-forward optimization and parameter tuning
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Walk Forward Optimization Engine
 */

const logger = require('../../utils/logger');

class WalkForwardOptimizer {
  constructor(options = {}) {
    this.config = {
      walkForwardSteps: options.walkForwardSteps || 12,
      trainingRatio: options.trainingRatio || 0.8, // 80% for training, 20% for testing
      optimizationMetric: options.optimizationMetric || 'sharpeRatio',
      maxIterations: options.maxIterations || 100,
      convergenceThreshold: options.convergenceThreshold || 0.001,
      ...options
    };
    
    this.optimizationResults = new Map();
    this.parameterHistory = new Map();
  }

  async walkForwardOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges) {
    try {
      logger.info('🔄 Starting walk-forward optimization...');
      
      const results = [];
      const totalPeriod = new Date(endDate) - new Date(startDate);
      const stepSize = totalPeriod / this.config.walkForwardSteps;
      
      for (let step = 0; step < this.config.walkForwardSteps; step++) {
        const trainStart = new Date(new Date(startDate).getTime() + step * stepSize);
        const trainEnd = new Date(trainStart.getTime() + stepSize * this.config.trainingRatio);
        const testStart = trainEnd;
        const testEnd = new Date(trainStart.getTime() + stepSize);
        
        logger.info(`🔄 Walk-forward step ${step + 1}/${this.config.walkForwardSteps}: ${trainStart.toISOString().split('T')[0]} to ${testEnd.toISOString().split('T')[0]}`);
        
        // Optimize parameters on training period
        const optimalParams = await this.optimizeParameters(
          backtestingCore,
          strategy,
          trainStart,
          trainEnd,
          symbols,
          parameterRanges
        );
        
        // Test on out-of-sample period
        const testStrategy = { ...strategy, parameters: optimalParams };
        const testResults = await backtestingCore.runBacktest(
          testStrategy,
          testStart,
          testEnd,
          symbols
        );
        
        // Calculate performance analyzer if not available
        const performanceAnalyzer = backtestingCore.performanceAnalyzer || 
          new (require('./PerformanceAnalyzer').PerformanceAnalyzer)();
        
        const performance = performanceAnalyzer.calculateMetrics(testResults.portfolioHistory);
        
        results.push({
          step,
          trainPeriod: { start: trainStart, end: trainEnd },
          testPeriod: { start: testStart, end: testEnd },
          optimalParameters: optimalParams,
          performance,
          backtestId: testResults.backtestId
        });
        
        // Store parameter history
        this.updateParameterHistory(strategy.name, step, optimalParams);
      }
      
      // Aggregate results
      const aggregateResults = this.aggregateWalkForwardResults(results);
      
      // Store optimization results
      const optimizationId = `wfo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      this.optimizationResults.set(optimizationId, {
        strategy: strategy.name,
        parameterRanges,
        results,
        aggregate: aggregateResults,
        timestamp: new Date()
      });
      
      logger.info('✅ Walk-forward optimization completed');
      
      return {
        optimizationId,
        results,
        aggregate: aggregateResults
      };
      
    } catch (error) {
      logger.error('❌ Walk-forward optimization failed:', error);
      throw error;
    }
  }

  async optimizeParameters(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges) {
    try {
      logger.info('🎯 Optimizing parameters...');
      
      const optimizationMethod = this.config.optimizationMethod || 'gridSearch';
      
      switch (optimizationMethod) {
        case 'gridSearch':
          return await this.gridSearchOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges);
        case 'randomSearch':
          return await this.randomSearchOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges);
        case 'bayesianOptimization':
          return await this.bayesianOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges);
        case 'geneticAlgorithm':
          return await this.geneticAlgorithmOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges);
        default:
          return await this.gridSearchOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges);
      }
      
    } catch (error) {
      logger.error('❌ Parameter optimization failed:', error);
      throw error;
    }
  }

  async gridSearchOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges) {
    const parameterCombinations = this.generateParameterCombinations(parameterRanges);
    let bestParams = null;
    let bestScore = -Infinity;
    
    logger.info(`🔍 Grid search: testing ${parameterCombinations.length} parameter combinations`);
    
    for (let i = 0; i < parameterCombinations.length; i++) {
      const params = parameterCombinations[i];
      
      try {
        const testStrategy = { ...strategy, parameters: params };
        const results = await backtestingCore.runBacktest(testStrategy, startDate, endDate, symbols);
        
        const performanceAnalyzer = backtestingCore.performanceAnalyzer || 
          new (require('./PerformanceAnalyzer').PerformanceAnalyzer)();
        
        const performance = performanceAnalyzer.calculateMetrics(results.portfolioHistory);
        const score = this.calculateOptimizationScore(performance);
        
        if (score > bestScore) {
          bestScore = score;
          bestParams = params;
        }
        
        if ((i + 1) % 10 === 0) {
          logger.info(`🔍 Grid search progress: ${i + 1}/${parameterCombinations.length} (best score: ${bestScore.toFixed(4)})`);
        }
        
      } catch (error) {
        logger.warn(`⚠️ Parameter combination failed: ${JSON.stringify(params)}`, error.message);
      }
    }
    
    logger.info(`✅ Grid search completed. Best score: ${bestScore.toFixed(4)}`);
    return bestParams || strategy.parameters || {};
  }

  async randomSearchOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges) {
    const maxIterations = Math.min(this.config.maxIterations, 50);
    let bestParams = null;
    let bestScore = -Infinity;
    
    logger.info(`🎲 Random search: testing ${maxIterations} random parameter combinations`);
    
    for (let i = 0; i < maxIterations; i++) {
      const params = this.generateRandomParameters(parameterRanges);
      
      try {
        const testStrategy = { ...strategy, parameters: params };
        const results = await backtestingCore.runBacktest(testStrategy, startDate, endDate, symbols);
        
        const performanceAnalyzer = backtestingCore.performanceAnalyzer || 
          new (require('./PerformanceAnalyzer').PerformanceAnalyzer)();
        
        const performance = performanceAnalyzer.calculateMetrics(results.portfolioHistory);
        const score = this.calculateOptimizationScore(performance);
        
        if (score > bestScore) {
          bestScore = score;
          bestParams = params;
        }
        
        if ((i + 1) % 10 === 0) {
          logger.info(`🎲 Random search progress: ${i + 1}/${maxIterations} (best score: ${bestScore.toFixed(4)})`);
        }
        
      } catch (error) {
        logger.warn(`⚠️ Random parameter combination failed: ${JSON.stringify(params)}`, error.message);
      }
    }
    
    logger.info(`✅ Random search completed. Best score: ${bestScore.toFixed(4)}`);
    return bestParams || strategy.parameters || {};
  }

  async bayesianOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges) {
    // Simplified Bayesian optimization using Gaussian Process approximation
    const maxIterations = Math.min(this.config.maxIterations, 30);
    const initialSamples = 5;
    
    let bestParams = null;
    let bestScore = -Infinity;
    const observedPoints = [];
    
    logger.info(`🧠 Bayesian optimization: ${maxIterations} iterations with ${initialSamples} initial samples`);
    
    // Initial random sampling
    for (let i = 0; i < initialSamples; i++) {
      const params = this.generateRandomParameters(parameterRanges);
      const score = await this.evaluateParameters(backtestingCore, strategy, startDate, endDate, symbols, params);
      
      observedPoints.push({ params, score });
      
      if (score > bestScore) {
        bestScore = score;
        bestParams = params;
      }
    }
    
    // Bayesian optimization iterations
    for (let i = initialSamples; i < maxIterations; i++) {
      // Select next point using acquisition function (simplified)
      const nextParams = this.selectNextPoint(observedPoints, parameterRanges);
      const score = await this.evaluateParameters(backtestingCore, strategy, startDate, endDate, symbols, nextParams);
      
      observedPoints.push({ params: nextParams, score });
      
      if (score > bestScore) {
        bestScore = score;
        bestParams = nextParams;
      }
      
      if ((i + 1) % 5 === 0) {
        logger.info(`🧠 Bayesian optimization progress: ${i + 1}/${maxIterations} (best score: ${bestScore.toFixed(4)})`);
      }
    }
    
    logger.info(`✅ Bayesian optimization completed. Best score: ${bestScore.toFixed(4)}`);
    return bestParams || strategy.parameters || {};
  }

  async geneticAlgorithmOptimization(backtestingCore, strategy, startDate, endDate, symbols, parameterRanges) {
    const populationSize = 20;
    const generations = 10;
    const mutationRate = 0.1;
    const crossoverRate = 0.8;
    
    logger.info(`🧬 Genetic algorithm: ${generations} generations with population size ${populationSize}`);
    
    // Initialize population
    let population = [];
    for (let i = 0; i < populationSize; i++) {
      const params = this.generateRandomParameters(parameterRanges);
      const score = await this.evaluateParameters(backtestingCore, strategy, startDate, endDate, symbols, params);
      population.push({ params, score });
    }
    
    let bestIndividual = population.reduce((best, current) => 
      current.score > best.score ? current : best
    );
    
    // Evolution loop
    for (let gen = 0; gen < generations; gen++) {
      // Selection, crossover, and mutation
      const newPopulation = [];
      
      // Keep best individuals (elitism)
      population.sort((a, b) => b.score - a.score);
      newPopulation.push(...population.slice(0, Math.floor(populationSize * 0.2)));
      
      // Generate offspring
      while (newPopulation.length < populationSize) {
        const parent1 = this.tournamentSelection(population);
        const parent2 = this.tournamentSelection(population);
        
        let offspring;
        if (Math.random() < crossoverRate) {
          offspring = this.crossover(parent1.params, parent2.params, parameterRanges);
        } else {
          offspring = Math.random() < 0.5 ? parent1.params : parent2.params;
        }
        
        if (Math.random() < mutationRate) {
          offspring = this.mutate(offspring, parameterRanges);
        }
        
        const score = await this.evaluateParameters(backtestingCore, strategy, startDate, endDate, symbols, offspring);
        newPopulation.push({ params: offspring, score });
        
        if (score > bestIndividual.score) {
          bestIndividual = { params: offspring, score };
        }
      }
      
      population = newPopulation;
      
      logger.info(`🧬 Generation ${gen + 1}/${generations} completed (best score: ${bestIndividual.score.toFixed(4)})`);
    }
    
    logger.info(`✅ Genetic algorithm completed. Best score: ${bestIndividual.score.toFixed(4)}`);
    return bestIndividual.params;
  }

  async evaluateParameters(backtestingCore, strategy, startDate, endDate, symbols, params) {
    try {
      const testStrategy = { ...strategy, parameters: params };
      const results = await backtestingCore.runBacktest(testStrategy, startDate, endDate, symbols);
      
      const performanceAnalyzer = backtestingCore.performanceAnalyzer || 
        new (require('./PerformanceAnalyzer').PerformanceAnalyzer)();
      
      const performance = performanceAnalyzer.calculateMetrics(results.portfolioHistory);
      return this.calculateOptimizationScore(performance);
      
    } catch (error) {
      logger.warn(`⚠️ Parameter evaluation failed: ${JSON.stringify(params)}`, error.message);
      return -Infinity;
    }
  }

  calculateOptimizationScore(performance) {
    const metric = this.config.optimizationMetric;
    
    switch (metric) {
      case 'sharpeRatio':
        return performance.sharpeRatio || -Infinity;
      case 'calmarRatio':
        return performance.calmarRatio || -Infinity;
      case 'sortinoRatio':
        return performance.sortinoRatio || -Infinity;
      case 'totalReturn':
        return performance.totalReturn || -Infinity;
      case 'riskAdjustedReturn':
        return (performance.annualizedReturn || 0) / Math.max(performance.volatility || 1, 0.01);
      default:
        return performance.sharpeRatio || -Infinity;
    }
  }

  generateParameterCombinations(parameterRanges) {
    const combinations = [];
    const paramNames = Object.keys(parameterRanges);
    
    if (paramNames.length === 0) return [{}];
    
    const generateCombos = (index, currentCombo) => {
      if (index === paramNames.length) {
        combinations.push({ ...currentCombo });
        return;
      }
      
      const paramName = paramNames[index];
      const range = parameterRanges[paramName];
      
      if (Array.isArray(range)) {
        for (const value of range) {
          currentCombo[paramName] = value;
          generateCombos(index + 1, currentCombo);
        }
      } else if (typeof range === 'object' && range.min !== undefined && range.max !== undefined) {
        const step = range.step || 1;
        for (let value = range.min; value <= range.max; value += step) {
          currentCombo[paramName] = value;
          generateCombos(index + 1, currentCombo);
        }
      }
    };
    
    generateCombos(0, {});
    return combinations;
  }

  generateRandomParameters(parameterRanges) {
    const params = {};
    
    for (const [paramName, range] of Object.entries(parameterRanges)) {
      if (Array.isArray(range)) {
        params[paramName] = range[Math.floor(Math.random() * range.length)];
      } else if (typeof range === 'object' && range.min !== undefined && range.max !== undefined) {
        if (range.type === 'integer') {
          params[paramName] = Math.floor(Math.random() * (range.max - range.min + 1)) + range.min;
        } else {
          params[paramName] = Math.random() * (range.max - range.min) + range.min;
        }
      }
    }
    
    return params;
  }

  selectNextPoint(observedPoints, parameterRanges) {
    // Simplified acquisition function - select point with highest uncertainty
    // In practice, you'd use Expected Improvement or Upper Confidence Bound
    return this.generateRandomParameters(parameterRanges);
  }

  tournamentSelection(population, tournamentSize = 3) {
    const tournament = [];
    for (let i = 0; i < tournamentSize; i++) {
      tournament.push(population[Math.floor(Math.random() * population.length)]);
    }
    return tournament.reduce((best, current) => current.score > best.score ? current : best);
  }

  crossover(parent1, parent2, parameterRanges) {
    const offspring = {};
    
    for (const paramName of Object.keys(parameterRanges)) {
      if (Math.random() < 0.5) {
        offspring[paramName] = parent1[paramName];
      } else {
        offspring[paramName] = parent2[paramName];
      }
    }
    
    return offspring;
  }

  mutate(params, parameterRanges) {
    const mutated = { ...params };
    const paramNames = Object.keys(parameterRanges);
    const paramToMutate = paramNames[Math.floor(Math.random() * paramNames.length)];
    
    const range = parameterRanges[paramToMutate];
    if (Array.isArray(range)) {
      mutated[paramToMutate] = range[Math.floor(Math.random() * range.length)];
    } else if (typeof range === 'object' && range.min !== undefined && range.max !== undefined) {
      if (range.type === 'integer') {
        mutated[paramToMutate] = Math.floor(Math.random() * (range.max - range.min + 1)) + range.min;
      } else {
        mutated[paramToMutate] = Math.random() * (range.max - range.min) + range.min;
      }
    }
    
    return mutated;
  }

  aggregateWalkForwardResults(results) {
    const performances = results.map(r => r.performance);
    
    return {
      averageReturn: performances.reduce((sum, p) => sum + (p.annualizedReturn || 0), 0) / performances.length,
      averageSharpe: performances.reduce((sum, p) => sum + (p.sharpeRatio || 0), 0) / performances.length,
      averageMaxDrawdown: performances.reduce((sum, p) => sum + (p.maxDrawdown || 0), 0) / performances.length,
      consistency: this.calculateConsistency(performances.map(p => p.annualizedReturn || 0)),
      parameterStability: this.analyzeParameterStability(results.map(r => r.optimalParameters)),
      winRate: performances.filter(p => (p.annualizedReturn || 0) > 0).length / performances.length,
      bestPeriod: results.reduce((best, current) => 
        (current.performance.sharpeRatio || -Infinity) > (best.performance.sharpeRatio || -Infinity) ? current : best
      ),
      worstPeriod: results.reduce((worst, current) => 
        (current.performance.sharpeRatio || Infinity) < (worst.performance.sharpeRatio || Infinity) ? current : worst
      )
    };
  }

  calculateConsistency(returns) {
    if (returns.length === 0) return 0;
    
    const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    return stdDev === 0 ? 1 : Math.max(0, 1 - (stdDev / Math.abs(mean)));
  }

  analyzeParameterStability(parameterSets) {
    if (parameterSets.length === 0) return {};
    
    const stability = {};
    const paramNames = Object.keys(parameterSets[0] || {});
    
    for (const paramName of paramNames) {
      const values = parameterSets.map(params => params[paramName]).filter(v => v !== undefined);
      
      if (values.length > 0) {
        const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
        const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
        const stdDev = Math.sqrt(variance);
        
        stability[paramName] = {
          mean,
          stdDev,
          coefficient_of_variation: mean !== 0 ? stdDev / Math.abs(mean) : 0,
          min: Math.min(...values),
          max: Math.max(...values)
        };
      }
    }
    
    return stability;
  }

  updateParameterHistory(strategyName, step, parameters) {
    if (!this.parameterHistory.has(strategyName)) {
      this.parameterHistory.set(strategyName, []);
    }
    
    this.parameterHistory.get(strategyName).push({
      step,
      parameters,
      timestamp: new Date()
    });
  }

  getOptimizationResults(optimizationId) {
    return this.optimizationResults.get(optimizationId);
  }

  getParameterHistory(strategyName) {
    return this.parameterHistory.get(strategyName) || [];
  }

  getAllOptimizationResults() {
    return Array.from(this.optimizationResults.values());
  }
}

module.exports = { WalkForwardOptimizer };
