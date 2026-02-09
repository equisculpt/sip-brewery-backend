const { spawn } = require('child_process');
const path = require('path');
const logger = require('../utils/logger');
const Redis = require('ioredis');

class MLInferenceService {
  constructor() {
    this.pythonPath = process.env.PYTHON_PATH || 'python';
    this.mlModelsPath = path.join(__dirname, '../../ml/models');
    this.redis = new Redis({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379,
      db: 1
    });
    this.cacheEnabled = process.env.ML_CACHE_ENABLED !== 'false';
    this.cacheTTL = 3600; // 1 hour
  }

  async optimizePortfolio(userId, portfolioData, userProfile) {
    try {
      logger.info('Optimizing portfolio with RL model', { userId });

      const cacheKey = `ml:portfolio:${userId}:${JSON.stringify(portfolioData).slice(0, 50)}`;
      
      if (this.cacheEnabled) {
        const cached = await this.redis.get(cacheKey);
        if (cached) {
          logger.info('Returning cached portfolio optimization');
          return JSON.parse(cached);
        }
      }

      const scriptPath = path.join(this.mlModelsPath, 'inference', 'portfolio_optimizer_inference.py');
      
      const input = {
        user_id: userId,
        portfolio: portfolioData,
        user_profile: userProfile
      };

      const result = await this.runPythonScript(scriptPath, input);

      if (this.cacheEnabled) {
        await this.redis.setex(cacheKey, this.cacheTTL, JSON.stringify(result));
      }

      return {
        success: true,
        data: {
          optimal_weights: result.weights,
          expected_return: result.expected_return,
          volatility: result.volatility,
          sharpe_ratio: result.sharpe_ratio,
          confidence: result.confidence,
          rebalancing_needed: result.turnover > 0.05,
          recommendations: this.generateRebalancingRecommendations(result)
        }
      };
    } catch (error) {
      logger.error('Portfolio optimization failed', { error: error.message, userId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async predictFundPerformance(fundIds, timeHorizon = '1Y') {
    try {
      logger.info('Predicting fund performance with GNN model', { fundIds, timeHorizon });

      const cacheKey = `ml:fund:${fundIds.join(',')}:${timeHorizon}`;
      
      if (this.cacheEnabled) {
        const cached = await this.redis.get(cacheKey);
        if (cached) {
          return JSON.parse(cached);
        }
      }

      const scriptPath = path.join(this.mlModelsPath, 'inference', 'fund_predictor_inference.py');
      
      const input = {
        fund_ids: fundIds,
        time_horizon: timeHorizon
      };

      const result = await this.runPythonScript(scriptPath, input);

      if (this.cacheEnabled) {
        await this.redis.setex(cacheKey, this.cacheTTL, JSON.stringify(result));
      }

      return {
        success: true,
        data: result.predictions
      };
    } catch (error) {
      logger.error('Fund performance prediction failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async predictRisk(userId, portfolioData, timeHorizon = '1M') {
    try {
      logger.info('Predicting portfolio risk with LSTM model', { userId, timeHorizon });

      const scriptPath = path.join(this.mlModelsPath, 'inference', 'risk_predictor_inference.py');
      
      const input = {
        user_id: userId,
        portfolio: portfolioData,
        time_horizon: timeHorizon
      };

      const result = await this.runPythonScript(scriptPath, input);

      return {
        success: true,
        data: {
          var_1d: result.var_1d,
          var_1w: result.var_1w,
          var_1m: result.var_1m,
          cvar_1d: result.cvar_1d,
          cvar_1w: result.cvar_1w,
          cvar_1m: result.cvar_1m,
          volatility: result.volatility_1m,
          max_drawdown: result.max_drawdown,
          confidence: result.confidence,
          risk_level: this.categorizeRisk(result),
          recommendations: this.generateRiskRecommendations(result)
        }
      };
    } catch (error) {
      logger.error('Risk prediction failed', { error: error.message, userId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async predictUserBehavior(userId, userHistory, currentContext) {
    try {
      logger.info('Predicting user behavior with BERT model', { userId });

      const scriptPath = path.join(this.mlModelsPath, 'inference', 'behavioral_predictor_inference.py');
      
      const input = {
        user_id: userId,
        history: userHistory,
        context: currentContext
      };

      const result = await this.runPythonScript(scriptPath, input);

      return {
        success: true,
        data: {
          next_action: result.top_actions[0],
          action_probabilities: result.top_actions,
          churn_probability: result.churn_probability,
          predicted_amount: result.predicted_amount,
          behavioral_biases: result.behavioral_biases,
          recommendations: this.generateBehavioralRecommendations(result)
        }
      };
    } catch (error) {
      logger.error('Behavioral prediction failed', { error: error.message, userId });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async detectMarketRegime() {
    try {
      logger.info('Detecting market regime');

      const cacheKey = 'ml:market:regime:current';
      
      if (this.cacheEnabled) {
        const cached = await this.redis.get(cacheKey);
        if (cached) {
          return JSON.parse(cached);
        }
      }

      const scriptPath = path.join(this.mlModelsPath, 'inference', 'regime_detector_inference.py');
      
      const result = await this.runPythonScript(scriptPath, {});

      if (this.cacheEnabled) {
        await this.redis.setex(cacheKey, 900, JSON.stringify(result)); // 15 min cache
      }

      return {
        success: true,
        data: {
          current_regime: result.regime,
          confidence: result.confidence,
          probabilities: result.probabilities,
          expected_duration: result.expected_duration_days,
          transition_probabilities: result.transition_probabilities,
          recommendations: this.generateRegimeRecommendations(result)
        }
      };
    } catch (error) {
      logger.error('Market regime detection failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async runPythonScript(scriptPath, input) {
    return new Promise((resolve, reject) => {
      const python = spawn(this.pythonPath, [scriptPath]);
      
      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code !== 0) {
          logger.error('Python script failed', { code, stderr });
          reject(new Error(`Python script exited with code ${code}: ${stderr}`));
        } else {
          try {
            const result = JSON.parse(stdout);
            resolve(result);
          } catch (error) {
            logger.error('Failed to parse Python output', { stdout, error: error.message });
            reject(new Error('Failed to parse Python output'));
          }
        }
      });

      python.stdin.write(JSON.stringify(input));
      python.stdin.end();

      setTimeout(() => {
        python.kill();
        reject(new Error('Python script timeout'));
      }, 30000);
    });
  }

  generateRebalancingRecommendations(optimizationResult) {
    const recommendations = [];
    
    if (optimizationResult.turnover > 0.10) {
      recommendations.push({
        type: 'REBALANCE',
        priority: 'HIGH',
        message: 'Significant portfolio rebalancing recommended',
        expected_benefit: `+${(optimizationResult.expected_return * 100).toFixed(2)}% return`
      });
    }

    if (optimizationResult.sharpe_ratio > 1.5) {
      recommendations.push({
        type: 'MAINTAIN',
        priority: 'LOW',
        message: 'Portfolio is well-optimized',
        expected_benefit: 'Continue current strategy'
      });
    }

    return recommendations;
  }

  categorizeRisk(riskMetrics) {
    const var_1m = Math.abs(riskMetrics.var_1m);
    
    if (var_1m < 0.05) return 'LOW';
    if (var_1m < 0.10) return 'MODERATE';
    if (var_1m < 0.15) return 'HIGH';
    return 'VERY_HIGH';
  }

  generateRiskRecommendations(riskMetrics) {
    const recommendations = [];
    const riskLevel = this.categorizeRisk(riskMetrics);

    if (riskLevel === 'HIGH' || riskLevel === 'VERY_HIGH') {
      recommendations.push({
        type: 'REDUCE_RISK',
        priority: 'HIGH',
        message: 'Consider reducing portfolio risk',
        actions: ['Increase debt allocation', 'Reduce equity exposure', 'Add gold/hybrid funds']
      });
    }

    if (Math.abs(riskMetrics.max_drawdown) > 0.20) {
      recommendations.push({
        type: 'DIVERSIFY',
        priority: 'HIGH',
        message: 'High drawdown risk detected',
        actions: ['Diversify across sectors', 'Add defensive funds']
      });
    }

    return recommendations;
  }

  generateBehavioralRecommendations(behaviorPrediction) {
    const recommendations = [];

    if (behaviorPrediction.churn_probability > 0.7) {
      recommendations.push({
        type: 'RETENTION',
        priority: 'CRITICAL',
        message: 'High churn risk detected',
        actions: ['Offer personalized insights', 'Provide portfolio review', 'Engage with support']
      });
    }

    const topBias = Object.entries(behaviorPrediction.behavioral_biases)
      .sort((a, b) => b[1] - a[1])[0];

    if (topBias && topBias[1] > 0.7) {
      recommendations.push({
        type: 'BEHAVIORAL_NUDGE',
        priority: 'MEDIUM',
        message: `${topBias[0]} detected`,
        actions: [`Provide educational content about ${topBias[0]}`]
      });
    }

    return recommendations;
  }

  generateRegimeRecommendations(regimeDetection) {
    const recommendations = [];
    const regime = regimeDetection.regime;

    const strategies = {
      'BULL': {
        message: 'Bull market detected - favor growth strategies',
        actions: ['Increase equity allocation', 'Focus on growth funds', 'Consider mid/small cap']
      },
      'BEAR': {
        message: 'Bear market detected - defensive positioning',
        actions: ['Increase debt allocation', 'Focus on quality funds', 'Consider gold/defensive sectors']
      },
      'SIDEWAYS': {
        message: 'Range-bound market - balanced approach',
        actions: ['Maintain diversified portfolio', 'Focus on dividend funds', 'Regular rebalancing']
      },
      'VOLATILE': {
        message: 'High volatility - risk management critical',
        actions: ['Reduce position sizes', 'Increase cash allocation', 'Use SIP for averaging']
      }
    };

    if (strategies[regime]) {
      recommendations.push({
        type: 'MARKET_STRATEGY',
        priority: 'HIGH',
        ...strategies[regime]
      });
    }

    return recommendations;
  }

  async clearCache(pattern = 'ml:*') {
    try {
      const keys = await this.redis.keys(pattern);
      if (keys.length > 0) {
        await this.redis.del(...keys);
        logger.info(`Cleared ${keys.length} ML cache entries`);
      }
    } catch (error) {
      logger.error('Failed to clear ML cache', { error: error.message });
    }
  }
}

module.exports = new MLInferenceService();
