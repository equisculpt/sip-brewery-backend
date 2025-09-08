/**
 * 📋 STRATEGY TEMPLATES
 * 
 * Pre-built trading strategy templates for backtesting
 * 
 * @author 35+ Years ASI Engineering Experience
 * @version 4.0.0 - Trading Strategy Library
 */

const logger = require('../../utils/logger');

class StrategyTemplates {
  constructor() {
    this.templates = new Map();
    this.initializeTemplates();
  }

  initializeTemplates() {
    // Buy and Hold Strategy
    this.templates.set('buyAndHold', {
      name: 'Buy and Hold',
      description: 'Simple buy and hold strategy with equal weighting',
      parameters: {
        initialAllocation: 0.8, // 80% of cash allocated initially
        rebalanceFrequency: 'never' // never, monthly, quarterly
      },
      execute: async (data, timestamp, portfolio, parameters) => {
        const signals = [];
        
        // Buy on first day, hold thereafter
        if (portfolio.positions.size === 0 && portfolio.cash > 1000) {
          const allocation = parameters.initialAllocation || 0.8;
          const cashToInvest = portfolio.cash * allocation;
          const perSymbol = cashToInvest / data.size;
          
          for (const [symbol, marketData] of data) {
            const quantity = Math.floor(perSymbol / marketData.price);
            if (quantity > 0) {
              signals.push({
                symbol,
                action: 'buy',
                quantity,
                reason: 'Initial buy and hold allocation'
              });
            }
          }
        }
        
        return signals;
      }
    });

    // Moving Average Crossover Strategy
    this.templates.set('movingAverageCrossover', {
      name: 'Moving Average Crossover',
      description: 'Buy when short MA crosses above long MA, sell when opposite',
      parameters: {
        shortPeriod: 20,
        longPeriod: 50,
        positionSize: 0.1 // 10% of portfolio per position
      },
      execute: async (data, timestamp, portfolio, parameters) => {
        const signals = [];
        const { shortPeriod, longPeriod, positionSize } = parameters;
        
        for (const [symbol, marketData] of data) {
          if (marketData.shortMA && marketData.longMA && 
              marketData.prevShortMA && marketData.prevLongMA) {
            
            const position = portfolio.positions.get(symbol);
            const currentCross = marketData.shortMA > marketData.longMA;
            const prevCross = marketData.prevShortMA > marketData.prevLongMA;
            
            // Buy signal: short MA crosses above long MA
            if (currentCross && !prevCross && !position && portfolio.cash > 1000) {
              const investAmount = portfolio.totalValue * positionSize;
              const quantity = Math.floor(investAmount / marketData.price);
              
              if (quantity > 0) {
                signals.push({
                  symbol,
                  action: 'buy',
                  quantity,
                  reason: `MA crossover buy (${shortPeriod}/${longPeriod})`
                });
              }
            }
            
            // Sell signal: short MA crosses below long MA
            if (!currentCross && prevCross && position) {
              signals.push({
                symbol,
                action: 'sell',
                quantity: position.quantity,
                reason: `MA crossover sell (${shortPeriod}/${longPeriod})`
              });
            }
          }
        }
        
        return signals;
      }
    });

    // Mean Reversion Strategy
    this.templates.set('meanReversion', {
      name: 'Mean Reversion',
      description: 'Buy oversold assets, sell overbought assets based on z-score',
      parameters: {
        lookback: 20,
        entryThreshold: 2.0,
        exitThreshold: 0.5,
        positionSize: 0.1,
        maxPositions: 5
      },
      execute: async (data, timestamp, portfolio, parameters) => {
        const signals = [];
        const { lookback, entryThreshold, exitThreshold, positionSize, maxPositions } = parameters;
        
        for (const [symbol, marketData] of data) {
          if (marketData.zScore !== undefined) {
            const position = portfolio.positions.get(symbol);
            const currentPositions = portfolio.positions.size;
            
            // Buy oversold (negative z-score)
            if (marketData.zScore < -entryThreshold && !position && 
                currentPositions < maxPositions && portfolio.cash > 1000) {
              
              const investAmount = portfolio.totalValue * positionSize;
              const quantity = Math.floor(investAmount / marketData.price);
              
              if (quantity > 0) {
                signals.push({
                  symbol,
                  action: 'buy',
                  quantity,
                  reason: `Mean reversion buy (z-score: ${marketData.zScore.toFixed(2)})`
                });
              }
            }
            
            // Sell when mean reverts or becomes overbought
            if (position && (marketData.zScore > exitThreshold || marketData.zScore > entryThreshold)) {
              signals.push({
                symbol,
                action: 'sell',
                quantity: position.quantity,
                reason: `Mean reversion sell (z-score: ${marketData.zScore.toFixed(2)})`
              });
            }
          }
        }
        
        return signals;
      }
    });

    // Momentum Strategy
    this.templates.set('momentum', {
      name: 'Momentum',
      description: 'Buy assets with strong recent performance, sell weak performers',
      parameters: {
        lookback: 60,
        topPercentile: 0.2, // Top 20% performers
        bottomPercentile: 0.2, // Bottom 20% performers
        positionSize: 0.15,
        rebalanceFrequency: 'monthly'
      },
      execute: async (data, timestamp, portfolio, parameters) => {
        const signals = [];
        const { lookback, topPercentile, bottomPercentile, positionSize } = parameters;
        
        // Calculate momentum scores
        const momentumScores = [];
        for (const [symbol, marketData] of data) {
          if (marketData.momentum !== undefined) {
            momentumScores.push({ symbol, momentum: marketData.momentum, data: marketData });
          }
        }
        
        // Sort by momentum
        momentumScores.sort((a, b) => b.momentum - a.momentum);
        
        const topCount = Math.floor(momentumScores.length * topPercentile);
        const bottomCount = Math.floor(momentumScores.length * bottomPercentile);
        
        // Buy top performers
        for (let i = 0; i < topCount; i++) {
          const { symbol, data: marketData } = momentumScores[i];
          const position = portfolio.positions.get(symbol);
          
          if (!position && portfolio.cash > 1000) {
            const investAmount = portfolio.totalValue * positionSize;
            const quantity = Math.floor(investAmount / marketData.price);
            
            if (quantity > 0) {
              signals.push({
                symbol,
                action: 'buy',
                quantity,
                reason: `Momentum buy (score: ${momentumScores[i].momentum.toFixed(3)})`
              });
            }
          }
        }
        
        // Sell bottom performers
        for (let i = momentumScores.length - bottomCount; i < momentumScores.length; i++) {
          const { symbol } = momentumScores[i];
          const position = portfolio.positions.get(symbol);
          
          if (position) {
            signals.push({
              symbol,
              action: 'sell',
              quantity: position.quantity,
              reason: `Momentum sell (score: ${momentumScores[i].momentum.toFixed(3)})`
            });
          }
        }
        
        return signals;
      }
    });

    // RSI Strategy
    this.templates.set('rsi', {
      name: 'RSI Strategy',
      description: 'Buy when RSI is oversold, sell when overbought',
      parameters: {
        rsiPeriod: 14,
        oversoldLevel: 30,
        overboughtLevel: 70,
        positionSize: 0.1,
        maxPositions: 8
      },
      execute: async (data, timestamp, portfolio, parameters) => {
        const signals = [];
        const { oversoldLevel, overboughtLevel, positionSize, maxPositions } = parameters;
        
        for (const [symbol, marketData] of data) {
          if (marketData.rsi !== undefined) {
            const position = portfolio.positions.get(symbol);
            const currentPositions = portfolio.positions.size;
            
            // Buy when oversold
            if (marketData.rsi < oversoldLevel && !position && 
                currentPositions < maxPositions && portfolio.cash > 1000) {
              
              const investAmount = portfolio.totalValue * positionSize;
              const quantity = Math.floor(investAmount / marketData.price);
              
              if (quantity > 0) {
                signals.push({
                  symbol,
                  action: 'buy',
                  quantity,
                  reason: `RSI oversold buy (RSI: ${marketData.rsi.toFixed(1)})`
                });
              }
            }
            
            // Sell when overbought
            if (marketData.rsi > overboughtLevel && position) {
              signals.push({
                symbol,
                action: 'sell',
                quantity: position.quantity,
                reason: `RSI overbought sell (RSI: ${marketData.rsi.toFixed(1)})`
              });
            }
          }
        }
        
        return signals;
      }
    });

    // Pairs Trading Strategy
    this.templates.set('pairsTrading', {
      name: 'Pairs Trading',
      description: 'Statistical arbitrage between correlated pairs',
      parameters: {
        correlationThreshold: 0.8,
        spreadThreshold: 2.0,
        positionSize: 0.05,
        maxPairs: 3
      },
      execute: async (data, timestamp, portfolio, parameters) => {
        const signals = [];
        const { correlationThreshold, spreadThreshold, positionSize, maxPairs } = parameters;
        
        // This is a simplified pairs trading implementation
        // In practice, you'd need to identify pairs and track their spread
        const symbols = Array.from(data.keys());
        
        for (let i = 0; i < symbols.length - 1; i++) {
          for (let j = i + 1; j < symbols.length; j++) {
            const symbol1 = symbols[i];
            const symbol2 = symbols[j];
            const data1 = data.get(symbol1);
            const data2 = data.get(symbol2);
            
            // Check if we have spread data for this pair
            if (data1.pairSpread && data2.pairSpread && 
                data1.pairSpread[symbol2] !== undefined) {
              
              const spread = data1.pairSpread[symbol2];
              const position1 = portfolio.positions.get(symbol1);
              const position2 = portfolio.positions.get(symbol2);
              
              // Enter long/short positions when spread is extreme
              if (Math.abs(spread) > spreadThreshold && !position1 && !position2) {
                const investAmount = portfolio.totalValue * positionSize;
                
                if (spread > 0) {
                  // Symbol1 is overvalued relative to Symbol2
                  // Short Symbol1, Long Symbol2
                  const qty1 = Math.floor(investAmount / data1.price / 2);
                  const qty2 = Math.floor(investAmount / data2.price / 2);
                  
                  if (qty1 > 0 && qty2 > 0) {
                    signals.push({
                      symbol: symbol1,
                      action: 'sell',
                      quantity: qty1,
                      reason: `Pairs trade short ${symbol1} (spread: ${spread.toFixed(2)})`
                    });
                    
                    signals.push({
                      symbol: symbol2,
                      action: 'buy',
                      quantity: qty2,
                      reason: `Pairs trade long ${symbol2} (spread: ${spread.toFixed(2)})`
                    });
                  }
                }
              }
              
              // Exit when spread normalizes
              if (Math.abs(spread) < 0.5 && (position1 || position2)) {
                if (position1) {
                  signals.push({
                    symbol: symbol1,
                    action: position1.quantity > 0 ? 'sell' : 'buy',
                    quantity: Math.abs(position1.quantity),
                    reason: `Pairs trade exit ${symbol1} (spread: ${spread.toFixed(2)})`
                  });
                }
                
                if (position2) {
                  signals.push({
                    symbol: symbol2,
                    action: position2.quantity > 0 ? 'sell' : 'buy',
                    quantity: Math.abs(position2.quantity),
                    reason: `Pairs trade exit ${symbol2} (spread: ${spread.toFixed(2)})`
                  });
                }
              }
            }
          }
        }
        
        return signals;
      }
    });

    // Bollinger Bands Strategy
    this.templates.set('bollingerBands', {
      name: 'Bollinger Bands',
      description: 'Buy at lower band, sell at upper band',
      parameters: {
        period: 20,
        stdDev: 2,
        positionSize: 0.1,
        maxPositions: 6
      },
      execute: async (data, timestamp, portfolio, parameters) => {
        const signals = [];
        const { positionSize, maxPositions } = parameters;
        
        for (const [symbol, marketData] of data) {
          if (marketData.bollingerUpper && marketData.bollingerLower && marketData.bollingerMiddle) {
            const position = portfolio.positions.get(symbol);
            const currentPositions = portfolio.positions.size;
            const price = marketData.price;
            
            // Buy when price touches lower band
            if (price <= marketData.bollingerLower && !position && 
                currentPositions < maxPositions && portfolio.cash > 1000) {
              
              const investAmount = portfolio.totalValue * positionSize;
              const quantity = Math.floor(investAmount / price);
              
              if (quantity > 0) {
                signals.push({
                  symbol,
                  action: 'buy',
                  quantity,
                  reason: `Bollinger lower band buy (price: ${price.toFixed(2)}, lower: ${marketData.bollingerLower.toFixed(2)})`
                });
              }
            }
            
            // Sell when price touches upper band or middle band (take profit)
            if (position && (price >= marketData.bollingerUpper || price >= marketData.bollingerMiddle)) {
              signals.push({
                symbol,
                action: 'sell',
                quantity: position.quantity,
                reason: `Bollinger band sell (price: ${price.toFixed(2)}, upper: ${marketData.bollingerUpper.toFixed(2)})`
              });
            }
          }
        }
        
        return signals;
      }
    });

    logger.info(`📋 Initialized ${this.templates.size} strategy templates`);
  }

  getTemplate(name) {
    return this.templates.get(name);
  }

  getAllTemplates() {
    return Array.from(this.templates.entries()).map(([name, template]) => ({
      name,
      ...template
    }));
  }

  getTemplateNames() {
    return Array.from(this.templates.keys());
  }

  addCustomTemplate(name, template) {
    if (!template.name || !template.execute) {
      throw new Error('Template must have name and execute function');
    }
    
    this.templates.set(name, template);
    logger.info(`📋 Added custom strategy template: ${name}`);
  }

  removeTemplate(name) {
    const removed = this.templates.delete(name);
    if (removed) {
      logger.info(`📋 Removed strategy template: ${name}`);
    }
    return removed;
  }

  validateTemplate(template) {
    const required = ['name', 'description', 'execute'];
    const missing = required.filter(field => !template[field]);
    
    if (missing.length > 0) {
      throw new Error(`Template missing required fields: ${missing.join(', ')}`);
    }
    
    if (typeof template.execute !== 'function') {
      throw new Error('Template execute must be a function');
    }
    
    return true;
  }
}

module.exports = { StrategyTemplates };
