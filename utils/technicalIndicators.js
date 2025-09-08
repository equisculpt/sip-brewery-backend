/**
 * Advanced Technical Indicators for Mutual Fund Analysis
 * Professional-grade calculations used by institutional traders
 */

class TechnicalIndicators {
  
  /**
   * Simple Moving Average (SMA)
   */
  static calculateSMA(data, period) {
    if (!data || data.length < period) return [];
    
    const result = [];
    for (let i = period - 1; i < data.length; i++) {
      const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
      result.push({
        index: i,
        value: parseFloat((sum / period).toFixed(4))
      });
    }
    return result;
  }

  /**
   * Exponential Moving Average (EMA)
   */
  static calculateEMA(data, period) {
    if (!data || data.length < period) return [];
    
    const multiplier = 2 / (period + 1);
    const result = [];
    
    // Start with SMA for first value
    const sma = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
    result.push({
      index: period - 1,
      value: parseFloat(sma.toFixed(4))
    });
    
    // Calculate EMA for remaining values
    for (let i = period; i < data.length; i++) {
      const ema = (data[i] - result[result.length - 1].value) * multiplier + result[result.length - 1].value;
      result.push({
        index: i,
        value: parseFloat(ema.toFixed(4))
      });
    }
    
    return result;
  }

  /**
   * Relative Strength Index (RSI)
   */
  static calculateRSI(data, period = 14) {
    if (!data || data.length < period + 1) return [];
    
    const gains = [];
    const losses = [];
    
    // Calculate gains and losses
    for (let i = 1; i < data.length; i++) {
      const change = data[i] - data[i - 1];
      gains.push(change > 0 ? change : 0);
      losses.push(change < 0 ? Math.abs(change) : 0);
    }
    
    const result = [];
    
    // Calculate initial average gain and loss
    let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
    let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
    
    // Calculate RSI for each subsequent period
    for (let i = period; i < gains.length; i++) {
      avgGain = ((avgGain * (period - 1)) + gains[i]) / period;
      avgLoss = ((avgLoss * (period - 1)) + losses[i]) / period;
      
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      const rsi = 100 - (100 / (1 + rs));
      
      result.push({
        index: i + 1,
        value: parseFloat(rsi.toFixed(2))
      });
    }
    
    return result;
  }

  /**
   * MACD (Moving Average Convergence Divergence)
   */
  static calculateMACD(data, fastPeriod = 12, slowPeriod = 26, signalPeriod = 9) {
    if (!data || data.length < slowPeriod) return { macd: [], signal: [], histogram: [] };
    
    const fastEMA = this.calculateEMA(data, fastPeriod);
    const slowEMA = this.calculateEMA(data, slowPeriod);
    
    const macdLine = [];
    const startIndex = slowPeriod - 1;
    
    // Calculate MACD line
    for (let i = 0; i < fastEMA.length && i < slowEMA.length; i++) {
      if (fastEMA[i].index >= startIndex && slowEMA[i].index >= startIndex) {
        const macdValue = fastEMA[i].value - slowEMA[i].value;
        macdLine.push({
          index: fastEMA[i].index,
          value: parseFloat(macdValue.toFixed(4))
        });
      }
    }
    
    // Calculate signal line (EMA of MACD)
    const macdValues = macdLine.map(item => item.value);
    const signalEMA = this.calculateEMA(macdValues, signalPeriod);
    
    const signalLine = signalEMA.map((item, index) => ({
      index: macdLine[item.index]?.index || startIndex + signalPeriod - 1 + index,
      value: item.value
    }));
    
    // Calculate histogram (MACD - Signal)
    const histogram = [];
    const minLength = Math.min(macdLine.length, signalLine.length);
    
    for (let i = 0; i < minLength; i++) {
      if (macdLine[i] && signalLine[i]) {
        histogram.push({
          index: macdLine[i].index,
          value: parseFloat((macdLine[i].value - signalLine[i].value).toFixed(4))
        });
      }
    }
    
    return {
      macd: macdLine,
      signal: signalLine,
      histogram: histogram
    };
  }

  /**
   * Bollinger Bands
   */
  static calculateBollingerBands(data, period = 20, stdDev = 2) {
    if (!data || data.length < period) return { upper: [], middle: [], lower: [] };
    
    const sma = this.calculateSMA(data, period);
    const upper = [];
    const middle = [];
    const lower = [];
    
    for (let i = 0; i < sma.length; i++) {
      const dataIndex = sma[i].index;
      const subset = data.slice(dataIndex - period + 1, dataIndex + 1);
      
      // Calculate standard deviation
      const mean = sma[i].value;
      const variance = subset.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / period;
      const standardDeviation = Math.sqrt(variance);
      
      middle.push({
        index: dataIndex,
        value: sma[i].value
      });
      
      upper.push({
        index: dataIndex,
        value: parseFloat((mean + (stdDev * standardDeviation)).toFixed(4))
      });
      
      lower.push({
        index: dataIndex,
        value: parseFloat((mean - (stdDev * standardDeviation)).toFixed(4))
      });
    }
    
    return { upper, middle, lower };
  }

  /**
   * Stochastic Oscillator
   */
  static calculateStochastic(highs, lows, closes, kPeriod = 14, dPeriod = 3) {
    if (!highs || !lows || !closes || highs.length < kPeriod) {
      return { k: [], d: [] };
    }
    
    const kValues = [];
    
    // Calculate %K
    for (let i = kPeriod - 1; i < closes.length; i++) {
      const highestHigh = Math.max(...highs.slice(i - kPeriod + 1, i + 1));
      const lowestLow = Math.min(...lows.slice(i - kPeriod + 1, i + 1));
      
      const kValue = ((closes[i] - lowestLow) / (highestHigh - lowestLow)) * 100;
      
      kValues.push({
        index: i,
        value: parseFloat(kValue.toFixed(2))
      });
    }
    
    // Calculate %D (SMA of %K)
    const kValuesOnly = kValues.map(item => item.value);
    const dValues = this.calculateSMA(kValuesOnly, dPeriod);
    
    const dLine = dValues.map((item, index) => ({
      index: kValues[item.index]?.index || kPeriod - 1 + dPeriod - 1 + index,
      value: item.value
    }));
    
    return {
      k: kValues,
      d: dLine
    };
  }

  /**
   * Average True Range (ATR)
   */
  static calculateATR(highs, lows, closes, period = 14) {
    if (!highs || !lows || !closes || highs.length < 2) return [];
    
    const trueRanges = [];
    
    // Calculate True Range for each period
    for (let i = 1; i < highs.length; i++) {
      const tr1 = highs[i] - lows[i];
      const tr2 = Math.abs(highs[i] - closes[i - 1]);
      const tr3 = Math.abs(lows[i] - closes[i - 1]);
      
      const trueRange = Math.max(tr1, tr2, tr3);
      trueRanges.push(trueRange);
    }
    
    // Calculate ATR using EMA
    const atr = this.calculateEMA(trueRanges, period);
    
    return atr.map(item => ({
      index: item.index + 1, // Adjust for the offset
      value: item.value
    }));
  }

  /**
   * Williams %R
   */
  static calculateWilliamsR(highs, lows, closes, period = 14) {
    if (!highs || !lows || !closes || highs.length < period) return [];
    
    const result = [];
    
    for (let i = period - 1; i < closes.length; i++) {
      const highestHigh = Math.max(...highs.slice(i - period + 1, i + 1));
      const lowestLow = Math.min(...lows.slice(i - period + 1, i + 1));
      
      const williamsR = ((highestHigh - closes[i]) / (highestHigh - lowestLow)) * -100;
      
      result.push({
        index: i,
        value: parseFloat(williamsR.toFixed(2))
      });
    }
    
    return result;
  }

  /**
   * Commodity Channel Index (CCI)
   */
  static calculateCCI(highs, lows, closes, period = 20) {
    if (!highs || !lows || !closes || highs.length < period) return [];
    
    const typicalPrices = [];
    
    // Calculate Typical Price
    for (let i = 0; i < highs.length; i++) {
      const tp = (highs[i] + lows[i] + closes[i]) / 3;
      typicalPrices.push(tp);
    }
    
    const result = [];
    
    for (let i = period - 1; i < typicalPrices.length; i++) {
      const subset = typicalPrices.slice(i - period + 1, i + 1);
      const sma = subset.reduce((sum, val) => sum + val, 0) / period;
      
      // Calculate Mean Deviation
      const meanDeviation = subset.reduce((sum, val) => sum + Math.abs(val - sma), 0) / period;
      
      const cci = (typicalPrices[i] - sma) / (0.015 * meanDeviation);
      
      result.push({
        index: i,
        value: parseFloat(cci.toFixed(2))
      });
    }
    
    return result;
  }

  /**
   * Money Flow Index (MFI)
   */
  static calculateMFI(highs, lows, closes, volumes, period = 14) {
    if (!highs || !lows || !closes || !volumes || highs.length < period + 1) return [];
    
    const typicalPrices = [];
    const rawMoneyFlows = [];
    
    // Calculate Typical Price and Raw Money Flow
    for (let i = 0; i < highs.length; i++) {
      const tp = (highs[i] + lows[i] + closes[i]) / 3;
      typicalPrices.push(tp);
      rawMoneyFlows.push(tp * volumes[i]);
    }
    
    const result = [];
    
    for (let i = period; i < typicalPrices.length; i++) {
      let positiveFlow = 0;
      let negativeFlow = 0;
      
      // Calculate positive and negative money flows
      for (let j = i - period + 1; j <= i; j++) {
        if (typicalPrices[j] > typicalPrices[j - 1]) {
          positiveFlow += rawMoneyFlows[j];
        } else if (typicalPrices[j] < typicalPrices[j - 1]) {
          negativeFlow += rawMoneyFlows[j];
        }
      }
      
      const moneyFlowRatio = negativeFlow === 0 ? 100 : positiveFlow / negativeFlow;
      const mfi = 100 - (100 / (1 + moneyFlowRatio));
      
      result.push({
        index: i,
        value: parseFloat(mfi.toFixed(2))
      });
    }
    
    return result;
  }
}

// Export individual functions for backward compatibility
module.exports = {
  calculateSMA: TechnicalIndicators.calculateSMA,
  calculateEMA: TechnicalIndicators.calculateEMA,
  calculateRSI: TechnicalIndicators.calculateRSI,
  calculateMACD: TechnicalIndicators.calculateMACD,
  calculateBollingerBands: TechnicalIndicators.calculateBollingerBands,
  calculateStochastic: TechnicalIndicators.calculateStochastic,
  calculateATR: TechnicalIndicators.calculateATR,
  calculateWilliamsR: TechnicalIndicators.calculateWilliamsR,
  calculateCCI: TechnicalIndicators.calculateCCI,
  calculateMFI: TechnicalIndicators.calculateMFI,
  TechnicalIndicators
};
