const axios = require('axios');
const { calculateSMA, calculateEMA, calculateRSI, calculateMACD, calculateBollingerBands, calculateStochastic } = require('../utils/technicalIndicators');
const logger = require('../utils/logger');

class MutualFundDataService {
  constructor() {
    this.cache = new Map();
    this.cacheExpiry = 5 * 60 * 1000; // 5 minutes
    this.supportedSchemes = new Map();
    this.initializeSupportedSchemes();
  }

  initializeSupportedSchemes() {
    // Popular mutual fund schemes with their identifiers
    this.supportedSchemes.set('HDFC_TOP_100', {
      name: 'HDFC Top 100 Fund',
      category: 'Large Cap',
      aum: '₹45,000 Cr',
      expense_ratio: 1.05,
      nav_history_url: 'https://api.mfapi.in/mf/119551'
    });
    
    this.supportedSchemes.set('SBI_BLUECHIP', {
      name: 'SBI Bluechip Fund',
      category: 'Large Cap',
      aum: '₹38,500 Cr',
      expense_ratio: 0.98,
      nav_history_url: 'https://api.mfapi.in/mf/119578'
    });
    
    this.supportedSchemes.set('ICICI_FOCUSED_BLUECHIP', {
      name: 'ICICI Prudential Focused Bluechip Equity Fund',
      category: 'Large Cap',
      aum: '₹15,200 Cr',
      expense_ratio: 1.75,
      nav_history_url: 'https://api.mfapi.in/mf/120503'
    });
    
    this.supportedSchemes.set('AXIS_MIDCAP', {
      name: 'Axis Midcap Fund',
      category: 'Mid Cap',
      aum: '₹12,800 Cr',
      expense_ratio: 1.95,
      nav_history_url: 'https://api.mfapi.in/mf/120503'
    });
    
    this.supportedSchemes.set('MIRAE_EMERGING_BLUECHIP', {
      name: 'Mirae Asset Emerging Bluechip Fund',
      category: 'Large & Mid Cap',
      aum: '₹28,400 Cr',
      expense_ratio: 1.15,
      nav_history_url: 'https://api.mfapi.in/mf/122639'
    });
  }

  async getMutualFundNAVHistory(schemeCode, period = '5Y') {
    const cacheKey = `${schemeCode}_${period}`;
    
    // Check cache first
    if (this.cache.has(cacheKey)) {
      const cached = this.cache.get(cacheKey);
      if (Date.now() - cached.timestamp < this.cacheExpiry) {
        return cached.data;
      }
    }

    try {
      const scheme = this.supportedSchemes.get(schemeCode);
      if (!scheme) {
        throw new Error(`Unsupported scheme: ${schemeCode}`);
      }

      // Fetch NAV history from API
      const response = await axios.get(scheme.nav_history_url, {
        timeout: 10000,
        headers: {
          'User-Agent': 'SIPBrewery-TradingView/1.0'
        }
      });

      let navData = response.data.data;
      
      // Filter data based on period
      const periodDays = this.getPeriodInDays(period);
      const cutoffDate = new Date();
      cutoffDate.setDate(cutoffDate.getDate() - periodDays);
      
      navData = navData.filter(item => {
        const itemDate = new Date(item.date);
        return itemDate >= cutoffDate;
      }).reverse(); // Reverse to get chronological order

      // Transform to OHLC format (for mutual funds, we simulate OHLC from NAV)
      const ohlcData = this.transformNAVToOHLC(navData);
      
      // Calculate technical indicators
      const technicalData = await this.calculateTechnicalIndicators(ohlcData);
      
      const result = {
        scheme: {
          code: schemeCode,
          name: scheme.name,
          category: scheme.category,
          aum: scheme.aum,
          expense_ratio: scheme.expense_ratio
        },
        period,
        data: ohlcData,
        technical: technicalData,
        metadata: {
          total_records: ohlcData.length,
          start_date: ohlcData[0]?.date,
          end_date: ohlcData[ohlcData.length - 1]?.date,
          last_updated: new Date().toISOString()
        }
      };

      // Cache the result
      this.cache.set(cacheKey, {
        data: result,
        timestamp: Date.now()
      });

      return result;

    } catch (error) {
      logger.error('Error fetching mutual fund data:', error);
      throw new Error(`Failed to fetch data for ${schemeCode}: ${error.message}`);
    }
  }

  transformNAVToOHLC(navData) {
    return navData.map((item, index) => {
      const nav = parseFloat(item.nav);
      const date = item.date;
      
      // For mutual funds, we simulate OHLC from NAV with small variations
      // This creates realistic-looking candlestick data for charting
      const variation = nav * 0.002; // 0.2% variation
      const random1 = (Math.random() - 0.5) * variation;
      const random2 = (Math.random() - 0.5) * variation;
      
      const open = index > 0 ? parseFloat(navData[index - 1].nav) : nav;
      const close = nav;
      const high = Math.max(open, close) + Math.abs(random1);
      const low = Math.min(open, close) - Math.abs(random2);
      
      return {
        date,
        timestamp: new Date(date).getTime(),
        open: parseFloat(open.toFixed(4)),
        high: parseFloat(high.toFixed(4)),
        low: parseFloat(low.toFixed(4)),
        close: parseFloat(close.toFixed(4)),
        volume: Math.floor(Math.random() * 1000000) + 500000, // Simulated volume
        nav: nav
      };
    });
  }

  async calculateTechnicalIndicators(ohlcData) {
    const closes = ohlcData.map(d => d.close);
    const highs = ohlcData.map(d => d.high);
    const lows = ohlcData.map(d => d.low);
    const volumes = ohlcData.map(d => d.volume);

    return {
      sma: {
        sma_20: calculateSMA(closes, 20),
        sma_50: calculateSMA(closes, 50),
        sma_100: calculateSMA(closes, 100),
        sma_200: calculateSMA(closes, 200)
      },
      ema: {
        ema_12: calculateEMA(closes, 12),
        ema_26: calculateEMA(closes, 26),
        ema_50: calculateEMA(closes, 50)
      },
      rsi: {
        rsi_14: calculateRSI(closes, 14)
      },
      macd: calculateMACD(closes),
      bollinger: calculateBollingerBands(closes, 20, 2),
      stochastic: calculateStochastic(highs, lows, closes, 14, 3),
      volume_sma: calculateSMA(volumes, 20)
    };
  }

  getPeriodInDays(period) {
    switch (period) {
      case '1M': return 30;
      case '3M': return 90;
      case '6M': return 180;
      case '1Y': return 365;
      case '2Y': return 730;
      case '3Y': return 1095;
      case '5Y': return 1825;
      case 'MAX': return 3650; // 10 years max
      default: return 1825; // 5 years default
    }
  }

  async getSchemeComparison(schemeCodes, period = '1Y') {
    try {
      const promises = schemeCodes.map(code => this.getMutualFundNAVHistory(code, period));
      const results = await Promise.all(promises);
      
      // Calculate relative performance
      const comparison = results.map(result => {
        const data = result.data;
        const startNAV = data[0]?.nav || 0;
        const endNAV = data[data.length - 1]?.nav || 0;
        const returns = ((endNAV - startNAV) / startNAV) * 100;
        
        return {
          scheme: result.scheme,
          returns: parseFloat(returns.toFixed(2)),
          start_nav: startNAV,
          end_nav: endNAV,
          volatility: this.calculateVolatility(data),
          sharpe_ratio: this.calculateSharpeRatio(data),
          max_drawdown: this.calculateMaxDrawdown(data)
        };
      });

      return {
        period,
        comparison,
        best_performer: comparison.reduce((best, current) => 
          current.returns > best.returns ? current : best
        ),
        metadata: {
          compared_schemes: schemeCodes.length,
          analysis_date: new Date().toISOString()
        }
      };

    } catch (error) {
      logger.error('Error in scheme comparison:', error);
      throw error;
    }
  }

  calculateVolatility(data) {
    const returns = [];
    for (let i = 1; i < data.length; i++) {
      const dailyReturn = (data[i].nav - data[i-1].nav) / data[i-1].nav;
      returns.push(dailyReturn);
    }
    
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance * 252) * 100; // Annualized volatility in %
  }

  calculateSharpeRatio(data, riskFreeRate = 0.06) {
    const returns = [];
    for (let i = 1; i < data.length; i++) {
      const dailyReturn = (data[i].nav - data[i-1].nav) / data[i-1].nav;
      returns.push(dailyReturn);
    }
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const annualizedReturn = avgReturn * 252;
    const volatility = this.calculateVolatility(data) / 100;
    
    return (annualizedReturn - riskFreeRate) / volatility;
  }

  calculateMaxDrawdown(data) {
    let maxDrawdown = 0;
    let peak = data[0]?.nav || 0;
    
    for (const point of data) {
      if (point.nav > peak) {
        peak = point.nav;
      }
      
      const drawdown = (peak - point.nav) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    
    return maxDrawdown * 100; // Return as percentage
  }

  async getTopPerformingSchemes(category = 'ALL', period = '1Y', limit = 10) {
    try {
      const allSchemes = Array.from(this.supportedSchemes.keys());
      const filteredSchemes = category === 'ALL' ? 
        allSchemes : 
        allSchemes.filter(code => {
          const scheme = this.supportedSchemes.get(code);
          return scheme.category.toLowerCase().includes(category.toLowerCase());
        });

      const comparison = await this.getSchemeComparison(filteredSchemes, period);
      
      return {
        category,
        period,
        top_performers: comparison.comparison
          .sort((a, b) => b.returns - a.returns)
          .slice(0, limit),
        metadata: {
          total_analyzed: filteredSchemes.length,
          analysis_date: new Date().toISOString()
        }
      };

    } catch (error) {
      logger.error('Error getting top performing schemes:', error);
      throw error;
    }
  }

  getSupportedSchemes() {
    return Array.from(this.supportedSchemes.entries()).map(([code, scheme]) => ({
      code,
      ...scheme
    }));
  }
}

module.exports = MutualFundDataService;
