const moment = require('moment');
const xirr = require('xirr');
const logger = require('../utils/logger');
const portfolioAnalyticsService = require('./portfolioAnalyticsService');
const navHistoryService = require('./navHistoryService');

class ChartDataService {
  constructor() {
    this.chartTypes = {
      PORTFOLIO_PERFORMANCE: 'portfolio_performance',
      SIP_PROJECTION: 'sip_projection',
      NAV_HISTORY: 'nav_history',
      ALLOCATION_PIE: 'allocation_pie',
      RISK_RETURN_SCATTER: 'risk_return_scatter',
      COMPARISON_CHART: 'comparison_chart',
      VOLATILITY_CHART: 'volatility_chart',
      DRAWDOWN_CHART: 'drawdown_chart',
      TAX_ANALYSIS: 'tax_analysis',
      GOAL_PROGRESS: 'goal_progress'
    };
  }

  /**
   * Generate chart data for various chart types
   */
  async generateChartData({ userId, chartType, period = '1y', fundCode, options = {} }) {
    try {
      logger.info(`Generating chart data for user: ${userId}, chartType: ${chartType}`);

      switch (chartType) {
        case this.chartTypes.PORTFOLIO_PERFORMANCE:
          return await this.generatePortfolioPerformanceChart(userId, period, options);
        
        case this.chartTypes.SIP_PROJECTION:
          return await this.generateSIPProjectionChart(userId, period, options);
        
        case this.chartTypes.NAV_HISTORY:
          return await this.generateNAVHistoryChart(fundCode, period, options);
        
        case this.chartTypes.ALLOCATION_PIE:
          return await this.generateAllocationPieChart(userId, options);
        
        case this.chartTypes.RISK_RETURN_SCATTER:
          return await this.generateRiskReturnScatterChart(userId, options);
        
        case this.chartTypes.COMPARISON_CHART:
          return await this.generateComparisonChart(userId, period, options);
        
        case this.chartTypes.VOLATILITY_CHART:
          return await this.generateVolatilityChart(userId, period, options);
        
        case this.chartTypes.DRAWDOWN_CHART:
          return await this.generateDrawdownChart(userId, period, options);
        
        case this.chartTypes.TAX_ANALYSIS:
          return await this.generateTaxAnalysisChart(userId, options);
        
        case this.chartTypes.GOAL_PROGRESS:
          return await this.generateGoalProgressChart(userId, options);
        
        default:
          throw new Error(`Unsupported chart type: ${chartType}`);
      }
    } catch (error) {
      logger.error('Error generating chart data:', error);
      throw error;
    }
  }

  /**
   * Generate portfolio performance chart data
   */
  async generatePortfolioPerformanceChart(userId, period, options) {
    try {
      const portfolioData = await portfolioAnalyticsService.getPortfolioSummary(userId);
      const historicalData = await this.getHistoricalPerformanceData(userId, period);
      
      const chartData = {
        type: 'line',
        title: 'Portfolio Performance',
        xAxis: {
          type: 'datetime',
          title: 'Date'
        },
        yAxis: {
          title: 'Portfolio Value (₹)',
          labels: {
            formatter: function() {
              return '₹' + this.value.toLocaleString('en-IN');
            }
          }
        },
        series: [
          {
            name: 'Portfolio Value',
            data: historicalData.map(point => [point.date, point.value]),
            color: '#2E8B57'
          },
          {
            name: 'Investment Amount',
            data: historicalData.map(point => [point.date, point.invested]),
            color: '#4682B4'
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${moment(this.x).format('DD MMM YYYY')}</b><br/>
                    <b>${this.series.name}:</b> ₹${this.y.toLocaleString('en-IN')}`;
          }
        },
        plotOptions: {
          line: {
            marker: {
              enabled: false
            }
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating portfolio performance chart:', error);
      throw error;
    }
  }

  /**
   * Generate SIP projection chart data
   */
  async generateSIPProjectionChart(userId, period, options) {
    try {
      const { monthlyAmount = 10000, duration = 60, expectedReturn = 12 } = options;
      
      const projectionData = this.calculateSIPProjectionData(monthlyAmount, duration, expectedReturn);
      
      const chartData = {
        type: 'line',
        title: 'SIP Projection',
        xAxis: {
          type: 'datetime',
          title: 'Month'
        },
        yAxis: {
          title: 'Amount (₹)',
          labels: {
            formatter: function() {
              return '₹' + this.value.toLocaleString('en-IN');
            }
          }
        },
        series: [
          {
            name: 'Total Investment',
            data: projectionData.map(point => [point.date, point.invested]),
            color: '#4682B4'
          },
          {
            name: 'Expected Value',
            data: projectionData.map(point => [point.date, point.expectedValue]),
            color: '#2E8B57'
          },
          {
            name: 'Wealth Gained',
            data: projectionData.map(point => [point.date, point.wealthGained]),
            color: '#FF6B6B'
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${moment(this.x).format('MMM YYYY')}</b><br/>
                    <b>${this.series.name}:</b> ₹${this.y.toLocaleString('en-IN')}`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating SIP projection chart:', error);
      throw error;
    }
  }

  /**
   * Generate NAV history chart data
   */
  async generateNAVHistoryChart(fundCode, period, options) {
    try {
      const navHistory = await navHistoryService.getNAVHistory({ fundCode, period });
      
      const chartData = {
        type: 'line',
        title: 'NAV History',
        xAxis: {
          type: 'datetime',
          title: 'Date'
        },
        yAxis: {
          title: 'NAV (₹)',
          labels: {
            formatter: function() {
              return '₹' + this.value.toFixed(2);
            }
          }
        },
        series: [
          {
            name: 'NAV',
            data: navHistory.data.map(point => [point.date, point.nav]),
            color: '#2E8B57'
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${moment(this.x).format('DD MMM YYYY')}</b><br/>
                    <b>NAV:</b> ₹${this.y.toFixed(2)}`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating NAV history chart:', error);
      throw error;
    }
  }

  /**
   * Generate allocation pie chart data
   */
  async generateAllocationPieChart(userId, options) {
    try {
      const portfolioData = await portfolioAnalyticsService.getPortfolioSummary(userId);
      
      const allocationData = portfolioData.holdings.map(holding => ({
        name: holding.fundName,
        y: holding.currentValue,
        color: this.getRandomColor()
      }));

      const chartData = {
        type: 'pie',
        title: 'Portfolio Allocation',
        series: [
          {
            name: 'Allocation',
            data: allocationData,
            size: '60%',
            dataLabels: {
              formatter: function() {
                return this.point.name + '<br/>' + 
                       '₹' + this.y.toLocaleString('en-IN') + '<br/>' +
                       '(' + this.percentage.toFixed(1) + '%)';
              }
            }
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${this.point.name}</b><br/>
                    Amount: ₹${this.y.toLocaleString('en-IN')}<br/>
                    Percentage: ${this.percentage.toFixed(1)}%`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating allocation pie chart:', error);
      throw error;
    }
  }

  /**
   * Generate risk-return scatter chart data
   */
  async generateRiskReturnScatterChart(userId, options) {
    try {
      const portfolioData = await portfolioAnalyticsService.getPortfolioSummary(userId);
      
      const scatterData = portfolioData.holdings.map(holding => ({
        name: holding.fundName,
        x: holding.volatility || 0,
        y: holding.return || 0,
        z: holding.currentValue,
        color: this.getRandomColor()
      }));

      const chartData = {
        type: 'scatter',
        title: 'Risk vs Return Analysis',
        xAxis: {
          title: 'Volatility (%)',
          labels: {
            formatter: function() {
              return this.value.toFixed(1) + '%';
            }
          }
        },
        yAxis: {
          title: 'Return (%)',
          labels: {
            formatter: function() {
              return this.value.toFixed(1) + '%';
            }
          }
        },
        series: [
          {
            name: 'Funds',
            data: scatterData,
            marker: {
              radius: function() {
                return Math.sqrt(this.z) / 1000;
              }
            }
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${this.point.name}</b><br/>
                    Volatility: ${this.x.toFixed(1)}%<br/>
                    Return: ${this.y.toFixed(1)}%<br/>
                    Value: ₹${this.point.z.toLocaleString('en-IN')}`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating risk-return scatter chart:', error);
      throw error;
    }
  }

  /**
   * Generate comparison chart data
   */
  async generateComparisonChart(userId, period, options) {
    try {
      const { benchmark = 'NIFTY50' } = options;
      const portfolioData = await portfolioAnalyticsService.getPortfolioSummary(userId);
      const benchmarkData = await this.getBenchmarkData(benchmark, period);
      
      const chartData = {
        type: 'line',
        title: 'Portfolio vs Benchmark',
        xAxis: {
          type: 'datetime',
          title: 'Date'
        },
        yAxis: {
          title: 'Cumulative Return (%)',
          labels: {
            formatter: function() {
              return this.value.toFixed(1) + '%';
            }
          }
        },
        series: [
          {
            name: 'Portfolio',
            data: portfolioData.performanceData.map(point => [point.date, point.return]),
            color: '#2E8B57'
          },
          {
            name: benchmark,
            data: benchmarkData.map(point => [point.date, point.return]),
            color: '#4682B4'
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${moment(this.x).format('DD MMM YYYY')}</b><br/>
                    <b>${this.series.name}:</b> ${this.y.toFixed(2)}%`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating comparison chart:', error);
      throw error;
    }
  }

  /**
   * Generate volatility chart data
   */
  async generateVolatilityChart(userId, period, options) {
    try {
      const portfolioData = await portfolioAnalyticsService.getPortfolioSummary(userId);
      const volatilityData = await this.calculateVolatilityData(userId, period);
      
      const chartData = {
        type: 'column',
        title: 'Portfolio Volatility',
        xAxis: {
          type: 'datetime',
          title: 'Date'
        },
        yAxis: {
          title: 'Volatility (%)',
          labels: {
            formatter: function() {
              return this.value.toFixed(1) + '%';
            }
          }
        },
        series: [
          {
            name: 'Volatility',
            data: volatilityData.map(point => [point.date, point.volatility]),
            color: '#FF6B6B'
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${moment(this.x).format('DD MMM YYYY')}</b><br/>
                    <b>Volatility:</b> ${this.y.toFixed(2)}%`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating volatility chart:', error);
      throw error;
    }
  }

  /**
   * Generate drawdown chart data
   */
  async generateDrawdownChart(userId, period, options) {
    try {
      const drawdownData = await this.calculateDrawdownData(userId, period);
      
      const chartData = {
        type: 'area',
        title: 'Portfolio Drawdown',
        xAxis: {
          type: 'datetime',
          title: 'Date'
        },
        yAxis: {
          title: 'Drawdown (%)',
          labels: {
            formatter: function() {
              return this.value.toFixed(1) + '%';
            }
          }
        },
        series: [
          {
            name: 'Drawdown',
            data: drawdownData.map(point => [point.date, point.drawdown]),
            color: '#FF6B6B',
            fillColor: {
              linearGradient: { x1: 0, y1: 0, x2: 0, y2: 1 },
              stops: [
                [0, 'rgba(255, 107, 107, 0.3)'],
                [1, 'rgba(255, 107, 107, 0.1)']
              ]
            }
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${moment(this.x).format('DD MMM YYYY')}</b><br/>
                    <b>Drawdown:</b> ${this.y.toFixed(2)}%`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating drawdown chart:', error);
      throw error;
    }
  }

  /**
   * Generate tax analysis chart data
   */
  async generateTaxAnalysisChart(userId, options) {
    try {
      const taxData = await this.getTaxAnalysisData(userId);
      
      const chartData = {
        type: 'column',
        title: 'Tax Analysis',
        xAxis: {
          categories: ['Short Term Gains', 'Long Term Gains', 'Dividend Income', 'Total Tax']
        },
        yAxis: {
          title: 'Amount (₹)',
          labels: {
            formatter: function() {
              return '₹' + this.value.toLocaleString('en-IN');
            }
          }
        },
        series: [
          {
            name: 'Taxable Amount',
            data: [
              taxData.shortTermGains,
              taxData.longTermGains,
              taxData.dividendIncome,
              taxData.totalTax
            ],
            color: '#FF6B6B'
          }
        ],
        tooltip: {
          formatter: function() {
            return `<b>${this.x}</b><br/>
                    <b>Amount:</b> ₹${this.y.toLocaleString('en-IN')}`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating tax analysis chart:', error);
      throw error;
    }
  }

  /**
   * Generate goal progress chart data
   */
  async generateGoalProgressChart(userId, options) {
    try {
      const goalData = await this.getGoalProgressData(userId);
      
      const chartData = {
        type: 'bar',
        title: 'Goal Progress',
        xAxis: {
          categories: goalData.map(goal => goal.name)
        },
        yAxis: {
          title: 'Progress (%)',
          max: 100,
          labels: {
            formatter: function() {
              return this.value + '%';
            }
          }
        },
        series: [
          {
            name: 'Progress',
            data: goalData.map(goal => goal.progress),
            color: '#2E8B57'
          }
        ],
        tooltip: {
          formatter: function() {
            const goal = goalData[this.point.index];
            return `<b>${goal.name}</b><br/>
                    Progress: ${this.y}%<br/>
                    Target: ₹${goal.target.toLocaleString('en-IN')}<br/>
                    Current: ₹${goal.current.toLocaleString('en-IN')}`;
          }
        }
      };

      return chartData;
    } catch (error) {
      logger.error('Error generating goal progress chart:', error);
      throw error;
    }
  }

  /**
   * Generate dashboard charts
   */
  async generateDashboardCharts(userId) {
    try {
      const charts = {
        portfolioPerformance: await this.generatePortfolioPerformanceChart(userId, '1y'),
        allocation: await this.generateAllocationPieChart(userId),
        riskReturn: await this.generateRiskReturnScatterChart(userId),
        comparison: await this.generateComparisonChart(userId, '1y')
      };

      return charts;
    } catch (error) {
      logger.error('Error generating dashboard charts:', error);
      throw error;
    }
  }

  /**
   * Calculate SIP projection data
   */
  calculateSIPProjectionData(monthlyAmount, duration, expectedReturn) {
    const data = [];
    let totalInvested = 0;
    let totalValue = 0;
    
    for (let month = 1; month <= duration; month++) {
      totalInvested += monthlyAmount;
      totalValue = this.calculateFutureValue(monthlyAmount, month, expectedReturn / 12 / 100);
      
      data.push({
        date: moment().add(month, 'months').valueOf(),
        invested: totalInvested,
        expectedValue: totalValue,
        wealthGained: totalValue - totalInvested
      });
    }
    
    return data;
  }

  /**
   * Calculate future value using compound interest formula
   */
  calculateFutureValue(monthlyAmount, months, monthlyRate) {
    if (monthlyRate === 0) {
      return monthlyAmount * months;
    }
    
    return monthlyAmount * ((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate);
  }

  /**
   * Get historical performance data
   */
  async getHistoricalPerformanceData(userId, period) {
    // This would typically fetch from database
    // For now, generating mock data
    const data = [];
    const startDate = moment().subtract(this.parsePeriod(period), 'days');
    
    for (let i = 0; i < 30; i++) {
      const date = moment(startDate).add(i, 'days');
      data.push({
        date: date.valueOf(),
        value: 100000 + (i * 1000) + (Math.random() * 5000),
        invested: 100000 + (i * 500)
      });
    }
    
    return data;
  }

  /**
   * Get benchmark data
   */
  async getBenchmarkData(benchmark, period) {
    // This would typically fetch from market data service
    // For now, generating mock data
    const data = [];
    const startDate = moment().subtract(this.parsePeriod(period), 'days');
    
    for (let i = 0; i < 30; i++) {
      const date = moment(startDate).add(i, 'days');
      data.push({
        date: date.valueOf(),
        return: (i * 0.5) + (Math.random() * 2)
      });
    }
    
    return data;
  }

  /**
   * Calculate volatility data
   */
  async calculateVolatilityData(userId, period) {
    // This would calculate actual volatility from historical data
    // For now, generating mock data
    const data = [];
    const startDate = moment().subtract(this.parsePeriod(period), 'days');
    
    for (let i = 0; i < 30; i++) {
      const date = moment(startDate).add(i, 'days');
      data.push({
        date: date.valueOf(),
        volatility: 15 + (Math.random() * 10)
      });
    }
    
    return data;
  }

  /**
   * Calculate drawdown data
   */
  async calculateDrawdownData(userId, period) {
    // This would calculate actual drawdown from historical data
    // For now, generating mock data
    const data = [];
    const startDate = moment().subtract(this.parsePeriod(period), 'days');
    
    for (let i = 0; i < 30; i++) {
      const date = moment(startDate).add(i, 'days');
      data.push({
        date: date.valueOf(),
        drawdown: -(Math.random() * 10)
      });
    }
    
    return data;
  }

  /**
   * Get tax analysis data
   */
  async getTaxAnalysisData(userId) {
    // This would fetch actual tax data from database
    return {
      shortTermGains: 50000,
      longTermGains: 150000,
      dividendIncome: 25000,
      totalTax: 45000
    };
  }

  /**
   * Get goal progress data
   */
  async getGoalProgressData(userId) {
    // This would fetch actual goal data from database
    return [
      {
        name: 'Emergency Fund',
        target: 500000,
        current: 300000,
        progress: 60
      },
      {
        name: 'House Down Payment',
        target: 2000000,
        current: 800000,
        progress: 40
      },
      {
        name: 'Retirement',
        target: 10000000,
        current: 2000000,
        progress: 20
      }
    ];
  }

  /**
   * Parse period string to days
   */
  parsePeriod(period) {
    const periodMap = {
      '1w': 7,
      '1m': 30,
      '3m': 90,
      '6m': 180,
      '1y': 365,
      '3y': 1095,
      '5y': 1825
    };
    
    return periodMap[period] || 365;
  }

  /**
   * Get random color for charts
   */
  getRandomColor() {
    const colors = [
      '#2E8B57', '#4682B4', '#FF6B6B', '#FFD93D', '#6BCF7F',
      '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
  }
}

module.exports = new ChartDataService(); 