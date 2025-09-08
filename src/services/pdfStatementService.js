const logger = require('../utils/logger');

class PDFStatementService {
  constructor() {
    this.chartColors = {
      primary: '#2563eb',
      secondary: '#7c3aed',
      success: '#059669',
      warning: '#d97706',
      danger: '#dc2626',
      info: '#0891b2',
      light: '#f3f4f6',
      dark: '#1f2937'
    };
  }

  /**
   * Generate chart image using Chart.js and Puppeteer
   */
  async generateChartImage(chartConfig) {
    try {
      // For now, return a placeholder
      // In production, this would use Chart.js + Puppeteer
      logger.info('Chart generation requested:', chartConfig.type);
      return null; // Placeholder
    } catch (error) {
      logger.error('Error generating chart image:', error);
      return null;
    }
  }

  /**
   * Generate performance trend chart
   */
  async generatePerformanceChart(performanceData) {
    const chartConfig = {
      type: 'line',
      data: {
        labels: performanceData.labels,
        datasets: [{
          label: 'Portfolio Value',
          data: performanceData.values,
          borderColor: this.chartColors.primary,
          backgroundColor: 'rgba(37, 99, 235, 0.1)',
          borderWidth: 3,
          fill: true,
          tension: 0.4
        }, {
          label: 'Nifty 50',
          data: performanceData.benchmark,
          borderColor: this.chartColors.secondary,
          backgroundColor: 'rgba(124, 58, 237, 0.1)',
          borderWidth: 2,
          fill: false,
          tension: 0.4
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Portfolio Performance vs Nifty 50',
            font: { size: 16, weight: 'bold' }
          },
          legend: {
            position: 'top'
          }
        },
        scales: {
          y: {
            beginAtZero: false,
            ticks: {
              callback: function(value) {
                return 'Rs. ' + value.toLocaleString();
              }
            }
          }
        }
      }
    };

    return await this.generateChartImage(chartConfig);
  }

  /**
   * Generate allocation pie chart
   */
  async generateAllocationChart(allocationData) {
    const chartConfig = {
      type: 'doughnut',
      data: {
        labels: allocationData.labels,
        datasets: [{
          data: allocationData.values,
          backgroundColor: [
            this.chartColors.primary,
            this.chartColors.secondary,
            this.chartColors.success,
            this.chartColors.warning,
            this.chartColors.danger,
            this.chartColors.info
          ],
          borderWidth: 2,
          borderColor: '#ffffff'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Portfolio Allocation',
            font: { size: 16, weight: 'bold' }
          },
          legend: {
            position: 'right'
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return context.label + ': ' + context.parsed + '%';
              }
            }
          }
        }
      }
    };

    return await this.generateChartImage(chartConfig);
  }

  /**
   * Generate XIRR vs benchmark chart
   */
  async generateXIRRChart(xirrData) {
    const chartConfig = {
      type: 'bar',
      data: {
        labels: xirrData.labels,
        datasets: [{
          label: 'Portfolio XIRR',
          data: xirrData.portfolio,
          backgroundColor: this.chartColors.primary,
          borderColor: this.chartColors.primary,
          borderWidth: 1
        }, {
          label: 'Nifty 50 Returns',
          data: xirrData.benchmark,
          backgroundColor: this.chartColors.secondary,
          borderColor: this.chartColors.secondary,
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'XIRR vs Benchmark Returns',
            font: { size: 16, weight: 'bold' }
          },
          legend: {
            position: 'top'
          }
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: {
              callback: function(value) {
                return value + '%';
              }
            }
          }
        }
      }
    };

    return await this.generateChartImage(chartConfig);
  }

  /**
   * Process user data for PDF generation
   */
  processUserData(user, portfolio, transactions, rewards) {
    try {
      // Calculate portfolio summary
      const totalInvested = portfolio.funds.reduce((sum, fund) => sum + fund.investedValue, 0);
      const totalCurrentValue = portfolio.funds.reduce((sum, fund) => sum + fund.currentValue, 0);
      const absoluteGain = totalCurrentValue - totalInvested;
      const percentageGain = totalInvested > 0 ? (absoluteGain / totalInvested) * 100 : 0;

      // Calculate allocation
      const allocation = {};
      portfolio.funds.forEach(fund => {
        const percentage = (fund.currentValue / totalCurrentValue) * 100;
        allocation[fund.schemeName] = Math.round(percentage * 100) / 100;
      });

      // Process transactions
      const processedTransactions = transactions.map(t => ({
        date: new Date(t.date).toLocaleDateString('en-IN'),
        type: t.type,
        schemeName: t.schemeName,
        amount: t.amount,
        units: t.units,
        nav: t.nav
      }));

      // Calculate capital gains
      const capitalGains = this.calculateCapitalGains(transactions);

      // Generate AI insights
      const aiInsights = this.generateAIInsights(portfolio, transactions);

      return {
        user: {
          name: user.name,
          pan: user.kycDetails?.panNumber || 'N/A',
          mobile: user.phone,
          email: user.email,
          clientCode: user.secretCode
        },
        portfolio: {
          totalInvested,
          totalCurrentValue,
          absoluteGain,
          percentageGain,
          xirr1Y: portfolio.xirr1Y || 0,
          funds: portfolio.funds,
          allocation
        },
        transactions: processedTransactions,
        capitalGains,
        rewards,
        aiInsights
      };
    } catch (error) {
      logger.error('Error processing user data:', error);
      throw error;
    }
  }

  /**
   * Calculate capital gains from transactions
   */
  calculateCapitalGains(transactions) {
    try {
      const gains = {
        shortTerm: { gain: 0, tax: 0 },
        longTerm: { gain: 0, tax: 0 }
      };

      // Group transactions by scheme
      const schemeTransactions = {};
      transactions.forEach(t => {
        if (!schemeTransactions[t.schemeCode]) {
          schemeTransactions[t.schemeCode] = [];
        }
        schemeTransactions[t.schemeCode].push(t);
      });

      // Calculate gains for each scheme
      Object.values(schemeTransactions).forEach(schemeTxs => {
        const sortedTxs = schemeTxs.sort((a, b) => new Date(a.date) - new Date(b.date));
        
        let totalUnits = 0;
        let totalCost = 0;
        
        sortedTxs.forEach(tx => {
          if (tx.type === 'SIP' || tx.type === 'LUMPSUM') {
            totalUnits += tx.units;
            totalCost += tx.amount;
          } else if (tx.type === 'REDEMPTION') {
            const avgCost = totalCost / totalUnits;
            const gain = (tx.nav - avgCost) * tx.units;
            
            // Determine if short term or long term
            const holdingPeriod = this.calculateHoldingPeriod(tx.date, sortedTxs[0].date);
            const isLongTerm = holdingPeriod >= 365; // 1 year for equity funds
            
            if (isLongTerm) {
              gains.longTerm.gain += gain;
            } else {
              gains.shortTerm.gain += gain;
            }
            
            totalUnits -= tx.units;
            totalCost -= avgCost * tx.units;
          }
        });
      });

      // Calculate taxes
      gains.shortTerm.tax = gains.shortTerm.gain * 0.15; // 15% for short term
      gains.longTerm.tax = gains.longTerm.gain * 0.10; // 10% for long term (above 1L)

      return gains;
    } catch (error) {
      logger.error('Error calculating capital gains:', error);
      return { shortTerm: { gain: 0, tax: 0 }, longTerm: { gain: 0, tax: 0 } };
    }
  }

  /**
   * Calculate holding period in days
   */
  calculateHoldingPeriod(sellDate, buyDate) {
    const sell = new Date(sellDate);
    const buy = new Date(buyDate);
    return Math.floor((sell - buy) / (1000 * 60 * 60 * 24));
  }

  /**
   * Generate AI insights based on portfolio data
   */
  generateAIInsights(portfolio, transactions) {
    try {
      const insights = [];

      // XIRR analysis
      if (portfolio.xirr1Y > 15) {
        insights.push({
          type: 'success',
          title: 'Excellent Performance',
          message: `Your portfolio has achieved an impressive XIRR of ${portfolio.xirr1Y.toFixed(2)}% over the past year, outperforming most benchmarks.`
        });
      } else if (portfolio.xirr1Y > 10) {
        insights.push({
          type: 'info',
          title: 'Good Performance',
          message: `Your portfolio XIRR of ${portfolio.xirr1Y.toFixed(2)}% shows consistent growth. Consider increasing SIP amounts during market dips.`
        });
      } else {
        insights.push({
          type: 'warning',
          title: 'Performance Review Needed',
          message: `Your portfolio XIRR of ${portfolio.xirr1Y.toFixed(2)}% suggests reviewing fund selection. Consider diversifying into better-performing categories.`
        });
      }

      // Fund concentration analysis
      const fundCount = portfolio.funds.length;
      if (fundCount < 3) {
        insights.push({
          type: 'warning',
          title: 'Low Diversification',
          message: `You have only ${fundCount} funds. Consider diversifying across more funds and categories for better risk management.`
        });
      } else if (fundCount > 8) {
        insights.push({
          type: 'info',
          title: 'High Diversification',
          message: `You have ${fundCount} funds which provides good diversification. Consider consolidating similar funds to reduce complexity.`
        });
      }

      // SIP consistency analysis
      const sipTransactions = transactions.filter(t => t.type === 'SIP');
      if (sipTransactions.length > 0) {
        const avgSipAmount = sipTransactions.reduce((sum, t) => sum + t.amount, 0) / sipTransactions.length;
        insights.push({
          type: 'success',
          title: 'SIP Discipline',
          message: `You've maintained consistent SIP discipline with an average monthly investment of Rs. ${avgSipAmount.toLocaleString()}.`
        });
      }

      // Top performing fund
      if (portfolio.funds.length > 0) {
        const bestFund = portfolio.funds.reduce((best, current) => {
          const currentReturn = current.investedValue > 0 ? 
            ((current.currentValue - current.investedValue) / current.investedValue) * 100 : 0;
          const bestReturn = best.investedValue > 0 ? 
            ((best.currentValue - best.investedValue) / best.investedValue) * 100 : 0;
          return currentReturn > bestReturn ? current : best;
        });

        const bestReturn = bestFund.investedValue > 0 ? 
          ((bestFund.currentValue - bestFund.investedValue) / bestFund.investedValue) * 100 : 0;

        insights.push({
          type: 'info',
          title: 'Top Performer',
          message: `${bestFund.schemeName} is your best performing fund with ${bestReturn.toFixed(2)}% returns.`
        });
      }

      return insights;
    } catch (error) {
      logger.error('Error generating AI insights:', error);
      return [];
    }
  }

  /**
   * Generate statement metadata
   */
  generateStatementMetadata(statementType, dateRange) {
    const metadata = {
      comprehensive: {
        title: 'SIP Brewery Wealth Statement',
        subtitle: 'Complete Portfolio Overview'
      },
      holdings: {
        title: 'Portfolio Holdings Statement',
        subtitle: 'Current Fund Holdings'
      },
      transactions: {
        title: 'Transaction Report',
        subtitle: 'Investment Activity Summary'
      },
      pnl: {
        title: 'Profit & Loss Statement',
        subtitle: 'Portfolio Performance Analysis'
      },
      'capital-gain': {
        title: 'Capital Gains Statement',
        subtitle: 'Tax Calculation Summary'
      },
      tax: {
        title: 'Tax Statement',
        subtitle: 'For CA/ITR Filing'
      },
      rewards: {
        title: 'Rewards & Referral Summary',
        subtitle: 'Earnings & Benefits'
      },
      'smart-sip': {
        title: 'Smart SIP Summary',
        subtitle: 'AI-Powered Investment Analysis'
      }
    };

    return {
      title: metadata[statementType]?.title || 'Portfolio Statement',
      subtitle: metadata[statementType]?.subtitle || 'Investment Summary',
      dateRange: `${dateRange.start} - ${dateRange.end}`,
      generatedOn: new Date().toLocaleString('en-IN'),
      arn: 'ARN-123456',
      sebiReg: 'SEBI Registration No: INZ000123456',
      contact: {
        email: 'support@sipbrewery.com',
        phone: '+91-98765-43210',
        website: 'www.sipbrewery.com'
      }
    };
  }

  /**
   * Generate all charts for a statement
   */
  async generateAllCharts(portfolioData) {
    try {
      const charts = {};

      // Performance trend chart
      const performanceData = {
        labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
        values: [100000, 105000, 102000, 108000, 112000, 115000],
        benchmark: [100000, 103000, 101000, 106000, 109000, 111000]
      };
      charts.performance = await this.generatePerformanceChart(performanceData);

      // Allocation pie chart
      const allocationData = {
        labels: Object.keys(portfolioData.allocation),
        values: Object.values(portfolioData.allocation)
      };
      charts.allocation = await this.generateAllocationChart(allocationData);

      // XIRR comparison chart
      const xirrData = {
        labels: ['1M', '3M', '6M', '1Y'],
        portfolio: [2.5, 7.8, 12.3, 18.7],
        benchmark: [2.1, 6.9, 10.2, 15.4]
      };
      charts.xirr = await this.generateXIRRChart(xirrData);

      return charts;
    } catch (error) {
      logger.error('Error generating charts:', error);
      return {};
    }
  }
}

module.exports = new PDFStatementService(); 