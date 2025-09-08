const moment = require('moment');
const logger = require('../utils/logger');

class TaxCalculationService {
  constructor() {
    this.taxRates = {
      // Capital Gains Tax
      shortTerm: 15, // Short term capital gains (STCG)
      longTerm: 10,  // Long term capital gains (LTCG) above 1L
      longTermExemption: 100000, // LTCG exemption limit
      
      // Dividend Distribution Tax
      dividend: 10,
      
      // Income Tax Slabs (FY 2024-25)
      incomeTaxSlabs: [
        { min: 0, max: 300000, rate: 0 },
        { min: 300000, max: 600000, rate: 5 },
        { min: 600000, max: 900000, rate: 10 },
        { min: 900000, max: 1200000, rate: 15 },
        { min: 1200000, max: 1500000, rate: 20 },
        { min: 1500000, max: Infinity, rate: 30 }
      ],
      
      // Surcharge
      surcharge: {
        threshold: 5000000,
        rate: 10
      },
      
      // Cess
      cess: 4
    };

    this.financialYear = '2024-25';
  }

  /**
   * Get comprehensive tax calculations for a user
   */
  async getTaxCalculations({ userId, financialYear, includeOptimization = true }) {
    try {
      logger.info(`Getting tax calculations for user: ${userId}, FY: ${financialYear}`);

      const taxCalculations = {
        summary: {},
        capitalGains: {},
        dividendIncome: {},
        totalTax: {},
        optimization: {},
        recommendations: {}
      };

      // Get user portfolio data
      const portfolioData = await this.getUserPortfolioData(userId);
      
      // Calculate capital gains
      taxCalculations.capitalGains = this.calculateCapitalGains(portfolioData);

      // Calculate dividend income
      taxCalculations.dividendIncome = this.calculateDividendIncome(portfolioData);

      // Calculate total tax liability
      taxCalculations.totalTax = this.calculateTotalTaxLiability(
        taxCalculations.capitalGains,
        taxCalculations.dividendIncome
      );

      // Generate tax summary
      taxCalculations.summary = this.generateTaxSummary(taxCalculations);

      // Generate optimization suggestions
      if (includeOptimization) {
        taxCalculations.optimization = this.generateTaxOptimization(taxCalculations);
      }

      // Generate recommendations
      taxCalculations.recommendations = this.generateTaxRecommendations(taxCalculations);

      return taxCalculations;
    } catch (error) {
      logger.error('Error getting tax calculations:', error);
      throw error;
    }
  }

  /**
   * Calculate capital gains tax
   */
  calculateCapitalGains(portfolioData) {
    const capitalGains = {
      shortTerm: {
        gains: 0,
        tax: 0,
        transactions: []
      },
      longTerm: {
        gains: 0,
        tax: 0,
        transactions: []
      },
      total: {
        gains: 0,
        tax: 0
      }
    };

    // Process each holding for capital gains
    portfolioData.holdings.forEach(holding => {
      const gains = this.calculateHoldingGains(holding);
      
      if (gains.holdingPeriod <= 12) {
        // Short term capital gains
        capitalGains.shortTerm.gains += gains.gainAmount;
        capitalGains.shortTerm.tax += gains.taxAmount;
        capitalGains.shortTerm.transactions.push({
          fundName: holding.fundName,
          fundCode: holding.fundCode,
          gainAmount: gains.gainAmount,
          taxAmount: gains.taxAmount,
          holdingPeriod: gains.holdingPeriod
        });
      } else {
        // Long term capital gains
        capitalGains.longTerm.gains += gains.gainAmount;
        capitalGains.longTerm.tax += gains.taxAmount;
        capitalGains.longTerm.transactions.push({
          fundName: holding.fundName,
          fundCode: holding.fundCode,
          gainAmount: gains.gainAmount,
          taxAmount: gains.taxAmount,
          holdingPeriod: gains.holdingPeriod
        });
      }
    });

    // Calculate total
    capitalGains.total.gains = capitalGains.shortTerm.gains + capitalGains.longTerm.gains;
    capitalGains.total.tax = capitalGains.shortTerm.tax + capitalGains.longTerm.tax;

    return capitalGains;
  }

  /**
   * Calculate gains for a single holding
   */
  calculateHoldingGains(holding) {
    const currentValue = holding.currentValue || 0;
    const totalInvestment = holding.totalInvestment || 0;
    const gainAmount = currentValue - totalInvestment;
    
    // Calculate holding period (in months)
    const purchaseDate = new Date(holding.firstPurchaseDate || new Date());
    const currentDate = new Date();
    const holdingPeriod = moment(currentDate).diff(moment(purchaseDate), 'months');

    let taxAmount = 0;

    if (gainAmount > 0) {
      if (holdingPeriod <= 12) {
        // Short term capital gains
        taxAmount = gainAmount * (this.taxRates.shortTerm / 100);
      } else {
        // Long term capital gains
        const taxableAmount = Math.max(0, gainAmount - this.taxRates.longTermExemption);
        taxAmount = taxableAmount * (this.taxRates.longTerm / 100);
      }
    }

    return {
      gainAmount,
      taxAmount,
      holdingPeriod,
      gainPercentage: totalInvestment > 0 ? (gainAmount / totalInvestment) * 100 : 0
    };
  }

  /**
   * Calculate dividend income tax
   */
  calculateDividendIncome(portfolioData) {
    const dividendIncome = {
      totalDividend: 0,
      taxableDividend: 0,
      tax: 0,
      breakdown: []
    };

    // Calculate dividend from holdings
    portfolioData.holdings.forEach(holding => {
      const dividend = holding.dividendIncome || 0;
      dividendIncome.totalDividend += dividend;
      
      if (dividend > 0) {
        dividendIncome.breakdown.push({
          fundName: holding.fundName,
          fundCode: holding.fundCode,
          dividend: dividend
        });
      }
    });

    // Calculate tax on dividend income
    dividendIncome.taxableDividend = dividendIncome.totalDividend;
    dividendIncome.tax = dividendIncome.taxableDividend * (this.taxRates.dividend / 100);

    return dividendIncome;
  }

  /**
   * Calculate total tax liability
   */
  calculateTotalTaxLiability(capitalGains, dividendIncome) {
    const totalTax = {
      capitalGainsTax: capitalGains.total.tax,
      dividendTax: dividendIncome.tax,
      totalTax: 0,
      effectiveRate: 0
    };

    totalTax.totalTax = totalTax.capitalGainsTax + totalTax.dividendTax;

    const totalIncome = capitalGains.total.gains + dividendIncome.totalDividend;
    totalTax.effectiveRate = totalIncome > 0 ? (totalTax.totalTax / totalIncome) * 100 : 0;

    return totalTax;
  }

  /**
   * Generate tax summary
   */
  generateTaxSummary(taxCalculations) {
    const summary = {
      financialYear: this.financialYear,
      totalIncome: 0,
      totalTax: 0,
      breakdown: {},
      keyMetrics: {}
    };

    summary.totalIncome = taxCalculations.capitalGains.total.gains + taxCalculations.dividendIncome.totalDividend;
    summary.totalTax = taxCalculations.totalTax.totalTax;

    summary.breakdown = {
      capitalGains: {
        shortTerm: taxCalculations.capitalGains.shortTerm.gains,
        longTerm: taxCalculations.capitalGains.longTerm.gains,
        total: taxCalculations.capitalGains.total.gains
      },
      dividendIncome: taxCalculations.dividendIncome.totalDividend,
      taxLiability: {
        capitalGains: taxCalculations.capitalGains.total.tax,
        dividend: taxCalculations.dividendIncome.tax,
        total: summary.totalTax
      }
    };

    summary.keyMetrics = {
      effectiveTaxRate: summary.totalIncome > 0 ? (summary.totalTax / summary.totalIncome) * 100 : 0,
      shortTermGainsPercentage: summary.totalIncome > 0 ? 
        (taxCalculations.capitalGains.shortTerm.gains / summary.totalIncome) * 100 : 0,
      longTermGainsPercentage: summary.totalIncome > 0 ? 
        (taxCalculations.capitalGains.longTerm.gains / summary.totalIncome) * 100 : 0,
      dividendPercentage: summary.totalIncome > 0 ? 
        (taxCalculations.dividendIncome.totalDividend / summary.totalIncome) * 100 : 0
    };

    return summary;
  }

  /**
   * Generate tax optimization suggestions
   */
  generateTaxOptimization(taxCalculations) {
    const optimization = {
      suggestions: [],
      potentialSavings: 0,
      strategies: {}
    };

    const suggestions = [];

    // Check for short-term gains optimization
    if (taxCalculations.capitalGains.shortTerm.gains > 0) {
      const stcgTax = taxCalculations.capitalGains.shortTerm.tax;
      const potentialSavings = stcgTax * 0.33; // Assuming 33% can be optimized

      suggestions.push({
        type: 'short_term_gains',
        title: 'Optimize Short-term Capital Gains',
        description: 'Consider holding investments for more than 12 months to qualify for lower LTCG rates',
        potentialSavings,
        action: 'Review holdings with less than 12 months holding period',
        priority: 'high'
      });

      optimization.potentialSavings += potentialSavings;
    }

    // Check for dividend optimization
    if (taxCalculations.dividendIncome.totalDividend > 0) {
      const dividendTax = taxCalculations.dividendIncome.tax;
      const potentialSavings = dividendTax * 0.2; // Assuming 20% can be optimized

      suggestions.push({
        type: 'dividend_optimization',
        title: 'Optimize Dividend Income',
        description: 'Consider growth funds instead of dividend-paying funds to defer tax',
        potentialSavings,
        action: 'Review dividend-paying funds and consider growth alternatives',
        priority: 'medium'
      });

      optimization.potentialSavings += potentialSavings;
    }

    // Check for tax-loss harvesting opportunities
    const lossHarvesting = this.identifyTaxLossHarvesting(taxCalculations);
    if (lossHarvesting.potentialSavings > 0) {
      suggestions.push({
        type: 'tax_loss_harvesting',
        title: 'Tax Loss Harvesting',
        description: 'Consider selling loss-making investments to offset gains',
        potentialSavings: lossHarvesting.potentialSavings,
        action: 'Identify and sell loss-making positions',
        priority: 'high'
      });

      optimization.potentialSavings += lossHarvesting.potentialSavings;
    }

    // Check for ELSS optimization
    const elssOptimization = this.calculateELSSOptimization(taxCalculations);
    if (elssOptimization.potentialSavings > 0) {
      suggestions.push({
        type: 'elss_investment',
        title: 'ELSS Investment for Tax Saving',
        description: 'Invest in ELSS funds to claim tax deduction under Section 80C',
        potentialSavings: elssOptimization.potentialSavings,
        action: 'Consider investing in ELSS funds up to ₹1.5L',
        priority: 'medium'
      });

      optimization.potentialSavings += elssOptimization.potentialSavings;
    }

    optimization.suggestions = suggestions;

    // Generate optimization strategies
    optimization.strategies = this.generateOptimizationStrategies(taxCalculations);

    return optimization;
  }

  /**
   * Identify tax loss harvesting opportunities
   */
  identifyTaxLossHarvesting(taxCalculations) {
    const lossHarvesting = {
      potentialSavings: 0,
      opportunities: []
    };

    // This would typically analyze actual portfolio data
    // For now, providing mock analysis
    const mockLosses = [
      { fundName: 'Fund A', loss: 50000 },
      { fundName: 'Fund B', loss: 30000 }
    ];

    mockLosses.forEach(opportunity => {
      const taxSavings = opportunity.loss * (this.taxRates.shortTerm / 100);
      lossHarvesting.potentialSavings += taxSavings;
      lossHarvesting.opportunities.push({
        ...opportunity,
        taxSavings
      });
    });

    return lossHarvesting;
  }

  /**
   * Calculate ELSS optimization potential
   */
  calculateELSSOptimization(taxCalculations) {
    const elssOptimization = {
      potentialSavings: 0,
      recommendedInvestment: 0
    };

    // Assuming user can invest up to ₹1.5L in ELSS
    const maxELSSInvestment = 150000;
    const currentELSSInvestment = 0; // This would come from portfolio data
    const availableELSSInvestment = maxELSSInvestment - currentELSSInvestment;

    if (availableELSSInvestment > 0) {
      // Calculate tax savings based on highest applicable tax rate
      const highestTaxRate = this.getHighestApplicableTaxRate(taxCalculations);
      elssOptimization.potentialSavings = availableELSSInvestment * (highestTaxRate / 100);
      elssOptimization.recommendedInvestment = availableELSSInvestment;
    }

    return elssOptimization;
  }

  /**
   * Get highest applicable tax rate
   */
  getHighestApplicableTaxRate(taxCalculations) {
    // This would typically calculate based on user's income
    // For now, returning a conservative estimate
    return 30; // 30% tax rate
  }

  /**
   * Generate optimization strategies
   */
  generateOptimizationStrategies(taxCalculations) {
    const strategies = {
      immediate: [],
      shortTerm: [],
      longTerm: []
    };

    // Immediate strategies
    if (taxCalculations.capitalGains.shortTerm.gains > 0) {
      strategies.immediate.push({
        title: 'Defer Short-term Sales',
        description: 'Hold investments for 12+ months to qualify for LTCG rates',
        impact: 'High',
        timeline: 'Immediate'
      });
    }

    // Short-term strategies
    strategies.shortTerm.push({
      title: 'Tax Loss Harvesting',
      description: 'Sell loss-making positions to offset gains',
      impact: 'Medium',
      timeline: '3-6 months'
    });

    // Long-term strategies
    strategies.longTerm.push({
      title: 'Asset Location Optimization',
      description: 'Allocate tax-efficient investments in taxable accounts',
      impact: 'High',
      timeline: '6-12 months'
    });

    return strategies;
  }

  /**
   * Generate tax recommendations
   */
  generateTaxRecommendations(taxCalculations) {
    const recommendations = {
      immediate: [],
      planning: [],
      compliance: []
    };

    // Immediate recommendations
    if (taxCalculations.totalTax.totalTax > 100000) {
      recommendations.immediate.push({
        type: 'warning',
        message: 'High tax liability detected',
        action: 'Consider tax optimization strategies immediately',
        priority: 'high'
      });
    }

    if (taxCalculations.capitalGains.shortTerm.gains > taxCalculations.capitalGains.longTerm.gains) {
      recommendations.immediate.push({
        type: 'info',
        message: 'High proportion of short-term gains',
        action: 'Consider longer holding periods for better tax efficiency',
        priority: 'medium'
      });
    }

    // Planning recommendations
    recommendations.planning.push({
      type: 'info',
      message: 'Plan for tax-efficient investing',
      action: 'Consider ELSS funds for tax deduction',
      priority: 'medium'
    });

    recommendations.planning.push({
      type: 'info',
      message: 'Regular portfolio review for tax optimization',
      action: 'Review portfolio quarterly for tax efficiency',
      priority: 'low'
    });

    // Compliance recommendations
    recommendations.compliance.push({
      type: 'warning',
      message: 'Ensure proper tax filing',
      action: 'File ITR with accurate capital gains details',
      priority: 'high'
    });

    recommendations.compliance.push({
      type: 'info',
      message: 'Maintain proper documentation',
      action: 'Keep records of all transactions for tax purposes',
      priority: 'medium'
    });

    return recommendations;
  }

  /**
   * Calculate tax for specific transaction
   */
  calculateTransactionTax(transaction) {
    const { type, amount, holdingPeriod, purchasePrice, salePrice } = transaction;

    let taxAmount = 0;

    if (type === 'CAPITAL_GAIN') {
      const gainAmount = salePrice - purchasePrice;
      
      if (gainAmount > 0) {
        if (holdingPeriod <= 12) {
          // Short term capital gains
          taxAmount = gainAmount * (this.taxRates.shortTerm / 100);
        } else {
          // Long term capital gains
          const taxableAmount = Math.max(0, gainAmount - this.taxRates.longTermExemption);
          taxAmount = taxableAmount * (this.taxRates.longTerm / 100);
        }
      }
    } else if (type === 'DIVIDEND') {
      taxAmount = amount * (this.taxRates.dividend / 100);
    }

    return {
      transactionType: type,
      amount,
      taxAmount,
      effectiveRate: amount > 0 ? (taxAmount / amount) * 100 : 0
    };
  }

  /**
   * Calculate tax liability for different scenarios
   */
  calculateTaxScenarios(portfolioData, scenarios) {
    const results = {};

    scenarios.forEach(scenario => {
      const modifiedPortfolio = this.applyScenarioToPortfolio(portfolioData, scenario);
      const taxCalculation = this.calculateCapitalGains(modifiedPortfolio);
      
      results[scenario.name] = {
        totalTax: taxCalculation.total.tax,
        savings: 0, // Will be calculated against baseline
        recommendations: this.generateScenarioRecommendations(scenario, taxCalculation)
      };
    });

    // Calculate savings against baseline
    const baselineTax = this.calculateCapitalGains(portfolioData).total.tax;
    Object.keys(results).forEach(scenarioName => {
      results[scenarioName].savings = baselineTax - results[scenarioName].totalTax;
    });

    return results;
  }

  /**
   * Apply scenario to portfolio
   */
  applyScenarioToPortfolio(portfolioData, scenario) {
    // This would modify portfolio data based on scenario
    // For now, returning original data
    return portfolioData;
  }

  /**
   * Generate scenario recommendations
   */
  generateScenarioRecommendations(scenario, taxCalculation) {
    const recommendations = [];

    if (taxCalculation.total.tax < scenario.targetTax) {
      recommendations.push({
        type: 'success',
        message: `Scenario achieves target tax reduction`,
        action: 'Consider implementing this strategy'
      });
    }

    return recommendations;
  }

  /**
   * Get user portfolio data (mock implementation)
   */
  async getUserPortfolioData(userId) {
    // This would typically fetch from database
    return {
      holdings: [
        {
          fundName: 'HDFC Mid-Cap Opportunities',
          fundCode: 'HDFCMIDCAP',
          currentValue: 150000,
          totalInvestment: 100000,
          firstPurchaseDate: '2023-01-15',
          dividendIncome: 5000
        },
        {
          fundName: 'ICICI Prudential Bluechip',
          fundCode: 'ICICIBLUECHIP',
          currentValue: 80000,
          totalInvestment: 120000,
          firstPurchaseDate: '2022-06-10',
          dividendIncome: 3000
        }
      ],
      transactions: [
        {
          type: 'PURCHASE',
          fundCode: 'HDFCMIDCAP',
          amount: 100000,
          date: '2023-01-15'
        },
        {
          type: 'PURCHASE',
          fundCode: 'ICICIBLUECHIP',
          amount: 120000,
          date: '2022-06-10'
        }
      ]
    };
  }
}

module.exports = new TaxCalculationService(); 