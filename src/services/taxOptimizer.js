const logger = require('../utils/logger');
const { User, UserPortfolio, Transaction, TaxOptimization, LTCGRecord, ELSSInvestment, TaxHarvesting } = require('../models');
const ollamaService = require('./ollamaService');

class TaxOptimizer {
  constructor() {
    this.taxRates = {
      LTCG_EQUITY: {
        rate: 10,
        threshold: 100000, // ₹1 lakh exemption
        holdingPeriod: 365, // 1 year
        description: 'Long-term capital gains on equity funds'
      },
      STCG_EQUITY: {
        rate: 15,
        threshold: 0,
        holdingPeriod: 365,
        description: 'Short-term capital gains on equity funds'
      },
      LTCG_DEBT: {
        rate: 20,
        threshold: 0,
        holdingPeriod: 1095, // 3 years
        description: 'Long-term capital gains on debt funds'
      },
      STCG_DEBT: {
        rate: 'slab_rate',
        threshold: 0,
        holdingPeriod: 1095,
        description: 'Short-term capital gains on debt funds'
      },
      DIVIDEND: {
        rate: 10,
        threshold: 5000, // ₹5000 exemption
        description: 'Dividend distribution tax'
      }
    };

    this.taxSavingOptions = {
      ELSS: {
        name: 'ELSS (Equity Linked Savings Scheme)',
        section: '80C',
        limit: 150000,
        lockInPeriod: 1095, // 3 years
        description: 'Tax-saving equity mutual funds',
        expectedReturn: 12.0,
        risk: 'moderate'
      },
      NPS: {
        name: 'NPS (National Pension System)',
        section: '80CCD(1B)',
        limit: 50000,
        lockInPeriod: 0, // Till retirement
        description: 'Pension scheme with tax benefits',
        expectedReturn: 8.0,
        risk: 'low'
      },
      PPF: {
        name: 'PPF (Public Provident Fund)',
        section: '80C',
        limit: 150000,
        lockInPeriod: 1825, // 5 years
        description: 'Government-backed savings scheme',
        expectedReturn: 7.1,
        risk: 'very_low'
      },
      SUKANYA_SAMRIDDHI: {
        name: 'Sukanya Samriddhi Yojana',
        section: '80C',
        limit: 150000,
        lockInPeriod: 3650, // Till girl child turns 21
        description: 'Girl child savings scheme',
        expectedReturn: 8.0,
        risk: 'very_low'
      },
      HOME_LOAN: {
        name: 'Home Loan Principal',
        section: '80C',
        limit: 150000,
        lockInPeriod: 0,
        description: 'Home loan principal repayment',
        expectedReturn: 0, // No return, just tax saving
        risk: 'none'
      },
      LIFE_INSURANCE: {
        name: 'Life Insurance Premium',
        section: '80C',
        limit: 150000,
        lockInPeriod: 0,
        description: 'Life insurance premium payment',
        expectedReturn: 0, // No return, just tax saving
        risk: 'none'
      },
      HEALTH_INSURANCE: {
        name: 'Health Insurance Premium',
        section: '80D',
        limit: 25000,
        lockInPeriod: 0,
        description: 'Health insurance premium payment',
        expectedReturn: 0, // No return, just tax saving
        risk: 'none'
      }
    };

    this.harvestingStrategies = {
      LOSS_HARVESTING: {
        name: 'Loss Harvesting',
        description: 'Sell investments at loss to offset gains',
        conditions: ['realized_losses', 'taxable_gains'],
        timing: 'year_end',
        frequency: 'annual'
      },
      GAIN_HARVESTING: {
        name: 'Gain Harvesting',
        description: 'Realize gains within exemption limit',
        conditions: ['ltcg_below_threshold', 'taxable_income'],
        timing: 'throughout_year',
        frequency: 'opportunistic'
      },
      REBALANCING_HARVESTING: {
        name: 'Rebalancing Harvesting',
        description: 'Use rebalancing to optimize tax position',
        conditions: ['portfolio_rebalancing', 'tax_optimization'],
        timing: 'quarterly',
        frequency: 'regular'
      }
    };
  }

  /**
   * Track LTCG and calculate tax liability
   */
  async trackLTCG(userId) {
    try {
      logger.info('Tracking LTCG', { userId });

      const user = await User.findById(userId);
      const transactions = await Transaction.find({ userId }).sort({ date: 1 });
      const userPortfolio = await UserPortfolio.findOne({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const ltcgAnalysis = await this.calculateLTCG(transactions, userPortfolio);
      const taxLiability = this.calculateTaxLiability(ltcgAnalysis, user);

      // Store LTCG analysis
      await this.storeLTCGAnalysis(userId, ltcgAnalysis, taxLiability);

      return {
        success: true,
        data: {
          ltcgAnalysis,
          taxLiability,
          recommendations: await this.getLTCGRecommendations(userId, ltcgAnalysis, taxLiability),
          harvestingOpportunities: await this.identifyHarvestingOpportunities(userId, ltcgAnalysis)
        }
      };
    } catch (error) {
      logger.error('Failed to track LTCG', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to track LTCG',
        error: error.message
      };
    }
  }

  /**
   * Get ELSS investment recommendations
   */
  async getELSSRecommendations(userId) {
    try {
      logger.info('Getting ELSS recommendations', { userId });

      const user = await User.findById(userId);
      const currentInvestments = await this.getCurrentTaxSavingInvestments(userId);
      const availableLimit = this.calculateAvailableLimit(currentInvestments);

      if (availableLimit <= 0) {
        return {
          success: true,
          data: {
            message: 'You have already utilized your Section 80C limit',
            currentInvestments,
            availableLimit: 0
          }
        };
      }

      const elssRecommendations = await this.generateELSSRecommendations(user, availableLimit);

      return {
        success: true,
        data: {
          availableLimit,
          currentInvestments,
          elssRecommendations,
          investmentStrategy: await this.getELSSInvestmentStrategy(userId, availableLimit),
          deadline: this.getTaxDeadline()
        }
      };
    } catch (error) {
      logger.error('Failed to get ELSS recommendations', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get ELSS recommendations',
        error: error.message
      };
    }
  }

  /**
   * Identify tax harvesting opportunities
   */
  async identifyHarvestingOpportunities(userId) {
    try {
      logger.info('Identifying harvesting opportunities', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });
      const transactions = await Transaction.find({ userId }).sort({ date: 1 });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const harvestingOpportunities = await this.analyzeHarvestingOpportunities(user, userPortfolio, transactions);
      const harvestingPlan = await this.createHarvestingPlan(userId, harvestingOpportunities);

      return {
        success: true,
        data: {
          harvestingOpportunities,
          harvestingPlan,
          recommendations: await this.getHarvestingRecommendations(userId, harvestingOpportunities),
          timeline: this.getHarvestingTimeline(harvestingOpportunities)
        }
      };
    } catch (error) {
      logger.error('Failed to identify harvesting opportunities', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to identify harvesting opportunities',
        error: error.message
      };
    }
  }

  /**
   * Calculate XIRR after tax
   */
  async calculateXIRRAfterTax(userId, portfolioId) {
    try {
      logger.info('Calculating XIRR after tax', { userId, portfolioId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findById(portfolioId);
      const transactions = await Transaction.find({ userId, portfolioId }).sort({ date: 1 });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const preTaxXIRR = this.calculatePreTaxXIRR(transactions);
      const taxImpact = await this.calculateTaxImpact(userId, transactions, preTaxXIRR);
      const postTaxXIRR = this.calculatePostTaxXIRR(preTaxXIRR, taxImpact);

      return {
        success: true,
        data: {
          preTaxXIRR,
          taxImpact,
          postTaxXIRR,
          breakdown: await this.getTaxBreakdown(userId, transactions),
          optimization: await this.getXIRROptimization(userId, postTaxXIRR)
        }
      };
    } catch (error) {
      logger.error('Failed to calculate XIRR after tax', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to calculate XIRR after tax',
        error: error.message
      };
    }
  }

  /**
   * Optimize SIP for tax efficiency
   */
  async optimizeSIPForTax(userId, sipAmount) {
    try {
      logger.info('Optimizing SIP for tax efficiency', { userId, sipAmount });

      const user = await User.findById(userId);
      const currentInvestments = await this.getCurrentTaxSavingInvestments(userId);
      const availableLimit = this.calculateAvailableLimit(currentInvestments);

      const optimization = await this.generateSIPTaxOptimization(user, sipAmount, availableLimit);

      return {
        success: true,
        data: {
          optimization,
          recommendations: await this.getSIPTaxRecommendations(userId, optimization),
          implementation: await this.getSIPTaxImplementation(userId, optimization)
        }
      };
    } catch (error) {
      logger.error('Failed to optimize SIP for tax', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to optimize SIP for tax',
        error: error.message
      };
    }
  }

  /**
   * Get comprehensive tax optimization plan
   */
  async getTaxOptimizationPlan(userId) {
    try {
      logger.info('Getting tax optimization plan', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });
      const currentInvestments = await this.getCurrentTaxSavingInvestments(userId);

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const taxPlan = await this.generateComprehensiveTaxPlan(user, userPortfolio, currentInvestments);

      return {
        success: true,
        data: {
          taxPlan,
          currentStatus: await this.getCurrentTaxStatus(userId),
          recommendations: await this.getTaxOptimizationRecommendations(userId, taxPlan),
          timeline: this.getTaxOptimizationTimeline(taxPlan)
        }
      };
    } catch (error) {
      logger.error('Failed to get tax optimization plan', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get tax optimization plan',
        error: error.message
      };
    }
  }

  // Helper methods
  async calculateLTCG(transactions, userPortfolio) {
    const holdings = userPortfolio.holdings || [];
    const ltcgRecords = [];

    for (const holding of holdings) {
      const fundTransactions = transactions.filter(t => 
        t.fundId === holding.fundId && t.type === 'purchase'
      );

      if (fundTransactions.length > 0) {
        const purchaseDate = new Date(fundTransactions[0].date);
        const currentDate = new Date();
        const holdingPeriod = Math.floor((currentDate - purchaseDate) / (1000 * 60 * 60 * 24));

        const currentValue = holding.currentValue || 0;
        const purchaseValue = holding.purchaseValue || 0;
        const gain = currentValue - purchaseValue;

        ltcgRecords.push({
          fundId: holding.fundId,
          fundName: holding.fundName,
          purchaseDate,
          holdingPeriod,
          purchaseValue,
          currentValue,
          gain,
          isLongTerm: holdingPeriod >= 365,
          taxRate: holdingPeriod >= 365 ? this.taxRates.LTCG_EQUITY.rate : this.taxRates.STCG_EQUITY.rate
        });
      }
    }

    return {
      records: ltcgRecords,
      totalGain: ltcgRecords.reduce((sum, record) => sum + record.gain, 0),
      longTermGain: ltcgRecords.filter(r => r.isLongTerm).reduce((sum, record) => sum + record.gain, 0),
      shortTermGain: ltcgRecords.filter(r => !r.isLongTerm).reduce((sum, record) => sum + record.gain, 0)
    };
  }

  calculateTaxLiability(ltcgAnalysis, user) {
    const { longTermGain, shortTermGain } = ltcgAnalysis;
    
    // LTCG calculation
    const ltcgExemption = this.taxRates.LTCG_EQUITY.threshold;
    const taxableLTCG = Math.max(0, longTermGain - ltcgExemption);
    const ltcgTax = taxableLTCG * (this.taxRates.LTCG_EQUITY.rate / 100);

    // STCG calculation
    const stcgTax = shortTermGain * (this.taxRates.STCG_EQUITY.rate / 100);

    return {
      ltcgTax,
      stcgTax,
      totalTax: ltcgTax + stcgTax,
      taxableLTCG,
      ltcgExemption,
      effectiveTaxRate: (ltcgTax + stcgTax) / (longTermGain + shortTermGain) * 100
    };
  }

  async getCurrentTaxSavingInvestments(userId) {
    const investments = [];

    // Get ELSS investments
    const elssInvestments = await ELSSInvestment.find({ userId });
    investments.push(...elssInvestments.map(inv => ({
      type: 'ELSS',
      amount: inv.amount,
      section: '80C',
      limit: this.taxSavingOptions.ELSS.limit
    })));

    // Add other tax-saving investments (simplified)
    investments.push({
      type: 'PPF',
      amount: 50000, // Example
      section: '80C',
      limit: this.taxSavingOptions.PPF.limit
    });

    return investments;
  }

  calculateAvailableLimit(currentInvestments) {
    const usedLimit = currentInvestments.reduce((sum, inv) => sum + inv.amount, 0);
    return Math.max(0, this.taxSavingOptions.ELSS.limit - usedLimit);
  }

  async generateELSSRecommendations(user, availableLimit) {
    const recommendations = [];

    if (availableLimit > 0) {
      recommendations.push({
        type: 'ELSS_INVESTMENT',
        amount: availableLimit,
        description: `Invest ₹${availableLimit.toLocaleString()} in ELSS for tax saving`,
        taxSaving: availableLimit * 0.3, // 30% tax bracket
        expectedReturn: this.taxSavingOptions.ELSS.expectedReturn,
        risk: this.taxSavingOptions.ELSS.risk
      });
    }

    return recommendations;
  }

  async analyzeHarvestingOpportunities(user, userPortfolio, transactions) {
    const opportunities = [];

    // Loss harvesting opportunities
    const lossOpportunities = await this.identifyLossHarvesting(userPortfolio);
    opportunities.push(...lossOpportunities);

    // Gain harvesting opportunities
    const gainOpportunities = await this.identifyGainHarvesting(userPortfolio);
    opportunities.push(...gainOpportunities);

    return opportunities;
  }

  async identifyLossHarvesting(userPortfolio) {
    const opportunities = [];
    const holdings = userPortfolio.holdings || [];

    for (const holding of holdings) {
      const currentValue = holding.currentValue || 0;
      const purchaseValue = holding.purchaseValue || 0;
      const loss = purchaseValue - currentValue;

      if (loss > 0 && loss > 10000) { // Minimum ₹10,000 loss
        opportunities.push({
          type: 'LOSS_HARVESTING',
          fundId: holding.fundId,
          fundName: holding.fundName,
          loss,
          taxSaving: loss * 0.15, // 15% tax saving
          recommendation: `Consider selling ${holding.fundName} to harvest ₹${loss.toLocaleString()} loss`
        });
      }
    }

    return opportunities;
  }

  async identifyGainHarvesting(userPortfolio) {
    const opportunities = [];
    const holdings = userPortfolio.holdings || [];

    for (const holding of holdings) {
      const currentValue = holding.currentValue || 0;
      const purchaseValue = holding.purchaseValue || 0;
      const gain = currentValue - purchaseValue;

      if (gain > 0 && gain <= this.taxRates.LTCG_EQUITY.threshold) {
        opportunities.push({
          type: 'GAIN_HARVESTING',
          fundId: holding.fundId,
          fundName: holding.fundName,
          gain,
          taxSaving: 0, // No tax if within exemption
          recommendation: `Consider realizing ₹${gain.toLocaleString()} gain within exemption limit`
        });
      }
    }

    return opportunities;
  }

  calculatePreTaxXIRR(transactions) {
    // Simplified XIRR calculation
    let totalInvestment = 0;
    let totalValue = 0;

    transactions.forEach(transaction => {
      if (transaction.type === 'purchase') {
        totalInvestment += transaction.amount;
      } else if (transaction.type === 'redemption') {
        totalValue += transaction.amount;
      }
    });

    // Simple return calculation (in real implementation, use proper XIRR formula)
    return totalInvestment > 0 ? ((totalValue - totalInvestment) / totalInvestment) * 100 : 0;
  }

  async calculateTaxImpact(userId, transactions, preTaxXIRR) {
    const ltcgAnalysis = await this.calculateLTCG(transactions, { holdings: [] });
    const user = await User.findById(userId);
    const taxLiability = this.calculateTaxLiability(ltcgAnalysis, user);

    return {
      taxRate: taxLiability.effectiveTaxRate,
      taxAmount: taxLiability.totalTax,
      impact: taxLiability.totalTax / (preTaxXIRR / 100)
    };
  }

  calculatePostTaxXIRR(preTaxXIRR, taxImpact) {
    return preTaxXIRR - (preTaxXIRR * taxImpact.taxRate / 100);
  }

  async generateSIPTaxOptimization(user, sipAmount, availableLimit) {
    const optimization = {
      currentSIP: sipAmount,
      recommendedAllocation: {
        elss: Math.min(availableLimit / 12, sipAmount * 0.3), // 30% to ELSS
        regular: sipAmount - Math.min(availableLimit / 12, sipAmount * 0.3)
      },
      taxSaving: Math.min(availableLimit / 12, sipAmount * 0.3) * 0.3, // 30% tax bracket
      expectedReturn: 12.0
    };

    return optimization;
  }

  async generateComprehensiveTaxPlan(user, userPortfolio, currentInvestments) {
    const plan = {
      currentYear: new Date().getFullYear(),
      taxBracket: this.getTaxBracket(user.income),
      currentInvestments,
      recommendations: [],
      timeline: []
    };

    // Add ELSS recommendations
    const availableLimit = this.calculateAvailableLimit(currentInvestments);
    if (availableLimit > 0) {
      plan.recommendations.push({
        type: 'ELSS_INVESTMENT',
        priority: 'high',
        amount: availableLimit,
        deadline: 'March 31',
        description: 'Maximize Section 80C deduction'
      });
    }

    // Add NPS recommendations
    plan.recommendations.push({
      type: 'NPS_CONTRIBUTION',
      priority: 'medium',
      amount: 50000,
      deadline: 'March 31',
      description: 'Additional tax deduction under Section 80CCD(1B)'
    });

    return plan;
  }

  getTaxBracket(income) {
    if (income <= 250000) return 0;
    if (income <= 500000) return 5;
    if (income <= 1000000) return 20;
    return 30;
  }

  getTaxDeadline() {
    const currentYear = new Date().getFullYear();
    return `${currentYear}-03-31`;
  }

  // Storage methods
  async storeLTCGAnalysis(userId, ltcgAnalysis, taxLiability) {
    const ltcgRecord = new LTCGRecord({
      userId,
      analysis: ltcgAnalysis,
      taxLiability,
      analysisDate: new Date()
    });

    await ltcgRecord.save();
  }

  // Recommendation methods
  async getLTCGRecommendations(userId, ltcgAnalysis, taxLiability) {
    const recommendations = [];

    if (taxLiability.taxableLTCG > 0) {
      recommendations.push('Consider tax-loss harvesting to offset gains');
    }

    if (ltcgAnalysis.longTermGain < this.taxRates.LTCG_EQUITY.threshold) {
      recommendations.push('Consider realizing gains within exemption limit');
    }

    return recommendations;
  }

  async getHarvestingRecommendations(userId, harvestingOpportunities) {
    return harvestingOpportunities.map(opp => opp.recommendation);
  }

  async getSIPTaxRecommendations(userId, optimization) {
    return [
      `Allocate ₹${optimization.recommendedAllocation.elss.toLocaleString()} to ELSS monthly`,
      `Expected tax saving: ₹${optimization.taxSaving.toLocaleString()} annually`,
      'Review allocation quarterly for optimal tax efficiency'
    ];
  }

  async getTaxOptimizationRecommendations(userId, taxPlan) {
    return taxPlan.recommendations.map(rec => rec.description);
  }

  // Implementation methods
  async getELSSInvestmentStrategy(userId, availableLimit) {
    return {
      monthlyInvestment: Math.ceil(availableLimit / 12),
      fundSelection: 'Top-performing ELSS funds',
      rebalancing: 'Annual',
      monitoring: 'Quarterly'
    };
  }

  async getSIPTaxImplementation(userId, optimization) {
    return {
      steps: [
        'Set up ELSS SIP for recommended amount',
        'Monitor tax-saving utilization',
        'Review and adjust annually'
      ],
      timeline: 'Immediate implementation'
    };
  }

  // Analysis methods
  async getTaxBreakdown(userId, transactions) {
    return {
      ltcg: 0,
      stcg: 0,
      dividend: 0,
      total: 0
    };
  }

  async getXIRROptimization(userId, postTaxXIRR) {
    return {
      current: postTaxXIRR,
      potential: postTaxXIRR * 1.1, // 10% improvement potential
      strategies: ['Tax-loss harvesting', 'Gain harvesting', 'Asset location optimization']
    };
  }

  async getCurrentTaxStatus(userId) {
    return {
      currentYear: new Date().getFullYear(),
      taxSavingUtilized: 0,
      taxSavingAvailable: 150000,
      deadline: 'March 31'
    };
  }

  getHarvestingTimeline(harvestingOpportunities) {
    return {
      immediate: harvestingOpportunities.filter(opp => opp.type === 'LOSS_HARVESTING'),
      quarterly: harvestingOpportunities.filter(opp => opp.type === 'GAIN_HARVESTING'),
      annual: 'Year-end tax optimization'
    };
  }

  getTaxOptimizationTimeline(taxPlan) {
    return {
      immediate: taxPlan.recommendations.filter(rec => rec.priority === 'high'),
      monthly: 'ELSS SIP contributions',
      quarterly: 'Tax-saving utilization review',
      annual: 'Comprehensive tax planning'
    };
  }

  async createHarvestingPlan(userId, harvestingOpportunities) {
    return {
      opportunities: harvestingOpportunities,
      implementation: 'Execute based on market conditions',
      monitoring: 'Weekly review of opportunities'
    };
  }
}

module.exports = new TaxOptimizer(); 