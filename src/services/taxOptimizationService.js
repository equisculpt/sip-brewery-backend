const logger = require('../utils/logger');
const dayjs = require('dayjs');

class TaxOptimizationService {
  constructor() {
    this.taxRates = {
      ltcg: {
        equity: 0.10, // 10% for gains > 1L
        debt: 0.20,   // 20% with indexation
        others: 0.20  // 20% flat
      },
      stcg: {
        equity: 0.15, // 15%
        debt: 0.30,   // 30% slab rate
        others: 0.30  // 30% slab rate
      }
    };
    
    this.section80CLimits = {
      total: 150000,
      categories: {
        elss: 150000,
        ppf: 150000,
        nps: 50000,
        insurance: 25000,
        homeLoan: 200000
      }
    };
  }

  /**
   * Optimize portfolio for tax efficiency
   */
  async optimizeForTaxes(portfolioData, userProfile) {
    try {
      const optimization = {
        taxLossHarvesting: await this.generateTaxLossHarvesting(portfolioData),
        section80COptimization: await this.optimizeSection80C(userProfile),
        assetLocation: await this.optimizeAssetLocation(portfolioData),
        withdrawalStrategy: await this.optimizeWithdrawalStrategy(portfolioData),
        internationalTax: await this.handleInternationalTax(portfolioData, userProfile)
      };

      return optimization;
    } catch (error) {
      logger.error('Error optimizing for taxes:', error);
      return null;
    }
  }

  /**
   * Generate tax-loss harvesting opportunities
   */
  async generateTaxLossHarvesting(portfolioData) {
    try {
      const opportunities = [];
      const currentDate = dayjs();
      
      for (const holding of portfolioData.holdings || []) {
        const taxAnalysis = this.analyzeTaxLossHarvesting(holding, currentDate);
        
        if (taxAnalysis.shouldHarvest) {
          opportunities.push({
            fundName: holding.fundName,
            schemeCode: holding.schemeCode,
            currentValue: holding.currentValue,
            investedValue: holding.investedValue,
            unrealizedLoss: holding.currentValue - holding.investedValue,
            taxBenefit: Math.abs(holding.currentValue - holding.investedValue) * 0.30, // Assuming 30% tax rate
            replacementFunds: await this.findReplacementFunds(holding),
            harvestStrategy: taxAnalysis.strategy,
            timing: taxAnalysis.timing
          });
        }
      }

      return {
        opportunities,
        totalTaxBenefit: opportunities.reduce((sum, opp) => sum + opp.taxBenefit, 0),
        recommendations: this.generateHarvestingRecommendations(opportunities)
      };
    } catch (error) {
      logger.error('Error generating tax-loss harvesting:', error);
      return { opportunities: [], totalTaxBenefit: 0, recommendations: [] };
    }
  }

  /**
   * Analyze tax-loss harvesting for a specific holding
   */
  analyzeTaxLossHarvesting(holding, currentDate) {
    const unrealizedLoss = holding.currentValue - holding.investedValue;
    const holdingPeriod = dayjs(currentDate).diff(dayjs(holding.purchaseDate), 'day');
    
    // Check if it's a good candidate for tax-loss harvesting
    const shouldHarvest = unrealizedLoss < 0 && 
                         holdingPeriod >= 30 && // Avoid wash sale rule
                         Math.abs(unrealizedLoss) > holding.investedValue * 0.05; // 5% threshold
    
    let strategy = 'HOLD';
    let timing = 'IMMEDIATE';
    
    if (shouldHarvest) {
      if (holdingPeriod >= 365) {
        strategy = 'HARVEST_LTCG_LOSS';
        timing = 'BEFORE_YEAR_END';
      } else {
        strategy = 'HARVEST_STCG_LOSS';
        timing = 'IMMEDIATE';
      }
    }

    return {
      shouldHarvest,
      strategy,
      timing,
      holdingPeriod,
      unrealizedLoss
    };
  }

  /**
   * Find replacement funds for tax-loss harvesting
   */
  async findReplacementFunds(holding) {
    try {
      // Find funds with similar characteristics but different enough to avoid wash sale
      const replacements = [];
      
      // Mock implementation - in production, use fund database
      const similarFunds = [
        {
          schemeCode: 'REPLACE001',
          schemeName: 'Similar Fund 1',
          category: holding.category,
          correlation: 0.85,
          expenseRatio: holding.expenseRatio * 1.1
        },
        {
          schemeCode: 'REPLACE002',
          schemeName: 'Similar Fund 2',
          category: holding.category,
          correlation: 0.80,
          expenseRatio: holding.expenseRatio * 0.95
        }
      ];

      return similarFunds.map(fund => ({
        ...fund,
        suitability: this.calculateReplacementSuitability(holding, fund)
      }));
    } catch (error) {
      logger.error('Error finding replacement funds:', error);
      return [];
    }
  }

  /**
   * Calculate replacement fund suitability
   */
  calculateReplacementSuitability(original, replacement) {
    let score = 0;
    
    // Category match
    if (original.category === replacement.category) score += 0.4;
    
    // Correlation (lower is better for tax-loss harvesting)
    if (replacement.correlation < 0.9) score += 0.3;
    
    // Expense ratio comparison
    if (replacement.expenseRatio <= original.expenseRatio) score += 0.3;
    
    return Math.min(1, score);
  }

  /**
   * Generate harvesting recommendations
   */
  generateHarvestingRecommendations(opportunities) {
    const recommendations = [];
    
    if (opportunities.length === 0) {
      recommendations.push({
        type: 'INFO',
        message: 'No tax-loss harvesting opportunities found at this time.',
        priority: 'LOW'
      });
      return recommendations;
    }

    // Sort opportunities by tax benefit
    const sortedOpportunities = opportunities.sort((a, b) => b.taxBenefit - a.taxBenefit);
    
    // High-value opportunities
    const highValue = sortedOpportunities.filter(opp => opp.taxBenefit > 10000);
    if (highValue.length > 0) {
      recommendations.push({
        type: 'HIGH_PRIORITY',
        message: `Consider harvesting losses in ${highValue.length} funds for significant tax savings.`,
        funds: highValue.map(opp => opp.fundName),
        estimatedSavings: highValue.reduce((sum, opp) => sum + opp.taxBenefit, 0),
        priority: 'HIGH'
      });
    }

    // Medium-value opportunities
    const mediumValue = sortedOpportunities.filter(opp => opp.taxBenefit > 5000 && opp.taxBenefit <= 10000);
    if (mediumValue.length > 0) {
      recommendations.push({
        type: 'MEDIUM_PRIORITY',
        message: `Consider harvesting losses in ${mediumValue.length} funds for moderate tax savings.`,
        funds: mediumValue.map(opp => opp.fundName),
        estimatedSavings: mediumValue.reduce((sum, opp) => sum + opp.taxBenefit, 0),
        priority: 'MEDIUM'
      });
    }

    return recommendations;
  }

  /**
   * Optimize Section 80C investments
   */
  async optimizeSection80C(userProfile) {
    try {
      const currentInvestments = userProfile.section80C || {};
      const optimization = {
        currentUtilization: this.calculateCurrentUtilization(currentInvestments),
        recommendations: await this.generate80CRecommendations(currentInvestments, userProfile),
        optimalAllocation: this.calculateOptimal80CAllocation(userProfile),
        taxSavings: this.calculate80CTaxSavings(currentInvestments)
      };

      return optimization;
    } catch (error) {
      logger.error('Error optimizing Section 80C:', error);
      return null;
    }
  }

  /**
   * Calculate current Section 80C utilization
   */
  calculateCurrentUtilization(currentInvestments) {
    const totalInvested = Object.values(currentInvestments).reduce((sum, amount) => sum + (amount || 0), 0);
    const utilization = (totalInvested / this.section80CLimits.total) * 100;
    
    return {
      totalInvested,
      limit: this.section80CLimits.total,
      utilization: Math.min(100, utilization),
      remaining: Math.max(0, this.section80CLimits.total - totalInvested)
    };
  }

  /**
   * Generate Section 80C recommendations
   */
  async generate80CRecommendations(currentInvestments, userProfile) {
    const recommendations = [];
    const utilization = this.calculateCurrentUtilization(currentInvestments);
    
    // Check if under-utilized
    if (utilization.utilization < 80) {
      recommendations.push({
        type: 'UNDER_UTILIZED',
        message: `You're only using ${utilization.utilization.toFixed(1)}% of your Section 80C limit.`,
        suggestion: `Consider investing ₹${utilization.remaining.toLocaleString()} more to maximize tax savings.`,
        priority: 'HIGH'
      });
    }

    // Check category-wise optimization
    for (const [category, limit] of Object.entries(this.section80CLimits.categories)) {
      const current = currentInvestments[category] || 0;
      const remaining = limit - current;
      
      if (remaining > 0) {
        recommendations.push({
          type: 'CATEGORY_OPTIMIZATION',
          category,
          current,
          limit,
          remaining,
          suggestion: `Consider investing ₹${remaining.toLocaleString()} in ${category.toUpperCase()} to maximize this category.`,
          priority: 'MEDIUM'
        });
      }
    }

    return recommendations;
  }

  /**
   * Calculate optimal Section 80C allocation
   */
  calculateOptimal80CAllocation(userProfile) {
    const allocation = {};
    const totalLimit = this.section80CLimits.total;
    
    // Based on user profile, suggest optimal allocation
    if (userProfile.age < 30) {
      // Younger investors can take more risk
      allocation.elss = Math.min(150000, totalLimit * 0.6);
      allocation.ppf = Math.min(150000, totalLimit * 0.3);
      allocation.nps = Math.min(50000, totalLimit * 0.1);
    } else if (userProfile.age < 50) {
      // Balanced approach
      allocation.elss = Math.min(150000, totalLimit * 0.4);
      allocation.ppf = Math.min(150000, totalLimit * 0.4);
      allocation.nps = Math.min(50000, totalLimit * 0.15);
      allocation.insurance = Math.min(25000, totalLimit * 0.05);
    } else {
      // Conservative approach for older investors
      allocation.ppf = Math.min(150000, totalLimit * 0.5);
      allocation.nps = Math.min(50000, totalLimit * 0.3);
      allocation.insurance = Math.min(25000, totalLimit * 0.2);
    }

    return allocation;
  }

  /**
   * Calculate Section 80C tax savings
   */
  calculate80CTaxSavings(currentInvestments) {
    const totalInvested = Object.values(currentInvestments).reduce((sum, amount) => sum + (amount || 0), 0);
    const taxRate = 0.30; // Assuming 30% tax bracket
    
    return {
      currentSavings: totalInvested * taxRate,
      potentialSavings: this.section80CLimits.total * taxRate,
      additionalSavings: (this.section80CLimits.total - totalInvested) * taxRate
    };
  }

  /**
   * Optimize asset location for tax efficiency
   */
  async optimizeAssetLocation(portfolioData) {
    try {
      const optimization = {
        equityLocation: this.optimizeEquityLocation(portfolioData),
        debtLocation: this.optimizeDebtLocation(portfolioData),
        internationalLocation: this.optimizeInternationalLocation(portfolioData),
        recommendations: this.generateLocationRecommendations(portfolioData)
      };

      return optimization;
    } catch (error) {
      logger.error('Error optimizing asset location:', error);
      return null;
    }
  }

  /**
   * Optimize equity asset location
   */
  optimizeEquityLocation(portfolioData) {
    const equityFunds = (portfolioData.holdings || []).filter(h => h.category.includes('Equity'));
    
    return {
      currentAllocation: equityFunds.reduce((sum, fund) => sum + fund.currentValue, 0),
      recommendedLocation: 'ELSS', // Tax-efficient equity funds
      taxEfficiency: this.calculateTaxEfficiency(equityFunds, 'equity'),
      suggestions: this.generateEquityLocationSuggestions(equityFunds)
    };
  }

  /**
   * Optimize debt asset location
   */
  optimizeDebtLocation(portfolioData) {
    const debtFunds = (portfolioData.holdings || []).filter(h => h.category.includes('Debt'));
    
    return {
      currentAllocation: debtFunds.reduce((sum, fund) => sum + fund.currentValue, 0),
      recommendedLocation: 'PPF', // Tax-efficient debt instruments
      taxEfficiency: this.calculateTaxEfficiency(debtFunds, 'debt'),
      suggestions: this.generateDebtLocationSuggestions(debtFunds)
    };
  }

  /**
   * Optimize international asset location
   */
  optimizeInternationalLocation(portfolioData) {
    const internationalFunds = (portfolioData.holdings || []).filter(h => h.category.includes('International'));
    
    return {
      currentAllocation: internationalFunds.reduce((sum, fund) => sum + fund.currentValue, 0),
      recommendedLocation: 'Dedicated International Funds',
      taxEfficiency: this.calculateTaxEfficiency(internationalFunds, 'international'),
      suggestions: this.generateInternationalLocationSuggestions(internationalFunds)
    };
  }

  /**
   * Calculate tax efficiency
   */
  calculateTaxEfficiency(funds, category) {
    let efficiency = 0;
    
    for (const fund of funds) {
      const holdingPeriod = dayjs().diff(dayjs(fund.purchaseDate), 'day');
      const taxRate = holdingPeriod >= 365 ? this.taxRates.ltcg[category] : this.taxRates.stcg[category];
      
      efficiency += (1 - taxRate) * fund.currentValue;
    }
    
    return efficiency;
  }

  /**
   * Optimize withdrawal strategy
   */
  async optimizeWithdrawalStrategy(portfolioData) {
    try {
      const strategy = {
        withdrawalOrder: this.calculateOptimalWithdrawalOrder(portfolioData),
        taxImplications: this.calculateWithdrawalTaxImplications(portfolioData),
        recommendations: this.generateWithdrawalRecommendations(portfolioData)
      };

      return strategy;
    } catch (error) {
      logger.error('Error optimizing withdrawal strategy:', error);
      return null;
    }
  }

  /**
   * Calculate optimal withdrawal order
   */
  calculateOptimalWithdrawalOrder(portfolioData) {
    const holdings = portfolioData.holdings || [];
    
    // Sort by tax efficiency (withdraw least tax-efficient first)
    return holdings
      .map(holding => ({
        ...holding,
        taxEfficiency: this.calculateHoldingTaxEfficiency(holding)
      }))
      .sort((a, b) => a.taxEfficiency - b.taxEfficiency)
      .map((holding, index) => ({
        ...holding,
        withdrawalOrder: index + 1,
        reason: this.getWithdrawalReason(holding, index)
      }));
  }

  /**
   * Calculate holding tax efficiency
   */
  calculateHoldingTaxEfficiency(holding) {
    const holdingPeriod = dayjs().diff(dayjs(holding.purchaseDate), 'day');
    const category = this.getFundCategory(holding.category);
    const taxRate = holdingPeriod >= 365 ? this.taxRates.ltcg[category] : this.taxRates.stcg[category];
    
    return taxRate;
  }

  /**
   * Get fund category for tax purposes
   */
  getFundCategory(category) {
    if (category.includes('Equity')) return 'equity';
    if (category.includes('Debt')) return 'debt';
    return 'others';
  }

  /**
   * Get withdrawal reason
   */
  getWithdrawalReason(holding, order) {
    if (order === 0) return 'Highest tax burden - withdraw first';
    if (order < 3) return 'High tax burden - consider early withdrawal';
    return 'Tax efficient - withdraw last';
  }

  /**
   * Handle international tax compliance
   */
  async handleInternationalTax(portfolioData, userProfile) {
    try {
      const internationalHoldings = (portfolioData.holdings || []).filter(h => h.category.includes('International'));
      
      if (internationalHoldings.length === 0) {
        return {
          hasInternationalExposure: false,
          compliance: 'N/A',
          recommendations: []
        };
      }

      const compliance = {
        hasInternationalExposure: true,
        fcraCompliance: this.checkFRCACompliance(internationalHoldings),
        taxTreatyBenefits: await this.checkTaxTreatyBenefits(userProfile),
        reportingRequirements: this.getReportingRequirements(internationalHoldings),
        recommendations: this.generateInternationalTaxRecommendations(internationalHoldings, userProfile)
      };

      return compliance;
    } catch (error) {
      logger.error('Error handling international tax:', error);
      return null;
    }
  }

  /**
   * Check FRCA compliance
   */
  checkFRCACompliance(internationalHoldings) {
    const totalExposure = internationalHoldings.reduce((sum, holding) => sum + holding.currentValue, 0);
    
    return {
      compliant: totalExposure <= 250000, // $250,000 limit
      currentExposure: totalExposure,
      limit: 250000,
      recommendations: totalExposure > 250000 ? ['Consider reducing international exposure to stay within FRCA limits'] : []
    };
  }

  /**
   * Check tax treaty benefits
   */
  async checkTaxTreatyBenefits(userProfile) {
    // Mock implementation - in production, check actual tax treaties
    return {
      applicable: false,
      countries: [],
      benefits: []
    };
  }

  /**
   * Get reporting requirements
   */
  getReportingRequirements(internationalHoldings) {
    const totalExposure = internationalHoldings.reduce((sum, holding) => sum + holding.currentValue, 0);
    
    return {
      fcraReporting: totalExposure > 100000,
      taxReporting: totalExposure > 50000,
      forms: this.getRequiredForms(totalExposure)
    };
  }

  /**
   * Get required forms
   */
  getRequiredForms(totalExposure) {
    const forms = [];
    
    if (totalExposure > 100000) forms.push('FCRA Annual Return');
    if (totalExposure > 50000) forms.push('Schedule FA (Tax Return)');
    
    return forms;
  }

  /**
   * Generate comprehensive tax report
   */
  async generateTaxReport(portfolioData, userProfile) {
    try {
      const report = {
        summary: await this.generateTaxSummary(portfolioData),
        taxLossHarvesting: await this.generateTaxLossHarvesting(portfolioData),
        section80C: await this.optimizeSection80C(userProfile),
        assetLocation: await this.optimizeAssetLocation(portfolioData),
        withdrawalStrategy: await this.optimizeWithdrawalStrategy(portfolioData),
        internationalTax: await this.handleInternationalTax(portfolioData, userProfile),
        recommendations: this.generateOverallTaxRecommendations(portfolioData, userProfile)
      };

      return report;
    } catch (error) {
      logger.error('Error generating tax report:', error);
      return null;
    }
  }

  /**
   * Generate tax summary
   */
  async generateTaxSummary(portfolioData) {
    const holdings = portfolioData.holdings || [];
    const currentYear = dayjs().year();
    
    const summary = {
      totalValue: holdings.reduce((sum, h) => sum + h.currentValue, 0),
      totalInvested: holdings.reduce((sum, h) => sum + h.investedValue, 0),
      unrealizedGains: holdings.reduce((sum, h) => sum + Math.max(0, h.currentValue - h.investedValue), 0),
      unrealizedLosses: holdings.reduce((sum, h) => sum + Math.min(0, h.currentValue - h.investedValue), 0),
      estimatedTaxLiability: this.calculateEstimatedTaxLiability(holdings),
      taxEfficiencyScore: this.calculateTaxEfficiencyScore(holdings)
    };

    return summary;
  }

  /**
   * Calculate estimated tax liability
   */
  calculateEstimatedTaxLiability(holdings) {
    let totalLiability = 0;
    
    for (const holding of holdings) {
      const gain = holding.currentValue - holding.investedValue;
      if (gain > 0) {
        const holdingPeriod = dayjs().diff(dayjs(holding.purchaseDate), 'day');
        const category = this.getFundCategory(holding.category);
        const taxRate = holdingPeriod >= 365 ? this.taxRates.ltcg[category] : this.taxRates.stcg[category];
        
        totalLiability += gain * taxRate;
      }
    }
    
    return totalLiability;
  }

  /**
   * Calculate tax efficiency score
   */
  calculateTaxEfficiencyScore(holdings) {
    if (holdings.length === 0) return 0;
    
    let totalScore = 0;
    
    for (const holding of holdings) {
      const efficiency = this.calculateHoldingTaxEfficiency(holding);
      totalScore += (1 - efficiency) * holding.currentValue;
    }
    
    const totalValue = holdings.reduce((sum, h) => sum + h.currentValue, 0);
    return totalValue > 0 ? (totalScore / totalValue) * 100 : 0;
  }

  /**
   * Generate overall tax recommendations
   */
  generateOverallTaxRecommendations(portfolioData, userProfile) {
    const recommendations = [];
    
    // Add recommendations from all optimization areas
    recommendations.push(...this.generateHarvestingRecommendations([]));
    recommendations.push(...this.generate80CRecommendations({}, userProfile));
    
    return recommendations.sort((a, b) => {
      const priorityOrder = { HIGH: 3, MEDIUM: 2, LOW: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }
}

module.exports = TaxOptimizationService; 