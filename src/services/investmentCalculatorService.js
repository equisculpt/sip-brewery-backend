const moment = require('moment');
const logger = require('../utils/logger');

class InvestmentCalculatorService {
  constructor() {
    this.inflationRate = 6.5; // Default inflation rate
    this.taxRates = {
      shortTerm: 15, // Short term capital gains tax
      longTerm: 10,  // Long term capital gains tax (above 1L)
      dividend: 10   // Dividend distribution tax
    };
  }

  /**
   * Calculate SIP projections with detailed breakdown
   */
  async calculateSIPProjections({
    monthlyAmount,
    duration,
    expectedReturn,
    fundCode,
    startDate = new Date(),
    includeInflation = true,
    includeTaxes = true
  }) {
    try {
      logger.info(`Calculating SIP projections: ${monthlyAmount} for ${duration} months at ${expectedReturn}%`);

      const projections = {
        summary: {},
        monthlyBreakdown: [],
        yearlyBreakdown: [],
        charts: {
          investmentVsWealth: [],
          wealthGained: [],
          inflationAdjusted: []
        }
      };

      let totalInvested = 0;
      let totalValue = 0;
      let totalWealthGained = 0;
      let totalTaxPaid = 0;

      // Calculate monthly breakdown
      for (let month = 1; month <= duration; month++) {
        const currentDate = moment(startDate).add(month, 'months');
        totalInvested += monthlyAmount;
        
        // Calculate future value using SIP formula
        const futureValue = this.calculateSIPFutureValue(monthlyAmount, month, expectedReturn);
        totalValue = futureValue;
        
        const wealthGained = totalValue - totalInvested;
        totalWealthGained = wealthGained;

        // Calculate inflation-adjusted value
        const inflationAdjustedValue = includeInflation ? 
          this.adjustForInflation(totalValue, month) : totalValue;

        // Calculate tax implications
        let taxPaid = 0;
        if (includeTaxes && wealthGained > 0) {
          taxPaid = this.calculateTaxOnGains(wealthGained, month);
          totalTaxPaid += taxPaid;
        }

        const monthlyData = {
          month,
          date: currentDate.toDate(),
          invested: totalInvested,
          expectedValue: totalValue,
          wealthGained,
          inflationAdjustedValue,
          taxPaid,
          netValue: totalValue - taxPaid,
          returnPercentage: (wealthGained / totalInvested) * 100
        };

        projections.monthlyBreakdown.push(monthlyData);

        // Add to charts data
        projections.charts.investmentVsWealth.push({
          date: currentDate.valueOf(),
          invested: totalInvested,
          expectedValue: totalValue,
          netValue: totalValue - taxPaid
        });

        projections.charts.wealthGained.push({
          date: currentDate.valueOf(),
          wealthGained,
          taxPaid
        });

        projections.charts.inflationAdjusted.push({
          date: currentDate.valueOf(),
          nominalValue: totalValue,
          inflationAdjustedValue
        });

        // Yearly breakdown
        if (month % 12 === 0) {
          const year = Math.floor(month / 12);
          projections.yearlyBreakdown.push({
            year,
            invested: totalInvested,
            expectedValue: totalValue,
            wealthGained,
            returnPercentage: (wealthGained / totalInvested) * 100,
            taxPaid: totalTaxPaid
          });
        }
      }

      // Calculate summary
      projections.summary = {
        totalInvestment: totalInvested,
        expectedValue: totalValue,
        wealthGained: totalWealthGained,
        totalTaxPaid,
        netValue: totalValue - totalTaxPaid,
        averageReturn: (totalWealthGained / totalInvested) * 100,
        inflationAdjustedValue: includeInflation ? 
          this.adjustForInflation(totalValue, duration) : totalValue,
        duration: {
          months: duration,
          years: duration / 12
        },
        monthlyAmount,
        expectedReturn
      };

      return projections;
    } catch (error) {
      logger.error('Error calculating SIP projections:', error);
      throw error;
    }
  }

  /**
   * Calculate goal-based investment requirements
   */
  async calculateGoalBasedInvestment({
    goalAmount,
    targetDate,
    currentSavings = 0,
    riskProfile = 'moderate',
    includeInflation = true
  }) {
    try {
      logger.info(`Calculating goal-based investment: ${goalAmount} by ${targetDate}`);

      const targetDateMoment = moment(targetDate);
      const currentDate = moment();
      const monthsToGoal = targetDateMoment.diff(currentDate, 'months');

      if (monthsToGoal <= 0) {
        throw new Error('Target date must be in the future');
      }

      // Adjust goal amount for inflation
      const inflationAdjustedGoal = includeInflation ? 
        this.adjustForInflation(goalAmount, monthsToGoal) : goalAmount;

      // Get expected return based on risk profile
      const expectedReturn = this.getExpectedReturnForRiskProfile(riskProfile);

      // Calculate required monthly investment
      const requiredMonthlyInvestment = this.calculateRequiredMonthlyInvestment(
        inflationAdjustedGoal,
        currentSavings,
        monthsToGoal,
        expectedReturn
      );

      // Calculate different scenarios
      const scenarios = this.calculateInvestmentScenarios(
        inflationAdjustedGoal,
        currentSavings,
        monthsToGoal,
        riskProfile
      );

      // Calculate SIP projections for the required amount
      const sipProjections = await this.calculateSIPProjections({
        monthlyAmount: requiredMonthlyInvestment,
        duration: monthsToGoal,
        expectedReturn,
        includeInflation,
        includeTaxes: true
      });

      const result = {
        goal: {
          amount: goalAmount,
          inflationAdjustedAmount: inflationAdjustedGoal,
          targetDate: targetDateMoment.toDate(),
          monthsToGoal
        },
        currentSavings,
        requiredMonthlyInvestment,
        expectedReturn,
        riskProfile,
        scenarios,
        sipProjections: sipProjections.summary,
        recommendations: this.generateGoalRecommendations(
          requiredMonthlyInvestment,
          riskProfile,
          monthsToGoal
        )
      };

      return result;
    } catch (error) {
      logger.error('Error calculating goal-based investment:', error);
      throw error;
    }
  }

  /**
   * Calculate lumpsum investment projections
   */
  async calculateLumpsumProjections({
    amount,
    duration,
    expectedReturn,
    includeInflation = true,
    includeTaxes = true
  }) {
    try {
      logger.info(`Calculating lumpsum projections: ${amount} for ${duration} months at ${expectedReturn}%`);

      const projections = {
        summary: {},
        yearlyBreakdown: [],
        charts: {
          valueGrowth: [],
          inflationAdjusted: []
        }
      };

      const years = Math.ceil(duration / 12);
      let totalValue = amount;
      let totalTaxPaid = 0;

      for (let year = 1; year <= years; year++) {
        // Calculate compound interest
        totalValue = amount * Math.pow(1 + expectedReturn / 100, year);
        
        const wealthGained = totalValue - amount;
        
        // Calculate tax
        let taxPaid = 0;
        if (includeTaxes && wealthGained > 0) {
          taxPaid = this.calculateTaxOnGains(wealthGained, year * 12);
          totalTaxPaid += taxPaid;
        }

        // Calculate inflation-adjusted value
        const inflationAdjustedValue = includeInflation ? 
          this.adjustForInflation(totalValue, year * 12) : totalValue;

        const yearlyData = {
          year,
          invested: amount,
          expectedValue: totalValue,
          wealthGained,
          returnPercentage: (wealthGained / amount) * 100,
          taxPaid,
          netValue: totalValue - taxPaid,
          inflationAdjustedValue
        };

        projections.yearlyBreakdown.push(yearlyData);

        // Add to charts
        projections.charts.valueGrowth.push({
          year,
          value: totalValue,
          netValue: totalValue - taxPaid
        });

        projections.charts.inflationAdjusted.push({
          year,
          nominalValue: totalValue,
          inflationAdjustedValue
        });
      }

      // Calculate summary
      projections.summary = {
        initialInvestment: amount,
        finalValue: totalValue,
        wealthGained: totalValue - amount,
        totalTaxPaid,
        netValue: totalValue - totalTaxPaid,
        totalReturn: ((totalValue - amount) / amount) * 100,
        inflationAdjustedValue: includeInflation ? 
          this.adjustForInflation(totalValue, duration) : totalValue,
        duration: {
          months: duration,
          years: years
        },
        expectedReturn
      };

      return projections;
    } catch (error) {
      logger.error('Error calculating lumpsum projections:', error);
      throw error;
    }
  }

  /**
   * Calculate mutual fund returns comparison
   */
  async calculateFundComparison({
    funds,
    amount,
    duration,
    includeExpenseRatio = true
  }) {
    try {
      logger.info(`Calculating fund comparison for ${funds.length} funds`);

      const comparison = {
        funds: [],
        summary: {},
        charts: {
          performanceComparison: [],
          expenseImpact: []
        }
      };

      let bestFund = null;
      let worstFund = null;

      for (const fund of funds) {
        const netReturn = includeExpenseRatio ? 
          fund.expectedReturn - fund.expenseRatio : fund.expectedReturn;

        const projections = await this.calculateLumpsumProjections({
          amount,
          duration,
          expectedReturn: netReturn,
          includeInflation: true,
          includeTaxes: true
        });

        const fundData = {
          fundCode: fund.fundCode,
          fundName: fund.fundName,
          category: fund.category,
          expectedReturn: fund.expectedReturn,
          expenseRatio: fund.expenseRatio,
          netReturn,
          finalValue: projections.summary.finalValue,
          wealthGained: projections.summary.wealthGained,
          totalReturn: projections.summary.totalReturn,
          projections: projections.summary
        };

        comparison.funds.push(fundData);

        // Track best and worst performers
        if (!bestFund || fundData.totalReturn > bestFund.totalReturn) {
          bestFund = fundData;
        }
        if (!worstFund || fundData.totalReturn < worstFund.totalReturn) {
          worstFund = fundData;
        }

        // Add to charts
        comparison.charts.performanceComparison.push({
          fundName: fund.fundName,
          finalValue: fundData.finalValue,
          totalReturn: fundData.totalReturn
        });
      }

      // Calculate summary
      comparison.summary = {
        totalFunds: funds.length,
        bestFund: {
          name: bestFund.fundName,
          return: bestFund.totalReturn,
          value: bestFund.finalValue
        },
        worstFund: {
          name: worstFund.fundName,
          return: worstFund.totalReturn,
          value: worstFund.finalValue
        },
        averageReturn: comparison.funds.reduce((sum, fund) => sum + fund.totalReturn, 0) / funds.length,
        returnDifference: bestFund.totalReturn - worstFund.totalReturn
      };

      return comparison;
    } catch (error) {
      logger.error('Error calculating fund comparison:', error);
      throw error;
    }
  }

  /**
   * Calculate retirement planning
   */
  async calculateRetirementPlanning({
    currentAge,
    retirementAge,
    lifeExpectancy,
    currentSavings,
    monthlyExpenses,
    expectedReturn,
    inflationRate = this.inflationRate
  }) {
    try {
      logger.info(`Calculating retirement planning for age ${currentAge} to ${retirementAge}`);

      const yearsToRetirement = retirementAge - currentAge;
      const retirementYears = lifeExpectancy - retirementAge;

      // Calculate required retirement corpus
      const inflationAdjustedMonthlyExpenses = this.adjustForInflation(
        monthlyExpenses * 12, 
        yearsToRetirement
      );

      const requiredCorpus = this.calculateRetirementCorpus(
        inflationAdjustedMonthlyExpenses,
        retirementYears,
        expectedReturn,
        inflationRate
      );

      // Calculate required monthly savings
      const requiredMonthlySavings = this.calculateRequiredMonthlyInvestment(
        requiredCorpus,
        currentSavings,
        yearsToRetirement * 12,
        expectedReturn
      );

      // Calculate different scenarios
      const scenarios = this.calculateRetirementScenarios({
        currentAge,
        retirementAge,
        lifeExpectancy,
        currentSavings,
        monthlyExpenses,
        expectedReturn,
        inflationRate
      });

      const result = {
        currentAge,
        retirementAge,
        lifeExpectancy,
        yearsToRetirement,
        retirementYears,
        currentSavings,
        monthlyExpenses,
        inflationAdjustedMonthlyExpenses,
        requiredCorpus,
        requiredMonthlySavings,
        expectedReturn,
        scenarios,
        recommendations: this.generateRetirementRecommendations(
          requiredMonthlySavings,
          yearsToRetirement,
          currentSavings
        )
      };

      return result;
    } catch (error) {
      logger.error('Error calculating retirement planning:', error);
      throw error;
    }
  }

  /**
   * Calculate SIP future value using the formula
   */
  calculateSIPFutureValue(monthlyAmount, months, annualReturn) {
    const monthlyRate = annualReturn / 12 / 100;
    
    if (monthlyRate === 0) {
      return monthlyAmount * months;
    }
    
    return monthlyAmount * ((Math.pow(1 + monthlyRate, months) - 1) / monthlyRate);
  }

  /**
   * Calculate required monthly investment for a goal
   */
  calculateRequiredMonthlyInvestment(goalAmount, currentSavings, months, expectedReturn) {
    const monthlyRate = expectedReturn / 12 / 100;
    const remainingAmount = goalAmount - currentSavings;
    
    if (remainingAmount <= 0) {
      return 0;
    }
    
    if (monthlyRate === 0) {
      return remainingAmount / months;
    }
    
    return remainingAmount * monthlyRate / (Math.pow(1 + monthlyRate, months) - 1);
  }

  /**
   * Adjust amount for inflation
   */
  adjustForInflation(amount, months) {
    const inflationRate = this.inflationRate / 12 / 100;
    return amount / Math.pow(1 + inflationRate, months);
  }

  /**
   * Calculate tax on capital gains
   */
  calculateTaxOnGains(gains, months) {
    if (gains <= 0) return 0;
    
    const years = months / 12;
    
    if (years <= 1) {
      // Short term capital gains
      return gains * (this.taxRates.shortTerm / 100);
    } else {
      // Long term capital gains (above 1L is taxable)
      const taxableAmount = Math.max(0, gains - 100000);
      return taxableAmount * (this.taxRates.longTerm / 100);
    }
  }

  /**
   * Get expected return for risk profile
   */
  getExpectedReturnForRiskProfile(riskProfile) {
    const returnRates = {
      conservative: 8,
      moderate: 12,
      aggressive: 16
    };
    
    return returnRates[riskProfile] || returnRates.moderate;
  }

  /**
   * Calculate investment scenarios
   */
  calculateInvestmentScenarios(goalAmount, currentSavings, months, riskProfile) {
    const scenarios = [];
    const riskProfiles = ['conservative', 'moderate', 'aggressive'];
    
    for (const profile of riskProfiles) {
      const expectedReturn = this.getExpectedReturnForRiskProfile(profile);
      const requiredMonthly = this.calculateRequiredMonthlyInvestment(
        goalAmount,
        currentSavings,
        months,
        expectedReturn
      );
      
      scenarios.push({
        riskProfile: profile,
        expectedReturn,
        requiredMonthlyInvestment: requiredMonthly,
        totalInvestment: requiredMonthly * months,
        projectedValue: this.calculateSIPFutureValue(requiredMonthly, months, expectedReturn)
      });
    }
    
    return scenarios;
  }

  /**
   * Calculate retirement corpus
   */
  calculateRetirementCorpus(annualExpenses, years, returnRate, inflationRate) {
    const realReturn = returnRate - inflationRate;
    const monthlyRealReturn = realReturn / 12 / 100;
    
    if (monthlyRealReturn === 0) {
      return annualExpenses * years;
    }
    
    return annualExpenses * (1 - Math.pow(1 + monthlyRealReturn, -years * 12)) / monthlyRealReturn;
  }

  /**
   * Calculate retirement scenarios
   */
  calculateRetirementScenarios(params) {
    const scenarios = [];
    const returnRates = [8, 10, 12, 14];
    
    for (const returnRate of returnRates) {
      const inflationAdjustedExpenses = this.adjustForInflation(
        params.monthlyExpenses * 12,
        params.retirementAge - params.currentAge
      );
      
      const requiredCorpus = this.calculateRetirementCorpus(
        inflationAdjustedExpenses,
        params.lifeExpectancy - params.retirementAge,
        returnRate,
        params.inflationRate
      );
      
      const requiredMonthly = this.calculateRequiredMonthlyInvestment(
        requiredCorpus,
        params.currentSavings,
        (params.retirementAge - params.currentAge) * 12,
        returnRate
      );
      
      scenarios.push({
        expectedReturn: returnRate,
        requiredCorpus,
        requiredMonthlySavings: requiredMonthly,
        totalSavings: requiredMonthly * (params.retirementAge - params.currentAge) * 12
      });
    }
    
    return scenarios;
  }

  /**
   * Generate goal recommendations
   */
  generateGoalRecommendations(requiredMonthly, riskProfile, months) {
    const recommendations = [];
    
    if (requiredMonthly > 50000) {
      recommendations.push({
        type: 'warning',
        message: 'Required monthly investment is high. Consider extending the timeline or adjusting the goal amount.',
        action: 'Review goal parameters'
      });
    }
    
    if (months < 12) {
      recommendations.push({
        type: 'warning',
        message: 'Short timeline detected. Consider longer investment horizon for better returns.',
        action: 'Extend timeline'
      });
    }
    
    if (riskProfile === 'conservative' && months > 60) {
      recommendations.push({
        type: 'suggestion',
        message: 'For long-term goals, consider moderate risk profile for better returns.',
        action: 'Review risk profile'
      });
    }
    
    return recommendations;
  }

  /**
   * Generate retirement recommendations
   */
  generateRetirementRecommendations(requiredMonthly, yearsToRetirement, currentSavings) {
    const recommendations = [];
    
    if (requiredMonthly > 100000) {
      recommendations.push({
        type: 'warning',
        message: 'High monthly savings required. Consider increasing current savings or adjusting retirement age.',
        action: 'Review retirement plan'
      });
    }
    
    if (currentSavings < 100000) {
      recommendations.push({
        type: 'warning',
        message: 'Low current savings. Start saving early to benefit from compound interest.',
        action: 'Increase current savings'
      });
    }
    
    if (yearsToRetirement < 10) {
      recommendations.push({
        type: 'warning',
        message: 'Limited time to retirement. Consider aggressive saving or extending retirement age.',
        action: 'Accelerate savings'
      });
    }
    
    return recommendations;
  }
}

module.exports = new InvestmentCalculatorService(); 