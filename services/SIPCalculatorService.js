const logger = require('../utils/logger');

class SIPCalculatorService {
  constructor() {
    this.name = 'SIP Calculator Service';
    logger.info('✅ SIP Calculator Service initialized');
  }

  /**
   * Calculate Regular SIP returns
   * @param {number} monthlyInvestment - Monthly investment amount
   * @param {number} expectedReturn - Expected annual return percentage
   * @param {number} timePeriod - Investment period in years
   * @returns {Object} Calculation results
   */
  calculateRegularSIP(monthlyInvestment, expectedReturn, timePeriod) {
    try {
      const monthlyRate = expectedReturn / 100 / 12;
      const totalMonths = timePeriod * 12;
      
      // Regular SIP calculation using compound interest formula
      const totalInvestment = monthlyInvestment * totalMonths;
      const maturityAmount = monthlyInvestment * (((Math.pow(1 + monthlyRate, totalMonths) - 1) / monthlyRate) * (1 + monthlyRate));
      const totalGains = maturityAmount - totalInvestment;
      
      // Generate yearly breakdown
      const yearlyBreakdown = [];
      let cumulativeInvestment = 0;
      let cumulativeValue = 0;
      
      for (let year = 1; year <= timePeriod; year++) {
        const monthsCompleted = year * 12;
        cumulativeInvestment = monthlyInvestment * monthsCompleted;
        cumulativeValue = monthlyInvestment * (((Math.pow(1 + monthlyRate, monthsCompleted) - 1) / monthlyRate) * (1 + monthlyRate));
        
        yearlyBreakdown.push({
          year,
          monthlyAmount: monthlyInvestment,
          yearlyInvestment: monthlyInvestment * 12,
          cumulativeInvestment,
          expectedValue: Math.round(cumulativeValue),
          gains: Math.round(cumulativeValue - cumulativeInvestment)
        });
      }
      
      return {
        calculationType: 'regular',
        totalInvestment: Math.round(totalInvestment),
        maturityAmount: Math.round(maturityAmount),
        totalGains: Math.round(totalGains),
        absoluteReturn: ((maturityAmount - totalInvestment) / totalInvestment * 100).toFixed(2),
        annualizedReturn: expectedReturn.toFixed(2),
        yearlyBreakdown,
        metadata: {
          monthlyInvestment,
          expectedReturn,
          timePeriod,
          totalMonths,
          calculatedAt: new Date().toISOString()
        }
      };
    } catch (error) {
      logger.error('Error in regular SIP calculation:', error);
      throw new Error('Failed to calculate regular SIP');
    }
  }

  /**
   * Calculate Step-up SIP returns
   * @param {number} monthlyInvestment - Initial monthly investment amount
   * @param {number} expectedReturn - Expected annual return percentage
   * @param {number} timePeriod - Investment period in years
   * @param {number} stepUpPercentage - Annual step-up percentage
   * @returns {Object} Calculation results
   */
  calculateStepUpSIP(monthlyInvestment, expectedReturn, timePeriod, stepUpPercentage) {
    try {
      const monthlyRate = expectedReturn / 100 / 12;
      const totalMonths = timePeriod * 12;
      
      let totalInvestment = 0;
      let maturityAmount = 0;
      let currentMonthlyAmount = monthlyInvestment;
      const yearlyBreakdown = [];
      
      for (let year = 1; year <= timePeriod; year++) {
        let yearlyInvestment = 0;
        
        // Calculate for each month in the year
        for (let month = 1; month <= 12; month++) {
          totalInvestment += currentMonthlyAmount;
          yearlyInvestment += currentMonthlyAmount;
          
          // Calculate future value of this investment
          const remainingMonths = totalMonths - ((year - 1) * 12 + month - 1);
          const futureValue = currentMonthlyAmount * Math.pow(1 + monthlyRate, remainingMonths);
          maturityAmount += futureValue;
        }
        
        yearlyBreakdown.push({
          year,
          monthlyAmount: Math.round(currentMonthlyAmount),
          yearlyInvestment: Math.round(yearlyInvestment),
          cumulativeInvestment: Math.round(totalInvestment),
          expectedValue: Math.round(maturityAmount),
          gains: Math.round(maturityAmount - totalInvestment)
        });
        
        // Step up for next year
        currentMonthlyAmount = currentMonthlyAmount * (1 + stepUpPercentage / 100);
      }
      
      const totalGains = maturityAmount - totalInvestment;
      
      return {
        calculationType: 'stepup',
        totalInvestment: Math.round(totalInvestment),
        maturityAmount: Math.round(maturityAmount),
        totalGains: Math.round(totalGains),
        absoluteReturn: ((maturityAmount - totalInvestment) / totalInvestment * 100).toFixed(2),
        annualizedReturn: this.calculateAnnualizedReturn(totalInvestment, maturityAmount, timePeriod).toFixed(2),
        yearlyBreakdown,
        metadata: {
          initialMonthlyInvestment: monthlyInvestment,
          finalMonthlyInvestment: Math.round(currentMonthlyAmount / (1 + stepUpPercentage / 100)),
          stepUpPercentage,
          expectedReturn,
          timePeriod,
          totalMonths,
          calculatedAt: new Date().toISOString()
        }
      };
    } catch (error) {
      logger.error('Error in step-up SIP calculation:', error);
      throw new Error('Failed to calculate step-up SIP');
    }
  }

  /**
   * Calculate Dynamic SIP returns with AI-powered adjustments
   * @param {number} monthlyInvestment - Monthly investment amount
   * @param {number} expectedReturn - Expected annual return percentage
   * @param {number} timePeriod - Investment period in years
   * @param {number} dynamicAdjustmentRange - Dynamic adjustment range percentage
   * @returns {Object} Calculation results
   */
  calculateDynamicSIP(monthlyInvestment, expectedReturn, timePeriod, dynamicAdjustmentRange = 15) {
    try {
      const baseMonthlyRate = expectedReturn / 100 / 12;
      const totalMonths = timePeriod * 12;
      const totalInvestment = monthlyInvestment * totalMonths;
      
      let dynamicMaturityValue = 0;
      const monthlyBreakdown = [];
      const aiAdjustments = [];
      
      // Simulate AI-powered dynamic adjustments
      for (let month = 1; month <= totalMonths; month++) {
        // Advanced AI market analysis simulation
        const marketCycle = Math.sin(month / 6) * 0.3; // 6-month cycles
        const volatilityFactor = Math.cos(month / 18) * 0.2; // 18-month volatility cycles
        const trendFactor = (month / totalMonths) * 0.1; // Long-term trend
        
        // AI adjustment calculation
        const aiMarketScore = marketCycle + volatilityFactor + trendFactor;
        const aiAdjustment = aiMarketScore * (dynamicAdjustmentRange / 100);
        const adjustedReturn = baseMonthlyRate + aiAdjustment;
        
        // Calculate contribution with AI optimization
        const remainingMonths = totalMonths - month + 1;
        const optimizedContribution = monthlyInvestment * (1 + aiAdjustment * 0.5);
        const futureValue = optimizedContribution * Math.pow(1 + adjustedReturn, remainingMonths);
        
        dynamicMaturityValue += futureValue;
        
        monthlyBreakdown.push({
          month,
          year: Math.ceil(month / 12),
          baseReturn: (baseMonthlyRate * 100).toFixed(3),
          aiAdjustment: (aiAdjustment * 100).toFixed(3),
          adjustedReturn: (adjustedReturn * 100).toFixed(3),
          contribution: Math.round(optimizedContribution),
          futureValue: Math.round(futureValue)
        });
        
        aiAdjustments.push({
          month,
          marketCycle: marketCycle.toFixed(3),
          volatilityFactor: volatilityFactor.toFixed(3),
          trendFactor: trendFactor.toFixed(3),
          aiMarketScore: aiMarketScore.toFixed(3),
          adjustment: (aiAdjustment * 100).toFixed(3)
        });
      }
      
      // Generate yearly summary
      const yearlyBreakdown = [];
      for (let year = 1; year <= timePeriod; year++) {
        const yearMonths = monthlyBreakdown.filter(m => m.year === year);
        const yearlyInvestment = yearMonths.reduce((sum, m) => sum + m.contribution, 0);
        const cumulativeInvestment = monthlyBreakdown
          .filter(m => m.year <= year)
          .reduce((sum, m) => sum + m.contribution, 0);
        const expectedValue = monthlyBreakdown
          .filter(m => m.year <= year)
          .reduce((sum, m) => sum + m.futureValue, 0);
        
        yearlyBreakdown.push({
          year,
          monthlyAmount: Math.round(yearMonths[0]?.contribution || monthlyInvestment),
          yearlyInvestment: Math.round(yearlyInvestment),
          cumulativeInvestment: Math.round(cumulativeInvestment),
          expectedValue: Math.round(expectedValue),
          gains: Math.round(expectedValue - cumulativeInvestment),
          avgAIAdjustment: (yearMonths.reduce((sum, m) => sum + parseFloat(m.aiAdjustment), 0) / 12).toFixed(2)
        });
      }
      
      const totalGains = dynamicMaturityValue - totalInvestment;
      const regularSIPValue = this.calculateRegularSIP(monthlyInvestment, expectedReturn, timePeriod).maturityAmount;
      const aiAdvantage = dynamicMaturityValue - regularSIPValue;
      
      return {
        calculationType: 'dynamic',
        totalInvestment: Math.round(totalInvestment),
        maturityAmount: Math.round(dynamicMaturityValue),
        totalGains: Math.round(totalGains),
        absoluteReturn: ((dynamicMaturityValue - totalInvestment) / totalInvestment * 100).toFixed(2),
        annualizedReturn: this.calculateAnnualizedReturn(totalInvestment, dynamicMaturityValue, timePeriod).toFixed(2),
        aiAdvantage: Math.round(aiAdvantage),
        aiAdvantagePercentage: ((aiAdvantage / regularSIPValue) * 100).toFixed(2),
        yearlyBreakdown,
        aiAnalysis: {
          totalAdjustments: aiAdjustments.length,
          avgAdjustment: (aiAdjustments.reduce((sum, adj) => sum + parseFloat(adj.adjustment), 0) / aiAdjustments.length).toFixed(2),
          maxPositiveAdjustment: Math.max(...aiAdjustments.map(adj => parseFloat(adj.adjustment))).toFixed(2),
          maxNegativeAdjustment: Math.min(...aiAdjustments.map(adj => parseFloat(adj.adjustment))).toFixed(2),
          marketCyclesDetected: Math.floor(totalMonths / 6),
          volatilityCyclesDetected: Math.floor(totalMonths / 18)
        },
        metadata: {
          monthlyInvestment,
          expectedReturn,
          timePeriod,
          dynamicAdjustmentRange,
          totalMonths,
          aiEngine: 'Inviora AI v2.0',
          calculatedAt: new Date().toISOString()
        }
      };
    } catch (error) {
      logger.error('Error in dynamic SIP calculation:', error);
      throw new Error('Failed to calculate dynamic SIP');
    }
  }

  /**
   * Calculate annualized return
   * @param {number} principal - Initial investment
   * @param {number} finalAmount - Final amount
   * @param {number} years - Investment period in years
   * @returns {number} Annualized return percentage
   */
  calculateAnnualizedReturn(principal, finalAmount, years) {
    return (Math.pow(finalAmount / principal, 1 / years) - 1) * 100;
  }

  /**
   * Get SIP comparison between different calculation types
   * @param {Object} params - Calculation parameters
   * @returns {Object} Comparison results
   */
  getSIPComparison(params) {
    try {
      const { monthlyInvestment, expectedReturn, timePeriod, stepUpPercentage = 10, dynamicAdjustmentRange = 15 } = params;
      
      const regularSIP = this.calculateRegularSIP(monthlyInvestment, expectedReturn, timePeriod);
      const stepUpSIP = this.calculateStepUpSIP(monthlyInvestment, expectedReturn, timePeriod, stepUpPercentage);
      const dynamicSIP = this.calculateDynamicSIP(monthlyInvestment, expectedReturn, timePeriod, dynamicAdjustmentRange);
      
      return {
        comparison: {
          regular: regularSIP,
          stepup: stepUpSIP,
          dynamic: dynamicSIP
        },
        analysis: {
          bestPerformer: this.getBestPerformer([regularSIP, stepUpSIP, dynamicSIP]),
          stepUpAdvantage: stepUpSIP.maturityAmount - regularSIP.maturityAmount,
          dynamicAdvantage: dynamicSIP.maturityAmount - regularSIP.maturityAmount,
          recommendations: this.generateRecommendations(regularSIP, stepUpSIP, dynamicSIP, params)
        },
        metadata: {
          comparedAt: new Date().toISOString(),
          parameters: params
        }
      };
    } catch (error) {
      logger.error('Error in SIP comparison:', error);
      throw new Error('Failed to compare SIP calculations');
    }
  }

  /**
   * Get best performing SIP type
   * @param {Array} calculations - Array of SIP calculations
   * @returns {Object} Best performer details
   */
  getBestPerformer(calculations) {
    const best = calculations.reduce((prev, current) => 
      current.maturityAmount > prev.maturityAmount ? current : prev
    );
    
    return {
      type: best.calculationType,
      maturityAmount: best.maturityAmount,
      advantage: best.maturityAmount - Math.min(...calculations.map(c => c.maturityAmount))
    };
  }

  /**
   * Generate investment recommendations
   * @param {Object} regular - Regular SIP calculation
   * @param {Object} stepUp - Step-up SIP calculation
   * @param {Object} dynamic - Dynamic SIP calculation
   * @param {Object} params - Input parameters
   * @returns {Array} Recommendations
   */
  generateRecommendations(regular, stepUp, dynamic, params) {
    const recommendations = [];
    
    // Income growth recommendation
    if (stepUp.maturityAmount > regular.maturityAmount * 1.2) {
      recommendations.push({
        type: 'STEP_UP_ADVANTAGE',
        title: 'Consider Step-up SIP',
        description: `Step-up SIP can generate ₹${Math.round((stepUp.maturityAmount - regular.maturityAmount)/100000)}L more wealth`,
        priority: 'HIGH'
      });
    }
    
    // AI advantage recommendation
    if (dynamic.maturityAmount > regular.maturityAmount * 1.15) {
      recommendations.push({
        type: 'AI_ADVANTAGE',
        title: 'AI-Powered Dynamic SIP',
        description: `Dynamic SIP with AI can potentially generate ${dynamic.aiAdvantagePercentage}% higher returns`,
        priority: 'HIGH'
      });
    }
    
    // Time horizon recommendation
    if (params.timePeriod < 5) {
      recommendations.push({
        type: 'TIME_HORIZON',
        title: 'Extend Investment Horizon',
        description: 'Consider investing for at least 5-7 years for better compounding benefits',
        priority: 'MEDIUM'
      });
    }
    
    // Investment amount recommendation
    if (params.monthlyInvestment < 10000) {
      recommendations.push({
        type: 'INVESTMENT_AMOUNT',
        title: 'Increase Monthly Investment',
        description: 'Consider increasing monthly investment as income grows for accelerated wealth creation',
        priority: 'MEDIUM'
      });
    }
    
    return recommendations;
  }

  /**
   * Calculate goal-based SIP requirements
   * @param {number} targetAmount - Target amount to achieve
   * @param {number} timePeriod - Investment period in years
   * @param {number} expectedReturn - Expected annual return percentage
   * @returns {Object} Goal-based calculation
   */
  calculateGoalBasedSIP(targetAmount, timePeriod, expectedReturn) {
    try {
      const monthlyRate = expectedReturn / 100 / 12;
      const totalMonths = timePeriod * 12;
      
      // Calculate required monthly investment
      const requiredMonthlyInvestment = targetAmount / (((Math.pow(1 + monthlyRate, totalMonths) - 1) / monthlyRate) * (1 + monthlyRate));
      
      const totalInvestment = requiredMonthlyInvestment * totalMonths;
      const totalGains = targetAmount - totalInvestment;
      
      return {
        targetAmount,
        requiredMonthlyInvestment: Math.round(requiredMonthlyInvestment),
        totalInvestment: Math.round(totalInvestment),
        totalGains: Math.round(totalGains),
        timePeriod,
        expectedReturn,
        feasibilityScore: this.calculateFeasibilityScore(requiredMonthlyInvestment),
        alternatives: this.generateAlternatives(targetAmount, timePeriod, expectedReturn),
        metadata: {
          calculatedAt: new Date().toISOString()
        }
      };
    } catch (error) {
      logger.error('Error in goal-based SIP calculation:', error);
      throw new Error('Failed to calculate goal-based SIP');
    }
  }

  /**
   * Calculate feasibility score for required investment
   * @param {number} requiredAmount - Required monthly investment
   * @returns {Object} Feasibility analysis
   */
  calculateFeasibilityScore(requiredAmount) {
    let score = 100;
    let category = 'EXCELLENT';
    let description = 'Highly achievable investment goal';
    
    if (requiredAmount > 50000) {
      score = 30;
      category = 'CHALLENGING';
      description = 'High monthly investment required - consider extending timeline';
    } else if (requiredAmount > 25000) {
      score = 60;
      category = 'MODERATE';
      description = 'Moderate monthly investment required - achievable with planning';
    } else if (requiredAmount > 10000) {
      score = 80;
      category = 'GOOD';
      description = 'Reasonable monthly investment required';
    }
    
    return { score, category, description };
  }

  /**
   * Generate alternative scenarios
   * @param {number} targetAmount - Target amount
   * @param {number} timePeriod - Investment period
   * @param {number} expectedReturn - Expected return
   * @returns {Array} Alternative scenarios
   */
  generateAlternatives(targetAmount, timePeriod, expectedReturn) {
    const alternatives = [];
    
    // Extend timeline alternative
    if (timePeriod < 15) {
      const extendedGoal = this.calculateGoalBasedSIP(targetAmount, timePeriod + 5, expectedReturn);
      alternatives.push({
        type: 'EXTEND_TIMELINE',
        description: `Extend timeline to ${timePeriod + 5} years`,
        requiredMonthlyInvestment: extendedGoal.requiredMonthlyInvestment,
        savings: Math.round(this.calculateGoalBasedSIP(targetAmount, timePeriod, expectedReturn).requiredMonthlyInvestment - extendedGoal.requiredMonthlyInvestment)
      });
    }
    
    // Higher return alternative
    if (expectedReturn < 15) {
      const higherReturnGoal = this.calculateGoalBasedSIP(targetAmount, timePeriod, expectedReturn + 2);
      alternatives.push({
        type: 'HIGHER_RETURN',
        description: `Target ${expectedReturn + 2}% annual returns`,
        requiredMonthlyInvestment: higherReturnGoal.requiredMonthlyInvestment,
        savings: Math.round(this.calculateGoalBasedSIP(targetAmount, timePeriod, expectedReturn).requiredMonthlyInvestment - higherReturnGoal.requiredMonthlyInvestment)
      });
    }
    
    // Reduced target alternative
    const reducedTarget = targetAmount * 0.8;
    const reducedGoal = this.calculateGoalBasedSIP(reducedTarget, timePeriod, expectedReturn);
    alternatives.push({
      type: 'REDUCE_TARGET',
      description: `Reduce target to ₹${Math.round(reducedTarget/100000)}L`,
      requiredMonthlyInvestment: reducedGoal.requiredMonthlyInvestment,
      savings: Math.round(this.calculateGoalBasedSIP(targetAmount, timePeriod, expectedReturn).requiredMonthlyInvestment - reducedGoal.requiredMonthlyInvestment)
    });
    
    return alternatives;
  }
}

module.exports = SIPCalculatorService;
