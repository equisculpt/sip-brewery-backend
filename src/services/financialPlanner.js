const logger = require('../utils/logger');
const { User, UserPortfolio, FinancialGoal, EmergencyFund, TaxDeadline, LifeEvent } = require('../models');
const ollamaService = require('./ollamaService');

class FinancialPlanner {
  constructor() {
    this.lifeStages = {
      EARLY_CAREER: {
        name: 'Early Career (22-30)',
        ageRange: [22, 30],
        priorities: ['emergency_fund', 'debt_repayment', 'basic_insurance', 'start_investing'],
        riskTolerance: 'moderate',
        timeHorizon: 'long'
      },
      ESTABLISHED_CAREER: {
        name: 'Established Career (31-45)',
        ageRange: [31, 45],
        priorities: ['goal_based_investing', 'insurance_adequacy', 'tax_optimization', 'retirement_planning'],
        riskTolerance: 'moderate_to_aggressive',
        timeHorizon: 'medium_to_long'
      },
      MID_CAREER: {
        name: 'Mid Career (46-55)',
        ageRange: [46, 55],
        priorities: ['retirement_planning', 'estate_planning', 'tax_efficiency', 'wealth_preservation'],
        riskTolerance: 'moderate',
        timeHorizon: 'medium'
      },
      PRE_RETIREMENT: {
        name: 'Pre-Retirement (56-60)',
        ageRange: [56, 60],
        priorities: ['retirement_readiness', 'debt_free', 'health_insurance', 'estate_planning'],
        riskTolerance: 'conservative',
        timeHorizon: 'short_to_medium'
      },
      RETIREMENT: {
        name: 'Retirement (60+)',
        ageRange: [60, 100],
        priorities: ['income_generation', 'wealth_preservation', 'health_care', 'legacy_planning'],
        riskTolerance: 'conservative',
        timeHorizon: 'short'
      }
    };

    this.goalTypes = {
      RETIREMENT: {
        name: 'Retirement',
        description: 'Build corpus for comfortable retirement',
        priority: 'high',
        timeHorizon: 'long',
        inflationAdjustment: true,
        taxConsideration: true
      },
      CHILD_EDUCATION: {
        name: 'Child Education',
        description: 'Fund children\'s education expenses',
        priority: 'high',
        timeHorizon: 'medium',
        inflationAdjustment: true,
        taxConsideration: true
      },
      EMERGENCY_FUND: {
        name: 'Emergency Fund',
        description: 'Maintain emergency fund for financial security',
        priority: 'critical',
        timeHorizon: 'immediate',
        inflationAdjustment: false,
        taxConsideration: false
      },
      HOME_PURCHASE: {
        name: 'Home Purchase',
        description: 'Save for down payment and home purchase',
        priority: 'high',
        timeHorizon: 'medium',
        inflationAdjustment: true,
        taxConsideration: true
      },
      VEHICLE_PURCHASE: {
        name: 'Vehicle Purchase',
        description: 'Save for vehicle purchase',
        priority: 'medium',
        timeHorizon: 'short',
        inflationAdjustment: true,
        taxConsideration: false
      },
      VACATION: {
        name: 'Vacation',
        description: 'Save for dream vacation',
        priority: 'low',
        timeHorizon: 'short',
        inflationAdjustment: true,
        taxConsideration: false
      },
      WEDDING: {
        name: 'Wedding',
        description: 'Save for wedding expenses',
        priority: 'high',
        timeHorizon: 'medium',
        inflationAdjustment: true,
        taxConsideration: false
      },
      BUSINESS_STARTUP: {
        name: 'Business Startup',
        description: 'Fund for business venture',
        priority: 'high',
        timeHorizon: 'medium',
        inflationAdjustment: true,
        taxConsideration: true
      }
    };

    this.inflationRates = {
      EDUCATION: 8.5, // Higher inflation for education
      HEALTHCARE: 7.5, // Healthcare inflation
      GENERAL: 6.0, // General inflation
      REAL_ESTATE: 5.5, // Real estate appreciation
      VEHICLE: 4.0, // Vehicle inflation
      VACATION: 5.0 // Travel inflation
    };

    this.taxDeadlines = {
      ELSS_INVESTMENT: {
        name: 'ELSS Investment for Tax Saving',
        deadline: 'March 31',
        description: 'Last date to invest in ELSS for current financial year tax deduction',
        amount: 150000,
        section: '80C'
      },
      NPS_CONTRIBUTION: {
        name: 'NPS Contribution',
        deadline: 'March 31',
        description: 'Last date for NPS contribution for tax deduction',
        amount: 50000,
        section: '80CCD(1B)'
      },
      HEALTH_INSURANCE: {
        name: 'Health Insurance Premium',
        deadline: 'March 31',
        description: 'Last date for health insurance premium payment',
        amount: 25000,
        section: '80D'
      },
      HOME_LOAN_EMI: {
        name: 'Home Loan EMI',
        deadline: 'March 31',
        description: 'Ensure home loan EMI payments for tax deduction',
        amount: 200000,
        section: '24(b)'
      }
    };
  }

  /**
   * Create comprehensive financial plan for user
   */
  async createFinancialPlan(userId) {
    try {
      logger.info('Creating financial plan', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      // Determine life stage
      const lifeStage = this.determineLifeStage(user.age);

      // Generate comprehensive plan using AGI
      const financialPlan = await this.generateFinancialPlan(user, userPortfolio, lifeStage);

      // Store the plan
      await this.storeFinancialPlan(userId, financialPlan);

      return {
        success: true,
        data: {
          financialPlan,
          lifeStage,
          nextReviewDate: this.calculateNextReviewDate(),
          recommendations: await this.getPriorityRecommendations(userId, financialPlan)
        }
      };
    } catch (error) {
      logger.error('Failed to create financial plan', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create financial plan',
        error: error.message
      };
    }
  }

  /**
   * Calculate retirement corpus requirement
   */
  async calculateRetirementCorpus(userId) {
    try {
      logger.info('Calculating retirement corpus', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const retirementData = await this.getRetirementData(user, userPortfolio);
      const corpusCalculation = this.calculateRetirementCorpusAmount(retirementData);

      // Store retirement goal
      await this.storeRetirementGoal(userId, corpusCalculation);

      return {
        success: true,
        data: {
          retirementCorpus: corpusCalculation,
          recommendations: await this.getRetirementRecommendations(userId, corpusCalculation),
          investmentStrategy: await this.getRetirementInvestmentStrategy(userId, corpusCalculation)
        }
      };
    } catch (error) {
      logger.error('Failed to calculate retirement corpus', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to calculate retirement corpus',
        error: error.message
      };
    }
  }

  /**
   * Plan child education funding
   */
  async planChildEducation(userId, childData) {
    try {
      logger.info('Planning child education', { userId, childCount: childData.length });

      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      const educationPlans = [];

      for (const child of childData) {
        const educationPlan = await this.calculateEducationCorpus(child, user);
        educationPlans.push(educationPlan);
      }

      // Store education goals
      await this.storeEducationGoals(userId, educationPlans);

      return {
        success: true,
        data: {
          educationPlans,
          totalCorpus: educationPlans.reduce((sum, plan) => sum + plan.totalCorpus, 0),
          monthlyInvestment: educationPlans.reduce((sum, plan) => sum + plan.monthlyInvestment, 0),
          recommendations: await this.getEducationRecommendations(userId, educationPlans)
        }
      };
    } catch (error) {
      logger.error('Failed to plan child education', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to plan child education',
        error: error.message
      };
    }
  }

  /**
   * Analyze emergency fund gap
   */
  async analyzeEmergencyFund(userId) {
    try {
      logger.info('Analyzing emergency fund', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const emergencyFundData = await this.getEmergencyFundData(user, userPortfolio);
      const gapAnalysis = this.calculateEmergencyFundGap(emergencyFundData);

      // Store emergency fund analysis
      await this.storeEmergencyFundAnalysis(userId, gapAnalysis);

      return {
        success: true,
        data: {
          emergencyFundAnalysis: gapAnalysis,
          recommendations: await this.getEmergencyFundRecommendations(userId, gapAnalysis),
          fundingStrategy: await this.getEmergencyFundStrategy(userId, gapAnalysis)
        }
      };
    } catch (error) {
      logger.error('Failed to analyze emergency fund', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to analyze emergency fund',
        error: error.message
      };
    }
  }

  /**
   * Get tax-saving deadline alerts
   */
  async getTaxDeadlineAlerts(userId) {
    try {
      logger.info('Getting tax deadline alerts', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const currentDate = new Date();
      const financialYear = this.getCurrentFinancialYear();
      const taxDeadlines = await this.calculateTaxDeadlines(user, userPortfolio, financialYear);

      // Store tax deadlines
      await this.storeTaxDeadlines(userId, taxDeadlines);

      return {
        success: true,
        data: {
          taxDeadlines,
          urgentAlerts: taxDeadlines.filter(d => d.daysRemaining <= 30),
          recommendations: await this.getTaxOptimizationRecommendations(userId, taxDeadlines)
        }
      };
    } catch (error) {
      logger.error('Failed to get tax deadline alerts', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get tax deadline alerts',
        error: error.message
      };
    }
  }

  /**
   * Track goal progress
   */
  async trackGoalProgress(userId, goalId) {
    try {
      logger.info('Tracking goal progress', { userId, goalId });

      const goal = await FinancialGoal.findById(goalId);
      if (!goal || goal.userId.toString() !== userId) {
        throw new Error('Goal not found');
      }

      const progress = await this.calculateGoalProgress(goal);
      const recommendations = await this.getGoalRecommendations(goal, progress);

      // Update goal progress
      goal.progress = progress;
      goal.lastUpdated = new Date();
      await goal.save();

      return {
        success: true,
        data: {
          goal,
          progress,
          recommendations,
          nextMilestone: this.getNextMilestone(goal, progress)
        }
      };
    } catch (error) {
      logger.error('Failed to track goal progress', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to track goal progress',
        error: error.message
      };
    }
  }

  /**
   * Get life event impact analysis
   */
  async analyzeLifeEventImpact(userId, lifeEvent) {
    try {
      logger.info('Analyzing life event impact', { userId, lifeEvent });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const impactAnalysis = await this.calculateLifeEventImpact(user, userPortfolio, lifeEvent);
      const recommendations = await this.getLifeEventRecommendations(userId, lifeEvent, impactAnalysis);

      // Store life event
      await this.storeLifeEvent(userId, lifeEvent, impactAnalysis);

      return {
        success: true,
        data: {
          lifeEvent,
          impactAnalysis,
          recommendations,
          actionPlan: await this.createLifeEventActionPlan(userId, lifeEvent, impactAnalysis)
        }
      };
    } catch (error) {
      logger.error('Failed to analyze life event impact', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to analyze life event impact',
        error: error.message
      };
    }
  }

  // Helper methods
  determineLifeStage(age) {
    for (const [stage, info] of Object.entries(this.lifeStages)) {
      if (age >= info.ageRange[0] && age <= info.ageRange[1]) {
        return stage;
      }
    }
    return 'EARLY_CAREER'; // Default
  }

  async generateFinancialPlan(user, userPortfolio, lifeStage) {
    const prompt = `
You are SipBrewery's AI Financial Planner. Create a comprehensive financial plan for an Indian investor.

USER PROFILE:
- Age: ${user.age}
- Income: ₹${user.income || 500000}/year
- Current Savings: ₹${userPortfolio?.totalValue || 0}
- Life Stage: ${this.lifeStages[lifeStage].name}
- Risk Profile: ${userPortfolio?.riskProfile || 'moderate'}

Create a comprehensive financial plan including:
1. Emergency Fund Requirement
2. Insurance Needs
3. Goal-Based Investment Plan
4. Tax Optimization Strategy
5. Retirement Planning
6. Risk Management

Format as JSON with detailed recommendations for each area.
    `;

    try {
      const ollamaResponse = await ollamaService.generateResponse(prompt, {
        model: 'mistral',
        temperature: 0.3,
        max_tokens: 2000
      });

      return this.parseFinancialPlanResponse(ollamaResponse);
    } catch (error) {
      logger.error('Failed to generate financial plan', { error: error.message });
      return this.getFallbackFinancialPlan(user, userPortfolio, lifeStage);
    }
  }

  parseFinancialPlanResponse(ollamaResponse) {
    try {
      const jsonMatch = ollamaResponse.match(/\{[\s\S]*\}/);
      if (!jsonMatch) {
        throw new Error('No valid JSON found in response');
      }

      return JSON.parse(jsonMatch[0]);
    } catch (error) {
      logger.error('Failed to parse financial plan response', { error: error.message });
      return this.getFallbackFinancialPlan();
    }
  }

  getFallbackFinancialPlan(user, userPortfolio, lifeStage) {
    return {
      emergencyFund: {
        required: user.income * 0.06, // 6 months of income
        current: userPortfolio?.liquidFunds || 0,
        gap: Math.max(0, user.income * 0.06 - (userPortfolio?.liquidFunds || 0))
      },
      insurance: {
        lifeInsurance: user.income * 10,
        healthInsurance: 500000,
        disabilityInsurance: user.income * 0.6
      },
      goals: [
        {
          type: 'RETIREMENT',
          targetAmount: user.income * 20,
          timeHorizon: 65 - user.age,
          monthlyInvestment: user.income * 0.15
        }
      ],
      taxOptimization: {
        elssInvestment: 150000,
        npsContribution: 50000,
        healthInsurance: 25000
      }
    };
  }

  async getRetirementData(user, userPortfolio) {
    const currentAge = user.age;
    const retirementAge = 60;
    const lifeExpectancy = 85;
    const currentIncome = user.income || 500000;
    const currentSavings = userPortfolio?.totalValue || 0;

    return {
      currentAge,
      retirementAge,
      lifeExpectancy,
      currentIncome,
      currentSavings,
      inflationRate: this.inflationRates.GENERAL,
      expectedReturn: 8.0,
      replacementRatio: 0.7 // 70% of current income
    };
  }

  calculateRetirementCorpusAmount(retirementData) {
    const {
      currentAge,
      retirementAge,
      lifeExpectancy,
      currentIncome,
      currentSavings,
      inflationRate,
      expectedReturn,
      replacementRatio
    } = retirementData;

    const yearsToRetirement = retirementAge - currentAge;
    const retirementYears = lifeExpectancy - retirementAge;

    // Calculate required monthly income at retirement
    const monthlyIncomeAtRetirement = (currentIncome * replacementRatio) / 12;
    
    // Adjust for inflation
    const inflatedMonthlyIncome = monthlyIncomeAtRetirement * Math.pow(1 + inflationRate / 100, yearsToRetirement);
    
    // Calculate corpus needed
    const corpusNeeded = inflatedMonthlyIncome * retirementYears * 12;
    
    // Calculate monthly investment needed
    const futureValueOfCurrentSavings = currentSavings * Math.pow(1 + expectedReturn / 100, yearsToRetirement);
    const additionalCorpusNeeded = Math.max(0, corpusNeeded - futureValueOfCurrentSavings);
    
    const monthlyInvestment = this.calculateMonthlyInvestment(additionalCorpusNeeded, yearsToRetirement, expectedReturn);

    return {
      corpusNeeded: Math.round(corpusNeeded),
      currentSavings: currentSavings,
      futureValueOfCurrentSavings: Math.round(futureValueOfCurrentSavings),
      additionalCorpusNeeded: Math.round(additionalCorpusNeeded),
      monthlyInvestment: Math.round(monthlyInvestment),
      yearsToRetirement,
      retirementYears
    };
  }

  calculateMonthlyInvestment(targetAmount, years, rate) {
    const monthlyRate = rate / 100 / 12;
    const months = years * 12;
    
    if (monthlyRate === 0) {
      return targetAmount / months;
    }
    
    return targetAmount * (monthlyRate / (Math.pow(1 + monthlyRate, months) - 1));
  }

  async calculateEducationCorpus(child, user) {
    const currentAge = child.age;
    const educationAge = child.educationAge || 18;
    const educationCost = child.educationCost || 2000000; // ₹20 lakhs
    const inflationRate = this.inflationRates.EDUCATION;
    const expectedReturn = 8.0;

    const yearsToEducation = educationAge - currentAge;
    const inflatedCost = educationCost * Math.pow(1 + inflationRate / 100, yearsToEducation);
    const monthlyInvestment = this.calculateMonthlyInvestment(inflatedCost, yearsToEducation, expectedReturn);

    return {
      childName: child.name,
      currentAge,
      educationAge,
      educationCost,
      inflatedCost: Math.round(inflatedCost),
      monthlyInvestment: Math.round(monthlyInvestment),
      yearsToEducation
    };
  }

  async getEmergencyFundData(user, userPortfolio) {
    const monthlyIncome = (user.income || 500000) / 12;
    const monthlyExpenses = monthlyIncome * 0.7; // Assume 70% of income as expenses
    const currentEmergencyFund = userPortfolio?.liquidFunds || 0;
    const requiredEmergencyFund = monthlyExpenses * 6; // 6 months of expenses

    return {
      monthlyIncome,
      monthlyExpenses,
      currentEmergencyFund,
      requiredEmergencyFund,
      gap: Math.max(0, requiredEmergencyFund - currentEmergencyFund)
    };
  }

  calculateEmergencyFundGap(emergencyFundData) {
    const { currentEmergencyFund, requiredEmergencyFund, gap } = emergencyFundData;
    const adequacy = (currentEmergencyFund / requiredEmergencyFund) * 100;

    return {
      currentFund: currentEmergencyFund,
      requiredFund: requiredEmergencyFund,
      gap,
      adequacy: Math.round(adequacy),
      status: adequacy >= 100 ? 'adequate' : adequacy >= 50 ? 'moderate' : 'inadequate',
      priority: adequacy >= 100 ? 'low' : adequacy >= 50 ? 'medium' : 'high'
    };
  }

  getCurrentFinancialYear() {
    const currentDate = new Date();
    const currentYear = currentDate.getFullYear();
    const currentMonth = currentDate.getMonth() + 1;
    
    // Financial year runs from April to March
    if (currentMonth >= 4) {
      return `${currentYear}-${currentYear + 1}`;
    } else {
      return `${currentYear - 1}-${currentYear}`;
    }
  }

  async calculateTaxDeadlines(user, userPortfolio, financialYear) {
    const currentDate = new Date();
    const deadlines = [];

    for (const [key, deadline] of Object.entries(this.taxDeadlines)) {
      const deadlineDate = new Date(`${financialYear.split('-')[1]}-03-31`);
      const daysRemaining = Math.ceil((deadlineDate - currentDate) / (1000 * 60 * 60 * 24));

      deadlines.push({
        type: key,
        name: deadline.name,
        deadline: deadline.deadline,
        description: deadline.description,
        amount: deadline.amount,
        section: deadline.section,
        daysRemaining,
        urgency: daysRemaining <= 30 ? 'high' : daysRemaining <= 60 ? 'medium' : 'low'
      });
    }

    return deadlines.sort((a, b) => a.daysRemaining - b.daysRemaining);
  }

  async calculateGoalProgress(goal) {
    const currentValue = goal.currentValue || 0;
    const targetValue = goal.targetAmount;
    const progress = (currentValue / targetValue) * 100;
    const remainingAmount = Math.max(0, targetValue - currentValue);

    return {
      currentValue,
      targetValue,
      progress: Math.round(progress),
      remainingAmount,
      status: progress >= 100 ? 'completed' : progress >= 75 ? 'on_track' : progress >= 50 ? 'moderate' : 'needs_attention'
    };
  }

  async getGoalRecommendations(goal, progress) {
    const recommendations = [];

    if (progress.status === 'needs_attention') {
      recommendations.push('Consider increasing monthly investment to meet goal timeline');
    }

    if (progress.status === 'on_track') {
      recommendations.push('Continue current investment strategy');
    }

    return recommendations;
  }

  getNextMilestone(goal, progress) {
    const milestones = [25, 50, 75, 100];
    const nextMilestone = milestones.find(m => m > progress.progress);
    
    if (!nextMilestone) {
      return { milestone: 100, status: 'completed' };
    }

    const amountForMilestone = (goal.targetAmount * nextMilestone) / 100;
    const remainingForMilestone = Math.max(0, amountForMilestone - progress.currentValue);

    return {
      milestone: nextMilestone,
      amountForMilestone,
      remainingForMilestone,
      estimatedMonths: Math.ceil(remainingForMilestone / (goal.monthlyInvestment || 10000))
    };
  }

  async calculateLifeEventImpact(user, userPortfolio, lifeEvent) {
    const impact = {
      financial: {
        incomeChange: 0,
        expenseChange: 0,
        savingsImpact: 0
      },
      investment: {
        riskProfileChange: 'none',
        timeHorizonChange: 'none',
        allocationAdjustment: 'none'
      },
      insurance: {
        coverageChange: 'none',
        premiumChange: 0
      }
    };

    switch (lifeEvent.type) {
      case 'marriage':
        impact.financial.expenseChange = user.income * 0.1; // 10% increase in expenses
        impact.insurance.coverageChange = 'increase';
        break;
      case 'child_birth':
        impact.financial.expenseChange = user.income * 0.15; // 15% increase in expenses
        impact.investment.timeHorizonChange = 'extend';
        break;
      case 'job_change':
        impact.financial.incomeChange = user.income * 0.2; // 20% income change
        break;
      case 'home_purchase':
        impact.financial.expenseChange = user.income * 0.3; // 30% increase in expenses
        impact.investment.riskProfileChange = 'reduce';
        break;
    }

    return impact;
  }

  async getLifeEventRecommendations(userId, lifeEvent, impact) {
    const recommendations = [];

    if (impact.financial.expenseChange > 0) {
      recommendations.push('Review and adjust monthly budget to accommodate new expenses');
    }

    if (impact.insurance.coverageChange === 'increase') {
      recommendations.push('Consider increasing life and health insurance coverage');
    }

    if (impact.investment.riskProfileChange === 'reduce') {
      recommendations.push('Consider reducing portfolio risk to accommodate new financial obligations');
    }

    return recommendations;
  }

  async createLifeEventActionPlan(userId, lifeEvent, impact) {
    return {
      immediate: [
        'Review current financial plan',
        'Update budget and expense tracking',
        'Assess insurance adequacy'
      ],
      shortTerm: [
        'Adjust investment allocation if needed',
        'Update financial goals',
        'Review emergency fund adequacy'
      ],
      longTerm: [
        'Monitor impact on retirement planning',
        'Update estate planning documents',
        'Review tax optimization strategies'
      ]
    };
  }

  calculateNextReviewDate() {
    const nextReview = new Date();
    nextReview.setMonth(nextReview.getMonth() + 3); // Review every 3 months
    return nextReview.toISOString();
  }

  async getPriorityRecommendations(userId, financialPlan) {
    return [
      'Build emergency fund to 6 months of expenses',
      'Start retirement planning early',
      'Optimize tax-saving investments',
      'Review insurance coverage adequacy'
    ];
  }

  async getRetirementRecommendations(userId, corpusCalculation) {
    return [
      `Invest ₹${corpusCalculation.monthlyInvestment.toLocaleString()} monthly for retirement`,
      'Consider increasing investment amount if possible',
      'Review retirement plan annually',
      'Diversify retirement portfolio'
    ];
  }

  async getRetirementInvestmentStrategy(userId, corpusCalculation) {
    return {
      assetAllocation: {
        equity: 70,
        debt: 20,
        gold: 10
      },
      fundTypes: [
        'Large-cap equity funds',
        'Multi-cap equity funds',
        'Corporate bond funds',
        'Gold ETFs'
      ],
      rebalancing: 'Annual',
      riskManagement: 'Gradual shift to debt as retirement approaches'
    };
  }

  async getEducationRecommendations(userId, educationPlans) {
    return [
      'Start education planning early',
      'Consider education-specific mutual funds',
      'Review education costs annually',
      'Plan for multiple children if applicable'
    ];
  }

  async getEmergencyFundRecommendations(userId, gapAnalysis) {
    const recommendations = [];

    if (gapAnalysis.status === 'inadequate') {
      recommendations.push('Prioritize building emergency fund before other investments');
    }

    recommendations.push('Keep emergency fund in liquid instruments like savings account or liquid funds');
    recommendations.push('Review emergency fund adequacy annually');

    return recommendations;
  }

  async getEmergencyFundStrategy(userId, gapAnalysis) {
    return {
      fundingPriority: gapAnalysis.priority === 'high' ? 'immediate' : 'gradual',
      monthlyContribution: Math.ceil(gapAnalysis.gap / 12),
      targetTimeline: gapAnalysis.priority === 'high' ? '3 months' : '6 months',
      instruments: ['Savings Account', 'Liquid Funds', 'Ultra Short-term Funds']
    };
  }

  async getTaxOptimizationRecommendations(userId, taxDeadlines) {
    return [
      'Maximize ELSS investment for Section 80C deduction',
      'Consider NPS contribution for additional tax benefit',
      'Ensure health insurance premium payment before deadline',
      'Review home loan EMI payments for tax deduction'
    ];
  }

  // Storage methods
  async storeFinancialPlan(userId, financialPlan) {
    // Implementation to store financial plan
    logger.info('Storing financial plan', { userId });
  }

  async storeRetirementGoal(userId, corpusCalculation) {
    const retirementGoal = new FinancialGoal({
      userId,
      type: 'RETIREMENT',
      targetAmount: corpusCalculation.corpusNeeded,
      currentValue: corpusCalculation.currentSavings,
      monthlyInvestment: corpusCalculation.monthlyInvestment,
      timeHorizon: corpusCalculation.yearsToRetirement,
      priority: 'high'
    });

    await retirementGoal.save();
  }

  async storeEducationGoals(userId, educationPlans) {
    for (const plan of educationPlans) {
      const educationGoal = new FinancialGoal({
        userId,
        type: 'CHILD_EDUCATION',
        targetAmount: plan.inflatedCost,
        monthlyInvestment: plan.monthlyInvestment,
        timeHorizon: plan.yearsToEducation,
        priority: 'high',
        metadata: { childName: plan.childName }
      });

      await educationGoal.save();
    }
  }

  async storeEmergencyFundAnalysis(userId, gapAnalysis) {
    const emergencyFund = new EmergencyFund({
      userId,
      currentAmount: gapAnalysis.currentFund,
      requiredAmount: gapAnalysis.requiredFund,
      gap: gapAnalysis.gap,
      adequacy: gapAnalysis.adequacy,
      status: gapAnalysis.status,
      priority: gapAnalysis.priority
    });

    await emergencyFund.save();
  }

  async storeTaxDeadlines(userId, taxDeadlines) {
    for (const deadline of taxDeadlines) {
      const taxDeadline = new TaxDeadline({
        userId,
        type: deadline.type,
        name: deadline.name,
        deadline: deadline.deadline,
        amount: deadline.amount,
        section: deadline.section,
        daysRemaining: deadline.daysRemaining,
        urgency: deadline.urgency
      });

      await taxDeadline.save();
    }
  }

  async storeLifeEvent(userId, lifeEvent, impactAnalysis) {
    const lifeEventRecord = new LifeEvent({
      userId,
      type: lifeEvent.type,
      date: lifeEvent.date,
      description: lifeEvent.description,
      impact: impactAnalysis,
      status: 'active'
    });

    await lifeEventRecord.save();
  }
}

module.exports = new FinancialPlanner(); 