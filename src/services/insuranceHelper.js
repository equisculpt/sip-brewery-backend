const logger = require('../utils/logger');
const { User, UserPortfolio, InsurancePolicy, InsuranceRecommendation, FamilyMember } = require('../models');
const ollamaService = require('./ollamaService');

class InsuranceHelper {
  constructor() {
    this.insuranceTypes = {
      LIFE_INSURANCE: {
        name: 'Life Insurance',
        description: 'Financial protection for family in case of death',
        categories: ['term_insurance', 'endowment', 'whole_life', 'ulip'],
        priority: 'high',
        mandatory: true
      },
      HEALTH_INSURANCE: {
        name: 'Health Insurance',
        description: 'Coverage for medical expenses and hospitalization',
        categories: ['individual', 'family_floater', 'senior_citizen', 'critical_illness'],
        priority: 'high',
        mandatory: true
      },
      DISABILITY_INSURANCE: {
        name: 'Disability Insurance',
        description: 'Income replacement in case of disability',
        categories: ['short_term', 'long_term', 'permanent'],
        priority: 'medium',
        mandatory: false
      },
      CRITICAL_ILLNESS: {
        name: 'Critical Illness',
        description: 'Lump sum payment for critical illnesses',
        categories: ['individual', 'family'],
        priority: 'medium',
        mandatory: false
      },
      ACCIDENT_INSURANCE: {
        name: 'Accident Insurance',
        description: 'Coverage for accidental death and disability',
        categories: ['personal_accident', 'travel_accident'],
        priority: 'low',
        mandatory: false
      }
    };

    this.coverageCalculators = {
      LIFE_INSURANCE: {
        method: 'income_multiple',
        baseMultiplier: 10,
        factors: {
          age: {
            '18-30': 1.0,
            '31-40': 1.2,
            '41-50': 1.5,
            '51-60': 2.0,
            '60+': 2.5
          },
          familySize: {
            1: 0.8,
            2: 1.0,
            3: 1.3,
            4: 1.5,
            '5+': 1.8
          },
          liabilities: {
            none: 1.0,
            low: 1.2,
            medium: 1.5,
            high: 2.0
          }
        }
      },
      HEALTH_INSURANCE: {
        method: 'medical_cost',
        baseAmount: 500000,
        factors: {
          age: {
            '18-30': 1.0,
            '31-40': 1.3,
            '41-50': 1.8,
            '51-60': 2.5,
            '60+': 3.0
          },
          familySize: {
            1: 1.0,
            2: 1.5,
            3: 2.0,
            4: 2.5,
            '5+': 3.0
          },
          location: {
            'tier1': 1.5,
            'tier2': 1.2,
            'tier3': 1.0
          }
        }
      },
      DISABILITY_INSURANCE: {
        method: 'income_replacement',
        basePercentage: 60,
        factors: {
          occupation: {
            'low_risk': 0.8,
            'medium_risk': 1.0,
            'high_risk': 1.5
          },
          age: {
            '18-30': 1.0,
            '31-40': 1.2,
            '41-50': 1.5,
            '51-60': 2.0
          }
        }
      }
    };

    this.premiumFactors = {
      age: {
        '18-25': 1.0,
        '26-30': 1.2,
        '31-35': 1.5,
        '36-40': 1.8,
        '41-45': 2.2,
        '46-50': 2.8,
        '51-55': 3.5,
        '56-60': 4.5,
        '60+': 6.0
      },
      health: {
        'excellent': 1.0,
        'good': 1.2,
        'average': 1.5,
        'poor': 2.0
      },
      lifestyle: {
        'non_smoker': 1.0,
        'occasional_smoker': 1.3,
        'regular_smoker': 1.8,
        'heavy_smoker': 2.5
      },
      occupation: {
        'desk_job': 1.0,
        'field_work': 1.2,
        'manual_labor': 1.5,
        'hazardous': 2.0
      }
    };

    this.insuranceCompanies = {
      LIFE_INSURANCE: [
        { name: 'LIC', rating: 4.5, claimRatio: 98.5 },
        { name: 'HDFC Life', rating: 4.3, claimRatio: 97.8 },
        { name: 'ICICI Prudential', rating: 4.2, claimRatio: 97.2 },
        { name: 'SBI Life', rating: 4.1, claimRatio: 96.8 },
        { name: 'Max Life', rating: 4.0, claimRatio: 96.5 }
      ],
      HEALTH_INSURANCE: [
        { name: 'Star Health', rating: 4.4, claimRatio: 95.2 },
        { name: 'HDFC ERGO', rating: 4.3, claimRatio: 94.8 },
        { name: 'ICICI Lombard', rating: 4.2, claimRatio: 94.5 },
        { name: 'Bajaj Allianz', rating: 4.1, claimRatio: 94.2 },
        { name: 'New India Assurance', rating: 4.0, claimRatio: 93.8 }
      ]
    };
  }

  /**
   * Calculate ideal life insurance coverage
   */
  async calculateLifeInsuranceCoverage(userId) {
    try {
      logger.info('Calculating life insurance coverage', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });
      const familyMembers = await FamilyMember.find({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const coverageData = await this.getLifeInsuranceData(user, userPortfolio, familyMembers);
      const coverageCalculation = this.calculateLifeCoverageAmount(coverageData);
      const premiumEstimate = this.estimateLifeInsurancePremium(coverageCalculation, user);

      // Store recommendation
      await this.storeInsuranceRecommendation(userId, 'LIFE_INSURANCE', coverageCalculation, premiumEstimate);

      return {
        success: true,
        data: {
          coverageCalculation,
          premiumEstimate,
          recommendations: await this.getLifeInsuranceRecommendations(userId, coverageCalculation),
          companies: await this.getRecommendedLifeInsuranceCompanies(coverageCalculation)
        }
      };
    } catch (error) {
      logger.error('Failed to calculate life insurance coverage', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to calculate life insurance coverage',
        error: error.message
      };
    }
  }

  /**
   * Calculate health insurance requirements
   */
  async calculateHealthInsuranceCoverage(userId) {
    try {
      logger.info('Calculating health insurance coverage', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });
      const familyMembers = await FamilyMember.find({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const healthData = await this.getHealthInsuranceData(user, userPortfolio, familyMembers);
      const coverageCalculation = this.calculateHealthCoverageAmount(healthData);
      const premiumEstimate = this.estimateHealthInsurancePremium(coverageCalculation, user, familyMembers);

      // Store recommendation
      await this.storeInsuranceRecommendation(userId, 'HEALTH_INSURANCE', coverageCalculation, premiumEstimate);

      return {
        success: true,
        data: {
          coverageCalculation,
          premiumEstimate,
          recommendations: await this.getHealthInsuranceRecommendations(userId, coverageCalculation),
          companies: await this.getRecommendedHealthInsuranceCompanies(coverageCalculation),
          topUpAnalysis: await this.analyzeTopUpGap(userId, coverageCalculation)
        }
      };
    } catch (error) {
      logger.error('Failed to calculate health insurance coverage', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to calculate health insurance coverage',
        error: error.message
      };
    }
  }

  /**
   * Analyze insurance portfolio and identify gaps
   */
  async analyzeInsurancePortfolio(userId) {
    try {
      logger.info('Analyzing insurance portfolio', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });
      const existingPolicies = await InsurancePolicy.find({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const portfolioAnalysis = await this.analyzeCurrentCoverage(user, userPortfolio, existingPolicies);
      const gapAnalysis = await this.identifyCoverageGaps(userId, portfolioAnalysis);
      const optimizationPlan = await this.createOptimizationPlan(userId, gapAnalysis);

      return {
        success: true,
        data: {
          portfolioAnalysis,
          gapAnalysis,
          optimizationPlan,
          recommendations: await this.getPortfolioRecommendations(userId, gapAnalysis),
          premiumOptimization: await this.optimizePremiumToBenefit(userId, existingPolicies)
        }
      };
    } catch (error) {
      logger.error('Failed to analyze insurance portfolio', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to analyze insurance portfolio',
        error: error.message
      };
    }
  }

  /**
   * Get comprehensive insurance recommendations
   */
  async getComprehensiveInsuranceRecommendations(userId) {
    try {
      logger.info('Getting comprehensive insurance recommendations', { userId });

      const user = await User.findById(userId);
      const userPortfolio = await UserPortfolio.findOne({ userId });
      const familyMembers = await FamilyMember.find({ userId });

      if (!user || !userPortfolio) {
        throw new Error('User or portfolio not found');
      }

      const recommendations = await this.generateComprehensiveRecommendations(user, userPortfolio, familyMembers);

      return {
        success: true,
        data: {
          recommendations,
          priority: await this.prioritizeRecommendations(recommendations),
          implementation: await this.getImplementationPlan(userId, recommendations),
          timeline: this.getInsuranceTimeline(recommendations)
        }
      };
    } catch (error) {
      logger.error('Failed to get comprehensive insurance recommendations', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get comprehensive insurance recommendations',
        error: error.message
      };
    }
  }

  /**
   * Calculate premium-to-benefit optimization
   */
  async optimizePremiumToBenefit(userId, insuranceType) {
    try {
      logger.info('Optimizing premium-to-benefit ratio', { userId, insuranceType });

      const user = await User.findById(userId);
      const existingPolicies = await InsurancePolicy.find({ userId, type: insuranceType });

      if (!user) {
        throw new Error('User not found');
      }

      const optimization = await this.calculatePremiumOptimization(user, existingPolicies, insuranceType);

      return {
        success: true,
        data: {
          optimization,
          recommendations: await this.getPremiumOptimizationRecommendations(userId, optimization),
          alternatives: await this.getAlternativePlans(userId, insuranceType)
        }
      };
    } catch (error) {
      logger.error('Failed to optimize premium-to-benefit', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to optimize premium-to-benefit',
        error: error.message
      };
    }
  }

  // Helper methods
  async getLifeInsuranceData(user, userPortfolio, familyMembers) {
    const monthlyIncome = (user.income || 500000) / 12;
    const monthlyExpenses = monthlyIncome * 0.7; // Assume 70% of income as expenses
    const familySize = familyMembers.length + 1; // Including user
    const liabilities = await this.calculateLiabilities(user, userPortfolio);

    return {
      annualIncome: user.income || 500000,
      monthlyIncome,
      monthlyExpenses,
      familySize,
      liabilities,
      age: user.age,
      occupation: user.occupation || 'desk_job',
      health: user.health || 'good',
      lifestyle: user.lifestyle || 'non_smoker'
    };
  }

  calculateLifeCoverageAmount(coverageData) {
    const { annualIncome, familySize, liabilities, age } = coverageData;
    
    // Base coverage calculation
    let baseCoverage = annualIncome * this.coverageCalculators.LIFE_INSURANCE.baseMultiplier;
    
    // Apply age factor
    const ageFactor = this.getAgeFactor(age);
    baseCoverage *= ageFactor;
    
    // Apply family size factor
    const familyFactor = this.getFamilySizeFactor(familySize);
    baseCoverage *= familyFactor;
    
    // Apply liability factor
    const liabilityFactor = this.getLiabilityFactor(liabilities);
    baseCoverage *= liabilityFactor;
    
    // Add liabilities to coverage
    const totalCoverage = baseCoverage + liabilities;
    
    return {
      baseCoverage: Math.round(baseCoverage),
      liabilities: liabilities,
      totalCoverage: Math.round(totalCoverage),
      recommendedTerm: this.getRecommendedTerm(age),
      factors: {
        age: ageFactor,
        familySize: familyFactor,
        liabilities: liabilityFactor
      }
    };
  }

  estimateLifeInsurancePremium(coverageCalculation, user) {
    const { totalCoverage, recommendedTerm } = coverageCalculation;
    const age = user.age;
    const health = user.health || 'good';
    const lifestyle = user.lifestyle || 'non_smoker';
    const occupation = user.occupation || 'desk_job';

    // Base premium calculation (simplified)
    let basePremium = (totalCoverage / 1000000) * 5000; // ₹5000 per ₹10 lakhs
    
    // Apply factors
    basePremium *= this.premiumFactors.age[this.getAgeGroup(age)] || 1.5;
    basePremium *= this.premiumFactors.health[health] || 1.2;
    basePremium *= this.premiumFactors.lifestyle[lifestyle] || 1.0;
    basePremium *= this.premiumFactors.occupation[occupation] || 1.0;

    return {
      annualPremium: Math.round(basePremium),
      monthlyPremium: Math.round(basePremium / 12),
      premiumToCoverageRatio: (basePremium / totalCoverage) * 100,
      costEffectiveness: this.assessCostEffectiveness(basePremium, totalCoverage)
    };
  }

  async getHealthInsuranceData(user, userPortfolio, familyMembers) {
    const familySize = familyMembers.length + 1;
    const location = user.location || 'tier1';
    const age = user.age;

    return {
      familySize,
      location,
      age,
      familyAges: [age, ...familyMembers.map(m => m.age)],
      existingCoverage: await this.getExistingHealthCoverage(userId),
      medicalHistory: await this.getMedicalHistory(userId)
    };
  }

  calculateHealthCoverageAmount(healthData) {
    const { familySize, location, age, familyAges } = healthData;
    
    // Base coverage calculation
    let baseCoverage = this.coverageCalculators.HEALTH_INSURANCE.baseAmount;
    
    // Apply family size factor
    const familyFactor = this.getFamilySizeFactor(familySize);
    baseCoverage *= familyFactor;
    
    // Apply location factor
    const locationFactor = this.coverageCalculators.HEALTH_INSURANCE.factors.location[location] || 1.0;
    baseCoverage *= locationFactor;
    
    // Apply age factor (considering eldest member)
    const maxAge = Math.max(...familyAges);
    const ageFactor = this.getHealthAgeFactor(maxAge);
    baseCoverage *= ageFactor;
    
    return {
      baseCoverage: Math.round(baseCoverage),
      familyCoverage: Math.round(baseCoverage * 1.2), // 20% extra for family floater
      individualCoverage: Math.round(baseCoverage / familySize),
      recommendedType: familySize > 1 ? 'family_floater' : 'individual',
      factors: {
        familySize: familyFactor,
        location: locationFactor,
        age: ageFactor
      }
    };
  }

  estimateHealthInsurancePremium(coverageCalculation, user, familyMembers) {
    const { familyCoverage, individualCoverage, recommendedType } = coverageCalculation;
    const coverage = recommendedType === 'family_floater' ? familyCoverage : individualCoverage;
    
    // Base premium calculation (simplified)
    let basePremium = (coverage / 100000) * 800; // ₹800 per ₹1 lakh
    
    // Apply age factor for eldest member
    const maxAge = Math.max(user.age, ...familyMembers.map(m => m.age));
    basePremium *= this.getHealthAgeFactor(maxAge);
    
    // Apply family size factor
    if (recommendedType === 'family_floater') {
      basePremium *= (1 + familyMembers.length * 0.3); // 30% extra per additional member
    }

    return {
      annualPremium: Math.round(basePremium),
      monthlyPremium: Math.round(basePremium / 12),
      premiumToCoverageRatio: (basePremium / coverage) * 100,
      costEffectiveness: this.assessCostEffectiveness(basePremium, coverage)
    };
  }

  async analyzeCurrentCoverage(user, userPortfolio, existingPolicies) {
    const analysis = {
      lifeInsurance: {
        current: 0,
        required: 0,
        gap: 0,
        adequacy: 0
      },
      healthInsurance: {
        current: 0,
        required: 0,
        gap: 0,
        adequacy: 0
      },
      disabilityInsurance: {
        current: 0,
        required: 0,
        gap: 0,
        adequacy: 0
      },
      criticalIllness: {
        current: 0,
        required: 0,
        gap: 0,
        adequacy: 0
      }
    };

    // Calculate current coverage
    existingPolicies.forEach(policy => {
      if (analysis[policy.type]) {
        analysis[policy.type].current += policy.sumAssured || 0;
      }
    });

    // Calculate required coverage
    const lifeData = await this.getLifeInsuranceData(user, userPortfolio, []);
    const healthData = await this.getHealthInsuranceData(user, userPortfolio, []);
    
    analysis.lifeInsurance.required = this.calculateLifeCoverageAmount(lifeData).totalCoverage;
    analysis.healthInsurance.required = this.calculateHealthCoverageAmount(healthData).familyCoverage;

    // Calculate gaps and adequacy
    Object.keys(analysis).forEach(type => {
      const coverage = analysis[type];
      coverage.gap = Math.max(0, coverage.required - coverage.current);
      coverage.adequacy = coverage.required > 0 ? (coverage.current / coverage.required) * 100 : 0;
    });

    return analysis;
  }

  async identifyCoverageGaps(userId, portfolioAnalysis) {
    const gaps = [];

    Object.entries(portfolioAnalysis).forEach(([type, analysis]) => {
      if (analysis.gap > 0) {
        gaps.push({
          type,
          gap: analysis.gap,
          priority: this.getGapPriority(type, analysis.adequacy),
          recommendation: this.getGapRecommendation(type, analysis.gap)
        });
      }
    });

    return gaps.sort((a, b) => b.priority - a.priority);
  }

  async createOptimizationPlan(userId, gapAnalysis) {
    const plan = {
      immediate: [],
      shortTerm: [],
      longTerm: []
    };

    gapAnalysis.forEach(gap => {
      if (gap.priority >= 8) {
        plan.immediate.push(gap);
      } else if (gap.priority >= 5) {
        plan.shortTerm.push(gap);
      } else {
        plan.longTerm.push(gap);
      }
    });

    return plan;
  }

  async calculatePremiumOptimization(user, existingPolicies, insuranceType) {
    const optimization = {
      currentPremium: 0,
      optimizedPremium: 0,
      savings: 0,
      recommendations: []
    };

    // Calculate current premium
    existingPolicies.forEach(policy => {
      optimization.currentPremium += policy.annualPremium || 0;
    });

    // Generate optimization recommendations
    optimization.recommendations = await this.generateOptimizationRecommendations(user, existingPolicies, insuranceType);
    
    // Calculate potential savings
    optimization.optimizedPremium = optimization.currentPremium * 0.8; // Assume 20% savings
    optimization.savings = optimization.currentPremium - optimization.optimizedPremium;

    return optimization;
  }

  // Utility methods
  getAgeFactor(age) {
    if (age <= 30) return this.coverageCalculators.LIFE_INSURANCE.factors.age['18-30'];
    if (age <= 40) return this.coverageCalculators.LIFE_INSURANCE.factors.age['31-40'];
    if (age <= 50) return this.coverageCalculators.LIFE_INSURANCE.factors.age['41-50'];
    if (age <= 60) return this.coverageCalculators.LIFE_INSURANCE.factors.age['51-60'];
    return this.coverageCalculators.LIFE_INSURANCE.factors.age['60+'];
  }

  getFamilySizeFactor(familySize) {
    if (familySize <= 1) return this.coverageCalculators.LIFE_INSURANCE.factors.familySize[1];
    if (familySize <= 2) return this.coverageCalculators.LIFE_INSURANCE.factors.familySize[2];
    if (familySize <= 3) return this.coverageCalculators.LIFE_INSURANCE.factors.familySize[3];
    if (familySize <= 4) return this.coverageCalculators.LIFE_INSURANCE.factors.familySize[4];
    return this.coverageCalculators.LIFE_INSURANCE.factors.familySize['5+'];
  }

  getLiabilityFactor(liabilities) {
    if (liabilities === 0) return this.coverageCalculators.LIFE_INSURANCE.factors.liabilities.none;
    if (liabilities < 1000000) return this.coverageCalculators.LIFE_INSURANCE.factors.liabilities.low;
    if (liabilities < 3000000) return this.coverageCalculators.LIFE_INSURANCE.factors.liabilities.medium;
    return this.coverageCalculators.LIFE_INSURANCE.factors.liabilities.high;
  }

  getRecommendedTerm(age) {
    return Math.max(65 - age, 10); // Minimum 10 years, maximum till 65
  }

  getAgeGroup(age) {
    if (age <= 25) return '18-25';
    if (age <= 30) return '26-30';
    if (age <= 35) return '31-35';
    if (age <= 40) return '36-40';
    if (age <= 45) return '41-45';
    if (age <= 50) return '46-50';
    if (age <= 55) return '51-55';
    if (age <= 60) return '56-60';
    return '60+';
  }

  getHealthAgeFactor(age) {
    if (age <= 30) return 1.0;
    if (age <= 40) return 1.3;
    if (age <= 50) return 1.8;
    if (age <= 60) return 2.5;
    return 3.0;
  }

  assessCostEffectiveness(premium, coverage) {
    const ratio = (premium / coverage) * 100;
    if (ratio < 0.5) return 'excellent';
    if (ratio < 1.0) return 'good';
    if (ratio < 2.0) return 'average';
    return 'poor';
  }

  async calculateLiabilities(user, userPortfolio) {
    // Simplified liability calculation
    return (user.income || 500000) * 0.5; // Assume 50% of income as liabilities
  }

  getGapPriority(type, adequacy) {
    const basePriority = {
      'lifeInsurance': 10,
      'healthInsurance': 9,
      'disabilityInsurance': 7,
      'criticalIllness': 6
    };

    return basePriority[type] * (1 - adequacy / 100);
  }

  getGapRecommendation(type, gap) {
    const recommendations = {
      'lifeInsurance': `Increase life insurance coverage by ₹${gap.toLocaleString()}`,
      'healthInsurance': `Increase health insurance coverage by ₹${gap.toLocaleString()}`,
      'disabilityInsurance': `Consider disability insurance coverage of ₹${gap.toLocaleString()}`,
      'criticalIllness': `Consider critical illness coverage of ₹${gap.toLocaleString()}`
    };

    return recommendations[type] || 'Review insurance coverage';
  }

  // Storage and recommendation methods
  async storeInsuranceRecommendation(userId, type, coverageCalculation, premiumEstimate) {
    const recommendation = new InsuranceRecommendation({
      userId,
      type,
      coverageCalculation,
      premiumEstimate,
      createdAt: new Date()
    });

    await recommendation.save();
  }

  async getLifeInsuranceRecommendations(userId, coverageCalculation) {
    return [
      `Consider term insurance of ₹${coverageCalculation.totalCoverage.toLocaleString()}`,
      'Choose term insurance over endowment for better coverage',
      'Review coverage every 3-5 years or on major life events',
      'Consider riders for critical illness and disability'
    ];
  }

  async getHealthInsuranceRecommendations(userId, coverageCalculation) {
    return [
      `Consider health insurance of ₹${coverageCalculation.familyCoverage.toLocaleString()}`,
      'Choose family floater for better value',
      'Consider top-up plans for additional coverage',
      'Review coverage annually'
    ];
  }

  async getPortfolioRecommendations(userId, gapAnalysis) {
    return gapAnalysis.map(gap => gap.recommendation);
  }

  async getPremiumOptimizationRecommendations(userId, optimization) {
    return [
      'Compare premiums across multiple insurers',
      'Consider online policies for lower premiums',
      'Review and remove unnecessary riders',
      'Consider higher deductibles for lower premiums'
    ];
  }

  async getRecommendedLifeInsuranceCompanies(coverageCalculation) {
    return this.insuranceCompanies.LIFE_INSURANCE
      .sort((a, b) => b.rating - a.rating)
      .slice(0, 3);
  }

  async getRecommendedHealthInsuranceCompanies(coverageCalculation) {
    return this.insuranceCompanies.HEALTH_INSURANCE
      .sort((a, b) => b.rating - a.rating)
      .slice(0, 3);
  }

  async analyzeTopUpGap(userId, coverageCalculation) {
    return {
      currentCoverage: 0,
      recommendedTopUp: coverageCalculation.baseCoverage * 0.5,
      gap: coverageCalculation.baseCoverage * 0.5,
      recommendation: 'Consider top-up health insurance for additional coverage'
    };
  }

  async getAlternativePlans(userId, insuranceType) {
    return [
      'Compare with other insurance providers',
      'Consider group insurance options',
      'Review employer-provided insurance',
      'Explore online insurance platforms'
    ];
  }

  async generateComprehensiveRecommendations(user, userPortfolio, familyMembers) {
    return [
      'Prioritize life insurance for family protection',
      'Ensure adequate health insurance coverage',
      'Consider disability insurance for income protection',
      'Review insurance portfolio annually'
    ];
  }

  async prioritizeRecommendations(recommendations) {
    return recommendations.map((rec, index) => ({
      recommendation: rec,
      priority: 10 - index,
      timeline: index === 0 ? 'immediate' : index === 1 ? 'short_term' : 'long_term'
    }));
  }

  async getImplementationPlan(userId, recommendations) {
    return {
      immediate: recommendations.slice(0, 2),
      shortTerm: recommendations.slice(2, 4),
      longTerm: recommendations.slice(4)
    };
  }

  getInsuranceTimeline(recommendations) {
    return {
      immediate: 'Within 30 days',
      shortTerm: 'Within 3 months',
      longTerm: 'Within 6 months'
    };
  }

  async generateOptimizationRecommendations(user, existingPolicies, insuranceType) {
    return [
      'Compare premiums across multiple insurers',
      'Consider online policies for better rates',
      'Review and remove unnecessary riders',
      'Consider higher deductibles for lower premiums'
    ];
  }

  // Placeholder methods for data retrieval
  async getExistingHealthCoverage(userId) {
    return 0; // Placeholder
  }

  async getMedicalHistory(userId) {
    return []; // Placeholder
  }
}

module.exports = new InsuranceHelper(); 