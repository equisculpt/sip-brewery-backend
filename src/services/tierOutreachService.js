const logger = require('../utils/logger');
const { User, UserProfile, Community, FinancialLiteracy, MicroInvestment } = require('../models');

class TierOutreachService {
  constructor() {
    this.tierCategories = {
      TIER_1: {
        cities: ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'],
        features: ['full_features', 'advanced_analytics', 'premium_support'],
        minInvestment: 1000,
        literacyLevel: 'advanced'
      },
      TIER_2: {
        cities: ['Jaipur', 'Lucknow', 'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam'],
        features: ['simplified_ui', 'basic_analytics', 'community_support'],
        minInvestment: 500,
        literacyLevel: 'intermediate'
      },
      TIER_3: {
        cities: ['Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana', 'Agra', 'Nashik', 'Faridabad', 'Meerut'],
        features: ['basic_ui', 'minimal_analytics', 'peer_support'],
        minInvestment: 100,
        literacyLevel: 'basic'
      },
      RURAL: {
        cities: ['villages', 'small_towns', 'district_centers'],
        features: ['ultra_simple_ui', 'voice_interface', 'community_learning'],
        minInvestment: 50,
        literacyLevel: 'beginner'
      }
    };

    this.vernacularContent = {
      HINDI: {
        onboarding: {
          title: 'आपका स्वागत है',
          subtitle: 'निवेश की दुनिया में आपका पहला कदम',
          steps: [
            'अपना नाम और फोन नंबर दर्ज करें',
            'एक सरल प्रश्नावली भरें',
            'अपना पहला निवेश शुरू करें'
          ]
        },
        investment_guide: {
          title: 'निवेश गाइड',
          content: 'म्यूचुअल फंड में निवेश करने के लिए आपको क्या जानना चाहिए'
        }
      },
      TAMIL: {
        onboarding: {
          title: 'வரவேற்கிறோம்',
          subtitle: 'முதலீட்டு உலகில் உங்கள் முதல் படி',
          steps: [
            'உங்கள் பெயர் மற்றும் தொலைபேசி எண்ணை உள்ளிடவும்',
            'ஒரு எளிய கேள்வித்தாளை நிரப்பவும்',
            'உங்கள் முதல் முதலீட்டைத் தொடங்கவும்'
          ]
        },
        investment_guide: {
          title: 'முதலீட்டு வழிகாட்டி',
          content: 'பரஸ்பர நிதியில் முதலீடு செய்ய நீங்கள் தெரிந்து கொள்ள வேண்டியவை'
        }
      },
      TELUGU: {
        onboarding: {
          title: 'స్వాగతం',
          subtitle: 'పెట్టుబడి ప్రపంచంలో మీ మొదటి అడుగు',
          steps: [
            'మీ పేరు మరియు ఫోన్ నంబర్‌ను నమోదు చేయండి',
            'ఒక సరళ ప్రశ్నావళిని నింపండి',
            'మీ మొదటి పెట్టుబడిని ప్రారంభించండి'
          ]
        },
        investment_guide: {
          title: 'పెట్టుబడి గైడ్',
          content: 'మ్యూచువల్ ఫండ్‌లో పెట్టుబడి పెట్టడానికి మీరు తెలుసుకోవాల్సినవి'
        }
      }
    };

    this.microInvestmentOptions = {
      DAILY_SIP: {
        name: 'Daily SIP',
        minAmount: 10,
        maxAmount: 1000,
        frequency: 'daily',
        description: 'Start with as little as ₹10 per day'
      },
      WEEKLY_SIP: {
        name: 'Weekly SIP',
        minAmount: 50,
        maxAmount: 5000,
        frequency: 'weekly',
        description: 'Invest ₹50-5000 every week'
      },
      GOAL_BASED: {
        name: 'Goal Based',
        minAmount: 100,
        maxAmount: 10000,
        frequency: 'monthly',
        description: 'Save for specific goals'
      },
      FESTIVAL_SAVINGS: {
        name: 'Festival Savings',
        minAmount: 25,
        maxAmount: 2000,
        frequency: 'monthly',
        description: 'Save for festivals and celebrations'
      }
    };

    this.financialLiteracyModules = {
      BASIC: {
        title: 'Basic Financial Literacy',
        modules: [
          'What is Money?',
          'Saving vs Spending',
          'Introduction to Banking',
          'Understanding Interest',
          'Basic Budgeting'
        ],
        duration: '2 weeks',
        difficulty: 'beginner'
      },
      INTERMEDIATE: {
        title: 'Investment Basics',
        modules: [
          'What are Mutual Funds?',
          'Types of Mutual Funds',
          'Risk and Returns',
          'SIP vs Lump Sum',
          'Tax Benefits'
        ],
        duration: '4 weeks',
        difficulty: 'intermediate'
      },
      ADVANCED: {
        title: 'Advanced Investment',
        modules: [
          'Portfolio Diversification',
          'Market Analysis',
          'Fund Selection',
          'Tax Planning',
          'Retirement Planning'
        ],
        duration: '6 weeks',
        difficulty: 'advanced'
      }
    };
  }

  /**
   * Determine user tier based on location and profile
   */
  async determineUserTier(userId, location) {
    try {
      logger.info('Determining user tier', { userId, location });

      const user = await User.findById(userId);
      if (!user) {
        throw new Error('User not found');
      }

      let tier = 'TIER_1';
      const city = location?.city || user.city || 'Unknown';

      // Determine tier based on city
      for (const [tierName, tierData] of Object.entries(this.tierCategories)) {
        if (tierData.cities.includes(city) || 
            (tierName === 'RURAL' && !this.isMetroCity(city))) {
          tier = tierName;
          break;
        }
      }

      // Adjust tier based on user profile
      const adjustedTier = await this.adjustTierBasedOnProfile(user, tier);

      // Update user profile with tier information
      await UserProfile.findOneAndUpdate(
        { userId },
        {
          tier: adjustedTier,
          city: city,
          tierFeatures: this.tierCategories[adjustedTier]?.features || [],
          minInvestment: this.tierCategories[adjustedTier]?.minInvestment || 1000,
          literacyLevel: this.tierCategories[adjustedTier]?.literacyLevel || 'basic',
          lastUpdated: new Date()
        },
        { upsert: true, new: true }
      );

      return {
        success: true,
        data: {
          tier: adjustedTier,
          features: this.tierCategories[adjustedTier]?.features || [],
          minInvestment: this.tierCategories[adjustedTier]?.minInvestment || 1000,
          literacyLevel: this.tierCategories[adjustedTier]?.literacyLevel || 'basic',
          city: city
        }
      };
    } catch (error) {
      logger.error('Failed to determine user tier', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to determine user tier',
        error: error.message
      };
    }
  }

  /**
   * Create simplified onboarding flow
   */
  async createSimplifiedOnboarding(userId, languageCode = 'en') {
    try {
      logger.info('Creating simplified onboarding', { userId, languageCode });

      const user = await User.findById(userId);
      const userProfile = await UserProfile.findOne({ userId });

      if (!user) {
        throw new Error('User not found');
      }

      const tier = userProfile?.tier || 'TIER_1';
      const vernacularContent = this.vernacularContent[languageCode.toUpperCase()] || this.vernacularContent.HINDI;

      const onboardingFlow = {
        step1: {
          title: vernacularContent.onboarding.title,
          subtitle: vernacularContent.onboarding.subtitle,
          fields: this.getOnboardingFields(tier),
          validation: this.getValidationRules(tier)
        },
        step2: {
          title: 'Investment Preferences',
          questions: this.getInvestmentQuestions(tier, languageCode),
          options: this.getInvestmentOptions(tier)
        },
        step3: {
          title: 'First Investment',
          options: this.getFirstInvestmentOptions(tier),
          guidance: this.getInvestmentGuidance(tier, languageCode)
        }
      };

      return {
        success: true,
        data: {
          onboardingFlow,
          tier,
          languageCode,
          estimatedDuration: this.getEstimatedDuration(tier),
          features: this.tierCategories[tier]?.features || []
        }
      };
    } catch (error) {
      logger.error('Failed to create simplified onboarding', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create simplified onboarding',
        error: error.message
      };
    }
  }

  /**
   * Get vernacular content for user
   */
  async getVernacularContent(userId, contentType, languageCode) {
    try {
      logger.info('Getting vernacular content', { userId, contentType, languageCode });

      const userProfile = await UserProfile.findOne({ userId });
      const tier = userProfile?.tier || 'TIER_1';

      const content = this.vernacularContent[languageCode.toUpperCase()];
      if (!content) {
        throw new Error(`Content not available for language: ${languageCode}`);
      }

      let vernacularContent;

      switch (contentType) {
        case 'onboarding':
          vernacularContent = content.onboarding;
          break;
        case 'investment_guide':
          vernacularContent = content.investment_guide;
          break;
        case 'educational':
          vernacularContent = this.getEducationalContent(languageCode, tier);
          break;
        case 'community':
          vernacularContent = this.getCommunityContent(languageCode, tier);
          break;
        default:
          throw new Error(`Unknown content type: ${contentType}`);
      }

      return {
        success: true,
        data: {
          contentType,
          languageCode,
          tier,
          content: vernacularContent,
          simplified: tier !== 'TIER_1'
        }
      };
    } catch (error) {
      logger.error('Failed to get vernacular content', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get vernacular content',
        error: error.message
      };
    }
  }

  /**
   * Create micro-investment options
   */
  async createMicroInvestmentOptions(userId) {
    try {
      logger.info('Creating micro-investment options', { userId });

      const userProfile = await UserProfile.findOne({ userId });
      const tier = userProfile?.tier || 'TIER_1';

      const availableOptions = Object.entries(this.microInvestmentOptions)
        .filter(([key, option]) => {
          const minAmount = this.tierCategories[tier]?.minInvestment || 1000;
          return option.minAmount <= minAmount;
        })
        .map(([key, option]) => ({
          id: key,
          ...option,
          recommended: this.isRecommendedOption(key, tier)
        }));

      const personalizedOptions = await this.getPersonalizedOptions(userId, availableOptions);

      return {
        success: true,
        data: {
          options: personalizedOptions,
          tier,
          minInvestment: this.tierCategories[tier]?.minInvestment || 1000,
          maxInvestment: this.getMaxInvestment(tier),
          totalOptions: personalizedOptions.length
        }
      };
    } catch (error) {
      logger.error('Failed to create micro-investment options', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create micro-investment options',
        error: error.message
      };
    }
  }

  /**
   * Create community features
   */
  async createCommunityFeatures(userId, location) {
    try {
      logger.info('Creating community features', { userId, location });

      const userProfile = await UserProfile.findOne({ userId });
      const tier = userProfile?.tier || 'TIER_1';

      const communityFeatures = {
        localGroups: await this.findLocalGroups(location, tier),
        mentors: await this.findMentors(location, tier),
        events: await this.findLocalEvents(location, tier),
        discussions: await this.getCommunityDiscussions(tier),
        successStories: await this.getSuccessStories(location, tier)
      };

      const communityGuidance = this.getCommunityGuidance(tier);

      return {
        success: true,
        data: {
          communityFeatures,
          communityGuidance,
          tier,
          location,
          features: this.getCommunityFeaturesByTier(tier)
        }
      };
    } catch (error) {
      logger.error('Failed to create community features', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create community features',
        error: error.message
      };
    }
  }

  /**
   * Create financial literacy program
   */
  async createFinancialLiteracyProgram(userId) {
    try {
      logger.info('Creating financial literacy program', { userId });

      const userProfile = await UserProfile.findOne({ userId });
      const tier = userProfile?.tier || 'TIER_1';
      const literacyLevel = this.tierCategories[tier]?.literacyLevel || 'basic';

      const program = this.financialLiteracyModules[literacyLevel.toUpperCase()];
      if (!program) {
        throw new Error(`No program available for literacy level: ${literacyLevel}`);
      }

      const personalizedProgram = await this.personalizeProgram(userId, program, tier);

      return {
        success: true,
        data: {
          program: personalizedProgram,
          tier,
          literacyLevel,
          duration: program.duration,
          difficulty: program.difficulty,
          modules: program.modules,
          progress: await this.getUserProgress(userId, literacyLevel)
        }
      };
    } catch (error) {
      logger.error('Failed to create financial literacy program', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to create financial literacy program',
        error: error.message
      };
    }
  }

  /**
   * Get tier-specific features
   */
  async getTierSpecificFeatures(userId) {
    try {
      logger.info('Getting tier-specific features', { userId });

      const userProfile = await UserProfile.findOne({ userId });
      const tier = userProfile?.tier || 'TIER_1';

      const features = {
        ui: this.getUIFeatures(tier),
        analytics: this.getAnalyticsFeatures(tier),
        support: this.getSupportFeatures(tier),
        investment: this.getInvestmentFeatures(tier),
        community: this.getCommunityFeaturesByTier(tier),
        education: this.getEducationFeatures(tier)
      };

      return {
        success: true,
        data: {
          tier,
          features,
          recommendations: this.getFeatureRecommendations(tier)
        }
      };
    } catch (error) {
      logger.error('Failed to get tier-specific features', { error: error.message, userId });
      return {
        success: false,
        message: 'Failed to get tier-specific features',
        error: error.message
      };
    }
  }

  // Helper methods
  isMetroCity(city) {
    const metroCities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad'];
    return metroCities.includes(city);
  }

  async adjustTierBasedOnProfile(user, initialTier) {
    try {
      // Adjust based on income
      if (user.income && user.income < 300000) {
        if (initialTier === 'TIER_1') return 'TIER_2';
        if (initialTier === 'TIER_2') return 'TIER_3';
      }

      // Adjust based on education
      if (user.education && user.education === 'high_school') {
        if (initialTier === 'TIER_1') return 'TIER_2';
      }

      // Adjust based on occupation
      if (user.occupation && ['farmer', 'daily_wage', 'small_business'].includes(user.occupation)) {
        if (initialTier === 'TIER_1') return 'TIER_2';
        if (initialTier === 'TIER_2') return 'TIER_3';
      }

      return initialTier;
    } catch (error) {
      logger.error('Failed to adjust tier based on profile', { error: error.message });
      return initialTier;
    }
  }

  getOnboardingFields(tier) {
    const fields = {
      TIER_1: ['name', 'email', 'phone', 'income', 'occupation', 'education', 'investment_goals'],
      TIER_2: ['name', 'phone', 'income', 'occupation', 'investment_goals'],
      TIER_3: ['name', 'phone', 'income', 'basic_goals'],
      RURAL: ['name', 'phone', 'village', 'basic_goals']
    };

    return fields[tier] || fields.TIER_2;
  }

  getValidationRules(tier) {
    const rules = {
      TIER_1: { strict: true, required: ['name', 'email', 'phone'] },
      TIER_2: { strict: false, required: ['name', 'phone'] },
      TIER_3: { strict: false, required: ['name', 'phone'] },
      RURAL: { strict: false, required: ['name', 'phone'] }
    };

    return rules[tier] || rules.TIER_2;
  }

  getInvestmentQuestions(tier, languageCode) {
    const questions = {
      TIER_1: [
        'What is your investment goal?',
        'What is your risk tolerance?',
        'What is your investment horizon?',
        'How much can you invest monthly?'
      ],
      TIER_2: [
        'What do you want to achieve?',
        'Are you comfortable with risk?',
        'How long can you invest?',
        'How much can you save monthly?'
      ],
      TIER_3: [
        'What is your main goal?',
        'Do you want safe or growth investments?',
        'How many years can you invest?',
        'How much can you invest?'
      ],
      RURAL: [
        'What do you want to save for?',
        'Do you want safe investments?',
        'How many years?',
        'How much money?'
      ]
    };

    return questions[tier] || questions.TIER_2;
  }

  getInvestmentOptions(tier) {
    const options = {
      TIER_1: ['wealth_creation', 'retirement', 'children_education', 'house_purchase', 'tax_saving'],
      TIER_2: ['wealth_creation', 'children_education', 'house_purchase', 'tax_saving'],
      TIER_3: ['children_education', 'house_purchase', 'emergency_fund'],
      RURAL: ['children_education', 'house_repair', 'emergency_fund']
    };

    return options[tier] || options.TIER_2;
  }

  getFirstInvestmentOptions(tier) {
    const options = {
      TIER_1: ['lump_sum', 'sip', 'goal_based'],
      TIER_2: ['sip', 'goal_based', 'micro_sip'],
      TIER_3: ['micro_sip', 'goal_based', 'festival_savings'],
      RURAL: ['micro_sip', 'festival_savings', 'daily_savings']
    };

    return options[tier] || options.TIER_2;
  }

  getInvestmentGuidance(tier, languageCode) {
    const guidance = {
      TIER_1: 'Start with a diversified portfolio based on your goals',
      TIER_2: 'Begin with SIP in balanced funds for steady growth',
      TIER_3: 'Start small with micro-SIP in safe funds',
      RURAL: 'Start with small daily savings in safe funds'
    };

    return guidance[tier] || guidance.TIER_2;
  }

  getEstimatedDuration(tier) {
    const durations = {
      TIER_1: '5 minutes',
      TIER_2: '3 minutes',
      TIER_3: '2 minutes',
      RURAL: '1 minute'
    };

    return durations[tier] || durations.TIER_2;
  }

  getEducationalContent(languageCode, tier) {
    // Mock educational content
    return {
      title: 'Investment Basics',
      content: 'Learn the basics of mutual fund investing',
      difficulty: tier === 'RURAL' ? 'very_basic' : 'basic'
    };
  }

  getCommunityContent(languageCode, tier) {
    // Mock community content
    return {
      title: 'Join Your Community',
      content: 'Connect with local investors',
      features: ['local_groups', 'mentors', 'events']
    };
  }

  isRecommendedOption(optionKey, tier) {
    const recommendations = {
      TIER_1: ['WEEKLY_SIP', 'GOAL_BASED'],
      TIER_2: ['WEEKLY_SIP', 'GOAL_BASED'],
      TIER_3: ['DAILY_SIP', 'FESTIVAL_SAVINGS'],
      RURAL: ['DAILY_SIP', 'FESTIVAL_SAVINGS']
    };

    return recommendations[tier]?.includes(optionKey) || false;
  }

  async getPersonalizedOptions(userId, availableOptions) {
    try {
      // Mock personalization based on user profile
      return availableOptions.map(option => ({
        ...option,
        personalized: true,
        recommendedAmount: this.getRecommendedAmount(option, userId)
      }));
    } catch (error) {
      logger.error('Failed to get personalized options', { error: error.message });
      return availableOptions;
    }
  }

  getRecommendedAmount(option, userId) {
    // Mock recommendation logic
    const baseAmounts = {
      DAILY_SIP: 25,
      WEEKLY_SIP: 100,
      GOAL_BASED: 500,
      FESTIVAL_SAVINGS: 200
    };

    return baseAmounts[option.id] || 100;
  }

  getMaxInvestment(tier) {
    const maxInvestments = {
      TIER_1: 100000,
      TIER_2: 50000,
      TIER_3: 25000,
      RURAL: 10000
    };

    return maxInvestments[tier] || 50000;
  }

  async findLocalGroups(location, tier) {
    try {
      // Mock local groups
      return [
        {
          name: 'Local Investors Group',
          members: 150,
          location: location?.city || 'Unknown',
          tier: tier,
          description: 'Connect with local investors'
        }
      ];
    } catch (error) {
      logger.error('Failed to find local groups', { error: error.message });
      return [];
    }
  }

  async findMentors(location, tier) {
    try {
      // Mock mentors
      return [
        {
          name: 'Local Financial Advisor',
          experience: '5 years',
          location: location?.city || 'Unknown',
          tier: tier,
          rating: 4.5
        }
      ];
    } catch (error) {
      logger.error('Failed to find mentors', { error: error.message });
      return [];
    }
  }

  async findLocalEvents(location, tier) {
    try {
      // Mock local events
      return [
        {
          name: 'Investment Workshop',
          date: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000),
          location: location?.city || 'Unknown',
          tier: tier,
          description: 'Learn about mutual funds'
        }
      ];
    } catch (error) {
      logger.error('Failed to find local events', { error: error.message });
      return [];
    }
  }

  async getCommunityDiscussions(tier) {
    try {
      // Mock discussions
      return [
        {
          topic: 'How to start investing?',
          participants: 25,
          tier: tier,
          language: 'local'
        }
      ];
    } catch (error) {
      logger.error('Failed to get community discussions', { error: error.message });
      return [];
    }
  }

  async getSuccessStories(location, tier) {
    try {
      // Mock success stories
      return [
        {
          name: 'Local Success Story',
          location: location?.city || 'Unknown',
          tier: tier,
          story: 'Started with ₹100, now investing ₹1000 monthly'
        }
      ];
    } catch (error) {
      logger.error('Failed to get success stories', { error: error.message });
      return [];
    }
  }

  getCommunityGuidance(tier) {
    const guidance = {
      TIER_1: 'Join professional investment groups',
      TIER_2: 'Connect with local investment communities',
      TIER_3: 'Learn from peer investors',
      RURAL: 'Start with family and friends'
    };

    return guidance[tier] || guidance.TIER_2;
  }

  getCommunityFeaturesByTier(tier) {
    const features = {
      TIER_1: ['professional_groups', 'mentors', 'events', 'discussions'],
      TIER_2: ['local_groups', 'mentors', 'events', 'discussions'],
      TIER_3: ['peer_groups', 'basic_mentors', 'local_events'],
      RURAL: ['family_groups', 'community_learning', 'basic_events']
    };

    return features[tier] || features.TIER_2;
  }

  async personalizeProgram(userId, program, tier) {
    try {
      // Mock personalization
      return {
        ...program,
        personalized: true,
        userSpecific: true,
        adaptive: tier !== 'TIER_1'
      };
    } catch (error) {
      logger.error('Failed to personalize program', { error: error.message });
      return program;
    }
  }

  async getUserProgress(userId, literacyLevel) {
    try {
      // Mock user progress
      return {
        completedModules: 0,
        totalModules: this.financialLiteracyModules[literacyLevel.toUpperCase()]?.modules?.length || 0,
        progress: 0,
        currentModule: 1
      };
    } catch (error) {
      logger.error('Failed to get user progress', { error: error.message });
      return { completedModules: 0, totalModules: 0, progress: 0, currentModule: 1 };
    }
  }

  getUIFeatures(tier) {
    const features = {
      TIER_1: ['advanced_ui', 'customizable_dashboard', 'detailed_analytics'],
      TIER_2: ['simplified_ui', 'basic_dashboard', 'simple_analytics'],
      TIER_3: ['basic_ui', 'minimal_dashboard', 'basic_charts'],
      RURAL: ['ultra_simple_ui', 'voice_interface', 'picture_based']
    };

    return features[tier] || features.TIER_2;
  }

  getAnalyticsFeatures(tier) {
    const features = {
      TIER_1: ['advanced_analytics', 'custom_reports', 'predictive_insights'],
      TIER_2: ['basic_analytics', 'standard_reports', 'simple_insights'],
      TIER_3: ['minimal_analytics', 'basic_reports', 'simple_charts'],
      RURAL: ['no_analytics', 'basic_summary', 'simple_numbers']
    };

    return features[tier] || features.TIER_2;
  }

  getSupportFeatures(tier) {
    const features = {
      TIER_1: ['premium_support', 'dedicated_manager', 'priority_support'],
      TIER_2: ['standard_support', 'community_support', 'chat_support'],
      TIER_3: ['basic_support', 'peer_support', 'email_support'],
      RURAL: ['community_support', 'voice_support', 'local_support']
    };

    return features[tier] || features.TIER_2;
  }

  getInvestmentFeatures(tier) {
    const features = {
      TIER_1: ['all_funds', 'advanced_strategies', 'custom_portfolios'],
      TIER_2: ['popular_funds', 'basic_strategies', 'model_portfolios'],
      TIER_3: ['safe_funds', 'simple_strategies', 'predefined_portfolios'],
      RURAL: ['basic_funds', 'micro_investments', 'goal_based_portfolios']
    };

    return features[tier] || features.TIER_2;
  }

  getEducationFeatures(tier) {
    const features = {
      TIER_1: ['advanced_courses', 'webinars', 'expert_sessions'],
      TIER_2: ['basic_courses', 'workshops', 'mentor_sessions'],
      TIER_3: ['simple_courses', 'local_workshops', 'peer_learning'],
      RURAL: ['basic_learning', 'community_learning', 'family_learning']
    };

    return features[tier] || features.TIER_2;
  }

  getFeatureRecommendations(tier) {
    const recommendations = {
      TIER_1: ['Use advanced analytics', 'Join professional groups', 'Attend expert sessions'],
      TIER_2: ['Start with basic analytics', 'Join local groups', 'Attend workshops'],
      TIER_3: ['Use simple interface', 'Join peer groups', 'Learn from community'],
      RURAL: ['Use voice interface', 'Learn from family', 'Start with small amounts']
    };

    return recommendations[tier] || recommendations.TIER_2;
  }
}

module.exports = new TierOutreachService(); 