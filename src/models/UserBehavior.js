const mongoose = require('mongoose');

const userBehaviorSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    unique: true,
    index: true
  },
  actions: [{
    action: {
      type: String,
      enum: [
        'login', 'logout', 'view_portfolio', 'view_funds', 'start_sip', 'stop_sip',
        'switch_fund', 'withdraw', 'add_funds', 'view_analytics', 'view_insights',
        'accept_recommendation', 'reject_recommendation', 'modify_recommendation',
        'view_learning', 'complete_lesson', 'take_quiz', 'view_rewards',
        'share_portfolio', 'view_leaderboard', 'participate_challenge',
        'contact_support', 'update_profile', 'change_preferences'
      ],
      required: true
    },
    timestamp: {
      type: Date,
      default: Date.now
    },
    context: {
      page: String,
      feature: String,
      device: String,
      sessionId: String
    },
    metadata: {
      duration: Number, // seconds
      clicks: Number,
      scrollDepth: Number,
      searchTerms: [String],
      selectedOptions: [String]
    }
  }],
  patterns: {
    loginFrequency: {
      daily: { type: Number, default: 0 },
      weekly: { type: Number, default: 0 },
      monthly: { type: Number, default: 0 }
    },
    sessionDuration: {
      average: { type: Number, default: 0 },
      longest: { type: Number, default: 0 },
      shortest: { type: Number, default: 0 }
    },
    preferredFeatures: [{
      feature: String,
      usageCount: { type: Number, default: 0 },
      lastUsed: Date
    }],
    preferredTimes: [{
      hour: Number,
      frequency: { type: Number, default: 0 }
    }],
    preferredDays: [{
      day: String,
      frequency: { type: Number, default: 0 }
    }]
  },
  preferences: {
    riskTolerance: {
      type: String,
      enum: ['conservative', 'moderate', 'aggressive'],
      default: 'moderate'
    },
    investmentHorizon: {
      type: String,
      enum: ['short_term', 'medium_term', 'long_term'],
      default: 'medium_term'
    },
    preferredFundTypes: [{
      type: String,
      enum: ['equity', 'debt', 'hybrid', 'liquid', 'sectoral', 'index', 'international']
    }],
    preferredFundHouses: [String],
    notificationPreferences: {
      email: { type: Boolean, default: true },
      sms: { type: Boolean, default: false },
      push: { type: Boolean, default: true },
      whatsapp: { type: Boolean, default: true }
    },
    language: {
      type: String,
      default: 'english'
    },
    theme: {
      type: String,
      enum: ['light', 'dark', 'auto'],
      default: 'light'
    }
  },
  learning: {
    completedLessons: [{
      lessonId: String,
      topic: String,
      completedAt: Date,
      score: Number,
      timeSpent: Number
    }],
    quizResults: [{
      quizId: String,
      topic: String,
      score: Number,
      totalQuestions: Number,
      completedAt: Date
    }],
    learningStreak: {
      current: { type: Number, default: 0 },
      longest: { type: Number, default: 0 },
      lastActivity: Date
    },
    preferredTopics: [{
      topic: String,
      interest: { type: Number, min: 0, max: 1, default: 0.5 }
    }],
    difficultyLevel: {
      type: String,
      enum: ['beginner', 'intermediate', 'advanced'],
      default: 'beginner'
    }
  },
  investment: {
    decisionPatterns: {
      quickDecisions: { type: Number, default: 0 },
      researchedDecisions: { type: Number, default: 0 },
      consultationDecisions: { type: Number, default: 0 }
    },
    riskBehavior: {
      averageRiskTaken: { type: Number, min: 0, max: 1, default: 0.5 },
      riskAdjustments: [{
        date: Date,
        oldRisk: String,
        newRisk: String,
        reason: String
      }]
    },
    goalAlignment: {
      primaryGoal: {
        type: String,
        enum: ['wealth_creation', 'retirement', 'child_education', 'house_purchase', 'emergency_fund', 'tax_savings'],
        default: 'wealth_creation'
      },
      goalProgress: [{
        goal: String,
        targetAmount: Number,
        currentAmount: Number,
        targetDate: Date,
        progress: { type: Number, min: 0, max: 1, default: 0 }
      }]
    },
    fundSelectionCriteria: [{
      criteria: String,
      weight: { type: Number, min: 0, max: 1, default: 0.5 }
    }]
  },
  engagement: {
    gamification: {
      points: { type: Number, default: 0 },
      level: { type: Number, default: 1 },
      badges: [{
        badgeId: String,
        name: String,
        earnedAt: Date,
        description: String
      }],
      achievements: [{
        achievementId: String,
        name: String,
        earnedAt: Date,
        description: String
      }]
    },
    social: {
      portfolioShares: { type: Number, default: 0 },
      referrals: { type: Number, default: 0 },
      communityParticipation: { type: Number, default: 0 },
      leaderboardRank: { type: Number, default: 0 }
    },
    support: {
      supportTickets: { type: Number, default: 0 },
      feedbackGiven: { type: Number, default: 0 },
      satisfactionRating: { type: Number, min: 1, max: 5, default: 0 }
    }
  },
  aiInteraction: {
    recommendationResponses: [{
      insightId: mongoose.Schema.Types.ObjectId,
      response: {
        type: String,
        enum: ['accepted', 'rejected', 'modified', 'ignored']
      },
      feedback: String,
      timestamp: Date
    }],
    autonomousMode: {
      enabled: { type: Boolean, default: false },
      lastToggle: Date,
      comfortLevel: { type: Number, min: 0, max: 1, default: 0.5 }
    },
    learningFromAI: {
      patternsLearned: [String],
      preferencesUpdated: [String],
      lastLearningUpdate: Date
    }
  },
  analytics: {
    lastCalculated: Date,
    behaviorScore: { type: Number, min: 0, max: 100, default: 50 },
    engagementScore: { type: Number, min: 0, max: 100, default: 50 },
    learningScore: { type: Number, min: 0, max: 100, default: 50 },
    investmentScore: { type: Number, min: 0, max: 100, default: 50 }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for better query performance
userBehaviorSchema.index({ userId: 1, 'actions.timestamp': -1 });
userBehaviorSchema.index({ 'patterns.loginFrequency.daily': -1 });
userBehaviorSchema.index({ 'engagement.gamification.points': -1 });
userBehaviorSchema.index({ 'analytics.behaviorScore': -1 });

// Virtual for total actions count
userBehaviorSchema.virtual('totalActions').get(function() {
  return this.actions.length;
});

// Virtual for recent activity
userBehaviorSchema.virtual('recentActivity').get(function() {
  const oneWeekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  return this.actions.filter(action => action.timestamp > oneWeekAgo).length;
});

// Virtual for engagement level
userBehaviorSchema.virtual('engagementLevel').get(function() {
  const score = this.analytics.engagementScore;
  if (score >= 80) return 'high';
  if (score >= 60) return 'medium';
  return 'low';
});

// Pre-save middleware
userBehaviorSchema.pre('save', function(next) {
  // Update analytics scores
  this.updateAnalyticsScores();
  next();
});

// Static methods
userBehaviorSchema.statics.findByEngagementLevel = function(level) {
  const scoreRanges = {
    high: { $gte: 80 },
    medium: { $gte: 60, $lt: 80 },
    low: { $lt: 60 }
  };
  
  return this.find({
    'analytics.engagementScore': scoreRanges[level]
  });
};

userBehaviorSchema.statics.findActiveUsers = function(days = 7) {
  const cutoffDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
  return this.find({
    'actions.timestamp': { $gte: cutoffDate }
  });
};

// Instance methods
userBehaviorSchema.methods.addAction = function(action, context = {}, metadata = {}) {
  this.actions.push({
    action,
    timestamp: new Date(),
    context,
    metadata
  });
  
  // Keep only last 1000 actions to prevent document size issues
  if (this.actions.length > 1000) {
    this.actions = this.actions.slice(-1000);
  }
  
  return this.save();
};

userBehaviorSchema.methods.updatePreferences = function(newPreferences) {
  this.preferences = { ...this.preferences, ...newPreferences };
  return this.save();
};

userBehaviorSchema.methods.addLearningActivity = function(lessonId, topic, score, timeSpent) {
  this.learning.completedLessons.push({
    lessonId,
    topic,
    completedAt: new Date(),
    score,
    timeSpent
  });
  
  // Update learning streak
  const today = new Date().toDateString();
  const lastActivity = this.learning.learningStreak.lastActivity?.toDateString();
  
  if (lastActivity === today) {
    // Already logged today
  } else if (lastActivity === new Date(Date.now() - 24 * 60 * 60 * 1000).toDateString()) {
    // Consecutive day
    this.learning.learningStreak.current += 1;
  } else {
    // Break in streak
    this.learning.learningStreak.current = 1;
  }
  
  this.learning.learningStreak.lastActivity = new Date();
  
  if (this.learning.learningStreak.current > this.learning.learningStreak.longest) {
    this.learning.learningStreak.longest = this.learning.learningStreak.current;
  }
  
  return this.save();
};

userBehaviorSchema.methods.updateAnalyticsScores = function() {
  // Calculate behavior score based on action patterns
  const totalActions = this.actions.length;
  const recentActions = this.actions.filter(a => 
    a.timestamp > new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
  ).length;
  
  this.analytics.behaviorScore = Math.min(100, Math.max(0, 
    (recentActions / 30) * 50 + (totalActions / 100) * 50
  ));
  
  // Calculate engagement score
  const engagementFactors = [
    this.engagement.gamification.points / 1000 * 25,
    this.engagement.social.portfolioShares * 10,
    this.engagement.social.referrals * 15,
    this.learning.learningStreak.current * 5
  ];
  
  this.analytics.engagementScore = Math.min(100, Math.max(0,
    engagementFactors.reduce((sum, factor) => sum + factor, 0)
  ));
  
  // Calculate learning score
  const completedLessons = this.learning.completedLessons.length;
  const averageQuizScore = this.learning.quizResults.length > 0 
    ? this.learning.quizResults.reduce((sum, quiz) => sum + quiz.score, 0) / this.learning.quizResults.length
    : 0;
  
  this.analytics.learningScore = Math.min(100, Math.max(0,
    (completedLessons / 20) * 50 + (averageQuizScore / 100) * 50
  ));
  
  // Calculate investment score
  const investmentFactors = [
    this.investment.decisionPatterns.researchedDecisions * 10,
    this.investment.goalAlignment.goalProgress.length * 15,
    this.aiInteraction.recommendationResponses.filter(r => r.response === 'accepted').length * 5
  ];
  
  this.analytics.investmentScore = Math.min(100, Math.max(0,
    investmentFactors.reduce((sum, factor) => sum + factor, 0)
  ));
  
  this.analytics.lastCalculated = new Date();
};

if (process.env.NODE_ENV === 'test') {
  const mockUserBehaviorModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockUserBehaviorId', ...data }),
    };
  };
  mockUserBehaviorModel.find = jest.fn().mockResolvedValue([]);
  mockUserBehaviorModel.findOne = jest.fn().mockResolvedValue(null);
  mockUserBehaviorModel.findById = jest.fn().mockResolvedValue(null);
  mockUserBehaviorModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockUserBehaviorModel.create = jest.fn().mockResolvedValue({ _id: 'mockUserBehaviorId' });
  module.exports = mockUserBehaviorModel;
} else {
  module.exports = mongoose.model('UserBehavior', userBehaviorSchema);
}