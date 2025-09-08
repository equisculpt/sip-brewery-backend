const mongoose = require('mongoose');

const agiInsightSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  insightType: {
    type: String,
    enum: ['FUND_SWITCH', 'SIP_UPDATE', 'REBALANCING', 'TAX_OPTIMIZATION', 'GOAL_ADJUSTMENT', 'RISK_MANAGEMENT', 'MARKET_OPPORTUNITY'],
    required: true
  },
  title: {
    type: String,
    required: true,
    maxlength: 200
  },
  description: {
    type: String,
    required: true,
    maxlength: 1000
  },
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'critical'],
    default: 'medium'
  },
  confidence: {
    type: Number,
    min: 0,
    max: 1,
    default: 0.7
  },
  reasoning: {
    type: String,
    required: true,
    maxlength: 2000
  },
  action: {
    type: String,
    enum: ['switch_fund', 'update_sip', 'rebalance_portfolio', 'tax_optimization', 'adjust_goal', 'monitor', 'no_action'],
    required: true
  },
  implementationSteps: [{
    step: {
      type: String,
      required: true,
      maxlength: 500
    },
    order: {
      type: Number,
      required: true
    },
    estimatedTime: {
      type: String,
      enum: ['immediate', '1_day', '1_week', '1_month'],
      default: '1_week'
    }
  }],
  alternatives: [{
    title: {
      type: String,
      required: true,
      maxlength: 200
    },
    description: {
      type: String,
      required: true,
      maxlength: 1000
    },
    pros: [String],
    cons: [String]
  }],
  marketContext: {
    currentTrend: {
      type: String,
      enum: ['bullish', 'bearish', 'sideways', 'volatile'],
      default: 'sideways'
    },
    volatility: {
      type: String,
      enum: ['low', 'medium', 'high'],
      default: 'medium'
    },
    sentiment: {
      type: String,
      enum: ['positive', 'negative', 'neutral'],
      default: 'neutral'
    }
  },
  economicFactors: [{
    factor: {
      type: String,
      enum: ['INFLATION', 'REPO_RATE', 'SECTOR_ROTATION', 'FISCAL_POLICY', 'GLOBAL_MARKETS'],
      required: true
    },
    impact: {
      type: String,
      enum: ['positive', 'negative', 'neutral'],
      required: true
    },
    description: String
  }],
  portfolioImpact: {
    expectedReturn: {
      type: Number,
      min: -100,
      max: 100
    },
    riskChange: {
      type: String,
      enum: ['decrease', 'increase', 'no_change'],
      default: 'no_change'
    },
    timeHorizon: {
      type: String,
      enum: ['short_term', 'medium_term', 'long_term'],
      default: 'medium_term'
    }
  },
  userResponse: {
    status: {
      type: String,
      enum: ['pending', 'accepted', 'rejected', 'modified', 'implemented'],
      default: 'pending'
    },
    feedback: {
      type: String,
      maxlength: 1000
    },
    implementationDate: Date,
    actualOutcome: {
      type: String,
      enum: ['better_than_expected', 'as_expected', 'worse_than_expected', 'not_implemented'],
      default: 'not_implemented'
    }
  },
  learningData: {
    patterns: [String],
    preferences: [String],
    riskTolerance: {
      type: String,
      enum: ['conservative', 'moderate', 'aggressive'],
      default: 'moderate'
    },
    goalAlignment: {
      type: Number,
      min: 0,
      max: 1,
      default: 0.7
    }
  },
  metadata: {
    source: {
      type: String,
      enum: ['agi_engine', 'market_analysis', 'user_behavior', 'economic_analysis'],
      default: 'agi_engine'
    },
    version: {
      type: String,
      default: '1.0'
    },
    tags: [String],
    relatedInsights: [{
      type: mongoose.Schema.Types.ObjectId,
      ref: 'AGIInsight'
    }]
  },
  status: {
    type: String,
    enum: ['active', 'archived', 'expired', 'superseded'],
    default: 'active'
  },
  expiresAt: {
    type: Date,
    default: function() {
      return new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); // 30 days from now
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for better query performance
agiInsightSchema.index({ userId: 1, insightType: 1 });
agiInsightSchema.index({ userId: 1, priority: 1 });
agiInsightSchema.index({ userId: 1, status: 1 });
agiInsightSchema.index({ 'userResponse.status': 1 });
agiInsightSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 });

// Virtual for insight age
agiInsightSchema.virtual('age').get(function() {
  return Math.floor((Date.now() - this.createdAt.getTime()) / (1000 * 60 * 60 * 24));
});

// Virtual for implementation status
agiInsightSchema.virtual('isImplemented').get(function() {
  return this.userResponse.status === 'implemented';
});

// Virtual for urgency
agiInsightSchema.virtual('isUrgent').get(function() {
  return this.priority === 'critical' || this.priority === 'high';
});

// Pre-save middleware
agiInsightSchema.pre('save', function(next) {
  // Auto-archive expired insights
  if (this.expiresAt && this.expiresAt < new Date() && this.status === 'active') {
    this.status = 'expired';
  }
  
  // Update status based on user response
  if (this.userResponse.status === 'implemented') {
    this.status = 'archived';
  }
  
  next();
});

// Static methods
agiInsightSchema.statics.findActiveInsights = function(userId) {
  return this.find({
    userId,
    status: 'active',
    expiresAt: { $gt: new Date() }
  }).sort({ priority: -1, createdAt: -1 });
};

agiInsightSchema.statics.findByType = function(userId, insightType) {
  return this.find({
    userId,
    insightType,
    status: 'active'
  }).sort({ createdAt: -1 });
};

agiInsightSchema.statics.findHighPriority = function(userId) {
  return this.find({
    userId,
    priority: { $in: ['high', 'critical'] },
    status: 'active'
  }).sort({ createdAt: -1 });
};

// Instance methods
agiInsightSchema.methods.markAsImplemented = function(outcome = 'as_expected') {
  this.userResponse.status = 'implemented';
  this.userResponse.implementationDate = new Date();
  this.userResponse.actualOutcome = outcome;
  this.status = 'archived';
  return this.save();
};

agiInsightSchema.methods.updateUserResponse = function(status, feedback = '') {
  this.userResponse.status = status;
  this.userResponse.feedback = feedback;
  return this.save();
};

agiInsightSchema.methods.extendExpiry = function(days = 30) {
  this.expiresAt = new Date(Date.now() + days * 24 * 60 * 60 * 1000);
  return this.save();
};

if (process.env.NODE_ENV === 'test') {
  const mockAGIInsightModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockAGIInsightId', ...data }),
    };
  };
  mockAGIInsightModel.find = jest.fn().mockResolvedValue([]);
  mockAGIInsightModel.findOne = jest.fn().mockResolvedValue(null);
  mockAGIInsightModel.findById = jest.fn().mockResolvedValue(null);
  mockAGIInsightModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockAGIInsightModel.create = jest.fn().mockResolvedValue({ _id: 'mockAGIInsightId' });
  module.exports = mockAGIInsightModel;
} else {
  module.exports = mongoose.model('AGIInsight', agiInsightSchema);
}