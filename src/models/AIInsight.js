const mongoose = require('mongoose');

const aiInsightSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
    index: true
  },
  type: {
    type: String,
    enum: ['PORTFOLIO_ANALYSIS', 'RECOMMENDATION', 'RISK_ASSESSMENT', 'PERFORMANCE_REVIEW', 'MARKET_INSIGHT'],
    required: true
  },
  title: {
    type: String,
    required: true
  },
  summary: {
    type: String,
    required: true
  },
  details: {
    type: String,
    required: true
  },
  metrics: {
    xirr: Number,
    volatility: Number,
    sharpeRatio: Number,
    beta: Number,
    alpha: Number,
    maxDrawdown: Number,
    correlation: Number
  },
  recommendations: [{
    action: String,
    reason: String,
    priority: {
      type: String,
      enum: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    },
    fundCode: String,
    amount: Number
  }],
  riskFactors: [{
    factor: String,
    impact: {
      type: String,
      enum: ['LOW', 'MEDIUM', 'HIGH']
    },
    description: String
  }],
  marketContext: {
    marketTrend: String,
    sectorPerformance: Object,
    economicFactors: [String]
  },
  generatedAt: {
    type: Date,
    default: Date.now
  },
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Indexes for performance optimization
// aiInsightSchema.index({ insightId: 1 }); // Duplicate of inline index: true

aiInsightSchema.index({ userId: 1, type: 1 });
aiInsightSchema.index({ userId: 1, generatedAt: -1 });
aiInsightSchema.index({ generatedAt: -1 });

if (process.env.NODE_ENV === 'test') {
  const mockAIInsightModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockAIInsightId', ...data }),
    };
  };
  mockAIInsightModel.find = jest.fn().mockResolvedValue([]);
  mockAIInsightModel.findOne = jest.fn().mockResolvedValue(null);
  mockAIInsightModel.findById = jest.fn().mockResolvedValue(null);
  mockAIInsightModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockAIInsightModel.create = jest.fn().mockResolvedValue({ _id: 'mockAIInsightId' });
  module.exports = mockAIInsightModel;
} else {
  module.exports = mongoose.model('AIInsight', aiInsightSchema);
}