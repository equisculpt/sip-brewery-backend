const mongoose = require('mongoose');

const rewardSummarySchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  // Total rewards earned
  totalPoints: {
    type: Number,
    default: 0,
    min: 0
  },
  totalCashback: {
    type: Number,
    default: 0,
    min: 0
  },
  totalReferralBonus: {
    type: Number,
    default: 0,
    min: 0
  },
  // SIP tracking for cashback
  totalSipInstallments: {
    type: Number,
    default: 0,
    min: 0
  },
  completedFunds: [{
    fundName: String,
    folioNumber: String,
    sipCount: Number,
    completedAt: Date
  }],
  // Referral tracking
  totalReferrals: {
    type: Number,
    default: 0,
    min: 0
  },
  successfulReferrals: {
    type: Number,
    default: 0,
    min: 0
  },
  // Payout tracking
  totalPaidOut: {
    type: Number,
    default: 0,
    min: 0
  },
  pendingPayout: {
    type: Number,
    default: 0,
    min: 0
  },
  // Statistics
  lastRewardDate: Date,
  lastPayoutDate: Date,
  lastUpdated: {
    type: Date,
    default: Date.now
  },
  // Anti-abuse tracking
  referralAttempts: {
    type: Number,
    default: 0,
    min: 0
  },
  lastReferralDate: Date,
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Indexes for performance optimization
// rewardSummarySchema.index({ userId: 1 }); // Duplicate of inline index: true

rewardSummarySchema.index({ totalPoints: -1 });
rewardSummarySchema.index({ totalReferralBonus: -1 });
rewardSummarySchema.index({ pendingPayout: -1 });
rewardSummarySchema.index({ lastUpdated: -1 });

if (process.env.NODE_ENV === 'test') {
  const mockRewardSummaryModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockRewardSummaryId', ...data }),
    };
  };
  mockRewardSummaryModel.find = jest.fn().mockResolvedValue([]);
  mockRewardSummaryModel.findOne = jest.fn().mockResolvedValue(null);
  mockRewardSummaryModel.findById = jest.fn().mockResolvedValue(null);
  mockRewardSummaryModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockRewardSummaryModel.create = jest.fn().mockResolvedValue({ _id: 'mockRewardSummaryId' });
  module.exports = mockRewardSummaryModel;
} else {
  module.exports = mongoose.model('RewardSummary', rewardSummarySchema);
}