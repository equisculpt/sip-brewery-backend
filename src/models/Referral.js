const mongoose = require('mongoose');

const referralSchema = new mongoose.Schema({
  referrerId: {
    type: String,
    required: true,
    ref: 'User',
    index: true
  },
  referredId: {
    type: String,
    required: true,
    ref: 'User',
    index: true
  },
  referralCode: {
    type: String,
    required: true,
    index: true
  },
  status: {
    type: String,
    enum: ['PENDING', 'KYC_COMPLETED', 'SIP_STARTED', 'BONUS_PAID', 'EXPIRED', 'CANCELLED'],
    default: 'PENDING'
  },
  // Anti-abuse fields
  ipAddress: String,
  userAgent: String,
  deviceFingerprint: String,
  // Timeline tracking
  referredAt: {
    type: Date,
    default: Date.now
  },
  kycCompletedAt: Date,
  sipStartedAt: Date,
  bonusPaidAt: Date,
  // SIP tracking for bonus validation
  firstSipId: {
    type: String,
    ref: 'Transaction'
  },
  sipCancelledAt: Date,
  // Bonus details
  bonusAmount: {
    type: Number,
    default: 100,
    min: 0
  },
  bonusPaid: {
    type: Boolean,
    default: false
  },
  bonusTransactionId: {
    type: String,
    ref: 'Reward'
  },
  // Validation fields
  isSelfReferral: {
    type: Boolean,
    default: false
  },
  isEligible: {
    type: Boolean,
    default: true
  },
  validationNotes: String,
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Indexes for performance optimization
// referralSchema.index({ referralId: 1 }); // Duplicate of inline index: true
// referralSchema.index({ referralCode: 1 }); // Duplicate of inline index: true
// referralSchema.index({ referredId: 1 }); // Duplicate of inline index: true

referralSchema.index({ referrerId: 1, status: 1 });
referralSchema.index({ status: 1 });
referralSchema.index({ referredAt: -1 });
referralSchema.index({ bonusPaid: 1 });

// Compound index for efficient queries
referralSchema.index({ referrerId: 1, referredAt: 1 });

if (process.env.NODE_ENV === 'test') {
  const mockReferralModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockReferralId', ...data }),
    };
  };
  mockReferralModel.find = jest.fn().mockResolvedValue([]);
  mockReferralModel.findOne = jest.fn().mockResolvedValue(null);
  mockReferralModel.findById = jest.fn().mockResolvedValue(null);
  mockReferralModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockReferralModel.create = jest.fn().mockResolvedValue({ _id: 'mockReferralId' });
  module.exports = mockReferralModel;
} else {
  module.exports = mongoose.model('Referral', referralSchema);
}