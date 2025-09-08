const mongoose = require('mongoose');

const rewardSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  type: {
    type: String,
    enum: ['REFERRAL_BONUS', 'SIP_LOYALTY_POINTS', 'CASHBACK_12_SIPS', 'SIP_COMPLETION'],
    required: true
  },
  amount: {
    type: Number,
    required: true,
    min: 0
  },
  points: {
    type: Number,
    default: 0,
    min: 0
  },
  description: {
    type: String,
    required: true
  },
  status: {
    type: String,
    enum: ['PENDING', 'CREDITED', 'REDEEMED', 'EXPIRED', 'REVOKED'],
    default: 'PENDING'
  },
  // SEBI Compliance fields
  isPaid: {
    type: Boolean,
    default: false
  },
  paidAt: {
    type: Date
  },
  paidBy: {
    type: String,
    ref: 'Admin'
  },
  // Transaction tracking
  sipId: {
    type: String,
    ref: 'Transaction'
  },
  referralId: {
    type: String,
    ref: 'Referral'
  },
  // Anti-abuse fields
  ipAddress: String,
  userAgent: String,
  // Expiry and validation
  expiryDate: {
    type: Date
  },
  validFrom: {
    type: Date,
    default: Date.now
  },
  // Metadata for compliance
  transactionTimestamp: {
    type: Date,
    default: Date.now
  },
  bseConfirmationId: String,
  folioNumber: String,
  fundName: String,
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Add validation for referential integrity
rewardSchema.pre('save', async function(next) {
  if (this.isNew || this.isModified('userId')) {
    try {
      const User = mongoose.model('User');
      const user = await User.findById(this.userId);
      if (!user) {
        throw new Error('Referenced user does not exist');
      }
    } catch (error) {
      return next(error);
    }
  }
  next();
});

// Explicit index declarations (no duplicates)
rewardSchema.index({ userId: 1 });
rewardSchema.index({ userId: 1, status: 1 });
rewardSchema.index({ userId: 1, type: 1 });
rewardSchema.index({ userId: 1, isPaid: 1 });
rewardSchema.index({ expiryDate: 1 });
rewardSchema.index({ sipId: 1 });
rewardSchema.index({ referralId: 1 });
rewardSchema.index({ transactionTimestamp: 1 });

if (process.env.NODE_ENV === 'test') {
  const mockRewardModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockRewardId', ...data }),
    };
  };
  mockRewardModel.find = jest.fn().mockResolvedValue([]);
  mockRewardModel.findOne = jest.fn().mockResolvedValue(null);
  mockRewardModel.findById = jest.fn().mockResolvedValue(null);
  mockRewardModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockRewardModel.create = jest.fn().mockResolvedValue({ _id: 'mockRewardId' });
  module.exports = mockRewardModel;
} else {
  module.exports = mongoose.model('Reward', rewardSchema);
}