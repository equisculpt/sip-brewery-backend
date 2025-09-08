const mongoose = require('mongoose');

const transactionSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  transactionId: {
    type: String,
    required: true,
    unique: true
  },
  type: {
    type: String,
    enum: ['SIP', 'LUMPSUM', 'REDEMPTION', 'SWITCH_IN', 'SWITCH_OUT', 'DIVIDEND_PAYOUT', 'DIVIDEND_REINVEST'],
    required: true
  },
  schemeCode: {
    type: String,
    required: true
  },
  schemeName: {
    type: String,
    required: true
  },
  folio: {
    type: String,
    required: true
  },
  amount: {
    type: Number,
    required: true
  },
  units: {
    type: Number,
    required: true
  },
  nav: {
    type: Number,
    required: true
  },
  date: {
    type: Date,
    required: true
  },
  status: {
    type: String,
    enum: ['PENDING', 'PROCESSING', 'SUCCESS', 'FAILED', 'CANCELLED'],
    default: 'PENDING'
  },
  sipId: {
    type: String,
    ref: 'SIP'
  },
  orderType: {
    type: String,
    enum: ['BUY', 'SELL'],
    required: true
  },
  charges: {
    type: Number,
    default: 0
  },
  tax: {
    type: Number,
    default: 0
  },
  netAmount: {
    type: Number,
    required: true
  },
  remarks: {
    type: String
  },
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Generate transaction ID
transactionSchema.pre('save', function(next) {
  if (!this.transactionId) {
    const timestamp = Date.now().toString();
    const random = Math.random().toString(36).substr(2, 6).toUpperCase();
    this.transactionId = `TXN${timestamp}${random}`;
  }
  next();
});

// Add validation for referential integrity
transactionSchema.pre('save', async function(next) {
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

// Indexes for performance optimization
transactionSchema.index({ userId: 1 });
transactionSchema.index({ userId: 1, date: -1 });
transactionSchema.index({ userId: 1, schemeCode: 1 });
transactionSchema.index({ date: -1 });

if (process.env.NODE_ENV === 'test') {
  const mockTransactionModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockTransactionId', ...data }),
    };
  };
  mockTransactionModel.find = jest.fn().mockResolvedValue([]);
  mockTransactionModel.findOne = jest.fn().mockResolvedValue(null);
  mockTransactionModel.findById = jest.fn().mockResolvedValue(null);
  mockTransactionModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockTransactionModel.create = jest.fn().mockResolvedValue({ _id: 'mockTransactionId' });
  module.exports = mockTransactionModel;
} else {
  module.exports = mongoose.model('Transaction', transactionSchema);
}