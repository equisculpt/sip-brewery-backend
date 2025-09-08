const mongoose = require('mongoose');

const portfolioCopySchema = new mongoose.Schema({
  sourceSecretCode: {
    type: String,
    required: true,
    index: true
  },
  sourceUserId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  targetUserId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  investmentType: {
    type: String,
    enum: ['SIP', 'LUMPSUM'],
    required: true
  },
  averageSip: {
    type: Number,
    min: 1000,
    max: 100000
  },
  copiedAllocation: {
    type: Map,
    of: Number
  },
  sourceReturnPercent: {
    type: Number,
    required: true
  },
  duration: {
    type: String,
    enum: ['1M', '3M', '6M', '1Y', '3Y'],
    required: true
  },
  status: {
    type: String,
    enum: ['PENDING', 'EXECUTED', 'FAILED', 'CANCELLED'],
    default: 'PENDING'
  },
  executionDetails: {
    executedAt: Date,
    executedAmount: Number,
    executedFunds: [{
      schemeCode: String,
      schemeName: String,
      amount: Number,
      units: Number,
      nav: Number
    }],
    errorMessage: String
  },
  metadata: {
    userAgent: String,
    ipAddress: String,
    deviceType: String
  },
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Method to get allocation as object
portfolioCopySchema.methods.getAllocationObject = function() {
  const allocation = {};
  this.copiedAllocation.forEach((value, key) => {
    allocation[key] = value;
  });
  return allocation;
};

// Method to update execution details
portfolioCopySchema.methods.updateExecution = function(executionDetails) {
  this.executionDetails = executionDetails;
  this.status = 'EXECUTED';
  return this.save();
};

// Method to mark as failed
portfolioCopySchema.methods.markFailed = function(errorMessage) {
  this.status = 'FAILED';
  this.executionDetails.errorMessage = errorMessage;
  return this.save();
};

// Indexes for performance optimization
// portfolioCopySchema.index({ copyId: 1 }); // Duplicate of inline index: true
// portfolioCopySchema.index({ sourceSecretCode: 1 }); // Duplicate of inline index: true

portfolioCopySchema.index({ sourceUserId: 1 });
portfolioCopySchema.index({ status: 1 });
portfolioCopySchema.index({ createdAt: -1 });
portfolioCopySchema.index({ isActive: 1 });

if (process.env.NODE_ENV === 'test') {
  const mockPortfolioCopyModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockPortfolioCopyId', ...data }),
    };
  };
  mockPortfolioCopyModel.find = jest.fn().mockResolvedValue([]);
  mockPortfolioCopyModel.findOne = jest.fn().mockResolvedValue(null);
  mockPortfolioCopyModel.findById = jest.fn().mockResolvedValue(null);
  mockPortfolioCopyModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockPortfolioCopyModel.create = jest.fn().mockResolvedValue({ _id: 'mockPortfolioCopyId' });
  module.exports = mockPortfolioCopyModel;
} else {
  module.exports = mongoose.model('PortfolioCopy', portfolioCopySchema);
}