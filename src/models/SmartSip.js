const mongoose = require('mongoose');

const smartSipSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
    index: true
  },
  sipType: {
    type: String,
    enum: ['STATIC', 'SMART'],
    required: true,
    default: 'STATIC'
  },
  averageSip: {
    type: Number,
    required: true,
    min: 1000,
    max: 100000
  },
  minSip: {
    type: Number,
    required: true,
    min: 500,
    max: 100000
  },
  maxSip: {
    type: Number,
    required: true,
    min: 1000,
    max: 100000
  },
  fundSelection: [{
    schemeCode: {
      type: String,
      required: true
    },
    schemeName: {
      type: String,
      required: true
    },
    allocation: {
      type: Number,
      required: true,
      min: 0,
      max: 100
    }
  }],
  lastSIPAmount: {
    type: Number,
    default: 0
  },
  nextSIPDate: {
    type: Date,
    required: true
  },
  sipDay: {
    type: Number,
    required: true,
    min: 1,
    max: 31,
    default: 1
  },
  status: {
    type: String,
    enum: ['ACTIVE', 'PAUSED', 'STOPPED'],
    default: 'ACTIVE'
  },
  sipHistory: [{
    date: {
      type: Date,
      required: true
    },
    amount: {
      type: Number,
      required: true
    },
    marketScore: {
      type: Number,
      min: -1,
      max: 1
    },
    marketReason: {
      type: String
    },
    fundSplit: {
      type: Map,
      of: Number
    },
    executed: {
      type: Boolean,
      default: false
    },
    transactionId: {
      type: String
    }
  }],
  marketAnalysis: {
    lastUpdated: {
      type: Date,
      default: Date.now
    },
    currentScore: {
      type: Number,
      min: -1,
      max: 1
    },
    currentReason: {
      type: String
    },
    recommendedAmount: {
      type: Number
    },
    indicators: {
      peRatio: Number,
      rsi: Number,
      breakout: Boolean,
      sentiment: String,
      fearGreedIndex: Number
    }
  },
  preferences: {
    riskTolerance: {
      type: String,
      enum: ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'],
      default: 'MODERATE'
    },
    marketTiming: {
      type: Boolean,
      default: true
    },
    aiEnabled: {
      type: Boolean,
      default: true
    },
    notifications: {
      type: Boolean,
      default: true
    }
  },
  performance: {
    totalInvested: {
      type: Number,
      default: 0
    },
    totalSIPs: {
      type: Number,
      default: 0
    },
    averageAmount: {
      type: Number,
      default: 0
    },
    bestSIPAmount: {
      type: Number,
      default: 0
    },
    worstSIPAmount: {
      type: Number,
      default: 0
    }
  },
  isActive: {
    type: Boolean,
    default: true
  },
  isSipOnly: {
    type: Boolean,
    default: false
  }
}, {
  timestamps: true
});

// Pre-save middleware to calculate min/max SIP amounts
smartSipSchema.pre('save', function(next) {
  if (this.sipType === 'SMART' && this.averageSip) {
    // Calculate min and max based on average SIP
    this.minSip = Math.round(this.averageSip * 0.8);
    this.maxSip = Math.round(this.averageSip * 1.2);
  }
  next();
});

// Method to add SIP to history
smartSipSchema.methods.addSIPToHistory = function(amount, marketScore, marketReason, fundSplit) {
  this.sipHistory.push({
    date: new Date(),
    amount: amount,
    marketScore: marketScore,
    marketReason: marketReason,
    fundSplit: fundSplit,
    executed: false
  });
  
  // Update performance metrics
  this.performance.totalInvested += amount;
  this.performance.totalSIPs += 1;
  this.performance.averageAmount = this.performance.totalInvested / this.performance.totalSIPs;
  
  if (amount > this.performance.bestSIPAmount) {
    this.performance.bestSIPAmount = amount;
  }
  
  if (this.performance.worstSIPAmount === 0 || amount < this.performance.worstSIPAmount) {
    this.performance.worstSIPAmount = amount;
  }
  
  this.lastSIPAmount = amount;
  this.nextSIPDate = this.calculateNextSIPDate();
  
  return this.save();
};

// Method to calculate next SIP date
smartSipSchema.methods.calculateNextSIPDate = function() {
  const today = new Date();
  let nextDate = new Date(today.getFullYear(), today.getMonth() + 1, this.sipDay);
  
  // If the day doesn't exist in the month (e.g., 31st in February), use last day
  if (nextDate.getMonth() !== (today.getMonth() + 1) % 12) {
    nextDate = new Date(today.getFullYear(), today.getMonth() + 2, 0);
  }
  
  return nextDate;
};

// Method to get current recommendation
smartSipSchema.methods.getCurrentRecommendation = function() {
  if (this.sipType === 'STATIC') {
    return {
      amount: this.averageSip,
      marketScore: 0,
      reason: 'Static SIP - Fixed amount',
      fundSplit: this.fundSelection.reduce((acc, fund) => {
        acc[fund.schemeName] = fund.allocation;
        return acc;
      }, {})
    };
  }
  
  // For SMART SIP, use the latest market analysis
  const marketScore = this.marketAnalysis.currentScore || 0;
  const reason = this.marketAnalysis.currentReason || 'Market analysis pending';
  
  // Calculate recommended amount based on market score
  let recommendedAmount = this.averageSip;
  
  if (marketScore > 0.5) {
    // Market is bottomed - invest maximum
    recommendedAmount = this.maxSip;
  } else if (marketScore < -0.5) {
    // Market is expensive - invest minimum
    recommendedAmount = this.minSip;
  } else {
    // Interpolate between min and max
    const range = this.maxSip - this.minSip;
    const normalizedScore = (marketScore + 1) / 2; // Convert -1 to 1 range to 0 to 1
    recommendedAmount = this.minSip + (range * normalizedScore);
  }
  
  // Round to nearest 100
  recommendedAmount = Math.round(recommendedAmount / 100) * 100;
  
  return {
    amount: recommendedAmount,
    marketScore: marketScore,
    reason: reason,
    fundSplit: this.fundSelection.reduce((acc, fund) => {
      acc[fund.schemeName] = fund.allocation;
      return acc;
    }, {})
  };
};

// Indexes for efficient queries
smartSipSchema.index({ userId: 1, status: 1 });
smartSipSchema.index({ userId: 1, sipType: 1 });
smartSipSchema.index({ nextSIPDate: 1 });
smartSipSchema.index({ status: 1, isActive: 1 });

if (process.env.NODE_ENV === 'test') {
  const mockSmartSipModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockSmartSipId', ...data }),
    };
  };
  mockSmartSipModel.find = jest.fn().mockResolvedValue([]);
  mockSmartSipModel.findOne = jest.fn().mockResolvedValue(null);
  mockSmartSipModel.findById = jest.fn().mockResolvedValue(null);
  mockSmartSipModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockSmartSipModel.create = jest.fn().mockResolvedValue({ _id: 'mockSmartSipId' });
  module.exports = mockSmartSipModel;
} else {
  module.exports = mongoose.model('SmartSip', smartSipSchema);
}