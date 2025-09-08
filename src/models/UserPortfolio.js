const mongoose = require('mongoose');

const userPortfolioSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true,
    index: true
  },
  funds: [{
    schemeCode: {
      type: String,
      required: true
    },
    schemeName: {
      type: String,
      required: true
    },
    investedValue: {
      type: Number,
      required: true,
      min: 0
    },
    currentValue: {
      type: Number,
      required: true,
      min: 0
    },
    units: {
      type: Number,
      required: true,
      min: 0
    },
    startDate: {
      type: Date,
      required: true
    },
    lastNav: {
      type: Number,
      required: true,
      min: 0
    },
    lastNavDate: {
      type: Date,
      required: true
    }
  }],
  // XIRR calculations for different time periods
  xirr1M: {
    type: Number,
    default: 0
  },
  xirr3M: {
    type: Number,
    default: 0
  },
  xirr6M: {
    type: Number,
    default: 0
  },
  xirr1Y: {
    type: Number,
    default: 0
  },
  xirr3Y: {
    type: Number,
    default: 0
  },
  // Total portfolio values
  totalInvested: {
    type: Number,
    default: 0
  },
  totalCurrentValue: {
    type: Number,
    default: 0
  },
  // Portfolio allocation percentages (calculated)
  allocation: {
    type: Map,
    of: Number
  },
  // Performance metrics
  performance: {
    absoluteReturn: {
      type: Number,
      default: 0
    },
    absoluteReturnPercent: {
      type: Number,
      default: 0
    },
    bestPerformingFund: {
      schemeCode: String,
      schemeName: String,
      returnPercent: Number
    },
    worstPerformingFund: {
      schemeCode: String,
      schemeName: String,
      returnPercent: Number
    }
  },
  // Transaction history for XIRR calculation
  transactions: [{
    date: {
      type: Date,
      required: true
    },
    type: {
      type: String,
      enum: ['SIP', 'LUMPSUM', 'REDEMPTION'],
      required: true
    },
    schemeCode: {
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
    }
  }],
  // Leaderboard history
  leaderboardHistory: [{
    duration: {
      type: String,
      enum: ['1M', '3M', '6M', '1Y', '3Y'],
      required: true
    },
    rank: {
      type: Number,
      required: true
    },
    returnPercent: {
      type: Number,
      required: true
    },
    date: {
      type: Date,
      default: Date.now
    }
  }],
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Pre-save middleware to calculate totals and allocation
userPortfolioSchema.pre('save', function(next) {
  // Calculate total invested and current values
  this.totalInvested = this.funds.reduce((sum, fund) => sum + fund.investedValue, 0);
  this.totalCurrentValue = this.funds.reduce((sum, fund) => sum + fund.currentValue, 0);
  
  // Calculate allocation percentages
  if (this.totalCurrentValue > 0) {
    this.allocation = new Map();
    this.funds.forEach(fund => {
      const percentage = (fund.currentValue / this.totalCurrentValue) * 100;
      this.allocation.set(fund.schemeName, Math.round(percentage * 100) / 100);
    });
  }
  
  // Calculate absolute return
  if (this.totalInvested > 0) {
    this.performance.absoluteReturn = this.totalCurrentValue - this.totalInvested;
    this.performance.absoluteReturnPercent = (this.performance.absoluteReturn / this.totalInvested) * 100;
  }
  
  // Find best and worst performing funds
  if (this.funds.length > 0) {
    const fundReturns = this.funds.map(fund => ({
      schemeCode: fund.schemeCode,
      schemeName: fund.schemeName,
      returnPercent: fund.investedValue > 0 ? ((fund.currentValue - fund.investedValue) / fund.investedValue) * 100 : 0
    }));
    
    const bestFund = fundReturns.reduce((best, current) => 
      current.returnPercent > best.returnPercent ? current : best
    );
    
    const worstFund = fundReturns.reduce((worst, current) => 
      current.returnPercent < worst.returnPercent ? current : worst
    );
    
    this.performance.bestPerformingFund = bestFund;
    this.performance.worstPerformingFund = worstFund;
  }
  
  next();
});

// Method to add transaction
userPortfolioSchema.methods.addTransaction = function(transaction) {
  this.transactions.push(transaction);
  return this.save();
};

// Method to update fund values
userPortfolioSchema.methods.updateFundValue = function(schemeCode, currentValue, nav, navDate) {
  const fund = this.funds.find(f => f.schemeCode === schemeCode);
  if (fund) {
    fund.currentValue = currentValue;
    fund.lastNav = nav;
    fund.lastNavDate = navDate;
    fund.units = currentValue / nav;
  }
  return this.save();
};

// Method to add fund
userPortfolioSchema.methods.addFund = function(fundData) {
  this.funds.push(fundData);
  return this.save();
};

// Method to remove fund
userPortfolioSchema.methods.removeFund = function(schemeCode) {
  this.funds = this.funds.filter(f => f.schemeCode !== schemeCode);
  return this.save();
};

// Method to get allocation as object
userPortfolioSchema.methods.getAllocationObject = function() {
  const allocation = {};
  this.allocation.forEach((value, key) => {
    allocation[key] = value;
  });
  return allocation;
};

// Method to add leaderboard history
userPortfolioSchema.methods.addLeaderboardHistory = function(duration, rank, returnPercent) {
  this.leaderboardHistory.push({
    duration,
    rank,
    returnPercent,
    date: new Date()
  });
  
  // Keep only last 50 entries
  if (this.leaderboardHistory.length > 50) {
    this.leaderboardHistory = this.leaderboardHistory.slice(-50);
  }
  
  return this.save();
};

// Indexes for performance optimization
// userPortfolioSchema.index({ userId: 1 }); // Duplicate of inline index: true

userPortfolioSchema.index({ 'funds.schemeCode': 1 });
userPortfolioSchema.index({ xirr1M: -1 });
userPortfolioSchema.index({ xirr3M: -1 });
userPortfolioSchema.index({ xirr6M: -1 });
userPortfolioSchema.index({ xirr1Y: -1 });
userPortfolioSchema.index({ xirr3Y: -1 });
userPortfolioSchema.index({ isActive: 1 });
userPortfolioSchema.index({ updatedAt: -1 });

if (process.env.NODE_ENV === 'test') {
  const mockUserPortfolioModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockPortfolioId', ...data }),
    };
  };
  mockUserPortfolioModel.find = jest.fn().mockResolvedValue([]);
  mockUserPortfolioModel.findOne = jest.fn().mockResolvedValue(null);
  mockUserPortfolioModel.findById = jest.fn().mockResolvedValue(null);
  mockUserPortfolioModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockUserPortfolioModel.create = jest.fn().mockResolvedValue({ _id: 'mockPortfolioId' });
  module.exports = mockUserPortfolioModel;
} else {
  module.exports = mongoose.model('UserPortfolio', userPortfolioSchema);
}