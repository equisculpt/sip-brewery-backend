const mongoose = require('mongoose');

const economicIndicatorSchema = new mongoose.Schema({
  date: {
    type: Date,
    required: true,
    index: true
  },
  inflation: {
    cpi: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      yearOverYear: { type: Number, required: true }
    },
    wpi: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      yearOverYear: { type: Number, required: true }
    },
    coreInflation: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  interestRates: {
    repoRate: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      lastChangeDate: Date
    },
    reverseRepoRate: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    mclr: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    gSec10Y: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  gdp: {
    growthRate: {
      value: { type: Number, required: true },
      quarter: { type: String, required: true },
      year: { type: Number, required: true }
    },
    nominal: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    perCapita: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  fiscalPolicy: {
    fiscalDeficit: {
      value: { type: Number, required: true },
      percentageOfGDP: { type: Number, required: true }
    },
    revenueDeficit: {
      value: { type: Number, required: true },
      percentageOfGDP: { type: Number, required: true }
    },
    primaryDeficit: {
      value: { type: Number, required: true },
      percentageOfGDP: { type: Number, required: true }
    }
  },
  externalSector: {
    currentAccountDeficit: {
      value: { type: Number, required: true },
      percentageOfGDP: { type: Number, required: true }
    },
    tradeBalance: {
      exports: { type: Number, required: true },
      imports: { type: Number, required: true },
      balance: { type: Number, required: true }
    },
    forexReserves: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  employment: {
    unemploymentRate: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    laborForceParticipation: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    employmentToPopulation: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  manufacturing: {
    pmi: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      expansion: { type: Boolean, required: true }
    },
    iip: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      yearOverYear: { type: Number, required: true }
    },
    capacityUtilization: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  services: {
    servicesPMI: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      expansion: { type: Boolean, required: true }
    },
    servicesGrowth: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  banking: {
    creditGrowth: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    depositGrowth: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    },
    npaRatio: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  globalFactors: {
    crudeOilPrice: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      currency: { type: String, default: 'USD' }
    },
    goldPrice: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      currency: { type: String, default: 'USD' }
    },
    dollarIndex: {
      value: { type: Number, required: true },
      change: { type: Number, required: true }
    }
  },
  sentiment: {
    consumerConfidence: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      level: {
        type: String,
        enum: ['very_low', 'low', 'neutral', 'high', 'very_high'],
        required: true
      }
    },
    businessConfidence: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      level: {
        type: String,
        enum: ['very_low', 'low', 'neutral', 'high', 'very_high'],
        required: true
      }
    },
    investorSentiment: {
      value: { type: Number, required: true },
      change: { type: Number, required: true },
      level: {
        type: String,
        enum: ['very_bearish', 'bearish', 'neutral', 'bullish', 'very_bullish'],
        required: true
      }
    }
  },
  metadata: {
    source: {
      type: String,
      enum: ['rbi', 'cso', 'nso', 'world_bank', 'imf', 'manual'],
      default: 'rbi'
    },
    lastUpdated: {
      type: Date,
      default: Date.now
    },
    dataQuality: {
      type: String,
      enum: ['high', 'medium', 'low'],
      default: 'high'
    },
    frequency: {
      type: String,
      enum: ['daily', 'weekly', 'monthly', 'quarterly', 'yearly'],
      default: 'monthly'
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for better query performance
economicIndicatorSchema.index({ date: -1 });
economicIndicatorSchema.index({ 'inflation.cpi.value': 1, date: -1 });
economicIndicatorSchema.index({ 'interestRates.repoRate.value': 1, date: -1 });

// Virtual for economic health score
economicIndicatorSchema.virtual('economicHealthScore').get(function() {
  let score = 50; // Base score
  
  // Inflation impact (lower is better)
  if (this.inflation.cpi.value < 4) score += 10;
  else if (this.inflation.cpi.value < 6) score += 5;
  else if (this.inflation.cpi.value > 8) score -= 10;
  
  // GDP growth impact (higher is better)
  if (this.gdp.growthRate.value > 7) score += 15;
  else if (this.gdp.growthRate.value > 5) score += 10;
  else if (this.gdp.growthRate.value < 3) score -= 10;
  
  // Interest rates impact (moderate is better)
  if (this.interestRates.repoRate.value > 6 && this.interestRates.repoRate.value < 8) score += 5;
  else if (this.interestRates.repoRate.value > 8) score -= 5;
  
  // Employment impact
  if (this.employment.unemploymentRate.value < 5) score += 10;
  else if (this.employment.unemploymentRate.value > 8) score -= 10;
  
  return Math.max(0, Math.min(100, score));
});

// Virtual for economic trend
economicIndicatorSchema.virtual('economicTrend').get(function() {
  const score = this.economicHealthScore;
  if (score >= 80) return 'excellent';
  if (score >= 60) return 'good';
  if (score >= 40) return 'moderate';
  if (score >= 20) return 'poor';
  return 'critical';
});

// Static methods
economicIndicatorSchema.statics.getLatestData = function() {
  return this.findOne().sort({ date: -1 });
};

economicIndicatorSchema.statics.getDataForPeriod = function(startDate, endDate) {
  return this.find({
    date: { $gte: startDate, $lte: endDate }
  }).sort({ date: 1 });
};

economicIndicatorSchema.statics.getEconomicTrend = function(days = 365) {
  const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
  return this.find({
    date: { $gte: startDate }
  }).sort({ date: 1 });
};

// Instance methods
economicIndicatorSchema.methods.getInflationTrend = function() {
  const cpi = this.inflation.cpi.value;
  if (cpi < 4) return 'low';
  if (cpi < 6) return 'moderate';
  if (cpi < 8) return 'high';
  return 'very_high';
};

economicIndicatorSchema.methods.getInterestRateEnvironment = function() {
  const repoRate = this.interestRates.repoRate.value;
  if (repoRate < 4) return 'accommodative';
  if (repoRate < 6) return 'neutral';
  if (repoRate < 8) return 'tightening';
  return 'very_tight';
};

economicIndicatorSchema.methods.getGrowthOutlook = function() {
  const gdpGrowth = this.gdp.growthRate.value;
  if (gdpGrowth > 7) return 'strong';
  if (gdpGrowth > 5) return 'moderate';
  if (gdpGrowth > 3) return 'weak';
  return 'recession';
};

if (process.env.NODE_ENV === 'test') {
  const mockEconomicIndicatorModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockEconomicIndicatorId', ...data }),
    };
  };
  mockEconomicIndicatorModel.find = jest.fn().mockResolvedValue([]);
  mockEconomicIndicatorModel.findOne = jest.fn().mockResolvedValue(null);
  mockEconomicIndicatorModel.findById = jest.fn().mockResolvedValue(null);
  mockEconomicIndicatorModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockEconomicIndicatorModel.create = jest.fn().mockResolvedValue({ _id: 'mockEconomicIndicatorId' });
  module.exports = mockEconomicIndicatorModel;
} else {
  module.exports = mongoose.model('EconomicIndicator', economicIndicatorSchema);
}