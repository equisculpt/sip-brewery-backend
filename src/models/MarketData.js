const mongoose = require('mongoose');

const marketDataSchema = new mongoose.Schema({
  date: {
    type: Date,
    required: true,
    index: true
  },
  nifty50: {
    open: { type: Number, required: true },
    high: { type: Number, required: true },
    low: { type: Number, required: true },
    close: { type: Number, required: true },
    volume: { type: Number, required: true },
    change: { type: Number, required: true },
    changePercent: { type: Number, required: true }
  },
  sensex: {
    open: { type: Number, required: true },
    high: { type: Number, required: true },
    low: { type: Number, required: true },
    close: { type: Number, required: true },
    volume: { type: Number, required: true },
    change: { type: Number, required: true },
    changePercent: { type: Number, required: true }
  },
  bankNifty: {
    open: { type: Number, required: true },
    high: { type: Number, required: true },
    low: { type: Number, required: true },
    close: { type: Number, required: true },
    volume: { type: Number, required: true },
    change: { type: Number, required: true },
    changePercent: { type: Number, required: true }
  },
  sectoralIndices: {
    it: {
      close: { type: Number, required: true },
      changePercent: { type: Number, required: true }
    },
    pharma: {
      close: { type: Number, required: true },
      changePercent: { type: Number, required: true }
    },
    auto: {
      close: { type: Number, required: true },
      changePercent: { type: Number, required: true }
    },
    metal: {
      close: { type: Number, required: true },
      changePercent: { type: Number, required: true }
    },
    realty: {
      close: { type: Number, required: true },
      changePercent: { type: Number, required: true }
    },
    energy: {
      close: { type: Number, required: true },
      changePercent: { type: Number, required: true }
    }
  },
  marketSentiment: {
    overall: {
      type: String,
      enum: ['bullish', 'bearish', 'neutral', 'volatile'],
      required: true
    },
    confidence: {
      type: Number,
      min: 0,
      max: 1,
      required: true
    },
    volatility: {
      type: String,
      enum: ['low', 'medium', 'high'],
      required: true
    }
  },
  technicalIndicators: {
    rsi: {
      nifty50: { type: Number, min: 0, max: 100 },
      sensex: { type: Number, min: 0, max: 100 },
      bankNifty: { type: Number, min: 0, max: 100 }
    },
    macd: {
      nifty50: {
        macd: Number,
        signal: Number,
        histogram: Number
      },
      sensex: {
        macd: Number,
        signal: Number,
        histogram: Number
      },
      bankNifty: {
        macd: Number,
        signal: Number,
        histogram: Number
      }
    },
    movingAverages: {
      nifty50: {
        sma20: Number,
        sma50: Number,
        sma200: Number
      },
      sensex: {
        sma20: Number,
        sma50: Number,
        sma200: Number
      },
      bankNifty: {
        sma20: Number,
        sma50: Number,
        sma200: Number
      }
    }
  },
  volumeAnalysis: {
    totalVolume: { type: Number, required: true },
    advanceDecline: {
      advances: { type: Number, required: true },
      declines: { type: Number, required: true },
      unchanged: { type: Number, required: true }
    },
    fiiDii: {
      fiiNetBuy: { type: Number, required: true },
      diiNetBuy: { type: Number, required: true }
    }
  },
  globalMarkets: {
    usMarkets: {
      sp500: { type: Number, required: true },
      nasdaq: { type: Number, required: true },
      dowJones: { type: Number, required: true }
    },
    asianMarkets: {
      nikkei: { type: Number, required: true },
      hangSeng: { type: Number, required: true },
      shanghai: { type: Number, required: true }
    },
    europeanMarkets: {
      ftse: { type: Number, required: true },
      dax: { type: Number, required: true },
      cac: { type: Number, required: true }
    }
  },
  currency: {
    usdInr: { type: Number, required: true },
    eurInr: { type: Number, required: true },
    gbpInr: { type: Number, required: true }
  },
  commodities: {
    gold: { type: Number, required: true },
    silver: { type: Number, required: true },
    crudeOil: { type: Number, required: true }
  },
  metadata: {
    source: {
      type: String,
      enum: ['nse', 'bse', 'yahoo_finance', 'alpha_vantage', 'manual'],
      default: 'nse'
    },
    lastUpdated: {
      type: Date,
      default: Date.now
    },
    dataQuality: {
      type: String,
      enum: ['high', 'medium', 'low'],
      default: 'high'
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for better query performance
marketDataSchema.index({ date: -1 });
marketDataSchema.index({ 'marketSentiment.overall': 1, date: -1 });
marketDataSchema.index({ 'nifty50.close': 1, date: -1 });

// Virtual for market trend
marketDataSchema.virtual('marketTrend').get(function() {
  const niftyChange = this.nifty50.changePercent;
  if (niftyChange > 1) return 'strong_bullish';
  if (niftyChange > 0.5) return 'bullish';
  if (niftyChange > -0.5) return 'sideways';
  if (niftyChange > -1) return 'bearish';
  return 'strong_bearish';
});

// Virtual for volatility level
marketDataSchema.virtual('volatilityLevel').get(function() {
  const highLowRange = (this.nifty50.high - this.nifty50.low) / this.nifty50.close * 100;
  if (highLowRange > 3) return 'high';
  if (highLowRange > 1.5) return 'medium';
  return 'low';
});

// Static methods
marketDataSchema.statics.getLatestData = function() {
  return this.findOne().sort({ date: -1 });
};

marketDataSchema.statics.getDataForPeriod = function(startDate, endDate) {
  return this.find({
    date: { $gte: startDate, $lte: endDate }
  }).sort({ date: 1 });
};

marketDataSchema.statics.getMarketTrend = function(days = 30) {
  const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
  return this.find({
    date: { $gte: startDate }
  }).sort({ date: 1 });
};

// Instance methods
marketDataSchema.methods.calculateReturns = function(period = 1) {
  // Calculate returns for different periods
  return {
    daily: this.nifty50.changePercent,
    weekly: null, // Would need historical data
    monthly: null  // Would need historical data
  };
};

marketDataSchema.methods.getSectorPerformance = function() {
  const sectors = Object.keys(this.sectoralIndices);
  return sectors.map(sector => ({
    sector,
    change: this.sectoralIndices[sector].changePercent,
    performance: this.sectoralIndices[sector].changePercent > 0 ? 'positive' : 'negative'
  })).sort((a, b) => b.change - a.change);
};

if (process.env.NODE_ENV === 'test') {
  const mockMarketDataModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockMarketDataId', ...data }),
    };
  };
  mockMarketDataModel.find = jest.fn().mockResolvedValue([]);
  mockMarketDataModel.findOne = jest.fn().mockResolvedValue(null);
  mockMarketDataModel.findById = jest.fn().mockResolvedValue(null);
  mockMarketDataModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockMarketDataModel.create = jest.fn().mockResolvedValue({ _id: 'mockMarketDataId' });
  module.exports = mockMarketDataModel;
} else {
  module.exports = mongoose.model('MarketData', marketDataSchema);
}