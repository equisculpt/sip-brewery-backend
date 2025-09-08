const mongoose = require('mongoose');

const holdingSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
    index: true
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
  units: {
    type: Number,
    required: true,
    default: 0
  },
  currentNav: {
    type: Number,
    required: true
  },
  value: {
    type: Number,
    required: true,
    default: 0
  },
  invested: {
    type: Number,
    required: true,
    default: 0
  },
  returns: {
    type: Number,
    required: true,
    default: 0
  },
  returnsPercentage: {
    type: Number,
    required: true,
    default: 0
  },
  sipStatus: {
    type: String,
    enum: ['ACTIVE', 'PAUSED', 'STOPPED'],
    default: 'ACTIVE'
  },
  sipAmount: {
    type: Number,
    default: 0
  },
  sipDate: {
    type: Number,
    default: 1 // Day of month
  },
  category: {
    type: String,
    required: true
  },
  fundHouse: {
    type: String,
    required: true
  },
  riskLevel: {
    type: String,
    enum: ['Low', 'Moderate', 'High'],
    default: 'Moderate'
  },
  lastUpdated: {
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

// Calculate value and returns before saving
holdingSchema.pre('save', function(next) {
  this.value = this.units * this.currentNav;
  this.returns = this.value - this.invested;
  this.returnsPercentage = this.invested > 0 ? (this.returns / this.invested) * 100 : 0;
  next();
});

// Indexes for performance optimization
// holdingSchema.index({ holdingId: 1 }); // Duplicate of inline index: true

holdingSchema.index({ userId: 1, schemeCode: 1 });
holdingSchema.index({ userId: 1, isActive: 1 });
holdingSchema.index({ schemeCode: 1 });

if (process.env.NODE_ENV === 'test') {
  const mockHoldingModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockHoldingId', ...data }),
    };
  };
  mockHoldingModel.find = jest.fn().mockResolvedValue([]);
  mockHoldingModel.findOne = jest.fn().mockResolvedValue(null);
  mockHoldingModel.findById = jest.fn().mockResolvedValue(null);
  mockHoldingModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockHoldingModel.create = jest.fn().mockResolvedValue({ _id: 'mockHoldingId' });
  module.exports = mockHoldingModel;
} else {
  module.exports = mongoose.model('Holding', holdingSchema);
}