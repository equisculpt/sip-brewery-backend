const mongoose = require('mongoose');

const mfOrderSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
    index: true
  },
  idempotencyKey: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  orderType: {
    type: String,
    enum: ['LUMPSUM', 'SIP', 'REDEMPTION', 'SWITCH'],
    required: true
  },
  schemeCode: {
    type: String,
    required: true,
    index: true
  },
  schemeName: {
    type: String
  },
  amount: {
    type: Number,
    required: true
  },
  units: {
    type: Number
  },
  folioNumber: {
    type: String
  },
  bseOrderId: {
    type: String,
    index: true
  },
  bseClientCode: {
    type: String
  },
  status: {
    type: String,
    enum: ['PENDING', 'SUBMITTED', 'ACCEPTED', 'REJECTED', 'COMPLETED', 'FAILED', 'CANCELLED'],
    default: 'PENDING',
    index: true
  },
  bseStatus: {
    type: String
  },
  bseResponse: {
    type: mongoose.Schema.Types.Mixed
  },
  nav: {
    type: Number
  },
  allottedUnits: {
    type: Number
  },
  allottedAmount: {
    type: Number
  },
  settlementDate: {
    type: Date
  },
  paymentMode: {
    type: String,
    enum: ['ONLINE', 'CHEQUE', 'DD', 'NEFT', 'RTGS'],
    default: 'ONLINE'
  },
  sipDetails: {
    frequency: {
      type: String,
      enum: ['MONTHLY', 'QUARTERLY', 'WEEKLY', 'DAILY']
    },
    startDate: Date,
    endDate: Date,
    installments: Number,
    mandateId: String
  },
  redemptionDetails: {
    redemptionType: {
      type: String,
      enum: ['FULL', 'PARTIAL']
    },
    redemptionAmount: Number,
    redemptionUnits: Number
  },
  switchDetails: {
    fromSchemeCode: String,
    toSchemeCode: String,
    switchType: {
      type: String,
      enum: ['AMOUNT', 'UNITS', 'ALL']
    }
  },
  errorMessage: {
    type: String
  },
  retryCount: {
    type: Number,
    default: 0
  },
  lastRetriedAt: {
    type: Date
  },
  reconciledAt: {
    type: Date
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed
  }
}, {
  timestamps: true
});

mfOrderSchema.index({ userId: 1, createdAt: -1 });
mfOrderSchema.index({ status: 1, createdAt: -1 });
mfOrderSchema.index({ bseOrderId: 1 });
mfOrderSchema.index({ idempotencyKey: 1 }, { unique: true });

mfOrderSchema.methods.markAsSubmitted = function(bseOrderId, bseResponse) {
  this.status = 'SUBMITTED';
  this.bseOrderId = bseOrderId;
  this.bseResponse = bseResponse;
  return this.save();
};

mfOrderSchema.methods.markAsCompleted = function(allottedUnits, nav, settlementDate) {
  this.status = 'COMPLETED';
  this.allottedUnits = allottedUnits;
  this.nav = nav;
  this.settlementDate = settlementDate;
  this.reconciledAt = new Date();
  return this.save();
};

mfOrderSchema.methods.markAsFailed = function(errorMessage) {
  this.status = 'FAILED';
  this.errorMessage = errorMessage;
  return this.save();
};

mfOrderSchema.methods.incrementRetry = function() {
  this.retryCount += 1;
  this.lastRetriedAt = new Date();
  return this.save();
};

mfOrderSchema.statics.findByIdempotencyKey = function(idempotencyKey) {
  return this.findOne({ idempotencyKey });
};

mfOrderSchema.statics.findPendingOrders = function() {
  return this.find({ 
    status: { $in: ['PENDING', 'SUBMITTED'] },
    createdAt: { $gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) }
  }).sort({ createdAt: 1 });
};

mfOrderSchema.statics.findOrdersForReconciliation = function() {
  return this.find({
    status: { $in: ['SUBMITTED', 'ACCEPTED'] },
    bseOrderId: { $exists: true, $ne: null },
    reconciledAt: null,
    createdAt: { $gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) }
  }).sort({ createdAt: 1 });
};

const MfOrder = mongoose.model('MfOrder', mfOrderSchema);

module.exports = MfOrder;
