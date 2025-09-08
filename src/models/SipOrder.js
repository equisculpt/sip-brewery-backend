const mongoose = require('mongoose');

const sipOrderSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
    ref: 'User',
    index: true
  },
  phoneNumber: {
    type: String,
    required: true,
    index: true
  },
  // Order details
  schemeName: {
    type: String,
    required: true,
    trim: true
  },
  schemeCode: {
    type: String,
    trim: true
  },
  fundHouse: {
    type: String,
    required: true,
    trim: true
  },
  fundType: {
    type: String,
    enum: ['EQUITY', 'DEBT', 'HYBRID', 'LIQUID', 'GOLD', 'INTERNATIONAL'],
    required: true
  },
  // Investment details
  amount: {
    type: Number,
    required: true,
    min: 100
  },
  frequency: {
    type: String,
    enum: ['MONTHLY', 'WEEKLY', 'QUARTERLY'],
    default: 'MONTHLY'
  },
  // Order status
  status: {
    type: String,
    enum: ['PENDING', 'CONFIRMED', 'ACTIVE', 'PAUSED', 'CANCELLED', 'FAILED'],
    default: 'PENDING'
  },
  // Dates
  startDate: {
    type: Date,
    required: true
  },
  nextInstallmentDate: {
    type: Date
  },
  endDate: {
    type: Date
  },
  // WhatsApp context
  whatsAppOrderId: {
    type: String,
    unique: true,
    index: true
  },
  orderSource: {
    type: String,
    enum: ['WHATSAPP', 'WEB', 'APP'],
    default: 'WHATSAPP'
  },
  // Confirmation details
  confirmationMessage: String,
  userConfirmation: {
    type: Boolean,
    default: false
  },
  confirmationTime: Date,
  // SEBI compliance
  folioNumber: String,
  bseOrderId: String,
  bseConfirmationId: String,
  // Tracking
  installmentsCompleted: {
    type: Number,
    default: 0
  },
  totalInvested: {
    type: Number,
    default: 0
  },
  currentValue: {
    type: Number,
    default: 0
  },
  // Metadata
  notes: String,
  tags: [String],
  isActive: {
    type: Boolean,
    default: true
  }
}, {
  timestamps: true
});

// Indexes for performance optimization
// sipOrderSchema.index({ sipOrderId: 1 }); // Duplicate of inline index: true
// sipOrderSchema.index({ phoneNumber: 1, status: 1 }); // phoneNumber has inline index: true
// sipOrderSchema.index({ whatsAppOrderId: 1 }); // whatsAppOrderId has inline index: true

// Keep other indexes as they don't duplicate inline declarations
sipOrderSchema.index({ userId: 1, status: 1 });
sipOrderSchema.index({ schemeName: 1 });
sipOrderSchema.index({ status: 1, nextInstallmentDate: 1 });

// Pre-save middleware to generate WhatsApp order ID
sipOrderSchema.pre('save', function(next) {
  if (!this.whatsAppOrderId) {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substr(2, 5);
    this.whatsAppOrderId = `SIP${timestamp}${random}`.toUpperCase();
  }
  next();
});

if (process.env.NODE_ENV === 'test') {
  const mockSipOrderModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockSipOrderId', ...data }),
    };
  };
  mockSipOrderModel.find = jest.fn().mockResolvedValue([]);
  mockSipOrderModel.findOne = jest.fn().mockResolvedValue(null);
  mockSipOrderModel.findById = jest.fn().mockResolvedValue(null);
  mockSipOrderModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockSipOrderModel.create = jest.fn().mockResolvedValue({ _id: 'mockSipOrderId' });
  module.exports = mockSipOrderModel;
} else {
  module.exports = mongoose.model('SipOrder', sipOrderSchema);
}