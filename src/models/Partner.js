const mongoose = require('mongoose');

const partnerSchema = new mongoose.Schema({
  name: {
    type: String,
    required: true
  },
  code: {
    type: String,
    required: true,
    unique: true
  },
  type: {
    type: String,
    enum: ['IFA', 'SUB_DISTRIBUTOR', 'BROKER', 'INSTITUTIONAL'],
    required: true
  },
  status: {
    type: String,
    enum: ['PENDING', 'ACTIVE', 'SUSPENDED', 'REVOKED'],
    default: 'PENDING'
  },
  contact: {
    name: String,
    email: String,
    phone: String
  },
  onboarding: {
    referralLink: String,
    onboardingCompleted: {
      type: Boolean,
      default: false
    },
    kycStatus: {
      type: String,
      enum: ['PENDING', 'VERIFIED', 'REJECTED'],
      default: 'PENDING'
    }
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  }
}, {
  timestamps: true
});

partnerSchema.index({ code: 1 }, { unique: true });
partnerSchema.index({ status: 1 });
partnerSchema.index({ type: 1 });

module.exports = mongoose.model('Partner', partnerSchema);
