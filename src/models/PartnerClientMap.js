const mongoose = require('mongoose');

const partnerClientMapSchema = new mongoose.Schema({
  partnerId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Partner',
    required: true
  },
  investorId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  relationshipType: {
    type: String,
    enum: ['REFERRAL', 'DASHBOARD_ONBOARDING'],
    required: true
  },
  status: {
    type: String,
    enum: ['ACTIVE', 'SUSPENDED', 'TERMINATED'],
    default: 'ACTIVE'
  },
  consentId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Consent',
    default: null
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  }
}, {
  timestamps: true
});

partnerClientMapSchema.index({ partnerId: 1, investorId: 1 }, { unique: true });
partnerClientMapSchema.index({ partnerId: 1, status: 1 });
partnerClientMapSchema.index({ investorId: 1, status: 1 });

module.exports = mongoose.model('PartnerClientMap', partnerClientMapSchema);
