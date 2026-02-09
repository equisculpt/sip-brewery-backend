const mongoose = require('mongoose');

const consentSchema = new mongoose.Schema({
  investorId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  partnerId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Partner',
    default: null
  },
  scope: {
    type: String,
    required: true
  },
  channel: {
    type: String,
    enum: ['OTP', 'ESIGN', 'VOICE', 'CHECKBOX', 'DOCUMENT'],
    required: true
  },
  artifactUri: {
    type: String,
    required: true
  },
  status: {
    type: String,
    enum: ['ACTIVE', 'REVOKED', 'EXPIRED'],
    default: 'ACTIVE'
  },
  capturedAt: {
    type: Date,
    default: Date.now
  },
  expiresAt: {
    type: Date,
    default: null
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  }
}, {
  timestamps: true
});

consentSchema.index({ investorId: 1, scope: 1, status: 1 });
consentSchema.index({ partnerId: 1, scope: 1, status: 1 });

module.exports = mongoose.model('Consent', consentSchema);
