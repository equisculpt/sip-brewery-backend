const mongoose = require('mongoose');

const userSchema = new mongoose.Schema({
  supabaseId: {
    type: String,
    required: true,
    unique: true
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true,
    trim: true
  },
  phone: {
    type: String,
    required: true,
    unique: true,
    trim: true
  },
  secretCode: {
    type: String,
    required: true,
    unique: true,
    default: function() {
      // Generate secret code: SBX + 2 random chars + - + 3 random chars
      const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
      const part1 = Array.from({length: 2}, () => chars[Math.floor(Math.random() * chars.length)]).join('');
      const part2 = Array.from({length: 3}, () => chars[Math.floor(Math.random() * chars.length)]).join('');
      return `SBX${part1}-${part2}`;
    }
  },
  name: {
    type: String,
    required: true,
    trim: true
  },
  role: {
    type: String,
    default: 'user'
  },
  kycStatus: {
    type: String,
    enum: ['PENDING', 'VERIFIED', 'REJECTED'],
    default: 'PENDING'
  },
  // Referral fields for rewards system
  referralCode: {
    type: String,
    unique: true,
    default: function() {
      // Generate referral code: REF + 6 random alphanumeric chars
      const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
      return 'REF' + Array.from({length: 6}, () => chars[Math.floor(Math.random() * chars.length)]).join('');
    }
  },
  referredBy: {
    type: String,
    ref: 'User'
  },
  referralCount: {
    type: Number,
    default: 0
  },
  totalReferralBonus: {
    type: Number,
    default: 0
  },
  kycDetails: {
    panNumber: String,
    aadharNumber: String,
    dateOfBirth: Date,
    address: {
      street: String,
      city: String,
      state: String,
      pincode: String
    },
    bankDetails: {
      accountNumber: String,
      ifscCode: String,
      bankName: String
    }
  },
  preferences: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },
  isActive: {
    type: Boolean,
    default: true
  },
  lastLoginAt: {
    type: Date,
    default: Date.now
  },
  otp: { type: String },
  otpExpiry: { type: Date },
  password: { type: String }, // hashed if provided
  authMode: { type: String, enum: ['1FA', '2FA'], default: '1FA' },
  isEmailVerified: { type: Boolean, default: false },
  // Email verification and password reset fields for Brevo integration
  emailToken: { type: String },
  emailTokenExpiry: { type: Date },
  resetPasswordToken: { type: String },
  resetPasswordExpiry: { type: Date },
  isPhoneVerified: { type: Boolean, default: false },
  createdAt: { type: Date, default: Date.now },
}, {
  timestamps: true
});

// Pre-save middleware to ensure secretCode and referralCode are generated
userSchema.pre('save', function(next) {
  if (!this.secretCode) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    const part1 = Array.from({length: 2}, () => chars[Math.floor(Math.random() * chars.length)]).join('');
    const part2 = Array.from({length: 3}, () => chars[Math.floor(Math.random() * chars.length)]).join('');
    this.secretCode = `SBX${part1}-${part2}`;
  }
  
  if (!this.referralCode) {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    this.referralCode = 'REF' + Array.from({length: 6}, () => chars[Math.floor(Math.random() * chars.length)]).join('');
  }
  
  next();
});

// Only declare indexes that don't duplicate inline index: true
userSchema.index({ referredBy: 1 });
userSchema.index({ phone: 1 }, { unique: true });

if (process.env.NODE_ENV === 'test') {
  // Return a jest mock model for User
  const mockUserModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockUserId', ...data }),
    };
  };
  mockUserModel.find = jest.fn().mockResolvedValue([]);
  mockUserModel.findOne = jest.fn().mockResolvedValue(null);
  mockUserModel.findById = jest.fn().mockResolvedValue(null);
  mockUserModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockUserModel.create = jest.fn().mockResolvedValue({ _id: 'mockUserId' });
  module.exports = mockUserModel;
} else {
  module.exports = mongoose.model('User', userSchema);
}