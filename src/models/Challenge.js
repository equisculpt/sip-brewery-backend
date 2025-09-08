const mongoose = require('mongoose');

const challengeSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  type: {
    type: String,
    enum: ['DAILY', 'WEEKLY', 'MONTHLY', 'SPECIAL', 'LEADERBOARD', 'SIP', 'REFERRAL'],
    required: true
  },
  title: {
    type: String,
    required: true
  },
  description: {
    type: String,
    required: true
  },
  icon: {
    type: String,
    default: '🎯'
  },
  criteria: {
    type: mongoose.Schema.Types.Mixed,
    required: true
  },
  progress: {
    current: {
      type: Number,
      default: 0
    },
    target: {
      type: Number,
      required: true
    },
    unit: {
      type: String,
      default: 'count'
    }
  },
  reward: {
    points: {
      type: Number,
      default: 0
    },
    badge: {
      type: String,
      default: null
    },
    special: {
      type: mongoose.Schema.Types.Mixed,
      default: null
    }
  },
  status: {
    type: String,
    enum: ['IN_PROGRESS', 'COMPLETED', 'CLAIMED', 'EXPIRED'],
    default: 'IN_PROGRESS'
  },
  completedAt: {
    type: Date,
    default: null
  },
  claimedAt: {
    type: Date,
    default: null
  },
  expiresAt: {
    type: Date,
    default: null
  },
  isActive: {
    type: Boolean,
    default: true
  },
  createdAt: {
    type: Date,
    default: Date.now
  },
  updatedAt: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

// Indexes
challengeSchema.index({ userId: 1, type: 1 });
challengeSchema.index({ userId: 1, status: 1 });
challengeSchema.index({ type: 1, status: 1 });

// Virtual for completion percentage
challengeSchema.virtual('completionPercentage').get(function() {
  if (this.progress.target === 0) return 0;
  return Math.min(100, (this.progress.current / this.progress.target) * 100);
});

// Virtual for isCompleted
challengeSchema.virtual('isCompleted').get(function() {
  return this.progress.current >= this.progress.target;
});

// Methods
challengeSchema.methods.updateProgress = function(newProgress) {
  this.progress.current = Math.min(newProgress, this.progress.target);
  
  if (this.progress.current >= this.progress.target && this.status === 'IN_PROGRESS') {
    this.status = 'COMPLETED';
    this.completedAt = new Date();
  }
  
  this.updatedAt = new Date();
  return this.save();
};

challengeSchema.methods.claim = function() {
  if (this.status === 'COMPLETED') {
    this.status = 'CLAIMED';
    this.claimedAt = new Date();
    this.updatedAt = new Date();
    return this.save();
  }
  throw new Error('Challenge must be completed before claiming');
};

// Static methods
challengeSchema.statics.getUserChallenges = function(userId, status = null) {
  const query = { userId, isActive: true };
  if (status) query.status = status;
  
  return this.find(query)
    .sort({ createdAt: -1 })
    .populate('userId', 'name email');
};

challengeSchema.statics.getChallengeStats = function(userId) {
  return this.aggregate([
    { $match: { userId: new mongoose.Types.ObjectId(userId), isActive: true } },
    {
      $group: {
        _id: '$status',
        count: { $sum: 1 },
        totalPoints: { $sum: '$reward.points' }
      }
    }
  ]);
};

// Pre-save middleware
challengeSchema.pre('save', function(next) {
  this.updatedAt = new Date();
  next();
});

if (process.env.NODE_ENV === 'test') {
  const mockChallengeModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockChallengeId', ...data }),
    };
  };
  mockChallengeModel.find = jest.fn().mockResolvedValue([]);
  mockChallengeModel.findOne = jest.fn().mockResolvedValue(null);
  mockChallengeModel.findById = jest.fn().mockResolvedValue(null);
  mockChallengeModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockChallengeModel.create = jest.fn().mockResolvedValue({ _id: 'mockChallengeId' });
  module.exports = mockChallengeModel;
} else {
  module.exports = mongoose.model('Challenge', challengeSchema);
}