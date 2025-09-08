const mongoose = require('mongoose');

const achievementSchema = new mongoose.Schema({
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  type: {
    type: String,
    enum: ['INVESTMENT', 'SIP', 'LEADERBOARD', 'REFERRAL', 'ENGAGEMENT', 'SPECIAL'],
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
    default: '🏆'
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
    enum: ['IN_PROGRESS', 'COMPLETED', 'CLAIMED'],
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
achievementSchema.index({ userId: 1, type: 1 });
achievementSchema.index({ userId: 1, status: 1 });
achievementSchema.index({ type: 1, status: 1 });

// Virtual for completion percentage
achievementSchema.virtual('completionPercentage').get(function() {
  if (this.progress.target === 0) return 0;
  return Math.min(100, (this.progress.current / this.progress.target) * 100);
});

// Virtual for isCompleted
achievementSchema.virtual('isCompleted').get(function() {
  return this.progress.current >= this.progress.target;
});

// Methods
achievementSchema.methods.updateProgress = function(newProgress) {
  this.progress.current = Math.min(newProgress, this.progress.target);
  
  if (this.progress.current >= this.progress.target && this.status === 'IN_PROGRESS') {
    this.status = 'COMPLETED';
    this.completedAt = new Date();
  }
  
  this.updatedAt = new Date();
  return this.save();
};

achievementSchema.methods.claim = function() {
  if (this.status === 'COMPLETED') {
    this.status = 'CLAIMED';
    this.claimedAt = new Date();
    this.updatedAt = new Date();
    return this.save();
  }
  throw new Error('Achievement must be completed before claiming');
};

// Static methods
achievementSchema.statics.getUserAchievements = function(userId, status = null) {
  const query = { userId, isActive: true };
  if (status) query.status = status;
  
  return this.find(query)
    .sort({ createdAt: -1 })
    .populate('userId', 'name email');
};

achievementSchema.statics.getAchievementStats = function(userId) {
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
achievementSchema.pre('save', function(next) {
  this.updatedAt = new Date();
  next();
});

if (process.env.NODE_ENV === 'test') {
  const mockAchievementModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockAchievementId', ...data }),
    };
  };
  mockAchievementModel.find = jest.fn().mockResolvedValue([]);
  mockAchievementModel.findOne = jest.fn().mockResolvedValue(null);
  mockAchievementModel.findById = jest.fn().mockResolvedValue(null);
  mockAchievementModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockAchievementModel.create = jest.fn().mockResolvedValue({ _id: 'mockAchievementId' });
  module.exports = mockAchievementModel;
} else {
  module.exports = mongoose.model('Achievement', achievementSchema);
}