const mongoose = require('mongoose');

const agentSchema = new mongoose.Schema({
  // Basic Information
  adminId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin',
    required: true
  },
  agentCode: {
    type: String,
    unique: true,
    required: true
  },
  name: {
    type: String,
    required: true,
    trim: true
  },
  email: {
    type: String,
    required: true,
    unique: true,
    lowercase: true
  },
  phone: {
    type: String,
    required: true,
    unique: true
  },
  avatar: {
    type: String,
    default: null
  },

  // Professional Details
  designation: {
    type: String,
    default: 'Investment Advisor'
  },
  experience: {
    type: Number,
    default: 0
  },
  qualification: {
    type: String,
    default: null
  },
  arnNumber: {
    type: String,
    default: null
  },

  // Regional & Territory
  region: {
    type: String,
    enum: ['North', 'South', 'East', 'West', 'Central'],
    required: true
  },
  city: {
    type: String,
    required: true
  },
  pincode: {
    type: String,
    required: true
  },
  address: {
    type: String,
    required: true
  },

  // Performance Metrics
  totalClients: {
    type: Number,
    default: 0
  },
  activeClients: {
    type: Number,
    default: 0
  },
  totalAUM: {
    type: Number,
    default: 0
  },
  monthlyAUMGrowth: {
    type: Number,
    default: 0
  },
  avgClientXIRR: {
    type: Number,
    default: 0
  },

  // Commission & Earnings
  commissionRate: {
    type: Number,
    default: 0,
    min: 0,
    max: 100
  },
  totalEarnings: {
    type: Number,
    default: 0
  },
  monthlyEarnings: {
    type: Number,
    default: 0
  },
  pendingPayout: {
    type: Number,
    default: 0
  },

  // Targets & Goals
  monthlyTarget: {
    type: Number,
    default: 0
  },
  yearlyTarget: {
    type: Number,
    default: 0
  },
  targetAchievement: {
    type: Number,
    default: 0
  },

  // Status & Activity
  status: {
    type: String,
    enum: ['active', 'inactive', 'suspended', 'pending'],
    default: 'pending'
  },
  isVerified: {
    type: Boolean,
    default: false
  },
  joinDate: {
    type: Date,
    default: Date.now
  },
  lastActivity: {
    type: Date,
    default: Date.now
  },

  // Supervisor & Hierarchy
  supervisor: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin'
  },
  teamMembers: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Agent'
  }],

  // Performance History
  performanceHistory: [{
    month: {
      type: String,
      required: true
    },
    newClients: {
      type: Number,
      default: 0
    },
    totalAUM: {
      type: Number,
      default: 0
    },
    commission: {
      type: Number,
      default: 0
    },
    targetAchievement: {
      type: Number,
      default: 0
    }
  }],

  // Settings & Preferences
  settings: {
    notifications: {
      email: { type: Boolean, default: true },
      sms: { type: Boolean, default: true },
      push: { type: Boolean, default: true }
    },
    autoAssignClients: {
      type: Boolean,
      default: true
    },
    maxClients: {
      type: Number,
      default: 100
    }
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes - Remove duplicates of inline indexes
// agentSchema.index({ agentCode: 1 }); // Duplicate of inline unique: true
// agentSchema.index({ email: 1 }); // Duplicate of inline unique: true
// agentSchema.index({ phone: 1 }); // Duplicate of inline unique: true
agentSchema.index({ region: 1 });
agentSchema.index({ status: 1 });
agentSchema.index({ totalAUM: -1 });
agentSchema.index({ monthlyEarnings: -1 });

// Virtuals
agentSchema.virtual('fullName').get(function() {
  return this.name;
});

agentSchema.virtual('isActive').get(function() {
  return this.status === 'active';
});

agentSchema.virtual('targetCompletion').get(function() {
  if (this.monthlyTarget === 0) return 0;
  return (this.totalAUM / this.monthlyTarget) * 100;
});

// Instance methods
agentSchema.methods = {
  // Update performance metrics
  async updatePerformanceMetrics() {
    const User = mongoose.model('User');
    const UserPortfolio = mongoose.model('UserPortfolio');
    
    // Get assigned clients
    const clients = await User.find({ 
      assignedAgent: this._id,
      isActive: true 
    });
    
    // Get portfolios
    const portfolios = await UserPortfolio.find({
      userId: { $in: clients.map(c => c._id) },
      isActive: true
    });
    
    // Calculate metrics
    const totalAUM = portfolios.reduce((sum, p) => sum + p.totalCurrentValue, 0);
    const avgXIRR = portfolios.length > 0 
      ? portfolios.reduce((sum, p) => sum + (p.xirr1Y || 0), 0) / portfolios.length 
      : 0;
    
    // Update agent
    this.totalClients = clients.length;
    this.activeClients = clients.filter(c => c.lastLogin > new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)).length;
    this.totalAUM = totalAUM;
    this.avgClientXIRR = avgXIRR;
    
    await this.save();
  },

  // Add performance record
  async addPerformanceRecord(month, data) {
    const existing = this.performanceHistory.find(p => p.month === month);
    if (existing) {
      Object.assign(existing, data);
    } else {
      this.performanceHistory.push({
        month,
        ...data
      });
    }
    await this.save();
  },

  // Update activity
  async updateActivity() {
    this.lastActivity = new Date();
    await this.save();
  }
};

// Static methods
agentSchema.statics = {
  // Get leaderboard
  async getLeaderboard(limit = 10, period = 'monthly') {
    const matchStage = { status: 'active' };
    
    let sortField = 'monthlyEarnings';
    if (period === 'yearly') {
      sortField = 'totalEarnings';
    } else if (period === 'aum') {
      sortField = 'totalAUM';
    }
    
    return await this.find(matchStage)
      .sort({ [sortField]: -1 })
      .limit(limit)
      .select('name agentCode totalAUM monthlyEarnings totalEarnings avatar region');
  },

  // Get regional stats
  async getRegionalStats() {
    return await this.aggregate([
      {
        $match: { status: 'active' }
      },
      {
        $group: {
          _id: '$region',
          totalAgents: { $sum: 1 },
          totalAUM: { $sum: '$totalAUM' },
          totalEarnings: { $sum: '$monthlyEarnings' },
          avgXIRR: { $avg: '$avgClientXIRR' }
        }
      },
      {
        $sort: { totalAUM: -1 }
      }
    ]);
  },

  // Get performance trends
  async getPerformanceTrends(months = 6) {
    const monthsAgo = new Date();
    monthsAgo.setMonth(monthsAgo.getMonth() - months);
    
    return await this.aggregate([
      {
        $match: {
          createdAt: { $gte: monthsAgo }
        }
      },
      {
        $unwind: '$performanceHistory'
      },
      {
        $match: {
          'performanceHistory.month': {
            $gte: monthsAgo.toISOString().slice(0, 7)
          }
        }
      },
      {
        $group: {
          _id: '$performanceHistory.month',
          totalAUM: { $sum: '$performanceHistory.totalAUM' },
          totalCommission: { $sum: '$performanceHistory.commission' },
          newClients: { $sum: '$performanceHistory.newClients' }
        }
      },
      {
        $sort: { _id: 1 }
      }
    ]);
  }
};

if (process.env.NODE_ENV === 'test') {
  const mockAgentModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockAgentId', ...data }),
    };
  };
  mockAgentModel.find = jest.fn().mockResolvedValue([]);
  mockAgentModel.findOne = jest.fn().mockResolvedValue(null);
  mockAgentModel.findById = jest.fn().mockResolvedValue(null);
  mockAgentModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockAgentModel.create = jest.fn().mockResolvedValue({ _id: 'mockAgentId' });
  module.exports = mockAgentModel;
} else {
  module.exports = mongoose.model('Agent', agentSchema);
}