const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');

const adminSchema = new mongoose.Schema({
  // Basic Information
  name: {
    type: String,
    required: true,
    trim: true
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
    unique: true
  },
  avatar: {
    type: String,
    default: null
  },

  // Authentication & Security
  password: {
    type: String,
    required: true,
    minlength: 8
  },
  supabaseId: {
    type: String,
    unique: true,
    sparse: true
  },
  twoFactorEnabled: {
    type: Boolean,
    default: false
  },
  twoFactorSecret: {
    type: String,
    default: null
  },
  lastLogin: {
    type: Date,
    default: null
  },
  loginAttempts: {
    type: Number,
    default: 0
  },
  lockUntil: {
    type: Date,
    default: null
  },

  // Role & Permissions
  role: {
    type: String,
    enum: ['SUPER_ADMIN', 'ADMIN', 'AGENT', 'VIEW_ONLY'],
    default: 'VIEW_ONLY',
    required: true
  },
  permissions: [{
    module: {
      type: String,
      enum: [
        'agents', 'clients', 'commission', 'analytics', 'kyc', 
        'transactions', 'rewards', 'pdf', 'settings', 'tax', 
        'leaderboard', 'notifications', 'audit'
      ]
    },
    actions: [{
      type: String,
      enum: ['create', 'read', 'update', 'delete', 'export', 'approve']
    }]
  }],

  // Regional Access (for ADMIN role)
  regions: [{
    type: String,
    enum: ['North', 'South', 'East', 'West', 'Central']
  }],

  // Agent-specific fields (for AGENT role)
  agentCode: {
    type: String,
    unique: true,
    sparse: true
  },
  assignedClients: [{
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User'
  }],
  supervisor: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin'
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

  // Status & Activity
  isActive: {
    type: Boolean,
    default: true
  },
  isVerified: {
    type: Boolean,
    default: false
  },
  status: {
    type: String,
    enum: ['active', 'inactive', 'suspended', 'pending'],
    default: 'pending'
  },

  // Preferences & Settings
  preferences: {
    dashboardLayout: {
      type: String,
      default: 'default'
    },
    notifications: {
      email: { type: Boolean, default: true },
      sms: { type: Boolean, default: false },
      push: { type: Boolean, default: true }
    },
    theme: {
      type: String,
      enum: ['light', 'dark', 'auto'],
      default: 'light'
    },
    language: {
      type: String,
      default: 'en'
    }
  },

  // Audit & Tracking
  createdBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin'
  },
  lastActivity: {
    type: Date,
    default: Date.now
  },
  ipWhitelist: [{
    ip: String,
    description: String,
    addedAt: { type: Date, default: Date.now }
  }]
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for performance - Remove duplicates of inline indexes
// adminSchema.index({ email: 1 }); // Duplicate of inline unique: true
// adminSchema.index({ phone: 1 }); // Duplicate of inline unique: true
adminSchema.index({ role: 1 });
// adminSchema.index({ agentCode: 1 }); // Duplicate of inline unique: true
adminSchema.index({ isActive: 1 });
adminSchema.index({ status: 1 });
adminSchema.index({ createdAt: -1 });

// Virtual for full name
adminSchema.virtual('fullName').get(function() {
  return this.name;
});

// Virtual for role display name
adminSchema.virtual('roleDisplay').get(function() {
  const roleNames = {
    'SUPER_ADMIN': 'Super Administrator',
    'ADMIN': 'Administrator',
    'AGENT': 'Agent',
    'VIEW_ONLY': 'View Only'
  };
  return roleNames[this.role] || this.role;
});

// Pre-save middleware
adminSchema.pre('save', async function(next) {
  if (this.isModified('password')) {
    this.password = await bcrypt.hash(this.password, 12);
  }
  
  // Generate agent code if not exists
  if (this.role === 'AGENT' && !this.agentCode) {
    this.agentCode = await this.generateAgentCode();
  }
  
  next();
});

// Instance methods
adminSchema.methods = {
  // Password verification
  async verifyPassword(candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
  },

  // Check if account is locked
  isLocked() {
    return !!(this.lockUntil && this.lockUntil > Date.now());
  },

  // Increment login attempts
  async incLoginAttempts() {
    if (this.lockUntil && this.lockUntil < Date.now()) {
      await this.updateOne({
        $unset: { lockUntil: 1 },
        $set: { loginAttempts: 1 }
      });
      return;
    }
    
    const updates = { $inc: { loginAttempts: 1 } };
    if (this.loginAttempts + 1 >= 5 && !this.isLocked()) {
      updates.$set = { lockUntil: Date.now() + 2 * 60 * 60 * 1000 }; // 2 hours
    }
    await this.updateOne(updates);
  },

  // Reset login attempts
  async resetLoginAttempts() {
    await this.updateOne({
      $unset: { loginAttempts: 1, lockUntil: 1 },
      $set: { lastLogin: new Date() }
    });
  },

  // Check permission
  hasPermission(module, action) {
    if (this.role === 'SUPER_ADMIN') return true;
    
    const permission = this.permissions.find(p => p.module === module);
    return permission && permission.actions.includes(action);
  },

  // Generate agent code
  async generateAgentCode() {
    const prefix = 'AG';
    const count = await this.constructor.countDocuments({ role: 'AGENT' });
    return `${prefix}${String(count + 1).padStart(6, '0')}`;
  },

  // Get accessible clients (for agents)
  async getAccessibleClients() {
    if (this.role === 'SUPER_ADMIN' || this.role === 'ADMIN') {
      return await mongoose.model('User').find({ isActive: true });
    }
    
    if (this.role === 'AGENT') {
      return await mongoose.model('User').find({
        _id: { $in: this.assignedClients },
        isActive: true
      });
    }
    
    return [];
  },

  // Update last activity
  async updateActivity() {
    this.lastActivity = new Date();
    await this.save();
  }
};

// Static methods
adminSchema.statics = {
  // Find by email
  async findByEmail(email) {
    return await this.findOne({ email: email.toLowerCase() });
  },

  // Find by phone
  async findByPhone(phone) {
    return await this.findOne({ phone });
  },

  // Find by agent code
  async findByAgentCode(agentCode) {
    return await this.findOne({ agentCode });
  },

  // Get agents with stats
  async getAgentsWithStats() {
    return await this.aggregate([
      {
        $match: { role: 'AGENT', isActive: true }
      },
      {
        $lookup: {
          from: 'users',
          localField: 'assignedClients',
          foreignField: '_id',
          as: 'clients'
        }
      },
      {
        $lookup: {
          from: 'userportfolios',
          localField: 'assignedClients',
          foreignField: 'userId',
          as: 'portfolios'
        }
      },
      {
        $addFields: {
          totalClients: { $size: '$clients' },
          totalAUM: {
            $sum: '$portfolios.totalCurrentValue'
          },
          avgXIRR: {
            $avg: '$portfolios.xirr1Y'
          }
        }
      },
      {
        $project: {
          name: 1,
          email: 1,
          phone: 1,
          agentCode: 1,
          totalClients: 1,
          totalAUM: 1,
          avgXIRR: 1,
          totalEarnings: 1,
          monthlyEarnings: 1,
          status: 1,
          createdAt: 1
        }
      }
    ]);
  },

  // Get admin dashboard stats
  async getDashboardStats() {
    const stats = await this.aggregate([
      {
        $facet: {
          totalAdmins: [
            { $match: { isActive: true } },
            { $count: 'count' }
          ],
          totalAgents: [
            { $match: { role: 'AGENT', isActive: true } },
            { $count: 'count' }
          ],
          activeAgents: [
            { $match: { role: 'AGENT', status: 'active' } },
            { $count: 'count' }
          ],
          totalEarnings: [
            { $match: { role: 'AGENT' } },
            { $group: { _id: null, total: { $sum: '$totalEarnings' } } }
          ]
        }
      }
    ]);

    return {
      totalAdmins: stats[0].totalAdmins[0]?.count || 0,
      totalAgents: stats[0].totalAgents[0]?.count || 0,
      activeAgents: stats[0].activeAgents[0]?.count || 0,
      totalEarnings: stats[0].totalEarnings[0]?.total || 0
    };
  }
};

if (process.env.NODE_ENV === 'test') {
  const mockAdminModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockAdminId', ...data }),
    };
  };
  mockAdminModel.find = jest.fn().mockResolvedValue([]);
  mockAdminModel.findOne = jest.fn().mockResolvedValue(null);
  mockAdminModel.findById = jest.fn().mockResolvedValue(null);
  mockAdminModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockAdminModel.create = jest.fn().mockResolvedValue({ _id: 'mockAdminId' });
  module.exports = mockAdminModel;
} else {
  module.exports = mongoose.model('Admin', adminSchema);
}