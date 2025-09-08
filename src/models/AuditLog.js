const mongoose = require('mongoose');

const auditLogSchema = new mongoose.Schema({
  // User Information
  userId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin',
    required: true
  },
  userEmail: {
    type: String,
    required: true
  },
  userRole: {
    type: String,
    required: true
  },
  userAgentCode: {
    type: String,
    default: null
  },

  // Action Details
  action: {
    type: String,
    required: true
  },
  module: {
    type: String,
    enum: [
      'auth', 'agents', 'clients', 'commission', 'analytics', 'kyc',
      'transactions', 'rewards', 'pdf', 'settings', 'tax', 'leaderboard',
      'notifications', 'audit', 'system'
    ],
    required: true
  },
  resourceType: {
    type: String,
    default: null
  },
  resourceId: {
    type: mongoose.Schema.Types.ObjectId,
    default: null
  },

  // Request Details
  method: {
    type: String,
    enum: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    default: null
  },
  endpoint: {
    type: String,
    default: null
  },
  ipAddress: {
    type: String,
    required: true
  },
  userAgent: {
    type: String,
    default: null
  },

  // Data Changes
  oldData: {
    type: mongoose.Schema.Types.Mixed,
    default: null
  },
  newData: {
    type: mongoose.Schema.Types.Mixed,
    default: null
  },
  changes: [{
    field: String,
    oldValue: mongoose.Schema.Types.Mixed,
    newValue: mongoose.Schema.Types.Mixed
  }],

  // Status & Result
  status: {
    type: String,
    enum: ['success', 'failure', 'error'],
    default: 'success'
  },
  errorMessage: {
    type: String,
    default: null
  },
  responseTime: {
    type: Number,
    default: null
  },

  // Security & Compliance
  severity: {
    type: String,
    enum: ['low', 'medium', 'high', 'critical'],
    default: 'low'
  },
  isSuspicious: {
    type: Boolean,
    default: false
  },
  flaggedBy: {
    type: String,
    enum: ['manual', 'automated', 'ai'],
    default: null
  },

  // Additional Context
  sessionId: {
    type: String,
    default: null
  },
  requestId: {
    type: String,
    default: null
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },

  // Timestamps
  timestamp: {
    type: Date,
    default: Date.now
  }
}, {
  timestamps: true
});

// Indexes for performance and queries
auditLogSchema.index({ userId: 1, timestamp: -1 });
auditLogSchema.index({ module: 1, timestamp: -1 });
auditLogSchema.index({ action: 1, timestamp: -1 });
auditLogSchema.index({ status: 1, timestamp: -1 });
auditLogSchema.index({ severity: 1, timestamp: -1 });
auditLogSchema.index({ ipAddress: 1, timestamp: -1 });
auditLogSchema.index({ isSuspicious: 1, timestamp: -1 });
auditLogSchema.index({ timestamp: -1 });

// Virtuals
auditLogSchema.virtual('isSuccess').get(function() {
  return this.status === 'success';
});

auditLogSchema.virtual('isFailure').get(function() {
  return this.status === 'failure' || this.status === 'error';
});

auditLogSchema.virtual('isHighSeverity').get(function() {
  return this.severity === 'high' || this.severity === 'critical';
});

// Static methods
auditLogSchema.statics = {
  // Log an action
  async logAction(data) {
    try {
      const log = new this(data);
      await log.save();
      return log;
    } catch (error) {
      console.error('Audit log save error:', error);
      // Don't throw error to avoid breaking main functionality
    }
  },

  // Get audit trail for a resource
  async getAuditTrail(resourceType, resourceId, limit = 50) {
    return await this.find({
      resourceType,
      resourceId
    })
    .sort({ timestamp: -1 })
    .limit(limit)
    .populate('userId', 'name email role');
  },

  // Get user activity
  async getUserActivity(userId, days = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    return await this.find({
      userId,
      timestamp: { $gte: startDate }
    })
    .sort({ timestamp: -1 })
    .populate('userId', 'name email role');
  },

  // Get suspicious activities
  async getSuspiciousActivities(limit = 100) {
    return await this.find({
      $or: [
        { isSuspicious: true },
        { severity: { $in: ['high', 'critical'] } },
        { status: 'error' }
      ]
    })
    .sort({ timestamp: -1 })
    .limit(limit)
    .populate('userId', 'name email role');
  },

  // Get security events
  async getSecurityEvents(days = 7) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    return await this.aggregate([
      {
        $match: {
          timestamp: { $gte: startDate },
          $or: [
            { module: 'auth' },
            { severity: { $in: ['high', 'critical'] } },
            { isSuspicious: true }
          ]
        }
      },
      {
        $group: {
          _id: {
            date: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } },
            severity: '$severity'
          },
          count: { $sum: 1 },
          events: { $push: '$$ROOT' }
        }
      },
      {
        $sort: { '_id.date': -1, '_id.severity': 1 }
      }
    ]);
  },

  // Get activity summary
  async getActivitySummary(days = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    return await this.aggregate([
      {
        $match: {
          timestamp: { $gte: startDate }
        }
      },
      {
        $group: {
          _id: {
            date: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } },
            module: '$module'
          },
          count: { $sum: 1 },
          successCount: {
            $sum: { $cond: [{ $eq: ['$status', 'success'] }, 1, 0] }
          },
          failureCount: {
            $sum: { $cond: [{ $ne: ['$status', 'success'] }, 1, 0] }
          }
        }
      },
      {
        $sort: { '_id.date': -1 }
      }
    ]);
  },

  // Get IP activity
  async getIPActivity(ipAddress, hours = 24) {
    const startDate = new Date();
    startDate.setHours(startDate.getHours() - hours);

    return await this.find({
      ipAddress,
      timestamp: { $gte: startDate }
    })
    .sort({ timestamp: -1 })
    .populate('userId', 'name email role');
  },

  // Clean old logs (retention policy)
  async cleanOldLogs(daysToKeep = 365) {
    const cutoffDate = new Date();
    cutoffDate.setDate(cutoffDate.getDate() - daysToKeep);

    const result = await this.deleteMany({
      timestamp: { $lt: cutoffDate },
      severity: { $ne: 'critical' } // Keep critical logs longer
    });

    return result.deletedCount;
  }
};

// Instance methods
auditLogSchema.methods = {
  // Mark as suspicious
  async markSuspicious(reason) {
    this.isSuspicious = true;
    this.metadata.suspiciousReason = reason;
    this.metadata.flaggedAt = new Date();
    await this.save();
  },

  // Add metadata
  async addMetadata(key, value) {
    this.metadata[key] = value;
    await this.save();
  }
};

if (process.env.NODE_ENV === 'test') {
  const mockAuditLogModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockAuditLogId', ...data }),
    };
  };
  mockAuditLogModel.find = jest.fn().mockResolvedValue([]);
  mockAuditLogModel.findOne = jest.fn().mockResolvedValue(null);
  mockAuditLogModel.findById = jest.fn().mockResolvedValue(null);
  mockAuditLogModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockAuditLogModel.create = jest.fn().mockResolvedValue({ _id: 'mockAuditLogId' });
  module.exports = mockAuditLogModel;
} else {
  module.exports = mongoose.model('AuditLog', auditLogSchema);
}