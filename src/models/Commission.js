const mongoose = require('mongoose');

const commissionSchema = new mongoose.Schema({
  // Basic Information
  agentId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Agent',
    required: true
  },
  clientId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'User',
    required: true
  },
  transactionId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Transaction',
    required: true
  },

  // Transaction Details
  schemeCode: {
    type: String,
    required: true
  },
  schemeName: {
    type: String,
    required: true
  },
  transactionType: {
    type: String,
    enum: ['SIP', 'LUMPSUM', 'SWITCH_IN', 'SWITCH_OUT', 'REDEMPTION'],
    required: true
  },
  transactionAmount: {
    type: Number,
    required: true
  },
  transactionDate: {
    type: Date,
    required: true
  },

  // Commission Calculation
  commissionRate: {
    type: Number,
    required: true,
    min: 0,
    max: 100
  },
  commissionAmount: {
    type: Number,
    required: true,
    min: 0
  },
  commissionType: {
    type: String,
    enum: ['UPFRONT', 'TRAIL', 'BONUS'],
    default: 'UPFRONT'
  },

  // Payout Information
  payoutStatus: {
    type: String,
    enum: ['PENDING', 'APPROVED', 'PAID', 'CANCELLED'],
    default: 'PENDING'
  },
  payoutDate: {
    type: Date,
    default: null
  },
  payoutAmount: {
    type: Number,
    default: 0
  },
  tdsAmount: {
    type: Number,
    default: 0
  },
  netPayout: {
    type: Number,
    default: 0
  },

  // Approval & Processing
  approvedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin'
  },
  approvedAt: {
    type: Date,
    default: null
  },
  processedBy: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin'
  },
  processedAt: {
    type: Date,
    default: null
  },

  // Additional Details
  folioNumber: {
    type: String,
    default: null
  },
  bseOrderId: {
    type: String,
    default: null
  },
  remarks: {
    type: String,
    default: null
  },

  // Status & Tracking
  isActive: {
    type: Boolean,
    default: true
  },
  status: {
    type: String,
    enum: ['active', 'cancelled', 'reversed'],
    default: 'active'
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes for performance
commissionSchema.index({ agentId: 1, transactionDate: -1 });
commissionSchema.index({ clientId: 1, transactionDate: -1 });
commissionSchema.index({ payoutStatus: 1 });
commissionSchema.index({ transactionDate: -1 });
commissionSchema.index({ schemeCode: 1 });
commissionSchema.index({ bseOrderId: 1 });

// Virtuals
commissionSchema.virtual('isPaid').get(function() {
  return this.payoutStatus === 'PAID';
});

commissionSchema.virtual('isPending').get(function() {
  return this.payoutStatus === 'PENDING';
});

commissionSchema.virtual('isApproved').get(function() {
  return this.payoutStatus === 'APPROVED';
});

commissionSchema.virtual('commissionPercentage').get(function() {
  if (this.transactionAmount === 0) return 0;
  return (this.commissionAmount / this.transactionAmount) * 100;
});

// Pre-save middleware
commissionSchema.pre('save', function(next) {
  // Calculate net payout
  this.netPayout = this.payoutAmount - this.tdsAmount;
  next();
});

// Instance methods
commissionSchema.methods = {
  // Approve commission
  async approve(adminId) {
    this.payoutStatus = 'APPROVED';
    this.approvedBy = adminId;
    this.approvedAt = new Date();
    await this.save();
  },

  // Process payout
  async processPayout(adminId, payoutAmount, tdsAmount = 0) {
    this.payoutStatus = 'PAID';
    this.processedBy = adminId;
    this.processedAt = new Date();
    this.payoutAmount = payoutAmount;
    this.tdsAmount = tdsAmount;
    this.netPayout = payoutAmount - tdsAmount;
    await this.save();
  },

  // Cancel commission
  async cancel(adminId, remarks) {
    this.payoutStatus = 'CANCELLED';
    this.status = 'cancelled';
    this.remarks = remarks;
    this.processedBy = adminId;
    this.processedAt = new Date();
    await this.save();
  }
};

// Static methods
commissionSchema.statics = {
  // Get commission report
  async getCommissionReport(filters = {}) {
    const matchStage = { isActive: true };
    
    if (filters.agentId) matchStage.agentId = filters.agentId;
    if (filters.clientId) matchStage.clientId = filters.clientId;
    if (filters.payoutStatus) matchStage.payoutStatus = filters.payoutStatus;
    if (filters.commissionType) matchStage.commissionType = filters.commissionType;
    
    if (filters.startDate || filters.endDate) {
      matchStage.transactionDate = {};
      if (filters.startDate) matchStage.transactionDate.$gte = new Date(filters.startDate);
      if (filters.endDate) matchStage.transactionDate.$lte = new Date(filters.endDate);
    }

    return await this.aggregate([
      { $match: matchStage },
      {
        $lookup: {
          from: 'agents',
          localField: 'agentId',
          foreignField: '_id',
          as: 'agent'
        }
      },
      {
        $lookup: {
          from: 'users',
          localField: 'clientId',
          foreignField: '_id',
          as: 'client'
        }
      },
      {
        $unwind: '$agent'
      },
      {
        $unwind: '$client'
      },
      {
        $project: {
          agentName: '$agent.name',
          agentCode: '$agent.agentCode',
          clientName: '$client.name',
          clientCode: '$client.secretCode',
          schemeName: 1,
          transactionType: 1,
          transactionAmount: 1,
          commissionAmount: 1,
          commissionRate: 1,
          payoutStatus: 1,
          transactionDate: 1,
          payoutDate: 1
        }
      },
      {
        $sort: { transactionDate: -1 }
      }
    ]);
  },

  // Get agent earnings summary
  async getAgentEarningsSummary(agentId, period = 'monthly') {
    const now = new Date();
    let startDate;
    
    if (period === 'monthly') {
      startDate = new Date(now.getFullYear(), now.getMonth(), 1);
    } else if (period === 'quarterly') {
      const quarter = Math.floor(now.getMonth() / 3);
      startDate = new Date(now.getFullYear(), quarter * 3, 1);
    } else if (period === 'yearly') {
      startDate = new Date(now.getFullYear(), 0, 1);
    }

    return await this.aggregate([
      {
        $match: {
          agentId: mongoose.Types.ObjectId(agentId),
          isActive: true,
          transactionDate: { $gte: startDate }
        }
      },
      {
        $group: {
          _id: null,
          totalCommission: { $sum: '$commissionAmount' },
          totalTransactions: { $sum: 1 },
          totalAmount: { $sum: '$transactionAmount' },
          pendingCommission: {
            $sum: {
              $cond: [
                { $eq: ['$payoutStatus', 'PENDING'] },
                '$commissionAmount',
                0
              ]
            }
          },
          paidCommission: {
            $sum: {
              $cond: [
                { $eq: ['$payoutStatus', 'PAID'] },
                '$commissionAmount',
                0
              ]
            }
          }
        }
      }
    ]);
  },

  // Get monthly earnings trend
  async getMonthlyEarningsTrend(agentId, months = 12) {
    const startDate = new Date();
    startDate.setMonth(startDate.getMonth() - months);

    return await this.aggregate([
      {
        $match: {
          agentId: mongoose.Types.ObjectId(agentId),
          isActive: true,
          transactionDate: { $gte: startDate }
        }
      },
      {
        $group: {
          _id: {
            year: { $year: '$transactionDate' },
            month: { $month: '$transactionDate' }
          },
          totalCommission: { $sum: '$commissionAmount' },
          totalTransactions: { $sum: 1 },
          totalAmount: { $sum: '$transactionAmount' }
        }
      },
      {
        $sort: { '_id.year': 1, '_id.month': 1 }
      }
    ]);
  },

  // Get pending payouts
  async getPendingPayouts() {
    return await this.aggregate([
      {
        $match: {
          payoutStatus: 'APPROVED',
          isActive: true
        }
      },
      {
        $lookup: {
          from: 'agents',
          localField: 'agentId',
          foreignField: '_id',
          as: 'agent'
        }
      },
      {
        $unwind: '$agent'
      },
      {
        $group: {
          _id: '$agentId',
          agentName: { $first: '$agent.name' },
          agentCode: { $first: '$agent.agentCode' },
          totalPending: { $sum: '$commissionAmount' },
          transactionCount: { $sum: 1 }
        }
      },
      {
        $sort: { totalPending: -1 }
      }
    ]);
  }
};

if (process.env.NODE_ENV === 'test') {
  const mockCommissionModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockCommissionId', ...data }),
    };
  };
  mockCommissionModel.find = jest.fn().mockResolvedValue([]);
  mockCommissionModel.findOne = jest.fn().mockResolvedValue(null);
  mockCommissionModel.findById = jest.fn().mockResolvedValue(null);
  mockCommissionModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockCommissionModel.create = jest.fn().mockResolvedValue({ _id: 'mockCommissionId' });
  module.exports = mockCommissionModel;
} else {
  module.exports = mongoose.model('Commission', commissionSchema);
}