const mongoose = require('mongoose');

const notificationSchema = new mongoose.Schema({
  // Recipient Information
  recipientId: {
    type: mongoose.Schema.Types.ObjectId,
    ref: 'Admin',
    required: true
  },
  recipientType: {
    type: String,
    enum: ['admin', 'agent', 'client'],
    required: true
  },

  // Notification Details
  type: {
    type: String,
    enum: [
      'system', 'commission', 'kyc', 'transaction', 'reward', 'alert',
      'reminder', 'approval', 'security', 'performance', 'compliance'
    ],
    required: true
  },
  title: {
    type: String,
    required: true,
    maxlength: 200
  },
  message: {
    type: String,
    required: true,
    maxlength: 1000
  },
  summary: {
    type: String,
    maxlength: 500
  },

  // Priority & Urgency
  priority: {
    type: String,
    enum: ['low', 'medium', 'high', 'urgent'],
    default: 'medium'
  },
  isUrgent: {
    type: Boolean,
    default: false
  },

  // Action & Navigation
  actionUrl: {
    type: String,
    default: null
  },
  actionText: {
    type: String,
    default: null
  },
  requiresAction: {
    type: Boolean,
    default: false
  },

  // Related Data
  relatedModule: {
    type: String,
    enum: [
      'agents', 'clients', 'commission', 'analytics', 'kyc',
      'transactions', 'rewards', 'pdf', 'settings', 'tax', 'leaderboard'
    ],
    default: null
  },
  relatedId: {
    type: mongoose.Schema.Types.ObjectId,
    default: null
  },
  metadata: {
    type: mongoose.Schema.Types.Mixed,
    default: {}
  },

  // Delivery Channels
  channels: {
    inApp: {
      type: Boolean,
      default: true
    },
    email: {
      type: Boolean,
      default: false
    },
    sms: {
      type: Boolean,
      default: false
    },
    push: {
      type: Boolean,
      default: false
    },
    whatsapp: {
      type: Boolean,
      default: false
    }
  },

  // Delivery Status
  deliveryStatus: {
    inApp: {
      sent: { type: Boolean, default: false },
      read: { type: Boolean, default: false },
      readAt: { type: Date, default: null }
    },
    email: {
      sent: { type: Boolean, default: false },
      delivered: { type: Boolean, default: false },
      opened: { type: Boolean, default: false },
      clicked: { type: Boolean, default: false },
      sentAt: { type: Date, default: null }
    },
    sms: {
      sent: { type: Boolean, default: false },
      delivered: { type: Boolean, default: false },
      sentAt: { type: Date, default: null }
    },
    push: {
      sent: { type: Boolean, default: false },
      delivered: { type: Boolean, default: false },
      sentAt: { type: Date, default: null }
    },
    whatsapp: {
      sent: { type: Boolean, default: false },
      delivered: { type: Boolean, default: false },
      read: { type: Boolean, default: false },
      sentAt: { type: Date, default: null }
    }
  },

  // Scheduling
  scheduledFor: {
    type: Date,
    default: null
  },
  expiresAt: {
    type: Date,
    default: null
  },

  // Status
  status: {
    type: String,
    enum: ['pending', 'sent', 'delivered', 'read', 'expired', 'failed'],
    default: 'pending'
  },
  isActive: {
    type: Boolean,
    default: true
  },

  // Timestamps
  createdAt: {
    type: Date,
    default: Date.now
  },
  sentAt: {
    type: Date,
    default: null
  },
  readAt: {
    type: Date,
    default: null
  }
}, {
  timestamps: true,
  toJSON: { virtuals: true },
  toObject: { virtuals: true }
});

// Indexes
notificationSchema.index({ recipientId: 1, createdAt: -1 });
notificationSchema.index({ type: 1, createdAt: -1 });
notificationSchema.index({ priority: 1, createdAt: -1 });
notificationSchema.index({ status: 1, createdAt: -1 });
notificationSchema.index({ isUrgent: 1, createdAt: -1 });
notificationSchema.index({ scheduledFor: 1 });
notificationSchema.index({ expiresAt: 1 });

// Virtuals
notificationSchema.virtual('isRead').get(function() {
  return this.deliveryStatus.inApp.read;
});

notificationSchema.virtual('isDelivered').get(function() {
  return this.status === 'delivered' || this.status === 'read';
});

notificationSchema.virtual('isExpired').get(function() {
  return this.expiresAt && this.expiresAt < new Date();
});

notificationSchema.virtual('canExpire').get(function() {
  return this.expiresAt !== null;
});

// Pre-save middleware
notificationSchema.pre('save', function(next) {
  // Auto-set urgent flag for high priority
  if (this.priority === 'urgent') {
    this.isUrgent = true;
  }
  
  // Set default expiration for non-urgent notifications
  if (!this.expiresAt && this.priority !== 'urgent') {
    this.expiresAt = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); // 30 days
  }
  
  next();
});

// Static methods
notificationSchema.statics = {
  // Create notification
  async createNotification(data) {
    const notification = new this(data);
    await notification.save();
    return notification;
  },

  // Get user notifications
  async getUserNotifications(userId, options = {}) {
    const {
      limit = 50,
      offset = 0,
      unreadOnly = false,
      type = null,
      priority = null
    } = options;

    const query = {
      recipientId: userId,
      isActive: true
    };

    if (unreadOnly) {
      query['deliveryStatus.inApp.read'] = false;
    }

    if (type) {
      query.type = type;
    }

    if (priority) {
      query.priority = priority;
    }

    return await this.find(query)
      .sort({ createdAt: -1 })
      .skip(offset)
      .limit(limit);
  },

  // Get unread count
  async getUnreadCount(userId) {
    return await this.countDocuments({
      recipientId: userId,
      isActive: true,
      'deliveryStatus.inApp.read': false
    });
  },

  // Mark as read
  async markAsRead(notificationId, userId) {
    return await this.updateOne(
      {
        _id: notificationId,
        recipientId: userId
      },
      {
        'deliveryStatus.inApp.read': true,
        'deliveryStatus.inApp.readAt': new Date(),
        status: 'read',
        readAt: new Date()
      }
    );
  },

  // Mark all as read
  async markAllAsRead(userId) {
    return await this.updateMany(
      {
        recipientId: userId,
        isActive: true,
        'deliveryStatus.inApp.read': false
      },
      {
        'deliveryStatus.inApp.read': true,
        'deliveryStatus.inApp.readAt': new Date(),
        status: 'read',
        readAt: new Date()
      }
    );
  },

  // Get urgent notifications
  async getUrgentNotifications() {
    return await this.find({
      isUrgent: true,
      isActive: true,
      status: { $in: ['pending', 'sent'] }
    })
    .sort({ createdAt: -1 })
    .populate('recipientId', 'name email role');
  },

  // Get scheduled notifications
  async getScheduledNotifications() {
    const now = new Date();
    return await this.find({
      scheduledFor: { $lte: now },
      status: 'pending',
      isActive: true
    });
  },

  // Get notification stats
  async getNotificationStats(days = 30) {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);

    return await this.aggregate([
      {
        $match: {
          createdAt: { $gte: startDate }
        }
      },
      {
        $group: {
          _id: {
            date: { $dateToString: { format: '%Y-%m-%d', date: '$createdAt' } },
            type: '$type'
          },
          count: { $sum: 1 },
          readCount: {
            $sum: { $cond: ['$deliveryStatus.inApp.read', 1, 0] }
          }
        }
      },
      {
        $sort: { '_id.date': -1 }
      }
    ]);
  },

  // Clean expired notifications
  async cleanExpiredNotifications() {
    const result = await this.updateMany(
      {
        expiresAt: { $lt: new Date() },
        status: { $ne: 'read' }
      },
      {
        status: 'expired',
        isActive: false
      }
    );

    return result.modifiedCount;
  }
};

// Instance methods
notificationSchema.methods = {
  // Mark as read
  async markAsRead() {
    this.deliveryStatus.inApp.read = true;
    this.deliveryStatus.inApp.readAt = new Date();
    this.status = 'read';
    this.readAt = new Date();
    await this.save();
  },

  // Update delivery status
  async updateDeliveryStatus(channel, status) {
    if (this.deliveryStatus[channel]) {
      Object.assign(this.deliveryStatus[channel], status);
      if (status.sent) {
        this.deliveryStatus[channel].sentAt = new Date();
      }
      await this.save();
    }
  },

  // Send notification
  async send() {
    this.status = 'sent';
    this.sentAt = new Date();
    await this.save();
  }
};

if (process.env.NODE_ENV === 'test') {
  const mockNotificationModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockNotificationId', ...data }),
    };
  };
  mockNotificationModel.find = jest.fn().mockResolvedValue([]);
  mockNotificationModel.findOne = jest.fn().mockResolvedValue(null);
  mockNotificationModel.findById = jest.fn().mockResolvedValue(null);
  mockNotificationModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockNotificationModel.create = jest.fn().mockResolvedValue({ _id: 'mockNotificationId' });
  module.exports = mockNotificationModel;
} else {
  module.exports = mongoose.model('Notification', notificationSchema);
}