const mongoose = require('mongoose');

const whatsAppMessageSchema = new mongoose.Schema({
  phoneNumber: {
    type: String,
    required: true,
    index: true
  },
  userId: {
    type: String,
    ref: 'User',
    index: true
  },
  // Message details
  messageId: {
    type: String,
    required: true,
    unique: true
  },
  direction: {
    type: String,
    enum: ['INBOUND', 'OUTBOUND'],
    required: true
  },
  messageType: {
    type: String,
    enum: ['TEXT', 'MEDIA', 'DOCUMENT', 'LOCATION', 'CONTACT', 'STICKER'],
    default: 'TEXT'
  },
  // Content
  content: {
    text: String,
    mediaUrl: String,
    fileName: String,
    fileSize: Number,
    mimeType: String
  },
  // Enhanced intent and context
  detectedIntent: {
    type: String,
    enum: [
      'GREETING', 'ONBOARDING', 'PORTFOLIO_VIEW', 'SIP_CREATE', 'SIP_STOP', 
      'SIP_STATUS', 'LUMP_SUM', 'AI_ANALYSIS', 'STATEMENT', 'REWARDS', 
      'REFERRAL', 'LEADERBOARD', 'COPY_PORTFOLIO', 'HELP', 'UNKNOWN',
      'CONFIRMATION', 'FUND_RESEARCH', 'MARKET_UPDATE', 'KYC_UPDATE', 'PASSWORD_RESET'
    ],
    default: 'UNKNOWN'
  },
  confidence: {
    type: Number,
    min: 0,
    max: 1,
    default: 0
  },
  // Enhanced processing
  processingTime: {
    type: Number, // in milliseconds
    default: 0
  },
  aiGenerated: {
    type: Boolean,
    default: false
  },
  aiProvider: {
    type: String,
    enum: ['GEMINI', 'OPENAI', 'NONE'],
    default: 'NONE'
  },
  // Response tracking
  responseMessageId: String,
  responseTime: Date,
  // Enhanced error handling
  error: {
    code: String,
    message: String,
    stack: String,
    retryCount: { type: Number, default: 0 },
    resolved: { type: Boolean, default: false }
  },
  // Enhanced metadata
  sessionId: String,
  conversationId: String,
  timestamp: {
    type: Date,
    default: Date.now
  },
  // Enhanced SEBI compliance
  disclaimerShown: {
    type: Boolean,
    default: false
  },
  disclaimerType: {
    type: String,
    enum: ['GENERAL', 'INVESTMENT_ADVICE', 'FUND_ANALYSIS', 'PORTFOLIO', 'RISK_DISCLOSURE', 'NONE'],
    default: 'NONE'
  },
  adviceType: {
    type: String,
    enum: ['GENERAL', 'FUND_ANALYSIS', 'PORTFOLIO', 'INVESTMENT', 'NONE'],
    default: 'NONE'
  },
  // Enhanced analytics
  userSatisfaction: {
    type: Number,
    min: 1,
    max: 5
  },
  tags: [String],
  isActive: {
    type: Boolean,
    default: true
  },
  // New fields for comprehensive audit logging
  auditLog: {
    // Conversation context
    conversationContext: {
      previousIntent: String,
      conversationFlow: [String],
      userPreferences: mongoose.Schema.Types.Mixed,
      sessionDuration: Number
    },
    // Compliance tracking
    complianceChecks: [{
      checkType: String,
      timestamp: Date,
      result: String,
      details: String
    }],
    // Performance metrics
    performanceMetrics: {
      intentDetectionTime: Number,
      aiProcessingTime: Number,
      databaseQueryTime: Number,
      totalProcessingTime: Number,
      memoryUsage: Number
    },
    // User behavior tracking
    userBehavior: {
      responseTime: Number,
      messageLength: Number,
      complexity: { type: String, enum: ['LOW', 'MEDIUM', 'HIGH'] },
      interactionPattern: String
    },
    // Security and validation
    securityChecks: [{
      checkType: String,
      timestamp: Date,
      result: String,
      riskLevel: { type: String, enum: ['LOW', 'MEDIUM', 'HIGH'] }
    }]
  },
  // Conversation continuity
  conversationContinuity: {
    isFollowUp: { type: Boolean, default: false },
    previousMessageId: String,
    contextCarried: Boolean,
    timeSinceLastMessage: Number,
    conversationResumed: Boolean
  },
  // Business logic tracking
  businessLogic: {
    actionTaken: String,
    dataProcessed: mongoose.Schema.Types.Mixed,
    rulesApplied: [String],
    decisionFactors: [String],
    outcome: String
  },
  // Quality assurance
  qualityMetrics: {
    intentAccuracy: Number,
    responseRelevance: Number,
    userIntentMet: Boolean,
    followUpRequired: Boolean,
    escalationNeeded: Boolean
  }
}, {
  timestamps: true
});

// Indexes for performance optimization
// whatsAppMessageSchema.index({ messageId: 1 }); // Duplicate of inline index: true
// whatsAppMessageSchema.index({ phoneNumber: 1, timestamp: -1 }); // phoneNumber has inline index: true

// Pre-save middleware for audit logging
whatsAppMessageSchema.pre('save', function(next) {
  // Auto-calculate message complexity
  if (this.content && this.content.text) {
    const text = this.content.text;
    const wordCount = text.split(' ').length;
    const charCount = text.length;
    
    if (wordCount > 20 || charCount > 200) {
      this.auditLog.userBehavior.complexity = 'HIGH';
    } else if (wordCount > 10 || charCount > 100) {
      this.auditLog.userBehavior.complexity = 'MEDIUM';
    } else {
      this.auditLog.userBehavior.complexity = 'LOW';
    }
    
    this.auditLog.userBehavior.messageLength = charCount;
  }
  
  next();
});

// Methods for audit logging
whatsAppMessageSchema.methods.addComplianceCheck = function(checkType, result, details) {
  this.auditLog.complianceChecks.push({
    checkType,
    timestamp: new Date(),
    result,
    details
  });
  return this;
};

whatsAppMessageSchema.methods.addSecurityCheck = function(checkType, result, riskLevel) {
  this.auditLog.securityChecks.push({
    checkType,
    timestamp: new Date(),
    result,
    riskLevel
  });
  return this;
};

whatsAppMessageSchema.methods.updatePerformanceMetrics = function(metrics) {
  this.auditLog.performanceMetrics = {
    ...this.auditLog.performanceMetrics,
    ...metrics
  };
  return this;
};

whatsAppMessageSchema.methods.setConversationContext = function(context) {
  this.auditLog.conversationContext = {
    ...this.auditLog.conversationContext,
    ...context
  };
  return this;
};

// Static methods for analytics
whatsAppMessageSchema.statics.getComplianceReport = function(startDate, endDate) {
  return this.aggregate([
    {
      $match: {
        timestamp: { $gte: startDate, $lte: endDate }
      }
    },
    {
      $group: {
        _id: '$disclaimerType',
        count: { $sum: 1 },
        avgConfidence: { $avg: '$confidence' }
      }
    }
  ]);
};

whatsAppMessageSchema.statics.getPerformanceMetrics = function(startDate, endDate) {
  return this.aggregate([
    {
      $match: {
        timestamp: { $gte: startDate, $lte: endDate }
      }
    },
    {
      $group: {
        _id: null,
        avgProcessingTime: { $avg: '$processingTime' },
        avgIntentAccuracy: { $avg: '$qualityMetrics.intentAccuracy' },
        totalMessages: { $sum: 1 },
        aiGeneratedCount: { $sum: { $cond: ['$aiGenerated', 1, 0] } }
      }
    }
  ]);
};

if (process.env.NODE_ENV === 'test') {
  const mockWhatsAppMessageModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockWhatsAppMessageId', ...data }),
    };
  };
  mockWhatsAppMessageModel.find = jest.fn().mockResolvedValue([]);
  mockWhatsAppMessageModel.findOne = jest.fn().mockResolvedValue(null);
  mockWhatsAppMessageModel.findById = jest.fn().mockResolvedValue(null);
  mockWhatsAppMessageModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockWhatsAppMessageModel.create = jest.fn().mockResolvedValue({ _id: 'mockWhatsAppMessageId' });
  module.exports = mockWhatsAppMessageModel;
} else {
  module.exports = mongoose.model('WhatsAppMessage', whatsAppMessageSchema);
}