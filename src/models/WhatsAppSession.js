const mongoose = require('mongoose');

const whatsAppSessionSchema = new mongoose.Schema({
  phoneNumber: {
    type: String,
    required: true,
    unique: true,
    index: true
  },
  userId: {
    type: String,
    ref: 'User',
    index: true
  },
  // Onboarding state
  onboardingState: {
    type: String,
    enum: ['INITIAL', 'STARTED', 'NAME_COLLECTED', 'EMAIL_COLLECTED', 'PAN_COLLECTED', 'KYC_VERIFIED', 'COMPLETED'],
    default: 'INITIAL'
  },
  // Collected data during onboarding
  onboardingData: {
    name: String,
    email: String,
    pan: String,
    kycStatus: {
      type: String,
      enum: ['PENDING', 'VERIFIED', 'REJECTED'],
      default: 'PENDING'
    }
  },
  // Enhanced conversation context and memory
  conversationMemory: {
    // Last 10 messages for context
    recentMessages: [{
      timestamp: Date,
      direction: { type: String, enum: ['INBOUND', 'OUTBOUND'] },
      content: String,
      intent: String,
      confidence: Number
    }],
    // Current conversation context
    currentContext: {
      lastIntent: String,
      pendingAction: String,
      tempData: mongoose.Schema.Types.Mixed,
      conversationFlow: [String], // Track conversation flow
      lastQueryTime: Date,
      sessionStartTime: Date
    },
    // User preferences and behavior
    userPreferences: {
      preferredLanguage: { type: String, default: 'en' },
      investmentStyle: { type: String, enum: ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'] },
      preferredFundTypes: [String],
      notificationPreferences: {
        portfolioUpdates: { type: Boolean, default: true },
        marketAlerts: { type: Boolean, default: true },
        sipReminders: { type: Boolean, default: true }
      }
    },
    // Conversation history for continuity
    conversationHistory: [{
      sessionId: String,
      startTime: Date,
      endTime: Date,
      messageCount: Number,
      topics: [String],
      actions: [String]
    }]
  },
  // Current intent and context
  currentIntent: {
    type: String,
    enum: [
      'GREETING', 'ONBOARDING', 'PORTFOLIO_VIEW', 'SIP_CREATE', 'SIP_STOP', 
      'SIP_STATUS', 'LUMP_SUM', 'AI_ANALYSIS', 'STATEMENT', 'REWARDS', 
      'REFERRAL', 'LEADERBOARD', 'COPY_PORTFOLIO', 'HELP', 'CONFIRMATION',
      'FUND_RESEARCH', 'MARKET_UPDATE', 'KYC_UPDATE', 'PASSWORD_RESET'
    ]
  },
  context: {
    lastMessage: String,
    pendingAction: String,
    tempData: mongoose.Schema.Types.Mixed,
    // Enhanced context for complex scenarios
    multiStepFlow: {
      isActive: { type: Boolean, default: false },
      currentStep: Number,
      totalSteps: Number,
      flowType: String,
      collectedData: mongoose.Schema.Types.Mixed
    },
    // Context for interrupted conversations
    interruptedConversation: {
      wasInterrupted: { type: Boolean, default: false },
      lastIntent: String,
      pendingData: mongoose.Schema.Types.Mixed,
      resumePoint: String
    }
  },
  // Session management
  isActive: {
    type: Boolean,
    default: true
  },
  lastActivity: {
    type: Date,
    default: Date.now
  },
  messageCount: {
    type: Number,
    default: 0
  },
  // Rate limiting
  lastMessageTime: Date,
  messageRate: {
    type: Number,
    default: 0
  },
  // Enhanced preferences
  preferences: {
    language: {
      type: String,
      default: 'en'
    },
    notifications: {
      type: Boolean,
      default: true
    },
    aiInsights: {
      type: Boolean,
      default: true
    },
    // New preferences for better UX
    responseStyle: {
      type: String,
      enum: ['CONCISE', 'DETAILED', 'BALANCED'],
      default: 'BALANCED'
    },
    autoSuggestions: {
      type: Boolean,
      default: true
    }
  },
  // Audit and compliance tracking
  auditTrail: {
    sessionStartTime: { type: Date, default: Date.now },
    totalMessages: { type: Number, default: 0 },
    disclaimersShown: { type: Number, default: 0 },
    complianceChecks: { type: Number, default: 0 },
    lastComplianceCheck: Date,
    sebiCompliant: { type: Boolean, default: true }
  },
  // Performance metrics
  performanceMetrics: {
    averageResponseTime: { type: Number, default: 0 },
    totalProcessingTime: { type: Number, default: 0 },
    successfulInteractions: { type: Number, default: 0 },
    failedInteractions: { type: Number, default: 0 },
    userSatisfactionScore: { type: Number, min: 1, max: 5 }
  }
}, {
  timestamps: true
});

// Remove duplicate index declarations
// whatsAppSessionSchema.index({ sessionId: 1 }); // Duplicate of inline index: true
// whatsAppSessionSchema.index({ phoneNumber: 1 }); // Duplicate of inline index: true

// Pre-save middleware to update lastActivity
whatsAppSessionSchema.pre('save', function(next) {
  this.lastActivity = new Date();
  next();
});

// Methods for session management
whatsAppSessionSchema.methods.addMessageToMemory = function(message, direction, intent, confidence) {
  const memoryEntry = {
    timestamp: new Date(),
    direction,
    content: message.substring(0, 200), // Limit content length
    intent,
    confidence
  };
  
  this.conversationMemory.recentMessages.push(memoryEntry);
  
  // Keep only last 10 messages
  if (this.conversationMemory.recentMessages.length > 10) {
    this.conversationMemory.recentMessages.shift();
  }
  
  // Update conversation flow
  this.conversationMemory.currentContext.conversationFlow.push(intent);
  this.conversationMemory.currentContext.lastQueryTime = new Date();
  
  return this;
};

whatsAppSessionSchema.methods.getConversationContext = function() {
  return {
    recentMessages: this.conversationMemory.recentMessages,
    currentIntent: this.currentIntent,
    pendingAction: this.context.pendingAction,
    userPreferences: this.conversationMemory.userPreferences,
    conversationFlow: this.conversationMemory.currentContext.conversationFlow.slice(-5) // Last 5 intents
  };
};

whatsAppSessionSchema.methods.updatePerformanceMetrics = function(responseTime, success) {
  this.performanceMetrics.totalProcessingTime += responseTime;
  this.performanceMetrics.averageResponseTime = 
    this.performanceMetrics.totalProcessingTime / this.messageCount;
  
  if (success) {
    this.performanceMetrics.successfulInteractions++;
  } else {
    this.performanceMetrics.failedInteractions++;
  }
  
  return this;
};

if (process.env.NODE_ENV === 'test') {
  const mockWhatsAppSessionModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockWhatsAppSessionId', ...data }),
    };
  };
  mockWhatsAppSessionModel.find = jest.fn().mockResolvedValue([]);
  mockWhatsAppSessionModel.findOne = jest.fn().mockResolvedValue(null);
  mockWhatsAppSessionModel.findById = jest.fn().mockResolvedValue(null);
  mockWhatsAppSessionModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockWhatsAppSessionModel.create = jest.fn().mockResolvedValue({ _id: 'mockWhatsAppSessionId' });
  module.exports = mockWhatsAppSessionModel;
} else {
  module.exports = mongoose.model('WhatsAppSession', whatsAppSessionSchema);
}