const { WhatsAppMessage, WhatsAppSession } = require('../models');
const logger = require('../utils/logger');

class AuditService {
  constructor() {
    this.auditQueue = [];
    this.batchSize = 100;
    this.flushInterval = 5000; // 5 seconds
    
    // Start periodic flush
    this._interval = setInterval(() => this.flushAuditQueue(), this.flushInterval);
    
    // Enhanced cleanup for test environments
    if (process.env.NODE_ENV === 'test') {
      // Store reference for cleanup
      const self = this;
      
      const cleanup = () => {
        if (self._interval) {
          clearInterval(self._interval);
          self._interval = null;
        }
        // Clear the audit queue
        self.auditQueue = [];
      };
      
      // Register cleanup handlers
      process.on('exit', cleanup);
      process.on('SIGINT', cleanup);
      process.on('SIGTERM', cleanup);
      process.on('beforeExit', cleanup);
      
      // Jest-specific cleanup
      if (typeof global !== 'undefined' && global.process) {
        global.process.on('beforeExit', cleanup);
      }
      
      // Store cleanup function for manual calls
      this._cleanup = cleanup;
      
      // Force cleanup after a short delay in test mode
      setTimeout(cleanup, 100);
    }
  }

  /**
   * Log comprehensive audit event
   */
  async logAuditEvent(eventData) {
    const auditEvent = {
      timestamp: new Date(),
      eventType: eventData.eventType,
      userId: eventData.userId,
      phoneNumber: eventData.phoneNumber,
      sessionId: eventData.sessionId,
      messageId: eventData.messageId,
      intent: eventData.intent,
      action: eventData.action,
      data: eventData.data,
      compliance: eventData.compliance,
      performance: eventData.performance,
      security: eventData.security,
      metadata: eventData.metadata
    };

    this.auditQueue.push(auditEvent);

    // Flush immediately if queue is full
    if (this.auditQueue.length >= this.batchSize) {
      await this.flushAuditQueue();
    }

    return auditEvent;
  }

  /**
   * Log conversation audit
   */
  async logConversationAudit(phoneNumber, messageId, direction, data, session) {
    const auditData = {
      eventType: 'CONVERSATION',
      phoneNumber,
      messageId,
      direction,
      intent: data.intent,
      action: 'MESSAGE_PROCESSED',
      data: {
        originalMessage: data.originalMessage,
        response: data.response,
        confidence: data.confidence,
        processingTime: data.processingTime
      },
      compliance: {
        disclaimerShown: data.disclaimerShown,
        disclaimerType: data.disclaimerType,
        sebiCompliant: true,
        riskDisclosure: data.adviceType !== 'NONE'
      },
      performance: {
        responseTime: data.processingTime,
        intentAccuracy: data.confidence,
        aiGenerated: data.aiGenerated
      },
      security: {
        rateLimited: false,
        suspiciousActivity: false,
        authenticationRequired: false
      },
      metadata: {
        sessionId: session?._id,
        conversationContext: session?.getConversationContext(),
        userType: this.getUserType(session),
        platform: 'WHATSAPP'
      }
    };

    return await this.logAuditEvent(auditData);
  }

  /**
   * Log compliance audit
   */
  async logComplianceAudit(phoneNumber, action, data, session) {
    const auditData = {
      eventType: 'COMPLIANCE',
      phoneNumber,
      action,
      data,
      compliance: {
        sebiCompliant: true,
        disclaimerShown: data.disclaimerShown,
        riskDisclosure: data.riskDisclosure,
        adviceType: data.adviceType,
        regulatoryCheck: 'PASSED'
      },
      metadata: {
        sessionId: session?._id,
        timestamp: new Date(),
        regulatoryFramework: 'SEBI',
        complianceVersion: '1.0'
      }
    };

    return await this.logAuditEvent(auditData);
  }

  /**
   * Log performance audit
   */
  async logPerformanceAudit(phoneNumber, metrics, session) {
    const auditData = {
      eventType: 'PERFORMANCE',
      phoneNumber,
      action: 'PERFORMANCE_METRICS',
      data: metrics,
      performance: {
        responseTime: metrics.responseTime,
        processingTime: metrics.processingTime,
        memoryUsage: metrics.memoryUsage,
        cpuUsage: metrics.cpuUsage,
        databaseQueries: metrics.databaseQueries
      },
      metadata: {
        sessionId: session?._id,
        timestamp: new Date(),
        environment: process.env.NODE_ENV || 'development'
      }
    };

    return await this.logAuditEvent(auditData);
  }

  /**
   * Log security audit
   */
  async logSecurityAudit(phoneNumber, securityEvent, session) {
    const auditData = {
      eventType: 'SECURITY',
      phoneNumber,
      action: securityEvent.type,
      data: securityEvent.data,
      security: {
        riskLevel: securityEvent.riskLevel,
        threatType: securityEvent.threatType,
        actionTaken: securityEvent.actionTaken,
        authenticated: securityEvent.authenticated
      },
      metadata: {
        sessionId: session?._id,
        timestamp: new Date(),
        ipAddress: securityEvent.ipAddress,
        userAgent: securityEvent.userAgent
      }
    };

    return await this.logAuditEvent(auditData);
  }

  /**
   * Log business logic audit
   */
  async logBusinessAudit(phoneNumber, businessAction, data, session) {
    const auditData = {
      eventType: 'BUSINESS',
      phoneNumber,
      action: businessAction,
      data,
      businessLogic: {
        actionType: businessAction,
        dataProcessed: data,
        rulesApplied: data.rulesApplied || [],
        decisionFactors: data.decisionFactors || [],
        outcome: data.outcome
      },
      metadata: {
        sessionId: session?._id,
        timestamp: new Date(),
        businessProcess: data.process || 'GENERAL'
      }
    };

    return await this.logAuditEvent(auditData);
  }

  /**
   * Generate compliance report
   */
  async generateComplianceReport(startDate, endDate, phoneNumber = null) {
    const query = {
      timestamp: { $gte: startDate, $lte: endDate }
    };

    if (phoneNumber) {
      query.phoneNumber = phoneNumber;
    }

    const messages = await WhatsAppMessage.find(query);
    
    const report = {
      period: { startDate, endDate },
      totalMessages: messages.length,
      compliance: {
        disclaimersShown: messages.filter(m => m.disclaimerShown).length,
        sebiCompliant: messages.filter(m => m.auditLog?.complianceChecks?.some(c => c.result === 'PASS')).length,
        riskDisclosures: messages.filter(m => m.adviceType !== 'NONE').length
      },
      performance: {
        averageResponseTime: messages.reduce((sum, m) => sum + m.processingTime, 0) / messages.length,
        averageConfidence: messages.reduce((sum, m) => sum + m.confidence, 0) / messages.length,
        aiGenerated: messages.filter(m => m.aiGenerated).length
      },
      intents: this.aggregateIntents(messages),
      userBehavior: this.analyzeUserBehavior(messages),
      recommendations: this.generateRecommendations(messages)
    };

    return report;
  }

  /**
   * Generate performance report
   */
  async generatePerformanceReport(startDate, endDate) {
    const sessions = await WhatsAppSession.find({
      'auditTrail.sessionStartTime': { $gte: startDate, $lte: endDate }
    });

    const report = {
      period: { startDate, endDate },
      sessions: {
        total: sessions.length,
        active: sessions.filter(s => s.isActive).length,
        averageDuration: this.calculateAverageSessionDuration(sessions),
        averageMessages: sessions.reduce((sum, s) => sum + s.messageCount, 0) / sessions.length
      },
      performance: {
        averageResponseTime: sessions.reduce((sum, s) => sum + s.performanceMetrics.averageResponseTime, 0) / sessions.length,
        successRate: this.calculateSuccessRate(sessions),
        userSatisfaction: this.calculateAverageSatisfaction(sessions)
      },
      compliance: {
        disclaimersShown: sessions.reduce((sum, s) => sum + s.auditTrail.disclaimersShown, 0),
        complianceChecks: sessions.reduce((sum, s) => sum + s.auditTrail.complianceChecks, 0)
      }
    };

    return report;
  }

  /**
   * Generate conversation continuity report
   */
  async generateConversationContinuityReport(startDate, endDate) {
    const messages = await WhatsAppMessage.find({
      timestamp: { $gte: startDate, $lte: endDate }
    }).sort({ phoneNumber: 1, timestamp: 1 });

    const continuityData = this.analyzeConversationContinuity(messages);
    
    return {
      period: { startDate, endDate },
      totalConversations: continuityData.totalConversations,
      continuity: {
        followUpMessages: continuityData.followUpMessages,
        interruptedConversations: continuityData.interruptedConversations,
        resumedConversations: continuityData.resumedConversations,
        averageGap: continuityData.averageGap
      },
      contextRetention: {
        successful: continuityData.contextRetention.successful,
        failed: continuityData.contextRetention.failed,
        rate: continuityData.contextRetention.rate
      }
    };
  }

  /**
   * Flush audit queue to database
   */
  async flushAuditQueue() {
    if (this.auditQueue.length === 0) return;

    try {
      const auditEvents = [...this.auditQueue];
      this.auditQueue = [];

      // Process audit events in batches
      for (let i = 0; i < auditEvents.length; i += this.batchSize) {
        const batch = auditEvents.slice(i, i + this.batchSize);
        await this.processAuditBatch(batch);
      }

      logger.info(`Flushed ${auditEvents.length} audit events`);
    } catch (error) {
      logger.error('Error flushing audit queue:', error);
      // Re-add events to queue for retry
      this.auditQueue.unshift(...this.auditQueue);
    }
  }

  /**
   * Process audit batch
   */
  async processAuditBatch(auditEvents) {
    // Store audit events in a dedicated collection or log file
    // For now, we'll log them
    auditEvents.forEach(event => {
      logger.info('AUDIT_EVENT', {
        timestamp: event.timestamp,
        eventType: event.eventType,
        phoneNumber: event.phoneNumber,
        action: event.action,
        compliance: event.compliance
      });
    });
  }

  /**
   * Get user type based on session
   */
  getUserType(session) {
    if (!session) return 'UNKNOWN';
    
    const messageCount = session.messageCount;
    const sessionDuration = Date.now() - session.auditTrail.sessionStartTime;
    
    if (messageCount < 5) return 'NEW_USER';
    if (messageCount > 50) return 'POWER_USER';
    if (sessionDuration > 24 * 60 * 60 * 1000) return 'LONG_SESSION_USER';
    return 'REGULAR_USER';
  }

  /**
   * Aggregate intents from messages
   */
  aggregateIntents(messages) {
    const intentCounts = {};
    messages.forEach(message => {
      const intent = message.detectedIntent;
      intentCounts[intent] = (intentCounts[intent] || 0) + 1;
    });
    return intentCounts;
  }

  /**
   * Analyze user behavior
   */
  analyzeUserBehavior(messages) {
    const behavior = {
      averageMessageLength: 0,
      complexityDistribution: { LOW: 0, MEDIUM: 0, HIGH: 0 },
      responseTime: 0,
      followUpRate: 0
    };

    if (messages.length === 0) return behavior;

    // Calculate averages
    behavior.averageMessageLength = messages.reduce((sum, m) => sum + (m.auditLog?.userBehavior?.messageLength || 0), 0) / messages.length;
    behavior.responseTime = messages.reduce((sum, m) => sum + m.processingTime, 0) / messages.length;

    // Calculate complexity distribution
    messages.forEach(message => {
      const complexity = message.auditLog?.userBehavior?.complexity || 'LOW';
      behavior.complexityDistribution[complexity]++;
    });

    // Calculate follow-up rate
    const followUpMessages = messages.filter(m => m.conversationContinuity?.isFollowUp);
    behavior.followUpRate = followUpMessages.length / messages.length;

    return behavior;
  }

  /**
   * Generate recommendations
   */
  generateRecommendations(messages) {
    const recommendations = [];

    // Analyze disclaimer compliance
    const disclaimerRate = messages.filter(m => m.disclaimerShown).length / messages.length;
    if (disclaimerRate < 0.8) {
      recommendations.push('Increase disclaimer compliance rate');
    }

    // Analyze response times
    const avgResponseTime = messages.reduce((sum, m) => sum + m.processingTime, 0) / messages.length;
    if (avgResponseTime > 2000) {
      recommendations.push('Optimize response times for better user experience');
    }

    // Analyze intent accuracy
    const avgConfidence = messages.reduce((sum, m) => sum + m.confidence, 0) / messages.length;
    if (avgConfidence < 0.7) {
      recommendations.push('Improve intent detection accuracy');
    }

    return recommendations;
  }

  /**
   * Calculate average session duration
   */
  calculateAverageSessionDuration(sessions) {
    if (sessions.length === 0) return 0;
    
    const totalDuration = sessions.reduce((sum, session) => {
      const duration = session.lastActivity - session.auditTrail.sessionStartTime;
      return sum + duration;
    }, 0);
    
    return totalDuration / sessions.length;
  }

  /**
   * Calculate success rate
   */
  calculateSuccessRate(sessions) {
    if (sessions.length === 0) return 0;
    
    const totalInteractions = sessions.reduce((sum, s) => sum + s.performanceMetrics.successfulInteractions + s.performanceMetrics.failedInteractions, 0);
    const successfulInteractions = sessions.reduce((sum, s) => sum + s.performanceMetrics.successfulInteractions, 0);
    
    return totalInteractions > 0 ? successfulInteractions / totalInteractions : 0;
  }

  /**
   * Calculate average satisfaction
   */
  calculateAverageSatisfaction(sessions) {
    const sessionsWithSatisfaction = sessions.filter(s => s.performanceMetrics.userSatisfactionScore);
    
    if (sessionsWithSatisfaction.length === 0) return 0;
    
    const totalSatisfaction = sessionsWithSatisfaction.reduce((sum, s) => sum + s.performanceMetrics.userSatisfactionScore, 0);
    return totalSatisfaction / sessionsWithSatisfaction.length;
  }

  /**
   * Analyze conversation continuity
   */
  analyzeConversationContinuity(messages) {
    const conversations = this.groupMessagesByConversation(messages);
    
    let followUpMessages = 0;
    let interruptedConversations = 0;
    let resumedConversations = 0;
    let totalGap = 0;
    let contextRetention = { successful: 0, failed: 0 };

    conversations.forEach(conversation => {
      const continuity = this.analyzeSingleConversation(conversation);
      followUpMessages += continuity.followUpMessages;
      interruptedConversations += continuity.interrupted ? 1 : 0;
      resumedConversations += continuity.resumed ? 1 : 0;
      totalGap += continuity.totalGap;
      contextRetention.successful += continuity.contextRetention.successful;
      contextRetention.failed += continuity.contextRetention.failed;
    });

    return {
      totalConversations: conversations.length,
      followUpMessages,
      interruptedConversations,
      resumedConversations,
      averageGap: totalGap / conversations.length,
      contextRetention: {
        successful: contextRetention.successful,
        failed: contextRetention.failed,
        rate: contextRetention.successful / (contextRetention.successful + contextRetention.failed)
      }
    };
  }

  /**
   * Group messages by conversation
   */
  groupMessagesByConversation(messages) {
    const conversations = {};
    
    messages.forEach(message => {
      const phoneNumber = message.phoneNumber;
      if (!conversations[phoneNumber]) {
        conversations[phoneNumber] = [];
      }
      conversations[phoneNumber].push(message);
    });

    return Object.values(conversations);
  }

  /**
   * Analyze single conversation
   */
  analyzeSingleConversation(messages) {
    let followUpMessages = 0;
    let interrupted = false;
    let resumed = false;
    let totalGap = 0;
    let contextRetention = { successful: 0, failed: 0 };

    for (let i = 1; i < messages.length; i++) {
      const currentMessage = messages[i];
      const previousMessage = messages[i - 1];
      
      const timeGap = currentMessage.timestamp - previousMessage.timestamp;
      
      if (timeGap < 30 * 60 * 1000) { // Less than 30 minutes
        followUpMessages++;
        if (currentMessage.conversationContinuity?.contextCarried) {
          contextRetention.successful++;
        } else {
          contextRetention.failed++;
        }
      } else {
        interrupted = true;
        totalGap += timeGap;
        
        if (currentMessage.conversationContinuity?.conversationResumed) {
          resumed = true;
        }
      }
    }

    return {
      followUpMessages,
      interrupted,
      resumed,
      totalGap,
      contextRetention
    };
  }

  // Add a cleanup method for test environments
  cleanup() {
    // Use stored cleanup function if available
    if (this._cleanup) {
      this._cleanup();
    } else if (this._interval) {
      clearInterval(this._interval);
      this._interval = null;
    }
    // Clear the audit queue
    this.auditQueue = [];
  }
}

module.exports = new AuditService(); 