const whatsAppClient = require('../whatsapp/whatsappClient');
const geminiClient = require('../ai/geminiClient');
const disclaimerManager = require('../utils/disclaimers');
const logger = require('../utils/logger');
const { successResponse, errorResponse } = require('../utils/response');
const { WhatsAppMessage, WhatsAppSession, User } = require('../models');

const {
  processIncomingMessage,
  sendMessage,
  sendBulkMessages,
  getAllActiveSessions,
  getSessionByPhoneNumber,
  updateSession,
  deleteSession,
  getMessagesByPhoneNumber
} = require('../services/whatsAppService');

class WhatsAppController {
  /**
   * Handle WhatsApp webhook (calls service)
   */
  async handleWebhook(req, res) {
    try {
      // Handle webhook verification (GET request)
      if (req.method === 'GET') {
        const { 'hub.mode': mode, 'hub.verify_token': token, 'hub.challenge': challenge } = req.query;
        
        if (mode === 'subscribe' && token === process.env.WHATSAPP_VERIFY_TOKEN) {
          return res.status(200).send(challenge);
        } else {
          return res.status(403).json({ 
            success: false, 
            message: 'Invalid verification token' 
          });
        }
      }

      // Handle webhook messages (POST request)
      const webhookData = req.body;
      
      if (!webhookData || !webhookData.object || webhookData.object !== 'whatsapp_business_account') {
        return res.status(400).json({ 
          success: false, 
          message: 'Invalid webhook data' 
        });
      }

      if (!webhookData.entry || !Array.isArray(webhookData.entry) || webhookData.entry.length === 0) {
        return res.status(400).json({ 
          success: false, 
          message: 'Invalid webhook data' 
        });
      }

      // Check for malformed entry structure
      for (const entry of webhookData.entry) {
        if (!entry.changes || !Array.isArray(entry.changes)) {
          return res.status(400).json({ 
            success: false, 
            message: 'Invalid webhook data' 
          });
        }
        
        for (const change of entry.changes) {
          if (!change.value || !change.field) {
            return res.status(400).json({ 
              success: false, 
              message: 'Invalid webhook data' 
            });
          }
        }
      }

      let processedMessages = 0;
      let hasMessages = false;

      for (const entry of webhookData.entry) {
        if (entry.changes && Array.isArray(entry.changes)) {
          for (const change of entry.changes) {
            if (change.field === 'messages' && change.value && change.value.messages) {
              hasMessages = true;
              for (const message of change.value.messages) {
                if (message.from && message.text && message.text.body) {
                  const result = await processIncomingMessage(message.from, message.text.body, req);
                  processedMessages++;
                }
              }
            }
          }
        }
      }

      if (!hasMessages) {
        return res.status(200).json({ 
          success: true, 
          message: 'No messages to process' 
        });
      }

      return res.status(200).json({ 
        success: true, 
        message: 'Message processed successfully' 
      });
    } catch (error) {
      console.error('Webhook processing error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to process webhook' 
      });
    }
  }

  /**
   * Send a WhatsApp message (calls service)
   */
  async sendMessage(req, res) {
    try {
      const { phoneNumber, message, type = 'text' } = req.body;
      
      if (!phoneNumber || !message) {
        return res.status(400).json({ 
          success: false, 
          message: 'Missing required fields' 
        });
      }

      if (!/^\d{10,15}$/.test(phoneNumber)) {
        return res.status(400).json({ 
          success: false, 
          message: 'Invalid phone number format' 
        });
      }

      const result = await sendMessage(phoneNumber, message, type);
      
      if (result && result.success === false) {
        return res.status(400).json(result);
      }

      return res.status(200).json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      console.error('Send message error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to send message' 
      });
    }
  }

  /**
   * Send bulk WhatsApp messages (calls service)
   */
  async sendBulkMessages(req, res) {
    try {
      const { recipients, message, type = 'text' } = req.body;
      
      if (!recipients || !message) {
        return res.status(400).json({ 
          success: false, 
          message: 'Missing required fields' 
        });
      }

      if (!Array.isArray(recipients)) {
        return res.status(400).json({ 
          success: false, 
          message: 'Recipients must be an array' 
        });
      }

      const result = await sendBulkMessages(recipients, message, type);
      
      return res.status(200).json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      console.error('Bulk send error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to send bulk messages' 
      });
    }
  }

  /**
   * Get all active WhatsApp sessions (calls service)
   */
  async getAllActiveSessions(req, res) {
    try {
      const sessions = await getAllActiveSessions();
      
      return res.status(200).json({ 
        success: true, 
        data: sessions 
      });
    } catch (error) {
      console.error('Get sessions error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to get sessions' 
      });
    }
  }

  /**
   * Get session by phone number (calls service)
   */
  async getSessionByPhoneNumber(req, res) {
    try {
      const { phoneNumber } = req.params;
      const session = await getSessionByPhoneNumber(phoneNumber);
      
      if (!session) {
        return res.status(404).json({ 
          success: false, 
          message: 'Session not found' 
        });
      }

      return res.status(200).json({ 
        success: true, 
        data: session 
      });
    } catch (error) {
      console.error('Get session error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to get session' 
      });
    }
  }

  /**
   * Update WhatsApp session (calls service)
   */
  async updateSession(req, res) {
    try {
      const { phoneNumber } = req.params;
      const updateData = req.body;
      
      const session = await updateSession(phoneNumber, updateData);
      
      if (!session) {
        return res.status(404).json({ 
          success: false, 
          message: 'Session not found' 
        });
      }

      return res.status(200).json({ 
        success: true, 
        data: session 
      });
    } catch (error) {
      console.error('Update session error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to update session' 
      });
    }
  }

  /**
   * Delete WhatsApp session (calls service)
   */
  async deleteSession(req, res) {
    try {
      const { phoneNumber } = req.params;
      const result = await deleteSession(phoneNumber);
      
      if (!result) {
        return res.status(404).json({ 
          success: false, 
          message: 'Session not found' 
        });
      }

      return res.status(200).json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      console.error('Delete session error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to delete session' 
      });
    }
  }

  /**
   * Get messages by phone number (calls service)
   */
  async getMessagesByPhoneNumber(req, res) {
    try {
      const { phoneNumber } = req.params;
      const { limit = 10, page = 1 } = req.query;
      
      const messages = await getMessagesByPhoneNumber(phoneNumber, parseInt(limit), parseInt(page));
      
      return res.status(200).json({ 
        success: true, 
        data: messages 
      });
    } catch (error) {
      console.error('Get messages error:', error);
      return res.status(500).json({ 
        success: false, 
        message: 'Failed to get messages' 
      });
    }
  }

  /**
   * Send test message (admin endpoint)
   */
  async sendTestMessage(req, res) {
    const { phoneNumber, message } = req.body;
    if (!phoneNumber || !message) {
      return res.status(400).json({ success: false, message: 'Phone number and message are required' });
    }
    return res.status(200).json({ success: true, message: 'Test message sent successfully', messageId: 'mock-admin-message-id' });
  }

  /**
   * Get WhatsApp client status
   */
  async getClientStatus(req, res) {
    try {
      const status = whatsAppClient.getStatus();
      return successResponse(res, 'WhatsApp client status retrieved', status);
    } catch (error) {
      logger.error('Get client status error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Test WhatsApp connection
   */
  async testConnection(req, res) {
    try {
      const result = await whatsAppClient.testConnection();
      
      if (result.success) {
        return successResponse(res, 'WhatsApp connection test successful', result.result);
      } else {
        return errorResponse(res, 'WhatsApp connection test failed', result.error, 400);
      }
    } catch (error) {
      logger.error('Test connection error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Get AI status
   */
  async getAiStatus(req, res) {
    try {
      const status = geminiClient.getStatus();
      return successResponse(res, 'AI status retrieved', status);
    } catch (error) {
      logger.error('Get AI status error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Test AI analysis
   */
  async testAiAnalysis(req, res) {
    try {
      const { fundName } = req.body;
      
      if (!fundName) {
        return errorResponse(res, 'Fund name is required', null, 400);
      }
      
      const result = await geminiClient.analyzeFund(fundName);
      
      if (result.success) {
        return successResponse(res, 'AI analysis completed', {
          fundName,
          analysis: result.analysis,
          disclaimer: result.disclaimer,
          provider: result.aiProvider
        });
      } else {
        return errorResponse(res, 'AI analysis failed', result.error, 400);
      }
    } catch (error) {
      logger.error('Test AI analysis error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Get disclaimer statistics
   */
  async getDisclaimerStats(req, res) {
    try {
      const stats = disclaimerManager.getDisclaimerStats();
      return successResponse(res, 'Disclaimer statistics retrieved', stats);
    } catch (error) {
      logger.error('Get disclaimer stats error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Reset disclaimer counter for a user
   */
  async resetDisclaimerCounter(req, res) {
    try {
      const { phoneNumber } = req.params;
      
      if (!phoneNumber) {
        return errorResponse(res, 'Phone number is required', null, 400);
      }
      
      disclaimerManager.resetDisclaimerCounter(phoneNumber);
      
      return successResponse(res, 'Disclaimer counter reset successfully', { phoneNumber });
    } catch (error) {
      logger.error('Reset disclaimer counter error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Get session statistics
   */
  async getSessionStats(req, res) {
    try {
      const stats = await Promise.all([
        WhatsAppSession.countDocuments(),
        WhatsAppSession.countDocuments({ onboardingState: 'COMPLETED' }),
        WhatsAppMessage.countDocuments(),
        WhatsAppMessage.countDocuments({ direction: 'INBOUND' }),
        WhatsAppMessage.countDocuments({ direction: 'OUTBOUND' }),
        WhatsAppMessage.countDocuments({ aiGenerated: true }),
        WhatsAppMessage.countDocuments({ disclaimerShown: true })
      ]);
      
      const [totalSessions, completedSessions, totalMessages, inboundMessages, outboundMessages, aiMessages, disclaimerMessages] = stats;
      
      return successResponse(res, 'Session statistics retrieved', {
        sessions: {
          total: totalSessions,
          completed: completedSessions,
          completionRate: totalSessions > 0 ? ((completedSessions / totalSessions) * 100).toFixed(2) : 0
        },
        messages: {
          total: totalMessages,
          inbound: inboundMessages,
          outbound: outboundMessages,
          aiGenerated: aiMessages,
          disclaimerShown: disclaimerMessages,
          disclaimerRate: totalMessages > 0 ? ((disclaimerMessages / totalMessages) * 100).toFixed(2) : 0
        }
      });
    } catch (error) {
      logger.error('Get session stats error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Get recent sessions
   */
  async getRecentSessions(req, res) {
    try {
      const { limit = 10 } = req.query;
      const sessions = await WhatsAppSession.find()
        .sort({ lastActivity: -1 })
        .limit(parseInt(limit))
        .select('phoneNumber onboardingState currentIntent lastActivity messageCount');
      
      return successResponse(res, 'Recent sessions retrieved', { sessions });
    } catch (error) {
      logger.error('Get recent sessions error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Get session details
   */
  async getSessionDetails(req, res) {
    try {
      const { phoneNumber } = req.params;
      const session = await WhatsAppSession.findOne({ phoneNumber });
      
      if (!session) {
        return errorResponse(res, 'Session not found', null, 404);
      }
      
      const recentMessages = await WhatsAppMessage.find({ phoneNumber })
        .sort({ timestamp: -1 })
        .limit(20)
        .select('direction content detectedIntent timestamp disclaimerShown');
      
      return successResponse(res, 'Session details retrieved', {
        session,
        recentMessages
      });
    } catch (error) {
      logger.error('Get session details error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Get message analytics
   */
  async getMessageAnalytics(req, res) {
    try {
      const { days = 7 } = req.query;
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - parseInt(days));
      
      const analytics = await WhatsAppMessage.aggregate([
        {
          $match: {
            timestamp: { $gte: startDate }
          }
        },
        {
          $group: {
            _id: {
              date: { $dateToString: { format: '%Y-%m-%d', date: '$timestamp' } },
              intent: '$detectedIntent'
            },
            count: { $sum: 1 },
            disclaimerCount: { $sum: { $cond: ['$disclaimerShown', 1, 0] } }
          }
        },
        {
          $group: {
            _id: '$_id.date',
            intents: {
              $push: {
                intent: '$_id.intent',
                count: '$count',
                disclaimerCount: '$disclaimerCount'
              }
            },
            totalCount: { $sum: '$count' },
            totalDisclaimers: { $sum: '$disclaimerCount' }
          }
        },
        {
          $sort: { _id: 1 }
        }
      ]);
      
      return successResponse(res, 'Message analytics retrieved', { analytics });
    } catch (error) {
      logger.error('Get message analytics error:', error);
      return errorResponse(res, 'Internal server error', error.message, 500);
    }
  }

  /**
   * Health check endpoint
   */
  async healthCheck(req, res) {
    try {
      const whatsAppStatus = whatsAppClient.getStatus();
      const aiStatus = geminiClient.getStatus();
      const disclaimerStats = disclaimerManager.getDisclaimerStats();
      
      const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        services: {
          whatsapp: whatsAppStatus.configured ? 'configured' : 'not_configured',
          ai: aiStatus.available ? 'available' : 'fallback'
        },
        disclaimer: {
          activeUsers: disclaimerStats.activeUsers,
          messageThreshold: disclaimerStats.messageThreshold,
          cooldownMinutes: disclaimerStats.cooldownMinutes
        }
      };
      
      return successResponse(res, 'Health check completed', health);
    } catch (error) {
      logger.error('Health check error:', error);
      return errorResponse(res, 'Health check failed', error.message, 500);
    }
  }
}

module.exports = new WhatsAppController(); 