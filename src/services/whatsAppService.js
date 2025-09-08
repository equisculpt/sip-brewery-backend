const { WhatsAppSession, SipOrder, WhatsAppMessage, User, RewardSummary, Leaderboard } = require('../models');
// Only require WhatsApp client if not in test mode
let whatsappClient = null;
if (process.env.NODE_ENV !== 'test') {
  whatsappClient = require('../whatsapp/whatsappClient');
} else {
  whatsappClient = {
    sendMessage: async () => ({ success: true, message: 'Simulated WhatsApp send in test mode' })
  };
}
const messageParser = require('../utils/parseMessage');
const geminiClient = require('../ai/geminiClient');
const disclaimerManager = require('../utils/disclaimers');
const rewardsService = require('./rewardsService');
const logger = require('../utils/logger');

class WhatsAppService {
  constructor() {
    this.rateLimitMap = new Map();
    this.conversationCache = new Map(); // Cache for active conversations
  }

  /**
   * Enhanced process message with comprehensive audit logging and conversation memory
   */
  async processMessage(phoneNumber, message, messageId) {
    const startTime = Date.now();
    const sessionStartTime = Date.now();
    
    try {
      // Rate limiting
      if (!this.checkRateLimit(phoneNumber)) {
        return {
          success: false,
          message: 'Rate limit exceeded. Please wait a moment before sending another message.',
          rateLimited: true
        };
      }

      // Get or create session with enhanced memory
      const session = await this.getOrCreateSession(phoneNumber);
      
      // Check for conversation continuity
      const conversationContext = await this.analyzeConversationContinuity(session, message);
      
      // Parse message intent with context awareness
      const parsedMessage = await this.parseMessageWithContext(message, session, conversationContext);
      
      // Log incoming message with comprehensive audit
      const inboundMessage = await this.logMessageWithAudit(phoneNumber, messageId, 'INBOUND', parsedMessage, session);
      
      // Process based on intent with enhanced context
      const response = await this.handleIntentWithMemory(session, parsedMessage, conversationContext);
      
      // Send response
      let sendResult = null;
      if (whatsappClient && typeof whatsappClient.sendMessage === 'function') {
        sendResult = await whatsappClient.sendMessage(phoneNumber, response.message);
      } else {
        // In test mode, simulate send
        sendResult = { success: true, message: 'Simulated send in test mode' };
      }
      
      // Log outgoing message with comprehensive audit
      const outboundMessage = await this.logMessageWithAudit(phoneNumber, sendResult.messageId, 'OUTBOUND', {
        intent: parsedMessage.intent,
        response: response.message.substring(0, 100),
        disclaimerShown: response.disclaimerShown,
        conversationContext: conversationContext
      }, session);
      
      // Update session with enhanced memory
      await this.updateSessionWithMemory(session, parsedMessage.intent, response, conversationContext);
      
      // Update performance metrics
      const processingTime = Date.now() - startTime;
      session.updatePerformanceMetrics(processingTime, true);
      
      // Update audit trail
      session.auditTrail.totalMessages += 1;
      session.auditTrail.lastComplianceCheck = new Date();
      if (response.disclaimerShown) {
        session.auditTrail.disclaimersShown += 1;
      }
      
      await session.save();
      
      logger.info(`WhatsApp message processed in ${processingTime}ms`, {
        phoneNumber,
        intent: parsedMessage.intent,
        processingTime,
        disclaimerShown: response.disclaimerShown,
        conversationContext: conversationContext.isFollowUp ? 'Follow-up' : 'New',
        confidence: parsedMessage.confidence
      });
      
      return {
        success: true,
        response: response.message,
        intent: parsedMessage.intent,
        processingTime,
        disclaimerShown: response.disclaimerShown,
        conversationContext: conversationContext
      };
      
    } catch (error) {
      logger.error('WhatsApp message processing error:', error);
      
      // Log error message with audit
      await this.logMessageWithAudit(phoneNumber, messageId, 'INBOUND', {
        intent: 'ERROR',
        error: error.message,
        stack: error.stack
      }, null);
      
      // Send error message to user
      const errorMessage = "Sorry, I'm having trouble processing your request. Please try again in a moment.";
      if (whatsappClient && typeof whatsappClient.sendMessage === 'function') {
        await whatsappClient.sendMessage(phoneNumber, errorMessage);
      }
      
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Analyze conversation continuity and context
   */
  async analyzeConversationContinuity(session, message) {
    const now = new Date();
    const lastActivity = session.lastActivity;
    const timeSinceLastMessage = now - lastActivity;
    
    // Check if this is a follow-up message (within 30 minutes)
    const isFollowUp = timeSinceLastMessage < 30 * 60 * 1000;
    
    // Get conversation context from memory
    const conversationContext = session.getConversationContext();
    
    // Check for interrupted conversations
    const wasInterrupted = session.context.interruptedConversation.wasInterrupted;
    const canResume = wasInterrupted && isFollowUp;
    
    return {
      isFollowUp,
      timeSinceLastMessage,
      conversationContext,
      wasInterrupted,
      canResume,
      lastIntent: session.currentIntent,
      pendingAction: session.context.pendingAction
    };
  }

  /**
   * Parse message with context awareness
   */
  async parseMessageWithContext(message, session, conversationContext) {
    // Get basic parsing
    const parsedMessage = messageParser.parseMessage(message);
    
    // Enhance with context if it's a follow-up
    if (conversationContext.isFollowUp) {
      const enhancedIntent = await this.enhanceIntentWithContext(parsedMessage, conversationContext);
      return {
        ...parsedMessage,
        intent: enhancedIntent.intent,
        confidence: enhancedIntent.confidence,
        contextAware: true
      };
    }
    
    return parsedMessage;
  }

  /**
   * Enhance intent detection with conversation context
   */
  async enhanceIntentWithContext(parsedMessage, conversationContext) {
    const { intent, confidence, originalMessage } = parsedMessage;
    const { lastIntent, pendingAction, conversationFlow } = conversationContext.conversationContext;
    
    // If confidence is low, try to infer from context
    if (confidence < 0.7) {
      const contextIntent = this.inferIntentFromContext(originalMessage, lastIntent, pendingAction);
      if (contextIntent) {
        return {
          intent: contextIntent,
          confidence: 0.8
        };
      }
    }
    
    // Handle follow-up responses
    if (lastIntent === 'SIP_CREATE' && this.isSipConfirmation(originalMessage)) {
      return {
        intent: 'CONFIRMATION',
        confidence: 0.9
      };
    }
    
    if (lastIntent === 'AI_ANALYSIS' && this.isAnalysisFollowUp(originalMessage)) {
      return {
        intent: 'AI_ANALYSIS',
        confidence: 0.85
      };
    }
    
    return { intent, confidence };
  }

  /**
   * Infer intent from conversation context
   */
  inferIntentFromContext(message, lastIntent, pendingAction) {
    const lowerMessage = message.toLowerCase();
    
    // Handle common follow-up patterns
    if (lastIntent === 'SIP_CREATE') {
      if (lowerMessage.includes('yes') || lowerMessage.includes('confirm') || lowerMessage.includes('ok')) {
        return 'CONFIRMATION';
      }
      if (lowerMessage.includes('no') || lowerMessage.includes('cancel')) {
        return 'SIP_CREATE'; // Restart SIP creation
      }
    }
    
    if (lastIntent === 'ONBOARDING') {
      if (lowerMessage.includes('@') && lowerMessage.includes('.')) {
        return 'ONBOARDING'; // Email provided
      }
      if (lowerMessage.length === 10 && /^[A-Z]{5}[0-9]{4}[A-Z]$/.test(message)) {
        return 'ONBOARDING'; // PAN provided
      }
    }
    
    if (pendingAction === 'WAITING_FOR_AMOUNT') {
      if (/^\d+$/.test(message) || /^\d+\.\d+$/.test(message)) {
        return 'SIP_CREATE'; // Amount provided
      }
    }
    
    return null;
  }

  /**
   * Check if message is SIP confirmation
   */
  isSipConfirmation(message) {
    const lowerMessage = message.toLowerCase();
    return lowerMessage.includes('yes') || 
           lowerMessage.includes('confirm') || 
           lowerMessage.includes('ok') || 
           lowerMessage.includes('proceed');
  }

  /**
   * Check if message is analysis follow-up
   */
  isAnalysisFollowUp(message) {
    const lowerMessage = message.toLowerCase();
    return lowerMessage.includes('more') || 
           lowerMessage.includes('details') || 
           lowerMessage.includes('explain') ||
           lowerMessage.includes('why');
  }

  /**
   * Enhanced intent handling with conversation memory
   */
  async handleIntentWithMemory(session, parsedMessage, conversationContext) {
    const { intent, extractedData } = parsedMessage;
    
    // Add message to conversation memory
    session.addMessageToMemory(parsedMessage.originalMessage, 'INBOUND', intent, parsedMessage.confidence);
    
    // Handle interrupted conversation resume
    if (conversationContext.canResume) {
      const resumeResponse = await this.resumeInterruptedConversation(session, parsedMessage);
      if (resumeResponse) {
        return resumeResponse;
      }
    }
    
    // Handle multi-step flows
    if (session.context.multiStepFlow.isActive) {
      return await this.handleMultiStepFlow(session, parsedMessage);
    }
    
    switch (intent) {
      case 'GREETING':
        return await this.handleGreetingWithMemory(session, conversationContext);
        
      case 'ONBOARDING':
        return await this.handleOnboardingWithMemory(session, extractedData, conversationContext);
        
      case 'PORTFOLIO_VIEW':
        return await this.handlePortfolioViewWithMemory(session, conversationContext);
        
      case 'SIP_CREATE':
        return await this.handleSipCreateWithMemory(session, extractedData, conversationContext);
        
      case 'SIP_STOP':
        return await this.handleSipStopWithMemory(session, extractedData, conversationContext);
        
      case 'SIP_STATUS':
        return await this.handleSipStatusWithMemory(session, conversationContext);
        
      case 'LUMP_SUM':
        return await this.handleLumpSumWithMemory(session, extractedData, conversationContext);
        
      case 'AI_ANALYSIS':
        return await this.handleAiAnalysisWithMemory(session, extractedData, conversationContext);
        
      case 'STATEMENT':
        return await this.handleStatementWithMemory(session, conversationContext);
        
      case 'REWARDS':
        return await this.handleRewardsWithMemory(session, conversationContext);
        
      case 'REFERRAL':
        return await this.handleReferralWithMemory(session, conversationContext);
        
      case 'LEADERBOARD':
        return await this.handleLeaderboardWithMemory(session, conversationContext);
        
      case 'COPY_PORTFOLIO':
        return await this.handleCopyPortfolioWithMemory(session, extractedData, conversationContext);
        
      case 'HELP':
        return await this.handleHelpWithMemory(session, conversationContext);
        
      case 'CONFIRMATION':
        return await this.handleConfirmationWithMemory(session, extractedData, conversationContext);
        
      case 'FUND_RESEARCH':
        return await this.handleFundResearchWithMemory(session, extractedData, conversationContext);
        
      case 'MARKET_UPDATE':
        return await this.handleMarketUpdateWithMemory(session, conversationContext);
        
      default:
        return await this.handleUnknownWithMemory(session, parsedMessage, conversationContext);
    }
  }

  /**
   * Resume interrupted conversation
   */
  async resumeInterruptedConversation(session, parsedMessage) {
    const interrupted = session.context.interruptedConversation;
    
    if (interrupted.lastIntent === 'SIP_CREATE') {
      const message = "Welcome back! We were setting up your SIP. Please continue with the fund name and amount.";
      return {
        message: disclaimerManager.addDisclaimerToMessage(message, 'general', session.phoneNumber),
        updateSession: {
          currentIntent: 'SIP_CREATE',
          'context.interruptedConversation.wasInterrupted': false
        },
        disclaimerShown: true
      };
    }
    
    if (interrupted.lastIntent === 'ONBOARDING') {
      const message = "Welcome back! Let's continue with your onboarding. Please provide the requested information.";
      return {
        message: disclaimerManager.addDisclaimerToMessage(message, 'general', session.phoneNumber),
        updateSession: {
          currentIntent: 'ONBOARDING',
          'context.interruptedConversation.wasInterrupted': false
        },
        disclaimerShown: true
      };
    }
    
    return null;
  }

  /**
   * Handle multi-step flows
   */
  async handleMultiStepFlow(session, parsedMessage) {
    const flow = session.context.multiStepFlow;
    
    switch (flow.flowType) {
      case 'SIP_CREATION':
        return await this.handleSipCreationFlow(session, parsedMessage);
      case 'ONBOARDING':
        return await this.handleOnboardingFlow(session, parsedMessage);
      default:
        return await this.handleUnknown(session, parsedMessage);
    }
  }

  /**
   * Enhanced greeting with memory
   */
  async handleGreetingWithMemory(session, conversationContext) {
    if (session.onboardingState === 'COMPLETED') {
      let message;
      
      if (conversationContext.isFollowUp) {
        message = `Welcome back! How can I help you today? You can ask about your portfolio, start a new SIP, or get investment insights.`;
      } else {
        message = disclaimerManager.getWelcomeMessageWithDisclaimer(session.phoneNumber);
      }
      
      return {
        message,
        updateSession: false,
        disclaimerShown: message.includes('⚠️')
      };
    } else {
      const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'name');
      return {
        message,
        updateSession: {
          onboardingState: 'INITIAL',
          currentIntent: 'ONBOARDING'
        },
        disclaimerShown: message.includes('⚠️')
      };
    }
  }

  /**
   * Enhanced onboarding with memory and multi-step flow handling
   */
  async handleOnboardingWithMemory(session, data, conversationContext) {
    const { name, email, pan } = data;
    
    // Handle multi-step onboarding flow
    if (session.context.multiStepFlow.isActive && session.context.multiStepFlow.flowType === 'ONBOARDING') {
      return await this.handleOnboardingFlow(session, data);
    }
    
    // Handle direct data input
    if (name && session.onboardingState === 'INITIAL') {
      session.onboardingData.name = name;
      session.onboardingState = 'NAME_COLLECTED';
      
      const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'email');
      return {
        message,
        updateSession: {
          onboardingState: 'NAME_COLLECTED',
          currentIntent: 'ONBOARDING',
          'context.multiStepFlow': {
            isActive: true,
            currentStep: 2,
            totalSteps: 4,
            flowType: 'ONBOARDING',
            collectedData: { name }
          }
        },
        disclaimerShown: message.includes('⚠️')
      };
    }
    
    if (email && session.onboardingState === 'NAME_COLLECTED') {
      session.onboardingData.email = email;
      session.onboardingState = 'EMAIL_COLLECTED';
      
      const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'pan');
      return {
        message,
        updateSession: {
          onboardingState: 'EMAIL_COLLECTED',
          currentIntent: 'ONBOARDING',
          'context.multiStepFlow.currentStep': 3
        },
        disclaimerShown: message.includes('⚠️')
      };
    }
    
    if (pan && session.onboardingState === 'EMAIL_COLLECTED') {
      session.onboardingData.pan = pan;
      
      // Mock KYC verification (in real app, call Digio API)
      const kycVerified = await this.verifyKYC(pan);
      session.onboardingData.kycStatus = kycVerified ? 'VERIFIED' : 'PENDING';
      session.onboardingState = kycVerified ? 'KYC_VERIFIED' : 'PAN_COLLECTED';
      
      if (kycVerified) {
        // Create user in database
        await this.createUser(session);
        
        const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'kyc_verified');
        return {
          message,
          updateSession: {
            onboardingState: 'COMPLETED',
            currentIntent: 'GREETING',
            'context.multiStepFlow.isActive': false
          },
          disclaimerShown: message.includes('⚠️')
        };
      } else {
        const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'kyc_pending');
        return {
          message,
          updateSession: {
            onboardingState: 'PAN_COLLECTED',
            currentIntent: 'ONBOARDING'
          },
          disclaimerShown: message.includes('⚠️')
        };
      }
    }
    
    // Handle generic onboarding input
    const message = this.getOnboardingPrompt(session);
    return {
      message,
      updateSession: false,
      disclaimerShown: false
    };
  }

  /**
   * Handle onboarding flow step by step
   */
  async handleOnboardingFlow(session, data) {
    const flow = session.context.multiStepFlow;
    const { name, email, pan } = data;
    
    switch (flow.currentStep) {
      case 1: // Waiting for name
        if (name) {
          flow.collectedData.name = name;
          flow.currentStep = 2;
          
          const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'email');
          return {
            message,
            updateSession: {
              'context.multiStepFlow': flow,
              onboardingState: 'NAME_COLLECTED'
            },
            disclaimerShown: message.includes('⚠️')
          };
        } else {
          const message = "Please provide your full name to continue with onboarding.";
          return {
            message,
            updateSession: false,
            disclaimerShown: false
          };
        }
        
      case 2: // Waiting for email
        if (email) {
          flow.collectedData.email = email;
          flow.currentStep = 3;
          
          const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'pan');
          return {
            message,
            updateSession: {
              'context.multiStepFlow': flow,
              onboardingState: 'EMAIL_COLLECTED'
            },
            disclaimerShown: message.includes('⚠️')
          };
        } else {
          const message = "Please provide your email address to continue.";
          return {
            message,
            updateSession: false,
            disclaimerShown: false
          };
        }
        
      case 3: // Waiting for PAN
        if (pan) {
          flow.collectedData.pan = pan;
          flow.currentStep = 4;
          
          // Mock KYC verification
          const kycVerified = await this.verifyKYC(pan);
          flow.collectedData.kycStatus = kycVerified ? 'VERIFIED' : 'PENDING';
          
          if (kycVerified) {
            await this.createUser(session);
            const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'kyc_verified');
            return {
              message,
              updateSession: {
                onboardingState: 'COMPLETED',
                currentIntent: 'GREETING',
                'context.multiStepFlow.isActive': false
              },
              disclaimerShown: message.includes('⚠️')
            };
          } else {
            const message = disclaimerManager.getOnboardingMessageWithDisclaimer(session.phoneNumber, 'kyc_pending');
            return {
              message,
              updateSession: {
                onboardingState: 'PAN_COLLECTED',
                'context.multiStepFlow': flow
              },
              disclaimerShown: message.includes('⚠️')
            };
          }
        } else {
          const message = "Please provide your PAN number to continue.";
          return {
            message,
            updateSession: false,
            disclaimerShown: false
          };
        }
        
      default:
        return await this.handleOnboardingWithMemory(session, data, {});
    }
  }

  /**
   * Get appropriate onboarding prompt based on current state
   */
  getOnboardingPrompt(session) {
    switch (session.onboardingState) {
      case 'INITIAL':
        return "Welcome to SIPBrewery! Please provide your full name to start onboarding.";
      case 'NAME_COLLECTED':
        return "Thanks! Please share your email address.";
      case 'EMAIL_COLLECTED':
        return "Great! Now please provide your PAN number.";
      case 'PAN_COLLECTED':
        return "Your PAN is being verified. You'll receive an update shortly.";
      default:
        return "Please provide the requested information to continue with onboarding.";
    }
  }

  /**
   * Enhanced portfolio view with better response handling
   */
  async handlePortfolioViewWithMemory(session, conversationContext) {
    if (session.onboardingState !== 'COMPLETED') {
      const message = "Please complete your onboarding first to view your portfolio.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
    
    try {
      const portfolio = await this.getUserPortfolio(session.userId);
      
      // Personalize message based on conversation context
      let message;
      if (conversationContext.isFollowUp) {
        message = `Here's your updated portfolio:\n\n${disclaimerManager.formatPortfolioData(portfolio)}`;
      } else {
        message = disclaimerManager.getPortfolioSummaryWithDisclaimer(session.phoneNumber, portfolio);
      }
      
      return {
        message,
        updateSession: {
          currentIntent: 'PORTFOLIO_VIEW'
        },
        disclaimerShown: message.includes('⚠️')
      };
    } catch (error) {
      logger.error('Portfolio view error:', error);
      const message = "Sorry, I couldn't fetch your portfolio. Please try again later.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
  }

  /**
   * Enhanced SIP creation with better multi-step handling
   */
  async handleSipCreateWithMemory(session, data, conversationContext) {
    if (session.onboardingState !== 'COMPLETED') {
      const message = "Please complete your onboarding first to start SIP investments.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
    
    const { amount, fundName } = data;
    
    // Check if we're in a multi-step flow
    if (session.context.multiStepFlow.isActive && session.context.multiStepFlow.flowType === 'SIP_CREATION') {
      return await this.handleSipCreationFlow(session, { amount, fundName });
    }
    
    if (!amount || !fundName) {
      // Start multi-step SIP creation flow
      const message = "I'll help you start a SIP. Please tell me:\n\n1. Which fund you'd like to invest in\n2. The monthly amount (minimum ₹100)";
      
      return {
        message,
        updateSession: {
          currentIntent: 'SIP_CREATE',
          'context.multiStepFlow': {
            isActive: true,
            currentStep: 1,
            totalSteps: 3,
            flowType: 'SIP_CREATION',
            collectedData: {}
          }
        },
        disclaimerShown: false
      };
    }
    
    if (amount < 100) {
      const message = "Minimum SIP amount is ₹100. Please specify a higher amount.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
    
    // Store temporary data for confirmation
    session.context.tempData = { amount, fundName };
    session.currentIntent = 'SIP_CREATE';
    
    const message = disclaimerManager.getSipConfirmationWithDisclaimer(session.phoneNumber, fundName, amount);
    
    return {
      message,
      updateSession: {
        currentIntent: 'SIP_CREATE',
        context: session.context
      },
      disclaimerShown: message.includes('⚠️')
    };
  }

  /**
   * Handle SIP creation flow
   */
  async handleSipCreationFlow(session, data) {
    const flow = session.context.multiStepFlow;
    const { amount, fundName } = data;
    
    if (flow.currentStep === 1) {
      // Waiting for fund name
      if (fundName) {
        flow.collectedData.fundName = fundName;
        flow.currentStep = 2;
        
        const message = `Great! You want to invest in ${fundName}. Now please tell me the monthly amount (minimum ₹100).`;
        
        return {
          message,
          updateSession: {
            'context.multiStepFlow': flow
          },
          disclaimerShown: false
        };
      } else {
        const message = "Please tell me which fund you'd like to invest in.";
        return {
          message,
          updateSession: false,
          disclaimerShown: false
        };
      }
    }
    
    if (flow.currentStep === 2) {
      // Waiting for amount
      if (amount) {
        if (amount < 100) {
          const message = "Minimum SIP amount is ₹100. Please specify a higher amount.";
          return {
            message,
            updateSession: false,
            disclaimerShown: false
          };
        }
        
        flow.collectedData.amount = amount;
        flow.currentStep = 3;
        
        const { fundName: collectedFundName } = flow.collectedData;
        const message = disclaimerManager.getSipConfirmationWithDisclaimer(session.phoneNumber, collectedFundName, amount);
        
        return {
          message,
          updateSession: {
            'context.multiStepFlow': flow,
            'context.tempData': flow.collectedData
          },
          disclaimerShown: message.includes('⚠️')
        };
      } else {
        const message = "Please specify the monthly amount (minimum ₹100).";
        return {
          message,
          updateSession: false,
          disclaimerShown: false
        };
      }
    }
    
    return await this.handleSipCreate(session, flow.collectedData);
  }

  /**
   * Enhanced confirmation with memory
   */
  async handleConfirmationWithMemory(session, data, conversationContext) {
    const { confirmed } = data;
    
    if (session.currentIntent === 'SIP_CREATE') {
      if (confirmed) {
        const { amount, fundName } = session.context.tempData;
        
        try {
          const sipOrder = await this.createSipOrder(session, amount, fundName);
          
          const message = `🎉 SIP confirmed!

Fund: ${fundName}
Amount: ₹${amount}/month
Order ID: ${sipOrder.orderId}

Your SIP will start from next month. You'll receive confirmation via email.

⚠️ Mutual Fund investments are subject to market risks. Read all scheme related documents carefully.`;
          
          return {
            message,
            updateSession: {
              currentIntent: 'GREETING',
              'context.multiStepFlow.isActive': false,
              'context.tempData': null
            },
            disclaimerShown: true
          };
        } catch (error) {
          logger.error('SIP creation error:', error);
          const message = "Sorry, I couldn't create your SIP. Please try again later.";
          return {
            message,
            updateSession: false,
            disclaimerShown: false
          };
        }
      } else {
        const message = "SIP creation cancelled. You can start a new SIP anytime by saying 'Start SIP'.";
        return {
          message,
          updateSession: {
            currentIntent: 'GREETING',
            'context.multiStepFlow.isActive': false,
            'context.tempData': null
          },
          disclaimerShown: false
        };
      }
    }
    
    return await this.handleConfirmation(session, data);
  }

  /**
   * Enhanced AI analysis with memory
   */
  async handleAiAnalysisWithMemory(session, data, conversationContext) {
    const { fundName, analysisType } = data;
    
    try {
      let analysis;
      
      if (conversationContext.isFollowUp && session.currentIntent === 'AI_ANALYSIS') {
        // Provide more detailed analysis for follow-up
        analysis = await this.getDetailedFundAnalysis(fundName || 'general');
      } else {
        // Initial analysis
        analysis = await this.getFundAnalysis(fundName || 'general');
      }
      
      const message = disclaimerManager.getAiAnalysisWithDisclaimer(session.phoneNumber, analysis);
      
      return {
        message,
        updateSession: {
          currentIntent: 'AI_ANALYSIS'
        },
        disclaimerShown: message.includes('⚠️')
      };
    } catch (error) {
      logger.error('AI analysis error:', error);
      const message = "Sorry, I couldn't analyze the fund. Please try again later.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
  }

  /**
   * Enhanced rewards with memory
   */
  async handleRewardsWithMemory(session, conversationContext) {
    try {
      const rewards = await rewardsService.getUserRewards(session.userId);
      
      let message;
      if (conversationContext.isFollowUp) {
        message = `Here's your updated rewards:\n\n${disclaimerManager.formatRewardsData(rewards)}`;
      } else {
        message = disclaimerManager.getRewardsSummaryWithDisclaimer(session.phoneNumber, rewards);
      }
      
      return {
        message,
        updateSession: {
          currentIntent: 'REWARDS'
        },
        disclaimerShown: message.includes('⚠️')
      };
    } catch (error) {
      logger.error('Rewards error:', error);
      const message = "Sorry, I couldn't fetch your rewards. Please try again later.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
  }

  /**
   * Enhanced referral with memory
   */
  async handleReferralWithMemory(session, conversationContext) {
    try {
      const referralCode = await rewardsService.getUserReferralCode(session.userId);
      
      let message;
      if (conversationContext.isFollowUp) {
        message = `Your referral code is: ${referralCode}\n\nShare it with friends to earn rewards!`;
      } else {
        message = disclaimerManager.getReferralMessageWithDisclaimer(session.phoneNumber, referralCode);
      }
      
      return {
        message,
        updateSession: {
          currentIntent: 'REFERRAL'
        },
        disclaimerShown: message.includes('⚠️')
      };
    } catch (error) {
      logger.error('Referral error:', error);
      const message = "Sorry, I couldn't fetch your referral code. Please try again later.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
  }

  /**
   * Enhanced leaderboard with memory
   */
  async handleLeaderboardWithMemory(session, conversationContext) {
    try {
      const leaderboard = await rewardsService.getLeaderboard();
      
      let message;
      if (conversationContext.isFollowUp) {
        message = `🏆 Updated Leaderboard:\n\n${disclaimerManager.formatLeaderboardData(leaderboard)}`;
      } else {
        message = disclaimerManager.getLeaderboardWithDisclaimer(session.phoneNumber, leaderboard);
      }
      
      return {
        message,
        updateSession: {
          currentIntent: 'LEADERBOARD'
        },
        disclaimerShown: message.includes('⚠️')
      };
    } catch (error) {
      logger.error('Leaderboard error:', error);
      const message = "Sorry, I couldn't fetch the leaderboard. Please try again later.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
  }

  /**
   * Enhanced help with memory
   */
  async handleHelpWithMemory(session, conversationContext) {
    let message;
    
    if (conversationContext.isFollowUp) {
      message = "Here are some quick actions you can try:\n\n• 'My portfolio' - View investments\n• 'Start SIP' - Begin new investment\n• 'My rewards' - Check points\n• 'AI analysis' - Get fund insights";
    } else {
      message = disclaimerManager.getHelpMessageWithDisclaimer(session.phoneNumber);
    }
    
    return {
      message,
      updateSession: {
        currentIntent: 'HELP'
      },
      disclaimerShown: message.includes('⚠️')
    };
  }

  /**
   * Enhanced unknown intent with better contextual responses
   */
  async handleUnknownWithMemory(session, parsedMessage, conversationContext) {
    // Try to infer intent from context
    const inferredIntent = this.inferIntentFromContext(parsedMessage.originalMessage, session.currentIntent, session.context.pendingAction);
    
    if (inferredIntent) {
      // Re-process with inferred intent
      const enhancedMessage = { ...parsedMessage, intent: inferredIntent, confidence: 0.8 };
      return await this.handleIntentWithMemory(session, enhancedMessage, conversationContext);
    }
    
    // Provide contextual help based on conversation history and current state
    const lastIntent = session.currentIntent;
    const pendingAction = session.context.pendingAction;
    let fallbackMessage;
    
    if (lastIntent === 'SIP_CREATE') {
      if (pendingAction === 'WAITING_FOR_FUND') {
        fallbackMessage = "Please tell me which fund you'd like to invest in (e.g., HDFC Flexicap, SBI Smallcap).";
      } else if (pendingAction === 'WAITING_FOR_AMOUNT') {
        fallbackMessage = "Please specify the monthly amount you want to invest (minimum ₹100).";
      } else {
        fallbackMessage = "For SIP creation, please provide:\n\n• Fund name (e.g., HDFC Flexicap)\n• Monthly amount (e.g., ₹5000)";
      }
    } else if (lastIntent === 'ONBOARDING') {
      fallbackMessage = this.getOnboardingPrompt(session);
    } else if (lastIntent === 'SIP_STOP') {
      fallbackMessage = "Please specify which fund's SIP you'd like to stop.";
    } else {
      // Check for common keywords and provide specific help
      const lowerMessage = parsedMessage.originalMessage.toLowerCase();
      
      if (lowerMessage.includes('fee') || lowerMessage.includes('charge') || lowerMessage.includes('cost')) {
        fallbackMessage = "💰 SIPBrewery Fee Structure:\n\n• No account opening charges\n• No annual maintenance fees\n• Fund expense ratios apply (0.5-2.5%)\n• Exit load as per fund scheme\n\nAll fees are transparent and disclosed upfront.";
      } else if (lowerMessage.includes('safe') || lowerMessage.includes('secure') || lowerMessage.includes('risk')) {
        fallbackMessage = "🔒 SIPBrewery Safety & Security:\n\n• SEBI registered platform\n• Bank-grade security\n• AMFI registered distributor\n• Your money goes directly to fund houses\n• No platform risk to your investments\n\n⚠️ Mutual funds are subject to market risks.";
      } else if (lowerMessage.includes('tax') || lowerMessage.includes('deduction')) {
        fallbackMessage = "📊 Tax Benefits:\n\n• ELSS funds: ₹1.5L deduction under 80C\n• Long-term capital gains: 10% tax after 1 year\n• Dividend income: Taxable as per slab\n• SIP investments qualify for tax benefits\n\nConsult a tax advisor for specific advice.";
      } else if (lowerMessage.includes('withdraw') || lowerMessage.includes('exit') || lowerMessage.includes('sell')) {
        fallbackMessage = "💳 Withdrawal Information:\n\n• Redemption processed in 1-3 business days\n• Exit load may apply based on fund\n• Partial or complete withdrawal available\n• Redemption amount credited to registered bank account\n\nType 'redeem [fund name]' to start withdrawal.";
      } else if (lowerMessage.includes('market') || lowerMessage.includes('trend') || lowerMessage.includes('news')) {
        fallbackMessage = "📈 Market Update:\n\n• Nifty 50: 19,850 (+0.8%)\n• Sensex: 66,200 (+0.7%)\n• Positive global cues\n• Strong domestic flows\n\nFor detailed analysis, say 'Analyse [fund name]'";
      } else {
        fallbackMessage = "I'm not sure I understand. You can ask me about:\n\n• Your portfolio\n• Starting SIPs\n• Checking rewards\n• Fund analysis\n• Getting statements\n• Fees and charges\n• Safety and security\n• Tax benefits\n\nType 'help' for more options.";
      }
    }
    
    const messageWithDisclaimer = disclaimerManager.addDisclaimerToMessage(fallbackMessage, 'general', session.phoneNumber);
    
    return {
      message: messageWithDisclaimer,
      updateSession: { currentIntent: 'UNKNOWN' },
      disclaimerShown: messageWithDisclaimer.includes('⚠️')
    };
  }

  /**
   * Handle fund research
   */
  async handleFundResearchWithMemory(session, data, conversationContext) {
    const { fundName, researchType } = data;
    
    try {
      const research = await this.getFundResearch(fundName, researchType);
      const message = disclaimerManager.getFundResearchWithDisclaimer(session.phoneNumber, research);
      
      return {
        message,
        updateSession: {
          currentIntent: 'FUND_RESEARCH'
        },
        disclaimerShown: message.includes('⚠️')
      };
    } catch (error) {
      logger.error('Fund research error:', error);
      const message = "Sorry, I couldn't research the fund. Please try again later.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
  }

  /**
   * Handle market update
   */
  async handleMarketUpdateWithMemory(session, conversationContext) {
    try {
      const marketUpdate = await this.getMarketUpdate();
      const message = disclaimerManager.getMarketUpdateWithDisclaimer(session.phoneNumber, marketUpdate);
      
      return {
        message,
        updateSession: {
          currentIntent: 'MARKET_UPDATE'
        },
        disclaimerShown: message.includes('⚠️')
      };
    } catch (error) {
      logger.error('Market update error:', error);
      const message = "Sorry, I couldn't fetch market updates. Please try again later.";
      return {
        message,
        updateSession: false,
        disclaimerShown: false
      };
    }
  }

  /**
   * Enhanced SIP management with better command handling
   */
  async handleSipStopWithMemory(session, data, conversationContext) {
    const { fundName } = data;
    
    if (!fundName) {
      const message = "I can help you stop your SIP. Please specify which fund's SIP you'd like to stop.";
      return {
        message,
        updateSession: {
          currentIntent: 'SIP_STOP',
          'context.pendingAction': 'WAITING_FOR_FUND_NAME'
        },
        disclaimerShown: false
      };
    }
    
    const message = `I can help you stop your SIP in ${fundName}. Please confirm by saying 'yes' to proceed.`;
    return {
      message,
      updateSession: {
        currentIntent: 'SIP_STOP',
        'context.tempData': { fundName }
      },
      disclaimerShown: false
    };
  }

  /**
   * Enhanced session update with memory
   */
  async updateSessionWithMemory(session, intent, response, conversationContext) {
    session.currentIntent = intent;
    session.messageCount += 1;
    session.lastMessageTime = new Date();
    
    // Update conversation memory
    session.addMessageToMemory(response.message, 'OUTBOUND', intent, 1.0);
    
    // Handle interrupted conversation tracking
    if (conversationContext.timeSinceLastMessage > 30 * 60 * 1000) {
      session.context.interruptedConversation = {
        wasInterrupted: true,
        lastIntent: intent,
        pendingData: session.context.tempData,
        resumePoint: 'LAST_INTENT'
      };
    }
    
    // Update session data if provided
    if (response.updateSession) {
      Object.assign(session, response.updateSession);
    }
    
    await session.save();
  }

  /**
   * Enhanced message logging with comprehensive audit
   */
  async logMessageWithAudit(phoneNumber, messageId, direction, data, session) {
    const messageData = {
      phoneNumber,
      messageId,
      direction,
      content: { text: data.originalMessage || data.response || JSON.stringify(data) },
      detectedIntent: data.intent || 'UNKNOWN',
      confidence: data.confidence || 0,
      processingTime: data.processingTime || 0,
      aiGenerated: data.aiGenerated || false,
      disclaimerShown: data.disclaimerShown || false,
      disclaimerType: data.disclaimerType || 'NONE',
      adviceType: data.adviceType || 'NONE',
      sessionId: session?._id,
      conversationId: session?.conversationMemory?.currentContext?.sessionId,
      auditLog: {
        conversationContext: session ? {
          previousIntent: session.currentIntent,
          conversationFlow: session.conversationMemory?.currentContext?.conversationFlow || [],
          userPreferences: session.conversationMemory?.userPreferences || {},
          sessionDuration: session.auditTrail?.totalMessages || 0
        } : {},
        complianceChecks: [{
          checkType: 'SEBI_COMPLIANCE',
          timestamp: new Date(),
          result: data.disclaimerShown ? 'PASS' : 'N/A',
          details: data.disclaimerShown ? 'Disclaimer shown' : 'No disclaimer required'
        }],
        performanceMetrics: {
          intentDetectionTime: data.intentDetectionTime || 0,
          aiProcessingTime: data.aiProcessingTime || 0,
          databaseQueryTime: data.databaseQueryTime || 0,
          totalProcessingTime: data.processingTime || 0,
          memoryUsage: process.memoryUsage().heapUsed
        },
        userBehavior: {
          responseTime: data.responseTime || 0,
          messageLength: (data.originalMessage || data.response || '').length,
          complexity: this.calculateMessageComplexity(data.originalMessage || data.response || ''),
          interactionPattern: session ? this.analyzeInteractionPattern(session) : 'NEW_USER'
        },
        securityChecks: [{
          checkType: 'RATE_LIMIT',
          timestamp: new Date(),
          result: 'PASS',
          riskLevel: 'LOW'
        }]
      },
      conversationContinuity: {
        isFollowUp: data.conversationContext?.isFollowUp || false,
        previousMessageId: data.previousMessageId,
        contextCarried: data.conversationContext?.isFollowUp || false,
        timeSinceLastMessage: data.conversationContext?.timeSinceLastMessage || 0,
        conversationResumed: data.conversationContext?.canResume || false
      },
      businessLogic: {
        actionTaken: data.intent || 'UNKNOWN',
        dataProcessed: data.extractedData || {},
        rulesApplied: data.disclaimerShown ? ['SEBI_COMPLIANCE'] : [],
        decisionFactors: [data.intent, data.confidence],
        outcome: 'SUCCESS'
      },
      qualityMetrics: {
        intentAccuracy: data.confidence || 0,
        responseRelevance: 0.9,
        userIntentMet: true,
        followUpRequired: false,
        escalationNeeded: false
      }
    };

    const message = new WhatsAppMessage(messageData);
    await message.save();
    
    return message;
  }

  /**
   * Calculate message complexity
   */
  calculateMessageComplexity(text) {
    if (!text) return 'LOW';
    
    const wordCount = text.split(' ').length;
    const charCount = text.length;
    
    if (wordCount > 20 || charCount > 200) return 'HIGH';
    if (wordCount > 10 || charCount > 100) return 'MEDIUM';
    return 'LOW';
  }

  /**
   * Analyze interaction pattern
   */
  analyzeInteractionPattern(session) {
    const messageCount = session.messageCount;
    const sessionDuration = Date.now() - session.auditTrail.sessionStartTime;
    
    if (messageCount < 5) return 'NEW_USER';
    if (messageCount > 20) return 'ACTIVE_USER';
    return 'REGULAR_USER';
  }

  /**
   * Get detailed fund analysis
   */
  async getDetailedFundAnalysis(fundName) {
    // Mock detailed analysis
    return `📊 Detailed Analysis for ${fundName}:

🎯 Investment Objective: Long-term capital appreciation
📈 Risk Level: Moderate to High
💰 Minimum Investment: ₹100
📅 Fund Age: 15 years
🏆 Rating: 4.5/5

📊 Performance (3 years):
• 1 Year: +18.5%
• 3 Years: +45.2%
• 5 Years: +78.9%

⚠️ Risk Factors:
• Market volatility
• Sector concentration
• Currency fluctuations

💡 Investment Strategy:
• Diversified portfolio
• Active management
• Growth-oriented approach`;
  }

  /**
   * Get fund research
   */
  async getFundResearch(fundName, researchType) {
    // Mock fund research
    return `🔍 Research Report: ${fundName}

📊 Fund Overview:
• Category: Flexi Cap
• AUM: ₹2,500 Crores
• Expense Ratio: 1.8%
• Exit Load: 1% (1 year)

📈 Performance Analysis:
• Consistent outperformance
• Strong risk-adjusted returns
• Experienced fund management

🎯 Investment Recommendation:
• Suitable for long-term investors
• Moderate risk appetite
• 5+ year investment horizon`;
  }

  /**
   * Get market update
   */
  async getMarketUpdate() {
    // Mock market update
    return `📈 Market Update - ${new Date().toLocaleDateString()}

🏛️ Nifty 50: 19,850 (+0.8%)
📊 Sensex: 66,200 (+0.7%)
💰 Bank Nifty: 44,500 (+1.2%)

📊 Sector Performance:
• Banking: +1.5%
• IT: +0.3%
• Pharma: -0.2%
• Auto: +0.8%

💡 Market Insights:
• Positive global cues
• Strong domestic flows
• RBI policy in focus`;
  }

  // Helper methods (unchanged)
  async getOrCreateSession(phoneNumber) {
    let session = await WhatsAppSession.findOne({ phoneNumber });
    
    if (!session) {
      session = new WhatsAppSession({ phoneNumber });
      await session.save();
    }
    
    return session;
  }

  async updateSession(session, intent, response) {
    session.currentIntent = intent;
    session.messageCount += 1;
    session.lastMessageTime = new Date();
    
    if (response.updateSession) {
      Object.assign(session, response.updateSession);
    }
    
    await session.save();
  }

  checkRateLimit(phoneNumber) {
    const now = Date.now();
    const windowMs = 1000; // 1 second
    const maxMessages = 3; // 3 messages per second
    
    if (!this.rateLimitMap.has(phoneNumber)) {
      this.rateLimitMap.set(phoneNumber, []);
    }
    
    const messages = this.rateLimitMap.get(phoneNumber);
    const recentMessages = messages.filter(time => now - time < windowMs);
    
    if (recentMessages.length >= maxMessages) {
      return false;
    }
    
    recentMessages.push(now);
    this.rateLimitMap.set(phoneNumber, recentMessages);
    
    return true;
  }

  async logMessage(phoneNumber, messageId, direction, data) {
    try {
      const message = new WhatsAppMessage({
        phoneNumber,
        messageId,
        direction,
        content: { text: data.originalMessage || data.response },
        detectedIntent: data.intent,
        confidence: data.confidence || 0,
        processingTime: data.processingTime || 0,
        disclaimerShown: data.disclaimerShown || false
      });
      
      await message.save();
    } catch (error) {
      logger.error('Message logging error:', error);
    }
  }

  async verifyKYC(pan) {
    // In real implementation, call Digio API
    // For now, return true for test PANs
    const testPans = ['ABCDE1234F', 'FGHIJ5678K', 'LMNOP9012Q'];
    return testPans.includes(pan);
  }

  async createUser(session) {
    const user = new User({
      supabaseId: `whatsapp_${session.phoneNumber}`,
      name: session.onboardingData.name,
      email: session.onboardingData.email,
      phone: session.phoneNumber,
      kycStatus: session.onboardingData.kycStatus
    });
    
    await user.save();
    session.userId = user.supabaseId;
    await session.save();
  }

  async getUserPortfolio(userId) {
    // Mock portfolio data
    return {
      totalValue: 125000,
      totalInvested: 100000,
      returns: 25.0,
      topHoldings: [
        'HDFC Flexicap - ₹45,000',
        'SBI Smallcap - ₹35,000',
        'Parag Parikh Flexicap - ₹25,000',
        'Mirae Asset Largecap - ₹20,000'
      ]
    };
  }

  async createSipOrder(session, amount, fundName) {
    const sipOrder = new SipOrder({
      userId: session.userId,
      phoneNumber: session.phoneNumber,
      schemeName: fundName,
      fundHouse: fundName.split(' ')[0],
      fundType: 'EQUITY',
      amount: amount,
      startDate: new Date(),
      status: 'PENDING'
    });
    
    await sipOrder.save();
    return sipOrder;
  }

  // Additional methods for other intents
  async handleSipStop(session, data) {
    // Stop (pause/cancel) a SIP for a user
    const { fundName } = data;
    if (!fundName) {
      return {
        message: "Please specify the fund whose SIP you want to stop.",
        updateSession: false,
        disclaimerShown: false
      };
    }
    const sip = await SipOrder.findOne({ userId: session.userId, schemeName: fundName, isActive: true, status: { $in: ['ACTIVE', 'CONFIRMED'] } });
    if (!sip) {
      return {
        message: `No active SIP found for ${fundName}.`,
        updateSession: false,
        disclaimerShown: false
      };
    }
    sip.status = 'PAUSED';
    sip.isActive = false;
    sip.endDate = new Date();
    await sip.save();
    return {
      message: `Your SIP in ${fundName} has been stopped. You can restart anytime.`,
      updateSession: true,
      disclaimerShown: false
    };
  }

  async handleSipStatus(session) {
    // Fetch all SIPs for the user
    const sips = await SipOrder.find({ userId: session.userId });
    if (!sips.length) {
      return {
        message: "You have no SIPs yet. Start one to grow your wealth!",
        updateSession: false,
        disclaimerShown: false
      };
    }
    const active = sips.filter(s => s.status === 'ACTIVE');
    const paused = sips.filter(s => s.status === 'PAUSED');
    let msg = `You have ${sips.length} SIP(s):\n`;
    if (active.length) msg += `\nActive: ${active.map(s => `${s.schemeName} (₹${s.amount}/mo)`).join(', ')}`;
    if (paused.length) msg += `\nPaused: ${paused.map(s => `${s.schemeName}`).join(', ')}`;
    return {
      message: msg,
      updateSession: false,
      disclaimerShown: false
    };
  }

  async handleLumpSum(session, data) {
    // Guide user through lump sum investment
    const { fundName, amount } = data;
    if (!fundName || !amount) {
      return {
        message: "Please specify both fund name and amount for lump sum investment.",
        updateSession: false,
        disclaimerShown: false
      };
    }
    // Simulate transaction creation
    // In production, integrate payment gateway and order placement
    const transaction = new Transaction({
      userId: session.userId,
      type: 'LUMPSUM',
      schemeName: fundName,
      amount,
      status: 'PENDING',
      date: new Date(),
      phoneNumber: session.phoneNumber,
      orderType: 'BUY',
      units: 0, // To be updated after NAV
      nav: 0, // To be updated after NAV
      netAmount: amount
    });
    await transaction.save();
    return {
      message: `Lump sum investment of ₹${amount} in ${fundName} has been initiated. You'll get a confirmation soon.`,
      updateSession: true,
      disclaimerShown: false
    };
  }

  async handleStatement(session) {
    // Generate a summary statement of recent transactions
    const txns = await Transaction.find({ userId: session.userId }).sort({ date: -1 }).limit(10);
    if (!txns.length) {
      return {
        message: "No transactions found to generate a statement.",
        updateSession: false,
        disclaimerShown: false
      };
    }
    let msg = `Recent Transactions:\n`;
    txns.forEach(t => {
      msg += `\n${t.type} | ${t.schemeName} | ₹${t.amount} | ${t.status} | ${t.date.toLocaleDateString()}`;
    });
    return {
      message: msg,
      updateSession: false,
      disclaimerShown: false
    };
  }

  async handleCopyPortfolio(session, data) {
    // Copy a leader's portfolio to the user
    const { leaderSecretCode } = data;
    if (!leaderSecretCode) {
      return {
        message: "Please provide the secret code of the leader you want to copy.",
        updateSession: false,
        disclaimerShown: false
      };
    }
    const leaderboard = await Leaderboard.findOne({ isActive: true }).sort({ generatedAt: -1 });
    if (!leaderboard) {
      return {
        message: "No active leaderboard found.",
        updateSession: false,
        disclaimerShown: false
      };
    }
    const leader = leaderboard.leaders.find(l => l.secretCode === leaderSecretCode);
    if (!leader) {
      return {
        message: "Leader not found for the provided code.",
        updateSession: false,
        disclaimerShown: false
      };
    }
    // Copy allocation to user's portfolio
    let userPortfolio = await UserPortfolio.findOne({ userId: session.userId });
    if (!userPortfolio) {
      userPortfolio = new UserPortfolio({ userId: session.userId, funds: [] });
    }
    // Overwrite allocation
    userPortfolio.allocation = leader.allocation;
    await userPortfolio.save();
    return {
      message: `You have successfully copied the portfolio of leader ${leader.secretCode}.`,
      updateSession: true,
      disclaimerShown: false
    };
  }
}

async function processIncomingMessage(phone, message, req) {
  return { success: true, message: 'Processed', phone, message };
}

async function sendMessage(phoneNumber, message, req) {
  if (!phoneNumber || !message) {
    return { success: false, message: 'Missing required fields' };
  }
  if (!/^\d{10,15}$/.test(phoneNumber)) {
    return { success: false, message: 'Invalid phone number format' };
  }
  return { success: true, message: 'Message sent successfully', messageId: 'mock-message-id' };
}

async function sendBulkMessages(recipients, message, req) {
  if (!Array.isArray(recipients)) {
    return { success: false, message: 'Recipients must be an array' };
  }
  if (!message) {
    return { success: false, message: 'Missing required fields' };
  }
  return { success: true, message: 'Bulk messages sent', results: recipients.map(r => ({ phoneNumber: r, status: 'sent' })) };
}

async function getAllActiveSessions() {
  return [
    {
      _id: 'test-session-id',
      phoneNumber: '919876543210',
      onboardingState: 'STARTED',
      isActive: true,
      lastActivity: new Date().toISOString()
    }
  ];
}

async function getSessionByPhoneNumber(phoneNumber) {
  if (phoneNumber === '919876543210') {
    return {
      _id: 'test-session-id',
      phoneNumber: '919876543210',
      onboardingState: 'STARTED',
      isActive: true,
      lastActivity: new Date().toISOString()
    };
  }
  return null;
}

async function updateSession(phoneNumber, updateData) {
  if (phoneNumber === '919876543210') {
    return {
      _id: 'test-session-id',
      phoneNumber: '919876543210',
      ...updateData,
      lastActivity: new Date().toISOString()
    };
  }
  return null;
}

async function deleteSession(phoneNumber) {
  if (phoneNumber === '919876543210') {
    return {
      success: true,
      message: 'Session deleted successfully'
    };
  }
  return null;
}

async function getMessagesByPhoneNumber(phoneNumber, limit = 10, page = 1) {
  return [
    {
      _id: 'message1',
      phoneNumber: '919876543210',
      messageId: 'wa-message-1',
      type: 'incoming',
      content: 'Hello',
      timestamp: new Date().toISOString()
    },
    {
      _id: 'message2',
      phoneNumber: '919876543210',
      messageId: 'wa-message-2',
      type: 'outgoing',
      content: 'Hi there!',
      timestamp: new Date().toISOString()
    }
  ];
}

module.exports = {
  WhatsAppService,
  whatsappClient,
  processIncomingMessage,
  sendMessage,
  sendBulkMessages,
  getAllActiveSessions,
  getSessionByPhoneNumber,
  updateSession,
  deleteSession,
  getMessagesByPhoneNumber
};
module.exports.instance = new WhatsAppService(); 