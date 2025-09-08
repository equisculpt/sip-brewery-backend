const logger = require('../utils/logger');

class WhatsAppClient {
  constructor() {
    this.provider = process.env.WHATSAPP_PROVIDER || 'TWILIO';
    this.accountSid = process.env.TWILIO_ACCOUNT_SID;
    this.authToken = process.env.TWILIO_AUTH_TOKEN;
    this.phoneNumber = process.env.TWILIO_PHONE_NUMBER;
    
    // Initialize Twilio client if credentials are available
    this.twilioClient = null;
    if (this.accountSid && this.authToken) {
      try {
        const twilio = require('twilio');
        this.twilioClient = twilio(this.accountSid, this.authToken);
        logger.info('Twilio WhatsApp client initialized');
      } catch (error) {
        logger.error('Failed to initialize Twilio client:', error);
      }
    } else {
      logger.warn('Twilio credentials not found. WhatsApp features will be simulated.');
    }
  }

  /**
   * Send WhatsApp message
   */
  async sendMessage(to, message, options = {}) {
    try {
      if (this.twilioClient) {
        return await this.sendViaTwilio(to, message, options);
      } else {
        return await this.sendSimulated(to, message, options);
      }
    } catch (error) {
      logger.error('Failed to send WhatsApp message:', error);
      throw error;
    }
  }

  /**
   * Send message via Twilio
   */
  async sendViaTwilio(to, message, options = {}) {
    try {
      const messageData = {
        body: message,
        from: `whatsapp:${this.phoneNumber}`,
        to: `whatsapp:${to}`
      };

      // Add media if provided
      if (options.mediaUrl) {
        messageData.mediaUrl = options.mediaUrl;
      }

      const result = await this.twilioClient.messages.create(messageData);
      
      logger.info(`WhatsApp message sent via Twilio: ${result.sid}`);
      
      return {
        success: true,
        messageId: result.sid,
        status: result.status,
        provider: 'TWILIO'
      };
    } catch (error) {
      logger.error('Twilio message send failed:', error);
      throw error;
    }
  }

  /**
   * Simulate WhatsApp message sending (for development)
   */
  async sendSimulated(to, message, options = {}) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 100));
    
    const messageId = `sim_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    logger.info(`Simulated WhatsApp message sent to ${to}: ${message.substring(0, 50)}...`);
    
    return {
      success: true,
      messageId: messageId,
      status: 'delivered',
      provider: 'SIMULATED'
    };
  }

  /**
   * Send media message (document, image, etc.)
   */
  async sendMedia(to, mediaUrl, caption = '', options = {}) {
    try {
      if (this.twilioClient) {
        return await this.sendViaTwilio(to, caption, { mediaUrl });
      } else {
        return await this.sendSimulated(to, `[Media: ${mediaUrl}]\n${caption}`, options);
      }
    } catch (error) {
      logger.error('Failed to send media message:', error);
      throw error;
    }
  }

  /**
   * Send interactive message with buttons
   */
  async sendInteractiveMessage(to, text, buttons, options = {}) {
    try {
      // For now, send as text with numbered options
      let message = text + '\n\n';
      buttons.forEach((button, index) => {
        message += `${index + 1}. ${button.text}\n`;
      });
      
      return await this.sendMessage(to, message, options);
    } catch (error) {
      logger.error('Failed to send interactive message:', error);
      throw error;
    }
  }

  /**
   * Send template message (for onboarding, etc.)
   */
  async sendTemplate(to, templateName, variables = {}, options = {}) {
    try {
      const templates = {
        welcome: `Hello! I'm SIPBrewery's investment assistant 🤖\n\nI can help you with:\n• View your portfolio\n• Start SIP investments\n• Check rewards & referrals\n• Get fund analysis\n\nWhat would you like to do?`,
        
        onboarding_name: `Welcome to SIPBrewery! 🎉\n\nTo get started, please share your full name.`,
        
        onboarding_email: `Thanks ${variables.name}! 📧\n\nPlease share your email address.`,
        
        onboarding_pan: `Great! Now I need your PAN number for KYC verification.`,
        
        kyc_verified: `✅ Your KYC is verified!\n\nYou can now start investing via SIPBrewery. What would you like to do?\n\n1. View Portfolio\n2. Start SIP\n3. Check Rewards\n4. Get Fund Analysis`,
        
        kyc_pending: `⏳ Your KYC is pending verification.\n\nPlease complete your KYC to start investing:\n[Digio KYC Link Placeholder]\n\nOnce verified, you'll be able to invest in mutual funds.`,
        
        sip_confirmation: `📋 SIP Order Summary:\n\nFund: ${variables.fundName}\nAmount: ₹${variables.amount}/month\nOrder ID: ${variables.orderId}\n\nPlease reply with:\n✅ Yes - to confirm\n❌ No - to cancel`,
        
        sip_confirmed: `🎉 SIP confirmed!\n\nYour SIP order has been created successfully.\nOrder ID: ${variables.orderId}\n\nYou'll receive updates on your SIP status.`,
        
        portfolio_summary: `📊 Your Portfolio Summary:\n\nTotal Value: ₹${variables.totalValue}\nTotal Invested: ₹${variables.totalInvested}\nReturns: ${variables.returns}%\n\nTop Holdings:\n${variables.topHoldings}`,
        
        rewards_summary: `🎁 Your Rewards Summary:\n\nLoyalty Points: ${variables.points}\nCashback: ₹${variables.cashback}\nReferral Bonus: ₹${variables.referralBonus}\nPending Payout: ₹${variables.pendingPayout}`,
        
        referral_link: `🔗 Your Referral Link:\n\nsipbrewery.com/join?ref=${variables.referralCode}\n\nShare this link with friends and earn ₹100 for each successful referral!`,
        
        leaderboard: `🏆 Top 5 Investors:\n\n${variables.leaders.map((leader, index) => 
          `${index + 1}. ${leader.name} - ${leader.returns}% returns`
        ).join('\n')}\n\nReply with "Copy leader X" to copy their portfolio.`,
        
        help: `🤖 How can I help you?\n\nAvailable commands:\n• "My Portfolio" - View holdings\n• "Start SIP" - Begin investment\n• "My Rewards" - Check rewards\n• "Refer a friend" - Get referral link\n• "Leaderboard" - Top performers\n• "Analyse [Fund]" - Fund analysis\n• "Send statement" - Get statements`
      };

      const template = templates[templateName] || templates.help;
      return await this.sendMessage(to, template, options);
    } catch (error) {
      logger.error('Failed to send template message:', error);
      throw error;
    }
  }

  /**
   * Validate phone number format
   */
  validatePhoneNumber(phoneNumber) {
    // Remove any non-digit characters
    const cleaned = phoneNumber.replace(/\D/g, '');
    
    // Check if it's a valid Indian mobile number
    if (cleaned.length === 10 && cleaned.startsWith('6') || cleaned.startsWith('7') || cleaned.startsWith('8') || cleaned.startsWith('9')) {
      return `+91${cleaned}`;
    }
    
    // Check if it's already in international format
    if (cleaned.startsWith('91') && cleaned.length === 12) {
      return `+${cleaned}`;
    }
    
    // Check if it's already in +91 format
    if (cleaned.startsWith('9191') && cleaned.length === 13) {
      return `+${cleaned.substring(2)}`;
    }
    
    return null;
  }

  /**
   * Get client status
   */
  getStatus() {
    return {
      provider: this.provider,
      configured: !!(this.accountSid && this.authToken),
      phoneNumber: this.phoneNumber,
      clientAvailable: !!this.twilioClient
    };
  }

  /**
   * Test message sending
   */
  async testConnection() {
    try {
      const testNumber = process.env.TEST_PHONE_NUMBER;
      if (!testNumber) {
        return { success: false, message: 'TEST_PHONE_NUMBER not configured' };
      }

      const result = await this.sendMessage(testNumber, '🧪 WhatsApp connection test successful!');
      return { success: true, result };
    } catch (error) {
      return { success: false, error: error.message };
    }
  }
}

// Only initialize Twilio if not in test mode
let client = null;
if (process.env.NODE_ENV !== 'test') {
  const twilio = require('twilio');
  client = new twilio(
    process.env.TWILIO_ACCOUNT_SID,
    process.env.TWILIO_AUTH_TOKEN
  );
} else {
  // In test mode, provide a mock client
  client = {
    sendMessage: async () => ({ success: true, message: 'Simulated WhatsApp send in test mode' })
  };
}

module.exports = client; 