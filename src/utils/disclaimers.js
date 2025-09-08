const logger = require('./logger');

class DisclaimerManager {
  constructor() {
    this.disclaimerCounter = new Map();
    this.messageThreshold = 5; // Show disclaimer every 5 messages
    this.lastDisclaimerTime = new Map();
    this.disclaimerCooldown = 5 * 60 * 1000; // 5 minutes cooldown
  }

  /**
   * Get comprehensive AMFI and SEBI disclaimers
   */
  getComprehensiveDisclaimer() {
    return `⚠️ *Important Disclaimers & Risk Disclosure*

📊 *Investment Risks*:
• Mutual fund investments are subject to market risks
• Please read all scheme related documents carefully before investing
• Past performance is not indicative of future returns
• NAV may go up or down based on market conditions

🎁 *Platform Rewards*:
• Rewards are discretionary, not guaranteed
• May be changed or withdrawn at any time
• Research and analysis are for informational purposes only
• Do not constitute investment advice

🏛️ *Regulatory Compliance*:
• We are AMFI registered mutual fund distributors
• Please see our Terms & Conditions and Commission Disclosure
• Complete details about fees and commissions available on our website

⚖️ *Legal Notice*:
• SIP Brewery is a trademark of Equisculpt Ventures Pvt. Ltd.
• Equisculpt Ventures Pvt. Ltd. is an AMFI Registered Mutual Fund Distributor
• We may earn commission when you invest through our platform
• We are NOT SEBI registered investment advisors
• All data is for educational purposes only

📞 *Contact*: For detailed information, visit sipbrewery.com or call our support team.`;
  }

  /**
   * Get short disclaimer for frequent messages
   */
  getShortDisclaimer() {
    return `⚠️ *Disclaimer*: Mutual funds are subject to market risks. We are AMFI registered distributors, not SEBI advisors. All data is for educational purposes only.`;
  }

  /**
   * Get investment-specific disclaimer
   */
  getInvestmentDisclaimer() {
    return `⚠️ *Investment Disclaimer*:
• Mutual funds are subject to market risks
• Past performance doesn't guarantee future returns
• Please read scheme documents carefully
• We are AMFI registered distributors, not SEBI advisors
• All analysis is for educational purposes only`;
  }

  /**
   * Get AI analysis disclaimer
   */
  getAiAnalysisDisclaimer() {
    return `🤖 *AI Analysis Disclaimer*:
• This is AI-generated analysis for educational purposes only
• Not investment advice or recommendation
• Please consult a SEBI-registered advisor before investing
• We are AMFI registered distributors, not SEBI advisors
• Past performance doesn't guarantee future returns`;
  }

  /**
   * Get portfolio disclaimer
   */
  getPortfolioDisclaimer() {
    return `📊 *Portfolio Disclaimer*:
• Portfolio values are subject to market fluctuations
• Historical data shown for reference only
• We are AMFI registered distributors, not SEBI advisors
• For investment advice, consult a SEBI-registered advisor`;
  }

  /**
   * Get rewards disclaimer
   */
  getRewardsDisclaimer() {
    return `🎁 *Rewards Disclaimer*:
• Rewards are discretionary and not guaranteed
• May be changed or withdrawn at any time
• Subject to terms and conditions
• We are AMFI registered distributors`;
  }

  /**
   * Check if disclaimer should be shown
   */
  shouldShowDisclaimer(phoneNumber, messageType = 'general') {
    const now = Date.now();
    const lastTime = this.lastDisclaimerTime.get(phoneNumber) || 0;
    
    // Check cooldown period
    if (now - lastTime < this.disclaimerCooldown) {
      return false;
    }
    
    // Get message count
    const count = this.disclaimerCounter.get(phoneNumber) || 0;
    
    // Show disclaimer every 5 messages or for specific types
    const shouldShow = count % this.messageThreshold === 0 || 
                      messageType === 'investment' ||
                      messageType === 'ai_analysis' ||
                      messageType === 'portfolio';
    
    if (shouldShow) {
      this.lastDisclaimerTime.set(phoneNumber, now);
    }
    
    return shouldShow;
  }

  /**
   * Increment message counter
   */
  incrementMessageCount(phoneNumber) {
    const count = this.disclaimerCounter.get(phoneNumber) || 0;
    this.disclaimerCounter.set(phoneNumber, count + 1);
  }

  /**
   * Get appropriate disclaimer based on context
   */
  getDisclaimerForContext(context, phoneNumber) {
    this.incrementMessageCount(phoneNumber);
    
    if (!this.shouldShowDisclaimer(phoneNumber, context)) {
      return null;
    }
    
    switch (context) {
      case 'investment':
      case 'sip_create':
      case 'lump_sum':
        return this.getInvestmentDisclaimer();
        
      case 'ai_analysis':
        return this.getAiAnalysisDisclaimer();
        
      case 'portfolio':
      case 'portfolio_view':
        return this.getPortfolioDisclaimer();
        
      case 'rewards':
      case 'referral':
        return this.getRewardsDisclaimer();
        
      case 'comprehensive':
        return this.getComprehensiveDisclaimer();
        
      default:
        return this.getShortDisclaimer();
    }
  }

  /**
   * Add disclaimer to message
   */
  addDisclaimerToMessage(message, context, phoneNumber) {
    const disclaimer = this.getDisclaimerForContext(context, phoneNumber);
    
    if (!disclaimer) {
      return message;
    }
    
    return `${message}\n\n${disclaimer}`;
  }

  /**
   * Get welcome message with disclaimer
   */
  getWelcomeMessageWithDisclaimer(phoneNumber) {
    const welcomeMessage = `Hello! I'm SIPBrewery's investment assistant 🤖

I can help you with:
• View your portfolio 📊
• Start SIP investments 💰
• Check rewards & referrals 🎁
• Get fund analysis 🤖
• Generate statements 📄

What would you like to do?`;
    
    return this.addDisclaimerToMessage(welcomeMessage, 'comprehensive', phoneNumber);
  }

  /**
   * Get onboarding message with disclaimer
   */
  getOnboardingMessageWithDisclaimer(phoneNumber, step) {
    let message = '';
    
    switch (step) {
      case 'name':
        message = `Welcome to SIPBrewery! 🎉

To get started, please share your full name.`;
        break;
        
      case 'email':
        message = `Thanks! 📧

Please share your email address.`;
        break;
        
      case 'pan':
        message = `Great! Now I need your PAN number for KYC verification.`;
        break;
        
      case 'kyc_verified':
        message = `✅ Your KYC is verified!

You can now start investing via SIPBrewery. What would you like to do?

1. View Portfolio
2. Start SIP
3. Check Rewards
4. Get Fund Analysis`;
        break;
        
      case 'kyc_pending':
        message = `⏳ Your KYC is pending verification.

Please complete your KYC to start investing:
[Digio KYC Link Placeholder]

Once verified, you'll be able to invest in mutual funds.`;
        break;
    }
    
    return this.addDisclaimerToMessage(message, 'comprehensive', phoneNumber);
  }

  /**
   * Get SIP confirmation with disclaimer
   */
  getSipConfirmationWithDisclaimer(phoneNumber, fundName, amount) {
    const message = `📋 SIP Order Summary:

Fund: ${fundName}
Amount: ₹${amount.toLocaleString()}/month

Please reply with:
✅ Yes - to confirm
❌ No - to cancel`;
    
    return this.addDisclaimerToMessage(message, 'investment', phoneNumber);
  }

  /**
   * Get portfolio summary with disclaimer
   */
  getPortfolioSummaryWithDisclaimer(phoneNumber, portfolioData) {
    const message = `📊 Your Portfolio Summary:

Total Value: ₹${portfolioData.totalValue.toLocaleString()}
Total Invested: ₹${portfolioData.totalInvested.toLocaleString()}
Returns: ${portfolioData.returns.toFixed(2)}%

Top Holdings:
${portfolioData.topHoldings.join('\n')}`;
    
    return this.addDisclaimerToMessage(message, 'portfolio', phoneNumber);
  }

  /**
   * Get AI analysis with disclaimer
   */
  getAiAnalysisWithDisclaimer(phoneNumber, analysis) {
    return this.addDisclaimerToMessage(analysis, 'ai_analysis', phoneNumber);
  }

  /**
   * Get rewards summary with disclaimer
   */
  getRewardsSummaryWithDisclaimer(phoneNumber, rewardsData) {
    const message = `🎁 Your Rewards Summary:

Loyalty Points: ${rewardsData.points}
Cashback: ₹${rewardsData.cashback}
Referral Bonus: ₹${rewardsData.referralBonus}
Pending Payout: ₹${rewardsData.pendingPayout}`;
    
    return this.addDisclaimerToMessage(message, 'rewards', phoneNumber);
  }

  /**
   * Get help message with disclaimer
   */
  getHelpMessageWithDisclaimer(phoneNumber) {
    const message = `🤖 How can I help you?

Available commands:
• "My Portfolio" - View holdings
• "Start SIP" - Begin investment
• "My Rewards" - Check rewards
• "Refer a friend" - Get referral link
• "Leaderboard" - Top performers
• "Analyse [Fund]" - Fund analysis
• "Send statement" - Get statements`;
    
    return this.addDisclaimerToMessage(message, 'comprehensive', phoneNumber);
  }

  /**
   * Reset disclaimer counter for a user
   */
  resetDisclaimerCounter(phoneNumber) {
    this.disclaimerCounter.delete(phoneNumber);
    this.lastDisclaimerTime.delete(phoneNumber);
  }

  /**
   * Get disclaimer statistics
   */
  getDisclaimerStats() {
    return {
      activeUsers: this.disclaimerCounter.size,
      messageThreshold: this.messageThreshold,
      cooldownMinutes: this.disclaimerCooldown / (60 * 1000)
    };
  }
}

module.exports = new DisclaimerManager(); 