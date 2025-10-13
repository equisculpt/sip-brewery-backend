/**
 * SEBI/AMFI Compliance Rules
 * Define compliance checks for financial operations
 */

const logger = require('../utils/logger');

// Compliance rules for SEBI/AMFI regulations
const complianceRules = [
  // Rule 1: Investment amount limits
  function checkInvestmentLimits(actionContext) {
    if (actionContext.body && actionContext.body.amount) {
      const amount = parseFloat(actionContext.body.amount);
      
      // Check minimum investment
      if (amount < 500) {
        return {
          compliant: false,
          reason: 'Investment amount below minimum threshold of ₹500 (SEBI guidelines)'
        };
      }
      
      // Check maximum single investment (for retail investors)
      if (amount > 50000000) {
        logger.warn('High value transaction detected', { amount, user: actionContext.user });
      }
    }
    
    return { compliant: true };
  },
  
  // Rule 2: KYC verification
  function checkKYCStatus(actionContext) {
    if (actionContext.path.includes('/invest') || actionContext.path.includes('/sip')) {
      if (!actionContext.user || !actionContext.user.kycStatus) {
        return {
          compliant: false,
          reason: 'KYC verification required for investment transactions (SEBI/PMLA compliance)'
        };
      }
      
      if (actionContext.user.kycStatus !== 'VERIFIED') {
        return {
          compliant: false,
          reason: 'KYC status must be VERIFIED to proceed with investments'
        };
      }
    }
    
    return { compliant: true };
  },
  
  // Rule 3: Risk disclosure acknowledgment
  function checkRiskDisclosure(actionContext) {
    if (actionContext.path.includes('/invest') && actionContext.body) {
      // Check if user has acknowledged risk
      if (actionContext.body.requiresRiskAcknowledgment && !actionContext.body.riskAcknowledged) {
        return {
          compliant: false,
          reason: 'Risk disclosure acknowledgment required (SEBI investor protection)'
        };
      }
    }
    
    return { compliant: true };
  },
  
  // Rule 4: Age verification for investments
  function checkInvestorAge(actionContext) {
    if (actionContext.user && actionContext.user.dateOfBirth) {
      const age = Math.floor((Date.now() - new Date(actionContext.user.dateOfBirth).getTime()) / (365.25 * 24 * 60 * 60 * 1000));
      
      if (age < 18) {
        return {
          compliant: false,
          reason: 'Investor must be 18 years or older (SEBI regulations)'
        };
      }
    }
    
    return { compliant: true };
  },
  
  // Rule 5: Transaction frequency limits (prevent suspicious activity)
  function checkTransactionFrequency(actionContext) {
    // This would check against a rate limit service in production
    // For now, basic validation
    return { compliant: true };
  },
  
  // Rule 6: PAN card requirement
  function checkPANRequirement(actionContext) {
    if (actionContext.path.includes('/invest') && actionContext.user) {
      if (!actionContext.user.panCard) {
        return {
          compliant: false,
          reason: 'Valid PAN card required for mutual fund investments (SEBI/Income Tax regulations)'
        };
      }
    }
    
    return { compliant: true };
  },
  
  // Rule 7: Bank account verification
  function checkBankAccountVerification(actionContext) {
    if (actionContext.path.includes('/redeem') || actionContext.path.includes('/withdraw')) {
      if (!actionContext.user || !actionContext.user.bankAccountVerified) {
        return {
          compliant: false,
          reason: 'Bank account verification required for redemptions (SEBI guidelines)'
        };
      }
    }
    
    return { compliant: true };
  },
  
  // Rule 8: Cooling period for new accounts
  function checkAccountCoolingPeriod(actionContext) {
    if (actionContext.user && actionContext.user.createdAt) {
      const accountAge = Date.now() - new Date(actionContext.user.createdAt).getTime();
      const oneDayInMs = 24 * 60 * 60 * 1000;
      
      // Some high-value operations require 24-hour cooling period
      if (actionContext.path.includes('/lumpsum') && accountAge < oneDayInMs) {
        const amountStr = actionContext.body?.amount;
        const amount = amountStr ? parseFloat(amountStr) : 0;
        
        if (amount > 100000) {
          return {
            compliant: false,
            reason: '24-hour cooling period required for new accounts on high-value transactions (fraud prevention)'
          };
        }
      }
    }
    
    return { compliant: true };
  }
];

// Additional regulatory checks
const regulatoryChecks = {
  SEBI: {
    minimumInvestment: 500,
    kycMandatory: true,
    panRequired: true,
    riskDisclosureRequired: true
  },
  AMFI: {
    kycCompliance: 'mandatory',
    investorProtection: 'enabled'
  },
  PMLA: {
    enhancedDueDiligence: true,
    transactionMonitoring: true
  }
};

module.exports = complianceRules;
module.exports.regulatoryChecks = regulatoryChecks;
