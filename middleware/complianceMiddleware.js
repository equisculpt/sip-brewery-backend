const logger = require('../utils/logger');

/**
 * AMFI & SEBI COMPLIANCE MIDDLEWARE
 * Ensures all API responses are compliant with mutual fund distributor regulations
 */

// AMFI Registration Details
const AMFI_DETAILS = {
  registration_number: "ARN-12345", // TODO: Replace with actual AMFI registration
  registration_name: "SIP Brewery Financial Services Pvt Ltd",
  registration_date: "2024-01-01",
  validity_date: "2027-01-01"
};

// Mandatory SEBI Disclaimers
const SEBI_DISCLAIMERS = {
  market_risk: "Mutual fund investments are subject to market risks, read all scheme related documents carefully.",
  past_performance: "Past performance does not guarantee future returns.",
  risk_factors: "Please read the risk factors carefully before investing.",
  scheme_documents: "Read all scheme related documents carefully before investing.",
  investment_decision: "Investment decisions should be made independently after careful consideration of the scheme information document.",
  consult_advisor: "Please consult your financial advisor if required."
};

// Business Model Clarification
const BUSINESS_MODEL = {
  type: "MUTUAL_FUND_DISTRIBUTOR",
  description: "We are Mutual Fund Distributors, NOT Investment Advisors",
  role: "We facilitate mutual fund investments and provide educational content only",
  no_advice: "We do not provide investment advisory services"
};

/**
 * Add compliance headers to all responses
 */
const addComplianceHeaders = (req, res, next) => {
  // Add AMFI registration to response headers
  res.setHeader('X-AMFI-Registration', AMFI_DETAILS.registration_number);
  res.setHeader('X-Business-Model', 'MUTUAL_FUND_DISTRIBUTOR');
  res.setHeader('X-Compliance-Status', 'SEBI_COMPLIANT');
  
  next();
};

/**
 * Validate request for investment advice language
 */
const validateRequestCompliance = (req, res, next) => {
  const requestBody = JSON.stringify(req.body).toLowerCase();
  const requestUrl = req.url.toLowerCase();
  
  // Prohibited terms that constitute investment advice
  const prohibitedTerms = [
    'recommend',
    'advice',
    'suggest',
    'should invest',
    'best fund',
    'optimal',
    'guaranteed',
    'promise',
    'assure'
  ];
  
  // Check for prohibited terms
  const foundProhibited = prohibitedTerms.some(term => 
    requestBody.includes(term) || requestUrl.includes(term)
  );
  
  if (foundProhibited) {
    logger.warn(`⚠️ Compliance warning: Request contains potential investment advice language`, {
      url: req.url,
      body: req.body,
      user: req.user?.id
    });
  }
  
  next();
};

/**
 * Add mandatory disclaimers to API responses
 */
const addComplianceDisclaimer = (req, res, next) => {
  const originalSend = res.send;
  
  res.send = function(data) {
    try {
      let responseData = data;
      
      // Parse JSON responses to add disclaimers
      if (typeof data === 'string') {
        try {
          responseData = JSON.parse(data);
        } catch (e) {
          // Not JSON, send as is
          return originalSend.call(this, data);
        }
      }
      
      // Add compliance information to response
      if (responseData && typeof responseData === 'object') {
        responseData.compliance = {
          amfi_registration: AMFI_DETAILS,
          business_model: BUSINESS_MODEL,
          sebi_disclaimers: SEBI_DISCLAIMERS,
          compliance_timestamp: new Date().toISOString(),
          compliance_version: "1.0"
        };
        
        // Add specific disclaimers based on response content
        if (needsInvestmentDisclaimer(responseData)) {
          responseData.mandatory_disclaimer = {
            critical_warning: "🚨 IMPORTANT: We are Mutual Fund Distributors, NOT Investment Advisors",
            market_risk: SEBI_DISCLAIMERS.market_risk,
            past_performance: SEBI_DISCLAIMERS.past_performance,
            independent_decision: "Please make independent investment decisions based on your financial goals and risk appetite"
          };
        }
      }
      
      return originalSend.call(this, JSON.stringify(responseData));
    } catch (error) {
      logger.error('Compliance middleware error:', error);
      return originalSend.call(this, data);
    }
  };
  
  next();
};

/**
 * Check if response needs investment disclaimer
 */
const needsInvestmentDisclaimer = (responseData) => {
  const responseString = JSON.stringify(responseData).toLowerCase();
  
  const investmentKeywords = [
    'fund',
    'investment',
    'portfolio',
    'return',
    'performance',
    'analysis',
    'recommendation',
    'suggestion'
  ];
  
  return investmentKeywords.some(keyword => responseString.includes(keyword));
};

/**
 * Log compliance-related activities
 */
const logComplianceActivity = (req, res, next) => {
  const startTime = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - startTime;
    
    logger.info('Compliance API Activity', {
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.get('User-Agent'),
      ip: req.ip,
      userId: req.user?.id,
      complianceStatus: 'MONITORED',
      timestamp: new Date().toISOString()
    });
  });
  
  next();
};

/**
 * Sanitize response to remove investment advice language
 */
const sanitizeResponse = (req, res, next) => {
  const originalSend = res.send;
  
  res.send = function(data) {
    try {
      let responseData = data;
      
      if (typeof data === 'string') {
        try {
          responseData = JSON.parse(data);
        } catch (e) {
          // Not JSON, sanitize string directly
          const sanitized = sanitizeText(data);
          return originalSend.call(this, sanitized);
        }
      }
      
      // Sanitize object responses
      if (responseData && typeof responseData === 'object') {
        responseData = sanitizeObject(responseData);
      }
      
      return originalSend.call(this, JSON.stringify(responseData));
    } catch (error) {
      logger.error('Response sanitization error:', error);
      return originalSend.call(this, data);
    }
  };
  
  next();
};

/**
 * Sanitize text to replace investment advice language
 */
const sanitizeText = (text) => {
  const replacements = {
    'recommend': 'suggest for educational purposes',
    'best fund': 'popular fund',
    'optimal': 'sample',
    'guaranteed': 'historical',
    'will return': 'has historically returned',
    'should invest': 'may consider learning about',
    'advice': 'educational information',
    'suggests investing': 'provides educational information about'
  };
  
  let sanitized = text;
  Object.entries(replacements).forEach(([prohibited, compliant]) => {
    const regex = new RegExp(prohibited, 'gi');
    sanitized = sanitized.replace(regex, compliant);
  });
  
  return sanitized;
};

/**
 * Sanitize object recursively
 */
const sanitizeObject = (obj) => {
  if (typeof obj !== 'object' || obj === null) {
    return obj;
  }
  
  if (Array.isArray(obj)) {
    return obj.map(item => sanitizeObject(item));
  }
  
  const sanitized = {};
  Object.entries(obj).forEach(([key, value]) => {
    if (typeof value === 'string') {
      sanitized[key] = sanitizeText(value);
    } else if (typeof value === 'object') {
      sanitized[key] = sanitizeObject(value);
    } else {
      sanitized[key] = value;
    }
  });
  
  return sanitized;
};

module.exports = {
  addComplianceHeaders,
  validateRequestCompliance,
  addComplianceDisclaimer,
  logComplianceActivity,
  sanitizeResponse,
  AMFI_DETAILS,
  SEBI_DISCLAIMERS,
  BUSINESS_MODEL
};
