const { body, param, query } = require('express-validator');

const bseValidationSchemas = {
  createClient: [
    body('clientData.firstName').trim().notEmpty().withMessage('First name is required'),
    body('clientData.lastName').trim().notEmpty().withMessage('Last name is required'),
    body('clientData.dateOfBirth').isISO8601().withMessage('Valid date of birth is required'),
    body('clientData.gender').optional().isIn(['MALE', 'FEMALE', 'OTHER']).withMessage('Invalid gender'),
    body('clientData.panNumber').matches(/^[A-Z]{5}[0-9]{4}[A-Z]{1}$/).withMessage('Invalid PAN number'),
    body('clientData.aadhaarNumber').optional().matches(/^[0-9]{12}$/).withMessage('Invalid Aadhaar number'),
    body('clientData.email').isEmail().withMessage('Valid email is required'),
    body('clientData.mobile').matches(/^[0-9]{10}$/).withMessage('Valid 10-digit mobile number is required'),
    body('clientData.address.line1').trim().notEmpty().withMessage('Address line 1 is required'),
    body('clientData.address.city').trim().notEmpty().withMessage('City is required'),
    body('clientData.address.state').trim().notEmpty().withMessage('State is required'),
    body('clientData.address.pincode').matches(/^[0-9]{6}$/).withMessage('Valid 6-digit pincode is required'),
    body('clientData.bankDetails.accountNumber').trim().notEmpty().withMessage('Bank account number is required'),
    body('clientData.bankDetails.ifscCode').matches(/^[A-Z]{4}0[A-Z0-9]{6}$/).withMessage('Valid IFSC code is required'),
    body('clientData.bankDetails.accountHolderName').trim().notEmpty().withMessage('Account holder name is required')
  ],

  modifyClient: [
    param('clientId').trim().notEmpty().withMessage('Client ID is required'),
    body('clientData.firstName').optional().trim().notEmpty().withMessage('First name cannot be empty'),
    body('clientData.lastName').optional().trim().notEmpty().withMessage('Last name cannot be empty'),
    body('clientData.email').optional().isEmail().withMessage('Valid email is required'),
    body('clientData.mobile').optional().matches(/^[0-9]{10}$/).withMessage('Valid 10-digit mobile number is required')
  ],

  placeLumpsumOrder: [
    body('orderData.clientId').trim().notEmpty().withMessage('Client ID is required'),
    body('orderData.schemeCode').trim().notEmpty().withMessage('Scheme code is required'),
    body('orderData.amount').isFloat({ min: 1000 }).withMessage('Amount must be at least ₹1000'),
    body('orderData.paymentMode').isIn(['ONLINE', 'CHEQUE', 'DD', 'NEFT', 'RTGS']).withMessage('Invalid payment mode'),
    body('orderData.folioNumber').optional().trim(),
    body('orderData.bankAccount').optional().trim(),
    body('orderData.nomineeDetails').optional().isArray().withMessage('Nominee details must be an array')
  ],

  placeRedemptionOrder: [
    body('redemptionData.clientId').trim().notEmpty().withMessage('Client ID is required'),
    body('redemptionData.schemeCode').trim().notEmpty().withMessage('Scheme code is required'),
    body('redemptionData.folioNumber').optional().trim(),
    body('redemptionData.redemptionType').isIn(['UNITS', 'AMOUNT', 'ALL']).withMessage('Invalid redemption type'),
    body('redemptionData.units').optional().isFloat({ min: 0.001 }).withMessage('Units must be greater than 0'),
    body('redemptionData.amount').optional().isFloat({ min: 100 }).withMessage('Amount must be at least ₹100'),
    body('redemptionData.bankAccount').optional().trim(),
    body('redemptionData.redemptionMode').optional().isIn(['NORMAL', 'SWITCH']).withMessage('Invalid redemption mode')
  ],

  setupEMandate: [
    body('mandateData.clientId').trim().notEmpty().withMessage('Client ID is required'),
    body('mandateData.bankAccount.accountNumber').trim().notEmpty().withMessage('Bank account number is required'),
    body('mandateData.bankAccount.ifscCode').matches(/^[A-Z]{4}0[A-Z0-9]{6}$/).withMessage('Valid IFSC code is required'),
    body('mandateData.bankAccount.accountHolderName').trim().notEmpty().withMessage('Account holder name is required'),
    body('mandateData.amount').isFloat({ min: 1000 }).withMessage('Mandate amount must be at least ₹1000'),
    body('mandateData.frequency').isIn(['MONTHLY', 'WEEKLY', 'QUARTERLY']).withMessage('Invalid frequency'),
    body('mandateData.startDate').isISO8601().withMessage('Valid start date is required'),
    body('mandateData.endDate').isISO8601().withMessage('Valid end date is required'),
    body('mandateData.purpose').optional().trim()
  ],

  getCurrentNAV: [
    body('schemeCodes').isArray({ min: 1 }).withMessage('Scheme codes array is required'),
    body('schemeCodes.*').trim().notEmpty().withMessage('Each scheme code must be non-empty')
  ],

  getOrderStatus: [
    param('orderId').trim().notEmpty().withMessage('Order ID is required')
  ],

  getSchemeDetails: [
    param('schemeCode').trim().notEmpty().withMessage('Scheme code is required')
  ],

  getSchemePerformance: [
    param('schemeCode').trim().notEmpty().withMessage('Scheme code is required'),
    query('period').optional().isIn(['1M', '3M', '6M', '1Y', '3Y', '5Y']).withMessage('Invalid period')
  ],

  getClientFolios: [
    param('clientId').trim().notEmpty().withMessage('Client ID is required')
  ],

  getEMandateStatus: [
    param('mandateId').trim().notEmpty().withMessage('Mandate ID is required')
  ],

  cancelEMandate: [
    param('mandateId').trim().notEmpty().withMessage('Mandate ID is required'),
    body('reason').optional().trim()
  ],

  getSchemeMasterData: [
    query('category').optional().trim(),
    query('fundHouse').optional().trim(),
    query('isActive').optional().isBoolean().withMessage('isActive must be boolean'),
    query('limit').optional().isInt({ min: 1, max: 100 }).withMessage('Limit must be between 1 and 100'),
    query('offset').optional().isInt({ min: 0 }).withMessage('Offset must be non-negative')
  ],

  getTransactionReport: [
    query('clientId').optional().trim(),
    query('schemeCode').optional().trim(),
    query('folioNumber').optional().trim(),
    query('startDate').optional().isISO8601().withMessage('Valid start date is required'),
    query('endDate').optional().isISO8601().withMessage('Valid end date is required'),
    query('orderType').optional().trim(),
    query('limit').optional().isInt({ min: 1, max: 100 }).withMessage('Limit must be between 1 and 100'),
    query('offset').optional().isInt({ min: 0 }).withMessage('Offset must be non-negative')
  ],

  getNAVAndHoldingReport: [
    query('clientId').optional().trim(),
    query('schemeCode').optional().trim(),
    query('folioNumber').optional().trim(),
    query('date').optional().isISO8601().withMessage('Valid date is required')
  ]
};

module.exports = bseValidationSchemas;
