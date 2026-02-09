const express = require('express');
const { body, param, query } = require('express-validator');
const partnerController = require('../controllers/partnerController');
const { authenticateToken } = require('../middleware/auth');
const { handleValidationErrors } = require('../middleware/validation');
const { requirePartnerRole, enforcePartnerClientAccess } = require('../middleware/partnerAccess');

const router = express.Router();

const partnerRoles = ['OWNER', 'ADMIN', 'ADVISOR', 'OPERATIONS', 'ANALYST'];

const validatePartnerOnboard = [
  body('name').isString().notEmpty().withMessage('Partner name is required'),
  body('code').isString().notEmpty().withMessage('Partner code is required'),
  body('type')
    .isIn(['IFA', 'SUB_DISTRIBUTOR', 'BROKER', 'INSTITUTIONAL'])
    .withMessage('Invalid partner type'),
  body('contact.name').optional().isString(),
  body('contact.email').optional().isEmail().withMessage('Invalid contact email'),
  body('contact.phone').optional().isString(),
  handleValidationErrors
];

const validateClientMap = [
  body('investorId').notEmpty().withMessage('Investor id is required'),
  body('relationshipType')
    .optional()
    .isIn(['REFERRAL', 'DASHBOARD_ONBOARDING'])
    .withMessage('Invalid relationship type'),
  body('consentId').optional().isMongoId().withMessage('Invalid consent id'),
  body('consentScope').optional().isString(),
  body('consentChannel')
    .optional()
    .isIn(['OTP', 'ESIGN', 'VOICE', 'CHECKBOX', 'DOCUMENT'])
    .withMessage('Invalid consent channel'),
  body('consentArtifactUri').optional().isString(),
  handleValidationErrors
];

const validateReferralMap = [
  param('partnerCode').isString().notEmpty().withMessage('Partner code is required'),
  body('consentChannel')
    .optional()
    .isIn(['OTP', 'ESIGN', 'VOICE', 'CHECKBOX', 'DOCUMENT'])
    .withMessage('Invalid consent channel'),
  body('consentArtifactUri').optional().isString(),
  handleValidationErrors
];

const validateClientList = [
  query('status')
    .optional()
    .isIn(['ACTIVE', 'SUSPENDED', 'TERMINATED'])
    .withMessage('Invalid status filter'),
  query('limit').optional().isInt({ min: 1, max: 100 }),
  query('offset').optional().isInt({ min: 0 }),
  handleValidationErrors
];

router.post('/onboard', authenticateToken, validatePartnerOnboard, partnerController.createPartner);
router.get('/profile', authenticateToken, requirePartnerRole(partnerRoles), partnerController.getPartnerProfile);
router.patch('/profile', authenticateToken, requirePartnerRole(partnerRoles), partnerController.updatePartnerProfile);

router.post('/clients', authenticateToken, requirePartnerRole(partnerRoles), validateClientMap, partnerController.mapClient);
router.get('/clients', authenticateToken, requirePartnerRole(partnerRoles), validateClientList, partnerController.listClients);
router.get(
  '/clients/:investorId',
  authenticateToken,
  requirePartnerRole(partnerRoles),
  enforcePartnerClientAccess,
  partnerController.getClientDetail
);

router.get(
  '/dashboard/summary',
  authenticateToken,
  requirePartnerRole(partnerRoles),
  partnerController.getDashboardSummary
);

router.post('/referral/:partnerCode', authenticateToken, validateReferralMap, partnerController.mapClientByReferral);

module.exports = router;
