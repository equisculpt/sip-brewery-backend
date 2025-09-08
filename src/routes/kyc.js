const express = require('express');
const { body, param } = require('express-validator');
const router = express.Router();
const kycController = require('../controllers/kycController');

// Validate userId param for all endpoints
const validateUserId = [
  param('userId').isMongoId().withMessage('Invalid userId')
];

// GET /kyc/status/:userId
const { handleValidationErrors } = require('../middleware/validation');

router.get('/status/:userId', validateUserId, handleValidationErrors, kycController.getKYCStatus);

// POST /kyc/retrigger/:userId
router.post('/retrigger/:userId', validateUserId, handleValidationErrors, kycController.retriggerKYC);

// GET /kyc/logs/:userId
router.get('/logs/:userId', validateUserId, handleValidationErrors, kycController.getKYCLogs);

module.exports = router;
