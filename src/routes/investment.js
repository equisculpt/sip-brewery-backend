const express = require('express');
const { body, param } = require('express-validator');
const router = express.Router();
const investmentController = require('../controllers/investmentController');
const logger = require('../utils/logger');

const validateInvestment = [
  body('userId').isMongoId().withMessage('Invalid userId'),
  body('schemeCode').isString().notEmpty().withMessage('Scheme code is required'),
  body('amount').isFloat({ min: 100 }).withMessage('Amount must be at least 100'),
  body('pan').matches(/[A-Z]{5}[0-9]{4}[A-Z]{1}/).withMessage('Invalid PAN'),
  body('email').isEmail().withMessage('Invalid email')
];

// POST /investment/lumpsum
const { handleValidationErrors } = require('../middleware/validation');

router.post('/lumpsum', validateInvestment, handleValidationErrors, investmentController.placeLumpsumOrder);

// POST /investment/sip
router.post('/sip', validateInvestment, handleValidationErrors, investmentController.placeSipOrder);

module.exports = router;
