const express = require('express');
const { body } = require('express-validator');
const { handleValidationErrors } = require('../middleware/validation');
const redemptionController = require('../controllers/redemptionController');
const router = express.Router();

const validateRedemption = [
  body('userId').isMongoId().withMessage('Invalid userId'),
  body('schemeCode').isString().notEmpty().withMessage('Scheme code is required'),
  body('amount').isFloat({ min: 100 }).withMessage('Amount must be at least 100'),
  body('pan').matches(/[A-Z]{5}[0-9]{4}[A-Z]{1}/).withMessage('Invalid PAN'),
  handleValidationErrors
];

// POST /redemption
router.post('/', validateRedemption, redemptionController.redeemFund);

module.exports = router;
