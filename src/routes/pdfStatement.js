const express = require('express');
const router = express.Router();
const pdfStatementController = require('../controllers/pdfStatementController');
const { authenticateToken } = require('../middleware/auth');

// Apply authentication middleware to all routes
router.use(authenticateToken);

/**
 * @route POST /api/pdf/statement/generate
 * @desc Generate PDF statement
 * @access Private
 */
router.post('/statement/generate', pdfStatementController.generateStatement);

/**
 * @route GET /api/pdf/statement/types
 * @desc Get available statement types
 * @access Private
 */
router.get('/statement/types', pdfStatementController.getStatementTypes);

/**
 * @route POST /api/pdf/statement/preview
 * @desc Preview statement data (without generating PDF)
 * @access Private
 */
router.post('/statement/preview', pdfStatementController.previewStatement);

module.exports = router; 