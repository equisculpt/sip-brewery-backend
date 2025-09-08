/**
 * 📋 DRHP API ROUTES - MERCHANT BANKER INTERFACE
 * 
 * RESTful API endpoints for DRHP generation and management
 * Enterprise-grade routes with authentication and validation
 * 
 * @author Senior ASI Engineer (35+ years experience)
 * @version 1.0.0 - Production Ready
 */

const express = require('express');
const multer = require('multer');
const rateLimit = require('express-rate-limit');
const { body, param, query, validationResult } = require('express-validator');
const DRHPController = require('../controllers/drhpController');
const auth = require('../middleware/auth');
const logger = require('../utils/logger');

const router = express.Router();

// Configure multer for document uploads
const storage = multer.memoryStorage();
const upload = multer({
    storage: storage,
    limits: {
        fileSize: 50 * 1024 * 1024, // 50MB per file
        files: 20 // Maximum 20 files
    },
    fileFilter: (req, file, cb) => {
        // Allow specific document and image types
        const allowedTypes = [
            // Document types
            'application/pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/vnd.openxmlformats-officedocument.presentationml.presentation',
            'text/plain',
            'application/msword',
            'application/vnd.ms-excel',
            // Image types
            'image/jpeg',
            'image/jpg',
            'image/png',
            'image/bmp',
            'image/tiff',
            'image/webp',
            'image/gif'
        ];
        
        if (allowedTypes.includes(file.mimetype)) {
            cb(null, true);
        } else {
            cb(new Error(`Unsupported file type: ${file.mimetype}`), false);
        }
    }
});

// Rate limiting for DRHP generation
const drhpGenerationLimit = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 5, // Maximum 5 DRHP generations per hour per merchant banker
    message: {
        error: 'Too many DRHP generation requests. Please try again later.',
        retryAfter: '1 hour'
    },
    standardHeaders: true,
    legacyHeaders: false
});

// Rate limiting for general API calls
const generalLimit = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Maximum 100 requests per 15 minutes
    message: {
        error: 'Too many requests. Please try again later.'
    }
});

// Apply general rate limiting to all routes
router.use(generalLimit);

/**
 * @route   GET /api/drhp/workflows
 * @desc    Get available DRHP generation workflows
 * @access  Private (Merchant Banker)
 */
router.get('/workflows',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    DRHPController.getWorkflows
);

/**
 * @route   POST /api/drhp/generate
 * @desc    Generate DRHP document
 * @access  Private (Merchant Banker)
 */
router.post('/generate',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    drhpGenerationLimit,
    upload.array('documents', 20),
    [
        body('companyName')
            .notEmpty()
            .withMessage('Company name is required')
            .isLength({ min: 2, max: 200 })
            .withMessage('Company name must be between 2 and 200 characters'),
        
        body('industry')
            .notEmpty()
            .withMessage('Industry is required')
            .isLength({ min: 2, max: 100 })
            .withMessage('Industry must be between 2 and 100 characters'),
        
        body('incorporationDate')
            .isISO8601()
            .withMessage('Valid incorporation date is required'),
        
        body('registeredAddress')
            .notEmpty()
            .withMessage('Registered address is required')
            .isLength({ min: 10, max: 500 })
            .withMessage('Address must be between 10 and 500 characters'),
        
        body('businessDescription')
            .notEmpty()
            .withMessage('Business description is required')
            .isLength({ min: 50, max: 2000 })
            .withMessage('Business description must be between 50 and 2000 characters'),
        
        body('workflow')
            .optional()
            .isIn(['express', 'standard', 'comprehensive'])
            .withMessage('Invalid workflow type'),
        
        body('priority')
            .optional()
            .isIn(['low', 'medium', 'high', 'urgent'])
            .withMessage('Invalid priority level'),
        
        body('expectedCompletionDate')
            .optional()
            .isISO8601()
            .withMessage('Valid expected completion date required')
    ],
    DRHPController.generateDRHP
);

/**
 * @route   GET /api/drhp/session/:sessionId/status
 * @desc    Get DRHP generation session status
 * @access  Private (Merchant Banker)
 */
router.get('/session/:sessionId/status',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    [
        param('sessionId')
            .isUUID()
            .withMessage('Valid session ID is required')
    ],
    DRHPController.getSessionStatus
);

/**
 * @route   GET /api/drhp/sessions
 * @desc    Get active DRHP sessions for merchant banker
 * @access  Private (Merchant Banker)
 */
router.get('/sessions',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    [
        query('status')
            .optional()
            .isIn(['active', 'completed', 'error', 'all'])
            .withMessage('Invalid status filter'),
        
        query('limit')
            .optional()
            .isInt({ min: 1, max: 100 })
            .withMessage('Limit must be between 1 and 100'),
        
        query('offset')
            .optional()
            .isInt({ min: 0 })
            .withMessage('Offset must be non-negative')
    ],
    DRHPController.getSessions
);

/**
 * @route   GET /api/drhp/download/:sessionId
 * @desc    Download generated DRHP document
 * @access  Private (Merchant Banker)
 */
router.get('/download/:sessionId',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    [
        param('sessionId')
            .isUUID()
            .withMessage('Valid session ID is required'),
        
        query('format')
            .optional()
            .isIn(['pdf', 'summary', 'compliance', 'all'])
            .withMessage('Invalid download format')
    ],
    DRHPController.downloadDRHP
);

/**
 * @route   POST /api/drhp/validate-company
 * @desc    Validate company data before DRHP generation
 * @access  Private (Merchant Banker)
 */
router.post('/validate-company',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    [
        body('companyName')
            .notEmpty()
            .withMessage('Company name is required'),
        
        body('industry')
            .notEmpty()
            .withMessage('Industry is required'),
        
        body('incorporationDate')
            .isISO8601()
            .withMessage('Valid incorporation date is required')
    ],
    DRHPController.validateCompanyData
);

/**
 * @route   POST /api/drhp/session/:sessionId/cancel
 * @desc    Cancel DRHP generation session
 * @access  Private (Merchant Banker)
 */
router.post('/session/:sessionId/cancel',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    [
        param('sessionId')
            .isUUID()
            .withMessage('Valid session ID is required')
    ],
    DRHPController.cancelSession
);

/**
 * @route   GET /api/drhp/templates
 * @desc    Get DRHP document templates and samples
 * @access  Private (Merchant Banker)
 */
router.get('/templates',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    DRHPController.getTemplates
);

/**
 * @route   POST /api/drhp/feedback
 * @desc    Submit feedback on generated DRHP
 * @access  Private (Merchant Banker)
 */
router.post('/feedback',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    [
        body('sessionId')
            .isUUID()
            .withMessage('Valid session ID is required'),
        
        body('rating')
            .isInt({ min: 1, max: 5 })
            .withMessage('Rating must be between 1 and 5'),
        
        body('feedback')
            .optional()
            .isLength({ max: 1000 })
            .withMessage('Feedback must not exceed 1000 characters'),
        
        body('improvements')
            .optional()
            .isArray()
            .withMessage('Improvements must be an array')
    ],
    DRHPController.submitFeedback
);

/**
 * @route   GET /api/drhp/analytics
 * @desc    Get DRHP generation analytics for merchant banker
 * @access  Private (Merchant Banker)
 */
router.get('/analytics',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    [
        query('period')
            .optional()
            .isIn(['week', 'month', 'quarter', 'year'])
            .withMessage('Invalid period'),
        
        query('metrics')
            .optional()
            .isIn(['generation_count', 'success_rate', 'processing_time', 'quality_scores', 'all'])
            .withMessage('Invalid metrics filter')
    ],
    DRHPController.getAnalytics
);

/**
 * @route   GET /api/drhp/compliance-check/:sessionId
 * @desc    Get detailed compliance check results
 * @access  Private (Merchant Banker)
 */
router.get('/compliance-check/:sessionId',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    [
        param('sessionId')
            .isUUID()
            .withMessage('Valid session ID is required')
    ],
    DRHPController.getComplianceCheck
);

/**
 * @route   POST /api/drhp/research-supplement
 * @desc    Add supplementary research data to existing DRHP
 * @access  Private (Merchant Banker)
 */
router.post('/research-supplement',
    auth.requireAuth,
    auth.requireRole(['merchant_banker', 'admin']),
    upload.array('supplementaryDocuments', 10),
    [
        body('sessionId')
            .isUUID()
            .withMessage('Valid session ID is required'),
        
        body('researchType')
            .isIn(['market_analysis', 'competitor_analysis', 'regulatory_update', 'financial_update'])
            .withMessage('Invalid research type'),
        
        body('priority')
            .optional()
            .isIn(['low', 'medium', 'high'])
            .withMessage('Invalid priority level')
    ],
    DRHPController.addSupplementaryResearch
);

// Error handling middleware for multer
router.use((error, req, res, next) => {
    if (error instanceof multer.MulterError) {
        if (error.code === 'LIMIT_FILE_SIZE') {
            return res.status(400).json({
                success: false,
                error: 'File size too large. Maximum size is 50MB per file.',
                code: 'FILE_SIZE_LIMIT'
            });
        }
        
        if (error.code === 'LIMIT_FILE_COUNT') {
            return res.status(400).json({
                success: false,
                error: 'Too many files. Maximum 20 files allowed.',
                code: 'FILE_COUNT_LIMIT'
            });
        }
        
        return res.status(400).json({
            success: false,
            error: 'File upload error: ' + error.message,
            code: 'UPLOAD_ERROR'
        });
    }
    
    if (error.message.includes('Unsupported file type')) {
        return res.status(400).json({
            success: false,
            error: error.message,
            code: 'UNSUPPORTED_FILE_TYPE',
            supportedTypes: [
                'PDF', 'DOCX', 'XLSX', 'PPTX', 'TXT', 'DOC', 'XLS',
                'JPEG', 'JPG', 'PNG', 'BMP', 'TIFF', 'WEBP', 'GIF'
            ]
        });
    }
    
    next(error);
});

// Validation error handler
router.use((req, res, next) => {
    const errors = validationResult(req);
    if (!errors.isEmpty()) {
        return res.status(400).json({
            success: false,
            error: 'Validation failed',
            details: errors.array(),
            code: 'VALIDATION_ERROR'
        });
    }
    next();
});

// General error handler
router.use((error, req, res, next) => {
    logger.error('DRHP API Error:', error);
    
    res.status(500).json({
        success: false,
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong',
        code: 'INTERNAL_ERROR'
    });
});

module.exports = router;
