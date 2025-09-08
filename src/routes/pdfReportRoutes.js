// Professional PDF Report Generation Routes - Institutional Grade
const express = require('express');
const rateLimit = require('express-rate-limit');
const { body, query, validationResult } = require('express-validator');
const PDFReportService = require('../services/PDFReportService');
const authMiddleware = require('../middleware/authMiddleware'); // Assuming you have auth middleware

const router = express.Router();

// Rate limiting for PDF generation (resource intensive)
const pdfRateLimit = rateLimit({
    windowMs: 5 * 60 * 1000, // 5 minutes
    max: 10, // 10 PDF requests per 5 minutes
    message: {
        success: false,
        message: 'Too many PDF generation requests. Please try again later.'
    },
    standardHeaders: true,
    legacyHeaders: false
});

// Apply authentication to all PDF routes
router.use(authMiddleware);

// Generate Portfolio Statement
router.post('/generate/portfolio-statement',
    pdfRateLimit,
    [
        body('userId').isUUID().withMessage('Valid user ID required'),
        body('options.dateRange')
            .optional()
            .isIn(['1M', '3M', '6M', '1Y', 'YTD', 'ALL'])
            .withMessage('Invalid date range'),
        body('options.format')
            .optional()
            .isIn(['detailed', 'summary', 'regulatory'])
            .withMessage('Invalid format'),
        body('options.includeTransactions')
            .optional()
            .isBoolean()
            .withMessage('Include transactions must be boolean'),
        body('options.includeTaxDetails')
            .optional()
            .isBoolean()
            .withMessage('Include tax details must be boolean')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userId, options = {} } = req.body;
            
            // Generate PDF
            const pdfDoc = await PDFReportService.generatePortfolioStatement(userId, options);
            
            // Set response headers
            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', `attachment; filename="Portfolio_Statement_${new Date().toISOString().split('T')[0]}.pdf"`);
            
            // Pipe PDF to response
            pdfDoc.pipe(res);
            pdfDoc.end();

        } catch (error) {
            console.error('Portfolio Statement Generation Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to generate portfolio statement'
            });
        }
    }
);

// Generate Performance Analysis Report
router.post('/generate/performance-analysis',
    pdfRateLimit,
    [
        body('userId').isUUID().withMessage('Valid user ID required'),
        body('options.period')
            .optional()
            .isIn(['1M', '3M', '6M', '1Y', '3Y', '5Y', 'ALL'])
            .withMessage('Invalid period'),
        body('options.benchmarkComparison')
            .optional()
            .isBoolean()
            .withMessage('Benchmark comparison must be boolean'),
        body('options.riskMetrics')
            .optional()
            .isBoolean()
            .withMessage('Risk metrics must be boolean')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userId, options = {} } = req.body;
            
            const pdfDoc = await PDFReportService.generatePerformanceAnalysis(userId, options);
            
            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', `attachment; filename="Performance_Analysis_${new Date().toISOString().split('T')[0]}.pdf"`);
            
            pdfDoc.pipe(res);
            pdfDoc.end();

        } catch (error) {
            console.error('Performance Analysis Generation Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to generate performance analysis report'
            });
        }
    }
);

// Generate Tax Statement
router.post('/generate/tax-statement',
    pdfRateLimit,
    [
        body('userId').isUUID().withMessage('Valid user ID required'),
        body('financialYear')
            .matches(/^\d{4}-\d{4}$/)
            .withMessage('Financial year must be in format YYYY-YYYY (e.g., 2023-2024)')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userId, financialYear } = req.body;
            
            const pdfDoc = await PDFReportService.generateTaxStatement(userId, financialYear);
            
            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', `attachment; filename="Tax_Statement_FY_${financialYear}.pdf"`);
            
            pdfDoc.pipe(res);
            pdfDoc.end();

        } catch (error) {
            console.error('Tax Statement Generation Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to generate tax statement'
            });
        }
    }
);

// Generate Capital Gains Report
router.post('/generate/capital-gains',
    pdfRateLimit,
    [
        body('userId').isUUID().withMessage('Valid user ID required'),
        body('options.financialYear')
            .matches(/^\d{4}-\d{4}$/)
            .withMessage('Financial year must be in format YYYY-YYYY'),
        body('options.gainType')
            .optional()
            .isIn(['STCG', 'LTCG', 'ALL'])
            .withMessage('Gain type must be STCG, LTCG, or ALL'),
        body('options.includeUnrealized')
            .optional()
            .isBoolean()
            .withMessage('Include unrealized must be boolean')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userId, options = {} } = req.body;
            
            const pdfDoc = await PDFReportService.generateCapitalGainsReport(userId, options);
            
            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', `attachment; filename="Capital_Gains_Report_${options.financialYear}.pdf"`);
            
            pdfDoc.pipe(res);
            pdfDoc.end();

        } catch (error) {
            console.error('Capital Gains Report Generation Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to generate capital gains report'
            });
        }
    }
);

// Generate SIP Analysis Report
router.post('/generate/sip-analysis',
    pdfRateLimit,
    [
        body('userId').isUUID().withMessage('Valid user ID required'),
        body('options.period')
            .optional()
            .isIn(['1Y', '3Y', '5Y', 'ALL'])
            .withMessage('Invalid period'),
        body('options.includeFutureProjections')
            .optional()
            .isBoolean()
            .withMessage('Include future projections must be boolean'),
        body('options.includeOptimization')
            .optional()
            .isBoolean()
            .withMessage('Include optimization must be boolean')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userId, options = {} } = req.body;
            
            const pdfDoc = await PDFReportService.generateSIPAnalysisReport(userId, options);
            
            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', `attachment; filename="SIP_Analysis_Report_${new Date().toISOString().split('T')[0]}.pdf"`);
            
            pdfDoc.pipe(res);
            pdfDoc.end();

        } catch (error) {
            console.error('SIP Analysis Report Generation Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to generate SIP analysis report'
            });
        }
    }
);

// Generate Risk Analysis Report
router.post('/generate/risk-analysis',
    pdfRateLimit,
    [
        body('userId').isUUID().withMessage('Valid user ID required'),
        body('options.includeStressTest')
            .optional()
            .isBoolean()
            .withMessage('Include stress test must be boolean'),
        body('options.includeScenarioAnalysis')
            .optional()
            .isBoolean()
            .withMessage('Include scenario analysis must be boolean'),
        body('options.includeRiskRecommendations')
            .optional()
            .isBoolean()
            .withMessage('Include risk recommendations must be boolean')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userId, options = {} } = req.body;
            
            const pdfDoc = await PDFReportService.generateRiskAnalysisReport(userId, options);
            
            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', `attachment; filename="Risk_Analysis_Report_${new Date().toISOString().split('T')[0]}.pdf"`);
            
            pdfDoc.pipe(res);
            pdfDoc.end();

        } catch (error) {
            console.error('Risk Analysis Report Generation Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to generate risk analysis report'
            });
        }
    }
);

// Generate Annual Investment Report
router.post('/generate/annual-report',
    pdfRateLimit,
    [
        body('userId').isUUID().withMessage('Valid user ID required'),
        body('year')
            .isInt({ min: 2020, max: new Date().getFullYear() })
            .withMessage('Valid year required')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userId, year } = req.body;
            
            const pdfDoc = await PDFReportService.generateAnnualReport(userId, year);
            
            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', `attachment; filename="Annual_Investment_Report_${year}.pdf"`);
            
            pdfDoc.pipe(res);
            pdfDoc.end();

        } catch (error) {
            console.error('Annual Report Generation Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to generate annual report'
            });
        }
    }
);

// Get Available Report Types
router.get('/report-types', async (req, res) => {
    try {
        const reportTypes = {
            'portfolio-statement': {
                name: 'Portfolio Statement',
                description: 'Comprehensive portfolio overview with holdings, performance, and transactions',
                options: {
                    dateRange: ['1M', '3M', '6M', '1Y', 'YTD', 'ALL'],
                    format: ['detailed', 'summary', 'regulatory'],
                    includeTransactions: 'boolean',
                    includeTaxDetails: 'boolean',
                    includePerformance: 'boolean'
                }
            },
            'performance-analysis': {
                name: 'Performance Analysis Report',
                description: 'Detailed performance metrics, benchmark comparison, and risk analysis',
                options: {
                    period: ['1M', '3M', '6M', '1Y', '3Y', '5Y', 'ALL'],
                    benchmarkComparison: 'boolean',
                    riskMetrics: 'boolean',
                    attribution: 'boolean'
                }
            },
            'tax-statement': {
                name: 'Tax Statement',
                description: 'Tax implications, capital gains, dividend income, and ELSS investments',
                options: {
                    financialYear: 'YYYY-YYYY format (required)'
                }
            },
            'capital-gains': {
                name: 'Capital Gains Report',
                description: 'Short-term and long-term capital gains with tax implications',
                options: {
                    financialYear: 'YYYY-YYYY format (required)',
                    gainType: ['STCG', 'LTCG', 'ALL'],
                    includeUnrealized: 'boolean'
                }
            },
            'sip-analysis': {
                name: 'SIP Analysis Report',
                description: 'SIP performance, rupee cost averaging, and future projections',
                options: {
                    period: ['1Y', '3Y', '5Y', 'ALL'],
                    includeFutureProjections: 'boolean',
                    includeOptimization: 'boolean'
                }
            },
            'risk-analysis': {
                name: 'Risk Analysis Report',
                description: 'Portfolio risk metrics, volatility analysis, and stress testing',
                options: {
                    includeStressTest: 'boolean',
                    includeScenarioAnalysis: 'boolean',
                    includeRiskRecommendations: 'boolean'
                }
            },
            'annual-report': {
                name: 'Annual Investment Report',
                description: 'Comprehensive year-end review with performance highlights and outlook',
                options: {
                    year: 'number (required)'
                }
            }
        };

        res.json({
            success: true,
            data: reportTypes,
            message: 'Available report types retrieved successfully'
        });

    } catch (error) {
        console.error('Get Report Types Error:', error);
        res.status(500).json({
            success: false,
            message: 'Failed to retrieve report types'
        });
    }
});

// Get Report Generation History
router.get('/history/:userId',
    [
        query('limit')
            .optional()
            .isInt({ min: 1, max: 100 })
            .withMessage('Limit must be between 1 and 100'),
        query('offset')
            .optional()
            .isInt({ min: 0 })
            .withMessage('Offset must be non-negative')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userId } = req.params;
            const { limit = 20, offset = 0 } = req.query;

            // Mock implementation - replace with actual database query
            const history = [
                {
                    id: 'report-1',
                    type: 'portfolio-statement',
                    generatedAt: new Date().toISOString(),
                    options: { dateRange: 'YTD', format: 'detailed' },
                    status: 'completed',
                    downloadUrl: '/api/reports/download/report-1'
                },
                {
                    id: 'report-2',
                    type: 'performance-analysis',
                    generatedAt: new Date(Date.now() - 86400000).toISOString(),
                    options: { period: '1Y', benchmarkComparison: true },
                    status: 'completed',
                    downloadUrl: '/api/reports/download/report-2'
                }
            ];

            res.json({
                success: true,
                data: {
                    reports: history.slice(offset, offset + limit),
                    total: history.length,
                    limit: parseInt(limit),
                    offset: parseInt(offset)
                },
                message: 'Report history retrieved successfully'
            });

        } catch (error) {
            console.error('Get Report History Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to retrieve report history'
            });
        }
    }
);

// Bulk Report Generation (for multiple users - admin only)
router.post('/generate/bulk',
    pdfRateLimit,
    [
        body('userIds')
            .isArray({ min: 1, max: 100 })
            .withMessage('User IDs must be an array with 1-100 items'),
        body('userIds.*')
            .isUUID()
            .withMessage('Each user ID must be valid UUID'),
        body('reportType')
            .isIn(['portfolio-statement', 'performance-analysis', 'tax-statement'])
            .withMessage('Invalid report type for bulk generation'),
        body('options')
            .optional()
            .isObject()
            .withMessage('Options must be an object')
    ],
    async (req, res) => {
        try {
            const errors = validationResult(req);
            if (!errors.isEmpty()) {
                return res.status(400).json({
                    success: false,
                    message: 'Validation failed',
                    errors: errors.array()
                });
            }

            const { userIds, reportType, options = {} } = req.body;
            
            // Start bulk generation process (should be handled asynchronously in production)
            const bulkJobId = `bulk-${Date.now()}`;
            
            // In production, this would be queued for background processing
            setTimeout(async () => {
                for (const userId of userIds) {
                    try {
                        // Generate report for each user
                        console.log(`Generating ${reportType} for user ${userId}`);
                        // Implementation would save PDFs to storage and notify users
                    } catch (error) {
                        console.error(`Failed to generate report for user ${userId}:`, error);
                    }
                }
            }, 1000);

            res.json({
                success: true,
                data: {
                    bulkJobId,
                    userCount: userIds.length,
                    reportType,
                    status: 'queued'
                },
                message: 'Bulk report generation started'
            });

        } catch (error) {
            console.error('Bulk Report Generation Error:', error);
            res.status(500).json({
                success: false,
                message: 'Failed to start bulk report generation'
            });
        }
    }
);

// Health check for PDF service
router.get('/health', async (req, res) => {
    try {
        res.json({
            success: true,
            message: 'PDF Report Service is healthy',
            timestamp: new Date().toISOString(),
            availableReports: Object.keys(PDFReportService.reportTypes).length
        });
    } catch (error) {
        res.status(500).json({
            success: false,
            message: 'PDF Report Service health check failed'
        });
    }
});

module.exports = router;
