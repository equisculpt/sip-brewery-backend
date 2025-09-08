const express = require('express');
const router = express.Router();
const ComprehensiveReportService = require('../services/ComprehensiveReportService');
const authMiddleware = require('../middleware/authMiddleware');
const rateLimit = require('express-rate-limit');

// Rate limiting for report generation
const reportRateLimit = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 10, // Limit each IP to 10 report requests per windowMs
    message: 'Too many report requests, please try again later.'
});

const reportService = new ComprehensiveReportService();

// Apply rate limiting and authentication to all routes
router.use(reportRateLimit);
router.use(authMiddleware);

/**
 * @swagger
 * /api/reports/generate:
 *   post:
 *     summary: Generate comprehensive financial report
 *     tags: [Reports]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - reportType
 *               - userId
 *             properties:
 *               reportType:
 *                 type: string
 *                 enum: [client_statement, asi_diagnostic, portfolio_allocation, performance_benchmark, fy_pnl, elss_investment, top_performer, asset_allocation_trends, sip_flow_retention, campaign_performance, compliance_audit, commission_brokerage, custom_builder]
 *               userId:
 *                 type: string
 *               options:
 *                 type: object
 *                 properties:
 *                   startDate:
 *                     type: string
 *                     format: date
 *                   endDate:
 *                     type: string
 *                     format: date
 *                   format:
 *                     type: string
 *                     enum: [pdf, csv, excel]
 *                     default: pdf
 *                   financialYear:
 *                     type: string
 *                     example: "2023-24"
 *     responses:
 *       200:
 *         description: Report generated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 filename:
 *                   type: string
 *                 downloadUrl:
 *                   type: string
 *                 reportType:
 *                   type: string
 *                 generatedAt:
 *                   type: string
 *                   format: date-time
 */
router.post('/generate', async (req, res) => {
    try {
        const { reportType, userId, options = {} } = req.body;

        // Validate required fields
        if (!reportType || !userId) {
            return res.status(400).json({
                success: false,
                error: 'reportType and userId are required'
            });
        }

        // Set default options
        const reportOptions = {
            format: 'pdf',
            startDate: options.startDate || new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
            endDate: options.endDate || new Date().toISOString().split('T')[0],
            financialYear: options.financialYear || '2023-24',
            ...options
        };

        // Generate report
        const result = await reportService.generateReport(reportType, userId, reportOptions);

        res.json({
            success: true,
            filename: result.filename,
            downloadUrl: `/api/reports/download/${result.filename}`,
            reportType: reportType,
            generatedAt: new Date().toISOString(),
            options: reportOptions
        });

    } catch (error) {
        console.error('Report generation error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to generate report',
            details: error.message
        });
    }
});

/**
 * @swagger
 * /api/reports/types:
 *   get:
 *     summary: Get available report types
 *     tags: [Reports]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: List of available report types
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 reportTypes:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       id:
 *                         type: string
 *                       name:
 *                         type: string
 *                       description:
 *                         type: string
 *                       category:
 *                         type: string
 */
router.get('/types', (req, res) => {
    const reportTypes = [
        {
            id: 'client_statement',
            name: 'Client Investment Statement',
            description: 'Monthly/Quarterly portfolio statement with holdings and SIP history',
            category: 'Client Reports',
            emoji: '📊'
        },
        {
            id: 'asi_diagnostic',
            name: 'ASI Portfolio Diagnostic',
            description: 'AI-powered portfolio analysis with ASI score and recommendations',
            category: 'AI Analysis',
            emoji: '🧠'
        },
        {
            id: 'portfolio_allocation',
            name: 'Portfolio Allocation & Overlap',
            description: 'Asset allocation analysis with fund overlap detection',
            category: 'Portfolio Analysis',
            emoji: '📁'
        },
        {
            id: 'performance_benchmark',
            name: 'Performance vs Benchmark',
            description: 'Fund performance comparison with benchmark analysis',
            category: 'Performance',
            emoji: '📈'
        },
        {
            id: 'fy_pnl',
            name: 'Financial Year P&L',
            description: 'Annual profit & loss with tax implications (April-March)',
            category: 'Tax & Compliance',
            emoji: '📆'
        },
        {
            id: 'elss_investment',
            name: 'ELSS Investment Report',
            description: 'Tax-saving ELSS funds with lock-in periods and 80C utilization',
            category: 'Tax Planning',
            emoji: '💸'
        },
        {
            id: 'top_performer',
            name: 'Top Performer & Laggard',
            description: 'Best and worst performing funds with movement analysis',
            category: 'Performance',
            emoji: '🚀'
        },
        {
            id: 'asset_allocation_trends',
            name: 'Asset Allocation Trends',
            description: 'Investment flow analysis with forecasting',
            category: 'Analytics',
            emoji: '🏗️'
        },
        {
            id: 'sip_flow_retention',
            name: 'SIP Flow & Retention',
            description: 'SIP analysis with cohort retention metrics',
            category: 'SIP Analytics',
            emoji: '📉'
        },
        {
            id: 'campaign_performance',
            name: 'Campaign Performance',
            description: 'Marketing campaign ROI and conversion analysis',
            category: 'Marketing',
            emoji: '📣'
        },
        {
            id: 'compliance_audit',
            name: 'Compliance & Audit',
            description: 'KYC status and regulatory compliance report',
            category: 'Compliance',
            emoji: '📋'
        },
        {
            id: 'commission_brokerage',
            name: 'Commission & Brokerage',
            description: 'IFA earnings and trail commission report',
            category: 'Financial',
            emoji: '💼'
        },
        {
            id: 'custom_builder',
            name: 'Custom Report Builder',
            description: 'Dynamic report with custom filters and charts',
            category: 'Custom',
            emoji: '🔧'
        }
    ];

    res.json({
        success: true,
        reportTypes: reportTypes,
        totalCount: reportTypes.length
    });
});

/**
 * @swagger
 * /api/reports/download/{filename}:
 *   get:
 *     summary: Download generated report
 *     tags: [Reports]
 *     parameters:
 *       - in: path
 *         name: filename
 *         required: true
 *         schema:
 *           type: string
 *     responses:
 *       200:
 *         description: Report file
 *         content:
 *           application/pdf:
 *             schema:
 *               type: string
 *               format: binary
 */
router.get('/download/:filename', (req, res) => {
    try {
        const filename = req.params.filename;
        const filepath = path.join(__dirname, '../../reports', filename);

        // Security check - ensure filename doesn't contain path traversal
        if (filename.includes('..') || filename.includes('/') || filename.includes('\\')) {
            return res.status(400).json({
                success: false,
                error: 'Invalid filename'
            });
        }

        // Check if file exists
        if (!fs.existsSync(filepath)) {
            return res.status(404).json({
                success: false,
                error: 'Report not found'
            });
        }

        // Set appropriate headers
        res.setHeader('Content-Type', 'application/pdf');
        res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);

        // Stream the file
        const fileStream = fs.createReadStream(filepath);
        fileStream.pipe(res);

    } catch (error) {
        console.error('Download error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to download report'
        });
    }
});

/**
 * @swagger
 * /api/reports/history/{userId}:
 *   get:
 *     summary: Get report generation history for user
 *     tags: [Reports]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: userId
 *         required: true
 *         schema:
 *           type: string
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           default: 0
 *     responses:
 *       200:
 *         description: Report history
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                 reports:
 *                   type: array
 *                   items:
 *                     type: object
 *                     properties:
 *                       filename:
 *                         type: string
 *                       reportType:
 *                         type: string
 *                       generatedAt:
 *                         type: string
 *                         format: date-time
 *                       downloadUrl:
 *                         type: string
 */
router.get('/history/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const { limit = 20, offset = 0 } = req.query;

        // Mock history data - replace with actual database query
        const mockHistory = [
            {
                filename: 'Client_Statement_user123_1640995200000.pdf',
                reportType: 'client_statement',
                generatedAt: '2024-01-15T10:30:00Z',
                downloadUrl: '/api/reports/download/Client_Statement_user123_1640995200000.pdf'
            },
            {
                filename: 'ASI_Diagnostic_user123_1640908800000.pdf',
                reportType: 'asi_diagnostic',
                generatedAt: '2024-01-14T08:15:00Z',
                downloadUrl: '/api/reports/download/ASI_Diagnostic_user123_1640908800000.pdf'
            }
        ];

        res.json({
            success: true,
            reports: mockHistory.slice(offset, offset + limit),
            totalCount: mockHistory.length,
            hasMore: (offset + limit) < mockHistory.length
        });

    } catch (error) {
        console.error('History fetch error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to fetch report history'
        });
    }
});

/**
 * @swagger
 * /api/reports/bulk-generate:
 *   post:
 *     summary: Generate multiple reports in batch
 *     tags: [Reports]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - requests
 *             properties:
 *               requests:
 *                 type: array
 *                 items:
 *                   type: object
 *                   properties:
 *                     reportType:
 *                       type: string
 *                     userId:
 *                       type: string
 *                     options:
 *                       type: object
 *     responses:
 *       200:
 *         description: Batch report generation results
 */
router.post('/bulk-generate', async (req, res) => {
    try {
        const { requests } = req.body;

        if (!Array.isArray(requests) || requests.length === 0) {
            return res.status(400).json({
                success: false,
                error: 'requests array is required'
            });
        }

        // Limit batch size
        if (requests.length > 10) {
            return res.status(400).json({
                success: false,
                error: 'Maximum 10 reports can be generated in one batch'
            });
        }

        const results = [];
        
        for (const request of requests) {
            try {
                const result = await reportService.generateReport(
                    request.reportType,
                    request.userId,
                    request.options || {}
                );
                
                results.push({
                    success: true,
                    reportType: request.reportType,
                    userId: request.userId,
                    filename: result.filename,
                    downloadUrl: `/api/reports/download/${result.filename}`
                });
            } catch (error) {
                results.push({
                    success: false,
                    reportType: request.reportType,
                    userId: request.userId,
                    error: error.message
                });
            }
        }

        res.json({
            success: true,
            results: results,
            totalRequests: requests.length,
            successCount: results.filter(r => r.success).length,
            failureCount: results.filter(r => !r.success).length
        });

    } catch (error) {
        console.error('Bulk generation error:', error);
        res.status(500).json({
            success: false,
            error: 'Failed to process bulk report generation'
        });
    }
});

module.exports = router;
