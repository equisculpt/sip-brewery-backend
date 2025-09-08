// ASI Portfolio Analysis Routes - API endpoints for AI-powered portfolio analysis
const express = require('express');
const rateLimit = require('express-rate-limit');
const ASIPortfolioAnalysisService = require('../services/ASIPortfolioAnalysisService');
const router = express.Router();

// Rate limiting for ASI analysis endpoints
const asiAnalysisLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // Limit each IP to 5 ASI analysis requests per windowMs
    message: {
        success: false,
        message: 'Too many ASI analysis requests, please try again later',
        error: 'RATE_LIMIT_EXCEEDED'
    },
    standardHeaders: true,
    legacyHeaders: false
});

// Input validation middleware
const validateASIAnalysisRequest = (req, res, next) => {
    const { userId } = req.params;
    const { 
        analysisDepth, 
        includeAIPredictions, 
        includeBehavioralAnalysis, 
        includeMarketSentiment,
        includeOptimizationSuggestions,
        timeHorizon 
    } = req.body;

    // Validate required parameters
    if (!userId) {
        return res.status(400).json({
            success: false,
            message: 'User ID is required',
            error: 'MISSING_USER_ID'
        });
    }

    // Validate optional parameters
    const validAnalysisDepths = ['basic', 'standard', 'comprehensive', 'institutional'];
    const validTimeHorizons = ['1Y', '3Y', '5Y', '10Y'];

    if (analysisDepth && !validAnalysisDepths.includes(analysisDepth)) {
        return res.status(400).json({
            success: false,
            message: 'Invalid analysis depth. Must be one of: ' + validAnalysisDepths.join(', '),
            error: 'INVALID_ANALYSIS_DEPTH'
        });
    }

    if (timeHorizon && !validTimeHorizons.includes(timeHorizon)) {
        return res.status(400).json({
            success: false,
            message: 'Invalid time horizon. Must be one of: ' + validTimeHorizons.join(', '),
            error: 'INVALID_TIME_HORIZON'
        });
    }

    next();
};

// Authentication middleware (mock implementation)
const authenticateUser = (req, res, next) => {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
        return res.status(401).json({
            success: false,
            message: 'Authentication required',
            error: 'UNAUTHORIZED'
        });
    }

    // Mock token validation - replace with actual JWT validation
    const token = authHeader.substring(7);
    if (token === 'invalid') {
        return res.status(401).json({
            success: false,
            message: 'Invalid authentication token',
            error: 'INVALID_TOKEN'
        });
    }

    // Add user info to request
    req.user = { id: req.params.userId || 'demo-user' };
    next();
};

// Audit logging middleware
const auditLogger = (req, res, next) => {
    const startTime = Date.now();
    
    res.on('finish', () => {
        const duration = Date.now() - startTime;
        console.log(`[ASI_AUDIT] ${new Date().toISOString()} - ${req.method} ${req.originalUrl} - Status: ${res.statusCode} - Duration: ${duration}ms - User: ${req.user?.id || 'anonymous'} - IP: ${req.ip}`);
    });
    
    next();
};

// Generate ASI Portfolio Analysis Report
router.post('/generate/:userId', 
    asiAnalysisLimiter,
    authenticateUser,
    validateASIAnalysisRequest,
    auditLogger,
    async (req, res) => {
        try {
            const { userId } = req.params;
            const options = req.body;

            console.log(`[ASI_SERVICE] Starting ASI portfolio analysis for user: ${userId}`);
            console.log(`[ASI_SERVICE] Analysis options:`, options);

            // Generate ASI analysis report
            const reportDoc = await ASIPortfolioAnalysisService.generateASIPortfolioAnalysis(userId, options);
            
            // Set response headers for PDF download
            res.setHeader('Content-Type', 'application/pdf');
            res.setHeader('Content-Disposition', `attachment; filename="ASI_Portfolio_Analysis_${userId}_${Date.now()}.pdf"`);
            res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
            res.setHeader('Pragma', 'no-cache');
            res.setHeader('Expires', '0');

            // Stream PDF to response
            reportDoc.pipe(res);
            reportDoc.end();

            console.log(`[ASI_SERVICE] ASI portfolio analysis completed successfully for user: ${userId}`);

        } catch (error) {
            console.error(`[ASI_SERVICE] Error generating ASI portfolio analysis:`, error);
            
            res.status(500).json({
                success: false,
                message: 'Failed to generate ASI portfolio analysis report',
                error: 'ANALYSIS_GENERATION_FAILED',
                details: process.env.NODE_ENV === 'development' ? error.message : undefined
            });
        }
    }
);

// Get ASI Analysis Preview (without generating full PDF)
router.get('/preview/:userId',
    asiAnalysisLimiter,
    authenticateUser,
    auditLogger,
    async (req, res) => {
        try {
            const { userId } = req.params;
            const { timeHorizon = '5Y' } = req.query;

            console.log(`[ASI_SERVICE] Generating ASI preview for user: ${userId}`);

            // Get portfolio data for preview
            const portfolioData = await ASIPortfolioAnalysisService.getPortfolioData(userId);
            const asiAnalysis = await ASIPortfolioAnalysisService.generateComprehensiveASIAnalysis(userId, portfolioData, { timeHorizon });

            // Return preview data
            res.json({
                success: true,
                message: 'ASI analysis preview generated successfully',
                data: {
                    overallASIScore: asiAnalysis.overallASIScore,
                    keyInsights: {
                        performanceAttribution: asiAnalysis.performanceAttribution.totalAttribution,
                        riskLevel: asiAnalysis.riskDecomposition.totalRisk,
                        expectedReturn: asiAnalysis.predictiveInsights.performanceForecast.oneYear.expectedReturn,
                        recommendationsCount: asiAnalysis.aiRecommendations.immediateActions.length
                    },
                    generatedAt: new Date().toISOString(),
                    analysisVersion: 'v2.1.0'
                }
            });

            console.log(`[ASI_SERVICE] ASI preview completed for user: ${userId}`);

        } catch (error) {
            console.error(`[ASI_SERVICE] Error generating ASI preview:`, error);
            
            res.status(500).json({
                success: false,
                message: 'Failed to generate ASI analysis preview',
                error: 'PREVIEW_GENERATION_FAILED',
                details: process.env.NODE_ENV === 'development' ? error.message : undefined
            });
        }
    }
);

// Get ASI Score Only (lightweight endpoint)
router.get('/score/:userId',
    rateLimit({
        windowMs: 5 * 60 * 1000, // 5 minutes
        max: 20, // Higher limit for score-only requests
        message: { success: false, message: 'Too many score requests' }
    }),
    authenticateUser,
    async (req, res) => {
        try {
            const { userId } = req.params;

            console.log(`[ASI_SERVICE] Calculating ASI score for user: ${userId}`);

            const portfolioData = await ASIPortfolioAnalysisService.getPortfolioData(userId);
            const asiScore = await ASIPortfolioAnalysisService.calculateOverallASIScore(portfolioData);

            res.json({
                success: true,
                message: 'ASI score calculated successfully',
                data: {
                    userId,
                    overallScore: asiScore.overallScore,
                    scoreInterpretation: asiScore.scoreInterpretation,
                    lastUpdated: new Date().toISOString()
                }
            });

        } catch (error) {
            console.error(`[ASI_SERVICE] Error calculating ASI score:`, error);
            
            res.status(500).json({
                success: false,
                message: 'Failed to calculate ASI score',
                error: 'SCORE_CALCULATION_FAILED'
            });
        }
    }
);

// Get ASI Analysis History
router.get('/history/:userId',
    authenticateUser,
    async (req, res) => {
        try {
            const { userId } = req.params;
            const { limit = 10, offset = 0 } = req.query;

            // Mock implementation - replace with actual database queries
            const history = [
                {
                    id: 'asi_001',
                    userId,
                    generatedAt: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
                    overallScore: 78,
                    analysisType: 'comprehensive',
                    status: 'completed'
                },
                {
                    id: 'asi_002',
                    userId,
                    generatedAt: new Date(Date.now() - 14 * 24 * 60 * 60 * 1000).toISOString(),
                    overallScore: 75,
                    analysisType: 'standard',
                    status: 'completed'
                }
            ];

            res.json({
                success: true,
                message: 'ASI analysis history retrieved successfully',
                data: {
                    history: history.slice(offset, offset + parseInt(limit)),
                    total: history.length,
                    pagination: {
                        limit: parseInt(limit),
                        offset: parseInt(offset),
                        hasMore: (offset + parseInt(limit)) < history.length
                    }
                }
            });

        } catch (error) {
            console.error(`[ASI_SERVICE] Error retrieving ASI history:`, error);
            
            res.status(500).json({
                success: false,
                message: 'Failed to retrieve ASI analysis history',
                error: 'HISTORY_RETRIEVAL_FAILED'
            });
        }
    }
);

// ASI Service Health Check
router.get('/health', (req, res) => {
    res.json({
        success: true,
        message: 'ASI Portfolio Analysis Service is operational',
        data: {
            service: 'ASI Portfolio Analysis',
            version: 'v2.1.0',
            status: 'healthy',
            timestamp: new Date().toISOString(),
            features: {
                performanceAttribution: true,
                riskDecomposition: true,
                aiPredictions: true,
                behavioralAnalysis: true,
                portfolioOptimization: true
            }
        }
    });
});

// ASI Model Information
router.get('/models', (req, res) => {
    res.json({
        success: true,
        message: 'ASI model information retrieved successfully',
        data: {
            models: {
                performanceAttribution: {
                    name: 'Multi-Factor Attribution Model',
                    version: 'v2.1',
                    accuracy: 0.89,
                    lastTrained: '2024-01-15'
                },
                riskDecomposition: {
                    name: 'Factor Risk Model',
                    version: 'v1.8',
                    accuracy: 0.92,
                    lastTrained: '2024-01-10'
                },
                predictiveModeling: {
                    name: 'LSTM Performance Predictor',
                    version: 'v3.0',
                    accuracy: 0.76,
                    lastTrained: '2024-01-20'
                }
            },
            totalModels: 3,
            lastModelUpdate: '2024-01-20'
        }
    });
});

// Error handling middleware
router.use((error, req, res, next) => {
    console.error(`[ASI_SERVICE] Unhandled error:`, error);
    
    res.status(500).json({
        success: false,
        message: 'An unexpected error occurred in ASI service',
        error: 'INTERNAL_SERVER_ERROR',
        timestamp: new Date().toISOString()
    });
});

module.exports = router;
