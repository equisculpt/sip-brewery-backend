const express = require('express');
const router = express.Router();
const ASIBlogGenerationEngine = require('../services/ASIBlogGenerationEngine');
const { PremiumBlogDatabaseService } = require('../services/PremiumBlogDatabaseService');
const AutomatedBlogScheduler = require('../services/AutomatedBlogScheduler');
const { authenticateToken } = require('../middleware/auth');
const { validateRequest } = require('../middleware/validation');
const { body, query, param } = require('express-validator');
const response = require('../utils/response');
const logger = require('../utils/logger');

// Initialize blog services
const blogEngine = new ASIBlogGenerationEngine();
const blogDatabase = new PremiumBlogDatabaseService();
const blogScheduler = new AutomatedBlogScheduler();

console.log('💎 PREMIUM BLOG API INITIALIZED');
console.log('📚 Goldman Sachs-quality content generation ready');
console.log('🤖 Automated daily blog publishing activated');
console.log('🎯 Target audience: Professional investors & fund managers');

/**
 * @route GET /api/blog/premium
 * @desc Get premium blogs for professional audience
 * @access Public
 */
router.get('/premium', [
  query('type').optional().isIn(['INVESTOR_EDUCATION', 'IPO_ANALYSIS', 'MUTUAL_FUND_ANALYSIS']),
  query('rating').optional().isIn(['SUPERIOR', 'EXCELLENT', 'GOOD']),
  query('limit').optional().isInt({ min: 1, max: 50 }),
  query('page').optional().isInt({ min: 1 }),
  validateRequest
], async (req, res) => {
  try {
    const { type, rating, limit = 20, page = 1 } = req.query;
    
    logger.info('📚 Fetching premium blogs for professional audience...');
    
    const filters = {
      type,
      goldman_sachs_rating: rating,
      limit: parseInt(limit),
      page: parseInt(page)
    };
    
    const blogs = await blogDatabase.getBlogsForProfessionals(filters);
    
    return response.success(res, 'Premium blogs retrieved successfully', {
      ...blogs,
      quality_assurance: 'Goldman Sachs-grade content',
      target_audience: 'Professional investors',
      content_features: [
        'ASI-powered insights',
        'Interactive charts',
        'Professional data tables',
        'Institutional-grade analysis'
      ],
      daily_generation: 'Automated 2 blogs per day',
      professional_optimization: true
    });
    
  } catch (error) {
    logger.error('Failed to fetch premium blogs:', error);
    return response.error(res, 'Failed to fetch premium blogs', error.message);
  }
});

/**
 * @route GET /api/blog/premium/:slug
 * @desc Get specific premium blog by slug
 * @access Public
 */
router.get('/premium/:slug', [
  param('slug').notEmpty().withMessage('Blog slug is required'),
  validateRequest
], async (req, res) => {
  try {
    const { slug } = req.params;
    const { track_engagement = true } = req.query;
    
    logger.info(`📖 Fetching premium blog: ${slug}`);
    
    const blog = await blogDatabase.getBlogBySlug(slug);
    
    if (!blog) {
      return response.error(res, 'Blog not found', 'The requested blog does not exist', 404);
    }
    
    // Track professional engagement
    if (track_engagement === 'true') {
      const engagementData = {
        is_professional: this.detectProfessionalUser(req),
        is_goldman_sachs: this.detectGoldmanSachsUser(req),
        user_agent: req.get('User-Agent'),
        ip: req.ip
      };
      
      await blogDatabase.trackProfessionalEngagement(blog.id, engagementData);
    }
    
    return response.success(res, 'Premium blog retrieved successfully', {
      blog,
      goldman_sachs_quality: true,
      professional_features: {
        downloadable_pdf: true,
        executive_summary: true,
        data_export: true,
        api_access: true
      },
      engagement_tracked: track_engagement === 'true'
    });
    
  } catch (error) {
    logger.error(`Failed to fetch blog ${req.params.slug}:`, error);
    return response.error(res, 'Failed to fetch blog', error.message);
  }
});

/**
 * @route POST /api/blog/generate
 * @desc Manually trigger blog generation
 * @access Private (Admin)
 */
router.post('/generate', [
  authenticateToken,
  body('type').isIn(['EDUCATIONAL', 'ANALYSIS']).withMessage('Valid blog type required'),
  body('priority').optional().isIn(['LOW', 'MEDIUM', 'HIGH', 'URGENT']),
  body('target_audience').optional().isIn(['RETAIL', 'PROFESSIONAL', 'INSTITUTIONAL']),
  validateRequest
], async (req, res) => {
  try {
    const { type, priority = 'MEDIUM', target_audience = 'PROFESSIONAL' } = req.body;
    
    logger.info(`🚀 Manual blog generation triggered: ${type}`);
    logger.info(`🎯 Priority: ${priority}, Audience: ${target_audience}`);
    
    const generationResult = await blogScheduler.generateBlogManually(type, {
      priority,
      target_audience,
      manual_trigger: true,
      triggered_by: req.user?.id || 'admin'
    });
    
    return response.success(res, 'Blog generated successfully', {
      ...generationResult,
      manual_generation: true,
      goldman_sachs_quality: true,
      professional_optimization: true,
      generation_priority: priority
    });
    
  } catch (error) {
    logger.error('Failed to generate blog manually:', error);
    return response.error(res, 'Blog generation failed', error.message);
  }
});

/**
 * @route GET /api/blog/analytics/daily
 * @desc Get daily blog analytics
 * @access Private (Admin)
 */
router.get('/analytics/daily', [
  authenticateToken,
  query('date').optional().isISO8601(),
  validateRequest
], async (req, res) => {
  try {
    const { date } = req.query;
    
    logger.info('📊 Generating daily blog analytics...');
    
    const analytics = await blogDatabase.generateDailyAnalytics(date);
    
    return response.success(res, 'Daily analytics generated successfully', {
      ...analytics,
      professional_metrics: {
        goldman_sachs_visits: analytics.professional_engagement?.goldman_sachs_visits || 0,
        professional_engagement_rate: analytics.professional_engagement?.professional_view_ratio || 0,
        superior_content_percentage: analytics.professional_engagement?.superior_content_ratio || 0
      },
      quality_assurance: 'Goldman Sachs-grade analytics',
      automated_generation: true
    });
    
  } catch (error) {
    logger.error('Failed to generate daily analytics:', error);
    return response.error(res, 'Analytics generation failed', error.message);
  }
});

/**
 * @route GET /api/blog/top-performing
 * @desc Get top performing blogs for professionals
 * @access Public
 */
router.get('/top-performing', [
  query('timeframe').optional().isIn(['7d', '30d', '90d', '1y']),
  query('metric').optional().isIn(['views', 'professional_views', 'goldman_sachs_visits', 'engagement']),
  validateRequest
], async (req, res) => {
  try {
    const { timeframe = '30d', metric = 'professional_views' } = req.query;
    
    logger.info(`🏆 Fetching top performing blogs: ${timeframe}, metric: ${metric}`);
    
    const topBlogs = await blogDatabase.getTopPerformingBlogs(timeframe, metric);
    
    return response.success(res, 'Top performing blogs retrieved successfully', {
      ...topBlogs,
      performance_metric: metric,
      timeframe,
      goldman_sachs_appeal: 'Maximum professional engagement',
      content_quality: 'Institutional-grade analysis'
    });
    
  } catch (error) {
    logger.error('Failed to fetch top performing blogs:', error);
    return response.error(res, 'Failed to fetch top performing blogs', error.message);
  }
});

/**
 * @route GET /api/blog/scheduler/status
 * @desc Get automated blog scheduler status
 * @access Private (Admin)
 */
router.get('/scheduler/status', [
  authenticateToken,
  validateRequest
], async (req, res) => {
  try {
    logger.info('🤖 Checking automated blog scheduler status...');
    
    const schedulerStatus = blogScheduler.getSchedulerStatus();
    
    return response.success(res, 'Scheduler status retrieved successfully', {
      ...schedulerStatus,
      automation_level: '$100 Million System',
      daily_blog_target: 2,
      quality_standard: 'Goldman Sachs Superior',
      professional_targeting: true
    });
    
  } catch (error) {
    logger.error('Failed to get scheduler status:', error);
    return response.error(res, 'Failed to get scheduler status', error.message);
  }
});

/**
 * @route POST /api/blog/scheduler/control
 * @desc Control automated blog scheduler
 * @access Private (Admin)
 */
router.post('/scheduler/control', [
  authenticateToken,
  body('action').isIn(['START', 'STOP', 'RESTART']).withMessage('Valid action required'),
  validateRequest
], async (req, res) => {
  try {
    const { action } = req.body;
    
    logger.info(`🎛️ Scheduler control action: ${action}`);
    
    let result;
    
    switch (action) {
      case 'START':
        result = await blogScheduler.startScheduler();
        break;
      case 'STOP':
        result = await blogScheduler.stopScheduler();
        break;
      case 'RESTART':
        await blogScheduler.stopScheduler();
        result = await blogScheduler.startScheduler();
        break;
    }
    
    return response.success(res, `Scheduler ${action.toLowerCase()} completed successfully`, {
      action,
      result,
      scheduler_status: 'CONTROLLED',
      automation_level: '$100 Million System'
    });
    
  } catch (error) {
    logger.error(`Failed to ${req.body.action} scheduler:`, error);
    return response.error(res, `Scheduler ${req.body.action} failed`, error.message);
  }
});

/**
 * @route GET /api/blog/upcoming-content
 * @desc Get upcoming content pipeline
 * @access Public
 */
router.get('/upcoming-content', async (req, res) => {
  try {
    logger.info('📅 Fetching upcoming content pipeline...');
    
    const upcomingContent = await blogEngine.getUpcomingContentPipeline();
    
    return response.success(res, 'Upcoming content pipeline retrieved successfully', {
      upcoming_content: upcomingContent,
      daily_generation: true,
      content_types: ['Investor Education', 'IPO Analysis', 'Mutual Fund Analysis'],
      quality_assurance: 'Goldman Sachs-grade content',
      automation_status: 'Fully automated',
      professional_targeting: true
    });
    
  } catch (error) {
    logger.error('Failed to fetch upcoming content:', error);
    return response.error(res, 'Failed to fetch upcoming content', error.message);
  }
});

/**
 * @route POST /api/blog/feedback
 * @desc Submit professional feedback on blog content
 * @access Public
 */
router.post('/feedback', [
  body('blog_id').notEmpty().withMessage('Blog ID is required'),
  body('rating').isInt({ min: 1, max: 5 }).withMessage('Rating must be between 1-5'),
  body('feedback_type').isIn(['QUALITY', 'ACCURACY', 'RELEVANCE', 'PRESENTATION']),
  body('professional_level').optional().isIn(['RETAIL', 'PROFESSIONAL', 'INSTITUTIONAL']),
  validateRequest
], async (req, res) => {
  try {
    const { blog_id, rating, feedback_type, professional_level, comments } = req.body;
    
    logger.info(`📝 Professional feedback received for blog ${blog_id}`);
    
    const feedbackResult = await blogDatabase.submitProfessionalFeedback({
      blog_id,
      rating,
      feedback_type,
      professional_level,
      comments,
      timestamp: new Date(),
      ip: req.ip,
      user_agent: req.get('User-Agent')
    });
    
    return response.success(res, 'Professional feedback submitted successfully', {
      feedback_id: feedbackResult.id,
      blog_id,
      professional_feedback: true,
      quality_improvement: 'Feedback will enhance future content',
      goldman_sachs_standards: 'Maintained through professional feedback'
    });
    
  } catch (error) {
    logger.error('Failed to submit professional feedback:', error);
    return response.error(res, 'Failed to submit feedback', error.message);
  }
});

// Helper methods
function detectProfessionalUser(req) {
  const userAgent = req.get('User-Agent') || '';
  const professionalIndicators = ['bloomberg', 'reuters', 'refinitiv', 'factset', 'morningstar'];
  return professionalIndicators.some(indicator => userAgent.toLowerCase().includes(indicator));
}

function detectGoldmanSachsUser(req) {
  const userAgent = req.get('User-Agent') || '';
  const ip = req.ip || '';
  
  // Check for Goldman Sachs indicators
  const gsIndicators = ['goldman', 'gs.com', 'goldmansachs'];
  return gsIndicators.some(indicator => 
    userAgent.toLowerCase().includes(indicator) || 
    ip.includes('goldman') // Simplified check
  );
}

module.exports = router;
