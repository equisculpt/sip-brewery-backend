const express = require('express');
const router = express.Router();

// Import route modules
const aiRoutes = require('./ai');
const analyticsRoutes = require('./analytics');
const benchmarkRoutes = require('./benchmarkRoutes');
const ollamaRoutes = require('./ollama');
const agiRoutes = require('./agi');
const voiceBotRoutes = require('./voiceBot');
const marketAnalyticsRoutes = require('./marketAnalytics');
const roboAdvisorRoutes = require('./roboAdvisor');
const complianceRoutes = require('./compliance');
const regionalLanguageRoutes = require('./regionalLanguage');
const tierOutreachRoutes = require('./tierOutreach');
const socialInvestingRoutes = require('./socialInvesting');
const learningRoutes = require('./learning');
const bseStarMFRoutes = require('./bseStarMF');
const digioRoutes = require('./digio');
// const authRoutes = require('./auth');
// const userRoutes = require('./users');
// const productRoutes = require('./products');

// Define routes
router.use('/ai', aiRoutes);
router.use('/analytics', analyticsRoutes);
router.use('/benchmark', benchmarkRoutes);
router.use('/ollama', ollamaRoutes);
router.use('/agi', agiRoutes);
router.use('/voice-bot', voiceBotRoutes);
router.use('/market-analytics', marketAnalyticsRoutes);
router.use('/robo-advisor', roboAdvisorRoutes);
router.use('/compliance', complianceRoutes);
router.use('/regional', regionalLanguageRoutes);
router.use('/tier', tierOutreachRoutes);
router.use('/social', socialInvestingRoutes);
router.use('/learning', learningRoutes);
router.use('/bse-star-mf', bseStarMFRoutes);
router.use('/digio', digioRoutes);
// router.use('/auth', authRoutes);
// router.use('/users', userRoutes);
// router.use('/products', productRoutes);

// Basic route for testing
router.get('/test', (req, res) => {
  res.json({
    message: 'Routes are working correctly',
    timestamp: new Date().toISOString()
  });
});

module.exports = router; 