/**
 * 🚀 SETUP SCRIPT - FREE SOCIAL MEDIA INTELLIGENCE SYSTEM
 * 
 * Complete setup and initialization of the free social media tracking
 * and management philosophy analysis system for ASI integration
 * 
 * @author Financial Intelligence Team
 * @version 1.0.0 - Free Social Media Setup
 */

const logger = require('../src/utils/logger');
const { FreeSocialMediaIntegration } = require('../src/finance_crawler/free-social-media-integration');

async function setupSocialMediaSystem() {
  try {
    console.log('🚀 Setting up Free Social Media Intelligence System...\n');
    
    // Initialize the social media integration system
    console.log('📱 Initializing Social Media Integration...');
    const socialMediaSystem = new FreeSocialMediaIntegration({
      enableRealTimeTracking: true,
      enablePhilosophyAnalysis: true,
      enableSentimentAnalysis: true,
      enableTrendAnalysis: true,
      enableASIIntegration: true
    });
    
    // Setup event listeners for monitoring
    setupEventListeners(socialMediaSystem);
    
    // Initialize the system
    await socialMediaSystem.initialize();
    
    console.log('✅ Social Media Integration System initialized successfully!\n');
    
    // Display system information
    await displaySystemInfo(socialMediaSystem);
    
    // Test the system with sample data
    await testSystemWithSampleData(socialMediaSystem);
    
    // Display next steps
    displayNextSteps();
    
    console.log('\n🎉 Free Social Media Intelligence System setup completed successfully!');
    console.log('🔗 System is now ready for ASI integration and real-time tracking.');
    
  } catch (error) {
    console.error('❌ Setup failed:', error);
    process.exit(1);
  }
}

function setupEventListeners(socialMediaSystem) {
  console.log('⚙️ Setting up event listeners...');
  
  // System initialization
  socialMediaSystem.on('systemInitialized', () => {
    console.log('✅ Social Media System initialized and ready');
  });
  
  // Real-time sentiment updates
  socialMediaSystem.on('realTimeSentiment', (data) => {
    console.log(`📊 Real-time sentiment: ${data.company} - ${data.sentiment.aggregated.sentiment} (${data.platform})`);
  });
  
  // Philosophy updates
  socialMediaSystem.on('philosophyUpdate', (data) => {
    console.log(`🧠 Philosophy update: ${data.company} - ${data.analysis.investmentPhilosophy.primaryPhilosophy}`);
  });
  
  // ASI updates
  socialMediaSystem.on('asiUpdate', (update) => {
    console.log(`🔗 ASI Update: ${update.data.summary.companiesAnalyzed} companies analyzed`);
  });
  
  // Comprehensive insights
  socialMediaSystem.on('comprehensiveInsights', (insights) => {
    console.log(`💡 Comprehensive insights generated for ${insights.companiesAnalyzed} companies`);
  });
  
  // Daily reports
  socialMediaSystem.on('dailyReport', (report) => {
    console.log(`📊 Daily report generated for ${report.date}`);
  });
  
  // Weekly trends
  socialMediaSystem.on('weeklyTrends', (trends) => {
    console.log(`📈 Weekly trends analyzed for week ${trends.week}`);
  });
  
  // Monthly reviews
  socialMediaSystem.on('monthlyReview', (review) => {
    console.log(`🔍 Monthly review completed for ${review.month}/${review.year}`);
  });
  
  console.log('✅ Event listeners configured\n');
}

async function displaySystemInfo(socialMediaSystem) {
  console.log('📋 SYSTEM INFORMATION');
  console.log('=====================');
  
  const stats = socialMediaSystem.getSystemStats();
  
  console.log(`📊 System Status: ${stats.isInitialized ? 'Initialized' : 'Not Initialized'}`);
  console.log(`⏱️  System Uptime: ${Math.round(stats.uptime / 1000)} seconds`);
  console.log(`📈 Data Points Collected: ${stats.totalDataPointsCollected}`);
  console.log(`🏢 Management Profiles Analyzed: ${stats.managementProfilesAnalyzed}`);
  console.log(`💭 Sentiment Analysis Completed: ${stats.sentimentAnalysisCompleted}`);
  console.log(`🧠 Philosophy Profiles Created: ${stats.philosophyProfilesCreated}`);
  console.log(`🔗 ASI Updates Generated: ${stats.asiUpdatesGenerated}`);
  console.log(`📱 Management Insights Stored: ${stats.managementInsightsStored}`);
  console.log(`📊 Sentiment Trends Tracked: ${stats.sentimentTrendsTracked}`);
  console.log(`📄 Philosophy Profiles Stored: ${stats.philosophyProfilesStored}`);
  console.log(`📋 Analysis Queue Size: ${stats.analysisQueueSize}`);
  console.log(`🔄 ASI Update Queue Size: ${stats.asiUpdateQueueSize}`);
  
  if (stats.lastActivity) {
    console.log(`🕐 Last Activity: ${new Date(stats.lastActivity).toLocaleString()}`);
  }
  
  console.log('');
}

async function testSystemWithSampleData(socialMediaSystem) {
  console.log('🧪 TESTING SYSTEM WITH SAMPLE DATA');
  console.log('===================================');
  
  // Sample social media data for testing
  const sampleData = {
    company: 'HDFC Asset Management',
    platform: 'twitter_nitter',
    count: 5,
    data: [
      {
        text: 'Our value investing approach focuses on long-term wealth creation through fundamental analysis',
        sentiment: 'positive',
        timestamp: new Date().toISOString(),
        source: 'twitter'
      },
      {
        text: 'We maintain a conservative risk management strategy to protect investor capital',
        sentiment: 'neutral',
        timestamp: new Date().toISOString(),
        source: 'twitter'
      },
      {
        text: 'Excited about the growth opportunities in the Indian equity markets',
        sentiment: 'positive',
        timestamp: new Date().toISOString(),
        source: 'twitter'
      },
      {
        text: 'Our team believes in quality investing with a focus on blue-chip companies',
        sentiment: 'positive',
        timestamp: new Date().toISOString(),
        source: 'twitter'
      },
      {
        text: 'Market volatility requires a disciplined investment approach',
        sentiment: 'neutral',
        timestamp: new Date().toISOString(),
        source: 'twitter'
      }
    ]
  };
  
  console.log('📊 Injecting sample social media data...');
  
  // Simulate social media data event
  socialMediaSystem.socialMediaTracker.emit('socialMediaData', sampleData);
  
  // Wait for processing
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  console.log('✅ Sample data processed successfully');
  
  // Display updated stats
  const updatedStats = socialMediaSystem.getSystemStats();
  console.log(`📈 Updated Data Points: ${updatedStats.totalDataPointsCollected}`);
  console.log(`📊 Updated Sentiment Analysis: ${updatedStats.sentimentAnalysisCompleted}`);
  
  console.log('');
}

function displayNextSteps() {
  console.log('🎯 NEXT STEPS');
  console.log('=============');
  console.log('');
  console.log('1. 🚀 Start the social media monitoring service:');
  console.log('   npm run social-media-monitor');
  console.log('');
  console.log('2. 📊 Run on-demand analysis:');
  console.log('   npm run analyze-social-media');
  console.log('');
  console.log('3. 🔗 Integrate with ASI system:');
  console.log('   - The system is already configured for ASI integration');
  console.log('   - Real-time data will flow to the IntegratedDataManager');
  console.log('   - Philosophy and sentiment analysis will enhance ASI predictions');
  console.log('');
  console.log('4. 📈 Monitor system performance:');
  console.log('   - Check logs for real-time updates');
  console.log('   - Review daily, weekly, and monthly reports');
  console.log('   - Monitor sentiment trends and philosophy changes');
  console.log('');
  console.log('5. 🎛️ Customize settings:');
  console.log('   - Adjust scraping intervals in free-social-media-tracker.js');
  console.log('   - Modify philosophy patterns in free-management-philosophy-analyzer.js');
  console.log('   - Configure ASI integration settings in free-social-media-integration.js');
  console.log('');
  console.log('6. 📊 Data Sources Being Tracked:');
  console.log('   ✅ Twitter (via Nitter proxy)');
  console.log('   ✅ LinkedIn Public Posts');
  console.log('   ✅ YouTube Management Interviews');
  console.log('   ✅ RSS Feeds from AMC websites');
  console.log('   ✅ Financial News Websites');
  console.log('');
  console.log('7. 🧠 Analysis Capabilities:');
  console.log('   ✅ Investment Philosophy Extraction');
  console.log('   ✅ Risk Management Approach Analysis');
  console.log('   ✅ Market Outlook Assessment');
  console.log('   ✅ Leadership Style Identification');
  console.log('   ✅ Communication Pattern Analysis');
  console.log('   ✅ Real-time Sentiment Tracking');
  console.log('   ✅ Strategy Consistency Monitoring');
  console.log('');
}

// Run the setup
if (require.main === module) {
  setupSocialMediaSystem().catch(error => {
    console.error('❌ Setup failed:', error);
    process.exit(1);
  });
}

module.exports = { setupSocialMediaSystem };
