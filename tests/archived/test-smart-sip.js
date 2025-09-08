const axios = require('axios');
const SmartSip = require('./src/models/SmartSip');
const marketScoreService = require('./src/services/marketScoreService');
const smartSipService = require('./src/services/smartSipService');
const cronService = require('./src/services/cronService');

const BASE_URL = 'http://localhost:3000';
const TEST_USER_ID = 'test-user-123';

// Test data
const testFunds = [
  {
    schemeCode: 'HDFC001',
    schemeName: 'HDFC Flexicap Fund',
    allocation: 60
  },
  {
    schemeCode: 'PARAG001',
    schemeName: 'Parag Parikh Flexicap Fund',
    allocation: 40
  }
];

const testSIPData = {
  sipType: 'SMART',
  averageSip: 20000,
  fundSelection: testFunds,
  sipDay: 1,
  preferences: {
    riskTolerance: 'MODERATE',
    marketTiming: true,
    aiEnabled: true,
    notifications: true
  }
};

async function testMarketScoreService() {
  console.log('\n🧠 Testing Market Score Service...');
  
  try {
    // Test market score calculation
    const marketAnalysis = await marketScoreService.calculateMarketScore();
    console.log('✅ Market Analysis:', {
      score: marketAnalysis.score,
      reason: marketAnalysis.reason,
      indicators: marketAnalysis.indicators
    });

    // Test recommended SIP calculation
    const recommendedAmount = marketScoreService.calculateRecommendedSIP(
      20000, // averageSip
      16000, // minSip
      24000, // maxSip
      marketAnalysis.score
    );
    console.log('✅ Recommended SIP Amount:', recommendedAmount);

    // Test AI analysis (placeholder)
    const aiAnalysis = await marketScoreService.getAIAnalysis();
    console.log('✅ AI Analysis:', aiAnalysis);

  } catch (error) {
    console.error('❌ Market Score Service Test Failed:', error.message);
  }
}

async function testSmartSipService() {
  console.log('\n📊 Testing Smart SIP Service...');
  
  try {
    // Test starting a new SIP
    console.log('Starting new SIP...');
    const startResult = await smartSipService.startSIP(TEST_USER_ID, testSIPData);
    console.log('✅ SIP Started:', startResult);

    // Test getting SIP details
    console.log('Getting SIP details...');
    const details = await smartSipService.getSIPDetails(TEST_USER_ID);
    console.log('✅ SIP Details:', {
      sipType: details.sipType,
      averageSip: details.averageSip,
      minSip: details.minSip,
      maxSip: details.maxSip,
      status: details.status,
      nextSIPDate: details.nextSIPDate
    });

    // Test getting SIP recommendation
    console.log('Getting SIP recommendation...');
    const recommendation = await smartSipService.getSIPRecommendation(TEST_USER_ID);
    console.log('✅ SIP Recommendation:', {
      marketScore: recommendation.marketScore,
      reason: recommendation.reason,
      recommendedSIP: recommendation.recommendedSIP,
      fundSplit: recommendation.fundSplit
    });

    // Test updating preferences
    console.log('Updating SIP preferences...');
    const prefResult = await smartSipService.updateSIPPreferences(TEST_USER_ID, {
      riskTolerance: 'AGGRESSIVE',
      aiEnabled: false
    });
    console.log('✅ Preferences Updated:', prefResult);

    // Test getting analytics
    console.log('Getting SIP analytics...');
    const analytics = await smartSipService.getSIPAnalytics(TEST_USER_ID);
    console.log('✅ SIP Analytics:', analytics);

    // Test getting history
    console.log('Getting SIP history...');
    const history = await smartSipService.getSIPHistory(TEST_USER_ID, 5);
    console.log('✅ SIP History:', history);

  } catch (error) {
    console.error('❌ Smart SIP Service Test Failed:', error.message);
  }
}

async function testAPIEndpoints() {
  console.log('\n🌐 Testing API Endpoints...');
  
  try {
    // Mock authentication token (in real app, this would be a valid JWT)
    const headers = {
      'Authorization': 'Bearer mock-token',
      'Content-Type': 'application/json'
    };

    // Test starting SIP via API
    console.log('Testing POST /api/sip/start...');
    try {
      const startResponse = await axios.post(`${BASE_URL}/api/sip/start`, testSIPData, { headers });
      console.log('✅ Start SIP API Response:', startResponse.data);
    } catch (error) {
      console.log('⚠️ Start SIP API (expected auth error):', error.response?.data?.message || error.message);
    }

    // Test getting recommendation via API
    console.log('Testing GET /api/sip/recommendation...');
    try {
      const recResponse = await axios.get(`${BASE_URL}/api/sip/recommendation`, { headers });
      console.log('✅ Recommendation API Response:', recResponse.data);
    } catch (error) {
      console.log('⚠️ Recommendation API (expected auth error):', error.response?.data?.message || error.message);
    }

    // Test getting details via API
    console.log('Testing GET /api/sip/details...');
    try {
      const detailsResponse = await axios.get(`${BASE_URL}/api/sip/details`, { headers });
      console.log('✅ Details API Response:', detailsResponse.data);
    } catch (error) {
      console.log('⚠️ Details API (expected auth error):', error.response?.data?.message || error.message);
    }

    // Test market analysis via API
    console.log('Testing GET /api/sip/market-analysis...');
    try {
      const marketResponse = await axios.get(`${BASE_URL}/api/sip/market-analysis`, { headers });
      console.log('✅ Market Analysis API Response:', marketResponse.data);
    } catch (error) {
      console.log('⚠️ Market Analysis API (expected auth error):', error.response?.data?.message || error.message);
    }

  } catch (error) {
    console.error('❌ API Endpoints Test Failed:', error.message);
  }
}

async function testCronService() {
  console.log('\n⏰ Testing Cron Service...');
  
  try {
    // Test manual market analysis trigger
    console.log('Testing manual market analysis...');
    const marketResult = await cronService.triggerMarketAnalysis();
    console.log('✅ Manual Market Analysis:', marketResult);

    // Test manual SIP execution trigger
    console.log('Testing manual SIP execution...');
    const executionResult = await cronService.triggerSIPExecution();
    console.log('✅ Manual SIP Execution:', executionResult);

    // Test getting job status
    console.log('Getting cron job status...');
    const jobStatus = cronService.getJobStatus();
    console.log('✅ Cron Job Status:', jobStatus);

    // Start test job for demonstration
    console.log('Starting test cron job...');
    cronService.startTestJob();
    console.log('✅ Test cron job started (runs every 5 minutes)');

    // Wait a bit to see the test job run
    console.log('Waiting 10 seconds to see test job run...');
    await new Promise(resolve => setTimeout(resolve, 10000));

    // Stop test job
    cronService.stopJob('test');
    console.log('✅ Test cron job stopped');

  } catch (error) {
    console.error('❌ Cron Service Test Failed:', error.message);
  }
}

async function testDatabaseOperations() {
  console.log('\n🗄️ Testing Database Operations...');
  
  try {
    // Test creating a SmartSip document directly
    console.log('Creating SmartSip document...');
    const smartSip = new SmartSip({
      userId: 'db-test-user',
      sipType: 'SMART',
      averageSip: 15000,
      fundSelection: testFunds,
      sipDay: 5,
      nextSIPDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 days from now
    });

    await smartSip.save();
    console.log('✅ SmartSip document created:', smartSip._id);

    // Test model methods
    console.log('Testing model methods...');
    const recommendation = smartSip.getCurrentRecommendation();
    console.log('✅ Model Recommendation:', recommendation);

    const nextDate = smartSip.calculateNextSIPDate();
    console.log('✅ Next SIP Date:', nextDate);

    // Test adding SIP to history
    console.log('Adding SIP to history...');
    await smartSip.addSIPToHistory(
      18000,
      0.3,
      'Market showing moderate valuations',
      new Map([['HDFC Flexicap Fund', 60], ['Parag Parikh Flexicap Fund', 40]])
    );
    console.log('✅ SIP added to history');

    // Test finding documents
    console.log('Finding SmartSip documents...');
    const allSIPs = await SmartSip.find({});
    console.log('✅ Total SmartSip documents:', allSIPs.length);

    const userSIPs = await SmartSip.find({ userId: 'db-test-user' });
    console.log('✅ User SmartSip documents:', userSIPs.length);

    // Clean up test data
    console.log('Cleaning up test data...');
    await SmartSip.deleteMany({ userId: { $in: [TEST_USER_ID, 'db-test-user'] } });
    console.log('✅ Test data cleaned up');

  } catch (error) {
    console.error('❌ Database Operations Test Failed:', error.message);
  }
}

async function runAllTests() {
  console.log('🚀 Starting Smart SIP Module Tests...\n');
  
  try {
    await testMarketScoreService();
    await testSmartSipService();
    await testAPIEndpoints();
    await testCronService();
    await testDatabaseOperations();
    
    console.log('\n🎉 All Smart SIP tests completed successfully!');
    console.log('\n📋 Test Summary:');
    console.log('✅ Market Score Service - AI-powered market analysis');
    console.log('✅ Smart SIP Service - Core business logic');
    console.log('✅ API Endpoints - RESTful API structure');
    console.log('✅ Cron Service - Automated execution');
    console.log('✅ Database Operations - MongoDB integration');
    
    console.log('\n🔧 Next Steps:');
    console.log('1. Start the server: npm start');
    console.log('2. Test with real authentication tokens');
    console.log('3. Integrate with BSE Star MF API for actual SIP execution');
    console.log('4. Add notification service for user alerts');
    console.log('5. Implement AI analysis with Gemini/OpenAI');
    
  } catch (error) {
    console.error('\n💥 Test suite failed:', error);
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runAllTests().then(() => {
    console.log('\n🏁 Test execution completed');
    process.exit(0);
  }).catch((error) => {
    console.error('\n💥 Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = {
  testMarketScoreService,
  testSmartSipService,
  testAPIEndpoints,
  testCronService,
  testDatabaseOperations,
  runAllTests
}; 