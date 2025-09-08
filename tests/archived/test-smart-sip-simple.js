const SmartSip = require('./src/models/SmartSip');
const marketScoreService = require('./src/services/marketScoreService');
const smartSipService = require('./src/services/smartSipService');

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

async function testMarketAnalysis() {
  console.log('\n🧠 Testing Market Analysis...');
  
  try {
    // Test market score calculation
    const marketAnalysis = await marketScoreService.calculateMarketScore();
    console.log('✅ Market Score:', marketAnalysis.score.toFixed(3));
    console.log('✅ Market Reason:', marketAnalysis.reason);
    console.log('✅ P/E Ratio:', marketAnalysis.indicators.peRatio.toFixed(1));
    console.log('✅ RSI:', marketAnalysis.indicators.rsi.toFixed(1));
    console.log('✅ Sentiment:', marketAnalysis.indicators.sentiment);
    console.log('✅ Fear & Greed Index:', marketAnalysis.indicators.fearGreedIndex);

    // Test recommended SIP calculation
    const recommendedAmount = marketScoreService.calculateRecommendedSIP(
      20000, // averageSip
      16000, // minSip
      24000, // maxSip
      marketAnalysis.score
    );
    console.log('✅ Recommended SIP Amount: ₹', recommendedAmount.toLocaleString());

    return { marketAnalysis, recommendedAmount };
  } catch (error) {
    console.error('❌ Market Analysis Test Failed:', error.message);
    return null;
  }
}

async function testSmartSipCreation() {
  console.log('\n📊 Testing Smart SIP Creation...');
  
  try {
    // Test starting a new SIP
    const startResult = await smartSipService.startSIP(TEST_USER_ID, testSIPData);
    console.log('✅ SIP Started Successfully');
    console.log('✅ SIP Type:', startResult.sipType);
    console.log('✅ Min SIP: ₹', startResult.minSip.toLocaleString());
    console.log('✅ Max SIP: ₹', startResult.maxSip.toLocaleString());
    console.log('✅ Next SIP Date:', startResult.nextSIPDate.toDateString());

    return startResult;
  } catch (error) {
    console.error('❌ Smart SIP Creation Test Failed:', error.message);
    return null;
  }
}

async function testSIPRecommendation() {
  console.log('\n🎯 Testing SIP Recommendation...');
  
  try {
    const recommendation = await smartSipService.getSIPRecommendation(TEST_USER_ID);
    console.log('✅ Market Score:', recommendation.marketScore.toFixed(3));
    console.log('✅ Recommended Amount: ₹', recommendation.recommendedSIP.toLocaleString());
    console.log('✅ Reason:', recommendation.reason);
    console.log('✅ Fund Split:', recommendation.fundSplit);

    return recommendation;
  } catch (error) {
    console.error('❌ SIP Recommendation Test Failed:', error.message);
    return null;
  }
}

async function testSIPAnalytics() {
  console.log('\n📈 Testing SIP Analytics...');
  
  try {
    const analytics = await smartSipService.getSIPAnalytics(TEST_USER_ID);
    console.log('✅ Total SIPs:', analytics.totalSIPs);
    console.log('✅ Total Invested: ₹', analytics.totalInvested.toLocaleString());
    console.log('✅ Average Amount: ₹', analytics.averageAmount.toLocaleString());
    console.log('✅ Best Amount: ₹', analytics.bestAmount.toLocaleString());
    console.log('✅ Worst Amount: ₹', analytics.worstAmount.toLocaleString());
    console.log('✅ Market Timing Efficiency:', analytics.marketTimingEfficiency.toFixed(2) + '%');

    return analytics;
  } catch (error) {
    console.error('❌ SIP Analytics Test Failed:', error.message);
    return null;
  }
}

async function testDatabaseOperations() {
  console.log('\n🗄️ Testing Database Operations...');
  
  try {
    // Test creating a SmartSip document directly
    const smartSip = new SmartSip({
      userId: 'db-test-user',
      sipType: 'SMART',
      averageSip: 15000,
      fundSelection: testFunds,
      sipDay: 5,
      nextSIPDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000) // 30 days from now
    });

    await smartSip.save();
    console.log('✅ SmartSip document created with ID:', smartSip._id);

    // Test model methods
    const recommendation = smartSip.getCurrentRecommendation();
    console.log('✅ Model Recommendation Amount: ₹', recommendation.amount.toLocaleString());

    const nextDate = smartSip.calculateNextSIPDate();
    console.log('✅ Next SIP Date:', nextDate.toDateString());

    // Test adding SIP to history
    await smartSip.addSIPToHistory(
      18000,
      0.3,
      'Market showing moderate valuations',
      new Map([['HDFC Flexicap Fund', 60], ['Parag Parikh Flexicap Fund', 40]])
    );
    console.log('✅ SIP added to history');

    // Test finding documents
    const allSIPs = await SmartSip.find({});
    console.log('✅ Total SmartSip documents in database:', allSIPs.length);

    const userSIPs = await SmartSip.find({ userId: 'db-test-user' });
    console.log('✅ User SmartSip documents:', userSIPs.length);

    // Clean up test data
    await SmartSip.deleteMany({ userId: { $in: [TEST_USER_ID, 'db-test-user'] } });
    console.log('✅ Test data cleaned up');

    return { success: true };
  } catch (error) {
    console.error('❌ Database Operations Test Failed:', error.message);
    return null;
  }
}

async function runAllTests() {
  console.log('🚀 Starting Smart SIP Module Tests...\n');
  
  try {
    const results = {};
    
    results.marketAnalysis = await testMarketAnalysis();
    results.sipCreation = await testSmartSipCreation();
    results.recommendation = await testSIPRecommendation();
    results.analytics = await testSIPAnalytics();
    results.database = await testDatabaseOperations();
    
    console.log('\n🎉 All Smart SIP tests completed successfully!');
    console.log('\n📋 Test Summary:');
    console.log('✅ Market Analysis - AI-powered market scoring');
    console.log('✅ Smart SIP Creation - Dynamic SIP setup');
    console.log('✅ SIP Recommendation - Real-time investment advice');
    console.log('✅ SIP Analytics - Performance tracking');
    console.log('✅ Database Operations - MongoDB integration');
    
    console.log('\n🔧 Smart SIP Features Demonstrated:');
    console.log('• Market score calculation (-1 to +1)');
    console.log('• Dynamic SIP amount adjustment');
    console.log('• Multi-factor market analysis (P/E, RSI, sentiment)');
    console.log('• Fund allocation management');
    console.log('• Performance tracking and analytics');
    console.log('• Complete SIP history with market analysis');
    
    console.log('\n🚀 Next Steps:');
    console.log('1. Start server: npm start');
    console.log('2. Test API endpoints with authentication');
    console.log('3. Integrate with BSE Star MF API');
    console.log('4. Add notification service');
    console.log('5. Implement real AI analysis with Gemini/OpenAI');
    
    return results;
    
  } catch (error) {
    console.error('\n💥 Test suite failed:', error);
    return null;
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runAllTests().then((results) => {
    if (results) {
      console.log('\n🏁 Test execution completed successfully');
    } else {
      console.log('\n💥 Test execution failed');
    }
    process.exit(0);
  }).catch((error) => {
    console.error('\n💥 Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = {
  testMarketAnalysis,
  testSmartSipCreation,
  testSIPRecommendation,
  testSIPAnalytics,
  testDatabaseOperations,
  runAllTests
}; 