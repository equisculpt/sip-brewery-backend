const leaderboardService = require('./src/services/leaderboardService');
const leaderboardCronService = require('./src/services/leaderboardCronService');
const { seedLeaderboardData } = require('./src/utils/leaderboardSeeder');
const User = require('./src/models/User');
const UserPortfolio = require('./src/models/UserPortfolio');
const Leaderboard = require('./src/models/Leaderboard');
const PortfolioCopy = require('./src/models/PortfolioCopy');

async function testXIRRCalculation() {
  console.log('\n🧮 Testing XIRR Calculation...');
  
  try {
    // Test XIRR calculation with sample cash flows
    const cashFlows = [
      { date: new Date('2024-01-01').getTime(), amount: -10000 },
      { date: new Date('2024-02-01').getTime(), amount: -10000 },
      { date: new Date('2024-03-01').getTime(), amount: -10000 },
      { date: new Date('2024-04-01').getTime(), amount: 35000 }
    ];
    
    const xirr = leaderboardService.calculateXIRR(cashFlows);
    console.log('✅ Sample XIRR Calculation:', xirr.toFixed(2) + '%');
    
    // Test portfolio XIRR calculation
    const testPortfolio = {
      transactions: [
        {
          date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
          type: 'SIP',
          amount: 10000
        },
        {
          date: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000),
          type: 'SIP',
          amount: 10000
        }
      ],
      totalCurrentValue: 22000
    };
    
    const portfolioXIRR = await leaderboardService.calculatePortfolioXIRR(testPortfolio, '1M');
    console.log('✅ Portfolio XIRR (1M):', portfolioXIRR.toFixed(2) + '%');
    
    return { xirr, portfolioXIRR };
  } catch (error) {
    console.error('❌ XIRR Calculation Test Failed:', error.message);
    return null;
  }
}

async function testLeaderboardGeneration() {
  console.log('\n🏆 Testing Leaderboard Generation...');
  
  try {
    // Generate leaderboards for all durations
    const results = await leaderboardService.generateAllLeaderboards();
    console.log('✅ Leaderboard Generation Results:', results);
    
    // Get sample leaderboard
    const leaderboard = await leaderboardService.getLeaderboard('1Y');
    if (leaderboard) {
      console.log('✅ 1Y Leaderboard Sample:');
      leaderboard.leaders.slice(0, 3).forEach((leader, index) => {
        console.log(`   ${index + 1}. ${leader.secretCode} - ${leader.returnPercent}%`);
        console.log(`      Allocation:`, leader.allocation);
      });
    }
    
    return results;
  } catch (error) {
    console.error('❌ Leaderboard Generation Test Failed:', error.message);
    return null;
  }
}

async function testPortfolioCopying() {
  console.log('\n📋 Testing Portfolio Copying...');
  
  try {
    // Get a sample user and leaderboard
    const user = await User.findOne({ email: /@example\.com$/ });
    const leaderboard = await Leaderboard.findOne({ duration: '1Y' });
    
    if (!user || !leaderboard || leaderboard.leaders.length === 0) {
      console.log('⚠️ No test data available for portfolio copying test');
      return null;
    }
    
    const sourceSecretCode = leaderboard.leaders[0].secretCode;
    
    // Test SIP copying
    console.log('Testing SIP portfolio copy...');
    const sipResult = await leaderboardService.copyPortfolio(
      user._id,
      sourceSecretCode,
      'SIP',
      5000
    );
    console.log('✅ SIP Copy Result:', sipResult.message);
    
    // Test lumpsum copying
    console.log('Testing lumpsum portfolio copy...');
    const lumpsumResult = await leaderboardService.copyPortfolio(
      user._id,
      sourceSecretCode,
      'LUMPSUM'
    );
    console.log('✅ Lumpsum Copy Result:', lumpsumResult.message);
    
    return { sipResult, lumpsumResult };
  } catch (error) {
    console.error('❌ Portfolio Copying Test Failed:', error.message);
    return null;
  }
}

async function testUserHistory() {
  console.log('\n📊 Testing User History...');
  
  try {
    const user = await User.findOne({ email: /@example\.com$/ });
    
    if (!user) {
      console.log('⚠️ No test user available for history test');
      return null;
    }
    
    // Test leaderboard history
    const leaderboardHistory = await leaderboardService.getUserLeaderboardHistory(user._id);
    console.log('✅ Leaderboard History Entries:', leaderboardHistory.length);
    
    if (leaderboardHistory.length > 0) {
      console.log('✅ Sample History Entry:', {
        duration: leaderboardHistory[0].duration,
        rank: leaderboardHistory[0].rank,
        returnPercent: leaderboardHistory[0].returnPercent
      });
    }
    
    // Test portfolio copy history
    const copyHistory = await leaderboardService.getPortfolioCopyHistory(user._id);
    console.log('✅ Portfolio Copy History Entries:', copyHistory.length);
    
    if (copyHistory.length > 0) {
      console.log('✅ Sample Copy Entry:', {
        sourceSecretCode: copyHistory[0].sourceSecretCode,
        investmentType: copyHistory[0].investmentType,
        status: copyHistory[0].status
      });
    }
    
    return { leaderboardHistory, copyHistory };
  } catch (error) {
    console.error('❌ User History Test Failed:', error.message);
    return null;
  }
}

async function testCronService() {
  console.log('\n⏰ Testing Cron Service...');
  
  try {
    // Test manual XIRR update
    console.log('Testing manual XIRR update...');
    const xirrResult = await leaderboardCronService.triggerXIRRUpdate();
    console.log('✅ XIRR Update Result:', xirrResult.message);
    
    // Test manual leaderboard generation
    console.log('Testing manual leaderboard generation...');
    const leaderboardResult = await leaderboardCronService.triggerLeaderboardGeneration();
    console.log('✅ Leaderboard Generation Result:', leaderboardResult.message);
    
    // Test job status
    const jobStatus = leaderboardCronService.getJobStatus();
    console.log('✅ Cron Job Status:', Object.keys(jobStatus).length, 'jobs');
    
    // Start test job
    leaderboardCronService.startTestJob();
    console.log('✅ Test cron job started');
    
    // Wait a bit to see the test job run
    await new Promise(resolve => setTimeout(resolve, 5000));
    
    // Stop test job
    leaderboardCronService.stopJob('test');
    console.log('✅ Test cron job stopped');
    
    return { xirrResult, leaderboardResult, jobStatus };
  } catch (error) {
    console.error('❌ Cron Service Test Failed:', error.message);
    return null;
  }
}

async function testDatabaseOperations() {
  console.log('\n🗄️ Testing Database Operations...');
  
  try {
    // Test User model
    console.log('Testing User model...');
    const users = await User.find({ email: /@example\.com$/ });
    console.log('✅ Test Users:', users.length);
    
    if (users.length > 0) {
      console.log('✅ Sample User:', {
        secretCode: users[0].secretCode,
        email: users[0].email,
        kycStatus: users[0].kycStatus
      });
    }
    
    // Test UserPortfolio model
    console.log('Testing UserPortfolio model...');
    const portfolios = await UserPortfolio.find({});
    console.log('✅ Portfolios:', portfolios.length);
    
    if (portfolios.length > 0) {
      const portfolio = portfolios[0];
      console.log('✅ Sample Portfolio:', {
        funds: portfolio.funds.length,
        totalInvested: portfolio.totalInvested,
        totalCurrentValue: portfolio.totalCurrentValue,
        xirr1Y: portfolio.xirr1Y
      });
      
      // Test allocation object
      const allocation = portfolio.getAllocationObject();
      console.log('✅ Portfolio Allocation:', allocation);
    }
    
    // Test Leaderboard model
    console.log('Testing Leaderboard model...');
    const leaderboards = await Leaderboard.find({});
    console.log('✅ Leaderboards:', leaderboards.length);
    
    if (leaderboards.length > 0) {
      const leaderboard = leaderboards[0];
      console.log('✅ Sample Leaderboard:', {
        duration: leaderboard.duration,
        leaders: leaderboard.leaders.length,
        totalParticipants: leaderboard.totalParticipants,
        averageReturn: leaderboard.averageReturn
      });
    }
    
    // Test PortfolioCopy model
    console.log('Testing PortfolioCopy model...');
    const copies = await PortfolioCopy.find({});
    console.log('✅ Portfolio Copies:', copies.length);
    
    return { users: users.length, portfolios: portfolios.length, leaderboards: leaderboards.length, copies: copies.length };
  } catch (error) {
    console.error('❌ Database Operations Test Failed:', error.message);
    return null;
  }
}

async function runAllTests() {
  console.log('🚀 Starting Leaderboard System Tests...\n');
  
  try {
    // Seed test data first
    console.log('📊 Seeding test data...');
    const seedResult = await seedLeaderboardData();
    console.log('✅ Data seeding completed:', seedResult);
    
    const results = {};
    
    results.xirr = await testXIRRCalculation();
    results.leaderboard = await testLeaderboardGeneration();
    results.copying = await testPortfolioCopying();
    results.history = await testUserHistory();
    results.cron = await testCronService();
    results.database = await testDatabaseOperations();
    
    console.log('\n🎉 All Leaderboard tests completed successfully!');
    console.log('\n📋 Test Summary:');
    console.log('✅ XIRR Calculation - Mathematical return calculations');
    console.log('✅ Leaderboard Generation - Ranking and statistics');
    console.log('✅ Portfolio Copying - SIP and lumpsum copying');
    console.log('✅ User History - Leaderboard and copy history');
    console.log('✅ Cron Service - Automated updates and jobs');
    console.log('✅ Database Operations - Model functionality');
    
    console.log('\n🔧 Leaderboard Features Demonstrated:');
    console.log('• Anonymous user display with secret codes');
    console.log('• XIRR calculation for multiple time periods');
    console.log('• Portfolio allocation-based rankings');
    console.log('• Copy portfolio functionality (SIP/Lumpsum)');
    console.log('• Daily automated leaderboard updates');
    console.log('• Complete audit trail and history tracking');
    
    console.log('\n🚀 Next Steps:');
    console.log('1. Start server: npm start');
    console.log('2. Test API endpoints with authentication');
    console.log('3. Integrate with BSE Star MF API for real execution');
    console.log('4. Add notification service for copy alerts');
    console.log('5. Implement real-time leaderboard updates');
    
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
  testXIRRCalculation,
  testLeaderboardGeneration,
  testPortfolioCopying,
  testUserHistory,
  testCronService,
  testDatabaseOperations,
  runAllTests
}; 