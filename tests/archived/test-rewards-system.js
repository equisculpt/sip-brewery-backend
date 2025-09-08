const mongoose = require('mongoose');
const { User, Reward, RewardSummary, Referral } = require('./src/models');
const rewardsService = require('./src/services/rewardsService');
const logger = require('./src/utils/logger');

// Test configuration
const TEST_USERS = [
  {
    supabaseId: 'test-user-1',
    name: 'John Doe',
    email: 'john@example.com',
    phone: '9876543210',
    kycStatus: 'VERIFIED',
    isActive: true
  },
  {
    supabaseId: 'test-user-2',
    name: 'Jane Smith',
    email: 'jane@example.com',
    phone: '9876543211',
    kycStatus: 'VERIFIED',
    isActive: true
  },
  {
    supabaseId: 'test-user-3',
    name: 'Bob Wilson',
    email: 'bob@example.com',
    phone: '9876543212',
    kycStatus: 'VERIFIED',
    isActive: true
  }
];

class RewardsSystemTest {
  constructor() {
    this.testResults = [];
  }

  async connectToMongoDB() {
    try {
      const uri = process.env.MONGODB_URI || 'mongodb://localhost:27017/sip-brewery-test';
      await mongoose.connect(uri);
      console.log('✅ Connected to MongoDB');
      return true;
    } catch (error) {
      console.error('❌ MongoDB connection failed:', error.message);
      return false;
    }
  }

  async cleanupDatabase() {
    try {
      await User.deleteMany({ supabaseId: { $in: TEST_USERS.map(u => u.supabaseId) } });
      await Reward.deleteMany({ userId: { $in: TEST_USERS.map(u => u.supabaseId) } });
      await RewardSummary.deleteMany({ userId: { $in: TEST_USERS.map(u => u.supabaseId) } });
      await Referral.deleteMany({ 
        referrerId: { $in: TEST_USERS.map(u => u.supabaseId) },
        referredId: { $in: TEST_USERS.map(u => u.supabaseId) }
      });
      console.log('✅ Database cleaned up');
    } catch (error) {
      console.error('❌ Database cleanup failed:', error.message);
    }
  }

  async createTestUsers() {
    try {
      const users = [];
      for (const userData of TEST_USERS) {
        const user = new User(userData);
        await user.save();
        users.push(user);
        console.log(`✅ Created user: ${user.name} (${user.referralCode})`);
      }
      return users;
    } catch (error) {
      console.error('❌ Failed to create test users:', error.message);
      return [];
    }
  }

  async testSipLoyaltyPoints() {
    console.log('\n🧪 Testing SIP Loyalty Points...');
    
    try {
      const user = await User.findOne({ supabaseId: 'test-user-1' });
      
      // Test awarding SIP loyalty points
      const result = await rewardsService.awardSipLoyaltyPoints(
        user.supabaseId,
        'SIP_001',
        'HDFC Mid-Cap Opportunities Fund',
        'HDFC123456',
        'BSE_CONFIRM_001'
      );
      
      this.testResults.push({
        test: 'SIP Loyalty Points',
        status: result.success ? 'PASS' : 'FAIL',
        message: result.message
      });
      
      console.log(`✅ SIP loyalty point awarded: ${result.message}`);
      
      // Verify reward was created
      const reward = await Reward.findOne({
        userId: user.supabaseId,
        type: 'SIP_LOYALTY_POINTS',
        sipId: 'SIP_001'
      });
      
      if (reward) {
        console.log(`✅ Reward record created: ${reward.description}`);
      } else {
        console.log('❌ Reward record not found');
      }
      
      return result;
    } catch (error) {
      console.error('❌ SIP loyalty points test failed:', error.message);
      this.testResults.push({
        test: 'SIP Loyalty Points',
        status: 'FAIL',
        message: error.message
      });
      return null;
    }
  }

  async testCashbackFor12Sips() {
    console.log('\n🧪 Testing Cashback for 12 SIPs...');
    
    try {
      const user = await User.findOne({ supabaseId: 'test-user-1' });
      const fundName = 'HDFC Mid-Cap Opportunities Fund';
      const folioNumber = 'HDFC123456';
      
      // Award 12 SIP loyalty points to trigger cashback
      for (let i = 1; i <= 12; i++) {
        await rewardsService.awardSipLoyaltyPoints(
          user.supabaseId,
          `SIP_${i.toString().padStart(3, '0')}`,
          fundName,
          folioNumber,
          `BSE_CONFIRM_${i.toString().padStart(3, '0')}`
        );
      }
      
      // Check for cashback
      const cashbackResult = await rewardsService.checkAndAwardCashback(
        user.supabaseId,
        fundName,
        folioNumber
      );
      
      this.testResults.push({
        test: 'Cashback for 12 SIPs',
        status: cashbackResult.success ? 'PASS' : 'FAIL',
        message: cashbackResult.message
      });
      
      if (cashbackResult.success) {
        console.log(`✅ Cashback awarded: ${cashbackResult.message}`);
      } else {
        console.log(`ℹ️ Cashback check: ${cashbackResult.message}`);
      }
      
      return cashbackResult;
    } catch (error) {
      console.error('❌ Cashback test failed:', error.message);
      this.testResults.push({
        test: 'Cashback for 12 SIPs',
        status: 'FAIL',
        message: error.message
      });
      return null;
    }
  }

  async testReferralBonus() {
    console.log('\n🧪 Testing Referral Bonus...');
    
    try {
      const referrer = await User.findOne({ supabaseId: 'test-user-1' });
      const referred = await User.findOne({ supabaseId: 'test-user-2' });
      
      // Set up referral relationship
      referred.referredBy = referrer.supabaseId;
      await referred.save();
      
      // Award SIP to referred user to make them eligible
      await rewardsService.awardSipLoyaltyPoints(
        referred.supabaseId,
        'REF_SIP_001',
        'Axis Bluechip Fund',
        'AXIS789012',
        'BSE_REF_001'
      );
      
      // Validate and award referral bonus
      const result = await rewardsService.validateAndAwardReferralBonus(referred.supabaseId);
      
      this.testResults.push({
        test: 'Referral Bonus',
        status: result.success ? 'PASS' : 'FAIL',
        message: result.message
      });
      
      if (result.success) {
        console.log(`✅ Referral bonus awarded: ${result.message}`);
        
        // Verify referral record
        const referral = await Referral.findOne({
          referrerId: referrer.supabaseId,
          referredId: referred.supabaseId
        });
        
        if (referral) {
          console.log(`✅ Referral record created: ${referral.status}`);
        }
      } else {
        console.log(`ℹ️ Referral validation: ${result.message}`);
      }
      
      return result;
    } catch (error) {
      console.error('❌ Referral bonus test failed:', error.message);
      this.testResults.push({
        test: 'Referral Bonus',
        status: 'FAIL',
        message: error.message
      });
      return null;
    }
  }

  async testRewardSummary() {
    console.log('\n🧪 Testing Reward Summary...');
    
    try {
      const user = await User.findOne({ supabaseId: 'test-user-1' });
      
      const summary = await rewardsService.getUserRewardSummary(user.supabaseId);
      
      console.log('📊 Reward Summary:');
      console.log(`   Total Points: ${summary.totalPoints}`);
      console.log(`   Total Cashback: ₹${summary.totalCashback}`);
      console.log(`   Total Referral Bonus: ₹${summary.totalReferralBonus}`);
      console.log(`   Total SIP Installments: ${summary.totalSipInstallments}`);
      console.log(`   Pending Payout: ₹${summary.pendingPayout}`);
      console.log(`   Total Paid Out: ₹${summary.totalPaidOut}`);
      
      this.testResults.push({
        test: 'Reward Summary',
        status: 'PASS',
        message: `Summary generated with ${summary.recentTransactions.length} recent transactions`
      });
      
      return summary;
    } catch (error) {
      console.error('❌ Reward summary test failed:', error.message);
      this.testResults.push({
        test: 'Reward Summary',
        status: 'FAIL',
        message: error.message
      });
      return null;
    }
  }

  async testAntiAbuseMeasures() {
    console.log('\n🧪 Testing Anti-Abuse Measures...');
    
    try {
      const user = await User.findOne({ supabaseId: 'test-user-3' });
      
      // Test self-referral prevention
      try {
        await rewardsService.validateAndAwardReferralBonus(user.supabaseId);
        console.log('❌ Self-referral should have been blocked');
        this.testResults.push({
          test: 'Anti-Abuse: Self-Referral',
          status: 'FAIL',
          message: 'Self-referral was not blocked'
        });
      } catch (error) {
        if (error.message.includes('Self-referral not allowed')) {
          console.log('✅ Self-referral correctly blocked');
          this.testResults.push({
            test: 'Anti-Abuse: Self-Referral',
            status: 'PASS',
            message: 'Self-referral correctly blocked'
          });
        } else {
          console.log('❌ Unexpected error:', error.message);
          this.testResults.push({
            test: 'Anti-Abuse: Self-Referral',
            status: 'FAIL',
            message: error.message
          });
        }
      }
      
      // Test duplicate reward prevention
      try {
        await rewardsService.awardSipLoyaltyPoints(
          user.supabaseId,
          'SIP_001', // Same SIP ID
          'Test Fund',
          'TEST123',
          'BSE_TEST_001'
        );
        
        await rewardsService.awardSipLoyaltyPoints(
          user.supabaseId,
          'SIP_001', // Same SIP ID again
          'Test Fund',
          'TEST123',
          'BSE_TEST_001'
        );
        
        console.log('❌ Duplicate reward should have been blocked');
        this.testResults.push({
          test: 'Anti-Abuse: Duplicate Rewards',
          status: 'FAIL',
          message: 'Duplicate reward was not blocked'
        });
      } catch (error) {
        if (error.message.includes('already awarded')) {
          console.log('✅ Duplicate reward correctly blocked');
          this.testResults.push({
            test: 'Anti-Abuse: Duplicate Rewards',
            status: 'PASS',
            message: 'Duplicate reward correctly blocked'
          });
        } else {
          console.log('❌ Unexpected error:', error.message);
          this.testResults.push({
            test: 'Anti-Abuse: Duplicate Rewards',
            status: 'FAIL',
            message: error.message
          });
        }
      }
      
    } catch (error) {
      console.error('❌ Anti-abuse test failed:', error.message);
      this.testResults.push({
        test: 'Anti-Abuse Measures',
        status: 'FAIL',
        message: error.message
      });
    }
  }

  async testAdminFunctions() {
    console.log('\n🧪 Testing Admin Functions...');
    
    try {
      // Get unpaid rewards
      const unpaidRewards = await rewardsService.getUnpaidRewards();
      console.log(`📋 Found ${unpaidRewards.length} unpaid rewards`);
      
      // Get referral leaderboard
      const leaderboard = await rewardsService.getReferralLeaderboard(5);
      console.log(`🏆 Top ${leaderboard.length} referrers:`, leaderboard.map(u => ({
        name: u.name,
        referrals: u.referralCount,
        bonus: u.totalReferralBonus
      })));
      
      this.testResults.push({
        test: 'Admin Functions',
        status: 'PASS',
        message: `Found ${unpaidRewards.length} unpaid rewards and ${leaderboard.length} leaderboard entries`
      });
      
      return { unpaidRewards, leaderboard };
    } catch (error) {
      console.error('❌ Admin functions test failed:', error.message);
      this.testResults.push({
        test: 'Admin Functions',
        status: 'FAIL',
        message: error.message
      });
      return null;
    }
  }

  async runAllTests() {
    console.log('🚀 Starting SEBI-Compliant Rewards System Tests\n');
    
    // Connect to MongoDB
    const connected = await this.connectToMongoDB();
    if (!connected) {
      console.log('❌ Cannot proceed without MongoDB connection');
      return;
    }
    
    // Cleanup and setup
    await this.cleanupDatabase();
    const users = await createTestUsers();
    
    if (users.length === 0) {
      console.log('❌ Cannot proceed without test users');
      return;
    }
    
    // Run tests
    await this.testSipLoyaltyPoints();
    await this.testCashbackFor12Sips();
    await this.testReferralBonus();
    await this.testRewardSummary();
    await this.testAntiAbuseMeasures();
    await this.testAdminFunctions();
    
    // Print results
    this.printTestResults();
    
    // Cleanup
    await this.cleanupDatabase();
    await mongoose.connection.close();
    console.log('\n✅ Tests completed');
  }

  printTestResults() {
    console.log('\n📋 Test Results Summary:');
    console.log('='.repeat(50));
    
    const passed = this.testResults.filter(r => r.status === 'PASS').length;
    const failed = this.testResults.filter(r => r.status === 'FAIL').length;
    
    this.testResults.forEach(result => {
      const icon = result.status === 'PASS' ? '✅' : '❌';
      console.log(`${icon} ${result.test}: ${result.message}`);
    });
    
    console.log('='.repeat(50));
    console.log(`Total: ${this.testResults.length} | Passed: ${passed} | Failed: ${failed}`);
    
    if (failed === 0) {
      console.log('🎉 All tests passed! SEBI-compliant rewards system is working correctly.');
    } else {
      console.log('⚠️ Some tests failed. Please review the implementation.');
    }
  }
}

// Helper function to create test users
async function createTestUsers() {
  try {
    const users = [];
    for (const userData of TEST_USERS) {
      const user = new User(userData);
      await user.save();
      users.push(user);
      console.log(`✅ Created user: ${user.name} (${user.referralCode})`);
    }
    return users;
  } catch (error) {
    console.error('❌ Failed to create test users:', error.message);
    return [];
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  const tester = new RewardsSystemTest();
  tester.runAllTests().catch(error => {
    console.error('❌ Test execution failed:', error);
    process.exit(1);
  });
}

module.exports = RewardsSystemTest; 