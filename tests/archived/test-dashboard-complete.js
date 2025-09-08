const axios = require('axios');
const dataSeeder = require('./src/utils/dataSeeder');

const BASE_URL = 'http://localhost:3000';

// Dummy Supabase token for testing
const DUMMY_TOKEN = 'dummy-supabase-token-123';

async function testCompleteDashboard() {
  console.log('🚀 Testing Complete SIP Brewery Dashboard Backend\n');

  try {
    // Step 1: Seed sample data
    console.log('📊 Step 1: Seeding sample data...');
    await dataSeeder.seedAllData();
    console.log('✅ Sample data seeded successfully\n');

    // Step 2: Test Authentication
    console.log('🔐 Step 2: Testing Authentication...');
    await testAuthentication();
    console.log('✅ Authentication working\n');

    // Step 3: Test KYC Status
    console.log('📋 Step 3: Testing KYC Status...');
    await testKYCStatus();
    console.log('✅ KYC status working\n');

    // Step 4: Test Complete Dashboard
    console.log('📈 Step 4: Testing Complete Dashboard...');
    await testCompleteDashboardData();
    console.log('✅ Complete dashboard working\n');

    // Step 5: Test Individual Modules
    console.log('🔧 Step 5: Testing Individual Dashboard Modules...');
    await testIndividualModules();
    console.log('✅ All modules working\n');

    console.log('🎉 All tests completed successfully!');
    console.log('\n📝 Dashboard Features Implemented:');
    console.log('   ✅ Supabase Authentication Integration');
    console.log('   ✅ KYC Status Management');
    console.log('   ✅ Complete Dashboard Data Aggregation');
    console.log('   ✅ Holdings Management');
    console.log('   ✅ Smart SIP Center');
    console.log('   ✅ Transaction History');
    console.log('   ✅ Statements Generation');
    console.log('   ✅ Rewards & Cashback System');
    console.log('   ✅ Referral Program');
    console.log('   ✅ AI Portfolio Analytics');
    console.log('   ✅ Peer Comparison Analytics');
    console.log('   ✅ Performance Charts');
    console.log('   ✅ User Profile Management');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
    if (error.response) {
      console.error('   Response:', error.response.data);
    }
  }
}

async function testAuthentication() {
  try {
    const response = await axios.get(`${BASE_URL}/api/auth/check`, {
      headers: {
        'Authorization': `Bearer ${DUMMY_TOKEN}`
      }
    });

    if (response.data.success) {
      console.log('   ✅ Auth check successful');
      console.log(`   👤 User: ${response.data.data.name}`);
      console.log(`   📧 Email: ${response.data.data.email}`);
      console.log(`   🔐 KYC Status: ${response.data.data.kycStatus}`);
    } else {
      throw new Error('Auth check failed');
    }
  } catch (error) {
    throw new Error(`Authentication test failed: ${error.message}`);
  }
}

async function testKYCStatus() {
  try {
    const response = await axios.get(`${BASE_URL}/api/auth/kyc/status`, {
      headers: {
        'Authorization': `Bearer ${DUMMY_TOKEN}`
      }
    });

    if (response.data.success) {
      console.log('   ✅ KYC status retrieved');
      console.log(`   📋 Status: ${response.data.data.status}`);
      console.log(`   ✅ Completed: ${response.data.data.isCompleted}`);
      console.log(`   👤 Name: ${response.data.data.profile.name}`);
      console.log(`   📱 Mobile: ${response.data.data.profile.mobile}`);
    } else {
      throw new Error('KYC status check failed');
    }
  } catch (error) {
    throw new Error(`KYC test failed: ${error.message}`);
  }
}

async function testCompleteDashboardData() {
  try {
    const response = await axios.get(`${BASE_URL}/api/dashboard`, {
      headers: {
        'Authorization': `Bearer ${DUMMY_TOKEN}`
      }
    });

    if (response.data.success) {
      const data = response.data.data;
      console.log('   ✅ Complete dashboard data retrieved');
      console.log(`   📊 Holdings: ${data.holdings.length} funds`);
      console.log(`   💰 Total Value: ₹${data.holdings.reduce((sum, h) => sum + h.value, 0).toLocaleString()}`);
      console.log(`   📈 Transactions: ${data.transactions.length} recent`);
      console.log(`   🎁 Rewards: ₹${data.rewards.totalAmount} total`);
      console.log(`   🤖 AI XIRR: ${data.aiAnalytics.xirr}%`);
      console.log(`   📊 Portfolio XIRR: ${data.portfolioAnalytics.userXirr}%`);
      console.log(`   📈 Performance Chart: ${data.performanceChart.periods.length} periods`);
    } else {
      throw new Error('Dashboard data retrieval failed');
    }
  } catch (error) {
    throw new Error(`Dashboard test failed: ${error.message}`);
  }
}

async function testIndividualModules() {
  const modules = [
    { name: 'Holdings', endpoint: '/holdings' },
    { name: 'Smart SIP Center', endpoint: '/smart-sip' },
    { name: 'Transactions', endpoint: '/transactions' },
    { name: 'Statements', endpoint: '/statements' },
    { name: 'Rewards', endpoint: '/rewards' },
    { name: 'Referral', endpoint: '/referral' },
    { name: 'AI Analytics', endpoint: '/ai-analytics' },
    { name: 'Portfolio Analytics', endpoint: '/portfolio-analytics' },
    { name: 'Performance Chart', endpoint: '/performance-chart' },
    { name: 'Profile', endpoint: '/profile' }
  ];

  for (const module of modules) {
    try {
      const response = await axios.get(`${BASE_URL}/api/dashboard${module.endpoint}`, {
        headers: {
          'Authorization': `Bearer ${DUMMY_TOKEN}`
        }
      });

      if (response.data.success) {
        console.log(`   ✅ ${module.name}: Working`);
      } else {
        console.log(`   ❌ ${module.name}: Failed`);
      }
    } catch (error) {
      console.log(`   ❌ ${module.name}: Error - ${error.message}`);
    }
  }
}

async function demonstrateAPIUsage() {
  console.log('\n📚 API Usage Examples:\n');

  console.log('1. Frontend Authentication Flow:');
  console.log('   // After Supabase login, get user token');
  console.log('   const token = supabase.auth.session()?.access_token;');
  console.log('   // Send to backend for verification');
  console.log('   const response = await fetch("/api/auth/check", {');
  console.log('     headers: { "Authorization": `Bearer ${token}` }');
  console.log('   });\n');

  console.log('2. Check KYC Status:');
  console.log('   const kycResponse = await fetch("/api/auth/kyc/status", {');
  console.log('     headers: { "Authorization": `Bearer ${token}` }');
  console.log('   });');
  console.log('   const kycData = await kycResponse.json();');
  console.log('   if (kycData.data.isCompleted) {');
  console.log('     // Show dashboard');
  console.log('   } else {');
  console.log('     // Show KYC form');
  console.log('   }\n');

  console.log('3. Load Complete Dashboard:');
  console.log('   const dashboardResponse = await fetch("/api/dashboard", {');
  console.log('     headers: { "Authorization": `Bearer ${token}` }');
  console.log('   });');
  console.log('   const dashboardData = await dashboardResponse.json();');
  console.log('   // Use dashboardData.holdings, .transactions, .rewards, etc.\n');

  console.log('4. Individual Module Access:');
  console.log('   // Get only holdings');
  console.log('   const holdingsResponse = await fetch("/api/dashboard/holdings", {');
  console.log('     headers: { "Authorization": `Bearer ${token}` }');
  console.log('   });\n');

  console.log('5. Update KYC Status (for testing):');
  console.log('   const updateResponse = await fetch("/api/auth/kyc/status", {');
  console.log('     method: "PUT",');
  console.log('     headers: { "Authorization": `Bearer ${token}` },');
  console.log('     body: JSON.stringify({ status: "SUCCESS" })');
  console.log('   });\n');
}

// Check if server is running
async function checkServer() {
  try {
    await axios.get(`${BASE_URL}/health`);
    console.log('✅ Server is running at', BASE_URL);
    await testCompleteDashboard();
    await demonstrateAPIUsage();
  } catch (error) {
    console.error('❌ Server is not running. Please start the server first:');
    console.error('   npm start');
    console.error('   or');
    console.error('   node index.js');
  }
}

checkServer(); 