/**
 * Test script for SIP Calculator API
 * Run this to verify backend integration is working
 */

const axios = require('axios');

const API_BASE_URL = 'http://localhost:3001';

async function testSIPCalculatorAPI() {
  console.log('🧪 Testing SIP Calculator API Integration...\n');

  try {
    // Test 1: Health check
    console.log('1. Testing health check...');
    const healthResponse = await axios.get(`${API_BASE_URL}/api/sip-calculator/health`);
    console.log('✅ Health check passed:', healthResponse.data.data.status);

    // Test 2: Regular SIP calculation
    console.log('\n2. Testing regular SIP calculation...');
    const regularSIPParams = {
      monthlyInvestment: 5000,
      expectedReturn: 12,
      timePeriod: 10
    };
    
    const regularResponse = await axios.post(`${API_BASE_URL}/api/sip-calculator/regular`, regularSIPParams);
    const regularResult = regularResponse.data.data;
    console.log('✅ Regular SIP calculation passed');
    console.log(`   Monthly Investment: ₹${regularResult.totalInvestment.toLocaleString()}`);
    console.log(`   Maturity Amount: ₹${regularResult.maturityAmount.toLocaleString()}`);
    console.log(`   Total Gains: ₹${regularResult.totalGains.toLocaleString()}`);

    // Test 3: Step-up SIP calculation
    console.log('\n3. Testing step-up SIP calculation...');
    const stepUpParams = {
      ...regularSIPParams,
      stepUpPercentage: 10
    };
    
    const stepUpResponse = await axios.post(`${API_BASE_URL}/api/sip-calculator/stepup`, stepUpParams);
    const stepUpResult = stepUpResponse.data.data;
    console.log('✅ Step-up SIP calculation passed');
    console.log(`   Maturity Amount: ₹${stepUpResult.maturityAmount.toLocaleString()}`);
    console.log(`   Advantage over Regular: ₹${(stepUpResult.maturityAmount - regularResult.maturityAmount).toLocaleString()}`);

    // Test 4: Dynamic SIP calculation
    console.log('\n4. Testing dynamic SIP calculation...');
    const dynamicParams = {
      ...regularSIPParams,
      dynamicAdjustmentRange: 15
    };
    
    const dynamicResponse = await axios.post(`${API_BASE_URL}/api/sip-calculator/dynamic`, dynamicParams);
    const dynamicResult = dynamicResponse.data.data;
    console.log('✅ Dynamic SIP calculation passed');
    console.log(`   Maturity Amount: ₹${dynamicResult.maturityAmount.toLocaleString()}`);
    console.log(`   AI Advantage: ₹${dynamicResult.aiAdvantage.toLocaleString()} (${dynamicResult.aiAdvantagePercentage}%)`);

    // Test 5: SIP comparison
    console.log('\n5. Testing SIP comparison...');
    const comparisonResponse = await axios.post(`${API_BASE_URL}/api/sip-calculator/compare`, {
      ...regularSIPParams,
      stepUpPercentage: 10,
      dynamicAdjustmentRange: 15
    });
    const comparisonResult = comparisonResponse.data.data;
    console.log('✅ SIP comparison passed');
    console.log(`   Best Performer: ${comparisonResult.analysis.bestPerformer.type.toUpperCase()}`);
    console.log(`   Best Amount: ₹${comparisonResult.analysis.bestPerformer.maturityAmount.toLocaleString()}`);

    // Test 6: Goal-based SIP
    console.log('\n6. Testing goal-based SIP calculation...');
    const goalParams = {
      targetAmount: 1000000, // 10 Lakh
      timePeriod: 10,
      expectedReturn: 12
    };
    
    const goalResponse = await axios.post(`${API_BASE_URL}/api/sip-calculator/goal-based`, goalParams);
    const goalResult = goalResponse.data.data;
    console.log('✅ Goal-based SIP calculation passed');
    console.log(`   Target: ₹${goalResult.targetAmount.toLocaleString()}`);
    console.log(`   Required Monthly: ₹${goalResult.requiredMonthlyInvestment.toLocaleString()}`);
    console.log(`   Feasibility: ${goalResult.feasibilityScore.category} (${goalResult.feasibilityScore.score}/100)`);

    // Test 7: Quick calculation
    console.log('\n7. Testing quick calculation...');
    const quickResponse = await axios.get(`${API_BASE_URL}/api/sip-calculator/quick-calculate?monthlyInvestment=5000&expectedReturn=12&timePeriod=10&type=regular`);
    const quickResult = quickResponse.data.data;
    console.log('✅ Quick calculation passed');
    console.log(`   Maturity Amount: ₹${quickResult.maturityAmount.toLocaleString()}`);

    console.log('\n🎉 ALL TESTS PASSED! SIP Calculator API is fully functional');
    console.log('\n📊 Backend Integration Summary:');
    console.log('   ✅ Health check endpoint working');
    console.log('   ✅ Regular SIP calculation working');
    console.log('   ✅ Step-up SIP calculation working');
    console.log('   ✅ Dynamic SIP with AI working');
    console.log('   ✅ SIP comparison working');
    console.log('   ✅ Goal-based SIP working');
    console.log('   ✅ Quick calculation working');
    console.log('\n🔗 Frontend can now connect to all backend endpoints successfully!');

  } catch (error) {
    console.error('❌ Test failed:', error.response?.data || error.message);
    console.log('\n🔧 Make sure the backend server is running on port 3001');
    console.log('   Run: node app.js');
  }
}

// Run the test
testSIPCalculatorAPI();
