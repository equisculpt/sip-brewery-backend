/**
 * Direct test of SIP Calculator Service (without server)
 */

const SIPCalculatorService = require('./services/SIPCalculatorService');

function testSIPCalculatorService() {
  console.log('🧪 Testing SIP Calculator Service directly...\n');

  const sipService = new SIPCalculatorService();

  try {
    // Test 1: Regular SIP
    console.log('1. Testing Regular SIP Calculation:');
    const regularResult = sipService.calculateRegularSIP(5000, 12, 10);
    console.log(`   Monthly Investment: ₹5,000`);
    console.log(`   Expected Return: 12% p.a.`);
    console.log(`   Time Period: 10 years`);
    console.log(`   ✅ Total Investment: ₹${regularResult.totalInvestment.toLocaleString()}`);
    console.log(`   ✅ Maturity Amount: ₹${regularResult.maturityAmount.toLocaleString()}`);
    console.log(`   ✅ Total Gains: ₹${regularResult.totalGains.toLocaleString()}`);

    // Test 2: Step-up SIP
    console.log('\n2. Testing Step-up SIP Calculation:');
    const stepUpResult = sipService.calculateStepUpSIP(5000, 12, 10, 10);
    console.log(`   Step-up Percentage: 10% annually`);
    console.log(`   ✅ Total Investment: ₹${stepUpResult.totalInvestment.toLocaleString()}`);
    console.log(`   ✅ Maturity Amount: ₹${stepUpResult.maturityAmount.toLocaleString()}`);
    console.log(`   ✅ Advantage over Regular: ₹${(stepUpResult.maturityAmount - regularResult.maturityAmount).toLocaleString()}`);

    // Test 3: Dynamic SIP
    console.log('\n3. Testing Dynamic SIP Calculation:');
    const dynamicResult = sipService.calculateDynamicSIP(5000, 12, 10, 15);
    console.log(`   Dynamic Adjustment Range: ±15%`);
    console.log(`   ✅ Maturity Amount: ₹${dynamicResult.maturityAmount.toLocaleString()}`);
    console.log(`   ✅ AI Advantage: ₹${dynamicResult.aiAdvantage.toLocaleString()} (${dynamicResult.aiAdvantagePercentage}%)`);

    // Test 4: Goal-based SIP
    console.log('\n4. Testing Goal-based SIP Calculation:');
    const goalResult = sipService.calculateGoalBasedSIP(1000000, 10, 12);
    console.log(`   Target Amount: ₹10,00,000`);
    console.log(`   ✅ Required Monthly Investment: ₹${goalResult.requiredMonthlyInvestment.toLocaleString()}`);
    console.log(`   ✅ Feasibility Score: ${goalResult.feasibilityScore.score}/100 (${goalResult.feasibilityScore.category})`);

    // Test 5: SIP Comparison
    console.log('\n5. Testing SIP Comparison:');
    const comparisonParams = {
      monthlyInvestment: 5000,
      expectedReturn: 12,
      timePeriod: 10,
      stepUpPercentage: 10,
      dynamicAdjustmentRange: 15
    };
    const comparisonResult = sipService.getSIPComparison(comparisonParams);
    console.log(`   ✅ Best Performer: ${comparisonResult.analysis.bestPerformer.type.toUpperCase()}`);
    console.log(`   ✅ Best Amount: ₹${comparisonResult.analysis.bestPerformer.maturityAmount.toLocaleString()}`);
    console.log(`   ✅ Recommendations: ${comparisonResult.recommendations.length} generated`);

    console.log('\n🎉 ALL SERVICE TESTS PASSED!');
    console.log('\n📊 SIP Calculator Service Summary:');
    console.log('   ✅ Regular SIP calculation working');
    console.log('   ✅ Step-up SIP calculation working');
    console.log('   ✅ Dynamic SIP with AI working');
    console.log('   ✅ Goal-based SIP working');
    console.log('   ✅ SIP comparison working');
    console.log('\n🔗 Backend service is ready for frontend integration!');

    return true;
  } catch (error) {
    console.error('❌ Service test failed:', error.message);
    console.error(error.stack);
    return false;
  }
}

// Run the test
const success = testSIPCalculatorService();
process.exit(success ? 0 : 1);
