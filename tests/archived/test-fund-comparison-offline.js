const fundComparisonService = require('./src/services/fundComparisonService');

// Test data
const testCases = [
  {
    name: 'Mid-Cap Fund Comparison',
    data: {
      fundCodes: ['HDFCMIDCAP', 'MIRAEEMERGING', 'AXISBLUECHIP'],
      category: 'Mid Cap',
      period: '1y',
      investmentAmount: 100000,
      includeRatings: true,
      includeRecommendations: true
    }
  },
  {
    name: 'Large-Cap Fund Comparison',
    data: {
      fundCodes: ['ICICIBLUECHIP', 'AXISBLUECHIP'],
      category: 'Large Cap',
      period: '3y',
      investmentAmount: 500000,
      includeRatings: true,
      includeRecommendations: true
    }
  },
  {
    name: 'Mixed Category Comparison',
    data: {
      fundCodes: ['HDFCMIDCAP', 'ICICIBLUECHIP', 'SBISMALLCAP', 'AXISBLUECHIP', 'MIRAEEMERGING'],
      period: '1y',
      investmentAmount: 200000,
      includeRatings: true,
      includeRecommendations: true
    }
  }
];

// Test results
const results = {
  total: 0,
  passed: 0,
  failed: 0,
  details: []
};

/**
 * Test fund comparison service directly
 */
async function testFundComparison(testCase) {
  try {
    console.log(`\n🧪 Testing: ${testCase.name}`);
    console.log(`📊 Comparing funds: ${testCase.data.fundCodes.join(', ')}`);
    
    const comparison = await fundComparisonService.compareFunds(testCase.data);
    
    // Validate response structure
    const validation = validateResponse(comparison, testCase);
    
    if (validation.isValid) {
      console.log(`✅ PASSED: ${testCase.name}`);
      console.log(`📈 Top performer: ${comparison.summary.topPerformer.fundName} (${comparison.summary.topPerformer.rating}⭐)`);
      console.log(`💰 Best value: ${comparison.summary.bestValue.fundName}`);
      console.log(`🛡️  Safest choice: ${comparison.summary.safestChoice?.fundName || 'N/A'}`);
      
      results.passed++;
      results.details.push({
        test: testCase.name,
        status: 'PASSED',
        topPerformer: comparison.summary.topPerformer.fundName,
        bestValue: comparison.summary.bestValue.fundName
      });
    } else {
      console.log(`❌ FAILED: ${testCase.name} - ${validation.error}`);
      results.failed++;
      results.details.push({
        test: testCase.name,
        status: 'FAILED',
        error: validation.error
      });
    }
  } catch (error) {
    console.log(`❌ FAILED: ${testCase.name} - ${error.message}`);
    results.failed++;
    results.details.push({
      test: testCase.name,
      status: 'FAILED',
      error: error.message
    });
  }
}

/**
 * Validate response structure
 */
function validateResponse(data, testCase) {
  try {
    // Check required top-level fields
    const requiredFields = ['comparisonPeriod', 'investmentAmount', 'totalFunds', 'funds'];
    for (const field of requiredFields) {
      if (!data.hasOwnProperty(field)) {
        return { isValid: false, error: `Missing required field: ${field}` };
      }
    }

    // Check funds array
    if (!Array.isArray(data.funds) || data.funds.length !== testCase.data.fundCodes.length) {
      return { isValid: false, error: `Invalid funds array length` };
    }

    // Check each fund structure
    for (const fund of data.funds) {
      const fundValidation = validateFundStructure(fund);
      if (!fundValidation.isValid) {
        return fundValidation;
      }
    }

    // Check ratings if included
    if (testCase.data.includeRatings) {
      if (!data.ratings || !Array.isArray(data.ratings)) {
        return { isValid: false, error: 'Missing or invalid ratings array' };
      }
      
      for (const rating of data.ratings) {
        const ratingValidation = validateRatingStructure(rating);
        if (!ratingValidation.isValid) {
          return ratingValidation;
        }
      }
    }

    // Check recommendations if included
    if (testCase.data.includeRecommendations) {
      if (!data.recommendations) {
        return { isValid: false, error: 'Missing recommendations' };
      }
      
      const recValidation = validateRecommendationsStructure(data.recommendations);
      if (!recValidation.isValid) {
        return recValidation;
      }
    }

    // Check summary
    if (!data.summary) {
      return { isValid: false, error: 'Missing summary' };
    }

    return { isValid: true };
  } catch (error) {
    return { isValid: false, error: `Validation error: ${error.message}` };
  }
}

/**
 * Validate fund structure
 */
function validateFundStructure(fund) {
  const requiredFields = ['fundCode', 'fundDetails', 'performance', 'riskMetrics', 'analysis'];
  
  for (const field of requiredFields) {
    if (!fund.hasOwnProperty(field)) {
      return { isValid: false, error: `Fund missing required field: ${field}` };
    }
  }

  // Check fund details
  const detailFields = ['name', 'fundHouse', 'category', 'subCategory', 'aum', 'expenseRatio', 'nav'];
  for (const field of detailFields) {
    if (!fund.fundDetails.hasOwnProperty(field)) {
      return { isValid: false, error: `Fund details missing field: ${field}` };
    }
  }

  // Check performance
  const perfFields = ['totalReturn', 'annualizedReturn'];
  for (const field of perfFields) {
    if (!fund.performance.hasOwnProperty(field)) {
      return { isValid: false, error: `Performance missing field: ${field}` };
    }
  }

  // Check risk metrics
  const riskFields = ['volatility', 'maxDrawdown', 'sharpeRatio'];
  for (const field of riskFields) {
    if (!fund.riskMetrics.hasOwnProperty(field)) {
      return { isValid: false, error: `Risk metrics missing field: ${field}` };
    }
  }

  return { isValid: true };
}

/**
 * Validate rating structure
 */
function validateRatingStructure(rating) {
  const requiredFields = ['fundCode', 'fundName', 'overallRating', 'totalScore', 'categoryRatings'];
  
  for (const field of requiredFields) {
    if (!rating.hasOwnProperty(field)) {
      return { isValid: false, error: `Rating missing required field: ${field}` };
    }
  }

  // Check category ratings
  const categories = ['performance', 'risk', 'cost', 'consistency', 'fundHouse', 'taxEfficiency', 'liquidity'];
  for (const category of categories) {
    if (!rating.categoryRatings.hasOwnProperty(category)) {
      return { isValid: false, error: `Rating missing category: ${category}` };
    }
  }

  return { isValid: true };
}

/**
 * Validate recommendations structure
 */
function validateRecommendationsStructure(recommendations) {
  const requiredFields = ['topPick', 'bestValue', 'recommendations'];
  
  for (const field of requiredFields) {
    if (!recommendations.hasOwnProperty(field)) {
      return { isValid: false, error: `Recommendations missing field: ${field}` };
    }
  }

  return { isValid: true };
}

/**
 * Test error cases
 */
async function testErrorCases() {
  console.log('\n🔍 Testing Error Cases...');
  
  const errorTests = [
    {
      name: 'Invalid fund codes (empty array)',
      data: { fundCodes: [] },
      expectedError: 'Please provide 2-5 fund codes for comparison'
    },
    {
      name: 'Invalid fund codes (single fund)',
      data: { fundCodes: ['HDFCMIDCAP'] },
      expectedError: 'Please provide 2-5 fund codes for comparison'
    },
    {
      name: 'Invalid fund codes (too many)',
      data: { fundCodes: ['FUND1', 'FUND2', 'FUND3', 'FUND4', 'FUND5', 'FUND6'] },
      expectedError: 'Please provide 2-5 fund codes for comparison'
    }
  ];

  for (const test of errorTests) {
    try {
      await fundComparisonService.compareFunds(test.data);
      console.log(`❌ FAILED: ${test.name} - Expected error but got success`);
      results.failed++;
    } catch (error) {
      if (error.message.includes('2-5 fund codes')) {
        console.log(`✅ PASSED: ${test.name}`);
        results.passed++;
      } else {
        console.log(`❌ FAILED: ${test.name} - ${error.message}`);
        results.failed++;
      }
    }
  }
}

/**
 * Run performance test
 */
async function testPerformance() {
  console.log('\n⚡ Testing Performance...');
  
  const startTime = Date.now();
  
  try {
    const comparison = await fundComparisonService.compareFunds({
      fundCodes: ['HDFCMIDCAP', 'ICICIBLUECHIP', 'SBISMALLCAP', 'AXISBLUECHIP', 'MIRAEEMERGING'],
      period: '1y',
      investmentAmount: 100000,
      includeRatings: true,
      includeRecommendations: true
    });

    const endTime = Date.now();
    const duration = endTime - startTime;

    if (duration < 5000) { // Should complete within 5 seconds
      console.log(`✅ Performance test PASSED - Completed in ${duration}ms`);
      results.passed++;
    } else {
      console.log(`❌ Performance test FAILED - Took ${duration}ms (too slow)`);
      results.failed++;
    }
  } catch (error) {
    console.log(`❌ Performance test FAILED - ${error.message}`);
    results.failed++;
  }
}

/**
 * Main test runner
 */
async function runTests() {
  console.log('🚀 Starting Offline Fund Comparison Service Tests...\n');
  
  // Test basic functionality
  for (const testCase of testCases) {
    results.total++;
    await testFundComparison(testCase);
  }

  // Test error cases
  await testErrorCases();

  // Test performance
  await testPerformance();

  // Print summary
  console.log('\n📊 Test Summary:');
  console.log(`Total Tests: ${results.total}`);
  console.log(`Passed: ${results.passed} ✅`);
  console.log(`Failed: ${results.failed} ❌`);
  console.log(`Success Rate: ${((results.passed / results.total) * 100).toFixed(1)}%`);

  if (results.failed === 0) {
    console.log('\n🎉 All tests passed! Fund comparison service is working correctly.');
  } else {
    console.log('\n⚠️  Some tests failed. Please check the implementation.');
  }

  // Print detailed results
  console.log('\n📋 Detailed Results:');
  for (const detail of results.details) {
    const status = detail.status === 'PASSED' ? '✅' : '❌';
    console.log(`${status} ${detail.test}`);
    if (detail.error) {
      console.log(`   Error: ${detail.error}`);
    }
    if (detail.topPerformer) {
      console.log(`   Top Performer: ${detail.topPerformer}`);
      console.log(`   Best Value: ${detail.bestValue}`);
    }
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = { runTests, testFundComparison }; 