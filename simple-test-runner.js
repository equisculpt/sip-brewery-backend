const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Test categories with existing working tests
const WORKING_TESTS = {
  controllers: [
    'authController.test.js',
    'agiController.test.js',
    'aiPortfolioController.test.js',
    'smartSipController.test.js',
    'whatsAppController.test.js'
  ],
  services: [
    'agiService.test.js',
    'aiPortfolioOptimizer.test.js',
    'smartSipService.test.js',
    'portfolioAnalyticsService.test.js'
  ],
  models: [
    'comprehensive-model-tests.js',
    'phase3-model-tests.js'
  ],
  middleware: [
    'comprehensive-middleware-tests.js',
    'phase3-middleware-tests.js'
  ]
};

// Utility functions
function log(message, type = 'info') {
  const timestamp = new Date().toISOString();
  const colors = {
    info: '\x1b[36m',    // Cyan
    success: '\x1b[32m', // Green
    error: '\x1b[31m',   // Red
    warning: '\x1b[33m', // Yellow
    reset: '\x1b[0m'     // Reset
  };
  
  console.log(`${colors[type]}[${timestamp}] ${message}${colors.reset}`);
}

function runTest(testFile, category) {
  try {
    log(`Running test: ${testFile}`, 'info');
    
    const testPath = path.join(__dirname, `__tests__/${category}`, testFile);
    
    if (!fs.existsSync(testPath)) {
      log(`Test file not found: ${testPath}`, 'warning');
      return { success: false, error: 'Test file not found' };
    }

    const command = `npx jest ${testPath} --verbose --detectOpenHandles --forceExit --maxWorkers=1 --timeout=30000`;
    const result = execSync(command, { 
      encoding: 'utf8', 
      stdio: 'pipe',
      timeout: 60000 // 1 minute timeout
    });

    // Parse test results
    const testCount = (result.match(/Tests:\s+(\d+)/) || [])[1];
    const passedCount = (result.match(/✓\s+(\d+)/) || [])[1];
    const failedCount = (result.match(/✗\s+(\d+)/) || [])[1];
    const coverageMatch = result.match(/All files\s+\|\s+(\d+(?:\.\d+)?)/);

    return {
      success: true,
      testCount: parseInt(testCount) || 0,
      passedCount: parseInt(passedCount) || 0,
      failedCount: parseInt(failedCount) || 0,
      coverage: coverageMatch ? parseFloat(coverageMatch[1]) : 0,
      output: result
    };

  } catch (error) {
    log(`Test failed: ${testFile} - ${error.message}`, 'error');
    return {
      success: false,
      error: error.message,
      output: error.stdout || error.stderr || error.message
    };
  }
}

function testCategory(category) {
  log(`\n🚀 Testing category: ${category.toUpperCase()}`, 'info');
  
  const testFiles = WORKING_TESTS[category];
  let categoryPassed = 0;
  let categoryFailed = 0;
  let totalTests = 0;
  let totalPassed = 0;
  
  for (const testFile of testFiles) {
    const result = runTest(testFile, category);
    
    if (result.success) {
      categoryPassed++;
      totalTests += result.testCount;
      totalPassed += result.passedCount;
      log(`✅ ${testFile} PASSED - ${result.passedCount}/${result.testCount} tests passed, ${result.coverage}% coverage`, 'success');
    } else {
      categoryFailed++;
      log(`❌ ${testFile} FAILED - ${result.error}`, 'error');
    }
  }
  
  log(`\n📊 ${category.toUpperCase()} Results: ${categoryPassed} passed, ${categoryFailed} failed`, 
      categoryFailed === 0 ? 'success' : 'warning');
  
  if (totalTests > 0) {
    const passRate = ((totalPassed / totalTests) * 100).toFixed(2);
    log(`   Total Tests: ${totalTests}, Passed: ${totalPassed}, Pass Rate: ${passRate}%`, 'info');
  }
  
  return {
    category,
    passed: categoryPassed,
    failed: categoryFailed,
    totalTests,
    totalPassed
  };
}

function runSimpleTests() {
  log('🧪 Starting Simple Test Runner', 'info');
  log('Testing existing working tests only', 'info');
  
  const results = {
    totalCategories: 0,
    passedCategories: 0,
    failedCategories: 0,
    totalTests: 0,
    totalPassed: 0,
    categoryResults: {}
  };
  
  for (const category of Object.keys(WORKING_TESTS)) {
    const result = testCategory(category);
    results.categoryResults[category] = result;
    results.totalCategories++;
    
    if (result.failed === 0) {
      results.passedCategories++;
    } else {
      results.failedCategories++;
    }
    
    results.totalTests += result.totalTests;
    results.totalPassed += result.totalPassed;
  }
  
  // Calculate overall pass rate
  const overallPassRate = results.totalTests > 0 ? ((results.totalPassed / results.totalTests) * 100).toFixed(2) : '0.00';
  
  log(`\n🏆 FINAL RESULTS:`, 'info');
  log(`   Categories: ${results.passedCategories}/${results.totalCategories} passed`, 
      results.failedCategories === 0 ? 'success' : 'warning');
  log(`   Tests: ${results.totalPassed}/${results.totalTests} passed`, 
      results.totalPassed === results.totalTests ? 'success' : 'warning');
  log(`   Overall Pass Rate: ${overallPassRate}%`, 
      overallPassRate === '100.00' ? 'success' : 'warning');
  
  // Save results
  const report = {
    timestamp: new Date().toISOString(),
    ...results,
    overallPassRate
  };
  
  fs.writeFileSync('simple-test-results.json', JSON.stringify(report, null, 2));
  log(`   Results saved to: simple-test-results.json`, 'info');
  
  return report;
}

// Start the simple tests
if (require.main === module) {
  runSimpleTests().catch(error => {
    log(`Fatal error in test runner: ${error.message}`, 'error');
    process.exit(1);
  });
}

module.exports = {
  runSimpleTests,
  WORKING_TESTS
}; 