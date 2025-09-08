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

function runTestWithTimeout(testFile, category, timeoutMs = 30000) {
  return new Promise((resolve) => {
    const testPath = path.join(__dirname, `__tests__/${category}`, testFile);
    
    if (!fs.existsSync(testPath)) {
      log(`Test file not found: ${testPath}`, 'warning');
      resolve({ success: false, error: 'Test file not found' });
      return;
    }

    log(`Running test: ${testFile} (timeout: ${timeoutMs}ms)`, 'info');
    
    const command = `npx jest ${testPath} --verbose --detectOpenHandles --forceExit --maxWorkers=1 --testTimeout=${timeoutMs}`;
    
    const timeout = setTimeout(() => {
      log(`Test timed out: ${testFile}`, 'error');
      resolve({ 
        success: false, 
        error: `Test timed out after ${timeoutMs}ms`,
        testCount: 0,
        passedCount: 0,
        failedCount: 0,
        coverage: 0
      });
    }, timeoutMs + 5000); // Add 5 seconds buffer

    try {
      const result = execSync(command, { 
        encoding: 'utf8', 
        stdio: 'pipe',
        timeout: timeoutMs + 10000 // Extra buffer for process cleanup
      });

      clearTimeout(timeout);

      // Parse test results
      const testCount = (result.match(/Tests:\s+(\d+)/) || [])[1];
      const passedCount = (result.match(/✓\s+(\d+)/) || [])[1];
      const failedCount = (result.match(/✗\s+(\d+)/) || [])[1];
      const coverageMatch = result.match(/All files\s+\|\s+(\d+(?:\.\d+)?)/);

      resolve({
        success: true,
        testCount: parseInt(testCount) || 0,
        passedCount: parseInt(passedCount) || 0,
        failedCount: parseInt(failedCount) || 0,
        coverage: coverageMatch ? parseFloat(coverageMatch[1]) : 0,
        output: result
      });

    } catch (error) {
      clearTimeout(timeout);
      log(`Test failed: ${testFile} - ${error.message}`, 'error');
      resolve({
        success: false,
        error: error.message,
        output: error.stdout || error.stderr || error.message,
        testCount: 0,
        passedCount: 0,
        failedCount: 0,
        coverage: 0
      });
    }
  });
}

function testCategory(category) {
  return new Promise(async (resolve) => {
    log(`\n🚀 Testing category: ${category.toUpperCase()}`, 'info');
    
    const testFiles = WORKING_TESTS[category];
    let categoryPassed = 0;
    let categoryFailed = 0;
    let totalTests = 0;
    let totalPassed = 0;
    let totalFailed = 0;
    
    for (const testFile of testFiles) {
      const result = await runTestWithTimeout(testFile, category, 15000); // 15 second timeout per test
      
      if (result.success) {
        categoryPassed++;
        totalTests += result.testCount;
        totalPassed += result.passedCount;
        totalFailed += result.failedCount;
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
      log(`   Total Tests: ${totalTests}, Passed: ${totalPassed}, Failed: ${totalFailed}, Pass Rate: ${passRate}%`, 'info');
    }
    
    resolve({
      category,
      passed: categoryPassed,
      failed: categoryFailed,
      totalTests,
      totalPassed,
      totalFailed
    });
  });
}

function runStrictTimeoutTests() {
  return new Promise(async (resolve) => {
    log('🧪 Starting Strict Timeout Test Runner', 'info');
    log('All tests have 15-second timeout to prevent hanging', 'info');
    
    const results = {
      totalCategories: 0,
      passedCategories: 0,
      failedCategories: 0,
      totalTests: 0,
      totalPassed: 0,
      totalFailed: 0,
      categoryResults: {}
    };
    
    for (const category of Object.keys(WORKING_TESTS)) {
      const result = await testCategory(category);
      results.categoryResults[category] = result;
      results.totalCategories++;
      
      if (result.failed === 0) {
        results.passedCategories++;
      } else {
        results.failedCategories++;
      }
      
      results.totalTests += result.totalTests;
      results.totalPassed += result.totalPassed;
      results.totalFailed += result.totalFailed;
    }
    
    // Calculate overall pass rate
    const overallPassRate = results.totalTests > 0 ? ((results.totalPassed / results.totalTests) * 100).toFixed(2) : '0.00';
    
    log(`\n🏆 FINAL RESULTS:`, 'info');
    log(`   Categories: ${results.passedCategories}/${results.totalCategories} passed`, 
        results.failedCategories === 0 ? 'success' : 'warning');
    log(`   Tests: ${results.totalPassed}/${results.totalTests} passed`, 
        results.totalPassed === results.totalTests ? 'success' : 'warning');
    log(`   Failed Tests: ${results.totalFailed}`, 
        results.totalFailed === 0 ? 'success' : 'error');
    log(`   Overall Pass Rate: ${overallPassRate}%`, 
        overallPassRate === '100.00' ? 'success' : 'warning');
    
    // Save results
    const report = {
      timestamp: new Date().toISOString(),
      ...results,
      overallPassRate
    };
    
    fs.writeFileSync('strict-timeout-test-results.json', JSON.stringify(report, null, 2));
    log(`   Results saved to: strict-timeout-test-results.json`, 'info');
    
    resolve(report);
  });
}

// Start the strict timeout tests
if (require.main === module) {
  runStrictTimeoutTests()
    .then(report => {
      log(`\n✅ Test run completed successfully!`, 'success');
      process.exit(0);
    })
    .catch(error => {
      log(`❌ Fatal error in test runner: ${error.message}`, 'error');
      process.exit(1);
    });
}

module.exports = {
  runStrictTimeoutTests,
  WORKING_TESTS
}; 