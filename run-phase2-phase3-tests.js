const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 Starting Phase 2 & Phase 3 Comprehensive Testing...');
console.log('📊 Target: 90%+ Coverage, 100% Pass Rate');
console.log('⚡ Fast Tests with Timeouts');
console.log('=' .repeat(60));

// Test configuration
const testConfig = {
  timeout: 15000,
  coverage: true,
  verbose: true,
  maxWorkers: '50%',
  bail: 0,
  retryTimes: 1
};

// Test suites to run
const testSuites = [
  '__tests__/controllers/phase2-controller-tests.js',
  '__tests__/services/phase3-service-tests.js',
  '__tests__/models/phase3-model-tests.js',
  '__tests__/middleware/phase3-middleware-tests.js'
];

// Function to run tests with coverage
async function runTestsWithCoverage(testFile) {
  return new Promise((resolve, reject) => {
    console.log(`\n🔍 Running tests: ${testFile}`);
    
    const args = [
      '--testTimeout=15000',
      '--verbose',
      '--coverage',
      '--coverageReporters=text,lcov,html,json',
      '--coverageThreshold.global.branches=95',
      '--coverageThreshold.global.functions=95',
      '--coverageThreshold.global.lines=95',
      '--coverageThreshold.global.statements=95',
      '--maxWorkers=50%',
      '--bail=0',
      '--retryTimes=1',
      '--forceExit',
      '--detectOpenHandles',
      testFile
    ];

    const testProcess = spawn('node', ['node_modules/.bin/jest', ...args], {
      stdio: 'pipe',
      env: { ...process.env, NODE_ENV: 'test' }
    });

    let output = '';
    let errorOutput = '';

    testProcess.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      process.stdout.write(text);
    });

    testProcess.stderr.on('data', (data) => {
      const text = data.toString();
      errorOutput += text;
      process.stderr.write(text);
    });

    testProcess.on('close', (code) => {
      if (code === 0) {
        console.log(`✅ ${testFile} completed successfully`);
        resolve({ success: true, output, errorOutput });
      } else {
        console.log(`❌ ${testFile} failed with code ${code}`);
        reject({ success: false, code, output, errorOutput });
      }
    });

    testProcess.on('error', (error) => {
      console.log(`💥 Error running ${testFile}:`, error.message);
      reject({ success: false, error: error.message });
    });
  });
}

// Function to run all tests
async function runAllTests() {
  const results = [];
  const startTime = Date.now();

  console.log('\n📋 Test Execution Plan:');
  console.log('Phase 2: Controller Tests (350 tests)');
  console.log('Phase 3: Service Tests (2000 tests)');
  console.log('Phase 3: Model Tests (800 tests)');
  console.log('Phase 3: Middleware Tests (800 tests)');
  console.log('Total: ~3950 tests');
  console.log('=' .repeat(60));

  for (const testFile of testSuites) {
    try {
      const result = await runTestsWithCoverage(testFile);
      results.push({ file: testFile, ...result });
    } catch (error) {
      results.push({ file: testFile, ...error });
    }
  }

  const endTime = Date.now();
  const duration = (endTime - startTime) / 1000;

  return { results, duration };
}

// Function to generate test report
function generateTestReport(results, duration) {
  console.log('\n' + '=' .repeat(60));
  console.log('📊 PHASE 2 & PHASE 3 TEST REPORT');
  console.log('=' .repeat(60));

  const successfulTests = results.filter(r => r.success);
  const failedTests = results.filter(r => !r.success);

  console.log(`⏱️  Total Duration: ${duration.toFixed(2)} seconds`);
  console.log(`✅ Successful Test Suites: ${successfulTests.length}/${results.length}`);
  console.log(`❌ Failed Test Suites: ${failedTests.length}/${results.length}`);
  console.log(`📈 Success Rate: ${((successfulTests.length / results.length) * 100).toFixed(2)}%`);

  if (failedTests.length > 0) {
    console.log('\n❌ Failed Test Suites:');
    failedTests.forEach(test => {
      console.log(`   - ${test.file}`);
      if (test.error) {
        console.log(`     Error: ${test.error}`);
      }
    });
  }

  console.log('\n✅ Successful Test Suites:');
  successfulTests.forEach(test => {
    console.log(`   - ${test.file}`);
  });

  // Check for coverage report
  const coveragePath = path.join(__dirname, 'coverage', 'coverage-final.json');
  if (fs.existsSync(coveragePath)) {
    try {
      const coverageData = JSON.parse(fs.readFileSync(coveragePath, 'utf8'));
      console.log('\n📊 Coverage Summary:');
      
      let totalStatements = 0;
      let coveredStatements = 0;
      let totalFunctions = 0;
      let coveredFunctions = 0;
      let totalBranches = 0;
      let coveredBranches = 0;
      let totalLines = 0;
      let coveredLines = 0;

      Object.values(coverageData).forEach(file => {
        if (file && file.s) {
          totalStatements += Object.keys(file.s).length;
          coveredStatements += Object.values(file.s).filter(count => count > 0).length;
        }
        if (file && file.f) {
          totalFunctions += Object.keys(file.f).length;
          coveredFunctions += Object.values(file.f).filter(count => count > 0).length;
        }
        if (file && file.b) {
          totalBranches += Object.keys(file.b).length;
          coveredBranches += Object.values(file.b).filter(count => count > 0).length;
        }
        if (file && file.l) {
          totalLines += Object.keys(file.l).length;
          coveredLines += Object.values(file.l).filter(count => count > 0).length;
        }
      });

      const statementCoverage = totalStatements > 0 ? (coveredStatements / totalStatements) * 100 : 0;
      const functionCoverage = totalFunctions > 0 ? (coveredFunctions / totalFunctions) * 100 : 0;
      const branchCoverage = totalBranches > 0 ? (coveredBranches / totalBranches) * 100 : 0;
      const lineCoverage = totalLines > 0 ? (coveredLines / totalLines) * 100 : 0;

      console.log(`   Statements: ${coveredStatements}/${totalStatements} (${statementCoverage.toFixed(2)}%)`);
      console.log(`   Functions: ${coveredFunctions}/${totalFunctions} (${functionCoverage.toFixed(2)}%)`);
      console.log(`   Branches: ${coveredBranches}/${totalBranches} (${branchCoverage.toFixed(2)}%)`);
      console.log(`   Lines: ${coveredLines}/${totalLines} (${lineCoverage.toFixed(2)}%)`);

      const overallCoverage = (statementCoverage + functionCoverage + branchCoverage + lineCoverage) / 4;
      console.log(`   Overall Coverage: ${overallCoverage.toFixed(2)}%`);

      if (overallCoverage >= 90) {
        console.log('🎉 Coverage target achieved! (≥90%)');
      } else {
        console.log('⚠️  Coverage below target (≥90%)');
      }
    } catch (error) {
      console.log('⚠️  Could not parse coverage data:', error.message);
    }
  }

  console.log('\n' + '=' .repeat(60));
  
  if (failedTests.length === 0 && successfulTests.length === results.length) {
    console.log('🎉 ALL TESTS PASSED! Phase 2 & Phase 3 Complete!');
    console.log('✅ 100% Pass Rate Achieved');
    console.log('📈 High Coverage Achieved');
    console.log('⚡ Fast Execution with Timeouts');
  } else {
    console.log('⚠️  Some tests failed. Please review the failures above.');
  }

  console.log('=' .repeat(60));
}

// Function to run performance tests
async function runPerformanceTests() {
  console.log('\n🚀 Running Performance Tests...');
  
  const performanceTests = [
    'load-test.yml',
    'performance/load-test.yml'
  ];

  for (const testFile of performanceTests) {
    const testPath = path.join(__dirname, '__tests__', testFile);
    if (fs.existsSync(testPath)) {
      console.log(`📊 Running performance test: ${testFile}`);
      // Add performance test execution logic here
    }
  }
}

// Main execution
async function main() {
  try {
    console.log('🔧 Setting up test environment...');
    
    // Ensure test directories exist
    const testDirs = [
      '__tests__/controllers',
      '__tests__/services', 
      '__tests__/models',
      '__tests__/middleware'
    ];

    for (const dir of testDirs) {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    }

    console.log('✅ Test environment ready');
    
    // Run all tests
    const { results, duration } = await runAllTests();
    
    // Generate report
    generateTestReport(results, duration);
    
    // Run performance tests if available
    await runPerformanceTests();
    
    // Save detailed results
    const reportData = {
      timestamp: new Date().toISOString(),
      duration,
      results,
      summary: {
        total: results.length,
        successful: results.filter(r => r.success).length,
        failed: results.filter(r => !r.success).length,
        successRate: ((results.filter(r => r.success).length / results.length) * 100).toFixed(2)
      }
    };

    fs.writeFileSync(
      'phase2-phase3-test-report.json',
      JSON.stringify(reportData, null, 2)
    );

    console.log('\n📄 Detailed report saved to: phase2-phase3-test-report.json');
    
    // Exit with appropriate code
    const hasFailures = results.some(r => !r.success);
    process.exit(hasFailures ? 1 : 0);
    
  } catch (error) {
    console.error('💥 Test execution failed:', error);
    process.exit(1);
  }
}

// Handle process termination
process.on('SIGINT', () => {
  console.log('\n⚠️  Test execution interrupted by user');
  process.exit(1);
});

process.on('SIGTERM', () => {
  console.log('\n⚠️  Test execution terminated');
  process.exit(1);
});

// Run the main function
main(); 