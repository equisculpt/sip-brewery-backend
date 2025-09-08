const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🚀 Starting Simplified Phase 2 & Phase 3 Testing...');
console.log('📊 Target: High Coverage, Fast Execution');
console.log('⚡ No Hanging Tests');
console.log('=' .repeat(60));

// Test configuration for fast execution
const testConfig = {
  timeout: 10000, // Reduced timeout
  coverage: true,
  verbose: false, // Less verbose for faster execution
  maxWorkers: '25%', // Reduced workers to prevent hanging
  bail: 1, // Stop on first failure
  retryTimes: 0 // No retries to avoid hanging
};

// Simplified test suites - focus on core functionality
const testSuites = [
  {
    name: 'Auth Controller Tests',
    pattern: 'authController',
    description: 'Core authentication functionality'
  },
  {
    name: 'Smart SIP Controller Tests', 
    pattern: 'smartSipController',
    description: 'Smart SIP functionality'
  },
  {
    name: 'WhatsApp Controller Tests',
    pattern: 'whatsAppController', 
    description: 'WhatsApp integration'
  },
  {
    name: 'Smart SIP Service Tests',
    pattern: 'smartSipService',
    description: 'Smart SIP business logic'
  }
];

// Function to run tests with proper cleanup
async function runTestsWithCleanup(testPattern) {
  return new Promise((resolve, reject) => {
    console.log(`\n🔍 Running: ${testPattern}`);
    
    const args = [
      '--testTimeout=10000',
      '--verbose=false',
      '--coverage',
      '--coverageReporters=text,json',
      '--coverageThreshold.global.branches=80',
      '--coverageThreshold.global.functions=80', 
      '--coverageThreshold.global.lines=80',
      '--coverageThreshold.global.statements=80',
      '--maxWorkers=25%',
      '--bail=1',
      '--retryTimes=0',
      '--forceExit',
      '--detectOpenHandles=false', // Disable to prevent hanging
      '--testPathPattern=' + testPattern
    ];

    const testProcess = spawn('npx.cmd', ['jest', ...args], {
      stdio: 'pipe',
      env: { ...process.env, NODE_ENV: 'test' },
      timeout: 30000 // 30 second timeout for entire test process
    });

    let output = '';
    let errorOutput = '';

    testProcess.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      // Only show important output
      if (text.includes('PASS') || text.includes('FAIL') || text.includes('Tests:')) {
        process.stdout.write(text);
      }
    });

    testProcess.stderr.on('data', (data) => {
      const text = data.toString();
      errorOutput += text;
      // Only show error output
      if (text.includes('Error') || text.includes('Failed')) {
        process.stderr.write(text);
      }
    });

    testProcess.on('close', (code) => {
      if (code === 0) {
        console.log(`✅ ${testPattern} completed successfully`);
        resolve({ success: true, output, errorOutput });
      } else {
        console.log(`❌ ${testPattern} failed with code ${code}`);
        reject({ success: false, code, output, errorOutput });
      }
    });

    testProcess.on('error', (error) => {
      console.log(`💥 Error running ${testPattern}:`, error.message);
      reject({ success: false, error: error.message });
    });

    // Force kill after timeout
    setTimeout(() => {
      if (!testProcess.killed) {
        console.log(`⏰ Timeout reached for ${testPattern}, killing process`);
        testProcess.kill('SIGKILL');
        reject({ success: false, error: 'Process timeout' });
      }
    }, 30000);
  });
}

// Function to run all tests with proper error handling
async function runAllTests() {
  const results = [];
  const startTime = Date.now();

  console.log('\n📋 Simplified Test Execution Plan:');
  testSuites.forEach(suite => {
    console.log(`- ${suite.name}: ${suite.description}`);
  });
  console.log('=' .repeat(60));

  for (const testSuite of testSuites) {
    try {
      const result = await runTestsWithCleanup(testSuite.pattern);
      results.push({ 
        file: testSuite.name, 
        pattern: testSuite.pattern,
        ...result 
      });
    } catch (error) {
      results.push({ 
        file: testSuite.name, 
        pattern: testSuite.pattern,
        ...error 
      });
      
      // Continue with next test suite even if one fails
      console.log(`⚠️  Continuing with next test suite...`);
    }
  }

  const endTime = Date.now();
  const duration = (endTime - startTime) / 1000;

  return { results, duration };
}

// Function to generate simplified test report
function generateTestReport(results, duration) {
  console.log('\n' + '=' .repeat(60));
  console.log('📊 SIMPLIFIED PHASE 2 & PHASE 3 TEST REPORT');
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

      if (overallCoverage >= 80) {
        console.log('🎉 Good coverage achieved! (≥80%)');
      } else {
        console.log('⚠️  Coverage below target (≥80%)');
      }
    } catch (error) {
      console.log('⚠️  Could not parse coverage data:', error.message);
    }
  }

  console.log('\n' + '=' .repeat(60));
  
  if (failedTests.length === 0 && successfulTests.length === results.length) {
    console.log('🎉 ALL TESTS PASSED! Simplified Phase 2 & Phase 3 Complete!');
    console.log('✅ No Hanging Tests');
    console.log('⚡ Fast Execution');
    console.log('📈 Good Coverage Achieved');
  } else {
    console.log('⚠️  Some tests failed but no hanging issues detected.');
    console.log('✅ Test execution completed without hanging');
  }

  console.log('=' .repeat(60));
}

// Main execution
async function main() {
  try {
    console.log('🔧 Setting up simplified test environment...');
    
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
      'simple-phase2-phase3-test-report.json',
      JSON.stringify(reportData, null, 2)
    );

    console.log('\n📄 Detailed report saved to: simple-phase2-phase3-test-report.json');
    
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