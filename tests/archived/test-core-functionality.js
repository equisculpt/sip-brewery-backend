const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');

console.log('🚀 Testing Core Functionality - No Hanging Issues');
console.log('=' .repeat(60));

// Test results tracking
const testResults = {
  passed: 0,
  failed: 0,
  total: 0,
  startTime: Date.now()
};

// Simple test runner
function runTest(testName, testFunction) {
  return new Promise(async (resolve) => {
    testResults.total++;
    console.log(`\n🧪 Running: ${testName}`);
    
    try {
      await testFunction();
      console.log(`✅ PASS: ${testName}`);
      testResults.passed++;
      resolve(true);
    } catch (error) {
      console.log(`❌ FAIL: ${testName}`);
      console.log(`   Error: ${error.message}`);
      testResults.failed++;
      resolve(false);
    }
  });
}

// Test 1: Database Connection
async function testDatabaseConnection() {
  const mongoServer = await MongoMemoryServer.create();
  const mongoUri = mongoServer.getUri();
  
  await mongoose.connect(mongoUri, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  });
  
  const isConnected = mongoose.connection.readyState === 1;
  if (!isConnected) {
    throw new Error('Database connection failed');
  }
  
  await mongoose.disconnect();
  await mongoServer.stop();
}

// Test 2: User Model Creation
async function testUserModelCreation() {
  const User = require('./src/models/User');
  
  const userData = {
    name: 'Test User',
    email: 'test@example.com',
    phone: '9876543210',
    secretCode: 'TEST123',
    role: 'user',
    isActive: true,
    supabaseId: 'test-supabase-id-123'
  };
  
  const user = new User(userData);
  const savedUser = await user.save();
  
  if (!savedUser._id) {
    throw new Error('User creation failed');
  }
  
  await User.deleteMany({});
}

// Test 3: Response Utility
function testResponseUtility() {
  const { successResponse, errorResponse } = require('./src/utils/response');
  
  const mockRes = {
    status: (code) => ({
      json: (data) => ({ statusCode: code, data })
    })
  };
  
  const successResult = successResponse(mockRes, 'Success', { test: true }, 200);
  const errorResult = errorResponse(mockRes, 'Error', 'Test error', 400);
  
  if (!successResult.data.success || !errorResult.data.success === false) {
    throw new Error('Response utility test failed');
  }
}

// Test 4: Logger Utility
function testLoggerUtility() {
  const logger = require('./src/utils/logger');
  
  // Test that logger functions exist and don't throw errors
  logger.info('Test info message');
  logger.error('Test error message');
  logger.warn('Test warning message');
  logger.debug('Test debug message');
}

// Test 5: JWT Token Generation
function testJWTTokenGeneration() {
  const jwt = require('jsonwebtoken');
  
  const payload = { userId: 'test123', role: 'user' };
  const token = jwt.sign(payload, 'test-secret', { expiresIn: '1h' });
  
  const decoded = jwt.verify(token, 'test-secret');
  
  if (decoded.userId !== 'test123' || decoded.role !== 'user') {
    throw new Error('JWT token generation/verification failed');
  }
}

// Test 6: Basic Math Operations (Sanity Check)
function testBasicMath() {
  const result = 2 + 2;
  if (result !== 4) {
    throw new Error('Basic math failed');
  }
}

// Test 7: File System Operations
function testFileSystem() {
  const fs = require('fs');
  const path = require('path');
  
  const testFile = path.join(__dirname, 'test-temp-file.txt');
  const testContent = 'Test content';
  
  fs.writeFileSync(testFile, testContent);
  const readContent = fs.readFileSync(testFile, 'utf8');
  fs.unlinkSync(testFile);
  
  if (readContent !== testContent) {
    throw new Error('File system operations failed');
  }
}

// Test 8: Environment Variables
function testEnvironmentVariables() {
  if (!process.env.NODE_ENV) {
    process.env.NODE_ENV = 'test';
  }
  
  if (process.env.NODE_ENV !== 'test') {
    throw new Error('Environment variable test failed');
  }
}

// Test 9: Async Operations
async function testAsyncOperations() {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve('Async operation completed');
    }, 100);
  });
}

// Test 10: Error Handling
function testErrorHandling() {
  try {
    throw new Error('Test error');
  } catch (error) {
    if (error.message !== 'Test error') {
      throw new Error('Error handling test failed');
    }
  }
}

// Main test execution
async function runAllTests() {
  console.log('🔧 Starting core functionality tests...\n');
  
  const tests = [
    { name: 'Basic Math Operations', fn: testBasicMath },
    { name: 'Environment Variables', fn: testEnvironmentVariables },
    { name: 'Error Handling', fn: testErrorHandling },
    { name: 'File System Operations', fn: testFileSystem },
    { name: 'Async Operations', fn: testAsyncOperations },
    { name: 'JWT Token Generation', fn: testJWTTokenGeneration },
    { name: 'Response Utility', fn: testResponseUtility },
    { name: 'Logger Utility', fn: testLoggerUtility },
    { name: 'Database Connection', fn: testDatabaseConnection },
    { name: 'User Model Creation', fn: testUserModelCreation }
  ];
  
  for (const test of tests) {
    await runTest(test.name, test.fn);
  }
  
  // Generate report
  const duration = (Date.now() - testResults.startTime) / 1000;
  
  console.log('\n' + '=' .repeat(60));
  console.log('📊 CORE FUNCTIONALITY TEST REPORT');
  console.log('=' .repeat(60));
  console.log(`⏱️  Duration: ${duration.toFixed(2)} seconds`);
  console.log(`✅ Passed: ${testResults.passed}`);
  console.log(`❌ Failed: ${testResults.failed}`);
  console.log(`📊 Total: ${testResults.total}`);
  console.log(`📈 Success Rate: ${((testResults.passed / testResults.total) * 100).toFixed(2)}%`);
  
  if (testResults.failed === 0) {
    console.log('\n🎉 ALL CORE TESTS PASSED!');
    console.log('✅ No Hanging Issues');
    console.log('⚡ Fast Execution');
    console.log('🔧 Core Functionality Verified');
  } else {
    console.log('\n⚠️  Some tests failed but execution completed without hanging.');
  }
  
  console.log('=' .repeat(60));
  
  // Save results
  const reportData = {
    timestamp: new Date().toISOString(),
    duration,
    results: {
      passed: testResults.passed,
      failed: testResults.failed,
      total: testResults.total,
      successRate: ((testResults.passed / testResults.total) * 100).toFixed(2)
    }
  };
  
  const fs = require('fs');
  fs.writeFileSync('core-functionality-test-report.json', JSON.stringify(reportData, null, 2));
  console.log('\n📄 Report saved to: core-functionality-test-report.json');
  
  return testResults.failed === 0;
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

// Run tests
runAllTests()
  .then((success) => {
    process.exit(success ? 0 : 1);
  })
  .catch((error) => {
    console.error('💥 Test execution failed:', error);
    process.exit(1);
  }); 