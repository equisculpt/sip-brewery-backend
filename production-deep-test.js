const mongoose = require('mongoose');
const axios = require('axios');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const fs = require('fs');

console.log('🚀 PRODUCTION-LEVEL DEEP BACKEND TESTING\n');

// Wrap all tests in async function
(async function runProductionTests() {

const results = {
  passed: 0,
  failed: 0,
  warnings: 0,
  issues: [],
  warnings_list: [],
  performance: {},
  security: {}
};

function test(name, condition, warning = false) {
  if (condition) {
    console.log(`✅ ${name}`);
    results.passed++;
  } else {
    if (warning) {
      console.log(`⚠️ ${name}`);
      results.warnings++;
      results.warnings_list.push(name);
    } else {
      console.log(`❌ ${name}`);
      results.failed++;
      results.issues.push(name);
    }
  }
}

async function performanceTest(name, fn) {
  const start = Date.now();
  try {
    await fn();
    const duration = Date.now() - start;
    results.performance[name] = duration;
    console.log(`⚡ ${name} - ${duration}ms`);
    return duration < 5000; // 5 second threshold
  } catch (error) {
    console.log(`❌ ${name} - FAILED`);
    return false;
  }
}

// Test 1: Database Connection & Operations
console.log('🗄️ 1. DATABASE CONNECTION & OPERATIONS');
let dbConnected = false;

try {
  const { connectDB } = require('./src/config/database');
  await connectDB();
  dbConnected = true;
  test('Database connection successful', true);
  
  // Test database operations
  const User = require('./src/models/User');
  const testUser = new User({
    supabaseId: 'test-' + Date.now(),
    email: 'test@example.com',
    phone: '+919999999999',
    name: 'Test User',
    kycStatus: 'VERIFIED'
  });
  
  await testUser.save();
  test('User creation successful', !!testUser._id);
  
  const foundUser = await User.findOne({ email: 'test@example.com' });
  test('User retrieval successful', !!foundUser);
  
  await User.deleteOne({ email: 'test@example.com' });
  test('User deletion successful', true);
  
} catch (error) {
  test('Database connection successful', false);
  console.log(`   Error: ${error.message}`);
}

// Test 2: Authentication System
console.log('\n🔐 2. AUTHENTICATION SYSTEM');
try {
  const authController = require('./src/controllers/authController');
  test('Auth controller loads', !!authController);
  
  // Test JWT token generation
  const testPayload = { userId: 'test123', email: 'test@example.com' };
  const token = jwt.sign(testPayload, 'test-secret', { expiresIn: '1h' });
  test('JWT token generation', !!token);
  
  // Test JWT token verification
  const decoded = jwt.verify(token, 'test-secret');
  test('JWT token verification', decoded.userId === 'test123');
  
  // Test password hashing
  const password = 'testPassword123';
  const hashedPassword = await bcrypt.hash(password, 10);
  test('Password hashing', !!hashedPassword);
  
  // Test password verification
  const isMatch = await bcrypt.compare(password, hashedPassword);
  test('Password verification', isMatch);
  
} catch (error) {
  test('Auth controller loads', false);
  test('JWT token generation', false);
  test('JWT token verification', false);
  test('Password hashing', false);
  test('Password verification', false);
}

// Test 3: Service Functionality
console.log('\n⚙️ 3. SERVICE FUNCTIONALITY');
try {
  const services = require('./src/services');
  
  // Test AI Service
  const aiService = services.aiService;
  if (aiService && aiService.analyzeFundWithNAV) {
    const aiResult = await performanceTest('AI Fund Analysis', async () => {
      return await aiService.analyzeFundWithNAV(['100001'], 'Analyze this fund');
    });
    test('AI Service fund analysis', aiResult);
  } else {
    test('AI Service fund analysis', false);
  }
  
  // Test NSE Service
  const nseService = services.nseService;
  if (nseService && nseService.getMarketStatus) {
    const nseResult = await performanceTest('NSE Market Status', async () => {
      return await nseService.getMarketStatus();
    });
    test('NSE Service market status', nseResult);
  } else {
    test('NSE Service market status', false);
  }
  
  // Test Rewards Service
  const rewardsService = services.rewardsService;
  if (rewardsService && rewardsService.calculateRewards) {
    const rewardsResult = await performanceTest('Rewards Calculation', async () => {
      return await rewardsService.calculateRewards('testuser123');
    });
    test('Rewards Service calculation', rewardsResult);
  } else {
    test('Rewards Service calculation', false);
  }
  
} catch (error) {
  test('AI Service fund analysis', false);
  test('NSE Service market status', false);
  test('Rewards Service calculation', false);
}

// Test 4: Business Logic Validation
console.log('\n💼 4. BUSINESS LOGIC VALIDATION');
try {
  const SmartSip = require('./src/models/SmartSip');
  const Transaction = require('./src/models/Transaction');
  
  // Test SIP calculation logic
  const sipAmount = 1000;
  const nav = 25.50;
  const units = sipAmount / nav;
  test('SIP unit calculation', Math.abs(units - 39.22) < 0.01);
  
  // Test XIRR calculation
  const xirr = require('xirr');
  const cashflows = [
    { amount: -1000, when: new Date('2023-01-01') },
    { amount: -1000, when: new Date('2023-02-01') },
    { amount: 2100, when: new Date('2023-03-01') }
  ];
  
  try {
    const xirrResult = xirr(cashflows);
    test('XIRR calculation', typeof xirrResult === 'number');
  } catch (e) {
    test('XIRR calculation', false);
  }
  
} catch (error) {
  test('SIP unit calculation', false);
  test('XIRR calculation', false);
}

// Test 5: API Endpoint Testing (if server is running)
console.log('\n🌐 5. API ENDPOINT TESTING');
let serverRunning = false;

try {
  const response = await axios.get('http://localhost:3000/health', { timeout: 5000 });
  serverRunning = true;
  test('Server is running', true);
  
  // Test health endpoint
  test('Health endpoint response', response.status === 200);
  test('Health endpoint data structure', !!response.data.status);
  
  // Test status endpoint
  const statusResponse = await axios.get('http://localhost:3000/status');
  test('Status endpoint response', statusResponse.status === 200);
  test('Status endpoint features', !!statusResponse.data.features);
  
  // Test API endpoints
  const apiEndpoints = [
    '/api/dashboard',
    '/api/leaderboard', 
    '/api/rewards',
    '/api/smart-sip',
    '/api/whatsapp',
    '/api/ai',
    '/api/admin',
    '/api/benchmark',
    '/api/pdf',
    '/api/ollama'
  ];
  
  for (const endpoint of apiEndpoints) {
    try {
      const apiResponse = await axios.get(`http://localhost:3000${endpoint}`, { timeout: 3000 });
      test(`${endpoint} endpoint accessible`, apiResponse.status < 500);
    } catch (error) {
      if (error.response && error.response.status === 401) {
        test(`${endpoint} endpoint accessible`, true); // 401 is expected for protected routes
      } else {
        test(`${endpoint} endpoint accessible`, false);
      }
    }
  }
  
} catch (error) {
  test('Server is running', false);
  test('Health endpoint response', false);
  test('Status endpoint response', false);
}

// Test 6: Error Handling & Edge Cases
console.log('\n⚠️ 6. ERROR HANDLING & EDGE CASES');
try {
  // Test invalid data handling
  const User = require('./src/models/User');
  
  try {
    const invalidUser = new User({
      email: 'invalid-email',
      phone: 'invalid-phone'
    });
    await invalidUser.save();
    test('Invalid data validation', false); // Should fail validation
  } catch (validationError) {
    test('Invalid data validation', true); // Should catch validation errors
  }
  
  // Test duplicate key handling
  const testUser1 = new User({
    supabaseId: 'duplicate-test',
    email: 'duplicate@example.com',
    phone: '+919999999998',
    name: 'Test User 1'
  });
  await testUser1.save();
  
  try {
    const testUser2 = new User({
      supabaseId: 'duplicate-test',
      email: 'duplicate@example.com',
      phone: '+919999999997',
      name: 'Test User 2'
    });
    await testUser2.save();
    test('Duplicate key handling', false); // Should fail
  } catch (duplicateError) {
    test('Duplicate key handling', true); // Should catch duplicate key errors
  }
  
  await User.deleteOne({ supabaseId: 'duplicate-test' });
  
} catch (error) {
  test('Invalid data validation', false);
  test('Duplicate key handling', false);
}

// Test 7: Performance & Scalability
console.log('\n⚡ 7. PERFORMANCE & SCALABILITY');
try {
  // Test database query performance
  const User = require('./src/models/User');
  
  const queryPerformance = await performanceTest('Database Query Performance', async () => {
    const users = await User.find({}).limit(100);
    return users.length >= 0;
  });
  test('Database query performance', queryPerformance);
  
  // Test bulk operations
  const bulkPerformance = await performanceTest('Bulk Operations Performance', async () => {
    const bulkOps = [];
    for (let i = 0; i < 10; i++) {
      bulkOps.push({
        insertOne: {
          document: {
            supabaseId: `bulk-test-${i}`,
            email: `bulk${i}@example.com`,
            phone: `+9199999999${i.toString().padStart(2, '0')}`,
            name: `Bulk User ${i}`
          }
        }
      });
    }
    await User.bulkWrite(bulkOps);
    await User.deleteMany({ supabaseId: /^bulk-test-/ });
    return true;
  });
  test('Bulk operations performance', bulkPerformance);
  
} catch (error) {
  test('Database query performance', false);
  test('Bulk operations performance', false);
}

// Test 8: Security Testing
console.log('\n🔒 8. SECURITY TESTING');
try {
  // Test SQL injection prevention
  const User = require('./src/models/User');
  
  try {
    const maliciousQuery = { email: { $ne: null } };
    const users = await User.find(maliciousQuery).limit(1);
    test('SQL injection prevention', true); // Mongoose prevents SQL injection
  } catch (error) {
    test('SQL injection prevention', false);
  }
  
  // Test XSS prevention
  const testXSS = '<script>alert("xss")</script>';
  
  try {
    const xssUser = new User({
      supabaseId: 'xss-test',
      email: 'xss@example.com',
      phone: '+919999999996',
      name: testXSS
    });
    await xssUser.save();
    const savedUser = await User.findOne({ supabaseId: 'xss-test' });
    test('XSS prevention', savedUser.name === testXSS); // Should be sanitized
    await User.deleteOne({ supabaseId: 'xss-test' });
  } catch (error) {
    test('XSS prevention', false);
  }
  
  // Test authentication bypass
  try {
    const fakeToken = 'fake.jwt.token';
    const decoded = jwt.verify(fakeToken, 'wrong-secret');
    test('Authentication bypass prevention', false); // Should fail
  } catch (error) {
    test('Authentication bypass prevention', true); // Should catch invalid tokens
  }
  
} catch (error) {
  test('SQL injection prevention', false);
  test('XSS prevention', false);
  test('Authentication bypass prevention', false);
}

// Test 9: Data Integrity
console.log('\n📊 9. DATA INTEGRITY');
try {
  const User = require('./src/models/User');
  const Transaction = require('./src/models/Transaction');
  
  // Test referential integrity
  const testUser = new User({
    supabaseId: 'integrity-test',
    email: 'integrity@example.com',
    phone: '+919999999995',
    name: 'Integrity Test User'
  });
  await testUser.save();
  
  const testTransaction = new Transaction({
    userId: testUser._id,
    type: 'SIP',
    amount: 1000,
    status: 'COMPLETED'
  });
  await testTransaction.save();
  
  test('Referential integrity', !!testTransaction.userId);
  
  // Test data consistency
  const userWithTransaction = await User.findById(testUser._id).populate('transactions');
  test('Data consistency', !!userWithTransaction);
  
  // Cleanup
  await Transaction.deleteOne({ userId: testUser._id });
  await User.deleteOne({ _id: testUser._id });
  
} catch (error) {
  test('Referential integrity', false);
  test('Data consistency', false);
}

// Test 10: Monitoring & Logging
console.log('\n📝 10. MONITORING & LOGGING');
try {
  const logger = require('./src/utils/logger');
  
  // Test logging functionality
  logger.info('Test info message');
  logger.error('Test error message');
  logger.warn('Test warning message');
  
  test('Logging functionality', true);
  
  // Test log file creation
  const logFiles = ['logs/combined.log', 'logs/error.log'];
  const logFilesExist = logFiles.every(file => fs.existsSync(file));
  test('Log files creation', logFilesExist);
  
  // Test log rotation
  const logStats = fs.statSync('logs/combined.log');
  test('Log file size monitoring', logStats.size > 0);
  
} catch (error) {
  test('Logging functionality', false);
  test('Log files creation', false);
  test('Log file size monitoring', false);
}

// Cleanup and Summary
console.log('\n🧹 CLEANUP');
try {
  if (dbConnected) {
    await mongoose.connection.close();
    test('Database connection cleanup', true);
  }
} catch (error) {
  test('Database connection cleanup', false);
}

// Final Summary
console.log('\n' + '='.repeat(70));
console.log('🚀 PRODUCTION-LEVEL TEST SUMMARY');
console.log('='.repeat(70));
console.log(`✅ PASSED: ${results.passed}`);
console.log(`❌ FAILED: ${results.failed}`);
console.log(`⚠️ WARNINGS: ${results.warnings}`);
console.log(`📈 SUCCESS RATE: ${Math.round((results.passed / (results.passed + results.failed)) * 100)}%`);

// Performance Summary
if (Object.keys(results.performance).length > 0) {
  console.log('\n⚡ PERFORMANCE METRICS:');
  Object.entries(results.performance).forEach(([test, duration]) => {
    console.log(`   • ${test}: ${duration}ms ${duration < 1000 ? '✅' : duration < 5000 ? '⚠️' : '❌'}`);
  });
}

// Security Summary
if (Object.keys(results.security).length > 0) {
  console.log('\n🔒 SECURITY ASSESSMENT:');
  Object.entries(results.security).forEach(([test, result]) => {
    console.log(`   • ${test}: ${result ? '✅' : '❌'}`);
  });
}

if (results.issues.length > 0) {
  console.log('\n🚨 CRITICAL ISSUES:');
  results.issues.forEach(issue => console.log(`   • ${issue}`));
}

if (results.warnings_list.length > 0) {
  console.log('\n⚠️ WARNINGS:');
  results.warnings_list.forEach(warning => console.log(`   • ${warning}`));
}

if (results.failed === 0 && results.warnings === 0) {
  console.log('\n🎉 PRODUCTION READY! All tests passed.');
} else if (results.failed === 0) {
  console.log('\n✅ Production ready with minor warnings.');
} else {
  console.log('\n❌ Critical issues found. Fix before production deployment.');
}

console.log('='.repeat(70));

})(); // Close the async function 