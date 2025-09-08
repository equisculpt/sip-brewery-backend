const fs = require('fs');
const path = require('path');

console.log('🔧 Starting comprehensive test fix...');

// 1. Fix all test files with syntax errors
const testFiles = [
  '__tests__/simple-working.test.js',
  '__tests__/simple-working-test.js',
  '__tests__/working-test.test.js'
];

// Remove problematic test files that are duplicates or have syntax errors
testFiles.forEach(file => {
  if (fs.existsSync(file)) {
    console.log(`🗑️  Removing problematic file: ${file}`);
    fs.unlinkSync(file);
  }
});

// 2. Create a clean, working test file
const cleanTestContent = `const mongoose = require('mongoose');
const { User, UserPortfolio, Transaction } = require('../src/models');

describe('Simple Working Test', () => {
  test('should connect to database', async () => {
    expect(mongoose.connection.readyState).toBe(1);
  });

  test('should create test user', async () => {
    const user = await global.testUtils.createTestUser();
    expect(user).toBeDefined();
    expect(user.name).toBe('Test User');
  });

  test('should create test portfolio', async () => {
    const user = await global.testUtils.createTestUser();
    const portfolio = await global.testUtils.createTestPortfolio(user._id);
    expect(portfolio).toBeDefined();
    expect(portfolio.userId.toString()).toBe(user._id.toString());
  });

  test('should create test transaction', async () => {
    const user = await global.testUtils.createTestUser();
    const transaction = await global.testUtils.createTestTransaction(user._id);
    expect(transaction).toBeDefined();
    expect(transaction.userId.toString()).toBe(user._id.toString());
    expect(transaction.nav).toBeDefined();
  });
});`;

fs.writeFileSync('__tests__/clean-working.test.js', cleanTestContent);
console.log('✅ Created clean test file');

// 3. Fix Jest configuration
const jestConfig = `module.exports = {
  testEnvironment: 'node',
  setupFilesAfterEnv: ['<rootDir>/__tests__/setup.js'],
  testMatch: [
    '**/__tests__/**/*.test.js',
    '**/__tests__/**/*.spec.js'
  ],
  collectCoverageFrom: [
    'src/**/*.js',
    '!src/**/*.test.js',
    '!src/**/*.spec.js'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html'],
  testTimeout: 30000,
  verbose: true,
  forceExit: true,
  detectOpenHandles: true
};`;

fs.writeFileSync('jest.config.js', jestConfig);
console.log('✅ Fixed Jest configuration');

// 4. Create a comprehensive test runner
const testRunnerContent = `const { execSync } = require('child_process');

console.log('🚀 Starting comprehensive test run...');

try {
  // Run the clean test first
  console.log('\\n📋 Running clean working test...');
  execSync('npm test -- --testPathPattern="clean-working" --verbose --no-coverage', { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  
  console.log('\\n✅ Clean test passed!');
  
  // Now run all tests
  console.log('\\n📋 Running all tests...');
  execSync('npm test -- --verbose --coverage', { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  
  console.log('\\n🎉 All tests completed successfully!');
  
} catch (error) {
  console.error('\\n❌ Test run failed:', error.message);
  process.exit(1);
}`;

fs.writeFileSync('run-comprehensive-tests.js', testRunnerContent);
console.log('✅ Created comprehensive test runner');

// 5. Fix package.json test script
const packageJsonPath = 'package.json';
if (fs.existsSync(packageJsonPath)) {
  const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
  packageJson.scripts.test = 'jest --detectOpenHandles --forceExit';
  fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
  console.log('✅ Fixed package.json test script');
}

// 6. Create a simple test verification script
const verificationScript = `const { execSync } = require('child_process');

console.log('🔍 Verifying test setup...');

try {
  // Test database connection
  console.log('\\n📊 Testing database connection...');
  execSync('node -e "const mongoose = require(\\"mongoose\\"); mongoose.connect(\\"mongodb://localhost:27017/test\\").then(() => { console.log(\\"✅ Database connection works\\"); mongoose.disconnect(); }).catch(console.error)"', { stdio: 'inherit' });
  
  // Test model loading
  console.log('\\n📦 Testing model loading...');
  execSync('node -e "const { User, Transaction, UserPortfolio } = require(\\"./src/models\\"); console.log(\\"✅ Models loaded successfully\\");"', { stdio: 'inherit' });
  
  // Test Jest setup
  console.log('\\n🧪 Testing Jest setup...');
  execSync('npm test -- --testPathPattern="clean-working" --verbose --no-coverage', { stdio: 'inherit' });
  
  console.log('\\n🎉 All verifications passed!');
  
} catch (error) {
  console.error('\\n❌ Verification failed:', error.message);
  process.exit(1);
}`;

fs.writeFileSync('verify-test-setup.js', verificationScript);
console.log('✅ Created verification script');

console.log('\\n🎯 Comprehensive fix completed!');
console.log('\\n📋 Next steps:');
console.log('1. Run: node verify-test-setup.js');
console.log('2. Run: node run-comprehensive-tests.js');
console.log('3. Or run: npm test'); 