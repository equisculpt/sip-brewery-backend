const { execSync } = require('child_process');

console.log('🚀 Starting comprehensive test run...');

try {
  // Run the clean test first
  console.log('\n📋 Running clean working test...');
  execSync('npm test -- --testPathPattern="clean-working" --verbose --no-coverage', { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  
  console.log('\n✅ Clean test passed!');
  
  // Now run all tests
  console.log('\n📋 Running all tests...');
  execSync('npm test -- --verbose --coverage', { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  
  console.log('\n🎉 All tests completed successfully!');
  
} catch (error) {
  console.error('\n❌ Test run failed:', error.message);
  process.exit(1);
}