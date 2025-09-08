const { execSync } = require('child_process');

console.log('🚀 Starting final test run...');

try {
  // Run simple test first
  console.log('\n📋 Running simple working test...');
  execSync('npm test -- --testPathPattern="simple-working" --verbose --no-coverage', { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  
  console.log('\n✅ Simple test passed!');
  
  // Run basic functionality test
  console.log('\n📋 Running basic functionality test...');
  execSync('npm test -- --testPathPattern="basic-functionality" --verbose --no-coverage', { 
    stdio: 'inherit',
    cwd: process.cwd()
  });
  
  console.log('\n✅ Basic functionality test passed!');
  
  // Run all tests
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