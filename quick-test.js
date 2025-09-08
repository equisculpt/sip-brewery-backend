const { execSync } = require('child_process');

console.log('🚀 Starting Quick Test Suite (60-second timeout)...\n');

try {
  // Run only the most critical tests with timeout
  const testCommand = 'npm test -- --testTimeout=60000 --maxWorkers=1 --verbose __tests__/services/smartSipService.test.js';
  
  console.log(`Running: ${testCommand}\n`);
  
  const result = execSync(testCommand, { 
    encoding: 'utf8',
    timeout: 60000, // 60 second timeout
    stdio: 'inherit'
  });
  
  console.log('\n✅ Quick test completed successfully!');
  console.log(result);
  
} catch (error) {
  console.log('\n❌ Quick test failed or timed out!');
  console.log('Error:', error.message);
  
  if (error.signal === 'SIGTERM') {
    console.log('⚠️  Test was terminated due to timeout (60 seconds)');
  }
  
  process.exit(1);
} 