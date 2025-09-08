const { execSync } = require('child_process');

console.log('🔍 Verifying test setup...');

try {
  // Test database connection
  console.log('\n📊 Testing database connection...');
  execSync('node -e "const mongoose = require(\"mongoose\"); mongoose.connect(\"mongodb://localhost:27017/test\").then(() => { console.log(\"✅ Database connection works\"); mongoose.disconnect(); }).catch(console.error)"', { stdio: 'inherit' });
  
  // Test model loading
  console.log('\n📦 Testing model loading...');
  execSync('node -e "const { User, Transaction, UserPortfolio } = require(\"./src/models\"); console.log(\"✅ Models loaded successfully\");"', { stdio: 'inherit' });
  
  // Test Jest setup
  console.log('\n🧪 Testing Jest setup...');
  execSync('npm test -- --testPathPattern="clean-working" --verbose --no-coverage', { stdio: 'inherit' });
  
  console.log('\n🎉 All verifications passed!');
  
} catch (error) {
  console.error('\n❌ Verification failed:', error.message);
  process.exit(1);
}