// Simple test to check basic functionality
console.log('🧪 Starting simple test...');

try {
  // Test basic require
  console.log('📦 Testing require...');
  const { RealTimeDataFeeds } = require('./src/ai/RealTimeDataFeeds');
  console.log('✅ RealTimeDataFeeds loaded');

  // Test instantiation
  console.log('🔧 Testing instantiation...');
  const dataFeeds = new RealTimeDataFeeds();
  console.log('✅ RealTimeDataFeeds instantiated');

  console.log('🎉 Simple test passed!');
} catch (error) {
  console.error('❌ Simple test failed:', error.message);
  console.error('Stack:', error.stack);
}
