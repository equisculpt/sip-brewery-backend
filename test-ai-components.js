/**
 * 🧪 TEST AI COMPONENTS
 * 
 * Simple test script to verify new AI components work correctly
 */

const { RealTimeDataFeeds } = require('./src/ai/RealTimeDataFeeds');
const { BacktestingFramework } = require('./src/ai/BacktestingFramework');
const { PerformanceMetrics } = require('./src/ai/PerformanceMetrics');

async function testAIComponents() {
  console.log('🧪 Testing AI Components...');

  try {
    // Test RealTimeDataFeeds
    console.log('📡 Testing RealTimeDataFeeds...');
    const dataFeeds = new RealTimeDataFeeds();
    console.log('✅ RealTimeDataFeeds created successfully');

    // Test BacktestingFramework
    console.log('🎯 Testing BacktestingFramework...');
    const backtesting = new BacktestingFramework();
    console.log('✅ BacktestingFramework created successfully');

    // Test PerformanceMetrics
    console.log('📊 Testing PerformanceMetrics...');
    const performance = new PerformanceMetrics();
    console.log('✅ PerformanceMetrics created successfully');

    // Test initialization
    console.log('🔄 Testing initialization...');
    await Promise.all([
      dataFeeds.initialize(),
      backtesting.initialize(),
      performance.initialize()
    ]);
    console.log('✅ All components initialized successfully');

    // Test metrics collection
    console.log('📈 Testing metrics collection...');
    const dataMetrics = dataFeeds.getMetrics();
    const backtestMetrics = backtesting.getMetrics();
    const performanceMetricsData = performance.getMetrics();

    console.log('📊 Data Feeds Metrics:', {
      totalRequests: dataMetrics.totalRequests,
      successfulRequests: dataMetrics.successfulRequests,
      cacheSize: dataMetrics.cacheSize
    });

    console.log('🎯 Backtesting Metrics:', {
      registeredStrategies: backtestMetrics.registeredStrategies,
      completedBacktests: backtestMetrics.completedBacktests
    });

    console.log('📊 Performance Metrics:', {
      totalModels: performanceMetricsData.totalModels,
      totalPredictions: performanceMetricsData.totalPredictions
    });

    console.log('🎉 All AI components tested successfully!');

  } catch (error) {
    console.error('❌ AI component test failed:', error);
    process.exit(1);
  }
}

// Run the test
testAIComponents().then(() => {
  console.log('✅ Test completed successfully');
  process.exit(0);
}).catch(error => {
  console.error('❌ Test failed:', error);
  process.exit(1);
});
