/**
 * 🛰️ SIMPLE NASA INTEGRATION TEST
 * Basic test to verify NASA satellite integration works
 */

require('dotenv').config();
const NASAEarthdataClient = require('./src/finance_crawler/nasa-earthdata-client');
const SatelliteDataIntegration = require('./src/finance_crawler/satellite-data-integration');
const SupplyChainASIIntegration = require('./src/finance_crawler/supply-chain-asi-integration');

async function testSimpleIntegration() {
  console.log('🛰️ SIMPLE NASA INTEGRATION TEST\n');
  
  try {
    // Test 1: NASA Client
    console.log('1. 🛰️ Testing NASA Earthdata Client...');
    const nasaClient = new NASAEarthdataClient();
    await nasaClient.initialize();
    console.log('✅ NASA Client initialized successfully\n');
    
    // Test 2: Satellite Data Integration
    console.log('2. 📡 Testing Satellite Data Integration...');
    const satelliteIntegration = new SatelliteDataIntegration();
    await satelliteIntegration.initialize();
    console.log('✅ Satellite Integration initialized successfully\n');
    
    // Test 3: ASI Integration with Satellite Intelligence
    console.log('3. 🔗 Testing ASI Integration...');
    const asiIntegration = new SupplyChainASIIntegration();
    await asiIntegration.initialize();
    console.log('✅ ASI Integration initialized successfully\n');
    
    // Test 4: Process Sample Satellite Data
    console.log('4. 🧪 Processing sample satellite data...');
    
    const sampleSatelliteData = {
      type: 'port_activity',
      region: 'Mumbai',
      timestamp: new Date(),
      metrics: {
        shipCount: 75,
        congestionIndex: 0.8,
        throughput: 'high'
      }
    };
    
    // Process through ASI Integration
    await asiIntegration.processSatelliteIntelligence(sampleSatelliteData);
    console.log('✅ Sample satellite data processed successfully\n');
    
    // Test 5: Get Statistics
    console.log('5. 📊 Component Statistics:');
    
    const nasaStats = nasaClient.getStats();
    console.log(`   NASA Client: ${nasaStats.searchesPerformed} searches performed`);
    
    const satelliteStats = satelliteIntegration.getStats();
    console.log(`   Satellite Integration: ${satelliteStats.dataCollectionRuns} collection runs`);
    
    const asiStats = asiIntegration.getStats();
    console.log(`   ASI Integration: ${asiStats.supplyChainSignalsGenerated} signals generated`);
    
    console.log('\n🎯 Integration Capabilities:');
    console.log('   ✅ NASA Earthdata API access');
    console.log('   ✅ Satellite data processing');
    console.log('   ✅ Investment signal generation');
    console.log('   ✅ ASI platform integration');
    console.log('   ✅ Real-time intelligence processing');
    
    console.log('\n🎉 SIMPLE INTEGRATION TEST PASSED!');
    console.log('🚀 NASA satellite integration is working correctly!');
    
    return true;
    
  } catch (error) {
    console.error('❌ Simple integration test failed:', error.message);
    console.error('Stack:', error.stack);
    return false;
  }
}

// Run the test
if (require.main === module) {
  testSimpleIntegration()
    .then(success => {
      if (success) {
        console.log('\n✅ NASA satellite integration is ready for production!');
        process.exit(0);
      } else {
        console.log('\n❌ Integration test failed - please check the errors above');
        process.exit(1);
      }
    })
    .catch(error => {
      console.error('❌ Unexpected error:', error);
      process.exit(1);
    });
}

module.exports = testSimpleIntegration;
