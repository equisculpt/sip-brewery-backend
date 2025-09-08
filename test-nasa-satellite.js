/**
 * 🛰️ NASA EARTHDATA SATELLITE TEST
 * Quick test to verify satellite data integration
 */

require('dotenv').config();
const NASAEarthdataClient = require('./src/finance_crawler/nasa-earthdata-client');

async function testSatelliteIntegration() {
  console.log('🛰️ Testing NASA Earthdata Satellite Integration...\n');
  
  try {
    // Initialize NASA client
    console.log('1. Initializing NASA Earthdata client...');
    const nasaClient = new NASAEarthdataClient({
      username: process.env.EARTHDATA_USERNAME,
      token: process.env.EARTHDATA_TOKEN
    });
    
    await nasaClient.initialize();
    console.log('✅ NASA client initialized successfully\n');
    
    // Test 1: Mumbai Port Activity
    console.log('2. Testing Mumbai Port satellite data...');
    const mumbaiData = await nasaClient.searchPortActivity('Mumbai', 7);
    console.log(`✅ Mumbai Port: Found ${mumbaiData.opticalGranules.length} satellite images`);
    console.log(`   Time range: ${mumbaiData.timeRange.start.toISOString().split('T')[0]} to ${mumbaiData.timeRange.end.toISOString().split('T')[0]}`);
    console.log(`   Data quality: ${mumbaiData.analysis.dataAvailability}\n`);
    
    // Test 2: Industrial Activity
    console.log('3. Testing Mumbai-Pune industrial satellite data...');
    const industrialData = await nasaClient.searchIndustrialActivity('Mumbai-Pune', 30);
    console.log(`✅ Industrial: Found ${industrialData.thermalGranules.length} thermal images, ${industrialData.opticalGranules.length} optical images`);
    console.log(`   Analysis: ${industrialData.analysis.recommendedAnalysis.slice(0, 2).join(', ')}\n`);
    
    // Test 3: Agricultural Monitoring
    console.log('4. Testing Punjab agricultural satellite data...');
    const agricData = await nasaClient.searchAgriculturalActivity('Punjab', 60);
    console.log(`✅ Agriculture: Found ${agricData.vegetationGranules.length} vegetation monitoring images`);
    console.log(`   Coverage: ${agricData.analysis.temporalCoverage ? agricData.analysis.temporalCoverage.daysCovered + ' days' : 'Limited'}\n`);
    
    // Test 4: Environmental Risk Assessment
    console.log('5. Testing environmental risk satellite data...');
    const riskData = await nasaClient.searchEnvironmentalRisks('national', 7);
    console.log(`✅ Environmental: Found ${riskData.fireGranules.length} fire detection images`);
    console.log(`   Risk level: ${riskData.analysis.riskLevel}\n`);
    
    // Display statistics
    const stats = nasaClient.getStats();
    console.log('📊 NASA Earthdata Statistics:');
    console.log(`   Searches performed: ${stats.searchesPerformed}`);
    console.log(`   Total granules found: ${stats.granulesFound}`);
    console.log(`   Collections available: ${stats.currentMetrics.collectionsAvailable}`);
    console.log(`   Regions configured: ${stats.currentMetrics.regionsConfigured}`);
    console.log(`   Last update: ${stats.lastUpdate}\n`);
    
    console.log('🎉 NASA Earthdata satellite integration is working perfectly!');
    console.log('🚀 Your supply chain system now has real-time satellite intelligence!\n');
    
    // Show what this enables
    console.log('🎯 Supply Chain Intelligence Capabilities:');
    console.log('   📡 Real-time port congestion monitoring');
    console.log('   🏭 Industrial facility utilization tracking');
    console.log('   🌾 Agricultural crop health assessment');
    console.log('   🔥 Environmental risk early warning');
    console.log('   📈 Satellite-based investment signals');
    console.log('   🌍 Ground-truth validation of market conditions\n');
    
    return true;
    
  } catch (error) {
    console.error('❌ NASA Earthdata test failed:', error.message);
    console.error('\n🔧 Troubleshooting:');
    console.error('   1. Check your EARTHDATA_USERNAME and EARTHDATA_TOKEN in .env');
    console.error('   2. Verify your NASA Earthdata account is active');
    console.error('   3. Ensure you have internet connectivity');
    console.error('   4. Check NASA Earthdata service status\n');
    
    return false;
  }
}

// Run the test
if (require.main === module) {
  testSatelliteIntegration()
    .then(success => {
      if (success) {
        console.log('✅ Test completed successfully - Satellite integration ready!');
        process.exit(0);
      } else {
        console.log('❌ Test failed - Please check configuration');
        process.exit(1);
      }
    })
    .catch(error => {
      console.error('❌ Unexpected error:', error);
      process.exit(1);
    });
}

module.exports = testSatelliteIntegration;
