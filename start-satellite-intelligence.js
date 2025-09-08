/**
 * 🛰️ SATELLITE INTELLIGENCE STARTUP SCRIPT
 * Launch your complete NASA-powered supply chain intelligence system
 */

require('dotenv').config();
const SupplyChainIntelligenceSystem = require('./src/finance_crawler/supply-chain-setup');

async function startSatelliteIntelligenceSystem() {
  console.log('🛰️ STARTING NASA SATELLITE INTELLIGENCE SYSTEM\n');
  
  try {
    // Initialize the complete system
    console.log('🚀 Initializing Supply Chain Intelligence System...');
    const system = new SupplyChainIntelligenceSystem({
      // Enable all components including satellite intelligence
      enableSatelliteIntelligence: true,
      enableASIIntegration: true,
      enableCommodityMonitoring: true,
      enableLogisticsTracking: true,
      enableManufacturingMonitoring: true,
      enableRiskEngine: true,
      
      // System configuration
      autoStart: true,
      enableHealthChecks: true,
      enableReporting: true,
      enableAlerting: true,
      
      // Data collection intervals
      healthCheckInterval: 10 * 60 * 1000, // 10 minutes
      reportGenerationInterval: 60 * 60 * 1000, // 1 hour
      alertCheckInterval: 5 * 60 * 1000, // 5 minutes
      
      dataPath: './data/satellite-intelligence-system'
    });
    
    // Set up graceful shutdown
    process.on('SIGINT', async () => {
      console.log('\n🛑 Received shutdown signal...');
      await system.stop();
      console.log('✅ System stopped gracefully');
      process.exit(0);
    });

    process.on('SIGTERM', async () => {
      console.log('\n🛑 Received termination signal...');
      await system.stop();
      console.log('✅ System stopped gracefully');
      process.exit(0);
    });

    // Initialize and start
    await system.initialize();
    console.log('✅ System initialized successfully\n');

    // Display system status
    const status = system.getSystemStatus();
    console.log('📊 SYSTEM STATUS:');
    console.log(`   Status: ${status.status}`);
    console.log(`   Components: ${Object.keys(status.componentsStatus).length}`);
    console.log(`   Start Time: ${status.startTime}`);

    console.log('\n📡 ACTIVE COMPONENTS:');
    for (const [component, componentStatus] of Object.entries(status.componentsStatus)) {
      const emoji = componentStatus === 'active' ? '✅' : componentStatus === 'initialized' ? '🔄' : '❌';
      console.log(`   ${emoji} ${component}: ${componentStatus}`);
    }

    // Show NASA credentials status
    console.log('\n🛰️ NASA EARTHDATA STATUS:');
    console.log(`   Username: ${process.env.EARTHDATA_USERNAME ? '✅ Configured' : '❌ Missing'}`);
    console.log(`   Token: ${process.env.EARTHDATA_TOKEN ? '✅ Configured' : '❌ Missing'}`);

    // Display capabilities
    console.log('\n🎯 SATELLITE INTELLIGENCE CAPABILITIES:');
    console.log('   🚢 Port Activity Monitoring');
    console.log('      • Mumbai, Chennai, Kolkata ports');
    console.log('      • Ship traffic analysis');
    console.log('      • Congestion assessment');
    console.log('      • Investment signals for logistics stocks');

    console.log('   🏭 Industrial Activity Tracking');
    console.log('      • Mumbai-Pune, Chennai-Bangalore corridors');
    console.log('      • Thermal signature analysis');
    console.log('      • Production utilization monitoring');
    console.log('      • Investment signals for manufacturing stocks');

    console.log('   🌾 Agricultural Monitoring');
    console.log('      • Punjab, Haryana, Maharashtra regions');
    console.log('      • Crop health via NDVI analysis');
    console.log('      • Harvest timing prediction');
    console.log('      • Investment signals for FMCG/Agri stocks');

    console.log('   ⚠️ Environmental Risk Assessment');
    console.log('      • Real-time fire detection');
    console.log('      • Flood monitoring');
    console.log('      • Air quality assessment');
    console.log('      • Risk-based defensive positioning');

    console.log('\n💰 INVESTMENT INTELLIGENCE:');
    console.log('   📈 Real-time satellite data → Investment signals');
    console.log('   🎯 Ground-truth validation of market conditions');
    console.log('   ⚡ Early warning for supply chain disruptions');
    console.log('   🔍 Competitive intelligence on facilities');
    console.log('   📊 Integration with ASI Financial Analysis Platform');

    console.log('\n🔄 DATA SOURCES:');
    console.log('   🛰️ NASA FIRMS - Fire detection (Real-time)');
    console.log('   🛰️ NASA MODIS - Land/ocean monitoring (Daily)');
    console.log('   🛰️ NASA VIIRS - Day/night imagery (Daily)');
    console.log('   🛰️ USGS Landsat - High-res optical (Weekly)');
    console.log('   🛰️ ESA Sentinel - SAR/optical (Weekly)');

    console.log('\n🎉 NASA SATELLITE INTELLIGENCE SYSTEM IS LIVE!');
    console.log('📡 Monitoring supply chains across India with satellite data...');
    console.log('💡 Generating investment insights from space-based intelligence...');
    console.log('🚀 Your ASI platform now has institutional-grade satellite capabilities!');

    console.log('\n⌨️  Press Ctrl+C to stop the system gracefully\n');

    // Keep the process running
    await new Promise(() => {}); // Run indefinitely until stopped

  } catch (error) {
    console.error('❌ Failed to start satellite intelligence system:', error.message);
    console.error('Stack:', error.stack);
    process.exit(1);
  }
}

// Start the system
if (require.main === module) {
  startSatelliteIntelligenceSystem()
    .catch(error => {
      console.error('❌ Startup failed:', error);
      process.exit(1);
    });
}

module.exports = startSatelliteIntelligenceSystem;
