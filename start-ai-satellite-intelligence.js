/**
 * 🤖🛰️ AI SATELLITE INTELLIGENCE STARTUP
 * Complete startup script for AI-powered satellite market prediction system
 * Integrates NASA, ESA satellite data with AI market prediction models
 */

require('dotenv').config();
const SatelliteDataIntegration = require('./src/finance_crawler/satellite-data-integration');
const SatelliteAIMarketPredictor = require('./src/finance_crawler/satellite-ai-market-predictor');
const SupplyChainASIIntegration = require('./src/finance_crawler/supply-chain-asi-integration');

class AISatelliteIntelligenceSystem {
  constructor() {
    this.satelliteIntegration = null;
    this.aiPredictor = null;
    this.asiIntegration = null;
    this.isRunning = false;
    this.predictionInterval = null;
    
    // Configuration for AI-powered satellite intelligence
    this.config = {
      satellite: {
        nasa: {
          enabled: true,
          baseUrl: 'https://cmr.earthdata.nasa.gov',
          firmsUrl: 'https://firms.modaps.eosdis.nasa.gov/api/area/csv',
          updateInterval: 3600000, // 1 hour
          regions: ['india', 'south_asia'],
          datasets: ['MODIS', 'VIIRS', 'Landsat']
        },
        esa: {
          enabled: true,
          baseUrl: 'https://catalogue.dataspace.copernicus.eu/resto/api',
          updateInterval: 3600000, // 1 hour
          regions: ['india', 'south_asia'],
          datasets: ['Sentinel-1', 'Sentinel-2', 'Sentinel-3', 'Sentinel-5P']
        },
        ai: {
          enabled: true,
          predictionInterval: 1800000, // 30 minutes
          sectors: ['agriculture', 'oil_gas', 'retail', 'shipping', 'mining'],
          confidence_threshold: 0.7,
          models: {
            ensemble: true,
            lstm: true,
            transformer: true,
            cnn: true
          }
        }
      },
      asi: {
        enabled: true,
        apiKey: process.env.ASI_API_KEY || 'demo-key',
        updateInterval: 1800000 // 30 minutes
      }
    };
  }
  
  async initialize() {
    console.log('🤖🛰️ INITIALIZING AI SATELLITE INTELLIGENCE SYSTEM\n');
    
    try {
      // Initialize Satellite Data Integration
      console.log('1. 🛰️ Initializing Satellite Data Integration...');
      this.satelliteIntegration = new SatelliteDataIntegration(this.config.satellite);
      await this.satelliteIntegration.initialize();
      console.log('✅ Satellite data integration initialized');
      
      // Initialize AI Market Predictor
      console.log('2. 🤖 Initializing AI Market Predictor...');
      this.aiPredictor = new SatelliteAIMarketPredictor(this.config.satellite.ai);
      await this.aiPredictor.initialize();
      console.log('✅ AI market predictor initialized');
      
      // Initialize ASI Integration
      console.log('3. 📊 Initializing ASI Integration...');
      this.asiIntegration = new SupplyChainASIIntegration(this.config.asi);
      await this.asiIntegration.initialize();
      console.log('✅ ASI integration initialized');
      
      // Set up event listeners
      this.setupEventListeners();
      console.log('✅ Event listeners configured');
      
      console.log('\n🎉 AI SATELLITE INTELLIGENCE SYSTEM READY!\n');
      
      return true;
      
    } catch (error) {
      console.error('❌ Initialization failed:', error.message);
      throw error;
    }
  }
  
  setupEventListeners() {
    // Listen for satellite data updates
    this.satelliteIntegration.on('dataUpdate', async (data) => {
      console.log(`📡 Satellite data update received: ${data.source}`);
      
      // Generate AI predictions from satellite data
      try {
        const predictions = await this.generateAIPredictions(data);
        
        // Send predictions to ASI
        if (predictions && this.asiIntegration) {
          await this.asiIntegration.processSatelliteIntelligence(predictions);
          console.log('🤖 AI predictions sent to ASI');
        }
        
      } catch (error) {
        console.error('❌ Error processing satellite data for AI:', error.message);
      }
    });
    
    // Listen for satellite alerts
    this.satelliteIntegration.on('alert', (alert) => {
      console.log(`🚨 Satellite alert: ${alert.type} - ${alert.message}`);
      
      // Process high-priority alerts immediately
      if (alert.priority === 'high') {
        this.processUrgentAlert(alert);
      }
    });
    
    // Listen for AI prediction updates
    if (this.aiPredictor) {
      this.aiPredictor.on('predictionUpdate', (prediction) => {
        console.log(`🔮 AI prediction update: ${prediction.sector} - ${prediction.outlook}`);
      });
    }
  }
  
  async generateAIPredictions(satelliteData) {
    if (!this.aiPredictor) return null;
    
    try {
      // Extract features from satellite data
      const features = this.extractAIFeatures(satelliteData);
      
      // Generate comprehensive market prediction
      const prediction = await this.aiPredictor.generateComprehensiveMarketPrediction(features);
      
      return {
        timestamp: new Date().toISOString(),
        source: 'ai_satellite_prediction',
        data: prediction,
        confidence: prediction.marketOutlook.confidence,
        sectors: Object.keys(prediction.predictions)
      };
      
    } catch (error) {
      console.error('❌ Error generating AI predictions:', error.message);
      return null;
    }
  }
  
  extractAIFeatures(satelliteData) {
    // Convert satellite data to AI model features
    const features = {
      // Agriculture features
      ndvi: satelliteData.vegetation?.ndvi || 0.5,
      soil_moisture: satelliteData.soil?.moisture || 0.5,
      rainfall: satelliteData.weather?.rainfall || 100,
      temperature: satelliteData.weather?.temperature || 25,
      
      // Oil & Gas features
      storage_levels: satelliteData.industrial?.storage || 0.7,
      flaring: satelliteData.industrial?.flaring || 0.3,
      tanker_count: satelliteData.shipping?.tankers || 20,
      refinery_thermal: satelliteData.industrial?.thermal || 0.8,
      
      // Retail features
      parking_density: satelliteData.commercial?.parking || 0.6,
      footfall: satelliteData.commercial?.activity || 0.7,
      construction: satelliteData.construction?.activity || 0.3,
      
      // Shipping features
      port_congestion: satelliteData.ports?.congestion || 0.5,
      vessel_count: satelliteData.shipping?.vessels || 30,
      containers: satelliteData.ports?.containers || 2000,
      
      // Mining features
      mine_activity: satelliteData.mining?.activity || 0.6,
      stockpiles: satelliteData.mining?.stockpiles || 0.5,
      rail_traffic: satelliteData.transport?.rail || 0.7
    };
    
    return features;
  }
  
  async processUrgentAlert(alert) {
    console.log('🚨 Processing urgent satellite alert...');
    
    try {
      // Generate immediate AI prediction for urgent situation
      const urgentFeatures = this.extractUrgentFeatures(alert);
      const prediction = await this.aiPredictor.generateComprehensiveMarketPrediction(urgentFeatures);
      
      // Send urgent prediction to ASI with high priority
      if (this.asiIntegration) {
        await this.asiIntegration.processUrgentSatelliteAlert({
          alert,
          prediction,
          timestamp: new Date().toISOString(),
          priority: 'urgent'
        });
      }
      
      console.log('✅ Urgent alert processed and sent to ASI');
      
    } catch (error) {
      console.error('❌ Error processing urgent alert:', error.message);
    }
  }
  
  extractUrgentFeatures(alert) {
    // Extract features from urgent alert for immediate AI processing
    const baseFeatures = {
      ndvi: 0.5, soil_moisture: 0.5, rainfall: 100, temperature: 25,
      storage_levels: 0.7, flaring: 0.3, tanker_count: 20, refinery_thermal: 0.8,
      parking_density: 0.6, footfall: 0.7, construction: 0.3,
      port_congestion: 0.5, vessel_count: 30, containers: 2000,
      mine_activity: 0.6, stockpiles: 0.5, rail_traffic: 0.7
    };
    
    // Modify features based on alert type
    switch (alert.type) {
      case 'fire':
        baseFeatures.ndvi = 0.2;
        baseFeatures.temperature = 35;
        break;
      case 'flood':
        baseFeatures.soil_moisture = 0.9;
        baseFeatures.rainfall = 300;
        break;
      case 'port_congestion':
        baseFeatures.port_congestion = 0.9;
        baseFeatures.vessel_count = 60;
        break;
      case 'industrial_activity':
        baseFeatures.refinery_thermal = 0.95;
        baseFeatures.flaring = 0.8;
        break;
    }
    
    return baseFeatures;
  }
  
  async start() {
    if (this.isRunning) {
      console.log('⚠️ AI Satellite Intelligence System is already running');
      return;
    }
    
    console.log('🚀 STARTING AI SATELLITE INTELLIGENCE SYSTEM...\n');
    
    try {
      // Initialize all components
      await this.initialize();
      
      // Start satellite data collection
      console.log('📡 Starting satellite data collection...');
      await this.satelliteIntegration.startDataCollection();
      
      // Start periodic AI predictions
      console.log('🤖 Starting AI prediction engine...');
      this.startPeriodicPredictions();
      
      // Start ASI integration
      console.log('📊 Starting ASI integration...');
      await this.asiIntegration.start();
      
      this.isRunning = true;
      
      console.log('\n🎉 AI SATELLITE INTELLIGENCE SYSTEM STARTED SUCCESSFULLY!\n');
      console.log('🌟 CAPABILITIES ACTIVE:');
      console.log('   🛰️ Real-time satellite data from NASA & ESA');
      console.log('   🤖 AI-powered market predictions');
      console.log('   📊 ASI integration for investment signals');
      console.log('   🚨 Automated alert processing');
      console.log('   📈 Multi-sector analysis (Agriculture, Oil & Gas, Retail, Shipping, Mining)');
      console.log('   🎯 Indian market focus with local stock recommendations');
      
      // Display system status
      this.displaySystemStatus();
      
    } catch (error) {
      console.error('❌ Failed to start AI Satellite Intelligence System:', error.message);
      throw error;
    }
  }
  
  startPeriodicPredictions() {
    const interval = this.config.satellite.ai.predictionInterval;
    
    this.predictionInterval = setInterval(async () => {
      try {
        console.log('🔮 Generating periodic AI market predictions...');
        
        // Get latest satellite data
        const latestData = await this.satelliteIntegration.getLatestData();
        
        if (latestData) {
          // Generate AI predictions
          const predictions = await this.generateAIPredictions(latestData);
          
          if (predictions) {
            // Send to ASI
            await this.asiIntegration.processSatelliteIntelligence(predictions);
            console.log('✅ Periodic AI predictions generated and sent to ASI');
          }
        }
        
      } catch (error) {
        console.error('❌ Error in periodic prediction generation:', error.message);
      }
    }, interval);
    
    console.log(`⏰ Periodic AI predictions scheduled every ${interval / 60000} minutes`);
  }
  
  displaySystemStatus() {
    console.log('\n📊 SYSTEM STATUS:');
    console.log(`   🛰️ Satellite Integration: ${this.satelliteIntegration ? '✅ Active' : '❌ Inactive'}`);
    console.log(`   🤖 AI Predictor: ${this.aiPredictor ? '✅ Active' : '❌ Inactive'}`);
    console.log(`   📊 ASI Integration: ${this.asiIntegration ? '✅ Active' : '❌ Inactive'}`);
    console.log(`   ⏰ Prediction Interval: ${this.config.satellite.ai.predictionInterval / 60000} minutes`);
    console.log(`   🎯 Sectors Monitored: ${this.config.satellite.ai.sectors.length}`);
    console.log(`   🌍 Regions: ${this.config.satellite.nasa.regions.join(', ')}`);
    
    console.log('\n🔗 DATA SOURCES:');
    console.log(`   NASA Earthdata: ${this.config.satellite.nasa.enabled ? '✅ Enabled' : '❌ Disabled'}`);
    console.log(`   ESA Copernicus: ${this.config.satellite.esa.enabled ? '✅ Enabled' : '❌ Disabled'}`);
    console.log(`   AI Models: ${Object.keys(this.config.satellite.ai.models).filter(m => this.config.satellite.ai.models[m]).join(', ')}`);
    
    console.log('\n💡 NEXT STEPS:');
    console.log('   1. Monitor console for real-time satellite data updates');
    console.log('   2. Check ASI dashboard for AI-generated investment signals');
    console.log('   3. Review AI predictions for market opportunities');
    console.log('   4. Set up alerts for high-confidence predictions');
    
    console.log('\n🛑 To stop the system, press Ctrl+C\n');
  }
  
  async stop() {
    if (!this.isRunning) {
      console.log('⚠️ AI Satellite Intelligence System is not running');
      return;
    }
    
    console.log('🛑 STOPPING AI SATELLITE INTELLIGENCE SYSTEM...');
    
    try {
      // Stop periodic predictions
      if (this.predictionInterval) {
        clearInterval(this.predictionInterval);
        this.predictionInterval = null;
        console.log('✅ Periodic predictions stopped');
      }
      
      // Stop satellite data collection
      if (this.satelliteIntegration) {
        await this.satelliteIntegration.stop();
        console.log('✅ Satellite data collection stopped');
      }
      
      // Stop ASI integration
      if (this.asiIntegration) {
        await this.asiIntegration.stop();
        console.log('✅ ASI integration stopped');
      }
      
      this.isRunning = false;
      console.log('✅ AI Satellite Intelligence System stopped successfully');
      
    } catch (error) {
      console.error('❌ Error stopping system:', error.message);
    }
  }
  
  getSystemStats() {
    return {
      isRunning: this.isRunning,
      components: {
        satelliteIntegration: !!this.satelliteIntegration,
        aiPredictor: !!this.aiPredictor,
        asiIntegration: !!this.asiIntegration
      },
      config: this.config,
      uptime: this.isRunning ? Date.now() - this.startTime : 0
    };
  }
}

// Create and start the AI Satellite Intelligence System
async function main() {
  const system = new AISatelliteIntelligenceSystem();
  
  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    console.log('\n🛑 Received shutdown signal...');
    await system.stop();
    process.exit(0);
  });
  
  process.on('SIGTERM', async () => {
    console.log('\n🛑 Received termination signal...');
    await system.stop();
    process.exit(0);
  });
  
  try {
    system.startTime = Date.now();
    await system.start();
    
    // Keep the process running
    console.log('🔄 AI Satellite Intelligence System is running...');
    console.log('   Press Ctrl+C to stop');
    
  } catch (error) {
    console.error('❌ Failed to start AI Satellite Intelligence System:', error);
    process.exit(1);
  }
}

// Run if this file is executed directly
if (require.main === module) {
  main().catch(error => {
    console.error('❌ Unexpected error:', error);
    process.exit(1);
  });
}

module.exports = AISatelliteIntelligenceSystem;
