console.log('🔍 Testing Services Module...\n');

try {
  console.log('Loading services module...');
  const services = require('./src/services');
  console.log('✅ Services module loaded successfully');
  
  console.log('Checking initializeServices function...');
  if (typeof services.initializeServices === 'function') {
    console.log('✅ initializeServices function exists');
  } else {
    console.log('❌ initializeServices function missing');
  }
  
  console.log('Checking healthCheck function...');
  if (typeof services.healthCheck === 'function') {
    console.log('✅ healthCheck function exists');
  } else {
    console.log('❌ healthCheck function missing');
  }
  
  console.log('Checking individual services...');
  const serviceNames = ['aiService', 'auditService', 'benchmarkService', 'cronService', 'dashboardService'];
  serviceNames.forEach(name => {
    if (services[name]) {
      console.log(`✅ ${name} exists`);
    } else {
      console.log(`❌ ${name} missing`);
    }
  });
  
} catch (error) {
  console.log('❌ Error loading services module:');
  console.log('Error message:', error.message);
  console.log('Error stack:', error.stack);
} 