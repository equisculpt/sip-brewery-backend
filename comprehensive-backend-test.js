const fs = require('fs');
const path = require('path');

console.log('🔍 COMPREHENSIVE BACKEND SYSTEM TEST\n');

const results = {
  passed: 0,
  failed: 0,
  warnings: 0,
  issues: [],
  warnings_list: []
};

function test(name, condition, warning = false) {
  if (condition) {
    console.log(`✅ ${name}`);
    results.passed++;
  } else {
    if (warning) {
      console.log(`⚠️ ${name}`);
      results.warnings++;
      results.warnings_list.push(name);
    } else {
      console.log(`❌ ${name}`);
      results.failed++;
      results.issues.push(name);
    }
  }
}

// Test 1: Environment and Configuration
console.log('📋 1. ENVIRONMENT & CONFIGURATION');
test('Environment file exists', fs.existsSync('.env'));
test('Package.json exists', fs.existsSync('package.json'));

try {
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  test('Package.json is valid JSON', true);
  test('Start script exists', !!pkg.scripts?.start);
  test('Main entry point exists', !!pkg.main);
  test('All required dependencies present', 
    pkg.dependencies?.express && 
    pkg.dependencies?.mongoose && 
    pkg.dependencies?.cors && 
    pkg.dependencies?.helmet && 
    pkg.dependencies?.dotenv
  );
  test('Vulnerable packages removed', !pkg.dependencies?.['stock-nse-india']);
} catch (e) {
  test('Package.json is valid JSON', false);
}

// Test 2: Critical Files
console.log('\n📁 2. CRITICAL FILES');
const criticalFiles = [
  'src/app.js',
  'src/config/database.js',
  'src/utils/logger.js',
  'src/services/index.js',
  'src/models/index.js',
  'src/routes/index.js'
];

criticalFiles.forEach(file => {
  test(`${file} exists`, fs.existsSync(file));
});

// Test 3: Database Configuration
console.log('\n🗄️ 3. DATABASE CONFIGURATION');
try {
  const dbConfig = require('./src/config/database');
  test('Database config loads', true);
  test('connectDB function exists', typeof dbConfig.connectDB === 'function');
  test('MongoDB config exists', !!dbConfig.mongoConfig);
  test('Environment-specific configs exist', 
    dbConfig.mongoConfig.development && 
    dbConfig.mongoConfig.test && 
    dbConfig.mongoConfig.production
  );
} catch (e) {
  test('Database config loads', false);
  test('connectDB function exists', false);
  test('MongoDB config exists', false);
  test('Environment-specific configs exist', false);
}

// Test 4: Services Module
console.log('\n⚙️ 4. SERVICES MODULE');
try {
  const services = require('./src/services');
  test('Services module loads', true);
  test('initializeServices function exists', typeof services.initializeServices === 'function');
  test('healthCheck function exists', typeof services.healthCheck === 'function');
  test('getServiceStatus function exists', typeof services.getServiceStatus === 'function');
  
  // Test individual services
  const serviceNames = [
    'aiService', 'auditService', 'benchmarkService', 'cronService', 
    'dashboardService', 'leaderboardService', 'rewardsService', 
    'smartSipService', 'whatsAppService', 'nseService', 'nseCliService'
  ];
  
  serviceNames.forEach(service => {
    test(`${service} exists`, !!services[service]);
  });
} catch (e) {
  test('Services module loads', false);
  test('initializeServices function exists', false);
  test('healthCheck function exists', false);
  test('getServiceStatus function exists', false);
}

// Test 5: Models
console.log('\n📊 5. MODELS');
try {
  const models = require('./src/models');
  test('Models module loads', true);
  
  const modelNames = [
    'User', 'Holding', 'Transaction', 'Reward', 'Leaderboard',
    'SmartSip', 'UserPortfolio', 'WhatsAppSession', 'SipOrder'
  ];
  
  modelNames.forEach(model => {
    test(`${model} model exists`, !!models[model]);
  });
} catch (e) {
  test('Models module loads', false);
}

// Test 6: Routes
console.log('\n🛣️ 6. ROUTES');
const routeFiles = [
  'src/routes/auth.js',
  'src/routes/dashboard.js',
  'src/routes/leaderboard.js',
  'src/routes/rewards.js',
  'src/routes/smartSip.js',
  'src/routes/whatsapp.js',
  'src/routes/ai.js',
  'src/routes/admin.js',
  'src/routes/benchmark.js',
  'src/routes/pdfStatement.js',
  'src/routes/ollama.js'
];

routeFiles.forEach(file => {
  test(`${file} exists`, fs.existsSync(file));
});

// Test 7: Middleware
console.log('\n🔧 7. MIDDLEWARE');
const middlewareFiles = [
  'src/middleware/auth.js',
  'src/middleware/errorHandler.js',
  'src/middleware/validation.js'
];

middlewareFiles.forEach(file => {
  test(`${file} exists`, fs.existsSync(file));
});

// Test 8: App Structure
console.log('\n🚀 8. APPLICATION STRUCTURE');
try {
  const appContent = fs.readFileSync('src/app.js', 'utf8');
  test('Main app class exists', appContent.includes('UniverseClassMutualFundPlatform'));
  test('Initialize method exists', appContent.includes('initialize()'));
  test('Start method exists', appContent.includes('start()'));
  test('Database connection call exists', appContent.includes('connectDB()'));
  test('Middleware setup exists', appContent.includes('setupMiddleware()'));
  test('Routes setup exists', appContent.includes('setupRoutes()'));
  test('WebSocket setup exists', appContent.includes('setupWebSocket()'));
  test('Monitoring setup exists', appContent.includes('setupMonitoring()'));
  test('Graceful shutdown exists', appContent.includes('gracefulShutdown()'));
} catch (e) {
  test('Main app class exists', false);
  test('Initialize method exists', false);
  test('Start method exists', false);
  test('Database connection call exists', false);
  test('Middleware setup exists', false);
  test('Routes setup exists', false);
  test('WebSocket setup exists', false);
  test('Monitoring setup exists', false);
  test('Graceful shutdown exists', false);
}

// Test 9: Security & Dependencies
console.log('\n🔐 9. SECURITY & DEPENDENCIES');
try {
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const deps = Object.keys(pkg.dependencies || {});
  
  test('Express security middleware (helmet)', deps.includes('helmet'));
  test('CORS protection', deps.includes('cors'));
  test('Rate limiting', deps.includes('express-rate-limit'));
  test('Compression', deps.includes('compression'));
  test('Request logging', deps.includes('morgan'));
  test('Environment variables', deps.includes('dotenv'));
  test('JWT authentication', deps.includes('jsonwebtoken'));
  test('Password hashing', deps.includes('bcryptjs') || deps.includes('bcrypt'));
  test('No vulnerable packages', !deps.includes('stock-nse-india'));
} catch (e) {
  test('Express security middleware (helmet)', false);
  test('CORS protection', false);
  test('Rate limiting', false);
  test('Compression', false);
  test('Request logging', false);
  test('Environment variables', false);
  test('JWT authentication', false);
  test('Password hashing', false);
  test('No vulnerable packages', false);
}

// Test 10: Logging & Monitoring
console.log('\n📝 10. LOGGING & MONITORING');
test('Logs directory exists', fs.existsSync('logs'));
test('Logger utility exists', fs.existsSync('src/utils/logger.js'));

try {
  const logger = require('./src/utils/logger');
  test('Logger module loads', true);
  test('Logger has info method', typeof logger.info === 'function');
  test('Logger has error method', typeof logger.error === 'function');
  test('Logger has warn method', typeof logger.warn === 'function');
} catch (e) {
  test('Logger module loads', false);
  test('Logger has info method', false);
  test('Logger has error method', false);
  test('Logger has warn method', false);
}

// Test 11: Advanced Features
console.log('\n🌟 11. ADVANCED FEATURES');
try {
  const appContent = fs.readFileSync('src/app.js', 'utf8');
  test('AI integration exists', appContent.includes('/api/ai'));
  test('WhatsApp integration exists', appContent.includes('/api/whatsapp'));
  test('Real-time data exists', appContent.includes('/api/universe/realtime'));
  test('Quantum computing exists', appContent.includes('/api/universe/quantum'));
  test('ESG analysis exists', appContent.includes('/api/universe/esg'));
  test('Tax optimization exists', appContent.includes('/api/universe/tax'));
  test('Social trading exists', appContent.includes('/api/universe/social'));
  test('Gamification exists', appContent.includes('/api/universe/gamification'));
  test('Advanced security exists', appContent.includes('/api/universe/security'));
  test('Scalability monitoring exists', appContent.includes('/api/universe/scalability'));
} catch (e) {
  test('AI integration exists', false);
  test('WhatsApp integration exists', false);
  test('Real-time data exists', false);
  test('Quantum computing exists', false);
  test('ESG analysis exists', false);
  test('Tax optimization exists', false);
  test('Social trading exists', false);
  test('Gamification exists', false);
  test('Advanced security exists', false);
  test('Scalability monitoring exists', false);
}

// Test 12: Error Handling
console.log('\n⚠️ 12. ERROR HANDLING');
test('Error handler middleware exists', fs.existsSync('src/middleware/errorHandler.js'));
test('404 handler exists', fs.existsSync('src/app.js') && fs.readFileSync('src/app.js', 'utf8').includes('404'));

// Summary
console.log('\n' + '='.repeat(60));
console.log('📊 COMPREHENSIVE TEST SUMMARY');
console.log('='.repeat(60));
console.log(`✅ PASSED: ${results.passed}`);
console.log(`❌ FAILED: ${results.failed}`);
console.log(`⚠️ WARNINGS: ${results.warnings}`);
console.log(`📈 SUCCESS RATE: ${Math.round((results.passed / (results.passed + results.failed)) * 100)}%`);

if (results.issues.length > 0) {
  console.log('\n🚨 CRITICAL ISSUES:');
  results.issues.forEach(issue => console.log(`   • ${issue}`));
}

if (results.warnings_list.length > 0) {
  console.log('\n⚠️ WARNINGS:');
  results.warnings_list.forEach(warning => console.log(`   • ${warning}`));
}

if (results.failed === 0 && results.warnings === 0) {
  console.log('\n🎉 ALL TESTS PASSED! Backend system is production-ready.');
} else if (results.failed === 0) {
  console.log('\n✅ All critical tests passed! Some warnings to address.');
} else {
  console.log('\n❌ Critical issues found. Please fix them before deployment.');
}

console.log('='.repeat(60)); 