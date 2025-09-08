const fs = require('fs');
const path = require('path');

console.log('🔍 Starting Comprehensive Backend System Test...\n');

// Test Results
const testResults = {
  passed: 0,
  failed: 0,
  issues: [],
  warnings: []
};

function logTest(testName, passed, message = '') {
  if (passed) {
    console.log(`✅ ${testName}`);
    testResults.passed++;
  } else {
    console.log(`❌ ${testName}: ${message}`);
    testResults.failed++;
    testResults.issues.push({ test: testName, message });
  }
}

function logWarning(testName, message) {
  console.log(`⚠️ ${testName}: ${message}`);
  testResults.warnings.push({ test: testName, message });
}

// Test 1: Check if .env file exists
console.log('📋 Test 1: Environment Configuration');
if (fs.existsSync('.env')) {
  logTest('Environment file exists', true);
} else {
  logTest('Environment file exists', false, 'Missing .env file');
}

// Test 2: Check package.json
console.log('\n📦 Test 2: Package Configuration');
try {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  
  if (packageJson.dependencies) {
    logTest('Package.json has dependencies', true);
  } else {
    logTest('Package.json has dependencies', false, 'No dependencies found');
  }
  
  if (packageJson.scripts && packageJson.scripts.start) {
    logTest('Start script exists', true);
  } else {
    logTest('Start script exists', false, 'Missing start script');
  }
  
  // Check for security vulnerabilities
  if (packageJson.dependencies && packageJson.dependencies['stock-nse-india']) {
    logTest('Vulnerable package removed', false, 'stock-nse-india still present');
  } else {
    logTest('Vulnerable package removed', true);
  }
} catch (error) {
  logTest('Package.json is valid JSON', false, error.message);
}

// Test 3: Check critical files exist
console.log('\n📁 Test 3: Critical Files');
const criticalFiles = [
  'src/app.js',
  'src/config/database.js',
  'src/utils/logger.js',
  'src/services/index.js',
  'src/models/index.js',
  'src/routes/index.js'
];

criticalFiles.forEach(file => {
  if (fs.existsSync(file)) {
    logTest(`${file} exists`, true);
  } else {
    logTest(`${file} exists`, false, `Missing ${file}`);
  }
});

// Test 4: Check database configuration
console.log('\n🗄️ Test 4: Database Configuration');
try {
  const dbConfig = require('./src/config/database');
  
  if (dbConfig.connectDB) {
    logTest('Database connection function exists', true);
  } else {
    logTest('Database connection function exists', false, 'Missing connectDB function');
  }
  
  if (dbConfig.mongoConfig) {
    logTest('MongoDB configuration exists', true);
  } else {
    logTest('MongoDB configuration exists', false, 'Missing MongoDB configuration');
  }
} catch (error) {
  logTest('Database configuration loads', false, error.message);
}

// Test 5: Check services
console.log('\n⚙️ Test 5: Services');
try {
  const services = require('./src/services');
  
  if (services.initializeServices) {
    logTest('Services initialization function exists', true);
  } else {
    logTest('Services initialization function exists', false, 'Missing initializeServices function');
  }
  
  if (services.healthCheck) {
    logTest('Health check function exists', true);
  } else {
    logTest('Health check function exists', false, 'Missing healthCheck function');
  }
} catch (error) {
  logTest('Services module loads', false, error.message);
}

// Test 6: Check models
console.log('\n📊 Test 6: Models');
try {
  const models = require('./src/models');
  
  const requiredModels = ['User', 'Holding', 'Transaction', 'Reward', 'Leaderboard'];
  requiredModels.forEach(model => {
    if (models[model]) {
      logTest(`${model} model exists`, true);
    } else {
      logTest(`${model} model exists`, false, `Missing ${model} model`);
    }
  });
} catch (error) {
  logTest('Models module loads', false, error.message);
}

// Test 7: Check routes
console.log('\n🛣️ Test 7: Routes');
const routeFiles = [
  'src/routes/auth.js',
  'src/routes/dashboard.js',
  'src/routes/leaderboard.js',
  'src/routes/rewards.js',
  'src/routes/smartSip.js',
  'src/routes/whatsapp.js',
  'src/routes/ai.js',
  'src/routes/admin.js'
];

routeFiles.forEach(file => {
  if (fs.existsSync(file)) {
    logTest(`${file} exists`, true);
  } else {
    logTest(`${file} exists`, false, `Missing ${file}`);
  }
});

// Test 8: Check middleware
console.log('\n🔧 Test 8: Middleware');
const middlewareFiles = [
  'src/middleware/auth.js',
  'src/middleware/errorHandler.js',
  'src/middleware/validation.js'
];

middlewareFiles.forEach(file => {
  if (fs.existsSync(file)) {
    logTest(`${file} exists`, true);
  } else {
    logTest(`${file} exists`, false, `Missing ${file}`);
  }
});

// Test 9: Check for common issues
console.log('\n🔍 Test 9: Common Issues');

// Check for hardcoded credentials
const filesToCheck = [
  'src/config/database.js',
  'src/services/aiService.js',
  'src/services/nseService.js'
];

filesToCheck.forEach(file => {
  if (fs.existsSync(file)) {
    const content = fs.readFileSync(file, 'utf8');
    
    // Check for hardcoded API keys
    if (content.includes('AIzaSyDc63xZUJktleMdGwfGILfp5oUITQ3znpM')) {
      logWarning(`${file}`, 'Contains hardcoded API key');
    }
    
    // Check for hardcoded database credentials
    if (content.includes('jgnDnev2mHVToKnJ')) {
      logWarning(`${file}`, 'Contains hardcoded database credentials');
    }
  }
});

// Test 10: Check app.js structure
console.log('\n🚀 Test 10: Application Structure');
try {
  const appContent = fs.readFileSync('src/app.js', 'utf8');
  
  if (appContent.includes('class UniverseClassMutualFundPlatform')) {
    logTest('Main application class exists', true);
  } else {
    logTest('Main application class exists', false, 'Missing UniverseClassMutualFundPlatform class');
  }
  
  if (appContent.includes('initialize()')) {
    logTest('Initialize method exists', true);
  } else {
    logTest('Initialize method exists', false, 'Missing initialize method');
  }
  
  if (appContent.includes('start()')) {
    logTest('Start method exists', true);
  } else {
    logTest('Start method exists', false, 'Missing start method');
  }
  
  if (appContent.includes('connectDB()')) {
    logTest('Database connection call exists', true);
  } else {
    logTest('Database connection call exists', false, 'Missing database connection call');
  }
} catch (error) {
  logTest('App.js structure check', false, error.message);
}

// Test 11: Check for missing dependencies
console.log('\n📦 Test 11: Dependencies');
try {
  const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  const dependencies = Object.keys(packageJson.dependencies || {});
  
  const requiredDeps = [
    'express',
    'mongoose',
    'cors',
    'helmet',
    'dotenv',
    'winston',
    'axios',
    'bcryptjs',
    'jsonwebtoken'
  ];
  
  requiredDeps.forEach(dep => {
    if (dependencies.includes(dep)) {
      logTest(`${dep} dependency exists`, true);
    } else {
      logTest(`${dep} dependency exists`, false, `Missing ${dep} dependency`);
    }
  });
} catch (error) {
  logTest('Dependencies check', false, error.message);
}

// Test 12: Check logs directory
console.log('\n📝 Test 12: Logging');
if (fs.existsSync('logs')) {
  logTest('Logs directory exists', true);
} else {
  logTest('Logs directory exists', false, 'Missing logs directory');
}

// Summary
console.log('\n' + '='.repeat(60));
console.log('📊 COMPREHENSIVE TEST SUMMARY');
console.log('='.repeat(60));
console.log(`✅ Passed: ${testResults.passed}`);
console.log(`❌ Failed: ${testResults.failed}`);
console.log(`⚠️ Warnings: ${testResults.warnings.length}`);

if (testResults.issues.length > 0) {
  console.log('\n🚨 CRITICAL ISSUES FOUND:');
  testResults.issues.forEach(issue => {
    console.log(`   • ${issue.test}: ${issue.message}`);
  });
}

if (testResults.warnings.length > 0) {
  console.log('\n⚠️ WARNINGS:');
  testResults.warnings.forEach(warning => {
    console.log(`   • ${warning.test}: ${warning.message}`);
  });
}

if (testResults.failed === 0 && testResults.warnings.length === 0) {
  console.log('\n🎉 All tests passed! Backend system is ready.');
} else if (testResults.failed === 0) {
  console.log('\n✅ All critical tests passed! Some warnings to address.');
} else {
  console.log('\n❌ Critical issues found. Please fix them before running the backend.');
}

console.log('\n' + '='.repeat(60)); 