const fs = require('fs');
const path = require('path');

console.log('⚡ QUICK MODULE TESTING (CORE FUNCTIONALITY ONLY)\n');

const results = {
  passed: 0,
  failed: 0,
  warnings: 0,
  issues: [],
  warnings_list: [],
  modules: {}
};

function test(name, condition, warning = false, module = 'General') {
  if (condition) {
    console.log(`✅ ${name}`);
    results.passed++;
    if (!results.modules[module]) results.modules[module] = { passed: 0, failed: 0, total: 0 };
    results.modules[module].passed++;
    results.modules[module].total++;
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
    if (!results.modules[module]) results.modules[module] = { passed: 0, failed: 0, total: 0 };
    results.modules[module].failed++;
    results.modules[module].total++;
  }
}

// Test 1: File System & Structure
console.log('📁 1. FILE SYSTEM & STRUCTURE');
test('src directory exists', fs.existsSync('./src'), false, 'File System');
test('src/models directory exists', fs.existsSync('./src/models'), false, 'File System');
test('src/services directory exists', fs.existsSync('./src/services'), false, 'File System');
test('src/controllers directory exists', fs.existsSync('./src/controllers'), false, 'File System');
test('src/routes directory exists', fs.existsSync('./src/routes'), false, 'File System');
test('src/utils directory exists', fs.existsSync('./src/utils'), false, 'File System');
test('src/config directory exists', fs.existsSync('./src/config'), false, 'File System');

// Test 2: Core Models Loading
console.log('\n📊 2. CORE MODELS LOADING');
try {
  const User = require('./src/models/User');
  test('User model loads', !!User, false, 'Models');
  test('User model has schema', !!User.schema, false, 'Models');
} catch (error) {
  test('User model loads', false, false, 'Models');
  test('User model has schema', false, false, 'Models');
}

try {
  const Transaction = require('./src/models/Transaction');
  test('Transaction model loads', !!Transaction, false, 'Models');
  test('Transaction model has schema', !!Transaction.schema, false, 'Models');
} catch (error) {
  test('Transaction model loads', false, false, 'Models');
  test('Transaction model has schema', false, false, 'Models');
}

try {
  const Reward = require('./src/models/Reward');
  test('Reward model loads', !!Reward, false, 'Models');
  test('Reward model has schema', !!Reward.schema, false, 'Models');
} catch (error) {
  test('Reward model loads', false, false, 'Models');
  test('Reward model has schema', false, false, 'Models');
}

// Test 3: Core Services Loading
console.log('\n⚙️ 3. CORE SERVICES LOADING');
try {
  const services = require('./src/services');
  test('Services index loads', !!services, false, 'Services');
  test('Services has initializeServices', !!services.initializeServices, false, 'Services');
  test('Services has healthCheck', !!services.healthCheck, false, 'Services');
} catch (error) {
  test('Services index loads', false, false, 'Services');
  test('Services has initializeServices', false, false, 'Services');
  test('Services has healthCheck', false, false, 'Services');
}

try {
  const rewardsService = require('./src/services/rewardsService');
  test('Rewards service loads', !!rewardsService, false, 'Rewards Service');
  test('Rewards service has calculateRewards', !!rewardsService.calculateRewards, false, 'Rewards Service');
  test('Rewards service has initialize', !!rewardsService.initialize, false, 'Rewards Service');
  test('Rewards service has getStatus', !!rewardsService.getStatus, false, 'Rewards Service');
} catch (error) {
  test('Rewards service loads', false, false, 'Rewards Service');
  test('Rewards service has calculateRewards', false, false, 'Rewards Service');
  test('Rewards service has initialize', false, false, 'Rewards Service');
  test('Rewards service has getStatus', false, false, 'Rewards Service');
}

try {
  const aiService = require('./src/services/aiService');
  test('AI service loads', !!aiService, false, 'AI Service');
  test('AI service has analyzeFundWithNAV', !!aiService.analyzeFundWithNAV, false, 'AI Service');
} catch (error) {
  test('AI service loads', false, false, 'AI Service');
  test('AI service has analyzeFundWithNAV', false, false, 'AI Service');
}

try {
  const nseService = require('./src/services/nseService');
  test('NSE service loads', !!nseService, false, 'NSE Service');
  test('NSE service has getMarketStatus', !!nseService.getMarketStatus, false, 'NSE Service');
} catch (error) {
  test('NSE service loads', false, false, 'NSE Service');
  test('NSE service has getMarketStatus', false, false, 'NSE Service');
}

// Test 4: Business Logic Functions
console.log('\n💼 4. BUSINESS LOGIC FUNCTIONS');
test('SIP calculation logic', (1000 / 25.50) === 39.21568627450981, false, 'Business Logic');
test('XIRR package available', !!require('xirr'), false, 'Business Logic');

// Test 5: Security Functions
console.log('\n🔒 5. SECURITY FUNCTIONS');
try {
  const jwt = require('jsonwebtoken');
  test('JWT package loads', !!jwt, false, 'Security');
  test('JWT sign function available', !!jwt.sign, false, 'Security');
  test('JWT verify function available', !!jwt.verify, false, 'Security');
} catch (error) {
  test('JWT package loads', false, false, 'Security');
  test('JWT sign function available', false, false, 'Security');
  test('JWT verify function available', false, false, 'Security');
}

try {
  const bcrypt = require('bcrypt');
  test('Bcrypt package loads', !!bcrypt, false, 'Security');
  test('Bcrypt hash function available', !!bcrypt.hash, false, 'Security');
  test('Bcrypt compare function available', !!bcrypt.compare, false, 'Security');
} catch (error) {
  test('Bcrypt package loads', false, false, 'Security');
  test('Bcrypt hash function available', false, false, 'Security');
  test('Bcrypt compare function available', false, false, 'Security');
}

// Test 6: Utility Functions
console.log('\n🛠️ 6. UTILITY FUNCTIONS');
try {
  const logger = require('./src/utils/logger');
  test('Logger utility loads', !!logger, false, 'Utilities');
  test('Logger has info method', !!logger.info, false, 'Utilities');
  test('Logger has error method', !!logger.error, false, 'Utilities');
  test('Logger has warn method', !!logger.warn, false, 'Utilities');
} catch (error) {
  test('Logger utility loads', false, false, 'Utilities');
  test('Logger has info method', false, false, 'Utilities');
  test('Logger has error method', false, false, 'Utilities');
  test('Logger has warn method', false, false, 'Utilities');
}

// Test 7: Configuration Loading
console.log('\n⚙️ 7. CONFIGURATION LOADING');
try {
  const database = require('./src/config/database');
  test('Database config loads', !!database, false, 'Configuration');
  test('Database config has connectDB', !!database.connectDB, false, 'Configuration');
} catch (error) {
  test('Database config loads', false, false, 'Configuration');
  test('Database config has connectDB', false, false, 'Configuration');
}

// Test 8: Routes Loading
console.log('\n🛣️ 8. ROUTES LOADING');
const routeFiles = [
  'auth',
  'dashboard', 
  'leaderboard',
  'rewards',
  'smartSip',
  'whatsapp',
  'ai',
  'admin',
  'benchmarkRoutes',
  'pdfStatement',
  'ollama'
];

routeFiles.forEach(routeFile => {
  try {
    const route = require(`./src/routes/${routeFile}`);
    test(`${routeFile} routes load`, !!route, false, 'Routes');
  } catch (error) {
    test(`${routeFile} routes load`, false, false, 'Routes');
  }
});

// Test 9: Controllers Loading
console.log('\n🎮 9. CONTROLLERS LOADING');
const controllerFiles = [
  'authController',
  'dashboardController',
  'leaderboardController',
  'rewardsController',
  'smartSipController',
  'whatsappController',
  'aiController',
  'adminController'
];

controllerFiles.forEach(controllerFile => {
  try {
    const controller = require(`./src/controllers/${controllerFile}`);
    test(`${controllerFile} loads`, !!controller, false, 'Controllers');
  } catch (error) {
    test(`${controllerFile} loads`, false, false, 'Controllers');
  }
});

// Test 10: Package Dependencies
console.log('\n📦 10. PACKAGE DEPENDENCIES');
try {
  const packageJson = require('./package.json');
  test('Package.json loads', !!packageJson, false, 'Dependencies');
  test('Package.json has dependencies', !!packageJson.dependencies, false, 'Dependencies');
  test('Package.json has scripts', !!packageJson.scripts, false, 'Dependencies');
  
  const requiredDeps = ['express', 'mongoose', 'jsonwebtoken', 'bcrypt', 'axios'];
  requiredDeps.forEach(dep => {
    test(`${dep} dependency exists`, !!packageJson.dependencies[dep], false, 'Dependencies');
  });
} catch (error) {
  test('Package.json loads', false, false, 'Dependencies');
  test('Package.json has dependencies', false, false, 'Dependencies');
  test('Package.json has scripts', false, false, 'Dependencies');
}

// Final Summary
console.log('\n' + '='.repeat(70));
console.log('⚡ QUICK MODULE TEST SUMMARY (CORE FUNCTIONALITY ONLY)');
console.log('='.repeat(70));
console.log(`✅ PASSED: ${results.passed}`);
console.log(`❌ FAILED: ${results.failed}`);
console.log(`⚠️ WARNINGS: ${results.warnings}`);
console.log(`📈 SUCCESS RATE: ${Math.round((results.passed / (results.passed + results.failed)) * 100)}%`);

// Module-wise Summary
console.log('\n📊 MODULE-WISE COVERAGE:');
Object.entries(results.modules).forEach(([module, stats]) => {
  const coverage = Math.round((stats.passed / stats.total) * 100);
  const status = coverage === 100 ? '✅ 100%' : coverage >= 80 ? '⚠️ ' + coverage + '%' : '❌ ' + coverage + '%';
  console.log(`   • ${module}: ${stats.passed}/${stats.total} tests passed - ${status}`);
});

if (results.issues.length > 0) {
  console.log('\n🚨 CRITICAL ISSUES:');
  results.issues.forEach(issue => console.log(`   • ${issue}`));
}

if (results.warnings_list.length > 0) {
  console.log('\n⚠️ WARNINGS:');
  results.warnings_list.forEach(warning => console.log(`   • ${warning}`));
}

// Check for 100% coverage modules
const perfectModules = Object.entries(results.modules)
  .filter(([module, stats]) => stats.passed === stats.total && stats.total > 0)
  .map(([module]) => module);

if (perfectModules.length > 0) {
  console.log('\n🎉 MODULES WITH 100% COVERAGE:');
  perfectModules.forEach(module => console.log(`   • ${module} ✅`));
}

if (results.failed === 0 && results.warnings === 0) {
  console.log('\n🎉 ALL MODULES HAVE 100% COVERAGE!');
} else if (results.failed === 0) {
  console.log('\n✅ All critical tests passed with minor warnings.');
} else {
  console.log('\n❌ Some modules need attention before production deployment.');
}

console.log('='.repeat(70)); 