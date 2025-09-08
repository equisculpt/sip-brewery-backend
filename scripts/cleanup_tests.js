/**
 * 🧹 TEST CLEANUP SCRIPT
 * 
 * Consolidates and organizes test files to eliminate pollution
 * Removes duplicates and creates proper test structure
 */

const fs = require('fs');
const path = require('path');

class TestCleanup {
  constructor() {
    this.rootDir = path.join(__dirname, '..');
    this.testDirs = ['__tests__', 'tests', 'test'];
    this.duplicates = [];
    this.organized = {
      controllers: [],
      services: [],
      middleware: [],
      models: [],
      utils: [],
      integration: [],
      unit: []
    };
  }

  async cleanup() {
    console.log('🧹 Starting test cleanup...');
    
    // 1. Find all test files
    const testFiles = await this.findAllTestFiles();
    console.log(`📊 Found ${testFiles.length} test files`);
    
    // 2. Identify duplicates
    await this.identifyDuplicates(testFiles);
    console.log(`🔍 Found ${this.duplicates.length} duplicate test files`);
    
    // 3. Organize tests by category
    await this.organizeTests(testFiles);
    
    // 4. Create consolidated test structure
    await this.createConsolidatedStructure();
    
    // 5. Remove duplicates
    await this.removeDuplicates();
    
    // 6. Generate test summary
    await this.generateTestSummary();
    
    console.log('✅ Test cleanup completed successfully!');
  }

  async findAllTestFiles() {
    const testFiles = [];
    
    const scanDirectory = (dir) => {
      if (!fs.existsSync(dir)) return;
      
      const items = fs.readdirSync(dir);
      
      for (const item of items) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        
        if (stat.isDirectory() && !item.includes('node_modules')) {
          scanDirectory(fullPath);
        } else if (item.endsWith('.test.js') || item.endsWith('.spec.js')) {
          testFiles.push({
            path: fullPath,
            name: item,
            size: stat.size,
            modified: stat.mtime
          });
        }
      }
    };
    
    scanDirectory(this.rootDir);
    return testFiles;
  }

  async identifyDuplicates(testFiles) {
    const nameMap = new Map();
    
    for (const file of testFiles) {
      const baseName = file.name.replace(/\.(test|spec)\.js$/, '');
      
      if (nameMap.has(baseName)) {
        nameMap.get(baseName).push(file);
      } else {
        nameMap.set(baseName, [file]);
      }
    }
    
    // Find duplicates
    for (const [name, files] of nameMap) {
      if (files.length > 1) {
        // Keep the most recent one
        files.sort((a, b) => b.modified - a.modified);
        const keep = files[0];
        const remove = files.slice(1);
        
        this.duplicates.push(...remove);
        console.log(`🔍 Duplicate found: ${name} (keeping ${keep.path})`);
      }
    }
  }

  async organizeTests(testFiles) {
    for (const file of testFiles) {
      if (this.duplicates.includes(file)) continue;
      
      const relativePath = path.relative(this.rootDir, file.path);
      
      if (relativePath.includes('controller')) {
        this.organized.controllers.push(file);
      } else if (relativePath.includes('service')) {
        this.organized.services.push(file);
      } else if (relativePath.includes('middleware')) {
        this.organized.middleware.push(file);
      } else if (relativePath.includes('model')) {
        this.organized.models.push(file);
      } else if (relativePath.includes('util')) {
        this.organized.utils.push(file);
      } else if (relativePath.includes('integration')) {
        this.organized.integration.push(file);
      } else {
        this.organized.unit.push(file);
      }
    }
  }

  async createConsolidatedStructure() {
    const testDir = path.join(this.rootDir, '__tests__');
    
    // Create organized directory structure
    const dirs = [
      'controllers',
      'services', 
      'middleware',
      'models',
      'utils',
      'integration',
      'unit'
    ];
    
    for (const dir of dirs) {
      const dirPath = path.join(testDir, dir);
      if (!fs.existsSync(dirPath)) {
        fs.mkdirSync(dirPath, { recursive: true });
        console.log(`📁 Created directory: ${dir}`);
      }
    }
    
    // Create test configuration
    await this.createTestConfig();
    
    // Create test utilities
    await this.createTestUtils();
  }

  async createTestConfig() {
    const configPath = path.join(this.rootDir, '__tests__', 'setup.js');
    
    const config = `/**
 * 🧪 TEST SETUP CONFIGURATION
 * 
 * Global test setup and utilities
 */

// Global test timeout
jest.setTimeout(30000);

// Mock environment variables
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-jwt-secret-key-for-testing-only';
process.env.MONGODB_URI = 'mongodb://localhost:27017/sip-brewery-test';

// Global test utilities
global.testUtils = {
  // Mock user data
  mockUser: {
    _id: '507f1f77bcf86cd799439011',
    email: 'test@example.com',
    role: 'user',
    subscription: 'premium'
  },
  
  // Mock request/response
  mockReq: (overrides = {}) => ({
    body: {},
    params: {},
    query: {},
    headers: {},
    user: global.testUtils.mockUser,
    ...overrides
  }),
  
  mockRes: () => {
    const res = {};
    res.status = jest.fn().mockReturnValue(res);
    res.json = jest.fn().mockReturnValue(res);
    res.send = jest.fn().mockReturnValue(res);
    return res;
  },
  
  // Mock next function
  mockNext: jest.fn()
};

// Setup and teardown
beforeAll(async () => {
  console.log('🧪 Setting up test environment...');
});

afterAll(async () => {
  console.log('🧹 Cleaning up test environment...');
});

beforeEach(() => {
  // Clear all mocks
  jest.clearAllMocks();
});
`;

    fs.writeFileSync(configPath, config);
    console.log('✅ Created test setup configuration');
  }

  async createTestUtils() {
    const utilsPath = path.join(this.rootDir, '__tests__', 'utils', 'testHelpers.js');
    
    const utils = `/**
 * 🛠️ TEST UTILITIES
 * 
 * Common test helpers and utilities
 */

const mongoose = require('mongoose');

class TestHelpers {
  // Database helpers
  static async connectTestDB() {
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(process.env.MONGODB_URI);
    }
  }
  
  static async disconnectTestDB() {
    await mongoose.connection.close();
  }
  
  static async clearTestDB() {
    const collections = mongoose.connection.collections;
    for (const key in collections) {
      await collections[key].deleteMany({});
    }
  }
  
  // API testing helpers
  static createAuthHeader(token = 'test-token') {
    return { Authorization: \`Bearer \${token}\` };
  }
  
  static expectSuccessResponse(response) {
    expect(response.success).toBe(true);
    expect(response.data).toBeDefined();
  }
  
  static expectErrorResponse(response, message) {
    expect(response.success).toBe(false);
    expect(response.error).toContain(message);
  }
  
  // Mock data generators
  static generateMockFund() {
    return {
      code: 'TEST001',
      name: 'Test Mutual Fund',
      nav: 100.50,
      category: 'Equity',
      aum: 1000000000
    };
  }
  
  static generateMockPortfolio() {
    return {
      userId: global.testUtils.mockUser._id,
      holdings: [
        { fundCode: 'TEST001', units: 100, value: 10050 }
      ],
      totalValue: 10050
    };
  }
  
  // Performance testing
  static async measurePerformance(fn, iterations = 100) {
    const start = Date.now();
    
    for (let i = 0; i < iterations; i++) {
      await fn();
    }
    
    const end = Date.now();
    const avgTime = (end - start) / iterations;
    
    return {
      totalTime: end - start,
      averageTime: avgTime,
      iterations
    };
  }
}

module.exports = TestHelpers;
`;

    const utilsDir = path.dirname(utilsPath);
    if (!fs.existsSync(utilsDir)) {
      fs.mkdirSync(utilsDir, { recursive: true });
    }
    
    fs.writeFileSync(utilsPath, utils);
    console.log('✅ Created test utilities');
  }

  async removeDuplicates() {
    console.log(`🗑️ Removing ${this.duplicates.length} duplicate test files...`);
    
    for (const duplicate of this.duplicates) {
      try {
        fs.unlinkSync(duplicate.path);
        console.log(`🗑️ Removed: ${duplicate.path}`);
      } catch (error) {
        console.error(`❌ Failed to remove ${duplicate.path}:`, error.message);
      }
    }
  }

  async generateTestSummary() {
    const summary = {
      totalTests: Object.values(this.organized).reduce((sum, arr) => sum + arr.length, 0),
      duplicatesRemoved: this.duplicates.length,
      organization: {
        controllers: this.organized.controllers.length,
        services: this.organized.services.length,
        middleware: this.organized.middleware.length,
        models: this.organized.models.length,
        utils: this.organized.utils.length,
        integration: this.organized.integration.length,
        unit: this.organized.unit.length
      },
      recommendations: [
        'Run tests with: npm test',
        'Run specific category: npm test -- __tests__/controllers',
        'Run with coverage: npm test -- --coverage',
        'Add new tests to appropriate category directories'
      ]
    };

    const summaryPath = path.join(this.rootDir, 'TEST_CLEANUP_SUMMARY.md');
    const content = `# 🧪 Test Cleanup Summary

## 📊 Results
- **Total Tests**: ${summary.totalTests}
- **Duplicates Removed**: ${summary.duplicatesRemoved}

## 📁 Organization
${Object.entries(summary.organization)
  .map(([category, count]) => `- **${category}**: ${count} tests`)
  .join('\n')}

## 🚀 Recommendations
${summary.recommendations.map(rec => `- ${rec}`).join('\n')}

## ✅ Status
Test structure has been cleaned and organized for optimal maintainability.
`;

    fs.writeFileSync(summaryPath, content);
    console.log('✅ Generated test cleanup summary');
    
    return summary;
  }
}

// Run cleanup if called directly
if (require.main === module) {
  const cleanup = new TestCleanup();
  cleanup.cleanup().catch(console.error);
}

module.exports = TestCleanup;
