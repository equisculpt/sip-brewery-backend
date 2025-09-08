#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

/**
 * Comprehensive Test File Cleanup Script
 * 
 * This script addresses the massive test file pollution in the root directory
 * by organizing files into proper directory structures and removing duplicates.
 */

class TestCleanupManager {
  constructor() {
    this.rootDir = process.cwd();
    this.stats = {
      iterationFiles: 0,
      testFiles: 0,
      reportFiles: 0,
      moved: 0,
      deleted: 0,
      errors: 0
    };
  }

  /**
   * Main cleanup execution
   */
  async cleanup() {
    console.log('🧹 Starting comprehensive test cleanup...\n');
    
    try {
      // Create organized directory structure
      await this.createDirectoryStructure();
      
      // Clean up test iteration files (400+ files)
      await this.cleanupTestIterations();
      
      // Organize remaining test files
      await this.organizeTestFiles();
      
      // Clean up report and output files
      await this.cleanupReportFiles();
      
      // Create proper test structure
      await this.createProperTestStructure();
      
      // Generate cleanup summary
      this.generateSummary();
      
    } catch (error) {
      console.error('❌ Cleanup failed:', error.message);
      process.exit(1);
    }
  }

  /**
   * Create organized directory structure
   */
  async createDirectoryStructure() {
    console.log('📁 Creating organized directory structure...');
    
    const directories = [
      'tests/archived',
      'tests/archived/iterations',
      'tests/archived/reports',
      'tests/archived/outputs',
      'tests/unit',
      'tests/integration',
      'tests/e2e',
      'tests/performance',
      'tests/fixtures',
      'tests/utils'
    ];

    for (const dir of directories) {
      const fullPath = path.join(this.rootDir, dir);
      if (!fs.existsSync(fullPath)) {
        fs.mkdirSync(fullPath, { recursive: true });
        console.log(`  ✓ Created: ${dir}`);
      }
    }
  }

  /**
   * Clean up the massive test iteration pollution
   */
  async cleanupTestIterations() {
    console.log('\n🗂️  Cleaning up test iteration files...');
    
    const files = fs.readdirSync(this.rootDir);
    const iterationFiles = files.filter(file => 
      file.match(/^test-iteration-\d+-report\.json$/)
    );

    console.log(`  Found ${iterationFiles.length} iteration files`);
    this.stats.iterationFiles = iterationFiles.length;

    // Move iteration files to archive
    const archiveDir = path.join(this.rootDir, 'tests/archived/iterations');
    
    for (const file of iterationFiles) {
      try {
        const sourcePath = path.join(this.rootDir, file);
        const targetPath = path.join(archiveDir, file);
        
        fs.renameSync(sourcePath, targetPath);
        this.stats.moved++;
        
        if (this.stats.moved % 50 === 0) {
          console.log(`  📦 Moved ${this.stats.moved}/${iterationFiles.length} files...`);
        }
      } catch (error) {
        console.error(`  ❌ Failed to move ${file}:`, error.message);
        this.stats.errors++;
      }
    }
    
    console.log(`  ✓ Moved ${this.stats.moved} iteration files to archive`);
  }

  /**
   * Organize remaining test files
   */
  async organizeTestFiles() {
    console.log('\n🔧 Organizing test files...');
    
    const files = fs.readdirSync(this.rootDir);
    const testFiles = files.filter(file => 
      file.startsWith('test-') && 
      file.endsWith('.js') && 
      !file.includes('iteration')
    );

    console.log(`  Found ${testFiles.length} test files to organize`);
    this.stats.testFiles = testFiles.length;

    const testCategories = {
      'integration': [
        'test-bse-digio-integration.js',
        'test-nse-integration.js',
        'test-ollama-integration.js',
        'test-mongodb-connection.js'
      ],
      'unit': [
        'test-services.js',
        'test-nse-service.js',
        'test-nse-cli.js'
      ],
      'e2e': [
        'test-whatsapp-chatbot.js',
        'test-whatsapp-comprehensive-deep.js',
        'test-whatsapp-100-percent-final.js',
        'test-whatsapp-world-class.js'
      ],
      'performance': [
        'test-whatsapp-100-percent-efficiency-deep.js',
        'test-whatsapp-efficiency-demo.js',
        'test-whatsapp-massive.js'
      ]
    };

    // Move categorized files
    for (const [category, categoryFiles] of Object.entries(testCategories)) {
      const categoryDir = path.join(this.rootDir, 'tests', category);
      
      for (const file of categoryFiles) {
        if (testFiles.includes(file)) {
          try {
            const sourcePath = path.join(this.rootDir, file);
            const targetPath = path.join(categoryDir, file);
            
            fs.renameSync(sourcePath, targetPath);
            console.log(`  ✓ Moved ${file} to ${category}/`);
            this.stats.moved++;
          } catch (error) {
            console.error(`  ❌ Failed to move ${file}:`, error.message);
            this.stats.errors++;
          }
        }
      }
    }

    // Move remaining test files to archived
    const remainingTestFiles = files.filter(file => 
      file.startsWith('test-') && 
      file.endsWith('.js') && 
      fs.existsSync(path.join(this.rootDir, file))
    );

    const archivedTestDir = path.join(this.rootDir, 'tests/archived');
    for (const file of remainingTestFiles) {
      try {
        const sourcePath = path.join(this.rootDir, file);
        const targetPath = path.join(archivedTestDir, file);
        
        fs.renameSync(sourcePath, targetPath);
        console.log(`  📦 Archived ${file}`);
        this.stats.moved++;
      } catch (error) {
        console.error(`  ❌ Failed to archive ${file}:`, error.message);
        this.stats.errors++;
      }
    }
  }

  /**
   * Clean up report and output files
   */
  async cleanupReportFiles() {
    console.log('\n📊 Cleaning up report and output files...');
    
    const files = fs.readdirSync(this.rootDir);
    const reportFiles = files.filter(file => 
      (file.includes('report') || 
      file.includes('results') || 
      file.includes('output') || 
      file.includes('failures') ||
      file.endsWith('.txt') ||
      file.endsWith('.json')) &&
      !file.includes('package') && 
      !file.includes('.env') &&
      !file.includes('README') &&
      !file.includes('SECURITY') &&
      !file.includes('cleanup-test-pollution') &&
      !file.includes('jest.config')
    );

    console.log(`  Found ${reportFiles.length} report/output files`);
    this.stats.reportFiles = reportFiles.length;

    const reportsDir = path.join(this.rootDir, 'tests/archived/reports');
    const outputsDir = path.join(this.rootDir, 'tests/archived/outputs');

    for (const file of reportFiles) {
      try {
        const sourcePath = path.join(this.rootDir, file);
        const targetDir = file.includes('output') || file.endsWith('.txt') ? outputsDir : reportsDir;
        const targetPath = path.join(targetDir, file);
        
        fs.renameSync(sourcePath, targetPath);
        this.stats.moved++;
      } catch (error) {
        console.error(`  ❌ Failed to move ${file}:`, error.message);
        this.stats.errors++;
      }
    }
    
    console.log(`  ✓ Organized ${reportFiles.length} report/output files`);
  }

  /**
   * Create proper test structure with examples
   */
  async createProperTestStructure() {
    console.log('\n🏗️  Creating proper test structure...');
    
    // Update existing jest config or create new one
    const jestConfig = {
      testEnvironment: 'node',
      roots: ['<rootDir>/tests'],
      testMatch: [
        '**/tests/**/*.test.js',
        '**/tests/**/*.spec.js'
      ],
      collectCoverageFrom: [
        'src/**/*.js',
        '!src/**/*.test.js',
        '!src/test/**'
      ],
      coverageDirectory: 'coverage',
      coverageReporters: ['text', 'lcov', 'html'],
      setupFilesAfterEnv: ['<rootDir>/tests/setup.js'],
      testTimeout: 30000
    };

    const jestConfigPath = path.join(this.rootDir, 'jest.config.js');
    fs.writeFileSync(
      jestConfigPath,
      `module.exports = ${JSON.stringify(jestConfig, null, 2)};`
    );

    // Create test setup file
    const testSetup = `// Test setup and global configurations
const mongoose = require('mongoose');

// Setup test database
beforeAll(async () => {
  // Connect to test database
  if (process.env.NODE_ENV !== 'test') {
    process.env.NODE_ENV = 'test';
  }
});

afterAll(async () => {
  // Cleanup after tests
  if (mongoose.connection.readyState === 1) {
    await mongoose.connection.close();
  }
});

// Global test utilities
global.testUtils = {
  // Add common test utilities here
};
`;

    fs.writeFileSync(
      path.join(this.rootDir, 'tests/setup.js'),
      testSetup
    );

    // Create example test files
    const exampleUnitTest = `// Example unit test
const { describe, test, expect } = require('@jest/globals');

describe('Example Unit Test', () => {
  test('should pass basic test', () => {
    expect(1 + 1).toBe(2);
  });
});
`;

    fs.writeFileSync(
      path.join(this.rootDir, 'tests/unit/example.test.js'),
      exampleUnitTest
    );

    console.log('  ✓ Created proper test structure');
    console.log('  ✓ Updated jest.config.js');
    console.log('  ✓ Created tests/setup.js');
    console.log('  ✓ Created example test files');
  }

  /**
   * Generate cleanup summary
   */
  generateSummary() {
    console.log('\n📋 Cleanup Summary');
    console.log('==================');
    console.log(`Test iteration files found: ${this.stats.iterationFiles}`);
    console.log(`Test files organized: ${this.stats.testFiles}`);
    console.log(`Report files organized: ${this.stats.reportFiles}`);
    console.log(`Total files moved: ${this.stats.moved}`);
    console.log(`Errors encountered: ${this.stats.errors}`);
    
    if (this.stats.errors === 0) {
      console.log('\n✅ Cleanup completed successfully!');
      console.log('\nNext steps:');
      console.log('1. Review the organized test structure in tests/');
      console.log('2. Update any remaining test imports');
      console.log('3. Run: npm test to verify test setup');
      console.log('4. Consider removing tests/archived/ after verification');
    } else {
      console.log(`\n⚠️  Cleanup completed with ${this.stats.errors} errors`);
      console.log('Please review the errors above and fix manually if needed');
    }

    // Create cleanup report
    const report = {
      timestamp: new Date().toISOString(),
      stats: this.stats,
      directories_created: [
        'tests/archived/iterations',
        'tests/archived/reports', 
        'tests/archived/outputs',
        'tests/unit',
        'tests/integration',
        'tests/e2e',
        'tests/performance'
      ],
      files_created: [
        'jest.config.js',
        'tests/setup.js',
        'tests/unit/example.test.js'
      ]
    };

    fs.writeFileSync(
      path.join(this.rootDir, 'cleanup-report.json'),
      JSON.stringify(report, null, 2)
    );

    console.log('\n📄 Detailed report saved to: cleanup-report.json');
  }
}

// Execute cleanup if run directly
if (require.main === module) {
  const cleanup = new TestCleanupManager();
  cleanup.cleanup().catch(error => {
    console.error('Fatal error:', error);
    process.exit(1);
  });
}

module.exports = TestCleanupManager;
