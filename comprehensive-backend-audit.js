const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class BackendAuditor {
  constructor() {
    this.results = {
      timestamp: new Date().toISOString(),
      totalModules: 0,
      passedModules: 0,
      failedModules: 0,
      passRate: 0,
      coverage: 0,
      moduleDetails: {},
      errors: [],
      fixes: []
    };
    
    this.testCategories = [
      { name: 'Basic Functionality', pattern: '__tests__/basic-functionality.test.js' },
      { name: 'Controllers', pattern: '__tests__/controllers/*.test.js' },
      { name: 'Services', pattern: '__tests__/services/*.test.js' },
      { name: 'Models', pattern: '__tests__/models/*.test.js' },
      { name: 'Middleware', pattern: '__tests__/middleware/*.test.js' },
      { name: 'Comprehensive', pattern: '__tests__/comprehensive.test.js' }
    ];
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const prefix = type === 'error' ? '❌' : type === 'success' ? '✅' : type === 'warning' ? '⚠️' : 'ℹ️';
    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  async runCommand(command, description) {
    try {
      this.log(`Running: ${description}`);
      const output = execSync(command, { 
        encoding: 'utf8', 
        stdio: 'pipe',
        timeout: 60000 // 60 second timeout
      });
      this.log(`✅ ${description} completed successfully`, 'success');
      return { success: true, output };
    } catch (error) {
      this.log(`❌ ${description} failed: ${error.message}`, 'error');
      return { success: false, error: error.message, output: error.stdout || error.stderr || '' };
    }
  }

  async testModule(testFile, category) {
    this.log(`Testing module: ${testFile}`);
    
    const result = await this.runCommand(
      `npx jest "${testFile}" --verbose --detectOpenHandles --forceExit --maxWorkers=1 --testTimeout=30000`,
      `Testing ${path.basename(testFile)}`
    );

    const moduleName = path.basename(testFile);
    this.results.moduleDetails[moduleName] = {
      category,
      success: result.success,
      testCount: 0,
      passedCount: 0,
      failedCount: 0,
      coverage: 0,
      error: result.success ? null : result.error,
      output: result.output
    };

    if (result.success) {
      this.results.passedModules++;
      this.log(`✅ ${moduleName} passed`, 'success');
    } else {
      this.results.failedModules++;
      this.log(`❌ ${moduleName} failed`, 'error');
    }

    this.results.totalModules++;
    return result;
  }

  async runAllTests() {
    this.log('🚀 Starting Comprehensive Backend Audit...');
    this.log('📊 Testing all modules systematically...');

    for (const category of this.testCategories) {
      this.log(`\n📋 Testing Category: ${category.name}`);
      
      if (category.pattern.includes('*')) {
        // Test multiple files in category
        const testFiles = this.getTestFiles(category.pattern);
        for (const testFile of testFiles) {
          await this.testModule(testFile, category.name);
        }
      } else {
        // Test single file
        await this.testModule(category.pattern, category.name);
      }
    }

    this.calculateResults();
    this.generateReport();
  }

  getTestFiles(pattern) {
    const glob = require('glob');
    const files = glob.sync(pattern);
    return files.filter(file => fs.existsSync(file));
  }

  calculateResults() {
    this.results.passRate = this.results.totalModules > 0 
      ? ((this.results.passedModules / this.results.totalModules) * 100).toFixed(2)
      : 0;
    
    this.log(`\n📊 AUDIT RESULTS:`);
    this.log(`Total Modules: ${this.results.totalModules}`);
    this.log(`Passed: ${this.results.passedModules}`);
    this.log(`Failed: ${this.results.failedModules}`);
    this.log(`Pass Rate: ${this.results.passRate}%`);
  }

  generateReport() {
    const reportFile = `backend-audit-report-${Date.now()}.json`;
    fs.writeFileSync(reportFile, JSON.stringify(this.results, null, 2));
    this.log(`📄 Detailed report saved to: ${reportFile}`);
    
    // Generate summary
    const summaryFile = `backend-audit-summary-${Date.now()}.txt`;
    const summary = this.generateSummary();
    fs.writeFileSync(summaryFile, summary);
    this.log(`📋 Summary saved to: ${summaryFile}`);
  }

  generateSummary() {
    let summary = `BACKEND AUDIT SUMMARY\n`;
    summary += `Generated: ${new Date().toISOString()}\n`;
    summary += `=====================================\n\n`;
    summary += `OVERALL RESULTS:\n`;
    summary += `Total Modules Tested: ${this.results.totalModules}\n`;
    summary += `Passed: ${this.results.passedModules}\n`;
    summary += `Failed: ${this.results.failedModules}\n`;
    summary += `Pass Rate: ${this.results.passRate}%\n\n`;

    if (this.results.failedModules > 0) {
      summary += `FAILED MODULES:\n`;
      Object.entries(this.results.moduleDetails)
        .filter(([name, details]) => !details.success)
        .forEach(([name, details]) => {
          summary += `- ${name} (${details.category})\n`;
          if (details.error) {
            summary += `  Error: ${details.error}\n`;
          }
        });
      summary += `\n`;
    }

    summary += `RECOMMENDATIONS:\n`;
    if (this.results.passRate >= 95) {
      summary += `✅ Backend is in excellent condition!\n`;
    } else if (this.results.passRate >= 80) {
      summary += `⚠️ Backend needs minor improvements\n`;
    } else {
      summary += `❌ Backend requires significant fixes\n`;
    }

    return summary;
  }

  async fixIssues() {
    this.log('\n🔧 Starting automatic issue fixes...');
    
    const failedModules = Object.entries(this.results.moduleDetails)
      .filter(([name, details]) => !details.success);

    for (const [moduleName, details] of failedModules) {
      this.log(`🔧 Attempting to fix: ${moduleName}`);
      
      // Try to fix common issues
      const fixResult = await this.attemptFix(moduleName, details);
      
      if (fixResult.success) {
        this.log(`✅ Fixed: ${moduleName}`, 'success');
        this.results.fixes.push({
          module: moduleName,
          fix: fixResult.fix,
          success: true
        });
      } else {
        this.log(`❌ Could not fix: ${moduleName}`, 'error');
        this.results.fixes.push({
          module: moduleName,
          fix: fixResult.fix,
          success: false,
          error: fixResult.error
        });
      }
    }
  }

  async attemptFix(moduleName, details) {
    try {
      // Check if it's a syntax error
      if (details.error && details.error.includes('SyntaxError')) {
        return await this.fixSyntaxError(moduleName, details);
      }
      
      // Check if it's a missing dependency
      if (details.error && details.error.includes('Cannot find module')) {
        return await this.fixMissingDependency(moduleName, details);
      }
      
      // Check if it's a test setup issue
      if (details.error && details.error.includes('setup')) {
        return await this.fixTestSetup(moduleName, details);
      }
      
      return { success: false, fix: 'Unknown issue type', error: details.error };
    } catch (error) {
      return { success: false, fix: 'Fix attempt failed', error: error.message };
    }
  }

  async fixSyntaxError(moduleName, details) {
    // This would require parsing the actual file and fixing syntax
    // For now, just report the issue
    return {
      success: false,
      fix: 'Syntax error detected - manual review required',
      error: details.error
    };
  }

  async fixMissingDependency(moduleName, details) {
    // Try to install missing dependencies
    const missingModule = this.extractMissingModule(details.error);
    if (missingModule) {
      try {
        await this.runCommand(
          `npm install ${missingModule}`,
          `Installing missing dependency: ${missingModule}`
        );
        return { success: true, fix: `Installed missing dependency: ${missingModule}` };
      } catch (error) {
        return { success: false, fix: `Failed to install ${missingModule}`, error: error.message };
      }
    }
    return { success: false, fix: 'Could not identify missing dependency', error: details.error };
  }

  extractMissingModule(error) {
    const match = error.match(/Cannot find module '([^']+)'/);
    return match ? match[1] : null;
  }

  async fixTestSetup(moduleName, details) {
    // Try to fix test setup issues
    return {
      success: false,
      fix: 'Test setup issue - manual review required',
      error: details.error
    };
  }
}

// Main execution
async function main() {
  const auditor = new BackendAuditor();
  
  try {
    // Run comprehensive tests
    await auditor.runAllTests();
    
    // If there are failures, attempt to fix them
    if (auditor.results.failedModules > 0) {
      await auditor.fixIssues();
      
      // Run tests again after fixes
      auditor.log('\n🔄 Running tests again after fixes...');
      await auditor.runAllTests();
    }
    
    // Final assessment
    if (auditor.results.passRate >= 95) {
      auditor.log('🎉 Backend audit completed successfully! System is in excellent condition.', 'success');
    } else {
      auditor.log('⚠️ Backend audit completed with issues. Manual review recommended.', 'warning');
    }
    
  } catch (error) {
    auditor.log(`❌ Audit failed: ${error.message}`, 'error');
    process.exit(1);
  }
}

// Run the audit
main().catch(console.error); 