const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

// Test configuration
const BASE_URL = 'http://localhost:3000/api';
const TEST_USER_ID = 'test-user-123';

// Test data
const testPortfolioData = {
  holdings: [
    {
      schemeCode: 'HDFC001',
      schemeName: 'HDFC Mid-Cap Opportunities Fund',
      category: 'Mid Cap',
      fundHouse: 'HDFC',
      currentValue: 50000,
      totalInvestment: 45000,
      return: 0.15,
      xirr: 12.5,
      beta: 1.2,
      marketCap: 'Mid Cap',
      sectorAllocation: {
        'Technology': 25,
        'Financial Services': 20,
        'Consumer Goods': 15
      }
    },
    {
      schemeCode: 'ICICI002',
      schemeName: 'ICICI Prudential Bluechip Fund',
      category: 'Large Cap',
      fundHouse: 'ICICI',
      currentValue: 75000,
      totalInvestment: 70000,
      return: 0.12,
      xirr: 10.8,
      beta: 0.9,
      marketCap: 'Large Cap',
      sectorAllocation: {
        'Financial Services': 30,
        'Technology': 20,
        'Energy': 15
      }
    }
  ],
  transactions: [
    {
      type: 'PURCHASE',
      amount: 45000,
      date: '2023-01-15',
      schemeCode: 'HDFC001'
    },
    {
      type: 'PURCHASE',
      amount: 70000,
      date: '2023-02-01',
      schemeCode: 'ICICI002'
    }
  ]
};

const testDocument = {
  id: 'test-document-001',
  content: 'This is a test document about Indian mutual funds. It contains information about SEBI regulations, expense ratios, and investment strategies suitable for Indian investors.',
  metadata: {
    title: 'Test Mutual Fund Document',
    category: 'education',
    source: 'Test Source',
    date: '2024'
  }
};

class OllamaIntegrationTester {
  constructor() {
    this.results = [];
    this.startTime = new Date();
  }

  /**
   * Log test result
   */
  logResult(testName, success, details = null) {
    const result = {
      test: testName,
      success,
      timestamp: new Date().toISOString(),
      details
    };
    
    this.results.push(result);
    console.log(`[${success ? '✅' : '❌'}] ${testName}: ${success ? 'PASSED' : 'FAILED'}`);
    
    if (details) {
      console.log(`   Details: ${JSON.stringify(details, null, 2)}`);
    }
  }

  /**
   * Test 1: System Health Check
   */
  async testSystemHealth() {
    try {
      const response = await axios.get(`${BASE_URL}/ollama/health`);
      
      if (response.data.success) {
        this.logResult('System Health Check', true, {
          status: response.data.status,
          services: response.data.services
        });
      } else {
        this.logResult('System Health Check', false, response.data);
      }
    } catch (error) {
      this.logResult('System Health Check', false, error.message);
    }
  }

  /**
   * Test 2: Ollama Connection Test
   */
  async testOllamaConnection() {
    try {
      const response = await axios.get(`${BASE_URL}/ollama/test`);
      
      if (response.data.success && response.data.connected) {
        this.logResult('Ollama Connection Test', true, {
          models: response.data.models.length
        });
      } else {
        this.logResult('Ollama Connection Test', false, response.data);
      }
    } catch (error) {
      this.logResult('Ollama Connection Test', false, error.message);
    }
  }

  /**
   * Test 3: Training Data Generation
   */
  async testTrainingDataGeneration() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/training/generate`);
      
      if (response.data.success) {
        this.logResult('Training Data Generation', true, {
          totalPairs: response.data.data.totalPairs,
          files: response.data.data.files
        });
      } else {
        this.logResult('Training Data Generation', false, response.data);
      }
    } catch (error) {
      this.logResult('Training Data Generation', false, error.message);
    }
  }

  /**
   * Test 4: RAG Document Addition
   */
  async testRAGDocumentAddition() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/rag/document`, {
        document: testDocument
      });
      
      if (response.data.success) {
        this.logResult('RAG Document Addition', true, {
          documentId: response.data.documentId,
          stats: response.data.ragStats
        });
      } else {
        this.logResult('RAG Document Addition', false, response.data);
      }
    } catch (error) {
      this.logResult('RAG Document Addition', false, error.message);
    }
  }

  /**
   * Test 5: RAG Document Search
   */
  async testRAGDocumentSearch() {
    try {
      const response = await axios.get(`${BASE_URL}/ollama/rag/search?query=mutual funds&limit=3`);
      
      if (response.data.success) {
        this.logResult('RAG Document Search', true, {
          resultsCount: response.data.results.length,
          query: response.data.query
        });
      } else {
        this.logResult('RAG Document Search', false, response.data);
      }
    } catch (error) {
      this.logResult('RAG Document Search', false, error.message);
    }
  }

  /**
   * Test 6: Basic Question Answering
   */
  async testBasicQuestionAnswering() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/ask`, {
        question: 'What is XIRR in mutual funds?',
        userId: TEST_USER_ID
      });
      
      if (response.data.success) {
        this.logResult('Basic Question Answering', true, {
          answerLength: response.data.answer.length,
          sources: response.data.sources.length,
          compliant: response.data.compliance.compliant
        });
      } else {
        this.logResult('Basic Question Answering', false, response.data);
      }
    } catch (error) {
      this.logResult('Basic Question Answering', false, error.message);
    }
  }

  /**
   * Test 7: Portfolio Analytics
   */
  async testPortfolioAnalytics() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/portfolio/analytics`, {
        portfolioData: testPortfolioData
      });
      
      if (response.data.success) {
        this.logResult('Portfolio Analytics', true, {
          totalFunds: response.data.analytics.basic.numberOfFunds,
          totalValue: response.data.analytics.basic.totalCurrentValue,
          xirr: response.data.analytics.basic.xirr
        });
      } else {
        this.logResult('Portfolio Analytics', false, response.data);
      }
    } catch (error) {
      this.logResult('Portfolio Analytics', false, error.message);
    }
  }

  /**
   * Test 8: Compliance Audit
   */
  async testComplianceAudit() {
    try {
      const testData = {
        aiResponse: 'This fund has guaranteed returns of 15% per year. You should definitely invest in it.',
        userQuery: 'Should I invest in this fund?',
        userId: TEST_USER_ID
      };

      const response = await axios.post(`${BASE_URL}/ollama/compliance/audit`, {
        data: testData
      });
      
      if (response.data.success) {
        this.logResult('Compliance Audit', true, {
          totalChecks: response.data.audit.summary.total,
          compliantChecks: response.data.audit.summary.compliant,
          criticalIssues: response.data.audit.summary.criticalIssues
        });
      } else {
        this.logResult('Compliance Audit', false, response.data);
      }
    } catch (error) {
      this.logResult('Compliance Audit', false, error.message);
    }
  }

  /**
   * Test 9: Financial Advice
   */
  async testFinancialAdvice() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/financial-advice`, {
        query: 'How should I allocate my portfolio for retirement?',
        portfolioData: testPortfolioData,
        userId: TEST_USER_ID
      });
      
      if (response.data.success) {
        this.logResult('Financial Advice', true, {
          answerLength: response.data.answer.length,
          compliant: response.data.compliance.compliant
        });
      } else {
        this.logResult('Financial Advice', false, response.data);
      }
    } catch (error) {
      this.logResult('Financial Advice', false, error.message);
    }
  }

  /**
   * Test 10: SIP Recommendation
   */
  async testSIPRecommendation() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/sip-recommendation`, {
        userId: TEST_USER_ID,
        riskProfile: 'moderate',
        investmentGoal: 'retirement',
        currentSIP: 10000
      });
      
      if (response.data.success) {
        this.logResult('SIP Recommendation', true, {
          answerLength: response.data.answer.length,
          compliant: response.data.compliance.compliant
        });
      } else {
        this.logResult('SIP Recommendation', false, response.data);
      }
    } catch (error) {
      this.logResult('SIP Recommendation', false, error.message);
    }
  }

  /**
   * Test 11: Fund Comparison
   */
  async testFundComparison() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/fund-comparison`, {
        fund1: 'HDFC Mid-Cap Opportunities Fund',
        fund2: 'ICICI Prudential Bluechip Fund',
        comparisonCriteria: 'performance and risk'
      });
      
      if (response.data.success) {
        this.logResult('Fund Comparison', true, {
          answerLength: response.data.answer.length,
          compliant: response.data.compliance.compliant
        });
      } else {
        this.logResult('Fund Comparison', false, response.data);
      }
    } catch (error) {
      this.logResult('Fund Comparison', false, error.message);
    }
  }

  /**
   * Test 12: Tax Optimization
   */
  async testTaxOptimization() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/tax-optimization`, {
        portfolioData: testPortfolioData,
        taxSlab: '30%',
        userId: TEST_USER_ID
      });
      
      if (response.data.success) {
        this.logResult('Tax Optimization', true, {
          answerLength: response.data.answer.length,
          compliant: response.data.compliance.compliant
        });
      } else {
        this.logResult('Tax Optimization', false, response.data);
      }
    } catch (error) {
      this.logResult('Tax Optimization', false, error.message);
    }
  }

  /**
   * Test 13: Chat Functionality
   */
  async testChatFunctionality() {
    try {
      const response = await axios.post(`${BASE_URL}/ollama/chat`, {
        message: 'What are the benefits of SIP?',
        userId: TEST_USER_ID,
        conversationId: 'conv-123'
      });
      
      if (response.data.success) {
        this.logResult('Chat Functionality', true, {
          answerLength: response.data.answer.length,
          compliant: response.data.compliance.compliant
        });
      } else {
        this.logResult('Chat Functionality', false, response.data);
      }
    } catch (error) {
      this.logResult('Chat Functionality', false, error.message);
    }
  }

  /**
   * Test 14: RAG Statistics
   */
  async testRAGStatistics() {
    try {
      const response = await axios.get(`${BASE_URL}/ollama/rag/stats`);
      
      if (response.data.success) {
        this.logResult('RAG Statistics', true, {
          totalDocuments: response.data.stats.totalDocuments,
          categories: Object.keys(response.data.stats.categories).length
        });
      } else {
        this.logResult('RAG Statistics', false, response.data);
      }
    } catch (error) {
      this.logResult('RAG Statistics', false, error.message);
    }
  }

  /**
   * Test 15: Model Status
   */
  async testModelStatus() {
    try {
      const response = await axios.get(`${BASE_URL}/ollama/status`);
      
      if (response.data.success) {
        this.logResult('Model Status', true, {
          ollamaRunning: response.data.ollamaRunning,
          totalModels: response.data.totalModels
        });
      } else {
        this.logResult('Model Status', false, response.data);
      }
    } catch (error) {
      this.logResult('Model Status', false, error.message);
    }
  }

  /**
   * Run all tests
   */
  async runAllTests() {
    console.log('🚀 Starting Ollama Integration Tests...\n');
    
    const tests = [
      this.testSystemHealth.bind(this),
      this.testOllamaConnection.bind(this),
      this.testTrainingDataGeneration.bind(this),
      this.testRAGDocumentAddition.bind(this),
      this.testRAGDocumentSearch.bind(this),
      this.testBasicQuestionAnswering.bind(this),
      this.testPortfolioAnalytics.bind(this),
      this.testComplianceAudit.bind(this),
      this.testFinancialAdvice.bind(this),
      this.testSIPRecommendation.bind(this),
      this.testFundComparison.bind(this),
      this.testTaxOptimization.bind(this),
      this.testChatFunctionality.bind(this),
      this.testRAGStatistics.bind(this),
      this.testModelStatus.bind(this)
    ];

    for (const test of tests) {
      try {
        await test();
        // Add small delay between tests
        await new Promise(resolve => setTimeout(resolve, 1000));
      } catch (error) {
        console.error(`Error running test: ${error.message}`);
      }
    }

    this.generateReport();
  }

  /**
   * Generate test report
   */
  generateReport() {
    const endTime = new Date();
    const duration = (endTime - this.startTime) / 1000;
    
    const passed = this.results.filter(r => r.success).length;
    const failed = this.results.filter(r => !r.success).length;
    const total = this.results.length;
    
    console.log('\n📊 Test Report');
    console.log('='.repeat(50));
    console.log(`Total Tests: ${total}`);
    console.log(`Passed: ${passed} ✅`);
    console.log(`Failed: ${failed} ❌`);
    console.log(`Success Rate: ${((passed / total) * 100).toFixed(2)}%`);
    console.log(`Duration: ${duration.toFixed(2)} seconds`);
    
    if (failed > 0) {
      console.log('\n❌ Failed Tests:');
      this.results.filter(r => !r.success).forEach(result => {
        console.log(`  - ${result.test}: ${result.details}`);
      });
    }
    
    console.log('\n✅ All Tests Completed!');
    
    // Save report to file
    const report = {
      summary: {
        total: total,
        passed: passed,
        failed: failed,
        successRate: ((passed / total) * 100).toFixed(2),
        duration: duration.toFixed(2)
      },
      results: this.results,
      timestamp: new Date().toISOString()
    };
    
    fs.writeFile('ollama-integration-test-report.json', JSON.stringify(report, null, 2))
      .then(() => console.log('📄 Test report saved to: ollama-integration-test-report.json'))
      .catch(err => console.error('Error saving report:', err));
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  const tester = new OllamaIntegrationTester();
  tester.runAllTests().catch(console.error);
}

module.exports = OllamaIntegrationTester; 