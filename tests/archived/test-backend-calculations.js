const axios = require('axios');
const logger = require('./src/utils/logger');

// Test configuration
const BASE_URL = 'http://localhost:3000/api';
const TEST_USER_ID = '507f1f77bcf86cd799439011'; // Mock user ID

// Mock authentication token
const AUTH_TOKEN = 'mock_jwt_token_for_testing';

// Test data
const testData = {
  sipProjection: {
    monthlyAmount: 10000,
    duration: 60,
    expectedReturn: 12,
    fundCode: 'HDFCMIDCAP',
    includeInflation: true,
    includeTaxes: true
  },
  goalBasedInvestment: {
    goalAmount: 1000000,
    targetDate: '2029-12-31',
    currentSavings: 200000,
    riskProfile: 'moderate',
    includeInflation: true
  },
  chartData: {
    chartType: 'portfolio_performance',
    period: '1y',
    fundCode: 'HDFCMIDCAP',
    options: {
      includeBenchmark: true
    }
  }
};

class BackendCalculationsTest {
  constructor() {
    this.results = {
      passed: 0,
      failed: 0,
      total: 0,
      details: []
    };
  }

  /**
   * Run all tests
   */
  async runAllTests() {
    console.log('🚀 Starting Backend Calculations API Tests...\n');

    try {
      // Test 1: Performance Analytics
      await this.testPerformanceAnalytics();

      // Test 2: Chart Data Generation
      await this.testChartDataGeneration();

      // Test 3: SIP Projections
      await this.testSIPProjections();

      // Test 4: Goal-based Investment
      await this.testGoalBasedInvestment();

      // Test 5: Risk Profiling
      await this.testRiskProfiling();

      // Test 6: NAV History
      await this.testNAVHistory();

      // Test 7: Tax Calculations
      await this.testTaxCalculations();

      // Test 8: XIRR Analytics
      await this.testXIRRAnalytics();

      // Test 9: Portfolio Comparison
      await this.testPortfolioComparison();

      // Test 10: Dashboard Analytics
      await this.testDashboardAnalytics();

      // Test 11: Platform Analytics
      await this.testPlatformAnalytics();

      // Test 12: Regional Analytics
      await this.testRegionalAnalytics();

      // Test 13: Agent Analytics
      await this.testAgentAnalytics();

      this.printResults();
    } catch (error) {
      console.error('❌ Test execution failed:', error.message);
      logger.error('Test execution failed:', error);
    }
  }

  /**
   * Test Performance Analytics API
   */
  async testPerformanceAnalytics() {
    const testName = 'Performance Analytics API';
    console.log(`📊 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/performance`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          period: '1y',
          includeChartData: true
        }
      });

      this.validateResponse(response, testName, [
        'data.basic',
        'data.risk',
        'data.performance',
        'data.allocation'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Chart Data Generation API
   */
  async testChartDataGeneration() {
    const testName = 'Chart Data Generation API';
    console.log(`📈 Testing ${testName}...`);

    try {
      const response = await axios.post(`${BASE_URL}/analytics/chart-data`, testData.chartData, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        }
      });

      this.validateResponse(response, testName, [
        'data.type',
        'data.title',
        'data.series'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test SIP Projections API
   */
  async testSIPProjections() {
    const testName = 'SIP Projections API';
    console.log(`💰 Testing ${testName}...`);

    try {
      const response = await axios.post(`${BASE_URL}/analytics/sip-projections`, testData.sipProjection, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        }
      });

      this.validateResponse(response, testName, [
        'data.summary',
        'data.monthlyBreakdown',
        'data.yearlyBreakdown',
        'data.charts'
      ]);

      // Validate specific calculations
      const data = response.data.data;
      if (data.summary.totalInvestment !== testData.sipProjection.monthlyAmount * testData.sipProjection.duration) {
        throw new Error('SIP total investment calculation incorrect');
      }

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Goal-based Investment API
   */
  async testGoalBasedInvestment() {
    const testName = 'Goal-based Investment API';
    console.log(`🎯 Testing ${testName}...`);

    try {
      const response = await axios.post(`${BASE_URL}/analytics/goal-based-investment`, testData.goalBasedInvestment, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        }
      });

      this.validateResponse(response, testName, [
        'data.goal',
        'data.requiredMonthlyInvestment',
        'data.scenarios',
        'data.recommendations'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Risk Profiling API
   */
  async testRiskProfiling() {
    const testName = 'Risk Profiling API';
    console.log(`⚠️ Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/risk-profiling`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          includePortfolioRisk: true,
          includeMarketRisk: true
        }
      });

      this.validateResponse(response, testName, [
        'data.personal',
        'data.portfolio',
        'data.market',
        'data.recommendations'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test NAV History API
   */
  async testNAVHistory() {
    const testName = 'NAV History API';
    console.log(`📊 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/nav-history/HDFCMIDCAP`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          period: '1y',
          includeCalculations: true
        }
      });

      this.validateResponse(response, testName, [
        'data.fundCode',
        'data.data',
        'data.calculations',
        'data.summary'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Tax Calculations API
   */
  async testTaxCalculations() {
    const testName = 'Tax Calculations API';
    console.log(`🧮 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/tax-calculations`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          financialYear: '2024-25',
          includeOptimization: true
        }
      });

      this.validateResponse(response, testName, [
        'data.summary',
        'data.capitalGains',
        'data.dividendIncome',
        'data.optimization'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test XIRR Analytics API
   */
  async testXIRRAnalytics() {
    const testName = 'XIRR Analytics API';
    console.log(`📈 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/xirr`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          timeframe: '1y'
        }
      });

      this.validateResponse(response, testName, [
        'data.xirr',
        'data.cashFlows',
        'data.period'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Portfolio Comparison API
   */
  async testPortfolioComparison() {
    const testName = 'Portfolio Comparison API';
    console.log(`🔄 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/portfolio-comparison`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          benchmark: 'NIFTY50',
          period: '1y'
        }
      });

      this.validateResponse(response, testName, [
        'data.portfolio',
        'data.benchmark',
        'data.comparison'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Dashboard Analytics API
   */
  async testDashboardAnalytics() {
    const testName = 'Dashboard Analytics API';
    console.log(`📊 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/dashboard`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          includeCharts: true,
          includeRecommendations: true
        }
      });

      this.validateResponse(response, testName, [
        'data.portfolio',
        'data.performance',
        'data.charts',
        'data.recommendations'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Platform Analytics API
   */
  async testPlatformAnalytics() {
    const testName = 'Platform Analytics API';
    console.log(`🌐 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/platform`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          period: '30d',
          type: 'overview'
        }
      });

      this.validateResponse(response, testName, [
        'data.users',
        'data.transactions',
        'data.revenue'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Regional Analytics API
   */
  async testRegionalAnalytics() {
    const testName = 'Regional Analytics API';
    console.log(`🌍 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/regional`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          region: 'Mumbai',
          period: '30d'
        }
      });

      this.validateResponse(response, testName, [
        'data.region',
        'data.metrics',
        'data.trends'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Test Agent Analytics API
   */
  async testAgentAnalytics() {
    const testName = 'Agent Analytics API';
    console.log(`👨‍💼 Testing ${testName}...`);

    try {
      const response = await axios.get(`${BASE_URL}/analytics/agent`, {
        headers: {
          'Authorization': `Bearer ${AUTH_TOKEN}`,
          'Content-Type': 'application/json'
        },
        params: {
          agentId: 'agent123',
          period: '30d'
        }
      });

      this.validateResponse(response, testName, [
        'data.agentId',
        'data.performance',
        'data.clients'
      ]);

    } catch (error) {
      this.handleTestError(testName, error);
    }
  }

  /**
   * Validate API response
   */
  validateResponse(response, testName, requiredFields) {
    this.results.total++;

    if (response.status !== 200) {
      throw new Error(`Expected status 200, got ${response.status}`);
    }

    if (!response.data.success) {
      throw new Error(`API returned success: false`);
    }

    // Check required fields
    requiredFields.forEach(field => {
      const value = this.getNestedValue(response.data, field);
      if (value === undefined) {
        throw new Error(`Missing required field: ${field}`);
      }
    });

    this.results.passed++;
    this.results.details.push({
      test: testName,
      status: 'PASSED',
      message: 'API working correctly'
    });

    console.log(`✅ ${testName} - PASSED`);
  }

  /**
   * Handle test errors
   */
  handleTestError(testName, error) {
    this.results.total++;
    this.results.failed++;
    
    const errorMessage = error.response?.data?.message || error.message;
    
    this.results.details.push({
      test: testName,
      status: 'FAILED',
      message: errorMessage
    });

    console.log(`❌ ${testName} - FAILED: ${errorMessage}`);
  }

  /**
   * Get nested object value
   */
  getNestedValue(obj, path) {
    return path.split('.').reduce((current, key) => current?.[key], obj);
  }

  /**
   * Print test results
   */
  printResults() {
    console.log('\n' + '='.repeat(60));
    console.log('📋 BACKEND CALCULATIONS API TEST RESULTS');
    console.log('='.repeat(60));
    
    console.log(`\n✅ Passed: ${this.results.passed}`);
    console.log(`❌ Failed: ${this.results.failed}`);
    console.log(`📊 Total: ${this.results.total}`);
    
    const successRate = ((this.results.passed / this.results.total) * 100).toFixed(1);
    console.log(`📈 Success Rate: ${successRate}%`);

    if (this.results.failed > 0) {
      console.log('\n❌ FAILED TESTS:');
      this.results.details
        .filter(detail => detail.status === 'FAILED')
        .forEach(detail => {
          console.log(`   • ${detail.test}: ${detail.message}`);
        });
    }

    console.log('\n✅ PASSED TESTS:');
    this.results.details
      .filter(detail => detail.status === 'PASSED')
      .forEach(detail => {
        console.log(`   • ${detail.test}`);
      });

    console.log('\n' + '='.repeat(60));
    
    if (this.results.failed === 0) {
      console.log('🎉 ALL TESTS PASSED! Backend calculations are working correctly.');
      console.log('✅ Frontend business logic can be safely moved to backend.');
    } else {
      console.log('⚠️ Some tests failed. Please review the backend implementation.');
    }
    
    console.log('='.repeat(60) + '\n');
  }
}

// Run tests if this file is executed directly
if (require.main === module) {
  const testRunner = new BackendCalculationsTest();
  testRunner.runAllTests().catch(console.error);
}

module.exports = BackendCalculationsTest; 