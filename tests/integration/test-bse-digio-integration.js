const axios = require('axios');
const logger = require('./src/utils/logger');

// Test configuration
const BASE_URL = 'http://localhost:3000/api';
const TEST_USER = {
  id: 'test-user-123',
  name: 'Test User',
  email: 'test@example.com'
};

// Mock JWT token for testing
const MOCK_TOKEN = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJ0ZXN0LXVzZXItMTIzIiwiaWF0IjoxNjM1NzI5NjAwfQ.test-signature';

class BSEStarMFDigioIntegrationTest {
  constructor() {
    this.testResults = {
      bseStarMF: {},
      digio: {},
      summary: {
        total: 0,
        passed: 0,
        failed: 0,
        errors: []
      }
    };
  }

  async makeRequest(endpoint, method = 'GET', data = null) {
    try {
      const config = {
        method,
        url: `${BASE_URL}${endpoint}`,
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${MOCK_TOKEN}`
        },
        timeout: 10000
      };

      if (data) {
        config.data = data;
      }

      const response = await axios(config);
      return {
        success: true,
        data: response.data,
        status: response.status
      };
    } catch (error) {
      return {
        success: false,
        error: error.response?.data || error.message,
        status: error.response?.status || 500
      };
    }
  }

  logTestResult(testName, success, details = {}) {
    this.testResults.summary.total++;
    
    if (success) {
      this.testResults.summary.passed++;
      logger.info(`✅ ${testName} - PASSED`, details);
    } else {
      this.testResults.summary.failed++;
      this.testResults.summary.errors.push({
        test: testName,
        error: details.error || 'Unknown error'
      });
      logger.error(`❌ ${testName} - FAILED`, details);
    }
  }

  // BSE Star MF Tests
  async testBSEStarMFHealthCheck() {
    const testName = 'BSE Star MF Health Check';
    const result = await this.makeRequest('/bse-star-mf/health');
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      data: result.data
    });
    
    this.testResults.bseStarMF.healthCheck = result;
  }

  async testBSEStarMFClientCreation() {
    const testName = 'BSE Star MF Client Creation';
    const clientData = {
      clientData: {
        firstName: 'John',
        lastName: 'Doe',
        dateOfBirth: '1990-01-01',
        gender: 'MALE',
        panNumber: 'ABCDE1234F',
        aadhaarNumber: '123456789012',
        email: 'john.doe@example.com',
        mobile: '9876543210',
        address: {
          line1: '123 Main Street',
          city: 'Mumbai',
          state: 'Maharashtra',
          pincode: '400001'
        },
        bankDetails: {
          accountNumber: '1234567890',
          ifscCode: 'SBIN0001234',
          accountHolderName: 'John Doe'
        }
      }
    };

    const result = await this.makeRequest('/bse-star-mf/client', 'POST', clientData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      data: result.data
    });
    
    this.testResults.bseStarMF.clientCreation = result;
    return result.data?.data?.clientId;
  }

  async testBSEStarMFSchemeMasterData() {
    const testName = 'BSE Star MF Scheme Master Data';
    const result = await this.makeRequest('/bse-star-mf/schemes?limit=5');
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      schemesCount: result.data?.data?.schemes?.length || 0
    });
    
    this.testResults.bseStarMF.schemeMasterData = result;
    return result.data?.data?.schemes?.[0]?.schemeCode;
  }

  async testBSEStarMFSchemeDetails(schemeCode) {
    const testName = 'BSE Star MF Scheme Details';
    const result = await this.makeRequest(`/bse-star-mf/schemes/${schemeCode}`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      schemeCode,
      schemeName: result.data?.data?.schemeName
    });
    
    this.testResults.bseStarMF.schemeDetails = result;
  }

  async testBSEStarMFLumpsumOrder(clientId, schemeCode) {
    const testName = 'BSE Star MF Lumpsum Order';
    const orderData = {
      orderData: {
        clientId,
        schemeCode,
        amount: 10000,
        paymentMode: 'ONLINE'
      }
    };

    const result = await this.makeRequest('/bse-star-mf/order/lumpsum', 'POST', orderData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      orderId: result.data?.data?.orderId
    });
    
    this.testResults.bseStarMF.lumpsumOrder = result;
    return result.data?.data?.orderId;
  }

  async testBSEStarMFOrderStatus(orderId) {
    const testName = 'BSE Star MF Order Status';
    const result = await this.makeRequest(`/bse-star-mf/order/status/${orderId}`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      orderStatus: result.data?.data?.status
    });
    
    this.testResults.bseStarMF.orderStatus = result;
  }

  async testBSEStarMFRedemptionOrder(clientId, schemeCode) {
    const testName = 'BSE Star MF Redemption Order';
    const redemptionData = {
      redemptionData: {
        clientId,
        schemeCode,
        redemptionType: 'UNITS',
        units: 50,
        bankAccount: '1234567890'
      }
    };

    const result = await this.makeRequest('/bse-star-mf/order/redemption', 'POST', redemptionData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      redemptionId: result.data?.data?.redemptionId
    });
    
    this.testResults.bseStarMF.redemptionOrder = result;
  }

  async testBSEStarMFTransactionReport() {
    const testName = 'BSE Star MF Transaction Report';
    const result = await this.makeRequest('/bse-star-mf/report/transactions?limit=5');
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      transactionsCount: result.data?.data?.transactions?.length || 0
    });
    
    this.testResults.bseStarMF.transactionReport = result;
  }

  async testBSEStarMFHoldingsReport() {
    const testName = 'BSE Star MF Holdings Report';
    const result = await this.makeRequest('/bse-star-mf/report/holdings');
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      holdingsCount: result.data?.data?.holdings?.length || 0
    });
    
    this.testResults.bseStarMF.holdingsReport = result;
  }

  async testBSEStarMFCurrentNAV() {
    const testName = 'BSE Star MF Current NAV';
    const navData = {
      schemeCodes: ['HDFCMIDCAP', 'ICICIBLUECHIP']
    };

    const result = await this.makeRequest('/bse-star-mf/nav/current', 'POST', navData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      navDataCount: Object.keys(result.data?.data?.navData || {}).length
    });
    
    this.testResults.bseStarMF.currentNAV = result;
  }

  async testBSEStarMFEMandateSetup(clientId) {
    const testName = 'BSE Star MF eMandate Setup';
    const mandateData = {
      mandateData: {
        clientId,
        bankAccount: {
          accountNumber: '1234567890',
          ifscCode: 'SBIN0001234',
          accountHolderName: 'John Doe'
        },
        amount: 10000,
        frequency: 'MONTHLY',
        startDate: '2024-02-01',
        endDate: '2025-02-01'
      }
    };

    const result = await this.makeRequest('/bse-star-mf/emandate/setup', 'POST', mandateData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      mandateId: result.data?.data?.mandateId
    });
    
    this.testResults.bseStarMF.emandateSetup = result;
    return result.data?.data?.mandateId;
  }

  async testBSEStarMFEMandateStatus(mandateId) {
    const testName = 'BSE Star MF eMandate Status';
    const result = await this.makeRequest(`/bse-star-mf/emandate/status/${mandateId}`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      mandateStatus: result.data?.data?.status
    });
    
    this.testResults.bseStarMF.emandateStatus = result;
  }

  async testBSEStarMFClientFolios(clientId) {
    const testName = 'BSE Star MF Client Folios';
    const result = await this.makeRequest(`/bse-star-mf/client/${clientId}/folios`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      foliosCount: result.data?.data?.folios?.length || 0
    });
    
    this.testResults.bseStarMF.clientFolios = result;
  }

  async testBSEStarMFSchemePerformance() {
    const testName = 'BSE Star MF Scheme Performance';
    const result = await this.makeRequest('/bse-star-mf/schemes/HDFCMIDCAP/performance?period=1Y');
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      performance: result.data?.data?.performance
    });
    
    this.testResults.bseStarMF.schemePerformance = result;
  }

  // Digio Tests
  async testDigioHealthCheck() {
    const testName = 'Digio Health Check';
    const result = await this.makeRequest('/digio/health');
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      data: result.data
    });
    
    this.testResults.digio.healthCheck = result;
  }

  async testDigioKYCInitiation() {
    const testName = 'Digio KYC Initiation';
    const kycData = {
      kycData: {
        customerDetails: {
          name: 'John Doe',
          dateOfBirth: '1990-01-01',
          gender: 'MALE',
          panNumber: 'ABCDE1234F',
          aadhaarNumber: '123456789012',
          mobile: '9876543210',
          email: 'john.doe@example.com',
          address: {
            line1: '123 Main Street',
            city: 'Mumbai',
            state: 'Maharashtra',
            pincode: '400001'
          }
        },
        kycType: 'AADHAAR_BASED'
      }
    };

    const result = await this.makeRequest('/digio/kyc/initiate', 'POST', kycData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      kycId: result.data?.data?.kycId
    });
    
    this.testResults.digio.kycInitiation = result;
    return result.data?.data?.kycId;
  }

  async testDigioKYCStatus(kycId) {
    const testName = 'Digio KYC Status';
    const result = await this.makeRequest(`/digio/kyc/status/${kycId}`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      kycStatus: result.data?.data?.status
    });
    
    this.testResults.digio.kycStatus = result;
  }

  async testDigioKYCDocuments(kycId) {
    const testName = 'Digio KYC Documents';
    const result = await this.makeRequest(`/digio/kyc/${kycId}/documents?type=ALL`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      documentsCount: result.data?.data?.documents?.length || 0
    });
    
    this.testResults.digio.kycDocuments = result;
  }

  async testDigioEMandateSetup() {
    const testName = 'Digio eMandate Setup';
    const mandateData = {
      mandateData: {
        customerDetails: {
          name: 'John Doe',
          mobile: '9876543210',
          email: 'john.doe@example.com',
          panNumber: 'ABCDE1234F'
        },
        bankDetails: {
          accountNumber: '1234567890',
          ifscCode: 'SBIN0001234',
          accountHolderName: 'John Doe'
        },
        mandateDetails: {
          amount: 10000,
          frequency: 'MONTHLY',
          startDate: '2024-02-01',
          endDate: '2025-02-01',
          purpose: 'MUTUAL_FUND_INVESTMENT'
        }
      }
    };

    const result = await this.makeRequest('/digio/emandate/setup', 'POST', mandateData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      mandateId: result.data?.data?.mandateId
    });
    
    this.testResults.digio.emandateSetup = result;
    return result.data?.data?.mandateId;
  }

  async testDigioEMandateStatus(mandateId) {
    const testName = 'Digio eMandate Status';
    const result = await this.makeRequest(`/digio/emandate/status/${mandateId}`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      mandateStatus: result.data?.data?.status
    });
    
    this.testResults.digio.emandateStatus = result;
  }

  async testDigioPANVerification() {
    const testName = 'Digio PAN Verification';
    const panData = {
      panNumber: 'ABCDE1234F'
    };

    const result = await this.makeRequest('/digio/pan/verify', 'POST', panData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      panStatus: result.data?.data?.status
    });
    
    this.testResults.digio.panVerification = result;
  }

  async testDigioCKYCPull() {
    const testName = 'Digio CKYC Pull';
    const ckycData = {
      ckycData: {
        panNumber: 'ABCDE1234F',
        aadhaarNumber: '123456789012',
        mobile: '9876543210',
        email: 'john.doe@example.com'
      }
    };

    const result = await this.makeRequest('/digio/ckyc/pull', 'POST', ckycData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      ckycStatus: result.data?.data?.status
    });
    
    this.testResults.digio.ckycPull = result;
  }

  async testDigioESignInitiation() {
    const testName = 'Digio eSign Initiation';
    const esignData = {
      esignData: {
        customerDetails: {
          name: 'John Doe',
          mobile: '9876543210',
          email: 'john.doe@example.com',
          panNumber: 'ABCDE1234F',
          aadhaarNumber: '123456789012'
        },
        documentDetails: {
          title: 'Investment Agreement',
          description: 'Mutual fund investment agreement',
          documentUrl: 'https://example.com/agreement.pdf',
          documentType: 'AGREEMENT'
        },
        signDetails: {
          signType: 'AADHAAR_BASED',
          signLocation: 'BOTTOM_RIGHT',
          signReason: 'AGREEMENT_SIGNING'
        }
      }
    };

    const result = await this.makeRequest('/digio/esign/initiate', 'POST', esignData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      esignId: result.data?.data?.esignId
    });
    
    this.testResults.digio.esignInitiation = result;
    return result.data?.data?.esignId;
  }

  async testDigioESignStatus(esignId) {
    const testName = 'Digio eSign Status';
    const result = await this.makeRequest(`/digio/esign/status/${esignId}`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      esignStatus: result.data?.data?.status
    });
    
    this.testResults.digio.esignStatus = result;
  }

  async testDigioSignedDocument(esignId) {
    const testName = 'Digio Signed Document';
    const result = await this.makeRequest(`/digio/esign/${esignId}/download`);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      documentUrl: result.data?.data?.documentUrl
    });
    
    this.testResults.digio.signedDocument = result;
  }

  async testDigioDocumentSignatureVerification() {
    const testName = 'Digio Document Signature Verification';
    const verificationData = {
      documentHash: 'hash_123456789'
    };

    const result = await this.makeRequest('/digio/esign/verify', 'POST', verificationData);
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      isValid: result.data?.data?.isValid
    });
    
    this.testResults.digio.documentVerification = result;
  }

  async testDigioConsentHistory() {
    const testName = 'Digio Consent History';
    const result = await this.makeRequest('/digio/consent/history/test-customer-123');
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      consentsCount: result.data?.data?.consents?.length || 0
    });
    
    this.testResults.digio.consentHistory = result;
  }

  async testDigioUsageStats() {
    const testName = 'Digio Usage Stats';
    const result = await this.makeRequest('/digio/stats/usage?startDate=2024-01-01&endDate=2024-12-31');
    
    this.logTestResult(testName, result.success && result.data.success, {
      status: result.status,
      totalRequests: result.data?.data?.totalRequests
    });
    
    this.testResults.digio.usageStats = result;
  }

  // Run all tests
  async runAllTests() {
    logger.info('🚀 Starting BSE Star MF and Digio Integration Tests');
    logger.info('=' .repeat(60));

    // BSE Star MF Tests
    logger.info('📊 Testing BSE Star MF APIs...');
    await this.testBSEStarMFHealthCheck();
    
    const clientId = await this.testBSEStarMFClientCreation();
    const schemeCode = await this.testBSEStarMFSchemeMasterData();
    
    if (schemeCode) {
      await this.testBSEStarMFSchemeDetails(schemeCode);
    }
    
    if (clientId && schemeCode) {
      const orderId = await this.testBSEStarMFLumpsumOrder(clientId, schemeCode);
      if (orderId) {
        await this.testBSEStarMFOrderStatus(orderId);
      }
      await this.testBSEStarMFRedemptionOrder(clientId, schemeCode);
    }
    
    await this.testBSEStarMFTransactionReport();
    await this.testBSEStarMFHoldingsReport();
    await this.testBSEStarMFCurrentNAV();
    
    if (clientId) {
      const mandateId = await this.testBSEStarMFEMandateSetup(clientId);
      if (mandateId) {
        await this.testBSEStarMFEMandateStatus(mandateId);
      }
      await this.testBSEStarMFClientFolios(clientId);
    }
    
    await this.testBSEStarMFSchemePerformance();

    // Digio Tests
    logger.info('📋 Testing Digio APIs...');
    await this.testDigioHealthCheck();
    
    const kycId = await this.testDigioKYCInitiation();
    if (kycId) {
      await this.testDigioKYCStatus(kycId);
      await this.testDigioKYCDocuments(kycId);
    }
    
    const mandateId = await this.testDigioEMandateSetup();
    if (mandateId) {
      await this.testDigioEMandateStatus(mandateId);
    }
    
    await this.testDigioPANVerification();
    await this.testDigioCKYCPull();
    
    const esignId = await this.testDigioESignInitiation();
    if (esignId) {
      await this.testDigioESignStatus(esignId);
      await this.testDigioSignedDocument(esignId);
    }
    
    await this.testDigioDocumentSignatureVerification();
    await this.testDigioConsentHistory();
    await this.testDigioUsageStats();

    // Print summary
    this.printSummary();
  }

  printSummary() {
    logger.info('=' .repeat(60));
    logger.info('📈 TEST SUMMARY');
    logger.info('=' .repeat(60));
    
    const { total, passed, failed, errors } = this.testResults.summary;
    
    logger.info(`Total Tests: ${total}`);
    logger.info(`Passed: ${passed} ✅`);
    logger.info(`Failed: ${failed} ❌`);
    logger.info(`Success Rate: ${((passed / total) * 100).toFixed(2)}%`);
    
    if (errors.length > 0) {
      logger.info('\n❌ Failed Tests:');
      errors.forEach(error => {
        logger.error(`  - ${error.test}: ${error.error}`);
      });
    }
    
    logger.info('\n🎯 BSE Star MF API Coverage:');
    const bseTests = Object.keys(this.testResults.bseStarMF).length;
    const bsePassed = Object.values(this.testResults.bseStarMF).filter(r => r.success && r.data?.success).length;
    logger.info(`  Tests: ${bseTests}, Passed: ${bsePassed}, Success Rate: ${((bsePassed / bseTests) * 100).toFixed(2)}%`);
    
    logger.info('\n🎯 Digio API Coverage:');
    const digioTests = Object.keys(this.testResults.digio).length;
    const digioPassed = Object.values(this.testResults.digio).filter(r => r.success && r.data?.success).length;
    logger.info(`  Tests: ${digioTests}, Passed: ${digioPassed}, Success Rate: ${((digioPassed / digioTests) * 100).toFixed(2)}%`);
    
    logger.info('=' .repeat(60));
    
    // Save results to file
    const fs = require('fs');
    const resultsFile = `bse-digio-integration-test-results-${Date.now()}.json`;
    fs.writeFileSync(resultsFile, JSON.stringify(this.testResults, null, 2));
    logger.info(`📄 Detailed results saved to: ${resultsFile}`);
  }
}

// Run the tests
async function runTests() {
  const tester = new BSEStarMFDigioIntegrationTest();
  await tester.runAllTests();
}

// Export for use in other files
module.exports = BSEStarMFDigioIntegrationTest;

// Run if this file is executed directly
if (require.main === module) {
  runTests().catch(error => {
    logger.error('Test execution failed:', error);
    process.exit(1);
  });
} 