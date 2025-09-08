const logger = require('../utils/logger');

class DemoDigioService {
  constructor() {
    this.kycRequests = new Map();
    this.mandates = new Map();
    this.esignRequests = new Map();
    this.panVerifications = new Map();
    this.ckycData = new Map();
  }

  /**
   * 1. KYC Verification API
   */
  async initiateKYC(kycData) {
    try {
      const kycId = `KYC_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const requestId = kycData.requestId || `REQ_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const kycRequest = {
        kycId,
        requestId,
        customerDetails: kycData.customerDetails,
        kycType: kycData.kycType || 'AADHAAR_BASED',
        status: 'PENDING',
        submittedAt: new Date().toISOString(),
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
      };

      this.kycRequests.set(kycId, kycRequest);

      // Simulate KYC completion after 30 seconds
      setTimeout(() => {
        kycRequest.status = 'COMPLETED';
        kycRequest.completedAt = new Date().toISOString();
        kycRequest.verificationDetails = {
          name: kycData.customerDetails.name,
          dateOfBirth: kycData.customerDetails.dateOfBirth,
          gender: kycData.customerDetails.gender,
          address: kycData.customerDetails.address,
          photo: 'https://demo.digio.in/kyc/photo.jpg',
          signature: 'https://demo.digio.in/kyc/signature.jpg'
        };
        kycRequest.documents = [
          {
            type: 'AADHAAR',
            status: 'VERIFIED',
            documentUrl: 'https://demo.digio.in/kyc/aadhaar.pdf'
          },
          {
            type: 'PAN',
            status: 'VERIFIED',
            documentUrl: 'https://demo.digio.in/kyc/pan.pdf'
          }
        ];
        this.kycRequests.set(kycId, kycRequest);
      }, 30000);

      logger.info('Demo Digio KYC initiated', { kycId, requestId });

      return {
        success: true,
        data: {
          kycId,
          requestId,
          status: 'PENDING',
          kycUrl: `https://demo.digio.in/kyc/${kycId}`,
          expiresAt: kycRequest.expiresAt,
          message: 'KYC initiated successfully'
        }
      };
    } catch (error) {
      logger.error('Demo Digio KYC initiation failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getKYCStatus(kycId) {
    try {
      const kycRequest = this.kycRequests.get(kycId);
      if (!kycRequest) {
        return {
          success: false,
          error: 'KYC request not found'
        };
      }

      logger.info('Demo Digio KYC status fetched', { kycId, status: kycRequest.status });

      return {
        success: true,
        data: {
          kycId: kycRequest.kycId,
          requestId: kycRequest.requestId,
          status: kycRequest.status,
          submittedAt: kycRequest.submittedAt,
          completedAt: kycRequest.completedAt,
          verificationDetails: kycRequest.verificationDetails || {},
          documents: kycRequest.documents || [],
          message: 'KYC status retrieved successfully'
        }
      };
    } catch (error) {
      logger.error('Demo Digio KYC status fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async downloadKYCDocuments(kycId, documentType = 'ALL') {
    try {
      const kycRequest = this.kycRequests.get(kycId);
      if (!kycRequest) {
        return {
          success: false,
          error: 'KYC request not found'
        };
      }

      const documents = kycRequest.documents || [];
      const downloadUrls = {};

      documents.forEach(doc => {
        if (documentType === 'ALL' || doc.type === documentType) {
          downloadUrls[doc.type] = doc.documentUrl;
        }
      });

      logger.info('Demo Digio KYC documents download', { kycId, documentType });

      return {
        success: true,
        data: {
          kycId: kycRequest.kycId,
          documents,
          downloadUrls
        }
      };
    } catch (error) {
      logger.error('Demo Digio KYC documents download failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 2. eMandate Setup API (via NPCI/NACH)
   */
  async setupEMandate(mandateData) {
    try {
      const mandateId = `MANDATE_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const requestId = mandateData.requestId || `REQ_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const mandate = {
        mandateId,
        requestId,
        customerDetails: mandateData.customerDetails,
        bankDetails: mandateData.bankDetails,
        mandateDetails: mandateData.mandateDetails,
        status: 'PENDING',
        submittedAt: new Date().toISOString(),
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
      };

      this.mandates.set(mandateId, mandate);

      // Simulate mandate activation after 45 seconds
      setTimeout(() => {
        mandate.status = 'ACTIVE';
        mandate.activatedAt = new Date().toISOString();
        mandate.mandateNumber = `MANDATE_${Date.now()}`;
        this.mandates.set(mandateId, mandate);
      }, 45000);

      logger.info('Demo Digio eMandate setup', { mandateId, requestId });

      return {
        success: true,
        data: {
          mandateId,
          requestId,
          status: 'PENDING',
          mandateUrl: `https://demo.digio.in/mandate/${mandateId}`,
          npcUrl: `https://demo.npc.in/mandate/${mandateId}`,
          expiresAt: mandate.expiresAt,
          message: 'eMandate setup initiated successfully'
        }
      };
    } catch (error) {
      logger.error('Demo Digio eMandate setup failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getEMandateStatus(mandateId) {
    try {
      const mandate = this.mandates.get(mandateId);
      if (!mandate) {
        return {
          success: false,
          error: 'Mandate not found'
        };
      }

      logger.info('Demo Digio eMandate status fetched', { mandateId, status: mandate.status });

      return {
        success: true,
        data: {
          mandateId: mandate.mandateId,
          requestId: mandate.requestId,
          status: mandate.status,
          customerDetails: mandate.customerDetails || {},
          bankDetails: mandate.bankDetails || {},
          mandateDetails: mandate.mandateDetails || {},
          submittedAt: mandate.submittedAt,
          activatedAt: mandate.activatedAt,
          mandateNumber: mandate.mandateNumber,
          message: 'Mandate status retrieved successfully'
        }
      };
    } catch (error) {
      logger.error('Demo Digio eMandate status fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async cancelEMandate(mandateId, reason = 'USER_REQUESTED') {
    try {
      const mandate = this.mandates.get(mandateId);
      if (!mandate) {
        return {
          success: false,
          error: 'Mandate not found'
        };
      }

      mandate.status = 'CANCELLED';
      mandate.cancelledAt = new Date().toISOString();
      mandate.cancellationReason = reason;
      this.mandates.set(mandateId, mandate);

      logger.info('Demo Digio eMandate cancelled', { mandateId, reason });

      return {
        success: true,
        data: {
          mandateId: mandate.mandateId,
          status: 'CANCELLED',
          cancellationDate: mandate.cancelledAt,
          message: 'eMandate cancelled successfully'
        }
      };
    } catch (error) {
      logger.error('Demo Digio eMandate cancellation failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 3. PAN Check + CKYC Pull
   */
  async verifyPAN(panNumber) {
    try {
      const requestId = `PAN_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Simulate PAN verification
      const isPANValid = /^[A-Z]{5}[0-9]{4}[A-Z]{1}$/.test(panNumber);
      
      const panVerification = {
        panNumber,
        requestId,
        status: isPANValid ? 'VALID' : 'INVALID',
        name: isPANValid ? 'JOHN DOE' : null,
        dateOfBirth: isPANValid ? '1990-01-01' : null,
        gender: isPANValid ? 'MALE' : null,
        verifiedAt: new Date().toISOString()
      };

      this.panVerifications.set(requestId, panVerification);

      logger.info('Demo Digio PAN verification', { panNumber, requestId, status: panVerification.status });

      return {
        success: true,
        data: {
          panNumber: panVerification.panNumber,
          requestId: panVerification.requestId,
          status: panVerification.status,
          name: panVerification.name,
          dateOfBirth: panVerification.dateOfBirth,
          gender: panVerification.gender,
          message: isPANValid ? 'PAN verification successful' : 'Invalid PAN number'
        }
      };
    } catch (error) {
      logger.error('Demo Digio PAN verification failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async pullCKYC(ckycData) {
    try {
      const requestId = `CKYC_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Simulate CKYC data pull
      const hasCKYC = Math.random() > 0.3; // 70% chance of having CKYC data

      const ckycPull = {
        requestId,
        panNumber: ckycData.panNumber,
        aadhaarNumber: ckycData.aadhaarNumber,
        mobile: ckycData.mobile,
        email: ckycData.email,
        status: hasCKYC ? 'FOUND' : 'NOT_FOUND',
        pulledAt: new Date().toISOString()
      };

      if (hasCKYC) {
        ckycPull.customerDetails = {
          name: 'JOHN DOE',
          dateOfBirth: '1990-01-01',
          gender: 'MALE',
          address: {
            line1: '123 MAIN STREET',
            city: 'MUMBAI',
            state: 'MAHARASHTRA',
            pincode: '400001'
          }
        };
        ckycPull.kycDetails = {
          kycNumber: `CKYC${Date.now()}`,
          kycDate: '2023-01-01',
          kycStatus: 'VERIFIED'
        };
      }

      this.ckycData.set(requestId, ckycPull);

      logger.info('Demo Digio CKYC pull', { requestId, status: ckycPull.status });

      return {
        success: true,
        data: {
          ckycNumber: ckycPull.kycDetails?.kycNumber,
          requestId: ckycPull.requestId,
          status: ckycPull.status,
          customerDetails: ckycPull.customerDetails || {},
          kycDetails: ckycPull.kycDetails || {},
          message: hasCKYC ? 'CKYC data found' : 'No CKYC data found'
        }
      };
    } catch (error) {
      logger.error('Demo Digio CKYC pull failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * 4. eSign flow for agreements
   */
  async initiateESign(esignData) {
    try {
      const esignId = `ESIGN_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const requestId = esignData.requestId || `REQ_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const esignRequest = {
        esignId,
        requestId,
        customerDetails: esignData.customerDetails,
        documentDetails: esignData.documentDetails,
        signDetails: esignData.signDetails,
        status: 'PENDING',
        submittedAt: new Date().toISOString(),
        expiresAt: new Date(Date.now() + 24 * 60 * 60 * 1000).toISOString()
      };

      this.esignRequests.set(esignId, esignRequest);

      // Simulate eSign completion after 60 seconds
      setTimeout(() => {
        esignRequest.status = 'COMPLETED';
        esignRequest.completedAt = new Date().toISOString();
        esignRequest.signedDocumentUrl = `https://demo.digio.in/esign/${esignId}/signed.pdf`;
        esignRequest.documentHash = `hash_${Date.now()}`;
        this.esignRequests.set(esignId, esignRequest);
      }, 60000);

      logger.info('Demo Digio eSign initiated', { esignId, requestId });

      return {
        success: true,
        data: {
          esignId,
          requestId,
          status: 'PENDING',
          esignUrl: `https://demo.digio.in/esign/${esignId}`,
          expiresAt: esignRequest.expiresAt,
          message: 'eSign initiated successfully'
        }
      };
    } catch (error) {
      logger.error('Demo Digio eSign initiation failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getESignStatus(esignId) {
    try {
      const esignRequest = this.esignRequests.get(esignId);
      if (!esignRequest) {
        return {
          success: false,
          error: 'eSign request not found'
        };
      }

      logger.info('Demo Digio eSign status fetched', { esignId, status: esignRequest.status });

      return {
        success: true,
        data: {
          esignId: esignRequest.esignId,
          requestId: esignRequest.requestId,
          status: esignRequest.status,
          customerDetails: esignRequest.customerDetails || {},
          documentDetails: esignRequest.documentDetails || {},
          signDetails: esignRequest.signDetails || {},
          submittedAt: esignRequest.submittedAt,
          completedAt: esignRequest.completedAt,
          signedDocumentUrl: esignRequest.signedDocumentUrl,
          message: 'eSign status retrieved successfully'
        }
      };
    } catch (error) {
      logger.error('Demo Digio eSign status fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async downloadSignedDocument(esignId) {
    try {
      const esignRequest = this.esignRequests.get(esignId);
      if (!esignRequest) {
        return {
          success: false,
          error: 'eSign request not found'
        };
      }

      if (esignRequest.status !== 'COMPLETED') {
        return {
          success: false,
          error: 'Document not yet signed'
        };
      }

      logger.info('Demo Digio signed document download', { esignId });

      return {
        success: true,
        data: {
          esignId: esignRequest.esignId,
          documentUrl: esignRequest.signedDocumentUrl,
          downloadUrl: esignRequest.signedDocumentUrl,
          documentHash: esignRequest.documentHash,
          signedAt: esignRequest.completedAt
        }
      };
    } catch (error) {
      logger.error('Demo Digio signed document download failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async verifyDocumentSignature(documentHash) {
    try {
      const requestId = `VERIFY_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Simulate signature verification
      const isValid = Math.random() > 0.1; // 90% chance of valid signature

      logger.info('Demo Digio document signature verification', { documentHash, requestId, isValid });

      return {
        success: true,
        data: {
          documentHash,
          requestId,
          isValid,
          signatureDetails: isValid ? {
            signerName: 'JOHN DOE',
            signerPAN: 'ABCDE1234F',
            signedAt: new Date().toISOString(),
            signatureType: 'AADHAAR_BASED'
          } : {},
          verificationDate: new Date().toISOString(),
          message: isValid ? 'Document signature is valid' : 'Document signature is invalid'
        }
      };
    } catch (error) {
      logger.error('Demo Digio document signature verification failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Additional helper methods
   */
  async getConsentHistory(customerId) {
    try {
      // Generate demo consent history
      const consents = [
        {
          consentId: `CONSENT_${customerId}_1`,
          type: 'KYC',
          status: 'ACTIVE',
          grantedAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
          expiresAt: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString()
        },
        {
          consentId: `CONSENT_${customerId}_2`,
          type: 'EMANDATE',
          status: 'ACTIVE',
          grantedAt: new Date(Date.now() - 15 * 24 * 60 * 60 * 1000).toISOString(),
          expiresAt: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000).toISOString()
        }
      ];

      logger.info('Demo Digio consent history fetched', { customerId });

      return {
        success: true,
        data: {
          customerId,
          consents,
          totalConsents: consents.length
        }
      };
    } catch (error) {
      logger.error('Demo Digio consent history fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async revokeConsent(consentId, reason = 'USER_REQUESTED') {
    try {
      logger.info('Demo Digio consent revoked', { consentId, reason });

      return {
        success: true,
        data: {
          consentId,
          status: 'REVOKED',
          revocationDate: new Date().toISOString(),
          message: 'Consent revoked successfully'
        }
      };
    } catch (error) {
      logger.error('Demo Digio consent revocation failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async healthCheck() {
    try {
      logger.info('Demo Digio health check');

      return {
        success: true,
        data: {
          status: 'HEALTHY',
          timestamp: new Date().toISOString(),
          version: '1.0.0',
          services: {
            kyc: 'ACTIVE',
            emandate: 'ACTIVE',
            esign: 'ACTIVE',
            panVerification: 'ACTIVE',
            ckyc: 'ACTIVE'
          }
        }
      };
    } catch (error) {
      logger.error('Demo Digio health check failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }

  async getUsageStats(startDate, endDate) {
    try {
      // Generate demo usage statistics
      const stats = {
        kycRequests: Math.floor(Math.random() * 100) + 50,
        mandateRequests: Math.floor(Math.random() * 50) + 20,
        esignRequests: Math.floor(Math.random() * 30) + 10,
        panVerifications: Math.floor(Math.random() * 200) + 100,
        ckycPulls: Math.floor(Math.random() * 80) + 40,
        totalRequests: 0,
        successRate: 0
      };

      stats.totalRequests = stats.kycRequests + stats.mandateRequests + stats.esignRequests + 
                           stats.panVerifications + stats.ckycPulls;
      stats.successRate = 95 + Math.random() * 5; // 95-100% success rate

      logger.info('Demo Digio usage stats fetched', { startDate, endDate });

      return {
        success: true,
        data: {
          period: { startDate, endDate },
          ...stats
        }
      };
    } catch (error) {
      logger.error('Demo Digio usage stats fetch failed', { error: error.message });
      return {
        success: false,
        error: error.message
      };
    }
  }
}

module.exports = new DemoDigioService(); 