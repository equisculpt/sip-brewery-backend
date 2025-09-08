const express = require('express');
const router = express.Router();
const digioController = require('../controllers/digioController');
const auth = require('../middleware/auth');
const { jtiReplayGuard } = require('../middleware/jtiReplayGuard');
const { validateRequest } = require('../middleware/validation');

/**
 * @swagger
 * components:
 *   schemas:
 *     KYCData:
 *       type: object
 *       required:
 *         - customerDetails
 *       properties:
 *         requestId:
 *           type: string
 *           description: Custom request ID (optional)
 *         customerDetails:
 *           type: object
 *           required:
 *             - name
 *             - dateOfBirth
 *             - panNumber
 *             - aadhaarNumber
 *             - mobile
 *             - email
 *             - address
 *           properties:
 *             name:
 *               type: string
 *               description: Full name of the customer
 *             dateOfBirth:
 *               type: string
 *               format: date
 *               description: Date of birth (YYYY-MM-DD)
 *             gender:
 *               type: string
 *               enum: [MALE, FEMALE, OTHER]
 *               description: Gender of the customer
 *             panNumber:
 *               type: string
 *               pattern: '^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
 *               description: PAN number
 *             aadhaarNumber:
 *               type: string
 *               pattern: '^[0-9]{12}$'
 *               description: Aadhaar number
 *             mobile:
 *               type: string
 *               pattern: '^[0-9]{10}$'
 *               description: Mobile number
 *             email:
 *               type: string
 *               format: email
 *               description: Email address
 *             address:
 *               type: object
 *               required:
 *                 - line1
 *                 - city
 *                 - state
 *                 - pincode
 *               properties:
 *                 line1:
 *                   type: string
 *                   description: Address line 1
 *                 line2:
 *                   type: string
 *                   description: Address line 2 (optional)
 *                 city:
 *                   type: string
 *                   description: City
 *                 state:
 *                   type: string
 *                   description: State
 *                 pincode:
 *                   type: string
 *                   pattern: '^[0-9]{6}$'
 *                   description: PIN code
 *         kycType:
 *           type: string
 *           enum: [AADHAAR_BASED, PAN_BASED, OFFLINE]
 *           default: AADHAAR_BASED
 *           description: Type of KYC
 *     
 *     MandateData:
 *       type: object
 *       required:
 *         - customerDetails
 *         - bankDetails
 *         - mandateDetails
 *       properties:
 *         requestId:
 *           type: string
 *           description: Custom request ID (optional)
 *         customerDetails:
 *           type: object
 *           required:
 *             - name
 *             - mobile
 *             - email
 *             - panNumber
 *           properties:
 *             name:
 *               type: string
 *               description: Customer name
 *             mobile:
 *               type: string
 *               pattern: '^[0-9]{10}$'
 *               description: Mobile number
 *             email:
 *               type: string
 *               format: email
 *               description: Email address
 *             panNumber:
 *               type: string
 *               pattern: '^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
 *               description: PAN number
 *         bankDetails:
 *           type: object
 *           required:
 *             - accountNumber
 *             - ifscCode
 *             - accountHolderName
 *           properties:
 *             accountNumber:
 *               type: string
 *               description: Bank account number
 *             ifscCode:
 *               type: string
 *               description: IFSC code
 *             accountHolderName:
 *               type: string
 *               description: Account holder name
 *         mandateDetails:
 *           type: object
 *           required:
 *             - amount
 *             - frequency
 *             - startDate
 *             - endDate
 *           properties:
 *             amount:
 *               type: number
 *               minimum: 1000
 *               description: Mandate amount
 *             frequency:
 *               type: string
 *               enum: [MONTHLY, WEEKLY, QUARTERLY]
 *               description: Mandate frequency
 *             startDate:
 *               type: string
 *               format: date
 *               description: Start date
 *             endDate:
 *               type: string
 *               format: date
 *               description: End date
 *             purpose:
 *               type: string
 *               default: MUTUAL_FUND_INVESTMENT
 *               description: Mandate purpose
 *     
 *     CKYCData:
 *       type: object
 *       required:
 *         - panNumber
 *       properties:
 *         panNumber:
 *           type: string
 *           pattern: '^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
 *           description: PAN number
 *         aadhaarNumber:
 *           type: string
 *           pattern: '^[0-9]{12}$'
 *           description: Aadhaar number
 *         mobile:
 *           type: string
 *           pattern: '^[0-9]{10}$'
 *           description: Mobile number
 *         email:
 *           type: string
 *           format: email
 *           description: Email address
 *     
 *     ESignData:
 *       type: object
 *       required:
 *         - customerDetails
 *         - documentDetails
 *         - signDetails
 *       properties:
 *         requestId:
 *           type: string
 *           description: Custom request ID (optional)
 *         customerDetails:
 *           type: object
 *           required:
 *             - name
 *             - mobile
 *             - email
 *             - panNumber
 *             - aadhaarNumber
 *           properties:
 *             name:
 *               type: string
 *               description: Customer name
 *             mobile:
 *               type: string
 *               pattern: '^[0-9]{10}$'
 *               description: Mobile number
 *             email:
 *               type: string
 *               format: email
 *               description: Email address
 *             panNumber:
 *               type: string
 *               pattern: '^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
 *               description: PAN number
 *             aadhaarNumber:
 *               type: string
 *               pattern: '^[0-9]{12}$'
 *               description: Aadhaar number
 *         documentDetails:
 *           type: object
 *           required:
 *             - title
 *             - description
 *             - documentUrl
 *           properties:
 *             title:
 *               type: string
 *               description: Document title
 *             description:
 *               type: string
 *               description: Document description
 *             documentUrl:
 *               type: string
 *               format: uri
 *               description: URL of the document to be signed
 *             documentType:
 *               type: string
 *               default: AGREEMENT
 *               description: Type of document
 *         signDetails:
 *           type: object
 *           required:
 *             - signType
 *           properties:
 *             signType:
 *               type: string
 *               enum: [AADHAAR_BASED, PAN_BASED, DSC]
 *               default: AADHAAR_BASED
 *               description: Type of signature
 *             signLocation:
 *               type: string
 *               default: BOTTOM_RIGHT
 *               description: Location for signature
 *             signReason:
 *               type: string
 *               default: AGREEMENT_SIGNING
 *               description: Reason for signing
 */

/**
 * @swagger
 * tags:
 *   name: Digio
 *   description: Digio API endpoints for KYC, eMandate, and eSign operations
 */

/**
 * @swagger
 * /api/digio/kyc/initiate:
 *   post:
 *     summary: Initiate KYC verification
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - kycData
 *             properties:
 *               kycData:
 *                 $ref: '#/components/schemas/KYCData'
 *     responses:
 *       201:
 *         description: KYC initiated successfully
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               properties:
 *                 success:
 *                   type: boolean
 *                   example: true
 *                 message:
 *                   type: string
 *                   example: KYC initiated successfully
 *                 data:
 *                   type: object
 *                   properties:
 *                     kycId:
 *                       type: string
 *                     requestId:
 *                       type: string
 *                     status:
 *                       type: string
 *                       example: PENDING
 *                     kycUrl:
 *                       type: string
 *                       format: uri
 *                     expiresAt:
 *                       type: string
 *                       format: date-time
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/kyc/initiate', auth, jtiReplayGuard(), validateRequest, digioController.initiateKYC);

/**
 * @swagger
 * /api/digio/kyc/status/{kycId}:
 *   get:
 *     summary: Get KYC status
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: kycId
 *         required: true
 *         schema:
 *           type: string
 *         description: KYC ID
 *     responses:
 *       200:
 *         description: KYC status retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: KYC request not found
 *       500:
 *         description: Internal server error
 */
router.get('/kyc/status/:kycId', auth, digioController.getKYCStatus);

/**
 * @swagger
 * /api/digio/kyc/{kycId}/documents:
 *   get:
 *     summary: Download KYC documents
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: kycId
 *         required: true
 *         schema:
 *           type: string
 *         description: KYC ID
 *       - in: query
 *         name: type
 *         schema:
 *           type: string
 *           default: ALL
 *           enum: [ALL, AADHAAR, PAN]
 *         description: Document type filter
 *     responses:
 *       200:
 *         description: KYC documents retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: KYC documents not found
 *       500:
 *         description: Internal server error
 */
router.get('/kyc/:kycId/documents', auth, digioController.downloadKYCDocuments);

/**
 * @swagger
 * /api/digio/emandate/setup:
 *   post:
 *     summary: Setup eMandate
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - mandateData
 *             properties:
 *               mandateData:
 *                 $ref: '#/components/schemas/MandateData'
 *     responses:
 *       201:
 *         description: eMandate setup initiated successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/emandate/setup', auth, jtiReplayGuard(), validateRequest, digioController.setupEMandate);

/**
 * @swagger
 * /api/digio/emandate/status/{mandateId}:
 *   get:
 *     summary: Get eMandate status
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: mandateId
 *         required: true
 *         schema:
 *           type: string
 *         description: Mandate ID
 *     responses:
 *       200:
 *         description: eMandate status retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Mandate not found
 *       500:
 *         description: Internal server error
 */
router.get('/emandate/status/:mandateId', auth, digioController.getEMandateStatus);

/**
 * @swagger
 * /api/digio/emandate/cancel/{mandateId}:
 *   post:
 *     summary: Cancel eMandate
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: mandateId
 *         required: true
 *         schema:
 *           type: string
 *         description: Mandate ID
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               reason:
 *                 type: string
 *                 default: USER_REQUESTED
 *                 description: Cancellation reason
 *     responses:
 *       200:
 *         description: eMandate cancelled successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Mandate not found
 *       500:
 *         description: Internal server error
 */
router.post('/emandate/cancel/:mandateId', auth, jtiReplayGuard(), validateRequest, digioController.cancelEMandate);

/**
 * @swagger
 * /api/digio/pan/verify:
 *   post:
 *     summary: Verify PAN number
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - panNumber
 *             properties:
 *               panNumber:
 *                 type: string
 *                 pattern: '^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
 *                 description: PAN number to verify
 *     responses:
 *       200:
 *         description: PAN verification completed
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/pan/verify', auth, jtiReplayGuard(), validateRequest, digioController.verifyPAN);

/**
 * @swagger
 * /api/digio/ckyc/pull:
 *   post:
 *     summary: Pull CKYC data
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - ckycData
 *             properties:
 *               ckycData:
 *                 $ref: '#/components/schemas/CKYCData'
 *     responses:
 *       200:
 *         description: CKYC data retrieved successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/ckyc/pull', auth, jtiReplayGuard(), validateRequest, digioController.pullCKYC);

/**
 * @swagger
 * /api/digio/esign/initiate:
 *   post:
 *     summary: Initiate eSign process
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - esignData
 *             properties:
 *               esignData:
 *                 $ref: '#/components/schemas/ESignData'
 *     responses:
 *       201:
 *         description: eSign initiated successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/esign/initiate', auth, jtiReplayGuard(), validateRequest, digioController.initiateESign);

/**
 * @swagger
 * /api/digio/esign/status/{esignId}:
 *   get:
 *     summary: Get eSign status
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: esignId
 *         required: true
 *         schema:
 *           type: string
 *         description: eSign ID
 *     responses:
 *       200:
 *         description: eSign status retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: eSign request not found
 *       500:
 *         description: Internal server error
 */
router.get('/esign/status/:esignId', auth, digioController.getESignStatus);

/**
 * @swagger
 * /api/digio/esign/{esignId}/download:
 *   get:
 *     summary: Download signed document
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: esignId
 *         required: true
 *         schema:
 *           type: string
 *         description: eSign ID
 *     responses:
 *       200:
 *         description: Signed document retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Signed document not found
 *       500:
 *         description: Internal server error
 */
router.get('/esign/:esignId/download', auth, digioController.downloadSignedDocument);

/**
 * @swagger
 * /api/digio/esign/verify:
 *   post:
 *     summary: Verify document signature
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - documentHash
 *             properties:
 *               documentHash:
 *                 type: string
 *                 description: Hash of the document to verify
 *     responses:
 *       200:
 *         description: Document signature verification completed
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/esign/verify', auth, jtiReplayGuard(), validateRequest, digioController.verifyDocumentSignature);

/**
 * @swagger
 * /api/digio/consent/history/{customerId}:
 *   get:
 *     summary: Get customer consent history
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: customerId
 *         required: true
 *         schema:
 *           type: string
 *         description: Customer ID
 *     responses:
 *       200:
 *         description: Consent history retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Customer not found
 *       500:
 *         description: Internal server error
 */
router.get('/consent/history/:customerId', auth, digioController.getConsentHistory);

/**
 * @swagger
 * /api/digio/consent/revoke/{consentId}:
 *   post:
 *     summary: Revoke customer consent
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: consentId
 *         required: true
 *         schema:
 *           type: string
 *         description: Consent ID
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               reason:
 *                 type: string
 *                 default: USER_REQUESTED
 *                 description: Revocation reason
 *     responses:
 *       200:
 *         description: Consent revoked successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Consent not found
 *       500:
 *         description: Internal server error
 */
router.post('/consent/revoke/:consentId', auth, jtiReplayGuard(), validateRequest, digioController.revokeConsent);

/**
 * @swagger
 * /api/digio/health:
 *   get:
 *     summary: Health check for Digio service
 *     tags: [Digio]
 *     responses:
 *       200:
 *         description: Service is healthy
 *       503:
 *         description: Service is unhealthy
 *       500:
 *         description: Internal server error
 */
router.get('/health', digioController.healthCheck);

/**
 * @swagger
 * /api/digio/stats/usage:
 *   get:
 *     summary: Get usage statistics
 *     tags: [Digio]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: startDate
 *         required: true
 *         schema:
 *           type: string
 *           format: date
 *         description: Start date for statistics
 *       - in: query
 *         name: endDate
 *         required: true
 *         schema:
 *           type: string
 *           format: date
 *         description: End date for statistics
 *     responses:
 *       200:
 *         description: Usage statistics retrieved successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/stats/usage', auth, digioController.getUsageStats);

/**
 * @swagger
 * /api/digio/webhook/kyc:
 *   post:
 *     summary: KYC webhook callback
 *     tags: [Digio]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               kycId:
 *                 type: string
 *                 description: KYC ID
 *               status:
 *                 type: string
 *                 enum: [PENDING, COMPLETED, FAILED, EXPIRED]
 *                 description: KYC status
 *               data:
 *                 type: object
 *                 description: Additional data
 *     responses:
 *       200:
 *         description: KYC callback processed successfully
 *       500:
 *         description: Callback processing failed
 */
router.post('/webhook/kyc', digioController.kycCallback);

/**
 * @swagger
 * /api/digio/webhook/mandate:
 *   post:
 *     summary: eMandate webhook callback
 *     tags: [Digio]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               mandateId:
 *                 type: string
 *                 description: Mandate ID
 *               status:
 *                 type: string
 *                 enum: [PENDING, ACTIVE, REJECTED, EXPIRED]
 *                 description: Mandate status
 *               data:
 *                 type: object
 *                 description: Additional data
 *     responses:
 *       200:
 *         description: Mandate callback processed successfully
 *       500:
 *         description: Callback processing failed
 */
router.post('/webhook/mandate', digioController.mandateCallback);

/**
 * @swagger
 * /api/digio/webhook/esign:
 *   post:
 *     summary: eSign webhook callback
 *     tags: [Digio]
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               esignId:
 *                 type: string
 *                 description: eSign ID
 *               status:
 *                 type: string
 *                 enum: [PENDING, COMPLETED, FAILED, EXPIRED]
 *                 description: eSign status
 *               data:
 *                 type: object
 *                 description: Additional data
 *     responses:
 *       200:
 *         description: eSign callback processed successfully
 *       500:
 *         description: Callback processing failed
 */
router.post('/webhook/esign', digioController.esignCallback);

module.exports = router; 