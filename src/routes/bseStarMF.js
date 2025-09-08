const express = require('express');
const router = express.Router();
const bseStarMFController = require('../controllers/bseStarMFController');
const auth = require('../middleware/auth');
const { validateRequest } = require('../middleware/validation');

/**
 * @swagger
 * components:
 *   schemas:
 *     ClientData:
 *       type: object
 *       required:
 *         - firstName
 *         - lastName
 *         - dateOfBirth
 *         - panNumber
 *         - email
 *         - mobile
 *         - address
 *         - bankDetails
 *       properties:
 *         firstName:
 *           type: string
 *           description: First name of the client
 *         lastName:
 *           type: string
 *           description: Last name of the client
 *         dateOfBirth:
 *           type: string
 *           format: date
 *           description: Date of birth (YYYY-MM-DD)
 *         gender:
 *           type: string
 *           enum: [MALE, FEMALE, OTHER]
 *           description: Gender of the client
 *         panNumber:
 *           type: string
 *           pattern: '^[A-Z]{5}[0-9]{4}[A-Z]{1}$'
 *           description: PAN number of the client
 *         aadhaarNumber:
 *           type: string
 *           pattern: '^[0-9]{12}$'
 *           description: Aadhaar number of the client
 *         email:
 *           type: string
 *           format: email
 *           description: Email address of the client
 *         mobile:
 *           type: string
 *           pattern: '^[0-9]{10}$'
 *           description: Mobile number of the client
 *         address:
 *           type: object
 *           required:
 *             - line1
 *             - city
 *             - state
 *             - pincode
 *           properties:
 *             line1:
 *               type: string
 *               description: Address line 1
 *             line2:
 *               type: string
 *               description: Address line 2 (optional)
 *             city:
 *               type: string
 *               description: City
 *             state:
 *               type: string
 *               description: State
 *             pincode:
 *               type: string
 *               pattern: '^[0-9]{6}$'
 *               description: PIN code
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
 *         nomineeDetails:
 *           type: array
 *           items:
 *             type: object
 *             properties:
 *               name:
 *                 type: string
 *               relationship:
 *                 type: string
 *               percentage:
 *                 type: number
 *               dateOfBirth:
 *                 type: string
 *                 format: date
 *         kycDetails:
 *           type: object
 *           properties:
 *             kycStatus:
 *               type: string
 *               enum: [PENDING, APPROVED, REJECTED]
 *             kycNumber:
 *               type: string
 *             kycDate:
 *               type: string
 *               format: date
 *     
 *     OrderData:
 *       type: object
 *       required:
 *         - clientId
 *         - schemeCode
 *         - amount
 *         - paymentMode
 *       properties:
 *         clientId:
 *           type: string
 *           description: Client ID
 *         schemeCode:
 *           type: string
 *           description: Scheme code
 *         amount:
 *           type: number
 *           minimum: 1000
 *           description: Investment amount
 *         paymentMode:
 *           type: string
 *           enum: [ONLINE, CHEQUE, DD]
 *           description: Payment mode
 *         sipDetails:
 *           type: object
 *           properties:
 *             frequency:
 *               type: string
 *               enum: [MONTHLY, WEEKLY, QUARTERLY]
 *             duration:
 *               type: number
 *               minimum: 1
 *         isSmartSIP:
 *           type: boolean
 *           default: false
 *     
 *     RedemptionData:
 *       type: object
 *       required:
 *         - clientId
 *         - schemeCode
 *         - redemptionType
 *       properties:
 *         clientId:
 *           type: string
 *           description: Client ID
 *         schemeCode:
 *           type: string
 *           description: Scheme code
 *         folioNumber:
 *           type: string
 *           description: Folio number
 *         redemptionType:
 *           type: string
 *           enum: [UNITS, AMOUNT]
 *           description: Redemption type
 *         units:
 *           type: number
 *           minimum: 0
 *           description: Number of units to redeem
 *         amount:
 *           type: number
 *           minimum: 0
 *           description: Amount to redeem
 *         bankAccount:
 *           type: string
 *           description: Bank account number
 *         nomineeDetails:
 *           type: array
 *           items:
 *             type: object
 *         redemptionMode:
 *           type: string
 *           enum: [NORMAL, SWITCH]
 *           default: NORMAL
 *     
 *     MandateData:
 *       type: object
 *       required:
 *         - clientId
 *         - bankAccount
 *         - amount
 *         - frequency
 *         - startDate
 *         - endDate
 *       properties:
 *         clientId:
 *           type: string
 *           description: Client ID
 *         bankAccount:
 *           type: object
 *           required:
 *             - accountNumber
 *             - ifscCode
 *             - accountHolderName
 *           properties:
 *             accountNumber:
 *               type: string
 *             ifscCode:
 *               type: string
 *             accountHolderName:
 *               type: string
 *         amount:
 *           type: number
 *           minimum: 1000
 *           description: Mandate amount
 *         frequency:
 *           type: string
 *           enum: [MONTHLY, WEEKLY, QUARTERLY]
 *           description: Mandate frequency
 *         startDate:
 *           type: string
 *           format: date
 *           description: Start date
 *         endDate:
 *           type: string
 *           format: date
 *           description: End date
 *         purpose:
 *           type: string
 *           default: MUTUAL_FUND_INVESTMENT
 */

/**
 * @swagger
 * tags:
 *   name: BSE Star MF
 *   description: BSE Star MF API endpoints for mutual fund operations
 */

/**
 * @swagger
 * /api/bse-star-mf/client:
 *   post:
 *     summary: Create a new client
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - clientData
 *             properties:
 *               clientData:
 *                 $ref: '#/components/schemas/ClientData'
 *     responses:
 *       201:
 *         description: Client created successfully
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
 *                   example: Client created successfully
 *                 data:
 *                   type: object
 *                   properties:
 *                     clientId:
 *                       type: string
 *                     bseClientId:
 *                       type: string
 *                     status:
 *                       type: string
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/client', auth, validateRequest, bseStarMFController.createClient);

/**
 * @swagger
 * /api/bse-star-mf/client/{clientId}:
 *   put:
 *     summary: Modify an existing client
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: clientId
 *         required: true
 *         schema:
 *           type: string
 *         description: Client ID
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - clientData
 *             properties:
 *               clientData:
 *                 $ref: '#/components/schemas/ClientData'
 *     responses:
 *       200:
 *         description: Client modified successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Client not found
 *       500:
 *         description: Internal server error
 */
router.put('/client/:clientId', auth, validateRequest, bseStarMFController.modifyClient);

/**
 * @swagger
 * /api/bse-star-mf/schemes:
 *   get:
 *     summary: Get scheme master data
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: category
 *         schema:
 *           type: string
 *         description: Fund category filter
 *       - in: query
 *         name: fundHouse
 *         schema:
 *           type: string
 *         description: Fund house filter
 *       - in: query
 *         name: isActive
 *         schema:
 *           type: boolean
 *         description: Active status filter
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *         description: Number of records to return
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           default: 0
 *         description: Number of records to skip
 *     responses:
 *       200:
 *         description: Scheme master data retrieved successfully
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/schemes', auth, bseStarMFController.getSchemeMasterData);

/**
 * @swagger
 * /api/bse-star-mf/schemes/{schemeCode}:
 *   get:
 *     summary: Get scheme details
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: schemeCode
 *         required: true
 *         schema:
 *           type: string
 *         description: Scheme code
 *     responses:
 *       200:
 *         description: Scheme details retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Scheme not found
 *       500:
 *         description: Internal server error
 */
router.get('/schemes/:schemeCode', auth, bseStarMFController.getSchemeDetails);

/**
 * @swagger
 * /api/bse-star-mf/order/lumpsum:
 *   post:
 *     summary: Place a lumpsum order
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - orderData
 *             properties:
 *               orderData:
 *                 $ref: '#/components/schemas/OrderData'
 *     responses:
 *       201:
 *         description: Order placed successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/order/lumpsum', auth, validateRequest, bseStarMFController.placeLumpsumOrder);

/**
 * @swagger
 * /api/bse-star-mf/order/status/{orderId}:
 *   get:
 *     summary: Get order status
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: orderId
 *         required: true
 *         schema:
 *           type: string
 *         description: Order ID
 *     responses:
 *       200:
 *         description: Order status retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Order not found
 *       500:
 *         description: Internal server error
 */
router.get('/order/status/:orderId', auth, bseStarMFController.getOrderStatus);

/**
 * @swagger
 * /api/bse-star-mf/order/redemption:
 *   post:
 *     summary: Place a redemption order
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - redemptionData
 *             properties:
 *               redemptionData:
 *                 $ref: '#/components/schemas/RedemptionData'
 *     responses:
 *       201:
 *         description: Redemption order placed successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/order/redemption', auth, validateRequest, bseStarMFController.placeRedemptionOrder);

/**
 * @swagger
 * /api/bse-star-mf/report/transactions:
 *   get:
 *     summary: Get transaction report
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: clientId
 *         schema:
 *           type: string
 *         description: Client ID filter
 *       - in: query
 *         name: schemeCode
 *         schema:
 *           type: string
 *         description: Scheme code filter
 *       - in: query
 *         name: folioNumber
 *         schema:
 *           type: string
 *         description: Folio number filter
 *       - in: query
 *         name: startDate
 *         schema:
 *           type: string
 *           format: date
 *         description: Start date filter
 *       - in: query
 *         name: endDate
 *         schema:
 *           type: string
 *           format: date
 *         description: End date filter
 *       - in: query
 *         name: orderType
 *         schema:
 *           type: string
 *         description: Order type filter
 *       - in: query
 *         name: limit
 *         schema:
 *           type: integer
 *           default: 20
 *         description: Number of records to return
 *       - in: query
 *         name: offset
 *         schema:
 *           type: integer
 *           default: 0
 *         description: Number of records to skip
 *     responses:
 *       200:
 *         description: Transaction report retrieved successfully
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/report/transactions', auth, bseStarMFController.getTransactionReport);

/**
 * @swagger
 * /api/bse-star-mf/report/holdings:
 *   get:
 *     summary: Get NAV and holding report
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: clientId
 *         schema:
 *           type: string
 *         description: Client ID filter
 *       - in: query
 *         name: schemeCode
 *         schema:
 *           type: string
 *         description: Scheme code filter
 *       - in: query
 *         name: folioNumber
 *         schema:
 *           type: string
 *         description: Folio number filter
 *       - in: query
 *         name: date
 *         schema:
 *           type: string
 *           format: date
 *         description: Report date
 *     responses:
 *       200:
 *         description: NAV and holding report retrieved successfully
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.get('/report/holdings', auth, bseStarMFController.getNAVAndHoldingReport);

/**
 * @swagger
 * /api/bse-star-mf/nav/current:
 *   post:
 *     summary: Get current NAV for schemes
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required:
 *               - schemeCodes
 *             properties:
 *               schemeCodes:
 *                 type: array
 *                 items:
 *                   type: string
 *                 description: Array of scheme codes
 *     responses:
 *       200:
 *         description: Current NAV retrieved successfully
 *       400:
 *         description: Bad request
 *       401:
 *         description: Unauthorized
 *       500:
 *         description: Internal server error
 */
router.post('/nav/current', auth, validateRequest, bseStarMFController.getCurrentNAV);

/**
 * @swagger
 * /api/bse-star-mf/emandate/setup:
 *   post:
 *     summary: Setup eMandate
 *     tags: [BSE Star MF]
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
router.post('/emandate/setup', auth, validateRequest, bseStarMFController.setupEMandate);

/**
 * @swagger
 * /api/bse-star-mf/emandate/status/{mandateId}:
 *   get:
 *     summary: Get eMandate status
 *     tags: [BSE Star MF]
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
router.get('/emandate/status/:mandateId', auth, bseStarMFController.getEMandateStatus);

/**
 * @swagger
 * /api/bse-star-mf/emandate/cancel/{mandateId}:
 *   post:
 *     summary: Cancel eMandate
 *     tags: [BSE Star MF]
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
router.post('/emandate/cancel/:mandateId', auth, validateRequest, bseStarMFController.cancelEMandate);

/**
 * @swagger
 * /api/bse-star-mf/client/{clientId}/folios:
 *   get:
 *     summary: Get client folios
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: clientId
 *         required: true
 *         schema:
 *           type: string
 *         description: Client ID
 *     responses:
 *       200:
 *         description: Client folios retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Client not found
 *       500:
 *         description: Internal server error
 */
router.get('/client/:clientId/folios', auth, bseStarMFController.getClientFolios);

/**
 * @swagger
 * /api/bse-star-mf/schemes/{schemeCode}/performance:
 *   get:
 *     summary: Get scheme performance
 *     tags: [BSE Star MF]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: schemeCode
 *         required: true
 *         schema:
 *           type: string
 *         description: Scheme code
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           default: 1Y
 *           enum: [1M, 3M, 6M, 1Y, 3Y, 5Y]
 *         description: Performance period
 *     responses:
 *       200:
 *         description: Scheme performance retrieved successfully
 *       401:
 *         description: Unauthorized
 *       404:
 *         description: Scheme not found
 *       500:
 *         description: Internal server error
 */
router.get('/schemes/:schemeCode/performance', auth, bseStarMFController.getSchemePerformance);

/**
 * @swagger
 * /api/bse-star-mf/health:
 *   get:
 *     summary: Health check for BSE Star MF service
 *     tags: [BSE Star MF]
 *     responses:
 *       200:
 *         description: Service is healthy
 *       503:
 *         description: Service is unhealthy
 *       500:
 *         description: Internal server error
 */
router.get('/health', bseStarMFController.healthCheck);

module.exports = router; 