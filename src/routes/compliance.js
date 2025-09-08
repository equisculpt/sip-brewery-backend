const express = require('express');
const router = express.Router();
const complianceController = require('../controllers/complianceController');
const { authenticateUser, adminAuth } = require('../middleware/auth');

/**
 * @swagger
 * /api/compliance/sebi-report:
 *   post:
 *     summary: Generate SEBI report
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               period:
 *                 type: string
 *                 enum: [monthly, quarterly, yearly]
 *                 description: Report period
 *     responses:
 *       200:
 *         description: SEBI report generated successfully
 *       500:
 *         description: Failed to generate SEBI report
 */
router.post('/sebi-report', authenticateUser, complianceController.generateSEBIReport);

/**
 * @swagger
 * /api/compliance/amfi-report:
 *   post:
 *     summary: Generate AMFI report
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               quarter:
 *                 type: string
 *                 enum: [Q1, Q2, Q3, Q4]
 *                 description: Quarter for report
 *     responses:
 *       200:
 *         description: AMFI report generated successfully
 *       500:
 *         description: Failed to generate AMFI report
 */
router.post('/amfi-report', authenticateUser, complianceController.generateAMFIReport);

/**
 * @swagger
 * /api/compliance/violations:
 *   get:
 *     summary: Check regulatory violations
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Regulatory violations checked successfully
 *       500:
 *         description: Failed to check regulatory violations
 */
router.get('/violations', authenticateUser, complianceController.checkRegulatoryViolations);

/**
 * @swagger
 * /api/compliance/admin-reports:
 *   post:
 *     summary: Generate admin reports (Admin only)
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               reportType:
 *                 type: string
 *                 enum: [comprehensive, compliance, user_behavior, risk_assessment, tax_compliance]
 *                 description: Type of report to generate
 *     responses:
 *       200:
 *         description: Admin reports generated successfully
 *       500:
 *         description: Failed to generate admin reports
 */
router.post('/admin-reports', adminAuth, complianceController.generateAdminReports);

/**
 * @swagger
 * /api/compliance/metrics:
 *   get:
 *     summary: Get compliance metrics (Admin only)
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Compliance metrics retrieved successfully
 *       500:
 *         description: Failed to get compliance metrics
 */
router.get('/metrics', adminAuth, complianceController.getComplianceMetrics);

/**
 * @swagger
 * /api/compliance/monitor:
 *   get:
 *     summary: Monitor real-time compliance (Admin only)
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Real-time compliance monitoring completed
 *       500:
 *         description: Failed to monitor real-time compliance
 */
router.get('/monitor', adminAuth, complianceController.monitorRealTimeCompliance);

/**
 * @swagger
 * /api/compliance/user-status:
 *   get:
 *     summary: Get user compliance status
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: User compliance status retrieved successfully
 *       500:
 *         description: Failed to get user compliance status
 */
router.get('/user-status', authenticateUser, complianceController.getUserComplianceStatus);

/**
 * @swagger
 * /api/compliance/dashboard:
 *   get:
 *     summary: Get compliance dashboard (Admin only)
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Compliance dashboard data retrieved successfully
 *       500:
 *         description: Failed to get compliance dashboard data
 */
router.get('/dashboard', adminAuth, complianceController.getComplianceDashboard);

/**
 * @swagger
 * /api/compliance/violation-trends:
 *   get:
 *     summary: Get violation trends (Admin only)
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: query
 *         name: period
 *         schema:
 *           type: string
 *           enum: [1d, 1w, 1m, 3m, 1y]
 *         description: Analysis period
 *     responses:
 *       200:
 *         description: Violation trends retrieved successfully
 *       500:
 *         description: Failed to get violation trends
 */
router.get('/violation-trends', adminAuth, complianceController.getViolationTrends);

/**
 * @swagger
 * /api/compliance/alerts:
 *   get:
 *     summary: Get compliance alerts (Admin only)
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Compliance alerts retrieved successfully
 *       500:
 *         description: Failed to get compliance alerts
 */
router.get('/alerts', adminAuth, complianceController.getComplianceAlerts);

/**
 * @swagger
 * /api/compliance/download/:reportType/:period:
 *   get:
 *     summary: Download compliance report
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     parameters:
 *       - in: path
 *         name: reportType
 *         required: true
 *         schema:
 *           type: string
 *           enum: [SEBI, AMFI]
 *         description: Type of report
 *       - in: path
 *         name: period
 *         required: true
 *         schema:
 *           type: string
 *         description: Report period
 *     responses:
 *       200:
 *         description: Report downloaded successfully
 *       404:
 *         description: Report file not found
 *       500:
 *         description: Failed to download report
 */
router.get('/download/:reportType/:period', authenticateUser, complianceController.downloadComplianceReport);

/**
 * @swagger
 * /api/compliance/summary:
 *   get:
 *     summary: Get compliance summary for user
 *     tags: [Compliance]
 *     security:
 *       - bearerAuth: []
 *     responses:
 *       200:
 *         description: Compliance summary retrieved successfully
 *       500:
 *         description: Failed to get compliance summary
 */
router.get('/summary', authenticateUser, complianceController.getComplianceSummary);

module.exports = router; 