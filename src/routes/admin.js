const express = require('express');
const router = express.Router();
const adminController = require('../controllers/adminController');
const agentController = require('../controllers/agentController');
const clientController = require('../controllers/clientController');
const commissionController = require('../controllers/commissionController');
const analyticsController = require('../controllers/analyticsController');
const kycController = require('../controllers/kycController');
const transactionController = require('../controllers/transactionController');
const rewardController = require('../controllers/rewardController');
const pdfController = require('../controllers/pdfStatementController');
const settingsController = require('../controllers/settingsController');
const leaderboardController = require('../controllers/leaderboardController');
const notificationController = require('../controllers/notificationController');

const {
  verifyToken,
  requireRole,
  requirePermission,
  requireAgentAccess,
  requireRegionalAccess,
  requireClientAccess,
  rateLimit,
  auditLog
} = require('../middleware/adminAuth');

// Apply rate limiting to all admin routes
router.use(rateLimit({ windowMs: 15 * 60 * 1000, max: 1000 }));

// ============================================================================
// AUTHENTICATION ROUTES
// ============================================================================

/**
 * @route POST /api/admin/auth/login
 * @desc Admin login
 * @access Public
 */
router.post('/auth/login', 
  auditLog('LOGIN_ATTEMPT', 'auth'),
  adminController.login
);

/**
 * @route POST /api/admin/auth/logout
 * @desc Admin logout
 * @access Private
 */
router.post('/auth/logout',
  verifyToken,
  auditLog('LOGOUT', 'auth'),
  adminController.logout
);

/**
 * @route GET /api/admin/auth/profile
 * @desc Get admin profile
 * @access Private
 */
router.get('/auth/profile',
  verifyToken,
  adminController.getProfile
);

/**
 * @route PUT /api/admin/auth/profile
 * @desc Update admin profile
 * @access Private
 */
router.put('/auth/profile',
  verifyToken,
  auditLog('UPDATE_PROFILE', 'admin'),
  adminController.updateProfile
);

// ============================================================================
// DASHBOARD ROUTES
// ============================================================================

/**
 * @route GET /api/admin/dashboard
 * @desc Get admin dashboard data
 * @access Private
 */
router.get('/dashboard',
  verifyToken,
  adminController.getDashboard
);

// ============================================================================
// AGENT MANAGEMENT ROUTES
// ============================================================================

/**
 * @route GET /api/admin/agents
 * @desc Get all agents with stats
 * @access Private - ADMIN, SUPER_ADMIN
 */
router.get('/agents',
  verifyToken,
  requireRole(['ADMIN', 'SUPER_ADMIN']),
  auditLog('VIEW_AGENTS', 'agents'),
  agentController.getAllAgents
);

/**
 * @route GET /api/admin/agents/:id
 * @desc Get agent details
 * @access Private - ADMIN, SUPER_ADMIN, AGENT (own data)
 */
router.get('/agents/:id',
  verifyToken,
  requireAgentAccess,
  auditLog('VIEW_AGENT_DETAILS', 'agents'),
  agentController.getAgentById
);

/**
 * @route POST /api/admin/agents
 * @desc Create new agent
 * @access Private - SUPER_ADMIN
 */
router.post('/agents',
  verifyToken,
  requireRole('SUPER_ADMIN'),
  auditLog('CREATE_AGENT', 'agents'),
  agentController.createAgent
);

/**
 * @route PUT /api/admin/agents/:id
 * @desc Update agent
 * @access Private - SUPER_ADMIN, ADMIN
 */
router.put('/agents/:id',
  verifyToken,
  requireRole(['SUPER_ADMIN', 'ADMIN']),
  auditLog('UPDATE_AGENT', 'agents'),
  agentController.updateAgent
);

/**
 * @route DELETE /api/admin/agents/:id
 * @desc Deactivate agent
 * @access Private - SUPER_ADMIN
 */
router.delete('/agents/:id',
  verifyToken,
  requireRole('SUPER_ADMIN'),
  auditLog('DEACTIVATE_AGENT', 'agents'),
  agentController.deactivateAgent
);

/**
 * @route GET /api/admin/agents/:id/dashboard
 * @desc Get agent dashboard
 * @access Private - ADMIN, SUPER_ADMIN, AGENT (own data)
 */
router.get('/agents/:id/dashboard',
  verifyToken,
  requireAgentAccess,
  auditLog('VIEW_AGENT_DASHBOARD', 'agents'),
  agentController.getAgentDashboard
);

// ============================================================================
// CLIENT MANAGEMENT ROUTES
// ============================================================================

/**
 * @route GET /api/admin/clients
 * @desc Get all clients with filters
 * @access Private - All roles
 */
router.get('/clients',
  verifyToken,
  auditLog('VIEW_CLIENTS', 'clients'),
  clientController.getAllClients
);

/**
 * @route GET /api/admin/clients/:id
 * @desc Get client details
 * @access Private - All roles (with access control)
 */
router.get('/clients/:id',
  verifyToken,
  requireClientAccess,
  auditLog('VIEW_CLIENT_DETAILS', 'clients'),
  clientController.getClientById
);

/**
 * @route GET /api/admin/clients/:id/dashboard
 * @desc Get client dashboard
 * @access Private - All roles (with access control)
 */
router.get('/clients/:id/dashboard',
  verifyToken,
  requireClientAccess,
  auditLog('VIEW_CLIENT_DASHBOARD', 'clients'),
  clientController.getClientDashboard
);

/**
 * @route PUT /api/admin/clients/:id/assign-agent
 * @desc Assign client to agent
 * @access Private - ADMIN, SUPER_ADMIN
 */
router.put('/clients/:id/assign-agent',
  verifyToken,
  requireRole(['ADMIN', 'SUPER_ADMIN']),
  auditLog('ASSIGN_CLIENT_TO_AGENT', 'clients'),
  clientController.assignAgent
);

// ============================================================================
// COMMISSION ROUTES
// ============================================================================

/**
 * @route GET /api/admin/commission
 * @desc Get commission report
 * @access Private - All roles
 */
router.get('/commission',
  verifyToken,
  auditLog('VIEW_COMMISSION_REPORT', 'commission'),
  commissionController.getCommissionReport
);

/**
 * @route GET /api/admin/commission/agent/:id
 * @desc Get agent commission details
 * @access Private - ADMIN, SUPER_ADMIN, AGENT (own data)
 */
router.get('/commission/agent/:id',
  verifyToken,
  requireAgentAccess,
  auditLog('VIEW_AGENT_COMMISSION', 'commission'),
  commissionController.getAgentCommission
);

/**
 * @route POST /api/admin/commission/approve
 * @desc Approve commission payout
 * @access Private - ADMIN, SUPER_ADMIN
 */
router.post('/commission/approve',
  verifyToken,
  requireRole(['ADMIN', 'SUPER_ADMIN']),
  auditLog('APPROVE_COMMISSION', 'commission'),
  commissionController.approveCommission
);

/**
 * @route POST /api/admin/commission/process-payout
 * @desc Process commission payout
 * @access Private - ADMIN, SUPER_ADMIN
 */
router.post('/commission/process-payout',
  verifyToken,
  requireRole(['ADMIN', 'SUPER_ADMIN']),
  auditLog('PROCESS_COMMISSION_PAYOUT', 'commission'),
  commissionController.processPayout
);

/**
 * @route GET /api/admin/commission/export
 * @desc Export commission report
 * @access Private - ADMIN, SUPER_ADMIN
 */
router.get('/commission/export',
  verifyToken,
  requireRole(['ADMIN', 'SUPER_ADMIN']),
  auditLog('EXPORT_COMMISSION_REPORT', 'commission'),
  commissionController.exportCommissionReport
);

// ============================================================================
// ANALYTICS ROUTES
// ============================================================================

/**
 * @route GET /api/admin/analytics/platform
 * @desc Get platform analytics
 * @access Private - SUPER_ADMIN
 */
router.get('/analytics/platform',
  verifyToken,
  requireRole('SUPER_ADMIN'),
  auditLog('VIEW_PLATFORM_ANALYTICS', 'analytics'),
  analyticsController.getPlatformAnalytics
);

/**
 * @route GET /api/admin/analytics/regional
 * @desc Get regional analytics
 * @access Private - ADMIN, SUPER_ADMIN
 */
router.get('/analytics/regional',
  verifyToken,
  requireRole(['ADMIN', 'SUPER_ADMIN']),
  requireRegionalAccess,
  auditLog('VIEW_REGIONAL_ANALYTICS', 'analytics'),
  analyticsController.getRegionalAnalytics
);

/**
 * @route GET /api/admin/analytics/agent/:id
 * @desc Get agent analytics
 * @access Private - ADMIN, SUPER_ADMIN, AGENT (own data)
 */
router.get('/analytics/agent/:id',
  verifyToken,
  requireAgentAccess,
  auditLog('VIEW_AGENT_ANALYTICS', 'analytics'),
  analyticsController.getAgentAnalytics
);

// ============================================================================
// KYC ROUTES
// ============================================================================

/**
 * @route GET /api/admin/kyc/status
 * @desc Get KYC status report
 * @access Private - All roles
 */
router.get('/kyc/status',
  verifyToken,
  auditLog('VIEW_KYC_STATUS', 'kyc'),
  kycController.getKYCStatus
);

/**
 * @route POST /api/admin/kyc/retrigger
 * @desc Re-trigger KYC for client
 * @access Private - ADMIN, SUPER_ADMIN
 */
router.post('/kyc/retrigger',
  verifyToken,
  requireRole(['ADMIN', 'SUPER_ADMIN']),
  auditLog('RETRIGGER_KYC', 'kyc'),
  kycController.retriggerKYC
);

/**
 * @route GET /api/admin/kyc/logs/:clientId
 * @desc Get KYC logs for client
 * @access Private - All roles (with access control)
 */
router.get('/kyc/logs/:clientId',
  verifyToken,
  requireClientAccess,
  auditLog('VIEW_KYC_LOGS', 'kyc'),
  kycController.getKYCLogs
);

// ============================================================================
// TRANSACTION ROUTES
// ============================================================================

/**
 * @route GET /api/admin/transactions
 * @desc Get transaction logs
 * @access Private - All roles
 */
router.get('/transactions',
  verifyToken,
  auditLog('VIEW_TRANSACTIONS', 'transactions'),
  transactionController.getTransactionLogs
);

/**
 * @route GET /api/admin/transactions/pending
 * @desc Get pending transactions
 * @access Private - All roles
 */
router.get('/transactions/pending',
  verifyToken,
  auditLog('VIEW_PENDING_TRANSACTIONS', 'transactions'),
  transactionController.getPendingTransactions
);

// ============================================================================
// REWARDS ROUTES
// ============================================================================

/**
 * @route GET /api/admin/rewards
 * @desc Get rewards report
 * @access Private - All roles
 */
router.get('/rewards',
  verifyToken,
  auditLog('VIEW_REWARDS_REPORT', 'rewards'),
  rewardController.getRewardsReport
);

/**
 * @route PUT /api/admin/rewards/:id/status
 * @desc Update reward status
 * @access Private - ADMIN, SUPER_ADMIN
 */
router.put('/rewards/:id/status',
  verifyToken,
  requireRole(['ADMIN', 'SUPER_ADMIN']),
  auditLog('UPDATE_REWARD_STATUS', 'rewards'),
  rewardController.updateRewardStatus
);

// ============================================================================
// PDF ROUTES
// ============================================================================

/**
 * @route POST /api/admin/pdf/generate
 * @desc Generate PDF statement
 * @access Private - All roles
 */
router.post('/pdf/generate',
  verifyToken,
  auditLog('GENERATE_PDF_STATEMENT', 'pdf'),
  pdfController.generateStatement
);

/**
 * @route POST /api/admin/pdf/send
 * @desc Send PDF to client
 * @access Private - All roles
 */
router.post('/pdf/send',
  verifyToken,
  auditLog('SEND_PDF_TO_CLIENT', 'pdf'),
  pdfController.sendPDFToClient
);

// ============================================================================
// LEADERBOARD ROUTES
// ============================================================================

/**
 * @route GET /api/admin/leaderboard/agents
 * @desc Get agent leaderboard
 * @access Private - All roles
 */
router.get('/leaderboard/agents',
  verifyToken,
  auditLog('VIEW_AGENT_LEADERBOARD', 'leaderboard'),
  leaderboardController.getAgentLeaderboard
);

/**
 * @route GET /api/admin/leaderboard/performance
 * @desc Get performance leaderboard
 * @access Private - All roles
 */
router.get('/leaderboard/performance',
  verifyToken,
  auditLog('VIEW_PERFORMANCE_LEADERBOARD', 'leaderboard'),
  leaderboardController.getPerformanceLeaderboard
);

// ============================================================================
// NOTIFICATION ROUTES
// ============================================================================

/**
 * @route GET /api/admin/notifications
 * @desc Get notifications
 * @access Private - All roles
 */
router.get('/notifications',
  verifyToken,
  notificationController.getNotifications
);

/**
 * @route PUT /api/admin/notifications/:id/read
 * @desc Mark notification as read
 * @access Private - All roles
 */
router.put('/notifications/:id/read',
  verifyToken,
  notificationController.markAsRead
);

/**
 * @route PUT /api/admin/notifications/read-all
 * @desc Mark all notifications as read
 * @access Private - All roles
 */
router.put('/notifications/read-all',
  verifyToken,
  notificationController.markAllAsRead
);

// ============================================================================
// SETTINGS ROUTES
// ============================================================================

/**
 * @route GET /api/admin/settings
 * @desc Get platform settings
 * @access Private - SUPER_ADMIN
 */
router.get('/settings',
  verifyToken,
  requireRole('SUPER_ADMIN'),
  auditLog('VIEW_SETTINGS', 'settings'),
  settingsController.getSettings
);

/**
 * @route PUT /api/admin/settings
 * @desc Update platform settings
 * @access Private - SUPER_ADMIN
 */
router.put('/settings',
  verifyToken,
  requireRole('SUPER_ADMIN'),
  auditLog('UPDATE_SETTINGS', 'settings'),
  settingsController.updateSettings
);

/**
 * @route GET /api/admin/settings/logs
 * @desc Get access logs
 * @access Private - SUPER_ADMIN
 */
router.get('/settings/logs',
  verifyToken,
  requireRole('SUPER_ADMIN'),
  auditLog('VIEW_ACCESS_LOGS', 'settings'),
  settingsController.getAccessLogs
);

// ============================================================================
// AUDIT ROUTES
// ============================================================================

/**
 * @route GET /api/admin/audit/logs
 * @desc Get audit logs
 * @access Private - SUPER_ADMIN
 */
router.get('/audit/logs',
  verifyToken,
  requireRole('SUPER_ADMIN'),
  auditLog('VIEW_AUDIT_LOGS', 'audit'),
  require('../controllers/auditController').getAuditLogs
);

/**
 * @route GET /api/admin/audit/security
 * @desc Get security events
 * @access Private - SUPER_ADMIN
 */
router.get('/audit/security',
  verifyToken,
  requireRole('SUPER_ADMIN'),
  auditLog('VIEW_SECURITY_EVENTS', 'audit'),
  require('../controllers/auditController').getSecurityEvents
);

module.exports = router; 