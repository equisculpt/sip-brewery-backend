/**
 * RewardsController handles user reward summary and transaction retrieval.
 * Enhanced with comprehensive security and validation.
 * @module controllers/rewardsController
 */
const rewardsService = require('../services/rewardsService');
const logger = require('../utils/logger');
const { validateRequest, commonSchemas } = require('../middleware/validation');
const validator = require('validator');

class RewardsController {
  /**
   * Get user's reward summary
   * @route GET /api/rewards/summary
   * @returns {Object} 200 - Success, summary data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 500 - Internal server error
   */
  async getRewardSummary(req, res) {
    try {
      // Enhanced authentication validation
      const userId = req.user?.supabaseId;
      if (!userId || !validator.isUUID(userId)) {
        logger.warn('Invalid authentication attempt in getRewardSummary', {
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          userId: userId ? 'present_but_invalid' : 'missing'
        });
        return res.status(401).json({ 
          success: false, 
          message: 'Valid authentication required.',
          error: 'AUTHENTICATION_REQUIRED'
        });
      }

      // Input validation and sanitization
      if (req.query && Object.keys(req.query).length > 0) {
        logger.warn('Unexpected query parameters in getRewardSummary', {
          userId,
          query: req.query,
          ip: req.ip
        });
      }

      const summary = await rewardsService.getUserRewardSummary(userId);
      
      logger.info('Reward summary retrieved successfully', { 
        userId,
        ip: req.ip,
        timestamp: new Date().toISOString()
      });
      
      res.json({
        success: true,
        data: summary,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error('Error in getRewardSummary:', { 
        error: error.message,
        stack: error.stack,
        userId: req.user?.supabaseId,
        ip: req.ip
      });
      
      res.status(500).json({
        success: false,
        message: 'Failed to fetch reward summary',
        error: 'INTERNAL_SERVER_ERROR',
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Get paginated reward transactions
   * @route GET /api/rewards/transactions
   * @param {number} [req.query.page=1] - Page number
   * @param {number} [req.query.limit=20] - Page size
   * @param {string} [req.query.type] - Transaction type filter
   * @param {string} [req.query.startDate] - Start date filter (ISO)
   * @param {string} [req.query.endDate] - End date filter (ISO)
   * @returns {Object} 200 - Success, transaction data
   * @returns {Object} 401 - Authentication required
   * @returns {Object} 400 - Invalid parameters
   * @returns {Object} 500 - Internal server error
   */
  async getRewardTransactions(req, res) {
    try {
      // Enhanced authentication validation
      const userId = req.user?.supabaseId;
      if (!userId || !validator.isUUID(userId)) {
        logger.warn('Invalid authentication attempt in getRewardTransactions', {
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          userId: userId ? 'present_but_invalid' : 'missing'
        });
        return res.status(401).json({ 
          success: false, 
          message: 'Valid authentication required.',
          error: 'AUTHENTICATION_REQUIRED'
        });
      }

      // Comprehensive input validation
      const { page, limit, type, startDate, endDate } = req.query;
      const errors = [];

      // Validate pagination parameters
      const pageNum = parseInt(page) || 1;
      const limitNum = parseInt(limit) || 20;
      
      if (pageNum < 1) {
        errors.push('Page must be greater than 0');
      }
      if (limitNum < 1 || limitNum > 100) {
        errors.push('Limit must be between 1 and 100');
      }

      // Validate transaction type
      const validTypes = ['REWARD', 'PENALTY', 'BONUS', 'REFERRAL', 'ACHIEVEMENT'];
      if (type && !validTypes.includes(type.toUpperCase())) {
        errors.push(`Invalid transaction type. Valid types: ${validTypes.join(', ')}`);
      }

      // Validate date formats
      if (startDate && !validator.isISO8601(startDate)) {
        errors.push('Start date must be in ISO 8601 format');
      }
      if (endDate && !validator.isISO8601(endDate)) {
        errors.push('End date must be in ISO 8601 format');
      }

      // Validate date range
      if (startDate && endDate) {
        const start = new Date(startDate);
        const end = new Date(endDate);
        if (start > end) {
          errors.push('Start date must be before end date');
        }
        // Prevent queries for more than 1 year of data
        const oneYearMs = 365 * 24 * 60 * 60 * 1000;
        if (end - start > oneYearMs) {
          errors.push('Date range cannot exceed 1 year');
        }
      }

      if (errors.length > 0) {
        logger.warn('Validation errors in getRewardTransactions', {
          userId,
          errors,
          query: req.query,
          ip: req.ip
        });
        return res.status(400).json({ 
          success: false, 
          message: 'Validation failed',
          errors,
          error: 'VALIDATION_ERROR'
        });
      }

      // Sanitize and prepare options
      const options = {
        page: pageNum,
        limit: limitNum,
        type: type ? validator.escape(type.toUpperCase()) : undefined,
        startDate: startDate ? validator.toDate(startDate) : undefined,
        endDate: endDate ? validator.toDate(endDate) : undefined
      };

      const result = await rewardsService.getRewardTransactions(userId, options);
      
      logger.info('Reward transactions retrieved successfully', { 
        userId,
        page: pageNum,
        limit: limitNum,
        type: options.type,
        dateRange: startDate && endDate ? `${startDate} to ${endDate}` : 'all',
        resultCount: result?.data?.length || 0,
        ip: req.ip,
        timestamp: new Date().toISOString()
      });
      
      res.json({
        success: true,
        data: result,
        pagination: {
          page: pageNum,
          limit: limitNum,
          total: result?.total || 0
        },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error('Error in getRewardTransactions:', { 
        error: error.message,
        stack: error.stack,
        userId: req.user?.supabaseId,
        query: req.query,
        ip: req.ip
      });
      
      res.status(500).json({
        success: false,
        message: 'Failed to fetch reward transactions',
        error: 'INTERNAL_SERVER_ERROR',
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Get a rewards report (admin only)
   * @route GET /api/rewards/admin/report
   * @param {Object} req.query - Query filters (optional)
   * @returns {Object} 200 - Rewards report data
   * @returns {Object} 401 - Unauthorized
   * @returns {Object} 500 - Internal server error
   */
  async getRewardsReport(req, res) {
    try {
      // Enhanced authentication and authorization validation
      const userId = req.user?.supabaseId;
      const userRole = req.user?.role;
      
      if (!userId || !validator.isUUID(userId)) {
        logger.warn('Invalid authentication attempt in getRewardsReport', {
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          userId: userId ? 'present_but_invalid' : 'missing'
        });
        return res.status(401).json({ 
          success: false, 
          message: 'Valid authentication required.',
          error: 'AUTHENTICATION_REQUIRED'
        });
      }

      // Authorization: Only admins allowed
      if (!userRole || !['ADMIN', 'SUPER_ADMIN'].includes(userRole.toUpperCase())) {
        logger.warn('Unauthorized admin access attempt in getRewardsReport', {
          userId,
          role: userRole,
          ip: req.ip,
          userAgent: req.get('User-Agent')
        });
        return res.status(403).json({ 
          success: false, 
          message: 'Admin access required.',
          error: 'INSUFFICIENT_PRIVILEGES'
        });
      }

      // Validate and sanitize query filters
      const filters = {};
      if (req.query.startDate && validator.isISO8601(req.query.startDate)) {
        filters.startDate = validator.toDate(req.query.startDate);
      }
      if (req.query.endDate && validator.isISO8601(req.query.endDate)) {
        filters.endDate = validator.toDate(req.query.endDate);
      }
      if (req.query.status && typeof req.query.status === 'string') {
        filters.status = validator.escape(req.query.status.toUpperCase());
      }
      if (req.query.type && typeof req.query.type === 'string') {
        filters.type = validator.escape(req.query.type.toUpperCase());
      }

      const report = await rewardsService.getRewardsReport(filters);
      
      logger.info('Rewards report generated successfully', { 
        adminId: userId,
        adminRole: userRole,
        filters,
        reportSize: report?.totalRecords || 0,
        ip: req.ip,
        timestamp: new Date().toISOString()
      });
      
      res.json({
        success: true,
        data: report,
        filters: filters,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error('Error in getRewardsReport:', { 
        error: error.message,
        stack: error.stack,
        adminId: req.user?.supabaseId,
        filters: req.query,
        ip: req.ip
      });
      
      res.status(500).json({
        success: false,
        message: 'Failed to generate rewards report',
        error: 'INTERNAL_SERVER_ERROR',
        timestamp: new Date().toISOString()
      });
    }
  }

  /**
   * Update the status of a reward (admin only)
   * @route PUT /api/rewards/admin/:id/status
   * @param {string} req.params.id - Reward ID
   * @param {string} req.body.status - New status (e.g., approved, rejected)
   * @returns {Object} 200 - Success message and updated reward
   * @returns {Object} 400 - Validation error
   * @returns {Object} 401 - Unauthorized
   * @returns {Object} 404 - Reward not found
   * @returns {Object} 500 - Internal server error
   */
  async updateRewardStatus(req, res) {
    try {
      // Enhanced authentication and authorization validation
      const userId = req.user?.supabaseId;
      const userRole = req.user?.role;
      
      if (!userId || !validator.isUUID(userId)) {
        logger.warn('Invalid authentication attempt in updateRewardStatus', {
          ip: req.ip,
          userAgent: req.get('User-Agent'),
          userId: userId ? 'present_but_invalid' : 'missing'
        });
        return res.status(401).json({ 
          success: false, 
          message: 'Valid authentication required.',
          error: 'AUTHENTICATION_REQUIRED'
        });
      }

      // Authorization: Only admins allowed
      if (!userRole || !['ADMIN', 'SUPER_ADMIN'].includes(userRole.toUpperCase())) {
        logger.warn('Unauthorized admin access attempt in updateRewardStatus', {
          userId,
          role: userRole,
          ip: req.ip,
          userAgent: req.get('User-Agent')
        });
        return res.status(403).json({ 
          success: false, 
          message: 'Admin access required.',
          error: 'INSUFFICIENT_PRIVILEGES'
        });
      }

      // Comprehensive input validation
      const { id } = req.params;
      const { status } = req.body;
      const errors = [];

      // Validate reward ID
      if (!id || !validator.isMongoId(id)) {
        errors.push('Valid reward ID is required');
      }

      // Validate status
      const validStatuses = ['PENDING', 'APPROVED', 'REJECTED', 'CANCELLED'];
      if (!status || typeof status !== 'string' || !status.trim()) {
        errors.push('Status is required');
      } else if (!validStatuses.includes(status.toUpperCase())) {
        errors.push(`Invalid status. Valid statuses: ${validStatuses.join(', ')}`);
      }

      if (errors.length > 0) {
        logger.warn('Validation errors in updateRewardStatus', {
          adminId: userId,
          errors,
          rewardId: id,
          status,
          ip: req.ip
        });
        return res.status(400).json({ 
          success: false, 
          message: 'Validation failed',
          errors,
          error: 'VALIDATION_ERROR'
        });
      }

      const sanitizedStatus = validator.escape(status.toUpperCase());
      const updated = await rewardsService.updateRewardStatus(id, sanitizedStatus);
      
      if (!updated) {
        logger.warn('Reward not found for status update', {
          adminId: userId,
          rewardId: id,
          status: sanitizedStatus,
          ip: req.ip
        });
        return res.status(404).json({ 
          success: false, 
          message: 'Reward not found.',
          error: 'REWARD_NOT_FOUND'
        });
      }
      
      logger.info('Reward status updated successfully', { 
        adminId: userId,
        adminRole: userRole,
        rewardId: id,
        oldStatus: updated.previousStatus,
        newStatus: sanitizedStatus,
        ip: req.ip,
        timestamp: new Date().toISOString()
      });
      
      res.json({
        success: true,
        message: 'Reward status updated successfully',
        data: {
          id: updated.id,
          status: sanitizedStatus,
          updatedAt: updated.updatedAt,
          updatedBy: userId
        },
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      logger.error('Error in updateRewardStatus:', { 
        error: error.message,
        stack: error.stack,
        adminId: req.user?.supabaseId,
        rewardId: req.params.id,
        status: req.body.status,
        ip: req.ip
      });
      
      res.status(500).json({
        success: false,
        message: 'Failed to update reward status',
        error: 'INTERNAL_SERVER_ERROR',
        timestamp: new Date().toISOString()
      });
    }
  }
}

module.exports = new RewardsController();
