const Admin = require('../models/Admin');
const Agent = require('../models/Agent');
const User = require('../models/User');
const UserPortfolio = require('../models/UserPortfolio');
const Commission = require('../models/Commission');
const AuditLog = require('../models/AuditLog');
const Notification = require('../models/Notification');
const response = require('../utils/response');
const logger = require('../utils/logger');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

class AdminController {
  /**
   * Admin Login
   */
  async login(req, res) {
    try {
      const { email, password } = req.body;

      if (!email || !password) {
        return response.error(res, 'Email and password are required', 400);
      }

      // Find admin by email
      const admin = await Admin.findOne({ email: email.toLowerCase() }).select('+password');
      
      if (!admin) {
        return response.error(res, 'Invalid credentials', 401);
      }

      // Check if account is active
      if (!admin.isActive) {
        return response.error(res, 'Account is deactivated', 401);
      }

      // Check if account is locked
      if (admin.isLocked()) {
        return response.error(res, 'Account temporarily locked due to multiple failed attempts', 423);
      }

      // Verify password
      const isPasswordValid = await admin.verifyPassword(password);
      
      if (!isPasswordValid) {
        await admin.incLoginAttempts();
        return response.error(res, 'Invalid credentials', 401);
      }

      // Reset login attempts on successful login
      await admin.resetLoginAttempts();

      // Generate JWT token
      const token = jwt.sign(
        { 
          userId: admin._id, 
          role: admin.role,
          email: admin.email 
        },
        process.env.JWT_SECRET,
        { expiresIn: '24h' }
      );

      // Update last login
      admin.lastLogin = new Date();
      await admin.save();

      // Log successful login
      await AuditLog.logAction({
        userId: admin._id,
        userEmail: admin.email,
        userRole: admin.role,
        action: 'LOGIN_SUCCESS',
        module: 'auth',
        method: req.method,
        endpoint: req.originalUrl,
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
        status: 'success'
      });

      // Return admin data (without password)
      const adminData = admin.toObject();
      delete adminData.password;

      return response.success(res, {
        admin: adminData,
        token
      }, 'Login successful');

    } catch (error) {
      logger.error('Admin login error:', error);
      return response.error(res, 'Login failed', 500, error);
    }
  }

  /**
   * Get Admin Dashboard
   */
  async getDashboard(req, res) {
    try {
      const admin = req.admin;

      // Get basic stats
      const stats = await this.getDashboardStats(admin);

      // Get recent activities
      const recentActivities = await this.getRecentActivities(admin);

      // Get notifications
      const notifications = await this.getNotifications(admin._id);

      // Get performance metrics
      const performanceMetrics = await this.getPerformanceMetrics(admin);

      return response.success(res, {
        stats,
        recentActivities,
        notifications,
        performanceMetrics
      }, 'Dashboard data retrieved successfully');

    } catch (error) {
      logger.error('Dashboard error:', error);
      return response.error(res, 'Failed to load dashboard', 500, error);
    }
  }

  /**
   * Get Dashboard Statistics
   */
  async getDashboardStats(admin) {
    try {
      let stats = {};

      if (admin.role === 'SUPER_ADMIN') {
        // Super admin sees all platform stats
        stats = await this.getPlatformStats();
      } else if (admin.role === 'ADMIN') {
        // Admin sees regional stats
        stats = await this.getRegionalStats(admin.regions);
      } else if (admin.role === 'AGENT') {
        // Agent sees personal stats
        stats = await this.getAgentStats(admin._id);
      }

      return stats;
    } catch (error) {
      logger.error('Dashboard stats error:', error);
      return {};
    }
  }

  /**
   * Get Platform Statistics (Super Admin)
   */
  async getPlatformStats() {
    try {
      const [
        totalUsers,
        totalAgents,
        totalAUM,
        monthlySIP,
        totalCommission,
        activeUsers
      ] = await Promise.all([
        User.countDocuments({ isActive: true }),
        Agent.countDocuments({ status: 'active' }),
        UserPortfolio.aggregate([
          { $match: { isActive: true } },
          { $group: { _id: null, total: { $sum: '$totalCurrentValue' } } }
        ]),
        UserPortfolio.aggregate([
          { $match: { isActive: true } },
          { $group: { _id: null, total: { $sum: '$monthlySIPAmount' } } }
        ]),
        Commission.aggregate([
          { $match: { isActive: true } },
          { $group: { _id: null, total: { $sum: '$commissionAmount' } } }
        ]),
        User.countDocuments({
          lastLogin: { $gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) }
        })
      ]);

      return {
        totalUsers: totalUsers,
        totalAgents: totalAgents,
        totalAUM: totalAUM[0]?.total || 0,
        monthlySIP: monthlySIP[0]?.total || 0,
        totalCommission: totalCommission[0]?.total || 0,
        activeUsers: activeUsers
      };
    } catch (error) {
      logger.error('Platform stats error:', error);
      return {};
    }
  }

  /**
   * Get Regional Statistics (Admin)
   */
  async getRegionalStats(regions) {
    try {
      const stats = await Agent.aggregate([
        {
          $match: {
            status: 'active',
            region: { $in: regions }
          }
        },
        {
          $group: {
            _id: '$region',
            totalAgents: { $sum: 1 },
            totalAUM: { $sum: '$totalAUM' },
            totalEarnings: { $sum: '$totalEarnings' }
          }
        }
      ]);

      return {
        regionalStats: stats,
        totalRegions: regions.length
      };
    } catch (error) {
      logger.error('Regional stats error:', error);
      return {};
    }
  }

  /**
   * Get Agent Statistics
   */
  async getAgentStats(adminId) {
    try {
      const agent = await Agent.findOne({ adminId });
      
      if (!agent) {
        return {};
      }

      const [
        totalClients,
        totalAUM,
        monthlyEarnings,
        pendingPayout
      ] = await Promise.all([
        User.countDocuments({ assignedAgent: agent._id, isActive: true }),
        UserPortfolio.aggregate([
          { $match: { userId: { $in: agent.assignedClients || [] } } },
          { $group: { _id: null, total: { $sum: '$totalCurrentValue' } } }
        ]),
        Commission.aggregate([
          {
            $match: {
              agentId: agent._id,
              isActive: true,
              transactionDate: {
                $gte: new Date(new Date().getFullYear(), new Date().getMonth(), 1)
              }
            }
          },
          { $group: { _id: null, total: { $sum: '$commissionAmount' } } }
        ]),
        Commission.aggregate([
          {
            $match: {
              agentId: agent._id,
              payoutStatus: 'APPROVED',
              isActive: true
            }
          },
          { $group: { _id: null, total: { $sum: '$commissionAmount' } } }
        ])
      ]);

      return {
        totalClients: totalClients,
        totalAUM: totalAUM[0]?.total || 0,
        monthlyEarnings: monthlyEarnings[0]?.total || 0,
        pendingPayout: pendingPayout[0]?.total || 0,
        agentCode: agent.agentCode,
        region: agent.region
      };
    } catch (error) {
      logger.error('Agent stats error:', error);
      return {};
    }
  }

  /**
   * Get Recent Activities
   */
  async getRecentActivities(admin) {
    try {
      let query = {};

      // Filter by admin role
      if (admin.role === 'AGENT') {
        query.userId = admin._id;
      } else if (admin.role === 'ADMIN') {
        // Admin sees activities from their region
        const agentIds = await Agent.find({ region: { $in: admin.regions } }).distinct('adminId');
        query.userId = { $in: agentIds };
      }

      const activities = await AuditLog.find(query)
        .sort({ timestamp: -1 })
        .limit(10)
        .populate('userId', 'name email role');

      return activities;
    } catch (error) {
      logger.error('Recent activities error:', error);
      return [];
    }
  }

  /**
   * Get Notifications
   */
  async getNotifications(adminId) {
    try {
      const notifications = await Notification.getUserNotifications(adminId, {
        limit: 10,
        unreadOnly: false
      });

      return notifications;
    } catch (error) {
      logger.error('Notifications error:', error);
      return [];
    }
  }

  /**
   * Get Performance Metrics
   */
  async getPerformanceMetrics(admin) {
    try {
      let metrics = {};

      if (admin.role === 'SUPER_ADMIN') {
        // Platform-wide metrics
        metrics = await this.getPlatformMetrics();
      } else if (admin.role === 'ADMIN') {
        // Regional metrics
        metrics = await this.getRegionalMetrics(admin.regions);
      } else if (admin.role === 'AGENT') {
        // Personal metrics
        metrics = await this.getPersonalMetrics(admin._id);
      }

      return metrics;
    } catch (error) {
      logger.error('Performance metrics error:', error);
      return {};
    }
  }

  /**
   * Get Platform Metrics
   */
  async getPlatformMetrics() {
    try {
      const [
        weeklyGrowth,
        monthlyGrowth,
        topSchemes,
        redemptionRatio
      ] = await Promise.all([
        this.calculateGrowth('weekly'),
        this.calculateGrowth('monthly'),
        this.getTopSchemes(),
        this.getRedemptionRatio()
      ]);

      return {
        weeklyGrowth,
        monthlyGrowth,
        topSchemes,
        redemptionRatio
      };
    } catch (error) {
      logger.error('Platform metrics error:', error);
      return {};
    }
  }

  /**
   * Calculate Growth
   */
  async calculateGrowth(period) {
    try {
      const now = new Date();
      let startDate;

      if (period === 'weekly') {
        startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
      } else if (period === 'monthly') {
        startDate = new Date(now.getFullYear(), now.getMonth() - 1, now.getDate());
      }

      const currentAUM = await UserPortfolio.aggregate([
        { $match: { isActive: true } },
        { $group: { _id: null, total: { $sum: '$totalCurrentValue' } } }
      ]);

      // This is a simplified calculation - in production, you'd compare with historical data
      return {
        period,
        growth: 0, // Placeholder
        currentValue: currentAUM[0]?.total || 0
      };
    } catch (error) {
      logger.error('Growth calculation error:', error);
      return { period, growth: 0, currentValue: 0 };
    }
  }

  /**
   * Get Top Schemes
   */
  async getTopSchemes() {
    try {
      const topSchemes = await UserPortfolio.aggregate([
        { $match: { isActive: true } },
        { $unwind: '$funds' },
        {
          $group: {
            _id: '$funds.schemeName',
            totalAUM: { $sum: '$funds.currentValue' },
            clientCount: { $addToSet: '$userId' }
          }
        },
        {
          $project: {
            schemeName: '$_id',
            totalAUM: 1,
            clientCount: { $size: '$clientCount' }
          }
        },
        { $sort: { totalAUM: -1 } },
        { $limit: 5 }
      ]);

      return topSchemes;
    } catch (error) {
      logger.error('Top schemes error:', error);
      return [];
    }
  }

  /**
   * Get Redemption Ratio
   */
  async getRedemptionRatio() {
    try {
      // This would require transaction data - simplified for now
      return {
        ratio: 0.15, // Placeholder
        totalRedemptions: 0,
        totalSIPs: 0
      };
    } catch (error) {
      logger.error('Redemption ratio error:', error);
      return { ratio: 0, totalRedemptions: 0, totalSIPs: 0 };
    }
  }

  /**
   * Get Regional Metrics
   */
  async getRegionalMetrics(regions) {
    try {
      const metrics = await Agent.aggregate([
        {
          $match: {
            region: { $in: regions },
            status: 'active'
          }
        },
        {
          $group: {
            _id: '$region',
            totalAgents: { $sum: 1 },
            totalAUM: { $sum: '$totalAUM' },
            avgXIRR: { $avg: '$avgClientXIRR' }
          }
        }
      ]);

      return { regionalMetrics: metrics };
    } catch (error) {
      logger.error('Regional metrics error:', error);
      return { regionalMetrics: [] };
    }
  }

  /**
   * Get Personal Metrics
   */
  async getPersonalMetrics(adminId) {
    try {
      const agent = await Agent.findOne({ adminId });
      
      if (!agent) {
        return {};
      }

      return {
        targetAchievement: agent.targetAchievement,
        clientGrowth: 0, // Placeholder
        aumGrowth: 0, // Placeholder
        commissionGrowth: 0 // Placeholder
      };
    } catch (error) {
      logger.error('Personal metrics error:', error);
      return {};
    }
  }

  /**
   * Logout
   */
  async logout(req, res) {
    try {
      const admin = req.admin;

      // Log logout
      await AuditLog.logAction({
        userId: admin._id,
        userEmail: admin.email,
        userRole: admin.role,
        action: 'LOGOUT',
        module: 'auth',
        method: req.method,
        endpoint: req.originalUrl,
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
        status: 'success'
      });

      return response.success(res, null, 'Logout successful');
    } catch (error) {
      logger.error('Logout error:', error);
      return response.error(res, 'Logout failed', 500, error);
    }
  }

  /**
   * Get Profile
   */
  async getProfile(req, res) {
    try {
      const admin = req.admin;
      
      // Remove sensitive data
      const profile = admin.toObject();
      delete profile.password;
      delete profile.twoFactorSecret;

      return response.success(res, profile, 'Profile retrieved successfully');
    } catch (error) {
      logger.error('Get profile error:', error);
      return response.error(res, 'Failed to get profile', 500, error);
    }
  }

  /**
   * Update Profile
   */
  async updateProfile(req, res) {
    try {
      const admin = req.admin;
      const { name, phone, preferences } = req.body;

      // Update allowed fields
      if (name) admin.name = name;
      if (phone) admin.phone = phone;
      if (preferences) admin.preferences = { ...admin.preferences, ...preferences };

      await admin.save();

      // Log profile update
      await AuditLog.logAction({
        userId: admin._id,
        userEmail: admin.email,
        userRole: admin.role,
        action: 'UPDATE_PROFILE',
        module: 'admin',
        method: req.method,
        endpoint: req.originalUrl,
        ipAddress: req.ip,
        userAgent: req.get('User-Agent'),
        status: 'success',
        newData: { name, phone, preferences }
      });

      const profile = admin.toObject();
      delete profile.password;
      delete profile.twoFactorSecret;

      return response.success(res, profile, 'Profile updated successfully');
    } catch (error) {
      logger.error('Update profile error:', error);
      return response.error(res, 'Failed to update profile', 500, error);
    }
  }
}

module.exports = new AdminController(); 