const jwt = require('jsonwebtoken');
const Admin = require('../models/Admin');
const Agent = require('../models/Agent');
const AuditLog = require('../models/AuditLog');
const response = require('../utils/response');
const logger = require('../utils/logger');

class AdminAuthMiddleware {
  /**
   * Verify JWT token and attach admin to request
   */
  async verifyToken(req, res, next) {
    try {
      const token = this.extractToken(req);
      
      if (!token) {
        return response.error(res, 'Access token required', 401);
      }

      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      
      // Check if admin exists and is active
      const admin = await Admin.findById(decoded.userId).select('+password');
      
      if (!admin || !admin.isActive) {
        return response.error(res, 'Invalid or inactive account', 401);
      }

      // Check if account is locked
      if (admin.isLocked()) {
        return response.error(res, 'Account temporarily locked due to multiple failed attempts', 423);
      }

      // Check IP whitelist if configured
      if (admin.ipWhitelist && admin.ipWhitelist.length > 0) {
        const clientIP = this.getClientIP(req);
        const isWhitelisted = admin.ipWhitelist.some(ip => ip.ip === clientIP);
        
        if (!isWhitelisted) {
          await this.logSecurityEvent(admin._id, 'IP_NOT_WHITELISTED', {
            ip: clientIP,
            userAgent: req.get('User-Agent')
          });
          return response.error(res, 'Access denied from this IP address', 403);
        }
      }

      // Update last activity
      admin.lastActivity = new Date();
      await admin.save();

      // Attach admin to request
      req.admin = admin;
      req.userId = admin._id;
      req.userRole = admin.role;

      next();
    } catch (error) {
      if (error.name === 'JsonWebTokenError') {
        return response.error(res, 'Invalid token', 401);
      }
      if (error.name === 'TokenExpiredError') {
        return response.error(res, 'Token expired', 401);
      }
      
      logger.error('Admin auth error:', error);
      return response.error(res, 'Authentication failed', 500);
    }
  }

  /**
   * Role-based access control
   */
  requireRole(roles) {
    return (req, res, next) => {
      if (!req.admin) {
        return response.error(res, 'Authentication required', 401);
      }

      const allowedRoles = Array.isArray(roles) ? roles : [roles];
      
      if (!allowedRoles.includes(req.admin.role)) {
        return response.error(res, 'Insufficient permissions', 403);
      }

      next();
    };
  }

  /**
   * Permission-based access control
   */
  requirePermission(module, action) {
    return (req, res, next) => {
      if (!req.admin) {
        return response.error(res, 'Authentication required', 401);
      }

      if (!req.admin.hasPermission(module, action)) {
        return response.error(res, 'Insufficient permissions', 403);
      }

      next();
    };
  }

  /**
   * Agent-specific access control
   */
  async requireAgentAccess(req, res, next) {
    try {
      if (!req.admin) {
        return response.error(res, 'Authentication required', 401);
      }

      // Super admins and admins can access all agents
      if (req.admin.role === 'SUPER_ADMIN' || req.admin.role === 'ADMIN') {
        return next();
      }

      // Agents can only access their own data
      if (req.admin.role === 'AGENT') {
        const agent = await Agent.findOne({ adminId: req.admin._id });
        if (!agent) {
          return response.error(res, 'Agent profile not found', 404);
        }
        
        req.agent = agent;
        return next();
      }

      return response.error(res, 'Insufficient permissions', 403);
    } catch (error) {
      logger.error('Agent access check error:', error);
      return response.error(res, 'Access check failed', 500);
    }
  }

  /**
   * Regional access control for admins
   */
  requireRegionalAccess(req, res, next) {
    if (!req.admin) {
      return response.error(res, 'Authentication required', 401);
    }

    // Super admins have access to all regions
    if (req.admin.role === 'SUPER_ADMIN') {
      return next();
    }

    // Admins have regional restrictions
    if (req.admin.role === 'ADMIN') {
      const requestedRegion = req.params.region || req.body.region;
      
      if (requestedRegion && !req.admin.regions.includes(requestedRegion)) {
        return response.error(res, 'Access denied to this region', 403);
      }
    }

    next();
  }

  /**
   * Rate limiting for sensitive operations
   */
  rateLimit(options = {}) {
    const {
      windowMs = 15 * 60 * 1000, // 15 minutes
      max = 100, // limit each IP to 100 requests per windowMs
      message = 'Too many requests from this IP'
    } = options;

    const requests = new Map();

    return (req, res, next) => {
      const key = this.getClientIP(req);
      const now = Date.now();
      const windowStart = now - windowMs;

      // Clean old entries
      if (requests.has(key)) {
        requests.set(key, requests.get(key).filter(timestamp => timestamp > windowStart));
      }

      const currentRequests = requests.get(key) || [];
      
      if (currentRequests.length >= max) {
        return response.error(res, message, 429);
      }

      currentRequests.push(now);
      requests.set(key, currentRequests);

      next();
    };
  }

  /**
   * Audit logging middleware
   */
  auditLog(action, module) {
    return async (req, res, next) => {
      const originalSend = res.send;
      const startTime = Date.now();

      res.send = function(data) {
        const responseTime = Date.now() - startTime;
        
        // Log the action asynchronously
        AuditLog.logAction({
          userId: req.admin?._id,
          userEmail: req.admin?.email,
          userRole: req.admin?.role,
          userAgentCode: req.admin?.agentCode,
          action,
          module,
          method: req.method,
          endpoint: req.originalUrl,
          ipAddress: this.getClientIP(req),
          userAgent: req.get('User-Agent'),
          status: res.statusCode < 400 ? 'success' : 'failure',
          responseTime,
          oldData: req.method === 'PUT' || req.method === 'PATCH' ? req.body : null,
          newData: req.method === 'POST' || req.method === 'PUT' || req.method === 'PATCH' ? req.body : null,
          metadata: {
            params: req.params,
            query: req.query
          }
        }).catch(err => {
          logger.error('Audit log error:', err);
        });

        originalSend.call(this, data);
      };

      next();
    };
  }

  /**
   * Extract token from request
   */
  extractToken(req) {
    if (req.headers.authorization && req.headers.authorization.startsWith('Bearer ')) {
      return req.headers.authorization.substring(7);
    }
    
    if (req.cookies && req.cookies.adminToken) {
      return req.cookies.adminToken;
    }
    
    return null;
  }

  /**
   * Get client IP address
   */
  getClientIP(req) {
    return req.ip || 
           req.connection.remoteAddress || 
           req.socket.remoteAddress ||
           (req.connection.socket ? req.connection.socket.remoteAddress : null) ||
           req.headers['x-forwarded-for'] ||
           req.headers['x-real-ip'];
  }

  /**
   * Log security events
   */
  async logSecurityEvent(userId, event, metadata = {}) {
    try {
      await AuditLog.logAction({
        userId,
        action: event,
        module: 'security',
        ipAddress: 'N/A',
        status: 'failure',
        severity: 'high',
        isSuspicious: true,
        metadata
      });
    } catch (error) {
      logger.error('Security event logging error:', error);
    }
  }

  /**
   * Check if admin has access to specific client
   */
  async requireClientAccess(req, res, next) {
    try {
      if (!req.admin) {
        return response.error(res, 'Authentication required', 401);
      }

      const clientId = req.params.clientId || req.body.clientId;
      
      if (!clientId) {
        return response.error(res, 'Client ID required', 400);
      }

      // Super admins and admins can access all clients
      if (req.admin.role === 'SUPER_ADMIN' || req.admin.role === 'ADMIN') {
        return next();
      }

      // Agents can only access their assigned clients
      if (req.admin.role === 'AGENT') {
        const hasAccess = req.admin.assignedClients.includes(clientId);
        
        if (!hasAccess) {
          return response.error(res, 'Access denied to this client', 403);
        }
      }

      next();
    } catch (error) {
      logger.error('Client access check error:', error);
      return response.error(res, 'Access check failed', 500);
    }
  }
}

// Create middleware instance
const adminAuth = new AdminAuthMiddleware();

// Export middleware functions
module.exports = {
  verifyToken: adminAuth.verifyToken.bind(adminAuth),
  requireRole: adminAuth.requireRole.bind(adminAuth),
  requirePermission: adminAuth.requirePermission.bind(adminAuth),
  requireAgentAccess: adminAuth.requireAgentAccess.bind(adminAuth),
  requireRegionalAccess: adminAuth.requireRegionalAccess.bind(adminAuth),
  requireClientAccess: adminAuth.requireClientAccess.bind(adminAuth),
  rateLimit: adminAuth.rateLimit.bind(adminAuth),
  auditLog: adminAuth.auditLog.bind(adminAuth)
}; 