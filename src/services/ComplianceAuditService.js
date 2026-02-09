const AuditLog = require('../models/AuditLog');

class ComplianceAuditService {
  async recordAction({
    userId,
    userEmail,
    userRole,
    action,
    module,
    ipAddress,
    method,
    endpoint,
    resourceType,
    resourceId,
    oldData,
    newData,
    status = 'success',
    severity = 'low',
    metadata = {}
  }) {
    return AuditLog.logAction({
      userId,
      userEmail,
      userRole,
      action,
      module,
      ipAddress,
      method,
      endpoint,
      resourceType,
      resourceId,
      oldData,
      newData,
      status,
      severity,
      metadata,
      timestamp: new Date()
    });
  }
}

module.exports = ComplianceAuditService;
