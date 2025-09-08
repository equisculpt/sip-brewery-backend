// Audit logger utility for sensitive/admin actions
const logger = require('./logger');

function auditLog(action, user, details = {}) {
  const entry = {
    timestamp: new Date().toISOString(),
    user: user ? { id: user.id, role: user.role } : null,
    action,
    details,
    ip: details.ip || null
  };
  logger.info('[AUDIT]', JSON.stringify(entry));
}

module.exports = { auditLog };
