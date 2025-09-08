// Compliance Audit Dashboard: Monitors and displays all compliance checks, violations, and actions
const fs = require('fs');
const path = require('path');
const COMPLIANCE_LOG = path.join(__dirname, '../middleware/compliance_audit_log.jsonl');

function logComplianceAudit({ id, actionType, user, path, method, result, violations, timestamp }) {
  const entry = {
    id,
    actionType,
    user,
    path,
    method,
    result,
    violations,
    timestamp: timestamp || Date.now()
  };
  fs.appendFileSync(COMPLIANCE_LOG, JSON.stringify(entry) + '\n');
  return entry;
}

function getComplianceAuditTrail(filter = {}) {
  if (!fs.existsSync(COMPLIANCE_LOG)) return [];
  return fs.readFileSync(COMPLIANCE_LOG, 'utf-8')
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line))
    .filter(entry => {
      for (const key in filter) {
        if (entry[key] !== filter[key]) return false;
      }
      return true;
    });
}

module.exports = { logComplianceAudit, getComplianceAuditTrail };
