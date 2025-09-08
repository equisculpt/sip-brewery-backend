// Compliance Middleware for SEBI/AMFI
const complianceRules = require('../compliance/rules');

function complianceCheck(actionContext) {
  // Check all rules, return { compliant: true/false, violations: [] }
  const violations = [];
  for (const rule of complianceRules) {
    const result = rule(actionContext);
    if (!result.compliant) violations.push(result.reason);
  }
  return { compliant: violations.length === 0, violations };
}

const { logComplianceAudit } = require('../asi/ComplianceAuditDashboard');

function complianceMiddleware(req, res, next) {
  const actionContext = {
    path: req.path,
    method: req.method,
    user: req.user,
    body: req.body,
    headers: req.headers
  };
  const check = complianceCheck(actionContext);
  logComplianceAudit({
    id: Date.now() + '-' + Math.random(),
    actionType: 'api_call',
    user: req.user,
    path: req.path,
    method: req.method,
    result: check.compliant ? 'pass' : 'fail',
    violations: check.violations,
    timestamp: Date.now()
  });
  if (!check.compliant) {
    res.status(403).json({ success: false, error: 'Compliance violation', violations: check.violations });
    return;
  }
  next();
}

module.exports = { complianceCheck, complianceMiddleware };
