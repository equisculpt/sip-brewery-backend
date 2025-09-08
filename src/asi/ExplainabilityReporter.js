// Explainability Reporter: Generates natural-language and structured reports for every prediction/action
const fs = require('fs');
const path = require('path');
const REPORT_LOG = path.join(__dirname, 'explainability_log.jsonl');

function generateExplainabilityReport({ id, type, input, output, model, rationale, compliance, timestamp }) {
  const report = {
    id,
    type,
    input,
    output,
    model,
    rationale,
    compliance,
    timestamp: timestamp || Date.now()
  };
  fs.appendFileSync(REPORT_LOG, JSON.stringify(report) + '\n');
  return report;
}

function getExplainabilityReports(filter = {}) {
  if (!fs.existsSync(REPORT_LOG)) return [];
  return fs.readFileSync(REPORT_LOG, 'utf-8')
    .split('\n')
    .filter(Boolean)
    .map(line => JSON.parse(line))
    .filter(rep => {
      for (const key in filter) {
        if (rep[key] !== filter[key]) return false;
      }
      return true;
    });
}

module.exports = { generateExplainabilityReport, getExplainabilityReports };
