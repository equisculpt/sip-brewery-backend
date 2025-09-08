// Error alert integration for Slack/email/ASI dashboard
const fs = require('fs');
const path = require('path');
const { processRequest } = require('../../asi/ASIMasterEngine');
const axios = require('axios');

const SLACK_WEBHOOK_URL = process.env.SLACK_WEBHOOK_URL || '';
const EMAIL_ALERT_API = process.env.EMAIL_ALERT_API || '';
const ASI_ALERT_API = process.env.ASI_ALERT_API || '';
const LOG_PATH = path.join(__dirname, 'crawl_errors.log');

function sendSlackAlert(message) {
  if (!SLACK_WEBHOOK_URL) return;
  return axios.post(SLACK_WEBHOOK_URL, { text: message });
}

function sendEmailAlert(subject, message) {
  if (!EMAIL_ALERT_API) return;
  return axios.post(EMAIL_ALERT_API, { subject, message });
}

async function sendASIAlert(site, error) {
  if (!ASI_ALERT_API) {
    // Optionally, route to ASI Master Engine directly
    await processRequest({ type: 'error_alert', site, error });
    return;
  }
  return axios.post(ASI_ALERT_API, { site, error });
}

// ASI self-healing and autonomous debugging
async function triggerASISelfHeal(site, error) {
  const healLogPath = path.join(__dirname, 'asi_self_heal.log');
  const dashboard = require('./asi_self_heal_dashboard');
  try {
    const result = await processRequest({ type: 'self_heal', site, error });
    let patchStatus = '', reloadStatus = '';
    // Attempt auto-patching if ASI returns patch_code
    if (result && result.patch_code && result.parser_file) {
      try {
        fs.writeFileSync(result.parser_file, result.patch_code);
        patchStatus = 'Patch applied';
        // Hot-reload parser module
        delete require.cache[require.resolve(result.parser_file)];
        require(result.parser_file);
        reloadStatus = 'Hot-reload successful';
      } catch (patchErr) {
        patchStatus = 'Patch failed: ' + patchErr.message;
        reloadStatus = 'Hot-reload failed';
      }
    }
    const entry = `[${new Date().toISOString()}] ${site}: ${error} | ASI Self-Heal: ${JSON.stringify(result)} | Patch: ${patchStatus} | Reload: ${reloadStatus}\n`;
    fs.appendFileSync(healLogPath, entry);
    if (dashboard && dashboard.notifyForNewHeals) await dashboard.notifyForNewHeals();
    console.log('ASI self-healing triggered:', entry);
  } catch (e) {
    const entry = `[${new Date().toISOString()}] ${site}: ${error} | ASI Self-Heal FAILED: ${e.message}\n`;
    fs.appendFileSync(healLogPath, entry);
    console.error('ASI self-healing failed:', entry);
  }
}

// Watch crawl_errors.log and alert on new errors
fs.watchFile(LOG_PATH, { interval: 10000 }, async (curr, prev) => {
  if (curr.size > prev.size) {
    const log = fs.readFileSync(LOG_PATH, 'utf-8');
    const lines = log.trim().split('\n');
    const lastLine = lines[lines.length - 1];
    if (lastLine) {
      const [timestamp, rest] = lastLine.split('] ');
      const [site, error] = rest.split(': ');
      const message = `🚨 Crawler Error\nSite: ${site}\nError: ${error}\nTime: ${timestamp.replace('[','')}`;
      await sendSlackAlert(message);
      await sendEmailAlert('Crawler Error', message);
      await sendASIAlert(site, error);
      await triggerASISelfHeal(site, error);
      console.log('Error alert sent:', message);
    }
  }
});

console.log('Error alert integration started. Watching for crawl errors...');
