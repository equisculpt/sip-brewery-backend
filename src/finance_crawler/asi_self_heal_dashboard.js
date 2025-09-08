// ASI Self-Heal Dashboard and Notification Integration
const fs = require('fs');
const path = require('path');
const { processRequest } = require('../../asi/ASIMasterEngine');
const axios = require('axios');

const HEAL_LOG_PATH = path.join(__dirname, 'asi_self_heal.log');
const DASHBOARD_API = process.env.DASHBOARD_API || '';

// Review log, summarize, and push notifications
function reviewSelfHealLog() {
  if (!fs.existsSync(HEAL_LOG_PATH)) {
    console.log('No asi_self_heal.log found.');
    return [];
  }
  const log = fs.readFileSync(HEAL_LOG_PATH, 'utf-8');
  const entries = log.trim().split('\n').map(line => {
    const [timestamp, rest] = line.split('] ');
    const [site, rest2] = rest.split(': ');
    const [error, healInfo] = rest2.split(' | ASI Self-Heal: ');
    return {
      time: timestamp.replace('[',''),
      site,
      error,
      healInfo: healInfo || ''
    };
  });
  return entries;
}

async function sendDashboardNotification(entry) {
  if (!DASHBOARD_API) return;
  return axios.post(DASHBOARD_API, { type: 'asi_self_heal', entry });
}

async function notifyForNewHeals() {
  const entries = reviewSelfHealLog();
  if (entries.length === 0) return;
  // Only notify for the latest entry
  const latest = entries[entries.length - 1];
  await sendDashboardNotification(latest);
  // Optionally, trigger further ASI review or improvement
  await processRequest({ type: 'self_heal_review', entry: latest });
  console.log('Dashboard/ASI notified for self-heal:', latest);
}

// Watch asi_self_heal.log for new entries
fs.watchFile(HEAL_LOG_PATH, { interval: 10000 }, async (curr, prev) => {
  if (curr.size > prev.size) {
    await notifyForNewHeals();
  }
});

console.log('ASI Self-Heal Dashboard integration started. Watching for self-heal outcomes...');
