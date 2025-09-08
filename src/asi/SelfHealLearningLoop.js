// ASI Self-Heal Learning Loop: Analyze self-heal outcomes and update heuristics
const fs = require('fs');
const path = require('path');
const ASIMasterEngine = require('./ASIMasterEngine');
const LOG_PATH = path.join(__dirname, '../finance_crawler/asi_self_heal.log');

async function selfHealLearningLoop() {
  if (!fs.existsSync(LOG_PATH)) return;
  const log = fs.readFileSync(LOG_PATH, 'utf-8').split('\n').filter(Boolean);
  for (const line of log.slice(-100)) { // Analyze last 100 events
    const match = line.match(/ASI Self-Heal: (\{.*?\})(?: |$)/);
    if (match) {
      const result = JSON.parse(match[1]);
      await ASIMasterEngine.updateHeuristicsFromSelfHeal(result);
    }
  }
}

module.exports = { selfHealLearningLoop };
