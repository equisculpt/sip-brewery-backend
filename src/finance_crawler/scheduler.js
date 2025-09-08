// Scheduler for orchestrator using node-cron
const cron = require('node-cron');
const { fork } = require('child_process');
const path = require('path');

function runOrchestrator() {
  const orchestratorPath = path.join(__dirname, 'orchestrator.js');
  const child = fork(orchestratorPath);
  child.on('exit', code => {
    if (code !== 0) {
      console.error('Orchestrator exited with code', code);
    }
  });
}

// Schedule: every hour at minute 0
cron.schedule('0 * * * *', runOrchestrator);
// Schedule: every 10 minutes
cron.schedule('*/10 * * * *', runOrchestrator);

console.log('Scheduler started: Orchestrator will run every hour and every 10 minutes.');
