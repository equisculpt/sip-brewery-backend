const messageParser = require('./src/utils/parseMessage');
const fs = require('fs');

// Pool of realistic user messages for all intents
const messagePool = [
  // Greetings
  'Hi', 'Hello', 'Hey', 'Good morning', 'Namaste',
  // Onboarding
  'My name is John Doe', 'email is john@example.com', 'PAN is ABCDE1234F',
  // Portfolio
  'Show my portfolio', 'What is my portfolio worth?', 'Holdings',
  // SIP
  'I want to invest ₹5000 in HDFC Flexicap', 'Start SIP ₹2000 in SBI Smallcap',
  'Invest ₹10000 in Parag Parikh Flexicap', '₹3000 in Mirae Asset Largecap',
  // SIP Stop
  'Stop SIP in HDFC Flexicap', 'Cancel my SIP',
  // Rewards
  'My rewards', 'Show cashback', 'Redeem points',
  // Referral
  'Refer a friend', 'Get my referral link',
  // Leaderboard
  'Leaderboard', 'Show top performers',
  // AI Analysis
  'Analyse HDFC Flexicap', 'Tell me about SBI Smallcap',
  // Statement
  'Send my statement', 'Download report',
  // Help
  'Help', 'What can you do?',
  // Edge cases
  'asdfghjkl', '', '12345', 'Invest', 'Show', 'Portfolio',
  'I want to invest in', 'My name is', 'email is', 'PAN is',
  // Rapid fire
  ...Array(20).fill('Hi'),
  ...Array(20).fill('Help'),
  ...Array(20).fill('Show my portfolio'),
  // Malformed
  'I want to invest $5000 in HDFC', 'Start SIP in', 'Invest in',
  'My email is notanemail', 'PAN is 1234',
];

// Generate a random conversation flow
function generateConversation() {
  const length = Math.floor(Math.random() * 8) + 3; // 3-10 messages
  const conversation = [];
  for (let i = 0; i < length; i++) {
    const msg = messagePool[Math.floor(Math.random() * messagePool.length)];
    conversation.push(msg);
  }
  return conversation;
}

const NUM_CONVERSATIONS = 10000;
const results = {
  total: 0,
  errors: 0,
  intentStats: {},
  extractionErrors: 0,
  emptyIntent: 0,
  performance: [],
  errorSamples: [],
};

console.log(`🚀 Starting massive WhatsApp chatbot test: ${NUM_CONVERSATIONS} conversations...`);

for (let c = 0; c < NUM_CONVERSATIONS; c++) {
  const conversation = generateConversation();
  let context = {};
  for (let m = 0; m < conversation.length; m++) {
    const message = conversation[m];
    const start = Date.now();
    try {
      const result = messageParser.parseMessage(message, context);
      const duration = Date.now() - start;
      results.performance.push(duration);
      results.total++;
      // Intent stats
      if (!result.intent) {
        results.emptyIntent++;
      } else {
        results.intentStats[result.intent] = (results.intentStats[result.intent] || 0) + 1;
      }
      // Extraction check for key intents
      if (["SIP_CREATE", "ONBOARDING", "AI_ANALYSIS"].includes(result.intent)) {
        if (!result.extractedData || Object.keys(result.extractedData).length === 0) {
          results.extractionErrors++;
          if (results.errorSamples.length < 10) {
            results.errorSamples.push({ message, intent: result.intent, extractedData: result.extractedData });
          }
        }
      }
    } catch (err) {
      results.errors++;
      if (results.errorSamples.length < 10) {
        results.errorSamples.push({ message, error: err.message });
      }
    }
  }
}

// Summary
const avgPerf = results.performance.reduce((a, b) => a + b, 0) / results.performance.length;
console.log('--- Massive Test Summary ---');
console.log(`Total messages tested: ${results.total}`);
console.log(`Total errors: ${results.errors}`);
console.log(`Empty intent detections: ${results.emptyIntent}`);
console.log(`Extraction errors (SIP_CREATE/ONBOARDING/AI_ANALYSIS): ${results.extractionErrors}`);
console.log('Intent distribution:', results.intentStats);
console.log(`Average parse time: ${avgPerf.toFixed(2)} ms`);
if (results.errorSamples.length > 0) {
  console.log('Sample errors/extraction issues:', results.errorSamples);
}

// Optionally, write results to a file
fs.writeFileSync('massive-test-results.json', JSON.stringify(results, null, 2));

console.log('✅ Massive WhatsApp chatbot test completed. Results saved to massive-test-results.json'); 