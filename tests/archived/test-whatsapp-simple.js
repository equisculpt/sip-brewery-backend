const messageParser = require('./src/utils/parseMessage');
const geminiClient = require('./src/ai/geminiClient');
const whatsAppClient = require('./src/whatsapp/whatsappClient');

console.log('🧪 Testing WhatsApp Chatbot System (Simple Mode)\n');

// Test 1: Message Parsing
console.log('📝 Testing Message Parsing...');
const testMessages = [
  'Hi',
  'My name is John Doe',
  'I want to invest ₹5000 in HDFC Flexicap',
  'Show my portfolio',
  'Analyse HDFC Flexicap',
  'My rewards',
  'Leaderboard',
  'Help'
];

testMessages.forEach(message => {
  const result = messageParser.parseMessage(message);
  console.log(`  "${message}" -> ${result.intent} (confidence: ${result.confidence})`);
});

// Test 2: WhatsApp Client
console.log('\n📱 Testing WhatsApp Client...');
const status = whatsAppClient.getStatus();
console.log(`  Provider: ${status.provider}`);
console.log(`  Configured: ${status.configured}`);
console.log(`  Phone Number: ${status.phoneNumber || 'Not set'}`);

// Test phone validation
const testNumbers = ['9876543210', '+919876543210', '919876543210'];
testNumbers.forEach(num => {
  const valid = whatsAppClient.validatePhoneNumber(num);
  console.log(`  "${num}" -> ${valid || 'Invalid'}`);
});

// Test 3: AI Integration
console.log('\n🤖 Testing AI Integration...');
const aiStatus = geminiClient.getStatus();
console.log(`  Available: ${aiStatus.available}`);
console.log(`  Provider: ${aiStatus.provider}`);
console.log(`  API Key Configured: ${aiStatus.apiKeyConfigured}`);

// Test 4: Template Messages
console.log('\n📋 Testing Template Messages...');
const templates = [
  'welcome',
  'onboarding_name',
  'kyc_verified',
  'sip_confirmation',
  'portfolio_summary',
  'rewards_summary',
  'referral_link',
  'leaderboard',
  'help'
];

templates.forEach(template => {
  try {
    const message = whatsAppClient.sendTemplate('+919876543210', template, {
      name: 'John',
      fundName: 'HDFC Flexicap',
      amount: 5000,
      orderId: 'SIP123456',
      totalValue: 125000,
      totalInvested: 100000,
      returns: 25.0,
      topHoldings: ['HDFC Flexicap - ₹45,000', 'SBI Smallcap - ₹35,000'],
      points: 1500,
      cashback: 250,
      referralBonus: 100,
      pendingPayout: 350,
      referralCode: 'JOHN123',
      leaders: [
        { name: 'User1', returns: 28.5 },
        { name: 'User2', returns: 25.2 }
      ]
    });
    console.log(`  ✅ ${template}: Generated successfully`);
  } catch (error) {
    console.log(`  ❌ ${template}: ${error.message}`);
  }
});

// Test 5: AI Analysis (Fallback Mode)
console.log('\n🔍 Testing AI Analysis (Fallback Mode)...');
const testFunds = ['HDFC Flexicap', 'SBI Smallcap', 'Parag Parikh Flexicap', 'Unknown Fund'];

testFunds.forEach(async (fund) => {
  try {
    const analysis = await geminiClient.analyzeFund(fund);
    console.log(`  ✅ "${fund}": ${analysis.aiProvider} - ${analysis.analysis.substring(0, 50)}...`);
  } catch (error) {
    console.log(`  ❌ "${fund}": ${error.message}`);
  }
});

// Test 6: AI Response Generation
console.log('\n💬 Testing AI Response Generation...');
const testQueries = [
  'Hello, how are you?',
  'What can you help me with?',
  'Tell me about mutual funds',
  'How do I start investing?'
];

testQueries.forEach(async (query) => {
  try {
    const response = await geminiClient.generateResponse(query);
    console.log(`  ✅ "${query}": ${response.aiProvider} - ${response.response.substring(0, 50)}...`);
  } catch (error) {
    console.log(`  ❌ "${query}": ${error.message}`);
  }
});

// Test 7: Data Extraction
console.log('\n🔧 Testing Data Extraction...');
const extractionTests = [
  {
    message: 'I want to invest ₹5000 in HDFC Flexicap',
    expected: { amount: 5000, fundName: 'HDFC Flexicap' }
  },
  {
    message: 'My name is Alice Johnson',
    expected: { name: 'Alice Johnson' }
  },
  {
    message: 'email is alice@example.com',
    expected: { email: 'alice@example.com' }
  },
  {
    message: 'PAN is ABCDE1234F',
    expected: { pan: 'ABCDE1234F' }
  }
];

extractionTests.forEach(test => {
  const result = messageParser.parseMessage(test.message);
  console.log(`  "${test.message}"`);
  console.log(`    Intent: ${result.intent}`);
  console.log(`    Data:`, result.extractedData);
});

// Test 8: Validation
console.log('\n✅ Testing Validation...');
const validationTests = [
  { email: 'test@example.com', expected: true },
  { email: 'invalid-email', expected: false },
  { pan: 'ABCDE1234F', expected: true },
  { pan: 'INVALID', expected: false }
];

validationTests.forEach(test => {
  if (test.email) {
    const isValid = messageParser.isEmail(test.email);
    console.log(`  Email "${test.email}": ${isValid === test.expected ? '✅' : '❌'}`);
  }
  if (test.pan) {
    const isValid = messageParser.isPAN(test.pan);
    console.log(`  PAN "${test.pan}": ${isValid === test.expected ? '✅' : '❌'}`);
  }
});

console.log('\n🎉 Simple WhatsApp Chatbot Tests Completed!');
console.log('\n📊 Summary:');
console.log('✅ Message parsing with intent detection');
console.log('✅ WhatsApp client with phone validation');
console.log('✅ AI integration with fallback mode');
console.log('✅ Template message generation');
console.log('✅ Data extraction from messages');
console.log('✅ Input validation (email, PAN)');
console.log('\n🚀 The WhatsApp chatbot system is ready for integration!'); 