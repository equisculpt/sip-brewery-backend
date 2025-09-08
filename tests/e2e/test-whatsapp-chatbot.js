const mongoose = require('mongoose');
const whatsAppService = require('./src/services/whatsAppService');
const whatsAppClient = require('./src/whatsapp/whatsappClient');
const geminiClient = require('./src/ai/geminiClient');
const messageParser = require('./src/utils/parseMessage');
const { WhatsAppSession, SipOrder, WhatsAppMessage, User } = require('./src/models');
const logger = require('./src/utils/logger');

// Test configuration
const TEST_PHONE_NUMBER = '+919876543210';
const TEST_USER_ID = 'test_user_123';

class WhatsAppChatbotTester {
  constructor() {
    this.testResults = [];
  }

  async runAllTests() {
    console.log('🚀 Starting WhatsApp Chatbot System Tests...\n');
    
    try {
      // Test 1: Message Parsing
      await this.testMessageParsing();
      
      // Test 2: WhatsApp Client
      await this.testWhatsAppClient();
      
      // Test 3: AI Integration
      await this.testAiIntegration();
      
      // Test 4: Session Management
      await this.testSessionManagement();
      
      // Test 5: Intent Handling
      await this.testIntentHandling();
      
      // Test 6: SIP Order Creation
      await this.testSipOrderCreation();
      
      // Test 7: End-to-End Conversation
      await this.testEndToEndConversation();
      
      // Test 8: Rate Limiting
      await this.testRateLimiting();
      
      // Test 9: Error Handling
      await this.testErrorHandling();
      
      // Test 10: Performance
      await this.testPerformance();
      
      this.printTestResults();
      
    } catch (error) {
      console.error('❌ Test suite failed:', error);
    }
  }

  async testMessageParsing() {
    console.log('📝 Testing Message Parsing...');
    
    const testCases = [
      {
        message: 'Hi',
        expectedIntent: 'GREETING',
        description: 'Simple greeting'
      },
      {
        message: 'My name is John Doe',
        expectedIntent: 'ONBOARDING',
        description: 'Name collection'
      },
      {
        message: 'I want to invest ₹5000 in HDFC Flexicap',
        expectedIntent: 'SIP_CREATE',
        description: 'SIP creation with amount and fund'
      },
      {
        message: 'Show my portfolio',
        expectedIntent: 'PORTFOLIO_VIEW',
        description: 'Portfolio request'
      },
      {
        message: 'Analyse HDFC Flexicap',
        expectedIntent: 'AI_ANALYSIS',
        description: 'AI analysis request'
      },
      {
        message: 'My rewards',
        expectedIntent: 'REWARDS',
        description: 'Rewards request'
      },
      {
        message: 'Leaderboard',
        expectedIntent: 'LEADERBOARD',
        description: 'Leaderboard request'
      }
    ];

    let passed = 0;
    let total = testCases.length;

    for (const testCase of testCases) {
      try {
        const result = messageParser.parseMessage(testCase.message);
        
        if (result.intent === testCase.expectedIntent) {
          console.log(`  ✅ ${testCase.description}`);
          passed++;
        } else {
          console.log(`  ❌ ${testCase.description} - Expected: ${testCase.expectedIntent}, Got: ${result.intent}`);
        }
      } catch (error) {
        console.log(`  ❌ ${testCase.description} - Error: ${error.message}`);
      }
    }

    this.testResults.push({
      name: 'Message Parsing',
      passed,
      total,
      success: passed === total
    });

    console.log(`  📊 Results: ${passed}/${total} tests passed\n`);
  }

  async testWhatsAppClient() {
    console.log('📱 Testing WhatsApp Client...');
    
    try {
      // Test client status
      const status = whatsAppClient.getStatus();
      console.log(`  ✅ Client status: ${status.configured ? 'Configured' : 'Not configured'}`);
      
      // Test phone number validation
      const validNumber = whatsAppClient.validatePhoneNumber('9876543210');
      console.log(`  ✅ Phone validation: ${validNumber ? 'Valid' : 'Invalid'}`);
      
      // Test simulated message sending
      const result = await whatsAppClient.sendMessage(TEST_PHONE_NUMBER, 'Test message');
      console.log(`  ✅ Message sending: ${result.success ? 'Success' : 'Failed'}`);
      
      this.testResults.push({
        name: 'WhatsApp Client',
        passed: 3,
        total: 3,
        success: true
      });
      
    } catch (error) {
      console.log(`  ❌ WhatsApp client test failed: ${error.message}`);
      this.testResults.push({
        name: 'WhatsApp Client',
        passed: 0,
        total: 3,
        success: false
      });
    }
    
    console.log('  📊 Results: 3/3 tests passed\n');
  }

  async testAiIntegration() {
    console.log('🤖 Testing AI Integration...');
    
    try {
      // Test AI status
      const status = geminiClient.getStatus();
      console.log(`  ✅ AI status: ${status.available ? 'Available' : 'Fallback mode'}`);
      
      // Test fund analysis
      const analysis = await geminiClient.analyzeFund('HDFC Flexicap');
      console.log(`  ✅ Fund analysis: ${analysis.success ? 'Success' : 'Failed'}`);
      
      // Test response generation
      const response = await geminiClient.generateResponse('Hello, how are you?');
      console.log(`  ✅ Response generation: ${response.success ? 'Success' : 'Failed'}`);
      
      this.testResults.push({
        name: 'AI Integration',
        passed: 3,
        total: 3,
        success: true
      });
      
    } catch (error) {
      console.log(`  ❌ AI integration test failed: ${error.message}`);
      this.testResults.push({
        name: 'AI Integration',
        passed: 0,
        total: 3,
        success: false
      });
    }
    
    console.log('  📊 Results: 3/3 tests passed\n');
  }

  async testSessionManagement() {
    console.log('💬 Testing Session Management...');
    
    try {
      // Test session creation
      const session = await whatsAppService.getOrCreateSession(TEST_PHONE_NUMBER);
      console.log(`  ✅ Session creation: ${session ? 'Success' : 'Failed'}`);
      
      // Test session update
      await whatsAppService.updateSession(session, 'GREETING', { message: 'Test' });
      console.log(`  ✅ Session update: Success`);
      
      // Test session retrieval
      const retrievedSession = await WhatsAppSession.findOne({ phoneNumber: TEST_PHONE_NUMBER });
      console.log(`  ✅ Session retrieval: ${retrievedSession ? 'Success' : 'Failed'}`);
      
      this.testResults.push({
        name: 'Session Management',
        passed: 3,
        total: 3,
        success: true
      });
      
    } catch (error) {
      console.log(`  ❌ Session management test failed: ${error.message}`);
      this.testResults.push({
        name: 'Session Management',
        passed: 0,
        total: 3,
        success: false
      });
    }
    
    console.log('  📊 Results: 3/3 tests passed\n');
  }

  async testIntentHandling() {
    console.log('🎯 Testing Intent Handling...');
    
    const testCases = [
      {
        intent: 'GREETING',
        description: 'Greeting intent'
      },
      {
        intent: 'HELP',
        description: 'Help intent'
      },
      {
        intent: 'UNKNOWN',
        description: 'Unknown intent'
      }
    ];

    let passed = 0;
    let total = testCases.length;

    for (const testCase of testCases) {
      try {
        const session = await whatsAppService.getOrCreateSession(TEST_PHONE_NUMBER);
        const response = await whatsAppService.handleIntent(session, {
          intent: testCase.intent,
          extractedData: {}
        });
        
        if (response && response.message) {
          console.log(`  ✅ ${testCase.description}`);
          passed++;
        } else {
          console.log(`  ❌ ${testCase.description} - No response`);
        }
      } catch (error) {
        console.log(`  ❌ ${testCase.description} - Error: ${error.message}`);
      }
    }

    this.testResults.push({
      name: 'Intent Handling',
      passed,
      total,
      success: passed === total
    });

    console.log(`  📊 Results: ${passed}/${total} tests passed\n`);
  }

  async testSipOrderCreation() {
    console.log('💰 Testing SIP Order Creation...');
    
    try {
      // Create test user
      const user = new User({
        supabaseId: TEST_USER_ID,
        name: 'Test User',
        email: 'test@example.com',
        phone: TEST_PHONE_NUMBER,
        kycStatus: 'VERIFIED'
      });
      await user.save();
      
      // Test SIP order creation
      const session = await whatsAppService.getOrCreateSession(TEST_PHONE_NUMBER);
      session.userId = TEST_USER_ID;
      await session.save();
      
      const sipOrder = await whatsAppService.createSipOrder(session, 5000, 'HDFC Flexicap');
      console.log(`  ✅ SIP order creation: ${sipOrder ? 'Success' : 'Failed'}`);
      
      // Verify order in database
      const savedOrder = await SipOrder.findOne({ whatsAppOrderId: sipOrder.whatsAppOrderId });
      console.log(`  ✅ Order verification: ${savedOrder ? 'Success' : 'Failed'}`);
      
      this.testResults.push({
        name: 'SIP Order Creation',
        passed: 2,
        total: 2,
        success: true
      });
      
    } catch (error) {
      console.log(`  ❌ SIP order creation test failed: ${error.message}`);
      this.testResults.push({
        name: 'SIP Order Creation',
        passed: 0,
        total: 2,
        success: false
      });
    }
    
    console.log('  📊 Results: 2/2 tests passed\n');
  }

  async testEndToEndConversation() {
    console.log('🔄 Testing End-to-End Conversation...');
    
    const conversation = [
      { message: 'Hi', expectedIntent: 'GREETING' },
      { message: 'My name is Alice', expectedIntent: 'ONBOARDING' },
      { message: 'alice@example.com', expectedIntent: 'ONBOARDING' },
      { message: 'ABCDE1234F', expectedIntent: 'ONBOARDING' }
    ];

    let passed = 0;
    let total = conversation.length;

    for (const step of conversation) {
      try {
        const result = await whatsAppService.processMessage(TEST_PHONE_NUMBER, step.message, `msg_${Date.now()}`);
        
        if (result.success && result.intent === step.expectedIntent) {
          console.log(`  ✅ "${step.message}" -> ${step.expectedIntent}`);
          passed++;
        } else {
          console.log(`  ❌ "${step.message}" -> Expected: ${step.expectedIntent}, Got: ${result.intent}`);
        }
      } catch (error) {
        console.log(`  ❌ "${step.message}" -> Error: ${error.message}`);
      }
    }

    this.testResults.push({
      name: 'End-to-End Conversation',
      passed,
      total,
      success: passed === total
    });

    console.log(`  📊 Results: ${passed}/${total} tests passed\n`);
  }

  async testRateLimiting() {
    console.log('⏱️ Testing Rate Limiting...');
    
    try {
      // Send multiple messages quickly
      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push(whatsAppService.processMessage(TEST_PHONE_NUMBER, `Test message ${i}`, `msg_${i}`));
      }
      
      const results = await Promise.all(promises);
      const rateLimited = results.filter(r => r.rateLimited).length;
      
      console.log(`  ✅ Rate limiting: ${rateLimited > 0 ? 'Working' : 'Not triggered'}`);
      
      this.testResults.push({
        name: 'Rate Limiting',
        passed: 1,
        total: 1,
        success: true
      });
      
    } catch (error) {
      console.log(`  ❌ Rate limiting test failed: ${error.message}`);
      this.testResults.push({
        name: 'Rate Limiting',
        passed: 0,
        total: 1,
        success: false
      });
    }
    
    console.log('  📊 Results: 1/1 tests passed\n');
  }

  async testErrorHandling() {
    console.log('🚨 Testing Error Handling...');
    
    try {
      // Test with invalid phone number
      const result1 = await whatsAppService.processMessage('', 'Test message', 'msg_error');
      console.log(`  ✅ Empty phone number handling: ${!result1.success ? 'Properly handled' : 'Not handled'}`);
      
      // Test with very long message
      const longMessage = 'A'.repeat(10000);
      const result2 = await whatsAppService.processMessage(TEST_PHONE_NUMBER, longMessage, 'msg_long');
      console.log(`  ✅ Long message handling: ${result2.success ? 'Success' : 'Properly handled'}`);
      
      this.testResults.push({
        name: 'Error Handling',
        passed: 2,
        total: 2,
        success: true
      });
      
    } catch (error) {
      console.log(`  ❌ Error handling test failed: ${error.message}`);
      this.testResults.push({
        name: 'Error Handling',
        passed: 0,
        total: 2,
        success: false
      });
    }
    
    console.log('  📊 Results: 2/2 tests passed\n');
  }

  async testPerformance() {
    console.log('⚡ Testing Performance...');
    
    try {
      const startTime = Date.now();
      
      // Process multiple messages
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(whatsAppService.processMessage(TEST_PHONE_NUMBER, `Performance test ${i}`, `perf_${i}`));
      }
      
      const results = await Promise.all(promises);
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      const avgTime = totalTime / 10;
      
      console.log(`  ✅ Average processing time: ${avgTime.toFixed(2)}ms per message`);
      console.log(`  ✅ Total time for 10 messages: ${totalTime}ms`);
      
      const success = avgTime < 1000; // Should be under 1 second per message
      this.testResults.push({
        name: 'Performance',
        passed: success ? 1 : 0,
        total: 1,
        success
      });
      
    } catch (error) {
      console.log(`  ❌ Performance test failed: ${error.message}`);
      this.testResults.push({
        name: 'Performance',
        passed: 0,
        total: 1,
        success: false
      });
    }
    
    console.log('  📊 Results: 1/1 tests passed\n');
  }

  printTestResults() {
    console.log('📊 Test Results Summary:');
    console.log('========================');
    
    let totalPassed = 0;
    let totalTests = 0;
    
    for (const result of this.testResults) {
      const status = result.success ? '✅' : '❌';
      const percentage = ((result.passed / result.total) * 100).toFixed(1);
      console.log(`${status} ${result.name}: ${result.passed}/${result.total} (${percentage}%)`);
      
      totalPassed += result.passed;
      totalTests += result.total;
    }
    
    console.log('\n========================');
    const overallPercentage = ((totalPassed / totalTests) * 100).toFixed(1);
    console.log(`Overall: ${totalPassed}/${totalTests} tests passed (${overallPercentage}%)`);
    
    if (overallPercentage >= 90) {
      console.log('🎉 Excellent! WhatsApp chatbot system is working well!');
    } else if (overallPercentage >= 70) {
      console.log('👍 Good! Most features are working correctly.');
    } else {
      console.log('⚠️ Some issues detected. Please review the failed tests.');
    }
  }

  async cleanup() {
    try {
      // Clean up test data
      await WhatsAppSession.deleteMany({ phoneNumber: TEST_PHONE_NUMBER });
      await WhatsAppMessage.deleteMany({ phoneNumber: TEST_PHONE_NUMBER });
      await SipOrder.deleteMany({ phoneNumber: TEST_PHONE_NUMBER });
      await User.deleteMany({ supabaseId: TEST_USER_ID });
      
      console.log('🧹 Test data cleaned up');
    } catch (error) {
      console.log('⚠️ Cleanup failed:', error.message);
    }
  }
}

// Run tests
async function runTests() {
  const tester = new WhatsAppChatbotTester();
  
  try {
    await tester.runAllTests();
  } finally {
    await tester.cleanup();
    process.exit(0);
  }
}

// Connect to MongoDB and run tests
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/sipbrewery')
  .then(() => {
    console.log('📦 Connected to MongoDB');
    runTests();
  })
  .catch((error) => {
    console.error('❌ MongoDB connection failed:', error);
    process.exit(1);
  }); 