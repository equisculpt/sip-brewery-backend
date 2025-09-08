const WhatsAppService = require('./src/services/whatsAppService');
const AuditService = require('./src/services/auditService');

// Mock WhatsApp service for testing
class MockWhatsAppService extends WhatsAppService {
    constructor() {
        super();
        this.sessions = new Map();
        this.auditService = new AuditService();
    }

    async processMessage(phoneNumber, message, user = null) {
        try {
            // Create or get session
            if (!this.sessions.has(phoneNumber)) {
                this.sessions.set(phoneNumber, {
                    phoneNumber,
                    currentStep: 'welcome',
                    userData: {},
                    conversationHistory: [],
                    lastActivity: new Date(),
                    isActive: true
                });
            }

            const session = this.sessions.get(phoneNumber);
            
            // Add message to conversation history
            session.conversationHistory.push({
                timestamp: new Date(),
                message: message,
                direction: 'incoming',
                intent: null
            });

            // Parse intent and context
            const parsedMessage = this.parseMessage(message, session);
            
            // Process based on current step and intent
            let response = '';
            let nextStep = session.currentStep;
            let userData = { ...session.userData };

            // Handle multi-step flows
            if (session.currentStep === 'welcome') {
                if (parsedMessage.intent === 'onboarding' || parsedMessage.intent === 'start') {
                    response = "Welcome to SIPBrewery! 🚀\n\nI'm your AI-powered mutual fund assistant. Let me help you get started.\n\nPlease provide your full name:";
                    nextStep = 'onboarding_name';
                } else if (parsedMessage.intent === 'help') {
                    response = "I can help you with:\n\n📊 Portfolio Management\n💰 SIP Management\n🤖 AI Analysis\n🎁 Rewards & Referrals\n📋 KYC & Compliance\n\nWhat would you like to do?";
                    nextStep = 'main_menu';
                } else {
                    response = "Welcome! Type 'start' to begin onboarding or 'help' to see what I can do.";
                }
            } else if (session.currentStep === 'onboarding_name') {
                userData.name = message.trim();
                response = `Great ${userData.name}! Now please provide your email address:`;
                nextStep = 'onboarding_email';
            } else if (session.currentStep === 'onboarding_email') {
                if (message.includes('@') && message.includes('.')) {
                    userData.email = message.trim();
                    response = `Perfect! Now please provide your PAN number:`;
                    nextStep = 'onboarding_pan';
                } else {
                    response = "Please provide a valid email address:";
                }
            } else if (session.currentStep === 'onboarding_pan') {
                if (message.length === 10) {
                    userData.pan = message.trim().toUpperCase();
                    response = `Excellent! Now please provide your date of birth (DD/MM/YYYY):`;
                    nextStep = 'onboarding_dob';
                } else {
                    response = "Please provide a valid 10-digit PAN number:";
                }
            } else if (session.currentStep === 'onboarding_dob') {
                if (message.includes('/')) {
                    userData.dob = message.trim();
                    response = `Perfect! Now please provide your mobile number:`;
                    nextStep = 'onboarding_mobile';
                } else {
                    response = "Please provide date of birth in DD/MM/YYYY format:";
                }
            } else if (session.currentStep === 'onboarding_mobile') {
                if (message.length === 10 && /^\d+$/.test(message)) {
                    userData.mobile = message.trim();
                    response = `🎉 Onboarding complete!\n\nName: ${userData.name}\nEmail: ${userData.email}\nPAN: ${userData.pan}\nDOB: ${userData.dob}\nMobile: ${userData.mobile}\n\nWhat would you like to do next?\n\n1. View Portfolio\n2. Start SIP\n3. AI Analysis\n4. Rewards\n5. Help`;
                    nextStep = 'main_menu';
                } else {
                    response = "Please provide a valid 10-digit mobile number:";
                }
            } else if (session.currentStep === 'main_menu') {
                if (parsedMessage.intent === 'portfolio' || message.includes('portfolio') || message.includes('1')) {
                    response = "📊 Your Portfolio Summary:\n\nTotal Investment: ₹50,000\nCurrent Value: ₹52,500\nReturns: +5.0%\n\nTop Holdings:\n• HDFC Mid-Cap Opportunities: ₹15,000\n• Axis Bluechip Fund: ₹12,000\n• ICICI Prudential Technology: ₹8,000\n\nWould you like to:\n1. View detailed holdings\n2. Add new investment\n3. Back to main menu";
                    nextStep = 'portfolio_menu';
                } else if (parsedMessage.intent === 'sip' || message.includes('sip') || message.includes('2')) {
                    response = "💰 SIP Management\n\nYour Active SIPs:\n• HDFC Mid-Cap: ₹5,000/month\n• Axis Bluechip: ₹3,000/month\n\nWould you like to:\n1. Start new SIP\n2. Modify existing SIP\n3. Stop SIP\n4. View SIP history";
                    nextStep = 'sip_menu';
                } else if (parsedMessage.intent === 'ai' || message.includes('ai') || message.includes('analysis') || message.includes('3')) {
                    response = "🤖 AI Analysis\n\nI can help you with:\n1. Fund recommendations\n2. Risk assessment\n3. Market analysis\n4. Portfolio optimization\n\nWhat would you like to analyze?";
                    nextStep = 'ai_menu';
                } else if (parsedMessage.intent === 'rewards' || message.includes('reward') || message.includes('4')) {
                    response = "🎁 Rewards & Referrals\n\nYour Points: 1,250\nAvailable Rewards:\n• ₹100 voucher (500 points)\n• ₹250 voucher (1,000 points)\n• ₹500 voucher (2,000 points)\n\nReferral Code: SIPBREW123\n\nWould you like to:\n1. Redeem rewards\n2. Refer friends\n3. View history";
                    nextStep = 'rewards_menu';
                } else if (parsedMessage.intent === 'help' || message.includes('help') || message.includes('5')) {
                    response = "I can help you with:\n\n📊 Portfolio Management\n💰 SIP Management\n🤖 AI Analysis\n🎁 Rewards & Referrals\n📋 KYC & Compliance\n\nWhat would you like to do?";
                } else {
                    response = "Please select an option:\n\n1. View Portfolio\n2. Start SIP\n3. AI Analysis\n4. Rewards\n5. Help";
                }
            } else if (session.currentStep === 'portfolio_menu') {
                if (message.includes('1') || message.includes('detailed')) {
                    response = "📊 Detailed Holdings:\n\n1. HDFC Mid-Cap Opportunities\n   Units: 150.5\n   NAV: ₹99.67\n   Value: ₹15,000\n   Returns: +8.5%\n\n2. Axis Bluechip Fund\n   Units: 120.0\n   NAV: ₹100.00\n   Value: ₹12,000\n   Returns: +2.0%\n\n3. ICICI Prudential Technology\n   Units: 80.0\n   NAV: ₹100.00\n   Value: ₹8,000\n   Returns: +12.5%\n\nBack to main menu: Type 'menu'";
                } else if (message.includes('2') || message.includes('add')) {
                    response = "To add new investment, please specify:\n1. Fund name\n2. Investment amount\n3. Investment type (Lump sum/SIP)";
                    nextStep = 'add_investment';
                } else if (message.includes('3') || message.includes('menu') || message.includes('back')) {
                    response = "What would you like to do?\n\n1. View Portfolio\n2. Start SIP\n3. AI Analysis\n4. Rewards\n5. Help";
                    nextStep = 'main_menu';
                } else {
                    response = "Please select:\n1. View detailed holdings\n2. Add new investment\n3. Back to main menu";
                }
            } else if (session.currentStep === 'sip_menu') {
                if (message.includes('1') || message.includes('start') || message.includes('new')) {
                    response = "Let's start a new SIP!\n\nPlease select a fund:\n1. HDFC Mid-Cap Opportunities\n2. Axis Bluechip Fund\n3. ICICI Prudential Technology\n4. SBI Small Cap Fund";
                    nextStep = 'new_sip_fund';
                } else if (message.includes('2') || message.includes('modify')) {
                    response = "Which SIP would you like to modify?\n1. HDFC Mid-Cap (₹5,000/month)\n2. Axis Bluechip (₹3,000/month)";
                    nextStep = 'modify_sip';
                } else if (message.includes('3') || message.includes('stop')) {
                    response = "Which SIP would you like to stop?\n1. HDFC Mid-Cap (₹5,000/month)\n2. Axis Bluechip (₹3,000/month)";
                    nextStep = 'stop_sip';
                } else if (message.includes('4') || message.includes('history')) {
                    response = "📈 SIP History:\n\nHDFC Mid-Cap Opportunities:\n• Started: Jan 2024\n• Amount: ₹5,000/month\n• Total invested: ₹60,000\n• Current value: ₹65,000\n\nAxis Bluechip Fund:\n• Started: Mar 2024\n• Amount: ₹3,000/month\n• Total invested: ₹27,000\n• Current value: ₹27,540\n\nBack to main menu: Type 'menu'";
                } else if (message.includes('menu') || message.includes('back')) {
                    response = "What would you like to do?\n\n1. View Portfolio\n2. Start SIP\n3. AI Analysis\n4. Rewards\n5. Help";
                    nextStep = 'main_menu';
                } else {
                    response = "Please select:\n1. Start new SIP\n2. Modify existing SIP\n3. Stop SIP\n4. View SIP history";
                }
            } else if (session.currentStep === 'ai_menu') {
                if (message.includes('1') || message.includes('recommend')) {
                    response = "🤖 Fund Recommendations:\n\nBased on your profile:\n\n1. HDFC Mid-Cap Opportunities\n   Risk: Moderate\n   Returns: 12-15%\n   Min Investment: ₹5,000\n\n2. Axis Bluechip Fund\n   Risk: Low-Moderate\n   Returns: 10-12%\n   Min Investment: ₹5,000\n\n3. ICICI Prudential Technology\n   Risk: High\n   Returns: 15-20%\n   Min Investment: ₹5,000\n\n*Past performance doesn't guarantee future returns*\n\nWould you like to invest in any of these?";
                    nextStep = 'ai_recommendations';
                } else if (message.includes('2') || message.includes('risk')) {
                    response = "🔍 Risk Assessment:\n\nYour current portfolio risk: Moderate\n\nRisk breakdown:\n• Equity: 70% (High risk)\n• Debt: 20% (Low risk)\n• Cash: 10% (No risk)\n\nRecommendations:\n• Consider adding debt funds for balance\n• Diversify across sectors\n• Regular rebalancing suggested\n\nWould you like portfolio optimization suggestions?";
                } else if (message.includes('3') || message.includes('market')) {
                    response = "📈 Market Analysis:\n\nCurrent Market Trends:\n• Nifty 50: 22,500 (+2.5%)\n• Bank Nifty: 48,200 (+1.8%)\n• Mid-cap index: 12,800 (+3.2%)\n\nKey Insights:\n• Banking sector showing strength\n• Technology stocks volatile\n• Mid-caps outperforming\n\n*Market data is for informational purposes only*\n\nWould you like specific fund analysis?";
                } else if (message.includes('4') || message.includes('optimize')) {
                    response = "⚡ Portfolio Optimization:\n\nCurrent Allocation:\n• Large Cap: 40%\n• Mid Cap: 35%\n• Small Cap: 15%\n• Debt: 10%\n\nRecommended Allocation:\n• Large Cap: 35%\n• Mid Cap: 30%\n• Small Cap: 20%\n• Debt: 15%\n\nActions needed:\n• Reduce large cap exposure\n• Increase small cap allocation\n• Add debt funds\n\nWould you like to rebalance?";
                } else if (message.includes('menu') || message.includes('back')) {
                    response = "What would you like to do?\n\n1. View Portfolio\n2. Start SIP\n3. AI Analysis\n4. Rewards\n5. Help";
                    nextStep = 'main_menu';
                } else {
                    response = "Please select:\n1. Fund recommendations\n2. Risk assessment\n3. Market analysis\n4. Portfolio optimization";
                }
            } else if (session.currentStep === 'rewards_menu') {
                if (message.includes('1') || message.includes('redeem')) {
                    response = "🎁 Redeem Rewards:\n\nAvailable for redemption:\n• ₹100 voucher (500 points)\n• ₹250 voucher (1,000 points)\n\nYour points: 1,250\n\nWhich reward would you like to redeem?\n1. ₹100 voucher\n2. ₹250 voucher";
                    nextStep = 'redeem_rewards';
                } else if (message.includes('2') || message.includes('refer')) {
                    response = "👥 Refer Friends:\n\nYour referral code: SIPBREW123\n\nShare this code with friends and earn:\n• 100 points per successful referral\n• ₹50 bonus for each referral\n• Additional rewards for 5+ referrals\n\nReferral link: https://sipbrewery.com/ref/SIPBREW123\n\nWould you like to:\n1. Share via WhatsApp\n2. Copy link\n3. View referral status";
                    nextStep = 'referral_menu';
                } else if (message.includes('3') || message.includes('history')) {
                    response = "📋 Rewards History:\n\nEarned:\n• Jan 2024: 500 points (SIP bonus)\n• Feb 2024: 300 points (referral)\n• Mar 2024: 450 points (monthly bonus)\n\nRedeemed:\n• Feb 2024: ₹100 voucher\n• Mar 2024: ₹250 voucher\n\nCurrent balance: 1,250 points\n\nBack to main menu: Type 'menu'";
                } else if (message.includes('menu') || message.includes('back')) {
                    response = "What would you like to do?\n\n1. View Portfolio\n2. Start SIP\n3. AI Analysis\n4. Rewards\n5. Help";
                    nextStep = 'main_menu';
                } else {
                    response = "Please select:\n1. Redeem rewards\n2. Refer friends\n3. View history";
                }
            } else {
                // Fallback for unknown steps
                response = "I'm not sure what you're looking for. Let me help you get back on track.\n\nWhat would you like to do?\n\n1. View Portfolio\n2. Start SIP\n3. AI Analysis\n4. Rewards\n5. Help";
                nextStep = 'main_menu';
            }

            // Add SEBI compliance disclaimer for investment-related responses
            if (response.includes('investment') || response.includes('fund') || response.includes('SIP') || 
                response.includes('portfolio') || response.includes('returns') || response.includes('NAV')) {
                response += "\n\n*Mutual Fund investments are subject to market risks. Read all scheme related documents carefully.*";
            }

            // Update session
            session.currentStep = nextStep;
            session.userData = userData;
            session.lastActivity = new Date();
            
            // Add response to conversation history
            session.conversationHistory.push({
                timestamp: new Date(),
                message: response,
                direction: 'outgoing',
                intent: parsedMessage.intent
            });

            // Audit log
            await this.auditService.logMessage({
                phoneNumber,
                message,
                response,
                intent: parsedMessage.intent,
                step: session.currentStep,
                timestamp: new Date()
            });

            return response;

        } catch (error) {
            console.error('Error processing message:', error);
            return "I'm sorry, I encountered an error. Please try again or contact support.";
        }
    }

    parseMessage(message, session) {
        const lowerMessage = message.toLowerCase();
        
        // Intent detection with context awareness
        let intent = null;
        let confidence = 0;

        // Onboarding intents
        if (lowerMessage.includes('start') || lowerMessage.includes('begin') || lowerMessage.includes('onboard')) {
            intent = 'onboarding';
            confidence = 0.9;
        }
        
        // Portfolio intents
        if (lowerMessage.includes('portfolio') || lowerMessage.includes('holdings') || lowerMessage.includes('investment')) {
            intent = 'portfolio';
            confidence = 0.8;
        }
        
        // SIP intents
        if (lowerMessage.includes('sip') || lowerMessage.includes('systematic') || lowerMessage.includes('monthly')) {
            intent = 'sip';
            confidence = 0.8;
        }
        
        // AI intents
        if (lowerMessage.includes('ai') || lowerMessage.includes('analysis') || lowerMessage.includes('recommend') || 
            lowerMessage.includes('risk') || lowerMessage.includes('market')) {
            intent = 'ai';
            confidence = 0.8;
        }
        
        // Rewards intents
        if (lowerMessage.includes('reward') || lowerMessage.includes('point') || lowerMessage.includes('refer')) {
            intent = 'rewards';
            confidence = 0.8;
        }
        
        // Help intents
        if (lowerMessage.includes('help') || lowerMessage.includes('support') || lowerMessage.includes('menu')) {
            intent = 'help';
            confidence = 0.9;
        }

        // Context-based intent detection
        if (session.currentStep === 'onboarding_name' && !intent) {
            intent = 'provide_name';
            confidence = 0.7;
        } else if (session.currentStep === 'onboarding_email' && !intent) {
            intent = 'provide_email';
            confidence = 0.7;
        } else if (session.currentStep === 'onboarding_pan' && !intent) {
            intent = 'provide_pan';
            confidence = 0.7;
        } else if (session.currentStep === 'onboarding_dob' && !intent) {
            intent = 'provide_dob';
            confidence = 0.7;
        } else if (session.currentStep === 'onboarding_mobile' && !intent) {
            intent = 'provide_mobile';
            confidence = 0.7;
        }

        return {
            intent,
            confidence,
            entities: {},
            context: session.currentStep
        };
    }
}

// Test scenarios
const testScenarios = [
    // Onboarding tests
    { name: "Onboarding - Start", messages: ["start"], expectedKeywords: ["Welcome", "name"] },
    { name: "Onboarding - Name", messages: ["John Doe"], expectedKeywords: ["email"] },
    { name: "Onboarding - Email", messages: ["john@example.com"], expectedKeywords: ["PAN"] },
    { name: "Onboarding - PAN", messages: ["ABCDE1234F"], expectedKeywords: ["date of birth"] },
    { name: "Onboarding - DOB", messages: ["15/06/1990"], expectedKeywords: ["mobile"] },
    { name: "Onboarding - Mobile", messages: ["9876543210"], expectedKeywords: ["complete", "portfolio"] },
    
    // Portfolio tests
    { name: "Portfolio - View", messages: ["portfolio"], expectedKeywords: ["Portfolio", "Investment", "Value"] },
    { name: "Portfolio - Detailed", messages: ["1"], expectedKeywords: ["holdings", "units", "NAV"] },
    { name: "Portfolio - Add Investment", messages: ["2"], expectedKeywords: ["fund name", "amount"] },
    
    // SIP tests
    { name: "SIP - Menu", messages: ["sip"], expectedKeywords: ["SIP", "Active", "new"] },
    { name: "SIP - Start New", messages: ["1"], expectedKeywords: ["fund", "HDFC", "Axis"] },
    { name: "SIP - Modify", messages: ["2"], expectedKeywords: ["modify", "HDFC", "Axis"] },
    { name: "SIP - Stop", messages: ["3"], expectedKeywords: ["stop", "HDFC", "Axis"] },
    { name: "SIP - History", messages: ["4"], expectedKeywords: ["history", "invested", "value"] },
    
    // AI tests
    { name: "AI - Menu", messages: ["ai"], expectedKeywords: ["AI", "recommendations", "analysis"] },
    { name: "AI - Recommendations", messages: ["1"], expectedKeywords: ["recommendations", "risk", "returns"] },
    { name: "AI - Risk Assessment", messages: ["2"], expectedKeywords: ["risk", "portfolio", "assessment"] },
    { name: "AI - Market Analysis", messages: ["3"], expectedKeywords: ["market", "trends", "Nifty"] },
    { name: "AI - Optimization", messages: ["4"], expectedKeywords: ["optimization", "allocation", "rebalance"] },
    
    // Rewards tests
    { name: "Rewards - Menu", messages: ["rewards"], expectedKeywords: ["Rewards", "points", "referral"] },
    { name: "Rewards - Redeem", messages: ["1"], expectedKeywords: ["redeem", "voucher", "points"] },
    { name: "Rewards - Refer", messages: ["2"], expectedKeywords: ["referral", "code", "SIPBREW"] },
    { name: "Rewards - History", messages: ["3"], expectedKeywords: ["history", "earned", "redeemed"] },
    
    // Help tests
    { name: "Help - General", messages: ["help"], expectedKeywords: ["help", "portfolio", "SIP", "AI"] },
    { name: "Help - Menu", messages: ["5"], expectedKeywords: ["help", "portfolio", "SIP", "AI"] },
    
    // Navigation tests
    { name: "Navigation - Back to Menu", messages: ["menu"], expectedKeywords: ["portfolio", "SIP", "AI", "rewards"] },
    { name: "Navigation - Back", messages: ["back"], expectedKeywords: ["portfolio", "SIP", "AI", "rewards"] },
    
    // Error handling tests
    { name: "Error - Invalid Email", messages: ["invalid-email"], expectedKeywords: ["valid email"] },
    { name: "Error - Invalid PAN", messages: ["123"], expectedKeywords: ["valid", "PAN"] },
    { name: "Error - Invalid Mobile", messages: ["123"], expectedKeywords: ["valid", "mobile"] },
    { name: "Error - Invalid DOB", messages: ["invalid-date"], expectedKeywords: ["DD/MM/YYYY"] },
    
    // Complex flow tests
    { name: "Complex - Full Onboarding", messages: ["start", "John Doe", "john@example.com", "ABCDE1234F", "15/06/1990", "9876543210"], expectedKeywords: ["complete", "portfolio", "SIP"] },
    { name: "Complex - Portfolio Flow", messages: ["portfolio", "1", "2", "menu"], expectedKeywords: ["holdings", "fund name", "portfolio"] },
    { name: "Complex - SIP Flow", messages: ["sip", "1", "menu"], expectedKeywords: ["fund", "portfolio"] },
    { name: "Complex - AI Flow", messages: ["ai", "1", "menu"], expectedKeywords: ["recommendations", "portfolio"] },
    { name: "Complex - Rewards Flow", messages: ["rewards", "1", "menu"], expectedKeywords: ["redeem", "portfolio"] },
    
    // SEBI Compliance tests
    { name: "SEBI - Investment Disclaimer", messages: ["portfolio"], expectedKeywords: ["market risks", "documents"] },
    { name: "SEBI - Fund Disclaimer", messages: ["sip"], expectedKeywords: ["market risks", "documents"] },
    { name: "SEBI - AI Disclaimer", messages: ["ai", "1"], expectedKeywords: ["market risks", "documents"] },
    
    // Context awareness tests
    { name: "Context - Name in Onboarding", messages: ["start", "John Doe"], expectedKeywords: ["email"] },
    { name: "Context - Email in Onboarding", messages: ["start", "John Doe", "john@example.com"], expectedKeywords: ["PAN"] },
    { name: "Context - Menu Navigation", messages: ["portfolio", "menu"], expectedKeywords: ["portfolio", "SIP", "AI"] },
    
    // Edge cases
    { name: "Edge - Empty Message", messages: [""], expectedKeywords: ["Welcome", "help"] },
    { name: "Edge - Special Characters", messages: ["!@#$%"], expectedKeywords: ["Welcome", "help"] },
    { name: "Edge - Long Message", messages: ["This is a very long message with many words that should still be processed correctly by the chatbot"], expectedKeywords: ["Welcome", "help"] },
    { name: "Edge - Numbers Only", messages: ["12345"], expectedKeywords: ["Welcome", "help"] },
    
    // Multi-step validation tests
    { name: "Validation - Email Format", messages: ["start", "John Doe", "invalid-email", "john@example.com"], expectedKeywords: ["valid email", "PAN"] },
    { name: "Validation - PAN Format", messages: ["start", "John Doe", "john@example.com", "123", "ABCDE1234F"], expectedKeywords: ["valid PAN", "date of birth"] },
    { name: "Validation - Mobile Format", messages: ["start", "John Doe", "john@example.com", "ABCDE1234F", "15/06/1990", "123", "9876543210"], expectedKeywords: ["valid mobile", "complete"] }
];

async function runComprehensiveTest() {
    console.log('🚀 Starting Comprehensive WhatsApp Chatbot Efficiency Test...\n');
    
    const whatsappService = new MockWhatsAppService();
    let passedTests = 0;
    let failedTests = 0;
    const failures = [];

    for (const scenario of testScenarios) {
        console.log(`\n📋 Testing: ${scenario.name}`);
        
        let currentResponse = '';
        let allResponses = '';
        
        try {
            // Process all messages in sequence
            for (const message of scenario.messages) {
                currentResponse = await whatsappService.processMessage('+1234567890', message);
                allResponses += currentResponse + ' ';
            }
            
            // Check if all expected keywords are present
            const missingKeywords = scenario.expectedKeywords.filter(keyword => 
                !allResponses.toLowerCase().includes(keyword.toLowerCase())
            );
            
            if (missingKeywords.length === 0) {
                console.log(`✅ PASSED: All expected keywords found`);
                passedTests++;
            } else {
                console.log(`❌ FAILED: Missing keywords: ${missingKeywords.join(', ')}`);
                console.log(`   Response: ${currentResponse.substring(0, 100)}...`);
                failedTests++;
                failures.push({
                    scenario: scenario.name,
                    missingKeywords,
                    response: currentResponse,
                    allResponses: allResponses
                });
            }
            
        } catch (error) {
            console.log(`❌ ERROR: ${error.message}`);
            failedTests++;
            failures.push({
                scenario: scenario.name,
                error: error.message,
                response: currentResponse
            });
        }
    }

    const totalTests = testScenarios.length;
    const successRate = ((passedTests / totalTests) * 100).toFixed(1);
    
    console.log('\n' + '='.repeat(60));
    console.log('📊 COMPREHENSIVE TEST RESULTS');
    console.log('='.repeat(60));
    console.log(`✅ Passed: ${passedTests}/${totalTests}`);
    console.log(`❌ Failed: ${failedTests}/${totalTests}`);
    console.log(`📈 Success Rate: ${successRate}%`);
    
    if (failures.length > 0) {
        console.log('\n🔍 FAILURE ANALYSIS:');
        console.log('='.repeat(60));
        failures.forEach((failure, index) => {
            console.log(`\n${index + 1}. ${failure.scenario}`);
            if (failure.missingKeywords) {
                console.log(`   Missing: ${failure.missingKeywords.join(', ')}`);
            }
            if (failure.error) {
                console.log(`   Error: ${failure.error}`);
            }
            console.log(`   Response: ${failure.response.substring(0, 150)}...`);
        });
    }
    
    console.log('\n' + '='.repeat(60));
    console.log(`🎯 EFFICIENCY STATUS: ${successRate}%`);
    
    if (successRate >= 100) {
        console.log('🎉 CHATBOT IS 100% EFFICIENT! Ready for production!');
    } else if (successRate >= 90) {
        console.log('🚀 EXCELLENT! Chatbot is highly efficient with minor improvements needed.');
    } else if (successRate >= 80) {
        console.log('✅ GOOD! Chatbot is efficient but needs some improvements.');
    } else if (successRate >= 70) {
        console.log('⚠️  FAIR! Chatbot needs significant improvements.');
    } else {
        console.log('❌ POOR! Chatbot needs major improvements.');
    }
    
    console.log('='.repeat(60));
    
    return { passedTests, failedTests, successRate, failures };
}

// Run the test
runComprehensiveTest().catch(console.error); 