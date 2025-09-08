const ASIWhatsAppService = require('./src/services/ASIWhatsAppService');
const fs = require('fs');
const path = require('path');

console.log('🧪 SIP BREWERY ASI WHATSAPP INTEGRATION TEST SUITE');
console.log('📱 Testing Complete Platform Operations via WhatsApp');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

class ASIWhatsAppTestSuite {
    constructor() {
        this.asiWhatsAppService = new ASIWhatsAppService();
        this.testPhoneNumber = '919876543210';
        this.testResults = [];
        this.startTime = Date.now();
    }

    async runAllTests() {
        console.log('🚀 Starting comprehensive ASI WhatsApp integration tests...\n');

        try {
            // Test 1: Service Initialization
            await this.testServiceInitialization();

            // Test 2: Intent Detection
            await this.testIntentDetection();

            // Test 3: User Onboarding Flow
            await this.testUserOnboardingFlow();

            // Test 4: Investment Operations
            await this.testInvestmentOperations();

            // Test 5: ASI Analysis
            await this.testASIAnalysis();

            // Test 6: Portfolio Management
            await this.testPortfolioManagement();

            // Test 7: Report Generation
            await this.testReportGeneration();

            // Test 8: Market Insights
            await this.testMarketInsights();

            // Test 9: Error Handling
            await this.testErrorHandling();

            // Test 10: Session Management
            await this.testSessionManagement();

            // Generate test report
            await this.generateTestReport();

        } catch (error) {
            console.error('❌ Test suite failed:', error);
            this.addTestResult('TEST_SUITE_EXECUTION', false, `Test suite execution failed: ${error.message}`);
        }
    }

    async testServiceInitialization() {
        console.log('🔧 Testing service initialization...');
        
        try {
            // Test service instantiation
            const service = new ASIWhatsAppService();
            this.addTestResult('SERVICE_INSTANTIATION', true, 'Service instantiated successfully');

            // Test required dependencies
            const hasGeminiClient = !!service.geminiClient;
            const hasWhatsAppClient = !!service.whatsAppClient;
            const hasReportSuite = !!service.reportSuite;

            this.addTestResult('GEMINI_CLIENT', hasGeminiClient, hasGeminiClient ? 'Gemini client initialized' : 'Gemini client missing');
            this.addTestResult('WHATSAPP_CLIENT', hasWhatsAppClient, hasWhatsAppClient ? 'WhatsApp client initialized' : 'WhatsApp client missing');
            this.addTestResult('REPORT_SUITE', hasReportSuite, hasReportSuite ? 'Report suite initialized' : 'Report suite missing');

            console.log('✅ Service initialization tests completed\n');

        } catch (error) {
            this.addTestResult('SERVICE_INITIALIZATION', false, `Service initialization failed: ${error.message}`);
            console.log('❌ Service initialization tests failed\n');
        }
    }

    async testIntentDetection() {
        console.log('🧠 Testing intent detection...');

        const testCases = [
            { message: 'Hello', expectedIntent: 'GREETING' },
            { message: 'I want to invest', expectedIntent: 'INVESTMENT' },
            { message: 'Show my portfolio', expectedIntent: 'PORTFOLIO_VIEW' },
            { message: 'Generate reports', expectedIntent: 'REPORTS' },
            { message: 'ASI analysis', expectedIntent: 'ASI_ANALYSIS' },
            { message: 'Market insights', expectedIntent: 'MARKET_INSIGHTS' },
            { message: 'Help me', expectedIntent: 'HELP' },
            { message: 'Random gibberish xyz', expectedIntent: 'UNKNOWN' }
        ];

        for (const testCase of testCases) {
            try {
                const result = await this.asiWhatsAppService.processMessage(
                    this.testPhoneNumber, 
                    testCase.message, 
                    `test-${Date.now()}`
                );

                const intentMatches = result.intent === testCase.expectedIntent;
                this.addTestResult(
                    `INTENT_${testCase.expectedIntent}`, 
                    intentMatches, 
                    `Message: "${testCase.message}" -> Intent: ${result.intent} (Expected: ${testCase.expectedIntent})`
                );

                if (intentMatches) {
                    console.log(`✅ "${testCase.message}" -> ${result.intent}`);
                } else {
                    console.log(`❌ "${testCase.message}" -> ${result.intent} (Expected: ${testCase.expectedIntent})`);
                }

            } catch (error) {
                this.addTestResult(`INTENT_${testCase.expectedIntent}`, false, `Intent detection failed: ${error.message}`);
                console.log(`❌ Intent detection failed for: "${testCase.message}"`);
            }
        }

        console.log('✅ Intent detection tests completed\n');
    }

    async testUserOnboardingFlow() {
        console.log('👤 Testing user onboarding flow...');

        try {
            const testPhone = '919876543211'; // Different number for onboarding test

            // Step 1: Initial greeting
            let result = await this.asiWhatsAppService.processMessage(testPhone, 'Hello', `onboard-1-${Date.now()}`);
            this.addTestResult('ONBOARDING_GREETING', result.success, `Greeting response: ${result.success ? 'Success' : result.error}`);

            // Step 2: Signup initiation
            result = await this.asiWhatsAppService.processMessage(testPhone, 'I want to sign up', `onboard-2-${Date.now()}`);
            this.addTestResult('ONBOARDING_SIGNUP', result.success, `Signup initiation: ${result.success ? 'Success' : result.error}`);

            // Step 3: Personal info collection
            result = await this.asiWhatsAppService.processMessage(testPhone, 'John Doe', `onboard-3-${Date.now()}`);
            this.addTestResult('ONBOARDING_NAME', result.success, `Name collection: ${result.success ? 'Success' : result.error}`);

            // Step 4: Email collection
            result = await this.asiWhatsAppService.processMessage(testPhone, 'john.doe@example.com', `onboard-4-${Date.now()}`);
            this.addTestResult('ONBOARDING_EMAIL', result.success, `Email collection: ${result.success ? 'Success' : result.error}`);

            console.log('✅ User onboarding flow tests completed\n');

        } catch (error) {
            this.addTestResult('ONBOARDING_FLOW', false, `Onboarding flow failed: ${error.message}`);
            console.log('❌ User onboarding flow tests failed\n');
        }
    }

    async testInvestmentOperations() {
        console.log('💰 Testing investment operations...');

        try {
            // Test investment intent
            let result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'I want to invest 5000 in equity funds', 
                `invest-1-${Date.now()}`
            );
            this.addTestResult('INVESTMENT_INTENT', result.success, `Investment intent: ${result.success ? 'Success' : result.error}`);

            // Test SIP creation
            result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'Create SIP for 2000 monthly', 
                `invest-2-${Date.now()}`
            );
            this.addTestResult('SIP_CREATION', result.success, `SIP creation: ${result.success ? 'Success' : result.error}`);

            // Test fund recommendation
            result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'Recommend best funds for me', 
                `invest-3-${Date.now()}`
            );
            this.addTestResult('FUND_RECOMMENDATION', result.success, `Fund recommendation: ${result.success ? 'Success' : result.error}`);

            console.log('✅ Investment operations tests completed\n');

        } catch (error) {
            this.addTestResult('INVESTMENT_OPERATIONS', false, `Investment operations failed: ${error.message}`);
            console.log('❌ Investment operations tests failed\n');
        }
    }

    async testASIAnalysis() {
        console.log('🎯 Testing ASI analysis...');

        try {
            // Test ASI portfolio analysis
            let result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'Analyze my portfolio with ASI', 
                `asi-1-${Date.now()}`
            );
            this.addTestResult('ASI_PORTFOLIO_ANALYSIS', result.success, `ASI portfolio analysis: ${result.success ? 'Success' : result.error}`);

            // Test ASI fund scoring
            result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'What is ASI score of my funds', 
                `asi-2-${Date.now()}`
            );
            this.addTestResult('ASI_FUND_SCORING', result.success, `ASI fund scoring: ${result.success ? 'Success' : result.error}`);

            // Test ASI recommendations
            result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'Give me ASI recommendations', 
                `asi-3-${Date.now()}`
            );
            this.addTestResult('ASI_RECOMMENDATIONS', result.success, `ASI recommendations: ${result.success ? 'Success' : result.error}`);

            console.log('✅ ASI analysis tests completed\n');

        } catch (error) {
            this.addTestResult('ASI_ANALYSIS', false, `ASI analysis failed: ${error.message}`);
            console.log('❌ ASI analysis tests failed\n');
        }
    }

    async testPortfolioManagement() {
        console.log('📊 Testing portfolio management...');

        try {
            // Test portfolio view
            let result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'Show my portfolio', 
                `portfolio-1-${Date.now()}`
            );
            this.addTestResult('PORTFOLIO_VIEW', result.success, `Portfolio view: ${result.success ? 'Success' : result.error}`);

            // Test SIP management
            result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'Manage my SIPs', 
                `portfolio-2-${Date.now()}`
            );
            this.addTestResult('SIP_MANAGEMENT', result.success, `SIP management: ${result.success ? 'Success' : result.error}`);

            // Test withdrawal
            result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'I want to withdraw 10000', 
                `portfolio-3-${Date.now()}`
            );
            this.addTestResult('WITHDRAWAL', result.success, `Withdrawal: ${result.success ? 'Success' : result.error}`);

            console.log('✅ Portfolio management tests completed\n');

        } catch (error) {
            this.addTestResult('PORTFOLIO_MANAGEMENT', false, `Portfolio management failed: ${error.message}`);
            console.log('❌ Portfolio management tests failed\n');
        }
    }

    async testReportGeneration() {
        console.log('📋 Testing report generation...');

        try {
            // Test general report request
            let result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'Generate my reports', 
                `report-1-${Date.now()}`
            );
            this.addTestResult('REPORT_GENERATION', result.success, `Report generation: ${result.success ? 'Success' : result.error}`);

            // Test specific report types
            const reportTypes = [
                'client statement',
                'ASI diagnostic',
                'portfolio allocation',
                'performance benchmark',
                'tax report'
            ];

            for (const reportType of reportTypes) {
                result = await this.asiWhatsAppService.processMessage(
                    this.testPhoneNumber, 
                    `Generate ${reportType} report`, 
                    `report-${reportType}-${Date.now()}`
                );
                this.addTestResult(
                    `REPORT_${reportType.toUpperCase().replace(' ', '_')}`, 
                    result.success, 
                    `${reportType} report: ${result.success ? 'Success' : result.error}`
                );
            }

            console.log('✅ Report generation tests completed\n');

        } catch (error) {
            this.addTestResult('REPORT_GENERATION', false, `Report generation failed: ${error.message}`);
            console.log('❌ Report generation tests failed\n');
        }
    }

    async testMarketInsights() {
        console.log('📈 Testing market insights...');

        try {
            // Test market insights
            let result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'Give me market insights', 
                `market-1-${Date.now()}`
            );
            this.addTestResult('MARKET_INSIGHTS', result.success, `Market insights: ${result.success ? 'Success' : result.error}`);

            // Test market trends
            result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                'What are current market trends', 
                `market-2-${Date.now()}`
            );
            this.addTestResult('MARKET_TRENDS', result.success, `Market trends: ${result.success ? 'Success' : result.error}`);

            console.log('✅ Market insights tests completed\n');

        } catch (error) {
            this.addTestResult('MARKET_INSIGHTS', false, `Market insights failed: ${error.message}`);
            console.log('❌ Market insights tests failed\n');
        }
    }

    async testErrorHandling() {
        console.log('⚠️ Testing error handling...');

        try {
            // Test invalid input
            let result = await this.asiWhatsAppService.processMessage(
                '', // Empty phone number
                'Hello', 
                `error-1-${Date.now()}`
            );
            this.addTestResult('ERROR_INVALID_PHONE', !result.success, `Invalid phone handling: ${!result.success ? 'Handled correctly' : 'Not handled'}`);

            // Test malformed message
            result = await this.asiWhatsAppService.processMessage(
                this.testPhoneNumber, 
                '', // Empty message
                `error-2-${Date.now()}`
            );
            this.addTestResult('ERROR_EMPTY_MESSAGE', !result.success, `Empty message handling: ${!result.success ? 'Handled correctly' : 'Not handled'}`);

            console.log('✅ Error handling tests completed\n');

        } catch (error) {
            this.addTestResult('ERROR_HANDLING', true, `Error handling working: ${error.message}`);
            console.log('✅ Error handling tests completed (caught expected error)\n');
        }
    }

    async testSessionManagement() {
        console.log('🔄 Testing session management...');

        try {
            const testPhone = '919876543212'; // Different number for session test

            // Create session
            const session1 = await this.asiWhatsAppService.getOrCreateSession(testPhone);
            this.addTestResult('SESSION_CREATION', !!session1, `Session creation: ${!!session1 ? 'Success' : 'Failed'}`);

            // Retrieve same session
            const session2 = await this.asiWhatsAppService.getOrCreateSession(testPhone);
            const sameSession = session1.phoneNumber === session2.phoneNumber;
            this.addTestResult('SESSION_RETRIEVAL', sameSession, `Session retrieval: ${sameSession ? 'Same session returned' : 'Different session returned'}`);

            console.log('✅ Session management tests completed\n');

        } catch (error) {
            this.addTestResult('SESSION_MANAGEMENT', false, `Session management failed: ${error.message}`);
            console.log('❌ Session management tests failed\n');
        }
    }

    async generateTestReport() {
        const endTime = Date.now();
        const duration = (endTime - this.startTime) / 1000;
        
        const totalTests = this.testResults.length;
        const passedTests = this.testResults.filter(test => test.passed).length;
        const failedTests = totalTests - passedTests;
        const successRate = ((passedTests / totalTests) * 100).toFixed(2);

        const report = {
            testSuite: 'ASI WhatsApp Integration Test Suite',
            timestamp: new Date().toISOString(),
            duration: `${duration} seconds`,
            summary: {
                totalTests,
                passedTests,
                failedTests,
                successRate: `${successRate}%`
            },
            results: this.testResults,
            environment: {
                nodeVersion: process.version,
                platform: process.platform,
                arch: process.arch
            }
        };

        // Save report to file
        const reportPath = path.join(__dirname, 'asi_whatsapp_test_report.json');
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

        console.log('📊 TEST REPORT SUMMARY');
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        console.log(`📱 Test Suite: ${report.testSuite}`);
        console.log(`⏱️ Duration: ${report.duration}`);
        console.log(`📊 Total Tests: ${totalTests}`);
        console.log(`✅ Passed: ${passedTests}`);
        console.log(`❌ Failed: ${failedTests}`);
        console.log(`📈 Success Rate: ${successRate}%`);
        console.log(`📄 Report saved to: ${reportPath}`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

        // Display failed tests
        if (failedTests > 0) {
            console.log('❌ FAILED TESTS:');
            this.testResults
                .filter(test => !test.passed)
                .forEach(test => {
                    console.log(`   • ${test.testName}: ${test.message}`);
                });
            console.log('');
        }

        console.log('🎉 ASI WhatsApp Integration Test Suite completed!');
        
        return report;
    }

    addTestResult(testName, passed, message) {
        this.testResults.push({
            testName,
            passed,
            message,
            timestamp: new Date().toISOString()
        });
    }
}

// Run the test suite
async function runTests() {
    const testSuite = new ASIWhatsAppTestSuite();
    await testSuite.runAllTests();
}

// Execute if run directly
if (require.main === module) {
    runTests().catch(console.error);
}

module.exports = ASIWhatsAppTestSuite;
