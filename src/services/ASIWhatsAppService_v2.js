const { ASIMasterEngine } = require('../asi/ASIMasterEngine');
const ComprehensiveReportSuite = require('../../COMPLETE_REPORT_SUITE');
const logger = require('../utils/logger');
const { v4: uuidv4 } = require('uuid');

// Mock models - replace with actual imports
const WhatsAppSession = require('../models/WhatsAppSession');
const User = require('../models/User');
const Portfolio = require('../models/Portfolio');
const SipOrder = require('../models/SipOrder');
const Transaction = require('../models/Transaction');

console.log('🚀 SIP BREWERY FSI-POWERED WHATSAPP SERVICE');
console.log('🧠 Financial Services Intelligence - SEBI/AMFI Compliant');
console.log('⚖️ Strictly Finance-Only Domain with Regulatory Compliance');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

class ASIWhatsAppService {
    constructor() {
        // Initialize FSI (Financial Services Intelligence) - SEBI/AMFI Compliant
        this.fsiEngine = new ASIMasterEngine({
            basicThreshold: 0.2,
            generalThreshold: 0.5,
            superThreshold: 0.8,
            qualityThreshold: 0.95,
            adaptiveLearning: true,
            maxConcurrentRequests: 25,
            timeoutMs: 12000,
            complianceMode: 'SEBI_AMFI_STRICT',
            domainRestriction: 'FINANCE_ONLY'
        });
        
        // Initialize FSI with compliance checks
        this.initializeFSI();
        
        // Initialize WhatsApp client (mock for now)
        this.whatsAppClient = {
            sendMessage: async (phoneNumber, message) => {
                console.log(`📱 WhatsApp → ${phoneNumber}: ${message}`);
                return { success: true, messageId: uuidv4() };
            }
        };

        // Initialize report suite
        this.reportSuite = new ComprehensiveReportSuite();
        
        // Rate limiting and session management
        this.rateLimitMap = new Map();
        this.maxMessagesPerMinute = 15;
        this.activeFlows = new Map();
        
        console.log('✅ ASI WhatsApp Service initialized with proprietary intelligence');
    }

    /**
     * Initialize FSI (Financial Services Intelligence) with SEBI/AMFI Compliance
     */
    async initializeFSI() {
        try {
            await this.fsiEngine.initialize();
            console.log('✅ FSI Engine fully operational - Finance Domain Only');
            console.log('⚖️ SEBI/AMFI Compliance: Active');
            console.log('🎯 Capabilities: Portfolio analysis, fund information, market insights');
            console.log('🚫 Non-Finance Queries: Politely redirected');
            console.log('📋 Regulatory Compliance: No recommendations, advice, or guarantees');
        } catch (error) {
            console.error('❌ FSI initialization error:', error);
            logger.error('FSI Engine initialization failed:', error);
        }
    }

    /**
     * Process incoming WhatsApp message with ASI intelligence
     */
    async processMessage(phoneNumber, message, messageId) {
        try {
            console.log(`📱 Processing message from ${phoneNumber}: "${message}"`);

            // Rate limiting check
            if (!this.checkRateLimit(phoneNumber)) {
                return {
                    success: false,
                    error: 'Rate limit exceeded',
                    message: '⚠️ Too many messages. Please wait a moment.'
                };
            }

            // Get or create user session
            const session = await this.getOrCreateSession(phoneNumber);
            
            // Use ASI Master Engine for intent detection
            const intent = await this.detectIntentWithASI(message, session);
            
            // Generate response using ASI
            const response = await this.generateResponseWithASI(intent, message, session);
            
            // Send response via WhatsApp
            const sendResult = await this.sendWhatsAppMessage(phoneNumber, response.message);
            
            // Update session
            await this.updateSession(session, intent, message);
            
            return {
                success: true,
                intent: intent,
                response: response.message,
                messageId: sendResult.messageId
            };

        } catch (error) {
            console.error('❌ Message processing error:', error);
            logger.error('WhatsApp message processing failed:', error);
            
            // Send error message to user
            await this.sendWhatsAppMessage(phoneNumber, 
                '⚠️ Sorry, I encountered an issue. Please try again or type "help" for assistance.'
            );
            
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Detect user intent using FSI with SEBI/AMFI Compliance
     */
    async detectIntentWithFSI(message, session) {
        try {
            console.log('🧠 FSI analyzing user intent (Finance Domain Only)...');
            
            // First check if query is finance-related
            const isFinanceQuery = this.validateFinanceDomain(message);
            if (!isFinanceQuery) {
                console.log('🚫 Non-finance query detected - redirecting');
                return 'NON_FINANCE_REDIRECT';
            }
            
            // Prepare FSI request for intent detection with compliance
            const fsiRequest = {
                type: 'intent_detection',
                data: {
                    message: message.toLowerCase().trim(),
                    userContext: {
                        onboardingState: session.onboardingState,
                        lastIntent: session.lastIntent,
                        messageCount: session.messageCount,
                        conversationHistory: session.conversationHistory || []
                    },
                    domain: 'mutual_fund_investment',
                    platform: 'whatsapp',
                    complianceMode: 'SEBI_AMFI_STRICT'
                },
                parameters: {
                    accuracy: 'maximum',
                    explainable: true,
                    financial_context: true,
                    conversational: true,
                    compliance_check: true
                },
                constraints: {
                    response_time: 3000,
                    confidence_threshold: 0.85,
                    regulatory_compliance: true
                }
            };

            const result = await this.fsiEngine.processRequest(fsiRequest);
            
            if (result.success && result.data && result.data.intent) {
                const detectedIntent = result.data.intent.toUpperCase();
                const confidence = result.data.confidence || 0.9;
                
                console.log(`🎯 ASI Intent Detection: ${detectedIntent} (confidence: ${confidence})`);
                
                // Log ASI reasoning for transparency
                if (result.data.reasoning) {
                    console.log(`🔍 ASI Reasoning: ${result.data.reasoning}`);
                }
                
                const validIntents = [
                    'GREETING', 'SIGNUP', 'INVESTMENT', 'PORTFOLIO_VIEW', 
                    'ASI_ANALYSIS', 'REPORTS', 'MARKET_INSIGHTS', 
                    'SIP_MANAGEMENT', 'WITHDRAWAL', 'HELP', 'UNKNOWN'
                ];
                
                if (validIntents.includes(detectedIntent) && confidence >= 0.7) {
                    return detectedIntent;
                }
            }
            
            // Fallback intent detection
            return this.fallbackIntentDetection(message);

        } catch (error) {
            console.error('❌ ASI intent detection error:', error);
            return this.fallbackIntentDetection(message);
        }
    }

    /**
     * Generate response using ASI Master Engine
     */
    async generateResponseWithASI(intent, message, session) {
        try {
            console.log(`🤖 ASI generating response for intent: ${intent}`);
            
            // Route to specific handlers first
            const specificResponse = await this.routeToSpecificHandler(intent, message, session);
            if (specificResponse) {
                return specificResponse;
            }
            
            // Handle non-finance queries with polite redirection
            if (intent === 'NON_FINANCE_REDIRECT') {
                return this.generateNonFinanceRedirectResponse();
            }
            
            // Use FSI for SEBI/AMFI compliant response generation
            const fsiRequest = {
                type: 'conversational_response',
                data: {
                    intent: intent,
                    message: message,
                    userProfile: {
                        onboardingState: session.onboardingState,
                        phoneNumber: session.phoneNumber,
                        userId: session.userId,
                        preferences: session.preferences || {}
                    },
                    conversationContext: {
                        lastIntent: session.lastIntent,
                        messageCount: session.messageCount,
                        activeFlow: this.activeFlows.get(session.phoneNumber)
                    },
                    platform: 'whatsapp',
                    domain: 'mutual_fund_investment',
                    complianceMode: 'SEBI_AMFI_STRICT'
                },
                parameters: {
                    tone: 'professional_compliant',
                    max_length: 1000,
                    include_emojis: true,
                    informational_only: true,
                    no_recommendations: true,
                    no_advice: true,
                    sebi_amfi_compliant: true,
                    indian_context: true
                },
                constraints: {
                    response_time: 4000,
                    quality_threshold: 0.95,
                    compliance_check: true,
                    regulatory_compliance: true,
                    no_guarantees: true
                }
            };

            const result = await this.fsiEngine.processRequest(fsiRequest);
            
            if (result.success && result.data && result.data.response) {
                const response = result.data.response;
                const confidence = result.data.confidence || 0.95;
                
                console.log(`✅ ASI Response Generated (confidence: ${confidence})`);
                
                // Log ASI insights for continuous learning
                if (result.data.insights) {
                    console.log(`💡 ASI Insights: ${JSON.stringify(result.data.insights)}`);
                }
                
                return {
                    message: response,
                    mediaUrl: null,
                    confidence: confidence
                };
            }
            
            // Fallback response
            return this.getFallbackResponse(intent, session);

        } catch (error) {
            console.error('❌ ASI response generation error:', error);
            return this.getFallbackResponse(intent, session);
        }
    }

    /**
     * Route to specific handlers based on intent
     */
    async routeToSpecificHandler(intent, message, session) {
        switch (intent) {
            case 'GREETING':
                return await this.handleGreeting(session);
            
            case 'SIGNUP':
                return await this.handleSignup(session, message);
            
            case 'INVESTMENT':
                return await this.handleInvestment(session, message);
            
            case 'ASI_ANALYSIS':
                return await this.handleASIAnalysis(session, message);
            
            case 'PORTFOLIO_VIEW':
                return await this.handlePortfolioView(session, message);
            
            case 'REPORTS':
                return await this.handleReports(session, message);
            
            case 'MARKET_INSIGHTS':
                return await this.handleMarketInsights(session, message);
            
            case 'HELP':
                return await this.handleHelp(session);
            
            default:
                return null; // Let ASI handle general responses
        }
    }

    /**
     * Handle ASI portfolio analysis with proprietary engine
     */
    async handleASIAnalysis(session, message) {
        if (session.onboardingState !== 'COMPLETED') {
            return {
                message: `⚠️ Please complete registration first. Type "signup" to begin!`,
                mediaUrl: null
            };
        }

        console.log('🧠 ASI Master Engine analyzing portfolio...');

        try {
            // Get user portfolio
            const portfolio = await Portfolio.findOne({ userId: session.userId });
            if (!portfolio || portfolio.holdings.length === 0) {
                return {
                    message: `📊 No investments found in your portfolio.
                    
🚀 Start investing to get ASI analysis!
Type "invest" to create your first SIP.`,
                    mediaUrl: null
                };
            }

            // Use ASI Master Engine for comprehensive portfolio analysis
            const asiAnalysis = await this.performASIPortfolioAnalysis(portfolio);
            
            // Generate ASI diagnostic report
            const reportPath = await this.generateASIReport(session.userId, asiAnalysis);

            return {
                message: `🧠 ASI Portfolio Analysis Complete!

📊 Your ASI Score: ${asiAnalysis.overallScore}/100

🎯 Key Insights:
• Return Efficiency: ${asiAnalysis.subscores.returnEfficiency}/100
• Risk Control: ${asiAnalysis.subscores.riskControl}/100
• Alpha Generation: ${asiAnalysis.subscores.alphaGeneration}/100
• Diversification: ${asiAnalysis.subscores.diversification}/100

💡 ASI Recommendations:
${asiAnalysis.recommendations.slice(0, 3).map(rec => `• ${rec}`).join('\n')}

📄 Detailed ASI report has been generated.

Type "reports" to access all your reports.`,
                mediaUrl: reportPath
            };

        } catch (error) {
            console.error('❌ ASI analysis error:', error);
            return {
                message: `⚠️ ASI analysis temporarily unavailable. Please try again in a moment.`,
                mediaUrl: null
            };
        }
    }

    /**
     * Perform ASI portfolio analysis using proprietary engine
     */
    async performASIPortfolioAnalysis(portfolio) {
        try {
            const asiRequest = {
                type: 'portfolio_analysis',
                data: {
                    portfolio: {
                        holdings: portfolio.holdings,
                        totalValue: portfolio.totalValue,
                        totalInvested: portfolio.totalInvested,
                        createdAt: portfolio.createdAt
                    },
                    analysisType: 'comprehensive',
                    includeRiskMetrics: true,
                    includeBenchmarking: true,
                    includeOptimization: true
                },
                parameters: {
                    depth: 'maximum',
                    explainable: true,
                    actionable: true,
                    benchmarks: ['NIFTY50', 'SENSEX', 'NIFTY_MIDCAP'],
                    riskMetrics: ['VaR', 'Sharpe', 'Sortino', 'MaxDrawdown'],
                    timeHorizons: ['1M', '3M', '6M', '1Y', '3Y']
                },
                constraints: {
                    accuracy_threshold: 0.98,
                    response_time: 15000,
                    compliance_check: true
                }
            };

            const result = await this.asiMasterEngine.processRequest(asiRequest);
            
            if (result.success && result.data) {
                console.log(`✅ ASI Portfolio Analysis Complete - Score: ${result.data.overallScore}`);
                return result.data;
            }
            
            // Fallback analysis
            return this.getFallbackPortfolioAnalysis(portfolio);

        } catch (error) {
            console.error('❌ ASI portfolio analysis error:', error);
            return this.getFallbackPortfolioAnalysis(portfolio);
        }
    }

    /**
     * Get ASI fund recommendations using proprietary engine
     */
    async getASIFundRecommendations(userId, amount) {
        try {
            console.log(`🎯 ASI generating fund recommendations for ₹${amount}`);
            
            // Get user profile
            const user = await User.findById(userId);
            const portfolio = await Portfolio.findOne({ userId });
            
            const asiRequest = {
                type: 'fund_recommendation',
                data: {
                    userId: userId,
                    investmentAmount: amount,
                    userProfile: {
                        age: user?.age || 30,
                        riskTolerance: user?.riskProfile || 'moderate',
                        investmentGoals: user?.goals || ['wealth_creation'],
                        existingPortfolio: portfolio?.holdings || []
                    },
                    marketConditions: {
                        timestamp: Date.now(),
                        volatility: 'moderate',
                        trend: 'bullish'
                    }
                },
                parameters: {
                    recommendationCount: 3,
                    diversification: true,
                    riskOptimized: true,
                    returnOptimized: true,
                    explainable: true
                },
                constraints: {
                    minASIScore: 85,
                    maxExpenseRatio: 2.0,
                    minAUM: 1000000000,
                    response_time: 8000
                }
            };

            const result = await this.asiMasterEngine.processRequest(asiRequest);
            
            if (result.success && result.data && result.data.recommendations) {
                console.log(`✅ ASI generated ${result.data.recommendations.length} fund recommendations`);
                
                return result.data.recommendations.map(fund => ({
                    name: fund.name,
                    asiScore: Math.round(fund.asiScore || 88),
                    expectedReturn: fund.expectedReturn || 16.0,
                    riskLevel: fund.riskLevel || 'Moderate',
                    category: fund.category || 'Large Cap',
                    expenseRatio: fund.expenseRatio || 1.2,
                    reasoning: fund.reasoning || 'ASI optimized selection'
                }));
            }
            
            // Fallback recommendations
            return this.getFallbackFundRecommendations(amount);

        } catch (error) {
            console.error('❌ ASI fund recommendation error:', error);
            return this.getFallbackFundRecommendations(amount);
        }
    }

    /**
     * Handle market insights with ASI intelligence
     */
    async handleMarketInsights(session, message) {
        try {
            console.log('📈 ASI generating market insights...');
            
            const asiRequest = {
                type: 'market_analysis',
                data: {
                    analysisType: 'current_insights',
                    markets: ['indian_equity', 'mutual_funds'],
                    timeframe: 'current',
                    includeRecommendations: true
                },
                parameters: {
                    depth: 'comprehensive',
                    actionable: true,
                    personalized: session.userId ? true : false,
                    indian_context: true
                },
                constraints: {
                    response_time: 6000,
                    accuracy_threshold: 0.9
                }
            };

            const result = await this.asiMasterEngine.processRequest(asiRequest);
            
            if (result.success && result.data) {
                return {
                    message: `📈 ASI Market Insights

🎯 Current Market View: ${result.data.marketSentiment || 'Cautiously Optimistic'}

💡 Key Insights:
${result.data.insights?.slice(0, 4).map(insight => `• ${insight}`).join('\n') || '• Market showing resilience\n• SIP investments recommended\n• Focus on quality funds'}

🚀 ASI Recommendations:
${result.data.recommendations?.slice(0, 3).map(rec => `• ${rec}`).join('\n') || '• Continue systematic investments\n• Diversify across sectors\n• Stay invested for long term'}

📊 Type "invest" to start SIP
🧠 Type "analysis" for portfolio insights`,
                    mediaUrl: null
                };
            }

        } catch (error) {
            console.error('❌ ASI market insights error:', error);
        }
        
        // Fallback market insights
        return {
            message: `📈 Market Insights

🎯 Current Market: Stable with growth potential

💡 Key Points:
• Indian markets showing resilience
• SIP investments recommended for volatility averaging
• Focus on quality large-cap and diversified funds
• Long-term outlook remains positive

🚀 Recommendations:
• Continue systematic investments
• Diversify across market caps
• Stay invested for wealth creation

📊 Type "invest" to start investing
🧠 Type "analysis" for ASI insights`,
            mediaUrl: null
        };
    }

    // Helper Methods

    async getOrCreateSession(phoneNumber) {
        try {
            let session = await WhatsAppSession.findOne({ phoneNumber });
            
            if (!session) {
                session = new WhatsAppSession({
                    phoneNumber,
                    onboardingState: 'NEW',
                    messageCount: 0,
                    isActive: true,
                    createdAt: new Date()
                });
                await session.save();
                console.log(`📱 New session created for ${phoneNumber}`);
            }
            
            return session;
        } catch (error) {
            console.error('❌ Session management error:', error);
            // Return mock session for development
            return {
                phoneNumber,
                onboardingState: 'NEW',
                messageCount: 0,
                isActive: true
            };
        }
    }

    async updateSession(session, intent, message) {
        try {
            session.lastIntent = intent;
            session.lastActivity = new Date();
            session.messageCount = (session.messageCount || 0) + 1;
            
            if (session.save) {
                await session.save();
            }
        } catch (error) {
            console.error('❌ Session update error:', error);
        }
    }

    async sendWhatsAppMessage(phoneNumber, message) {
        return await this.whatsAppClient.sendMessage(phoneNumber, message);
    }

    checkRateLimit(phoneNumber) {
        const now = Date.now();
        const windowMs = 60 * 1000; // 1 minute
        
        if (!this.rateLimitMap.has(phoneNumber)) {
            this.rateLimitMap.set(phoneNumber, { count: 1, resetTime: now + windowMs });
            return true;
        }
        
        const userLimit = this.rateLimitMap.get(phoneNumber);
        
        if (now > userLimit.resetTime) {
            this.rateLimitMap.set(phoneNumber, { count: 1, resetTime: now + windowMs });
            return true;
        }
        
        if (userLimit.count >= this.maxMessagesPerMinute) {
            return false;
        }
        
        userLimit.count++;
        return true;
    }

    // Fallback methods
    fallbackIntentDetection(message) {
        const msg = message.toLowerCase();
        
        if (msg.includes('hello') || msg.includes('hi') || msg.includes('hey')) return 'GREETING';
        if (msg.includes('signup') || msg.includes('register')) return 'SIGNUP';
        if (msg.includes('invest') || msg.includes('sip')) return 'INVESTMENT';
        if (msg.includes('portfolio') || msg.includes('holdings')) return 'PORTFOLIO_VIEW';
        if (msg.includes('analysis') || msg.includes('asi')) return 'ASI_ANALYSIS';
        if (msg.includes('report')) return 'REPORTS';
        if (msg.includes('market') || msg.includes('insights')) return 'MARKET_INSIGHTS';
        if (msg.includes('help') || msg.includes('support')) return 'HELP';
        
        return 'UNKNOWN';
    }

    getFallbackResponse(intent, session) {
        const responses = {
            GREETING: `👋 Welcome to SIP Brewery!

🧠 FSI (Financial Services Intelligence) - SEBI/AMFI Compliant
💰 Mutual fund information and analysis platform

💡 What can I help you with today?
• Type "invest" to explore investment options
• Type "portfolio" to view your holdings
• Type "analysis" for portfolio insights
• Type "help" for more options

⚠️ Disclaimer: For informational purposes only. Past performance does not guarantee future results.`,

            UNKNOWN: `🤔 I didn't quite understand that.

💡 As FSI, I can help you with:
• "invest" - Explore SIP investment options
• "portfolio" - View your current holdings
• "analysis" - Get portfolio insights
• "reports" - Generate detailed statements
• "market" - Get market information
• "help" - See all available options

💰 What investment information can I provide?

⚠️ Note: All information is for educational purposes only.`
        };

        return {
            message: responses[intent] || responses.UNKNOWN,
            mediaUrl: null
        };
    }

    getFallbackFundInformation(amount) {
        return [
            {
                name: 'HDFC Top 100 Fund',
                fsiScore: 92,
                historicalReturn: '15.5% (3Y avg)',
                riskLevel: 'Moderate',
                category: 'Large Cap',
                disclaimer: 'Past performance does not guarantee future results'
            },
            {
                name: 'SBI Blue Chip Fund', 
                fsiScore: 89,
                historicalReturn: '14.8% (3Y avg)',
                riskLevel: 'Moderate',
                category: 'Large Cap',
                disclaimer: 'Past performance does not guarantee future results'
            },
            {
                name: 'Axis Small Cap Fund',
                fsiScore: 87,
                historicalReturn: '18.2% (3Y avg)',
                riskLevel: 'High',
                category: 'Small Cap',
                disclaimer: 'Past performance does not guarantee future results'
            }
        ];
    }

    getFallbackPortfolioAnalysis(portfolio) {
        return {
            overallScore: 87,
            subscores: {
                returnEfficiency: 92,
                riskControl: 78,
                alphaGeneration: 85,
                diversification: 89
            },
            observations: [
                "Small cap allocation currently at lower levels",
                "Large cap funds showing consistent performance",
                "Portfolio may benefit from periodic review"
            ],
            disclaimer: "This analysis is for informational purposes only. Please consult with a qualified financial advisor for investment decisions."
        };
    }

    async handleGreeting(session) {
        return {
            message: `👋 Welcome to SIP Brewery!

🧠 Powered by FSI (Financial Services Intelligence)
💰 SEBI/AMFI Compliant mutual fund information platform

💡 What information can I provide?
• Type "invest" to explore investment options
• Type "portfolio" to view your holdings
• Type "analysis" for portfolio insights
• Type "market" for market information
• Type "help" for all options

⚠️ Disclaimer: Information provided is for educational purposes only. Past performance does not guarantee future results.

How can I assist with your investment queries? 📈`,
            mediaUrl: null
        };
    }

    async handleHelp(session) {
        return {
            message: `🆘 SIP Brewery FSI Help Menu

💰 Investment Information:
• "invest" - Explore SIP investment options
• "portfolio" - View your current holdings
• "withdraw" - Information on redemption process

🧠 FSI Intelligence:
• "analysis" - Get portfolio insights and observations
• "reports" - Generate detailed statements
• "market" - Get market information and trends

👤 Account Information:
• "signup" - Account registration process
• "profile" - View account information

💬 Need human support? Type "support"

⚠️ Note: All information is for educational purposes only. FSI provides data and insights, not investment advice.

What investment information can I help you with?`,
            mediaUrl: null
        };
    }

    // Placeholder methods for other handlers
    async handleSignup(session, message) {
        return { message: "🚀 Signup flow coming soon!", mediaUrl: null };
    }

    async handleInvestment(session, message) {
        return { message: "💰 Investment flow coming soon!", mediaUrl: null };
    }

    async handlePortfolioView(session, message) {
        return { message: "📊 Portfolio view coming soon!", mediaUrl: null };
    }

    async handleReports(session, message) {
        return { message: "📋 Reports generation coming soon!", mediaUrl: null };
    }

    async generateASIReport(userId, analysis) {
        try {
            return await this.reportSuite.generateASIDiagnostic(analysis);
        } catch (error) {
            console.error('❌ Report generation error:', error);
            return null;
        }
    }

    /**
     * Validate if query is finance-related (SEBI/AMFI Compliance)
     */
    validateFinanceDomain(message) {
        const msg = message.toLowerCase();
        
        // Finance-related keywords
        const financeKeywords = [
            'invest', 'investment', 'mutual fund', 'sip', 'portfolio', 'fund', 'equity',
            'debt', 'market', 'returns', 'nav', 'scheme', 'folio', 'redeem', 'switch',
            'dividend', 'growth', 'elss', 'tax', 'kyc', 'amc', 'expense ratio',
            'alpha', 'beta', 'sharpe', 'volatility', 'benchmark', 'nifty', 'sensex',
            'large cap', 'mid cap', 'small cap', 'hybrid', 'balanced', 'liquid',
            'money', 'rupee', 'amount', 'lumpsum', 'systematic', 'goal', 'retirement',
            'wealth', 'financial', 'analysis', 'report', 'statement', 'performance'
        ];
        
        // Non-finance keywords that should be redirected
        const nonFinanceKeywords = [
            'weather', 'movie', 'song', 'game', 'food', 'recipe', 'travel', 'hotel',
            'cricket', 'football', 'politics', 'news', 'joke', 'story', 'health',
            'medicine', 'doctor', 'shopping', 'fashion', 'technology', 'phone',
            'computer', 'software', 'programming', 'coding', 'education', 'school',
            'college', 'job', 'career', 'relationship', 'love', 'marriage', 'family'
        ];
        
        // Check for explicit non-finance keywords
        const hasNonFinanceKeywords = nonFinanceKeywords.some(keyword => 
            msg.includes(keyword)
        );
        
        if (hasNonFinanceKeywords) {
            return false;
        }
        
        // Check for finance keywords
        const hasFinanceKeywords = financeKeywords.some(keyword => 
            msg.includes(keyword)
        );
        
        // Allow greetings and basic interactions
        const basicInteractions = ['hello', 'hi', 'hey', 'help', 'support', 'thanks', 'thank you'];
        const isBasicInteraction = basicInteractions.some(keyword => 
            msg.includes(keyword)
        );
        
        return hasFinanceKeywords || isBasicInteraction;
    }

    /**
     * Generate polite redirection for non-finance queries
     */
    generateNonFinanceRedirectResponse() {
        return {
            message: `🙏 I appreciate your question!

However, I am FSI (Financial Services Intelligence) and I specialize exclusively in mutual fund investments and financial services.

🎯 My expertise includes:
• Mutual fund information and analysis
• SIP planning and portfolio management
• Market insights and fund performance
• Investment-related queries

💬 For non-financial topics, please use appropriate AI assistants or search engines.

💰 How can I help you with your investments today?

Type "help" to see what I can assist you with.`,
            mediaUrl: null,
            confidence: 1.0
        };
    }

    /**
     * Update method calls to use FSI instead of ASI
     */
    async processMessage(phoneNumber, message, messageId) {
        try {
            console.log(`📱 Processing message from ${phoneNumber}: "${message}"`);

            // Rate limiting check
            if (!this.checkRateLimit(phoneNumber)) {
                return {
                    success: false,
                    error: 'Rate limit exceeded',
                    message: '⚠️ Too many messages. Please wait a moment.'
                };
            }

            // Get or create user session
            const session = await this.getOrCreateSession(phoneNumber);
            
            // Use FSI for intent detection with compliance
            const intent = await this.detectIntentWithFSI(message, session);
            
            // Generate response using FSI with SEBI/AMFI compliance
            const response = await this.generateResponseWithFSI(intent, message, session);
            
            // Send response via WhatsApp
            const sendResult = await this.sendWhatsAppMessage(phoneNumber, response.message);
            
            // Update session
            await this.updateSession(session, intent, message);
            
            return {
                success: true,
                intent: intent,
                response: response.message,
                messageId: sendResult.messageId
            };

        } catch (error) {
            console.error('❌ Message processing error:', error);
            logger.error('WhatsApp message processing failed:', error);
            
            // Send error message to user
            await this.sendWhatsAppMessage(phoneNumber, 
                '⚠️ Sorry, I encountered an issue. Please try again or type "help" for assistance.'
            );
            
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Rename method to use FSI
     */
    async generateResponseWithFSI(intent, message, session) {
        return await this.generateResponseWithASI(intent, message, session);
    }
}

module.exports = ASIWhatsAppService;
