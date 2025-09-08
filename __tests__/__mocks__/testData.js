// Mock test data for SipBrewery backend tests
const { ObjectId } = require('mongodb');

const mockUsers = {
  regularUser: {
    _id: new ObjectId(),
    name: 'John Doe',
    email: 'john@example.com',
    phone: '9876543210',
    secretCode: 'USER123',
    role: 'user',
    isActive: true,
    supabaseId: 'test-supabase-id-1'
  },
  
  adminUser: {
    _id: new ObjectId(),
    name: 'Admin User',
    email: 'admin@example.com',
    phone: '9876543211',
    secretCode: 'ADMIN123',
    role: 'admin',
    isActive: true,
    supabaseId: 'test-supabase-id-2'
  },
  
  agentUser: {
    _id: new ObjectId(),
    name: 'Agent User',
    email: 'agent@example.com',
    phone: '9876543212',
    secretCode: 'AGENT123',
    role: 'agent',
    isActive: true,
    supabaseId: 'test-supabase-id-3'
  }
};

const mockPortfolios = {
  basicPortfolio: {
    _id: new ObjectId(),
    userId: new ObjectId(),
    funds: [
      {
        schemeCode: 'AXISBLUECHIP',
        schemeName: 'Axis Bluechip Fund',
        investedValue: 50000,
        currentValue: 55000,
        units: 500,
        lastNav: 110,
        lastNavDate: new Date(),
        startDate: new Date('2024-01-01')
      },
      {
        schemeCode: 'HDFCBANK',
        schemeName: 'HDFC Bank Fund',
        investedValue: 30000,
        currentValue: 31500,
        units: 300,
        lastNav: 105,
        lastNavDate: new Date(),
        startDate: new Date('2024-01-01')
      }
    ],
    totalInvested: 80000,
    totalCurrentValue: 86500,
    xirr1M: 2.5,
    xirr3M: 8.1,
    xirr6M: 12.3,
    xirr1Y: 18.7,
    xirr3Y: 45.2
  },
  
  emptyPortfolio: {
    _id: new ObjectId(),
    userId: new ObjectId(),
    funds: [],
    totalInvested: 0,
    totalCurrentValue: 0,
    xirr1M: 0,
    xirr3M: 0,
    xirr6M: 0,
    xirr1Y: 0,
    xirr3Y: 0
  }
};

const mockFunds = {
  axisBluechip: {
    schemeCode: 'AXISBLUECHIP',
    schemeName: 'Axis Bluechip Fund',
    category: 'Equity',
    nav: 110.50,
    navDate: new Date(),
    expenseRatio: 1.75,
    fundSize: 15000,
    rating: 4.5
  },
  
  hdfcBank: {
    schemeCode: 'HDFCBANK',
    schemeName: 'HDFC Bank Fund',
    category: 'Equity',
    nav: 105.25,
    navDate: new Date(),
    expenseRatio: 1.85,
    fundSize: 12000,
    rating: 4.2
  },
  
  sbiContra: {
    schemeCode: 'SBICONTRA',
    schemeName: 'SBI Contra Fund',
    category: 'Equity',
    nav: 95.80,
    navDate: new Date(),
    expenseRatio: 1.95,
    fundSize: 8000,
    rating: 4.0
  }
};

const mockTransactions = {
  sipTransaction: {
    _id: new ObjectId(),
    transactionId: 'TXN001',
    userId: new ObjectId(),
    schemeCode: 'AXISBLUECHIP',
    schemeName: 'Axis Bluechip Fund',
    transactionType: 'SIP',
    type: 'SIP',
    amount: 5000,
    netAmount: 5000,
    units: 45.25,
    nav: 110.50,
    date: new Date(),
    status: 'SUCCESS',
    orderType: 'SIP',
    folio: 'FOLIO001'
  },
  
  lumpsumTransaction: {
    _id: new ObjectId(),
    transactionId: 'TXN002',
    userId: new ObjectId(),
    schemeCode: 'HDFCBANK',
    schemeName: 'HDFC Bank Fund',
    transactionType: 'LUMPSUM',
    type: 'LUMPSUM',
    amount: 10000,
    netAmount: 10000,
    units: 95.02,
    nav: 105.25,
    date: new Date(),
    status: 'SUCCESS',
    orderType: 'LUMPSUM',
    folio: 'FOLIO002'
  }
};

const mockSmartSips = {
  basicSmartSip: {
    _id: new ObjectId(),
    userId: new ObjectId(),
    sipType: 'SMART',
    averageSip: 5000,
    minSip: 4000,
    maxSip: 6000,
    sipDay: 15,
    fundSelection: [
      {
        schemeCode: 'AXISBLUECHIP',
        schemeName: 'Axis Bluechip Fund',
        allocation: 60
      },
      {
        schemeCode: 'HDFCBANK',
        schemeName: 'HDFC Bank Fund',
        allocation: 40
      }
    ],
    status: 'active',
    isActive: true
  },
  
  staticSip: {
    _id: new ObjectId(),
    userId: new ObjectId(),
    sipType: 'STATIC',
    averageSip: 3000,
    fundSelection: [
      {
        schemeCode: 'AXISBLUECHIP',
        schemeName: 'Axis Bluechip Fund',
        allocation: 100
      }
    ],
    status: 'active',
    isActive: true
  }
};

const mockRewards = {
  sipReward: {
    _id: new ObjectId(),
    userId: new ObjectId(),
    type: 'SIP_COMPLETION',
    points: 100,
    description: 'Completed monthly SIP',
    status: 'earned',
    isPaid: false
  },
  
  referralReward: {
    _id: new ObjectId(),
    userId: new ObjectId(),
    type: 'REFERRAL_BONUS',
    points: 500,
    description: 'Referred a new user',
    status: 'earned',
    isPaid: false
  }
};

const mockLeaderboards = {
  monthlyLeaderboard: {
    _id: new ObjectId(),
    duration: '1M',
    leaders: [
      {
        userId: new ObjectId(),
        secretCode: 'USER123',
        name: 'John Doe',
        returnPercent: 15.5,
        rank: 1
      },
      {
        userId: new ObjectId(),
        secretCode: 'USER456',
        name: 'Jane Smith',
        returnPercent: 12.3,
        rank: 2
      }
    ],
    isActive: true,
    generatedAt: new Date()
  }
};

const mockWhatsAppMessages = {
  sipOrder: {
    _id: new ObjectId(),
    messageId: 'MSG001',
    phoneNumber: '9876543210',
    userId: new ObjectId(),
    message: 'Buy 1000 units of Axis Bluechip Fund',
    detectedIntent: 'SIP_ORDER',
    aiGenerated: false,
    timestamp: new Date()
  },
  
  portfolioQuery: {
    _id: new ObjectId(),
    messageId: 'MSG002',
    phoneNumber: '9876543210',
    userId: new ObjectId(),
    message: 'Show my portfolio performance',
    detectedIntent: 'PORTFOLIO_QUERY',
    aiGenerated: false,
    timestamp: new Date()
  }
};

const mockAIInsights = {
  portfolioInsight: {
    _id: new ObjectId(),
    userId: new ObjectId(),
    type: 'PORTFOLIO_ANALYSIS',
    insight: 'Your portfolio shows good diversification with 60% in large-cap funds.',
    confidence: 0.85,
    generatedAt: new Date()
  },
  
  marketInsight: {
    _id: new ObjectId(),
    userId: new ObjectId(),
    type: 'MARKET_ANALYSIS',
    insight: 'Current market conditions favor value investing strategies.',
    confidence: 0.78,
    generatedAt: new Date()
  }
};

module.exports = {
  mockUsers,
  mockPortfolios,
  mockFunds,
  mockTransactions,
  mockSmartSips,
  mockRewards,
  mockLeaderboards,
  mockWhatsAppMessages,
  mockAIInsights
}; 