/**
 * 🚀 SIMPLE SIP BREWERY BACKEND SERVER
 * For Frontend Integration Testing
 */

const express = require('express');
const cors = require('cors');
const asiPortfolioRoutes = require('./src/routes/asiPortfolioRoutes');
const asiAnalysisRoutes = require('./src/routes/asi-analysis');

const app = express();
const PORT = process.env.PORT || 3001; // Use 3001 to avoid conflict with frontend

// Middleware
app.use(cors({
  origin: ['http://localhost:3000', 'http://localhost:3001'],
  credentials: true
}));
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'OK',
    message: 'SIP Brewery Backend is running',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// Status endpoint
app.get('/status', (req, res) => {
  res.json({
    status: 'online',
    uptime: process.uptime(),
    asi_status: 'ready',
    database: 'connected',
    services: {
      fund_analysis: 'active',
      market_data: 'active',
      portfolio_optimization: 'active'
    }
  });
});

// Market indices endpoint
app.get('/api/market/indices', (req, res) => {
  res.json({
    success: true,
    message: 'Market indices retrieved successfully',
    data: {
      indices: [
        {
          name: 'NIFTY 50',
          value: 23249.50,
          change: 234.75,
          change_percent: 1.02,
          last_updated: new Date().toISOString()
        },
        {
          name: 'SENSEX',
          value: 76693.20,
          change: 445.87,
          change_percent: 0.58,
          last_updated: new Date().toISOString()
        },
        {
          name: 'BANK NIFTY',
          value: 49856.30,
          change: -123.45,
          change_percent: -0.25,
          last_updated: new Date().toISOString()
        },
        {
          name: 'NIFTY IT',
          value: 43567.80,
          change: 567.23,
          change_percent: 1.32,
          last_updated: new Date().toISOString()
        }
      ]
    }
  });
});

// ASI fund analysis endpoint
app.get('/api/asi/fund/:code/analyze', (req, res) => {
  const { code } = req.params;
  
  res.json({
    success: true,
    message: 'ASI fund analysis completed',
    data: {
      fund_code: code,
      analysis: {
        score: 8.5,
        recommendation: 'BUY',
        risk_level: 'MODERATE',
        expected_return: 12.5,
        confidence: 0.87
      },
      insights: [
        'Strong historical performance with consistent returns',
        'Well-diversified portfolio across sectors',
        'Low expense ratio compared to category average',
        'Experienced fund manager with proven track record'
      ],
      timestamp: new Date().toISOString()
    }
  });
});

// ASI portfolio optimization endpoint
app.post('/api/asi/portfolio/optimize', (req, res) => {
  const { funds, amount, risk_tolerance } = req.body;
  
  res.json({
    success: true,
    message: 'Portfolio optimization completed',
    data: {
      optimized_allocation: [
        { fund_code: 'HDFC0000123', allocation: 40, amount: amount * 0.4 },
        { fund_code: 'SBI0000456', allocation: 35, amount: amount * 0.35 },
        { fund_code: 'ICICI0000789', allocation: 25, amount: amount * 0.25 }
      ],
      expected_return: 11.8,
      risk_score: risk_tolerance || 'MODERATE',
      diversification_score: 0.92,
      timestamp: new Date().toISOString()
    }
  });
});

// ASI market insights endpoint
app.get('/api/asi/market/insights', (req, res) => {
  res.json({
    success: true,
    message: 'Market insights generated',
    data: {
      market_sentiment: 'BULLISH',
      key_insights: [
        'Technology sector showing strong momentum',
        'Banking stocks recovering from recent lows',
        'Mid-cap funds outperforming large-cap in current market',
        'SIP investments recommended in current volatile market'
      ],
      sentiment_score: 0.72,
      market_timing: 'FAVORABLE',
      timestamp: new Date().toISOString()
    }
  });
});

// Demo Account System
let demoUsers = new Map();
let demoPortfolios = new Map();
let demoTransactions = new Map();

// Demo account signup
app.post('/api/demo/signup', (req, res) => {
  const { name, email, phone } = req.body;
  
  if (!name || !email) {
    return res.status(400).json({
      success: false,
      message: 'Name and email are required'
    });
  }
  
  // Check if user already exists
  if (demoUsers.has(email)) {
    return res.status(400).json({
      success: false,
      message: 'Demo account already exists for this email'
    });
  }
  
  const userId = `demo_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  const demoUser = {
    id: userId,
    name,
    email,
    phone: phone || '',
    demo_balance: 100000, // ₹1,00,000 demo money
    created_at: new Date().toISOString(),
    last_login: new Date().toISOString()
  };
  
  // Initialize empty portfolio
  const portfolio = {
    user_id: userId,
    total_invested: 0,
    current_value: 0,
    total_returns: 0,
    return_percentage: 0,
    holdings: [],
    sip_investments: [],
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString()
  };
  
  demoUsers.set(email, demoUser);
  demoPortfolios.set(userId, portfolio);
  demoTransactions.set(userId, []);
  
  res.json({
    success: true,
    message: 'Demo account created successfully',
    data: {
      user: demoUser,
      portfolio,
      demo_token: `demo_token_${userId}`
    }
  });
});

// Demo account login
app.post('/api/demo/login', (req, res) => {
  const { email } = req.body;
  
  if (!email) {
    return res.status(400).json({
      success: false,
      message: 'Email is required'
    });
  }
  
  const user = demoUsers.get(email);
  if (!user) {
    return res.status(404).json({
      success: false,
      message: 'Demo account not found'
    });
  }
  
  // Update last login
  user.last_login = new Date().toISOString();
  demoUsers.set(email, user);
  
  const portfolio = demoPortfolios.get(user.id);
  
  res.json({
    success: true,
    message: 'Demo login successful',
    data: {
      user,
      portfolio,
      demo_token: `demo_token_${user.id}`
    }
  });
});

// Get demo portfolio
app.get('/api/demo/portfolio/:userId', (req, res) => {
  const { userId } = req.params;
  
  const portfolio = demoPortfolios.get(userId);
  if (!portfolio) {
    return res.status(404).json({
      success: false,
      message: 'Portfolio not found'
    });
  }
  
  // Simulate live market changes
  const updatedHoldings = portfolio.holdings.map(holding => {
    const marketChange = (Math.random() - 0.5) * 0.02; // ±1% random change
    const newValue = holding.current_value * (1 + marketChange);
    return {
      ...holding,
      current_value: Number(newValue.toFixed(2)),
      return_amount: Number((newValue - holding.invested_amount).toFixed(2)),
      return_percentage: Number(((newValue - holding.invested_amount) / holding.invested_amount * 100).toFixed(2))
    };
  });
  
  const totalCurrentValue = updatedHoldings.reduce((sum, holding) => sum + holding.current_value, 0);
  const totalInvested = updatedHoldings.reduce((sum, holding) => sum + holding.invested_amount, 0);
  const totalReturns = totalCurrentValue - totalInvested;
  
  const updatedPortfolio = {
    ...portfolio,
    holdings: updatedHoldings,
    current_value: Number(totalCurrentValue.toFixed(2)),
    total_invested: Number(totalInvested.toFixed(2)),
    total_returns: Number(totalReturns.toFixed(2)),
    return_percentage: totalInvested > 0 ? Number((totalReturns / totalInvested * 100).toFixed(2)) : 0,
    updated_at: new Date().toISOString()
  };
  
  demoPortfolios.set(userId, updatedPortfolio);
  
  res.json({
    success: true,
    message: 'Portfolio retrieved successfully',
    data: updatedPortfolio
  });
});

// Demo fund investment
app.post('/api/demo/invest', (req, res) => {
  const { userId, fundCode, fundName, amount, investmentType } = req.body;
  
  if (!userId || !fundCode || !amount) {
    return res.status(400).json({
      success: false,
      message: 'User ID, fund code, and amount are required'
    });
  }
  
  const user = Array.from(demoUsers.values()).find(u => u.id === userId);
  if (!user) {
    return res.status(404).json({
      success: false,
      message: 'Demo user not found'
    });
  }
  
  if (user.demo_balance < amount) {
    return res.status(400).json({
      success: false,
      message: 'Insufficient demo balance'
    });
  }
  
  const portfolio = demoPortfolios.get(userId);
  const transactions = demoTransactions.get(userId) || [];
  
  // Deduct from demo balance
  user.demo_balance -= amount;
  demoUsers.set(user.email, user);
  
  // Add to portfolio
  const existingHolding = portfolio.holdings.find(h => h.fund_code === fundCode);
  
  if (existingHolding) {
    // Add to existing holding
    existingHolding.invested_amount += amount;
    existingHolding.current_value += amount; // Start with same value
    existingHolding.units += amount / 100; // Assume ₹100 per unit
  } else {
    // Create new holding
    portfolio.holdings.push({
      fund_code: fundCode,
      fund_name: fundName || `Fund ${fundCode}`,
      invested_amount: amount,
      current_value: amount,
      units: amount / 100,
      return_amount: 0,
      return_percentage: 0,
      investment_date: new Date().toISOString()
    });
  }
  
  // Add transaction
  transactions.push({
    id: `txn_${Date.now()}`,
    type: investmentType || 'LUMPSUM',
    fund_code: fundCode,
    fund_name: fundName || `Fund ${fundCode}`,
    amount,
    status: 'COMPLETED',
    date: new Date().toISOString()
  });
  
  // Update portfolio totals
  portfolio.total_invested = portfolio.holdings.reduce((sum, h) => sum + h.invested_amount, 0);
  portfolio.current_value = portfolio.holdings.reduce((sum, h) => sum + h.current_value, 0);
  portfolio.total_returns = portfolio.current_value - portfolio.total_invested;
  portfolio.return_percentage = portfolio.total_invested > 0 ? 
    (portfolio.total_returns / portfolio.total_invested * 100) : 0;
  portfolio.updated_at = new Date().toISOString();
  
  demoPortfolios.set(userId, portfolio);
  demoTransactions.set(userId, transactions);
  
  res.json({
    success: true,
    message: 'Investment completed successfully',
    data: {
      transaction: transactions[transactions.length - 1],
      portfolio,
      remaining_balance: user.demo_balance
    }
  });
});

// Get demo transactions
app.get('/api/demo/transactions/:userId', (req, res) => {
  const { userId } = req.params;
  
  const transactions = demoTransactions.get(userId) || [];
  
  res.json({
    success: true,
    message: 'Transactions retrieved successfully',
    data: {
      transactions: transactions.reverse(), // Latest first
      total_transactions: transactions.length
    }
  });
});

// Demo SIP setup
app.post('/api/demo/sip/create', (req, res) => {
  const { userId, fundCode, fundName, monthlyAmount, duration } = req.body;
  
  if (!userId || !fundCode || !monthlyAmount || !duration) {
    return res.status(400).json({
      success: false,
      message: 'All SIP details are required'
    });
  }
  
  const portfolio = demoPortfolios.get(userId);
  if (!portfolio) {
    return res.status(404).json({
      success: false,
      message: 'Portfolio not found'
    });
  }
  
  const sipId = `sip_${Date.now()}`;
  const sip = {
    id: sipId,
    fund_code: fundCode,
    fund_name: fundName || `Fund ${fundCode}`,
    monthly_amount: monthlyAmount,
    duration_months: duration,
    start_date: new Date().toISOString(),
    next_installment: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
    total_invested: 0,
    installments_completed: 0,
    status: 'ACTIVE',
    created_at: new Date().toISOString()
  };
  
  portfolio.sip_investments.push(sip);
  demoPortfolios.set(userId, portfolio);
  
  res.json({
    success: true,
    message: 'SIP created successfully',
    data: sip
  });
});

// Get available demo funds
app.get('/api/demo/funds', (req, res) => {
  const demoFunds = [
    {
      code: 'HDFC0001',
      name: 'HDFC Top 100 Fund',
      category: 'Large Cap',
      nav: 756.23,
      returns_1y: 12.5,
      returns_3y: 15.2,
      expense_ratio: 1.05,
      risk: 'MODERATE'
    },
    {
      code: 'SBI0002',
      name: 'SBI Blue Chip Fund',
      category: 'Large Cap',
      nav: 432.18,
      returns_1y: 11.8,
      returns_3y: 14.7,
      expense_ratio: 0.98,
      risk: 'MODERATE'
    },
    {
      code: 'ICICI0003',
      name: 'ICICI Prudential Technology Fund',
      category: 'Sector',
      nav: 234.56,
      returns_1y: 18.3,
      returns_3y: 22.1,
      expense_ratio: 1.25,
      risk: 'HIGH'
    },
    {
      code: 'AXIS0004',
      name: 'Axis Small Cap Fund',
      category: 'Small Cap',
      nav: 89.45,
      returns_1y: 25.6,
      returns_3y: 28.9,
      expense_ratio: 1.45,
      risk: 'HIGH'
    },
    {
      code: 'KOTAK0005',
      name: 'Kotak Hybrid Equity Fund',
      category: 'Hybrid',
      nav: 345.67,
      returns_1y: 9.8,
      returns_3y: 11.5,
      expense_ratio: 0.85,
      risk: 'LOW'
    }
  ];
  
  res.json({
    success: true,
    message: 'Demo funds retrieved successfully',
    data: {
      funds: demoFunds,
      total_funds: demoFunds.length
    }
  });
});

// Mount ASI Portfolio Analysis routes
app.use('/api/asi', asiPortfolioRoutes);

// Mount ASI Analysis routes (includes FSI analysis)
app.use('/api/asi', asiAnalysisRoutes);

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: err.message
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Endpoint not found',
    path: req.originalUrl
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 SIP Brewery Backend running on port ${PORT}`);
  console.log(`📡 Health check: http://localhost:${PORT}/health`);
  console.log(`📊 Market data: http://localhost:${PORT}/api/market/indices`);
  console.log(`🧠 ASI endpoints available at /api/asi/*`);
});

module.exports = app;
