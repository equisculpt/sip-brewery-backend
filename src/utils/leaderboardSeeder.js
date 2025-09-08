const User = require('../models/User');
const UserPortfolio = require('../models/UserPortfolio');
const Leaderboard = require('../models/Leaderboard');
const logger = require('./logger');

// Sample fund data
const sampleFunds = [
  {
    schemeCode: 'HDFC001',
    schemeName: 'HDFC Flexicap Fund',
    nav: 45.67
  },
  {
    schemeCode: 'PARAG001',
    schemeName: 'Parag Parikh Flexicap Fund',
    nav: 52.34
  },
  {
    schemeCode: 'SBI001',
    schemeName: 'SBI Smallcap Fund',
    nav: 38.91
  },
  {
    schemeCode: 'QUANT001',
    schemeName: 'Quant Tax Saver Fund',
    nav: 67.23
  },
  {
    schemeCode: 'AXIS001',
    schemeName: 'Axis Bluechip Fund',
    nav: 41.56
  },
  {
    schemeCode: 'MIRAE001',
    schemeName: 'Mirae Asset Emerging Bluechip Fund',
    nav: 89.12
  }
];

// Generate random XIRR between 6% and 22%
function generateRandomXIRR() {
  return 6 + Math.random() * 16;
}

// Generate random portfolio allocation
function generateRandomAllocation() {
  const numFunds = 3 + Math.floor(Math.random() * 3); // 3-5 funds
  const selectedFunds = [];
  const usedIndices = new Set();
  
  while (selectedFunds.length < numFunds) {
    const index = Math.floor(Math.random() * sampleFunds.length);
    if (!usedIndices.has(index)) {
      usedIndices.add(index);
      selectedFunds.push(sampleFunds[index]);
    }
  }
  
  // Generate random allocation percentages that sum to 100
  const allocations = [];
  let remaining = 100;
  
  for (let i = 0; i < selectedFunds.length - 1; i++) {
    const maxAllocation = remaining - (selectedFunds.length - i - 1) * 10;
    const allocation = 10 + Math.random() * Math.min(maxAllocation - 10, 40);
    allocations.push(Math.round(allocation));
    remaining -= allocation;
  }
  
  allocations.push(remaining); // Last fund gets remaining allocation
  
  return selectedFunds.map((fund, index) => ({
    ...fund,
    allocation: allocations[index]
  }));
}

// Generate sample transactions
function generateSampleTransactions(funds, startDate) {
  const transactions = [];
  const now = new Date();
  let currentDate = new Date(startDate);
  
  while (currentDate <= now) {
    // Add SIP transaction for each fund
    funds.forEach(fund => {
      const sipAmount = fund.allocation * 100; // ₹100 per percentage point
      const nav = fund.nav * (0.9 + Math.random() * 0.2); // ±10% NAV variation
      const units = sipAmount / nav;
      
      transactions.push({
        date: new Date(currentDate),
        type: 'SIP',
        schemeCode: fund.schemeCode,
        amount: sipAmount,
        units: units,
        nav: nav
      });
    });
    
    // Move to next month
    currentDate.setMonth(currentDate.getMonth() + 1);
  }
  
  return transactions;
}

// Create sample user
async function createSampleUser(index) {
  const user = new User({
    supabaseId: `test-user-${index}`,
    email: `user${index}@example.com`,
    phone: `98765${index.toString().padStart(5, '0')}`,
    name: `Test User ${index}`,
    kycStatus: 'VERIFIED',
    preferences: {
      riskTolerance: ['CONSERVATIVE', 'MODERATE', 'AGGRESSIVE'][Math.floor(Math.random() * 3)],
      investmentGoal: 'WEALTH_CREATION',
      timeHorizon: 'LONG_TERM'
    }
  });
  
  await user.save();
  return user;
}

// Create sample portfolio
async function createSamplePortfolio(user) {
  const funds = generateRandomAllocation();
  const startDate = new Date();
  startDate.setMonth(startDate.getMonth() - Math.floor(Math.random() * 12) - 6); // 6-18 months ago
  
  const transactions = generateSampleTransactions(funds, startDate);
  
  // Calculate current values based on transactions
  const fundValues = {};
  transactions.forEach(t => {
    if (!fundValues[t.schemeCode]) {
      fundValues[t.schemeCode] = { units: 0, totalInvested: 0 };
    }
    fundValues[t.schemeCode].units += t.units;
    fundValues[t.schemeCode].totalInvested += t.amount;
  });
  
  // Generate current NAVs (with some growth)
  const currentNavs = {};
  funds.forEach(fund => {
    const growth = 1 + (generateRandomXIRR() / 100) * (Math.random() * 0.5 + 0.5);
    currentNavs[fund.schemeCode] = fund.nav * growth;
  });
  
  // Create portfolio funds
  const portfolioFunds = funds.map(fund => {
    const fundValue = fundValues[fund.schemeCode];
    const currentNav = currentNavs[fund.schemeCode];
    const currentValue = fundValue.units * currentNav;
    
    return {
      schemeCode: fund.schemeCode,
      schemeName: fund.schemeName,
      investedValue: fundValue.totalInvested,
      currentValue: currentValue,
      units: fundValue.units,
      startDate: startDate,
      lastNav: currentNav,
      lastNavDate: new Date()
    };
  });
  
  // Calculate XIRR values
  const xirr1M = generateRandomXIRR();
  const xirr3M = generateRandomXIRR();
  const xirr6M = generateRandomXIRR();
  const xirr1Y = generateRandomXIRR();
  const xirr3Y = generateRandomXIRR();
  
  const portfolio = new UserPortfolio({
    userId: user._id,
    funds: portfolioFunds,
    transactions: transactions,
    xirr1M: xirr1M,
    xirr3M: xirr3M,
    xirr6M: xirr6M,
    xirr1Y: xirr1Y,
    xirr3Y: xirr3Y
  });
  
  await portfolio.save();
  return portfolio;
}

// Seed leaderboard data
async function seedLeaderboardData() {
  try {
    logger.info('Starting leaderboard data seeding...');
    
    // Clear existing test data
    await User.deleteMany({ email: /@example\.com$/ });
    await UserPortfolio.deleteMany({});
    await Leaderboard.deleteMany({});
    
    logger.info('Cleared existing test data');
    
    // Create 20 sample users with portfolios
    const users = [];
    const portfolios = [];
    
    for (let i = 1; i <= 20; i++) {
      const user = await createSampleUser(i);
      users.push(user);
      
      const portfolio = await createSamplePortfolio(user);
      portfolios.push(portfolio);
      
      logger.info(`Created user ${i}/20: ${user.secretCode}`);
    }
    
    logger.info('Created all users and portfolios');
    
    // Generate leaderboards
    const leaderboardService = require('../services/leaderboardService');
    const results = await leaderboardService.generateAllLeaderboards();
    
    logger.info('Generated leaderboards:', results);
    
    // Display sample leaderboard data
    const leaderboard = await Leaderboard.findOne({ duration: '1Y' });
    if (leaderboard) {
      logger.info('Sample 1Y Leaderboard:');
      leaderboard.leaders.slice(0, 5).forEach((leader, index) => {
        logger.info(`${index + 1}. ${leader.secretCode} - ${leader.returnPercent}%`);
      });
    }
    
    logger.info('Leaderboard data seeding completed successfully!');
    
    return {
      usersCreated: users.length,
      portfoliosCreated: portfolios.length,
      leaderboardsGenerated: Object.keys(results).length
    };
    
  } catch (error) {
    logger.error('Error seeding leaderboard data:', error);
    throw error;
  }
}

// Export functions
module.exports = {
  seedLeaderboardData,
  createSampleUser,
  createSamplePortfolio,
  generateRandomXIRR,
  generateRandomAllocation,
  generateSampleTransactions
}; 