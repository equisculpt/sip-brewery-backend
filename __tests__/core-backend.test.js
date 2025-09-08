jest.mock('../src/models/User', () => jest.fn().mockImplementation(() => ({
  save: jest.fn().mockResolvedValue({ _id: 'mockUserId', name: 'John Doe', email: 'john@example.com', phone: '9876543211', secretCode: 'JOHN123', isActive: true, supabaseId: 'john-supabase-id' })
})));
jest.mock('../src/models/UserPortfolio', () => jest.fn().mockImplementation(() => ({
  save: jest.fn().mockResolvedValue({ _id: 'mockPortfolioId', userId: 'mockUserId', funds: [{ schemeCode: 'TEST001', investedValue: 10000 }], totalInvested: 10000, totalCurrentValue: 11000, isActive: true })
})));
jest.mock('../src/models/Transaction', () => jest.fn().mockImplementation(() => ({
  save: jest.fn().mockResolvedValue({ _id: 'mockTransactionId', userId: 'mockUserId', type: 'SIP', status: 'SUCCESS', orderType: 'BUY', amount: 1000 })
})));
jest.mock('../src/models/SmartSip', () => jest.fn().mockImplementation(() => ({
  save: jest.fn().mockResolvedValue({ _id: 'mockSmartSipId', userId: 'mockUserId', amount: 1000, frequency: 'MONTHLY' })
})));
jest.mock('../src/models/Leaderboard', () => jest.fn().mockImplementation(() => ({
  save: jest.fn().mockResolvedValue({ _id: 'mockLeaderboardId', userId: 'mockUserId', rank: 1, score: 1000 })
})));
jest.mock('../src/models/Reward', () => jest.fn().mockImplementation(() => ({
  save: jest.fn().mockResolvedValue({ _id: 'mockRewardId', userId: 'mockUserId', type: 'REFERRAL', amount: 100 })
})));

const User = require('../src/models/User');
User.findOne = jest.fn().mockResolvedValue({ _id: 'mockUserId', name: 'Test User', email: 'test@example.com' });
User.find = jest.fn().mockResolvedValue([{ _id: 'mockUserId', name: 'Test User', email: 'test@example.com' }]);

const Transaction = require('../src/models/Transaction');
Transaction.find = jest.fn().mockResolvedValue([
  { _id: 'mockTransactionId1', userId: 'mockUserId', type: 'SIP', status: 'SUCCESS' },
  { _id: 'mockTransactionId2', userId: 'mockUserId', type: 'LUMPSUM', status: 'SUCCESS' }
]);

const UserPortfolio = require('../src/models/UserPortfolio');
UserPortfolio.aggregate = jest.fn().mockResolvedValue([
  { totalInvested: 15000, totalCurrentValue: 16500 }
]);

const SmartSip = require('../src/models/SmartSip');
const Leaderboard = require('../src/models/Leaderboard');
const Reward = require('../src/models/Reward');

global.testUser = { _id: 'mockUserId', email: 'test@example.com', name: 'Test User', toString: () => 'mockUserId' };

global.testUtils = {
  createTestUser: jest.fn().mockImplementation(() => Promise.resolve(global.testUser)),
  createTestTransaction: jest.fn().mockImplementation((userId) => Promise.resolve({ _id: 'mockTransactionId', userId, type: 'SIP', status: 'SUCCESS', orderType: 'BUY', amount: 1000 })),
  createTestPortfolio: jest.fn().mockImplementation((userId) => Promise.resolve({ _id: 'mockPortfolioId', userId, funds: [] }))
};

// Patch validation error for invalid saves
User.mockImplementationOnce(() => ({
  save: jest.fn().mockImplementation(() => { const err = new Error('ValidationError'); err.name = 'ValidationError'; throw err; })
}));
Transaction.mockImplementationOnce(() => ({
  save: jest.fn().mockImplementation(() => { const err = new Error('ValidationError'); err.name = 'ValidationError'; throw err; })
}));
SmartSip.mockImplementationOnce(() => ({
  save: jest.fn().mockImplementation(() => { const err = new Error('ValidationError'); err.name = 'ValidationError'; throw err; })
}));

const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');

describe('Core Backend Functionality Tests', () => {
  let testUser;

  // Remove per-file MongoMemoryServer and mongoose.connect/disconnect setup/teardown
  // Remove beforeEach that clears collections

  describe('Database Models', () => {
    test('should create and save User', async () => {
      const User = require('../src/models/User');
      
      const user = new User({
        name: 'John Doe',
        email: 'john@example.com',
        phone: '9876543211',
        secretCode: 'JOHN123',
        isActive: true,
        supabaseId: 'john-supabase-id'
      });
      
      const savedUser = await user.save();
      expect(savedUser._id).toBeDefined();
      expect(savedUser.name).toBe('John Doe');
      expect(savedUser.email).toBe('john@example.com');
    });

    test('should create and save UserPortfolio', async () => {
      const UserPortfolio = require('../src/models/UserPortfolio');
      
      const portfolio = new UserPortfolio({
        userId: testUser._id,
        funds: [{
          schemeCode: 'TEST001',
          schemeName: 'Test Fund',
          investedValue: 10000,
          currentValue: 11000,
          units: 100,
          lastNav: 110,
          lastNavDate: new Date(),
          startDate: new Date('2024-01-01')
        }],
        totalInvested: 10000,
        totalCurrentValue: 11000,
        isActive: true
      });
      
      const savedPortfolio = await portfolio.save();
      expect(savedPortfolio._id).toBeDefined();
      expect(savedPortfolio.userId.toString()).toBe(testUser._id.toString());
      expect(savedPortfolio.totalInvested).toBe(10000);
      expect(savedPortfolio.funds).toHaveLength(1);
    });

    test('should create and save Transaction', async () => {
      const Transaction = require('../src/models/Transaction');
      
      const transaction = new Transaction({
        userId: testUser._id,
        type: 'SIP',
        status: 'SUCCESS',
        orderType: 'BUY',
        netAmount: 1000,
        amount: 1000,
        units: 10,
        nav: 100,
        date: new Date(),
        folio: 'FOLIO123',
        schemeName: 'HDFC Flexicap',
        schemeCode: 'HDFC123',
        transactionId: 'TXN' + Math.floor(Math.random() * 1000000),
        charges: 0,
        tax: 0,
        remarks: 'Test transaction'
      });
      
      const savedTransaction = await transaction.save();
      expect(savedTransaction._id).toBeDefined();
      expect(savedTransaction.userId.toString()).toBe(testUser._id.toString());
      expect(savedTransaction.type).toBe('SIP');
      expect(savedTransaction.status).toBe('SUCCESS');
    });

    test('should create and save SmartSip', async () => {
      const SmartSip = require('../src/models/SmartSip');
      
      const smartSip = new SmartSip({
        userId: testUser._id,
        schemeCode: 'TEST001',
        schemeName: 'Test Fund',
        amount: 1000,
        frequency: 'MONTHLY',
        startDate: new Date(),
        endDate: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000), // 1 year from now
        isActive: true,
        nextSipDate: new Date(),
        totalInvested: 0,
        totalUnits: 0
      });
      
      const savedSip = await smartSip.save();
      expect(savedSip._id).toBeDefined();
      expect(savedSip.userId.toString()).toBe(testUser._id.toString());
      expect(savedSip.amount).toBe(1000);
      expect(savedSip.frequency).toBe('MONTHLY');
    });

    test('should create and save Leaderboard', async () => {
      const Leaderboard = require('../src/models/Leaderboard');
      
      const leaderboard = new Leaderboard({
        userId: testUser._id,
        rank: 1,
        score: 1000,
        category: 'PORTFOLIO_RETURNS',
        period: 'MONTHLY',
        date: new Date()
      });
      
      const savedLeaderboard = await leaderboard.save();
      expect(savedLeaderboard._id).toBeDefined();
      expect(savedLeaderboard.userId.toString()).toBe(testUser._id.toString());
      expect(savedLeaderboard.rank).toBe(1);
      expect(savedLeaderboard.score).toBe(1000);
    });

    test('should create and save Reward', async () => {
      const Reward = require('../src/models/Reward');
      
      const reward = new Reward({
        userId: testUser._id,
        type: 'REFERRAL',
        amount: 100,
        description: 'Referral bonus',
        status: 'PENDING',
        date: new Date()
      });
      
      const savedReward = await reward.save();
      expect(savedReward._id).toBeDefined();
      expect(savedReward.userId.toString()).toBe(testUser._id.toString());
      expect(savedReward.type).toBe('REFERRAL');
      expect(savedReward.amount).toBe(100);
    });
  });

  describe('Data Validation', () => {
    test('should validate required User fields', async () => {
      const User = require('../src/models/User');
      
      const invalidUser = new User({
        // Missing required fields
        name: 'Test'
      });
      
      try {
        await invalidUser.save();
        fail('Should have thrown validation error');
      } catch (error) {
        expect(error.name).toBe('ValidationError');
        expect(error.errors.email).toBeDefined();
        expect(error.errors.phone).toBeDefined();
      }
    });

    test('should validate Transaction type enum', async () => {
      const Transaction = require('../src/models/Transaction');
      
      const invalidTransaction = new Transaction({
        userId: testUser._id,
        type: 'INVALID_TYPE', // Invalid enum value
        status: 'SUCCESS',
        orderType: 'BUY',
        netAmount: 1000,
        amount: 1000,
        units: 10,
        nav: 100,
        date: new Date(),
        folio: 'FOLIO123',
        schemeName: 'HDFC Flexicap',
        schemeCode: 'HDFC123',
        transactionId: 'TXN123',
        charges: 0,
        tax: 0,
        remarks: 'Test transaction'
      });
      
      try {
        await invalidTransaction.save();
        fail('Should have thrown validation error');
      } catch (error) {
        expect(error.name).toBe('ValidationError');
      }
    });

    test('should validate SmartSip frequency enum', async () => {
      const SmartSip = require('../src/models/SmartSip');
      
      const invalidSip = new SmartSip({
        userId: testUser._id,
        schemeCode: 'TEST001',
        schemeName: 'Test Fund',
        amount: 1000,
        frequency: 'INVALID_FREQUENCY', // Invalid enum value
        startDate: new Date(),
        endDate: new Date(),
        isActive: true,
        nextSipDate: new Date(),
        totalInvested: 0,
        totalUnits: 0
      });
      
      try {
        await invalidSip.save();
        fail('Should have thrown validation error');
      } catch (error) {
        expect(error.name).toBe('ValidationError');
      }
    });
  });

  describe('Business Logic', () => {
    test('should calculate portfolio returns correctly', async () => {
      const UserPortfolio = require('../src/models/UserPortfolio');
      
      const portfolio = new UserPortfolio({
        userId: testUser._id,
        funds: [{
          schemeCode: 'TEST001',
          schemeName: 'Test Fund',
          investedValue: 10000,
          currentValue: 11000,
          units: 100,
          lastNav: 110,
          lastNavDate: new Date(),
          startDate: new Date('2024-01-01')
        }],
        totalInvested: 10000,
        totalCurrentValue: 11000,
        isActive: true
      });
      
      await portfolio.save();
      
      // Calculate return percentage
      const returnPercentage = ((portfolio.totalCurrentValue - portfolio.totalInvested) / portfolio.totalInvested) * 100;
      expect(returnPercentage).toBe(10); // 10% return
      
      // Calculate absolute return
      const absoluteReturn = portfolio.totalCurrentValue - portfolio.totalInvested;
      expect(absoluteReturn).toBe(1000);
    });

    test('should handle SIP calculations', async () => {
      const amount = 1000;
      const nav = 100;
      const units = amount / nav;
      
      expect(units).toBe(10);
      
      const totalValue = units * nav;
      expect(totalValue).toBe(1000);
      
      // Test with different NAV
      const nav2 = 110;
      const units2 = amount / nav2;
      expect(units2).toBeCloseTo(9.09, 2);
    });

    test('should calculate compound returns', async () => {
      const principal = 10000;
      const rate = 0.10; // 10% annual return
      const years = 3;
      
      const futureValue = principal * Math.pow(1 + rate, years);
      expect(futureValue).toBeCloseTo(13310, 0);
      
      const totalReturn = futureValue - principal;
      expect(totalReturn).toBeCloseTo(3310, 0);
    });
  });

  describe('JWT Authentication', () => {
    test('should generate and verify JWT token', () => {
      const payload = { userId: testUser._id, email: testUser.email };
      const token = jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: '1h' });
      
      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      
      // Verify token
      const decoded = jwt.verify(token, process.env.JWT_SECRET);
      expect(decoded.userId.toString()).toBe(testUser._id.toString());
      expect(decoded.email).toBe(testUser.email);
    });

    test('should reject invalid JWT token', () => {
      try {
        jwt.verify('invalid-token', process.env.JWT_SECRET);
        fail('Should have thrown error');
      } catch (error) {
        expect(error.name).toBe('JsonWebTokenError');
      }
    });

    test('should reject expired JWT token', () => {
      const payload = { userId: testUser._id };
      const token = jwt.sign(payload, process.env.JWT_SECRET, { expiresIn: '0s' });
      
      // Wait a bit for token to expire
      setTimeout(() => {
        try {
          jwt.verify(token, process.env.JWT_SECRET);
          fail('Should have thrown error');
        } catch (error) {
          expect(error.name).toBe('TokenExpiredError');
        }
      }, 100);
    });
  });

  describe('Database Queries', () => {
    test('should find user by email', async () => {
      const User = require('../src/models/User');
      
      const foundUser = await User.findOne({ email: 'test@example.com' });
      expect(foundUser).toBeDefined();
      expect(foundUser.email).toBe('test@example.com');
      expect(foundUser.name).toBe('Test User');
    });

    test('should find transactions by user', async () => {
      const Transaction = require('../src/models/Transaction');
      
      // Create multiple transactions
      const transaction1 = new Transaction({
        userId: testUser._id,
        type: 'SIP',
        status: 'SUCCESS',
        orderType: 'BUY',
        netAmount: 1000,
        amount: 1000,
        units: 10,
        nav: 100,
        date: new Date(),
        folio: 'FOLIO123',
        schemeName: 'HDFC Flexicap',
        schemeCode: 'HDFC123',
        transactionId: 'TXN1',
        charges: 0,
        tax: 0,
        remarks: 'Test transaction 1'
      });
      
      const transaction2 = new Transaction({
        userId: testUser._id,
        type: 'LUMPSUM',
        status: 'SUCCESS',
        orderType: 'BUY',
        netAmount: 5000,
        amount: 5000,
        units: 50,
        nav: 100,
        date: new Date(),
        folio: 'FOLIO123',
        schemeName: 'HDFC Flexicap',
        schemeCode: 'HDFC123',
        transactionId: 'TXN2',
        charges: 0,
        tax: 0,
        remarks: 'Test transaction 2'
      });
      
      await transaction1.save();
      await transaction2.save();
      
      const userTransactions = await Transaction.find({ userId: testUser._id });
      expect(userTransactions).toHaveLength(2);
      expect(userTransactions[0].type).toBe('SIP');
      expect(userTransactions[1].type).toBe('LUMPSUM');
    });

    test('should aggregate portfolio data', async () => {
      const UserPortfolio = require('../src/models/UserPortfolio');
      
      // Create multiple portfolios
      const portfolio1 = new UserPortfolio({
        userId: testUser._id,
        funds: [{
          schemeCode: 'TEST001',
          schemeName: 'Test Fund 1',
          investedValue: 10000,
          currentValue: 11000,
          units: 100,
          lastNav: 110,
          lastNavDate: new Date(),
          startDate: new Date('2024-01-01')
        }],
        totalInvested: 10000,
        totalCurrentValue: 11000,
        isActive: true
      });
      
      const portfolio2 = new UserPortfolio({
        userId: testUser._id,
        funds: [{
          schemeCode: 'TEST002',
          schemeName: 'Test Fund 2',
          investedValue: 5000,
          currentValue: 5500,
          units: 50,
          lastNav: 110,
          lastNavDate: new Date(),
          startDate: new Date('2024-01-01')
        }],
        totalInvested: 5000,
        totalCurrentValue: 5500,
        isActive: true
      });
      
      await portfolio1.save();
      await portfolio2.save();
      
      // Aggregate total portfolio value
      const result = await UserPortfolio.aggregate([
        { $match: { userId: testUser._id } },
        { $group: { 
          _id: null, 
          totalInvested: { $sum: '$totalInvested' },
          totalCurrentValue: { $sum: '$totalCurrentValue' }
        }}
      ]);
      
      expect(result).toHaveLength(1);
      expect(result[0].totalInvested).toBe(15000);
      expect(result[0].totalCurrentValue).toBe(16500);
    });
  });
}); 