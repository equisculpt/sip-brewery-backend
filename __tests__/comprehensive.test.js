const mongoose = require('mongoose');
const { User, UserPortfolio, Transaction, SmartSip, Reward, WhatsAppSession } = require('../src/models');
const { createTestApp } = require('./setup');

describe('Comprehensive Backend Tests', () => {
  let app;
  
  beforeAll(async () => {
    app = createTestApp();
  });

  describe('Database Connection', () => {
    test('should connect to database', async () => {
      expect(mongoose.connection.readyState).toBe(1);
    });
  });

  describe('User Management', () => {
    test('should create test user', async () => {
      const user = await global.testUtils.createTestUser();
      expect(user).toBeDefined();
      expect(user.name).toBe('Test User');
      expect(user.email).toMatch(/test-.*@example.com/);
    });

    test('should create user with custom data', async () => {
      const userData = {
        name: 'Custom User',
        email: 'custom@example.com',
        phone: '9876543210',
        role: 'CLIENT'
      };
      const user = await global.testUtils.createTestUser(userData);
      expect(user.name).toBe('Custom User');
      expect(user.email).toBe('custom@example.com');
      expect(user.role).toBe('CLIENT');
    });
  });

  describe('Portfolio Management', () => {
    test('should create test portfolio', async () => {
      const user = await global.testUtils.createTestUser();
      const portfolio = await global.testUtils.createTestPortfolio(user._id);
      expect(portfolio).toBeDefined();
      expect(portfolio.userId.toString()).toBe(user._id.toString());
      expect(portfolio.funds).toHaveLength(1);
      expect(portfolio.totalInvested).toBe(10000);
    });

    test('should create portfolio with custom data', async () => {
      const user = await global.testUtils.createTestUser();
      const portfolioData = {
        funds: [{
          schemeCode: 'CUSTOM001',
          schemeName: 'Custom Fund',
          investedValue: 5000,
          currentValue: 5500,
          units: 50,
          lastNav: 110,
          lastNavDate: new Date(),
          startDate: new Date('2024-01-01')
        }],
        totalInvested: 5000,
        totalCurrentValue: 5500
      };
      const portfolio = await global.testUtils.createTestPortfolio(user._id, portfolioData);
      expect(portfolio.funds).toHaveLength(1);
      expect(portfolio.funds[0].schemeCode).toBe('CUSTOM001');
    });
  });

  describe('Transaction Management', () => {
    test('should create test transaction', async () => {
      const user = await global.testUtils.createTestUser();
      const transaction = await global.testUtils.createTestTransaction(user._id);
      expect(transaction).toBeDefined();
      expect(transaction.userId.toString()).toBe(user._id.toString());
      expect(transaction.nav).toBeDefined();
      expect(transaction.type).toBe('SIP');
      expect(transaction.status).toBe('SUCCESS');
    });

    test('should create transaction with custom data', async () => {
      const user = await global.testUtils.createTestUser();
      const transactionData = {
        type: 'LUMPSUM',
        status: 'PENDING',
        orderType: 'BUY',
        amount: 5000,
        units: 50,
        nav: 100,
        netAmount: 5000
      };
      const transaction = await global.testUtils.createTestTransaction(user._id, transactionData);
      expect(transaction.type).toBe('LUMPSUM');
      expect(transaction.status).toBe('PENDING');
      expect(transaction.amount).toBe(5000);
    });
  });

  describe('Smart SIP Management', () => {
    test('should create Smart SIP', async () => {
      const user = await global.testUtils.createTestUser();
      const smartSip = new SmartSip({
        userId: user._id,
        schemeCode: 'SIP001',
        schemeName: 'Smart SIP Fund',
        amount: 1000,
        frequency: 'MONTHLY',
        startDate: new Date(),
        endDate: new Date(Date.now() + 365 * 24 * 60 * 60 * 1000),
        status: 'ACTIVE',
        isActive: true,
        nextSIPDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000),
        maxSip: 5000,
        minSip: 500,
        averageSip: 1000
      });
      const savedSip = await smartSip.save();
      expect(savedSip).toBeDefined();
      expect(savedSip.userId.toString()).toBe(user._id.toString());
      expect(savedSip.amount).toBe(1000);
    });
  });

  describe('Rewards System', () => {
    test('should create reward', async () => {
      const user = await global.testUtils.createTestUser();
      const reward = new Reward({
        userId: user._id,
        type: 'SIGNUP',
        points: 100,
        amount: 50,
        description: 'Signup bonus',
        isActive: true
      });
      const savedReward = await reward.save();
      expect(savedReward).toBeDefined();
      expect(savedReward.userId.toString()).toBe(user._id.toString());
      expect(savedReward.points).toBe(100);
    });
  });

  describe('WhatsApp Integration', () => {
    test('should create WhatsApp session', async () => {
      const phoneNumber = '+919876543210';
      const session = await global.testUtils.createWhatsAppSession(phoneNumber);
      expect(session).toBeDefined();
      expect(session.phoneNumber).toBe(phoneNumber);
      expect(session.isActive).toBe(true);
    });

    test('should get conversation context', async () => {
      const phoneNumber = '+919876543211';
      await global.testUtils.createWhatsAppSession(phoneNumber);
      const context = await global.testUtils.getConversationContext(phoneNumber);
      expect(context).toBeDefined();
      expect(context.phoneNumber).toBe(phoneNumber);
    });
  });

  describe('Authentication', () => {
    test('should generate test JWT token', () => {
      const userId = new mongoose.Types.ObjectId();
      const token = global.testUtils.generateTestToken(userId);
      expect(token).toBeDefined();
      expect(typeof token).toBe('string');
      expect(token.length).toBeGreaterThan(0);
    });
  });

  describe('API Endpoints', () => {
    test('should have working app instance', () => {
      expect(app).toBeDefined();
      expect(typeof app.listen).toBe('function');
    });
  });
});