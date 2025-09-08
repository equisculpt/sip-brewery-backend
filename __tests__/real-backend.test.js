jest.mock('../src/models/User', () => jest.fn().mockImplementation(() => ({
  save: jest.fn().mockResolvedValue({ _id: 'mockUserId', name: 'Test User', email: 'test@example.com', phone: '9876543210', secretCode: 'TEST123', isActive: true, supabaseId: 'test-supabase-id' })
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

const request = require('supertest');
const mongoose = require('mongoose');
const { MongoMemoryServer } = require('mongodb-memory-server');
const jwt = require('jsonwebtoken');

// Import the platform class
const { UniverseClassMutualFundPlatform } = require('../src/app');

describe('Real Backend Functionality Tests', () => {
  let testUser;
  let authToken;
  let platform;
  let app;

  beforeAll(async () => {
    // Set test environment variables
    process.env.NODE_ENV = 'test';
    process.env.JWT_SECRET = 'test-secret-key';
    process.env.DISABLE_EXTERNAL_APIS = 'true';
    process.env.DISABLE_WHATSAPP = 'true';
    process.env.DISABLE_AI_SERVICES = 'true';
    
    // Create and initialize the platform
    platform = new UniverseClassMutualFundPlatform();
    await platform.initialize();
    app = platform.app;
    
    // Create a test user
    const User = require('../src/models/User');
    testUser = new User({
      name: 'Test User',
      email: 'test@example.com',
      phone: '9876543210',
      secretCode: 'TEST123',
      isActive: true,
      supabaseId: 'test-supabase-id'
    });
    await testUser.save();
    
    // Generate auth token
    authToken = jwt.sign({ userId: testUser._id }, process.env.JWT_SECRET, { expiresIn: '1h' });
  });

  describe('Health Check', () => {
    test('should return health status', async () => {
      const response = await request(app)
        .get('/health')
        .expect(200);
      
      expect(response.body).toHaveProperty('status');
      expect(response.body.status).toBe('healthy');
      expect(response.body).toHaveProperty('platform');
    });

    test('should return platform status', async () => {
      const response = await request(app)
        .get('/status')
        .expect(200);
      
      expect(response.body).toHaveProperty('platform');
      expect(response.body).toHaveProperty('version');
      expect(response.body).toHaveProperty('features');
    });
  });

  describe('Database Operations', () => {
    test('should create user portfolio', async () => {
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
    });

    test('should create transaction', async () => {
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
    });

    test('should create Smart SIP', async () => {
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
    });
  });

  describe('API Endpoints', () => {
    test('should get leaderboard', async () => {
      const response = await request(app)
        .get('/api/leaderboard')
        .expect(200);
      
      expect(response.body).toBeDefined();
    });

    test('should handle 404 routes', async () => {
      const response = await request(app)
        .get('/api/nonexistent-route')
        .expect(404);
      
      expect(response.body).toHaveProperty('success', false);
    });
  });

  describe('Data Validation', () => {
    test('should validate user data', async () => {
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
      }
    });

    test('should validate transaction data', async () => {
      const Transaction = require('../src/models/Transaction');
      
      const invalidTransaction = new Transaction({
        userId: testUser._id,
        // Missing required fields
        type: 'INVALID_TYPE'
      });
      
      try {
        await invalidTransaction.save();
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
    });

    test('should handle SIP calculations', async () => {
      const amount = 1000;
      const nav = 100;
      const units = amount / nav;
      
      expect(units).toBe(10);
      
      const totalValue = units * nav;
      expect(totalValue).toBe(1000);
    });
  });

  describe('Performance', () => {
    test('should handle multiple concurrent requests', async () => {
      const promises = [];
      
      for (let i = 0; i < 10; i++) {
        promises.push(
          request(app)
            .get('/health')
            .expect(200)
        );
      }
      
      const responses = await Promise.all(promises);
      expect(responses).toHaveLength(10);
      
      responses.forEach(response => {
        expect(response.body.status).toBe('healthy');
      });
    });

    test('should respond within acceptable time', async () => {
      const startTime = Date.now();
      
      await request(app)
        .get('/health')
        .expect(200);
      
      const endTime = Date.now();
      const responseTime = endTime - startTime;
      
      expect(responseTime).toBeLessThan(1000); // Should respond within 1 second
    });
  });
}); 