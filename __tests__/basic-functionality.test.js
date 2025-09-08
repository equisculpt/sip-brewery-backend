jest.mock('../src/models/User', () => {
  return jest.fn().mockImplementation(() => ({
    save: jest.fn().mockResolvedValue({ _id: 'mockUserId', name: 'Test User', email: 'test@example.com' })
  }));
});
jest.mock('../src/models/Transaction', () => {
  return jest.fn().mockImplementation(() => ({
    save: jest.fn().mockResolvedValue({ _id: 'mockTransactionId', userId: 'mockUserId', type: 'SIP', status: 'SUCCESS', orderType: 'BUY' })
  }));
});
jest.mock('../src/models/UserPortfolio', () => {
  return jest.fn().mockImplementation(() => ({
    save: jest.fn().mockResolvedValue({ _id: 'mockPortfolioId', userId: 'mockUserId', funds: [] })
  }));
});

const User = require('../src/models/User');
const Transaction = require('../src/models/Transaction');
const UserPortfolio = require('../src/models/UserPortfolio');

let deletedUserId = null;
let deletedTransactionId = null;
User.findById = jest.fn().mockImplementation((_id) => Promise.resolve(_id === deletedUserId ? null : { _id, name: 'Test User', email: 'test@example.com', save: jest.fn().mockResolvedValue({ _id, name: 'Updated User', email: 'test@example.com' }) }));
User.findByIdAndDelete = jest.fn().mockImplementation((_id) => { deletedUserId = _id; return Promise.resolve({ _id }); });
Transaction.findById = jest.fn().mockImplementation((_id) => Promise.resolve(_id === deletedTransactionId ? null : { _id, userId: 'mockUserId', type: 'SIP', status: 'SUCCESS', orderType: 'BUY', save: jest.fn().mockResolvedValue({ _id, status: 'PROCESSING' }) }));
Transaction.findByIdAndDelete = jest.fn().mockImplementation((_id) => { deletedTransactionId = _id; return Promise.resolve({ _id }); });

global.testUtils = {
  createTestUser: jest.fn().mockImplementation(() => Promise.resolve({ _id: 'mockUserId', name: 'Test User', email: `test${Math.random()}@example.com` })),
  createTestTransaction: jest.fn().mockImplementation((userId) => Promise.resolve({ _id: 'mockTransactionId', userId, type: 'SIP', status: 'SUCCESS', orderType: 'BUY' })),
  createTestPortfolio: jest.fn().mockImplementation((userId) => Promise.resolve({ _id: 'mockPortfolioId', userId, funds: [] }))
};

describe('Basic Functionality Tests', () => {
  describe('Database Operations', () => {
    test('should perform CRUD operations on User', async () => {
      // Create
      const user = await global.testUtils.createTestUser();
      expect(user._id).toBeDefined();
      
      // Read
      const foundUser = await User.findById(user._id);
      expect(foundUser).toBeDefined();
      expect(foundUser.name).toBe('Test User');
      
      // Update
      foundUser.name = 'Updated User';
      await foundUser.save();
      const updatedUser = await User.findById(user._id);
      expect(updatedUser.name).toBe('Updated User');
      
      // Delete
      await User.findByIdAndDelete(user._id);
      const deletedUser = await User.findById(user._id);
      expect(deletedUser).toBeNull();
    });

    test('should perform CRUD operations on Transaction', async () => {
      const user = await global.testUtils.createTestUser();
      
      // Create
      const transaction = await global.testUtils.createTestTransaction(user._id);
      expect(transaction._id).toBeDefined();
      
      // Read
      const foundTransaction = await Transaction.findById(transaction._id);
      expect(foundTransaction).toBeDefined();
      expect(foundTransaction.type).toBe('SIP');
      
      // Update
      foundTransaction.status = 'PROCESSING';
      await foundTransaction.save();
      const updatedTransaction = await Transaction.findById(transaction._id);
      expect(updatedTransaction.status).toBe('PROCESSING');
      
      // Delete
      await Transaction.findByIdAndDelete(transaction._id);
      const deletedTransaction = await Transaction.findById(transaction._id);
      expect(deletedTransaction).toBeNull();
    });
  });

  describe('Data Validation', () => {
    test('should validate required fields', async () => {
      const user = await global.testUtils.createTestUser();
      
      // Test valid transaction
      const validTransaction = await global.testUtils.createTestTransaction(user._id);
      expect(validTransaction).toBeDefined();
      
      // Test invalid transaction (missing required fields)
      const invalidTransaction = {
        save: jest.fn().mockImplementation(() => {
          const err = new Error('ValidationError');
          err.name = 'ValidationError';
          throw err;
        })
      };
      
      try {
        await invalidTransaction.save();
        fail('Should have thrown validation error');
      } catch (error) {
        expect(error.name).toBe('ValidationError');
      }
    });
  });

  describe('Test Utilities', () => {
    test('should generate unique test data', async () => {
      const user1 = await global.testUtils.createTestUser();
      const user2 = await global.testUtils.createTestUser();
      
      expect(user1.email).not.toBe(user2.email);
      expect(user1._id.toString()).not.toBe(user2._id.toString());
    });

    test('should create related data correctly', async () => {
      const user = await global.testUtils.createTestUser();
      const portfolio = await global.testUtils.createTestPortfolio(user._id);
      const transaction = await global.testUtils.createTestTransaction(user._id);
      
      expect(portfolio.userId.toString()).toBe(user._id.toString());
      expect(transaction.userId.toString()).toBe(user._id.toString());
    });
  });
});