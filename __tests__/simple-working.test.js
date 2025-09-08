const mongoose = require('mongoose');

describe('Simple Working Test', () => {
  test('should connect to database', async () => {
    expect(mongoose.connection.readyState).toBe(1);
  });

  test('should create test user', async () => {
    const user = await global.testUtils.createTestUser();
    expect(user).toBeDefined();
    expect(user.name).toBe('Test User');
  });

  test('should create test portfolio', async () => {
    const user = await global.testUtils.createTestUser();
    const portfolio = await global.testUtils.createTestPortfolio(user._id);
    expect(portfolio).toBeDefined();
    expect(portfolio.userId.toString()).toBe(user._id.toString());
  });

  test('should create test transaction', async () => {
    const user = await global.testUtils.createTestUser();
    const transaction = await global.testUtils.createTestTransaction(user._id);
    expect(transaction).toBeDefined();
    expect(transaction.userId.toString()).toBe(user._id.toString());
    expect(transaction.nav).toBeDefined();
  });
});