const mongoose = require('mongoose');
const { User, Transaction, UserPortfolio } = require('../src/models');

describe('Performance Tests', () => {
  test('should handle bulk user creation', async () => {
    const startTime = Date.now();
    const users = [];
    
    for (let i = 0; i < 10; i++) {
      const user = await global.testUtils.createTestUser({
        name: `Performance User ${i}`,
        email: `perf${i}@example.com`
      });
      users.push(user);
    }
    
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    expect(users).toHaveLength(10);
    expect(duration).toBeLessThan(5000); // Should complete within 5 seconds
  });

  test('should handle bulk transaction creation', async () => {
    const user = await global.testUtils.createTestUser();
    const startTime = Date.now();
    const transactions = [];
    
    for (let i = 0; i < 5; i++) {
      const transaction = await global.testUtils.createTestTransaction(user._id, {
        amount: 1000 + i * 100,
        units: 10 + i,
        nav: 100 + i
      });
      transactions.push(transaction);
    }
    
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    expect(transactions).toHaveLength(5);
    expect(duration).toBeLessThan(3000); // Should complete within 3 seconds
  });
});