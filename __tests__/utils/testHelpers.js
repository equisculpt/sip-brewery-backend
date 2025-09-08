/**
 * 🛠️ TEST UTILITIES
 * 
 * Common test helpers and utilities
 */

const mongoose = require('mongoose');

class TestHelpers {
  // Database helpers
  static async connectTestDB() {
    if (mongoose.connection.readyState === 0) {
      await mongoose.connect(process.env.MONGODB_URI);
    }
  }
  
  static async disconnectTestDB() {
    await mongoose.connection.close();
  }
  
  static async clearTestDB() {
    const collections = mongoose.connection.collections;
    for (const key in collections) {
      await collections[key].deleteMany({});
    }
  }
  
  // API testing helpers
  static createAuthHeader(token = 'test-token') {
    return { Authorization: `Bearer ${token}` };
  }
  
  static expectSuccessResponse(response) {
    expect(response.success).toBe(true);
    expect(response.data).toBeDefined();
  }
  
  static expectErrorResponse(response, message) {
    expect(response.success).toBe(false);
    expect(response.error).toContain(message);
  }
  
  // Mock data generators
  static generateMockFund() {
    return {
      code: 'TEST001',
      name: 'Test Mutual Fund',
      nav: 100.50,
      category: 'Equity',
      aum: 1000000000
    };
  }
  
  static generateMockPortfolio() {
    return {
      userId: global.testUtils.mockUser._id,
      holdings: [
        { fundCode: 'TEST001', units: 100, value: 10050 }
      ],
      totalValue: 10050
    };
  }
  
  // Performance testing
  static async measurePerformance(fn, iterations = 100) {
    const start = Date.now();
    
    for (let i = 0; i < iterations; i++) {
      await fn();
    }
    
    const end = Date.now();
    const avgTime = (end - start) / iterations;
    
    return {
      totalTime: end - start,
      averageTime: avgTime,
      iterations
    };
  }
}

module.exports = TestHelpers;
