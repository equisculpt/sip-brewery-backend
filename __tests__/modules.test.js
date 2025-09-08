const mongoose = require('mongoose');

describe('Module Tests', () => {
  test('should load all models', () => {
    const { User, Transaction, UserPortfolio, SmartSip, Reward, WhatsAppSession } = require('../src/models');
    expect(User).toBeDefined();
    expect(Transaction).toBeDefined();
    expect(UserPortfolio).toBeDefined();
    expect(SmartSip).toBeDefined();
    expect(Reward).toBeDefined();
    expect(WhatsAppSession).toBeDefined();
  });

  test('should load all controllers', () => {
    const authController = require('../src/controllers/authController');
    const dashboardController = require('../src/controllers/dashboardController');
    const smartSipController = require('../src/controllers/smartSipController');
    
    expect(authController).toBeDefined();
    expect(dashboardController).toBeDefined();
    expect(smartSipController).toBeDefined();
  });

  test('should load existing services', () => {
    const portfolioAnalyticsService = require('../src/services/portfolioAnalyticsService');
    const smartSipService = require('../src/services/smartSipService');
    const rewardsService = require('../src/services/rewardsService');
    
    expect(portfolioAnalyticsService).toBeDefined();
    expect(smartSipService).toBeDefined();
    expect(rewardsService).toBeDefined();
  });
});