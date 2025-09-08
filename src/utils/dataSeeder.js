const User = require('../models/User');
const Holding = require('../models/Holding');
const Transaction = require('../models/Transaction');
const Reward = require('../models/Reward');
const AIInsight = require('../models/AIInsight');
const logger = require('./logger');

class DataSeeder {
  /**
   * Seed all sample data
   */
  async seedAllData() {
    try {
      logger.info('Starting data seeding...');
      
      await this.seedUsers();
      await this.seedHoldings();
      await this.seedTransactions();
      await this.seedRewards();
      await this.seedAIInsights();
      
      logger.info('Data seeding completed successfully');
    } catch (error) {
      logger.error('Error seeding data:', error);
      throw error;
    }
  }

  /**
   * Seed sample users
   */
  async seedUsers() {
    try {
      const users = [
        {
          uid: 'dummy-uid-123',
          email: 'test@sipbrewery.com',
          name: 'Milin Raijada',
          mobile: '+91-9876543210',
          pan: 'ABCDE1234F',
          kycStatus: 'SUCCESS',
          riskProfile: 'Moderate',
          investorSince: new Date('2023-12-01'),
          profile: {
            dateOfBirth: new Date('1990-05-15'),
            address: {
              line1: '123 Main Street',
              line2: 'Apartment 4B',
              city: 'Mumbai',
              state: 'Maharashtra',
              pincode: '400001'
            },
            bankDetails: {
              accountNumber: '1234567890',
              ifscCode: 'SBIN0001234',
              bankName: 'State Bank of India'
            }
          },
          preferences: {
            notifications: {
              email: true,
              sms: true,
              push: true
            },
            theme: 'light'
          }
        }
      ];

      for (const userData of users) {
        const existingUser = await User.findOne({ uid: userData.uid });
        if (!existingUser) {
          const user = new User(userData);
          await user.save();
          logger.info(`Created user: ${userData.email}`);
        } else {
          logger.info(`User already exists: ${userData.email}`);
        }
      }
    } catch (error) {
      logger.error('Error seeding users:', error);
      throw error;
    }
  }

  /**
   * Seed sample holdings
   */
  async seedHoldings() {
    try {
      const holdings = [
        {
          userId: 'dummy-uid-123',
          schemeCode: 'HDFC001',
          schemeName: 'HDFC Flexi Cap Fund',
          folio: '123456789',
          units: 100.5,
          currentNav: 85.12,
          invested: 7000,
          sipStatus: 'ACTIVE',
          sipAmount: 5000,
          sipDate: 1,
          category: 'Flexi Cap',
          fundHouse: 'HDFC Mutual Fund',
          riskLevel: 'Moderate'
        },
        {
          userId: 'dummy-uid-123',
          schemeCode: 'SBI002',
          schemeName: 'SBI Bluechip Fund',
          folio: '123456789',
          units: 75.25,
          currentNav: 118.71,
          invested: 8000,
          sipStatus: 'ACTIVE',
          sipAmount: 3000,
          sipDate: 5,
          category: 'Large Cap',
          fundHouse: 'SBI Mutual Fund',
          riskLevel: 'Moderate'
        },
        {
          userId: 'dummy-uid-123',
          schemeCode: 'PARAG003',
          schemeName: 'Parag Parikh Flexi Cap Fund',
          folio: '123456789',
          units: 50.0,
          currentNav: 45.80,
          invested: 2000,
          sipStatus: 'ACTIVE',
          sipAmount: 2000,
          sipDate: 10,
          category: 'Flexi Cap',
          fundHouse: 'Parag Parikh Mutual Fund',
          riskLevel: 'Moderate'
        },
        {
          userId: 'dummy-uid-123',
          schemeCode: 'AXIS004',
          schemeName: 'Axis Midcap Fund',
          folio: '123456789',
          units: 30.0,
          currentNav: 65.45,
          invested: 1500,
          sipStatus: 'PAUSED',
          sipAmount: 1500,
          sipDate: 15,
          category: 'Mid Cap',
          fundHouse: 'Axis Mutual Fund',
          riskLevel: 'High'
        }
      ];

      for (const holdingData of holdings) {
        const existingHolding = await Holding.findOne({ 
          userId: holdingData.userId, 
          schemeCode: holdingData.schemeCode 
        });
        
        if (!existingHolding) {
          const holding = new Holding(holdingData);
          await holding.save();
          logger.info(`Created holding: ${holdingData.schemeName}`);
        } else {
          logger.info(`Holding already exists: ${holdingData.schemeName}`);
        }
      }
    } catch (error) {
      logger.error('Error seeding holdings:', error);
      throw error;
    }
  }

  /**
   * Seed sample transactions
   */
  async seedTransactions() {
    try {
      const transactions = [
        {
          userId: 'dummy-uid-123',
          type: 'SIP',
          schemeCode: 'HDFC001',
          schemeName: 'HDFC Flexi Cap Fund',
          folio: '123456789',
          amount: 5000,
          units: 58.73,
          nav: 85.12,
          date: new Date('2024-07-01'),
          status: 'SUCCESS',
          orderType: 'BUY',
          netAmount: 5000
        },
        {
          userId: 'dummy-uid-123',
          type: 'SIP',
          schemeCode: 'SBI002',
          schemeName: 'SBI Bluechip Fund',
          folio: '123456789',
          amount: 3000,
          units: 25.27,
          nav: 118.71,
          date: new Date('2024-07-05'),
          status: 'SUCCESS',
          orderType: 'BUY',
          netAmount: 3000
        },
        {
          userId: 'dummy-uid-123',
          type: 'LUMPSUM',
          schemeCode: 'PARAG003',
          schemeName: 'Parag Parikh Flexi Cap Fund',
          folio: '123456789',
          amount: 10000,
          units: 218.34,
          nav: 45.80,
          date: new Date('2024-06-15'),
          status: 'SUCCESS',
          orderType: 'BUY',
          netAmount: 10000
        },
        {
          userId: 'dummy-uid-123',
          type: 'REDEMPTION',
          schemeCode: 'AXIS004',
          schemeName: 'Axis Midcap Fund',
          folio: '123456789',
          amount: 2000,
          units: 30.55,
          nav: 65.45,
          date: new Date('2024-06-20'),
          status: 'SUCCESS',
          orderType: 'SELL',
          netAmount: 1950
        }
      ];

      for (const txnData of transactions) {
        const existingTxn = await Transaction.findOne({ 
          userId: txnData.userId, 
          schemeCode: txnData.schemeCode,
          date: txnData.date,
          amount: txnData.amount
        });
        
        if (!existingTxn) {
          const transaction = new Transaction(txnData);
          await transaction.save();
          logger.info(`Created transaction: ${txnData.type} - ${txnData.schemeName}`);
        } else {
          logger.info(`Transaction already exists: ${txnData.type} - ${txnData.schemeName}`);
        }
      }
    } catch (error) {
      logger.error('Error seeding transactions:', error);
      throw error;
    }
  }

  /**
   * Seed sample rewards
   */
  async seedRewards() {
    try {
      const rewards = [
        {
          userId: 'dummy-uid-123',
          type: 'SIGNUP',
          amount: 500,
          points: 500,
          description: 'Welcome bonus for new user registration',
          status: 'CREDITED'
        },
        {
          userId: 'dummy-uid-123',
          type: 'FIRST_SIP',
          amount: 200,
          points: 200,
          description: 'First SIP completion bonus',
          status: 'CREDITED'
        },
        {
          userId: 'dummy-uid-123',
          type: 'REFERRAL',
          amount: 200,
          points: 200,
          description: 'Referral bonus for new user',
          status: 'CREDITED'
        },
        {
          userId: 'dummy-uid-123',
          type: 'LOYALTY',
          amount: 100,
          points: 100,
          description: 'Monthly loyalty reward',
          status: 'CREDITED'
        },
        {
          userId: 'dummy-uid-123',
          type: 'CASHBACK',
          amount: 240,
          points: 240,
          description: 'Cashback on SIP transactions',
          status: 'CREDITED'
        }
      ];

      for (const rewardData of rewards) {
        const existingReward = await Reward.findOne({ 
          userId: rewardData.userId, 
          type: rewardData.type,
          description: rewardData.description
        });
        
        if (!existingReward) {
          const reward = new Reward(rewardData);
          await reward.save();
          logger.info(`Created reward: ${rewardData.type} - ${rewardData.description}`);
        } else {
          logger.info(`Reward already exists: ${rewardData.type} - ${rewardData.description}`);
        }
      }
    } catch (error) {
      logger.error('Error seeding rewards:', error);
      throw error;
    }
  }

  /**
   * Seed sample AI insights
   */
  async seedAIInsights() {
    try {
      const insights = [
        {
          userId: 'dummy-uid-123',
          type: 'PORTFOLIO_ANALYSIS',
          title: 'Portfolio Performance Analysis',
          summary: 'Your portfolio is performing well with a 14.5% XIRR, beating the market average.',
          details: 'Based on your current holdings, your portfolio shows strong diversification across different market caps. The HDFC Flexi Cap Fund and Parag Parikh Flexi Cap Fund are your top performers.',
          metrics: {
            xirr: 14.5,
            volatility: 12.3,
            sharpeRatio: 1.2,
            beta: 0.95,
            alpha: 2.1,
            maxDrawdown: -8.5,
            correlation: 0.85
          },
          recommendations: [
            {
              action: 'Increase SIP in smallcap fund',
              reason: 'Better diversification and growth potential',
              priority: 'MEDIUM',
              fundCode: 'AXIS004',
              amount: 2000
            },
            {
              action: 'Consider adding international exposure',
              reason: 'Reduce country-specific risk',
              priority: 'LOW',
              fundCode: null,
              amount: 0
            }
          ],
          riskFactors: [
            {
              factor: 'Market volatility',
              impact: 'MEDIUM',
              description: 'Current market conditions may affect short-term performance'
            }
          ],
          marketContext: {
            marketTrend: 'Bullish',
            sectorPerformance: {
              'IT': 15.2,
              'Banking': 8.5,
              'Pharma': 12.1
            },
            economicFactors: ['GDP Growth', 'Interest Rates', 'Inflation']
          }
        }
      ];

      for (const insightData of insights) {
        const existingInsight = await AIInsight.findOne({ 
          userId: insightData.userId, 
          type: insightData.type,
          title: insightData.title
        });
        
        if (!existingInsight) {
          const insight = new AIInsight(insightData);
          await insight.save();
          logger.info(`Created AI insight: ${insightData.title}`);
        } else {
          logger.info(`AI insight already exists: ${insightData.title}`);
        }
      }
    } catch (error) {
      logger.error('Error seeding AI insights:', error);
      throw error;
    }
  }
}

module.exports = new DataSeeder(); 