const rewardsService = require('../src/services/rewardsService');
const Reward = require('../src/models/Reward');
const RewardSummary = require('../src/models/RewardSummary');

jest.mock('../src/models/Reward');
jest.mock('../src/models/RewardSummary');

// Mock RewardSummary constructor to provide a save method
RewardSummary.mockImplementation(() => ({
  save: jest.fn().mockResolvedValue({ userId: '507f1f77bcf86cd799439011', totalPoints: 0 })
}));
// Mock countDocuments used in getRewardTransactions
Reward.countDocuments = jest.fn().mockResolvedValue(2);

describe('Rewards Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should return a user reward summary object', async () => {
    const userId = '507f1f77bcf86cd799439011';
    RewardSummary.findOne.mockResolvedValue({
      userId,
      totalPoints: 100,
      totalCashback: 200,
      totalReferralBonus: 300,
      totalSipInstallments: 5,
      pendingPayout: 0,
      totalPaidOut: 1000,
      lastUpdated: new Date(),
      save: jest.fn().mockResolvedValue(true),
    });
    Reward.find.mockReturnValue({
      sort: jest.fn().mockReturnThis(),
      limit: jest.fn().mockReturnThis(),
      select: jest.fn().mockResolvedValue([
        { type: 'REFERRAL_BONUS', amount: 100, points: 10, description: 'desc', status: 'PENDING', createdAt: new Date() }
      ])
    });
    const result = await rewardsService.getUserRewardSummary(userId);
    expect(result).toMatchObject({
      totalPoints: 100,
      totalCashback: 200,
      totalReferralBonus: 300,
      totalSipInstallments: 5,
      pendingPayout: 0,
      totalPaidOut: 1000,
      recentTransactions: expect.any(Array),
      lastUpdated: expect.any(Date)
    });
    expect(Array.isArray(result.recentTransactions)).toBe(true);
  });

  it('should create summary if not found', async () => {
    const userId = '507f1f77bcf86cd799439011';
    RewardSummary.findOne.mockResolvedValueOnce(null);
    Reward.find.mockReturnValue({
      sort: jest.fn().mockReturnThis(),
      limit: jest.fn().mockReturnThis(),
      select: jest.fn().mockResolvedValue([])
    });
    // Mock the RewardSummary constructor to return a summary with all fields
    RewardSummary.mockImplementationOnce(() => ({
      userId,
      totalPoints: 0,
      totalCashback: 0,
      totalReferralBonus: 0,
      totalSipInstallments: 0,
      pendingPayout: 0,
      totalPaidOut: 0,
      lastUpdated: new Date(),
      save: jest.fn().mockResolvedValue(true),
    }));
    const result = await rewardsService.getUserRewardSummary(userId);
    expect(result).toMatchObject({
      totalPoints: 0,
      totalCashback: 0,
      totalReferralBonus: 0,
      totalSipInstallments: 0,
      pendingPayout: 0,
      totalPaidOut: 0,
      recentTransactions: expect.any(Array),
      lastUpdated: expect.any(Date)
    });
    expect(Array.isArray(result.recentTransactions)).toBe(true);
  });

  it('should handle error in getUserRewardSummary', async () => {
    const userId = '507f1f77bcf86cd799439011';
    RewardSummary.findOne.mockRejectedValue(new Error('fail'));
    await expect(rewardsService.getUserRewardSummary(userId)).rejects.toThrow('fail');
  });

  it('should return paginated reward transactions', async () => {
    const userId = '507f1f77bcf86cd799439011';
    const options = { page: 1, limit: 10 };
    Reward.find.mockReturnValue({
      sort: jest.fn().mockReturnThis(),
      skip: jest.fn().mockReturnThis(),
      limit: jest.fn().mockReturnThis(),
      select: jest.fn().mockResolvedValue([{ amount: 10 }, { amount: 20 }])
    });
    Reward.countDocuments.mockResolvedValue(2);
    const result = await rewardsService.getRewardTransactions(userId, options);
    expect(result).toBeDefined();
    expect(result.transactions.length).toBe(2);
    expect(result.transactions[0].amount).toBe(10);
    expect(result).toMatchObject({
      transactions: expect.any(Array),
      pagination: {
        page: 1,
        limit: 10,
        total: 2,
        pages: 1
      }
    });
  });

  it('should handle error in getRewardTransactions', async () => {
    const userId = '507f1f77bcf86cd799439011';
    const options = { page: 1, limit: 10 };
    Reward.find.mockReturnValue({
      sort: jest.fn().mockReturnThis(),
      skip: jest.fn().mockReturnThis(),
      limit: jest.fn().mockReturnThis(),
      select: jest.fn().mockRejectedValue(new Error('fail'))
    });
    await expect(rewardsService.getRewardTransactions(userId, options)).rejects.toThrow('fail');
  });
});
