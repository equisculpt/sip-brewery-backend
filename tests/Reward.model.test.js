const mongoose = require('mongoose');
const Reward = require('../src/models/Reward');

jest.mock('mongoose', () => {
  const actual = jest.requireActual('mongoose');
  return {
    ...actual,
    model: jest.fn(() => ({ findById: jest.fn() })),
    Schema: actual.Schema,
    Types: actual.Types,
  };
});

describe('Reward model', () => {
  it('should be defined', () => {
    expect(Reward).toBeDefined();
  });

  it('should require userId, type, amount, and description', async () => {
    const reward = new Reward();
    let err;
    try {
      await reward.validate();
    } catch (e) {
      err = e;
    }
    expect(err).toBeDefined();
    expect(err.errors.userId).toBeDefined();
    expect(err.errors.type).toBeDefined();
    expect(err.errors.amount).toBeDefined();
    expect(err.errors.description).toBeDefined();
  });

  it('should enforce enum values for type and status', async () => {
    const reward = new Reward({
      userId: new mongoose.Types.ObjectId(),
      type: 'INVALID_TYPE',
      amount: 10,
      description: 'desc',
      status: 'INVALID_STATUS',
    });
    let err;
    try {
      await reward.validate();
    } catch (e) {
      err = e;
    }
    expect(err).toBeDefined();
    expect(err.errors.type).toBeDefined();
    expect(err.errors.status).toBeDefined();
  });

  it('should set default values', () => {
    const reward = new Reward({
      userId: new mongoose.Types.ObjectId(),
      type: 'REFERRAL_BONUS',
      amount: 100,
      description: 'desc',
    });
    expect(reward.status).toBe('PENDING');
    expect(reward.isPaid).toBe(false);
    expect(reward.isActive).toBe(true);
    expect(reward.points).toBe(0);
  });

  it('should call pre-save hook and throw if user does not exist', async () => {
    const reward = new Reward({
      userId: new mongoose.Types.ObjectId(),
      type: 'REFERRAL_BONUS',
      amount: 100,
      description: 'desc',
    });
    // Mock User model to simulate not found
    mongoose.model = jest.fn(() => ({
      findById: jest.fn().mockResolvedValue(null)
    }));
    let err;
    try {
      await reward.save();
    } catch (e) {
      err = e;
    }
    expect(err).toBeDefined();
    expect(err.message).toMatch(/Referenced user does not exist/);
  });
});
