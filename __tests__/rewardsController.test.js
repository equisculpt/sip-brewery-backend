const rewardsController = require('../src/controllers/rewardsController');
const rewardsService = require('../src/services/rewardsService');
jest.mock('../src/services/rewardsService');

function mockRes() {
  const res = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  return res;
}

describe('rewardsController', () => {
  afterEach(() => jest.clearAllMocks());

  it('should call service and return data for getRewardSummary', async () => {
    const req = { user: { supabaseId: 'u1' } };
    const res = mockRes();
    rewardsService.getUserRewardSummary.mockResolvedValue('summary-data');
    await rewardsController.getRewardSummary(req, res);
    expect(rewardsService.getUserRewardSummary).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalledWith({ success: true, data: 'summary-data' });
  });

  it('should return 400 if sipId missing for simulateSipReward', async () => {
    const req = { user: { supabaseId: 'u1' }, body: {} };
    const res = mockRes();
    await rewardsController.simulateSipReward(req, res);
    expect(res.status).toHaveBeenCalledWith(400);
  });

  // Similar tests can be written for other controller methods and error scenarios
});
