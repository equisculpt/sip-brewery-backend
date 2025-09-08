const aiPortfolioController = require('../src/controllers/aiPortfolioController');
const aiPortfolioOptimizer = require('../src/services/aiPortfolioOptimizer');
const { User, UserPortfolio, Holding } = require('../src/models');
jest.mock('../src/services/aiPortfolioOptimizer');
jest.mock('../src/models', () => ({
  User: { findById: jest.fn() },
  UserPortfolio: { findOne: jest.fn() },
  Holding: { find: jest.fn() }
}));

function mockRes() {
  const res = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  return res;
}

describe('aiPortfolioController', () => {
  afterEach(() => jest.clearAllMocks());

  it('should return 400 for missing riskTolerance/goals in optimizePortfolio', async () => {
    const req = { user: { userId: 'u1' }, body: {} };
    const res = mockRes();
    await aiPortfolioController.optimizePortfolio(req, res);
    expect(res.status).toHaveBeenCalledWith(400);
  });

  it('should return 404 for missing user in optimizePortfolio', async () => {
    const req = { user: { userId: 'u1' }, body: { riskTolerance: 'MODERATE', goals: [] } };
    const res = mockRes();
    User.findById.mockResolvedValue(null);
    await aiPortfolioController.optimizePortfolio(req, res);
    expect(res.status).toHaveBeenCalledWith(404);
  });

  it('should return 404 for missing portfolio in optimizePortfolio', async () => {
    const req = { user: { userId: 'u1' }, body: { riskTolerance: 'MODERATE', goals: [] } };
    const res = mockRes();
    User.findById.mockResolvedValue({});
    UserPortfolio.findOne.mockResolvedValue(null);
    await aiPortfolioController.optimizePortfolio(req, res);
    expect(res.status).toHaveBeenCalledWith(404);
  });

  // Add more tests for other methods as needed
});
