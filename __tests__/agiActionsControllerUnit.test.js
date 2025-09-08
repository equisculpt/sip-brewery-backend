const agiActionsController = require('../src/controllers/agiActionsController');
const agiActionsService = require('../src/services/agiActionsService');

jest.mock('../src/services/agiActionsService');

function mockRes() {
  const res = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  return res;
}

describe('agiActionsController', () => {
  afterEach(() => jest.clearAllMocks());

  describe('executeAutonomousActions', () => {
    it('should return 400 if userId or actions missing', async () => {
      const req = { body: { userId: null, actions: null } };
      const res = mockRes();
      const next = jest.fn();
      await agiActionsController.executeAutonomousActions(req, res, next);
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ success: false, message: expect.stringMatching(/BrewBot/) });
    });
    it('should call service and return BrewBot success', async () => {
      const req = { body: { userId: 'u1', actions: ['act'] } };
      const res = mockRes();
      const next = jest.fn();
      agiActionsService.executeAutonomousActions.mockResolvedValue('result-data');
      await agiActionsController.executeAutonomousActions(req, res, next);
      expect(agiActionsService.executeAutonomousActions).toHaveBeenCalledWith('u1', ['act']);
      expect(res.json).toHaveBeenCalledWith({ success: true, message: expect.stringMatching(/BrewBot/), data: 'result-data' });
    });
  });

  describe('toggleAutonomousMode', () => {
    it('should return 400 if userId or enable missing', async () => {
      const req = { body: { userId: null, enable: 'notbool' } };
      const res = mockRes();
      const next = jest.fn();
      await agiActionsController.toggleAutonomousMode(req, res, next);
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ success: false, message: expect.stringMatching(/BrewBot/) });
    });
    it('should call service and return BrewBot success', async () => {
      const req = { body: { userId: 'u2', enable: true } };
      const res = mockRes();
      const next = jest.fn();
      agiActionsService.toggleAutonomousMode.mockResolvedValue('mode-result');
      await agiActionsController.toggleAutonomousMode(req, res, next);
      expect(agiActionsService.toggleAutonomousMode).toHaveBeenCalledWith('u2', true);
      expect(res.json).toHaveBeenCalledWith({ success: true, message: expect.stringMatching(/BrewBot/), data: 'mode-result' });
    });
  });

  describe('getAGIInsights', () => {
    it('should return 400 if userId missing', async () => {
      const req = { query: {} };
      const res = mockRes();
      const next = jest.fn();
      await agiActionsController.getAGIInsights(req, res, next);
      expect(res.status).toHaveBeenCalledWith(400);
      expect(res.json).toHaveBeenCalledWith({ success: false, message: expect.stringMatching(/BrewBot/) });
    });
    it('should call service and return BrewBot insights', async () => {
      const req = { query: { userId: 'u3' } };
      const res = mockRes();
      const next = jest.fn();
      agiActionsService.getAGIInsights.mockResolvedValue('insights-data');
      await agiActionsController.getAGIInsights(req, res, next);
      expect(agiActionsService.getAGIInsights).toHaveBeenCalledWith('u3');
      expect(res.json).toHaveBeenCalledWith({ success: true, message: expect.stringMatching(/BrewBot/), data: 'insights-data' });
    });
  });
});
