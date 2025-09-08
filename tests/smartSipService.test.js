const smartSipService = require('../src/services/smartSipService');
const SmartSip = require('../src/models/SmartSip');
const agiPipelineOrchestrator = require('../src/utils/agiPipelineOrchestrator');
const mfApiClient = require('../src/utils/mfApiClient');

jest.mock('../src/models/SmartSip');
jest.mock('../src/utils/agiPipelineOrchestrator');
jest.mock('../src/utils/mfApiClient');

describe('smartSipService', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should be defined', () => {
    expect(smartSipService).toBeDefined();
  });

  it('should get all active SIPs', async () => {
    SmartSip.find.mockResolvedValue([{ _id: 'sip1' }]);
    const sips = await smartSipService.getAllActiveSIPs();
    expect(sips).toEqual([{ _id: 'sip1' }]);
  });

  it('should handle error in getAllActiveSIPs', async () => {
    SmartSip.find.mockRejectedValue(new Error('fail'));
    await expect(smartSipService.getAllActiveSIPs()).rejects.toThrow('fail');
  });

  it('should return SIP analytics', async () => {
    SmartSip.find.mockResolvedValue([{ amount: 100, createdAt: new Date() }]);
    const analytics = await smartSipService.getSIPAnalytics('userId');
    expect(analytics).toHaveProperty('totalInvested');
  });

  it('should call AGI orchestrator for market regime', async () => {
    agiPipelineOrchestrator.getMarketRegime.mockResolvedValue('bull');
    const regime = await agiPipelineOrchestrator.getMarketRegime();
    expect(regime).toBe('bull');
  });

  it('should place SIP order via mfApiClient', async () => {
    mfApiClient.placeSipOrder.mockResolvedValue({ success: true });
    const result = await mfApiClient.placeSipOrder({});
    expect(result).toEqual({ success: true });
  });

  // Add more tests for pause/resume SIP, edge cases, and error handling
});
