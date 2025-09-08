it('controller dummy', () => { expect(true).toBe(true); });

const dashboardService = require('../src/services/dashboardService');
jest.mock('../src/services/dashboardService');

function mockRes() {
  const res = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  return res;
}

describe('dashboardController', () => {
  afterEach(() => jest.clearAllMocks());

  it('should return 401 if no userId for getDashboard', async () => {
    const req = {};
    const res = mockRes();
    await dashboardController.getDashboard(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('should call service and return data for getDashboard', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getDashboardData.mockResolvedValue('dashboard-data');
    await dashboardController.getDashboard(req, res);
    expect(dashboardService.getDashboardData).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });

    it('should return 401 if no userId for getHoldings', async () => {
    const req = {};
    const res = mockRes();
    await dashboardController.getHoldings(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('should call service and return data for getHoldings', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getHoldings.mockResolvedValue('holdings-data');
    await dashboardController.getHoldings(req, res);
    expect(dashboardService.getHoldings).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle error in getHoldings', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getHoldings.mockRejectedValue(new Error('fail-holdings'));
    await dashboardController.getHoldings(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle error object in getRewards and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    const error = new Error('fail-rewards');
    dashboardService.getRewards.mockRejectedValue(error);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getRewards(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching rewards:', error);
    loggerSpy.mockRestore();
  });

  it('should handle null error in getRewards and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getRewards.mockRejectedValue(null);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getRewards(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching rewards:', null);
    loggerSpy.mockRestore();
  });

  it('should handle undefined error in getRewards and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getRewards.mockRejectedValue(undefined);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getRewards(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching rewards:', undefined);
    loggerSpy.mockRestore();
  });

  it('should handle primitive error in getRewards and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getRewards.mockRejectedValue('fail');
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getRewards(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching rewards:', 'fail');
    loggerSpy.mockRestore();
  });

  it('should handle empty object error in getRewards and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getRewards.mockRejectedValue({});
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getRewards(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching rewards:', {});
    loggerSpy.mockRestore();
  });

  it('should return 401 if no userId for getSmartSIPCenter', async () => {
    const req = {};
    const res = mockRes();
    await dashboardController.getSmartSIPCenter(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('should call service and return data for getSmartSIPCenter', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getSmartSIPCenter.mockResolvedValue('sip-center-data');
    await dashboardController.getSmartSIPCenter(req, res);
    expect(dashboardService.getSmartSIPCenter).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle error in getSmartSIPCenter', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getSmartSIPCenter.mockRejectedValue(new Error('fail-sip-center'));
    await dashboardController.getSmartSIPCenter(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle falsy/empty result from getSmartSIPCenter and log fallback', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getSmartSIPCenter = jest.fn().mockResolvedValue(undefined);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getSmartSIPCenter(req, res);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).not.toHaveBeenCalled(); // fallback does not log error unless thrown
    dashboardService.getSmartSIPCenter = jest.fn().mockResolvedValue(null);
    await dashboardController.getSmartSIPCenter(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getSmartSIPCenter = jest.fn().mockResolvedValue(false);
    await dashboardController.getSmartSIPCenter(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getSmartSIPCenter = jest.fn().mockResolvedValue({});
    await dashboardController.getSmartSIPCenter(req, res);
    expect(res.json).toHaveBeenCalled();
    loggerSpy.mockRestore();
  });

  it('should return 401 if no userId for getTransactions', async () => {
    const req = { query: {} };
    const res = mockRes();
    await dashboardController.getTransactions(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('should call service and return data for getTransactions', async () => {
    const req = { userId: 'u1', query: { limit: 5 } };
    const res = mockRes();
    dashboardService.getTransactions.mockResolvedValue('tx-data');
    await dashboardController.getTransactions(req, res);
    expect(dashboardService.getTransactions).toHaveBeenCalledWith('u1', 5);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle error in getTransactions', async () => {
    const req = { userId: 'u1', query: { limit: 5 } };
    const res = mockRes();
    dashboardService.getTransactions.mockRejectedValue(new Error('fail-tx'));
    await dashboardController.getTransactions(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle falsy/empty result from getTransactions and log fallback', async () => {
    const req = { userId: 'u1', query: {} };
    const res = mockRes();
    dashboardService.getTransactions = jest.fn().mockResolvedValue(undefined);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getTransactions(req, res);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).not.toHaveBeenCalled();
    dashboardService.getTransactions = jest.fn().mockResolvedValue(null);
    await dashboardController.getTransactions(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getTransactions = jest.fn().mockResolvedValue(false);
    await dashboardController.getTransactions(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getTransactions = jest.fn().mockResolvedValue({});
    await dashboardController.getTransactions(req, res);
    expect(res.json).toHaveBeenCalled();
    loggerSpy.mockRestore();
  });

  it('should return 401 if no userId for getStatements', async () => {
    const req = {};
    const res = mockRes();
    await dashboardController.getStatements(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('should call service and return data for getStatements', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getStatements.mockResolvedValue('stmt-data');
    await dashboardController.getStatements(req, res);
    expect(dashboardService.getStatements).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle error in getStatements and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    const error = new Error('fail-statements');
    dashboardService.getStatements.mockRejectedValue(error);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getStatements(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching statements:', error);
    loggerSpy.mockRestore();
  });

  it('should handle null error in getStatements', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getStatements.mockRejectedValue(null);
    await dashboardController.getStatements(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle undefined error in getStatements', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getStatements.mockRejectedValue(undefined);
    await dashboardController.getStatements(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle primitive error in getStatements', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getStatements.mockRejectedValue('fail');
    await dashboardController.getStatements(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle falsy service result in getStatements', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getStatements.mockResolvedValue(undefined);
    await dashboardController.getStatements(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getStatements.mockResolvedValue(null);
    await dashboardController.getStatements(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getStatements.mockResolvedValue(false);
    await dashboardController.getStatements(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getStatements.mockResolvedValue({});
    await dashboardController.getStatements(req, res);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle null error in getStatements and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getStatements.mockRejectedValue(null);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getStatements(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching statements:', null);
    loggerSpy.mockRestore();
  });

  it('should handle undefined error in getStatements and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getStatements.mockRejectedValue(undefined);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getStatements(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching statements:', undefined);
    loggerSpy.mockRestore();
  });

  it('should handle primitive error in getStatements and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getStatements.mockRejectedValue('fail');
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getStatements(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching statements:', 'fail');
    loggerSpy.mockRestore();
  });

  it('should return 401 if no userId for getPerformanceChart', async () => {
    const req = {};
    const res = mockRes();
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('should call service and return data for getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart.mockResolvedValue('performance-chart-data');
    await dashboardController.getPerformanceChart(req, res);
    expect(dashboardService.getPerformanceChart).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle error in getPerformanceChart and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    const error = new Error('fail-performance-chart');
    dashboardService.getPerformanceChart.mockRejectedValue(error);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching performance chart:', error);
    loggerSpy.mockRestore();
  });

  it('should handle null error in getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart.mockRejectedValue(null);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle undefined error in getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart.mockRejectedValue(undefined);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle primitive error in getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart.mockRejectedValue('fail');
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle falsy service result in getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart.mockResolvedValue(undefined);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPerformanceChart.mockResolvedValue(null);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPerformanceChart.mockResolvedValue(false);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPerformanceChart.mockResolvedValue({});
    await dashboardController.getPerformanceChart(req, res);
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle null error in getPerformanceChart and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart.mockRejectedValue(null);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching performance chart:', null);
    loggerSpy.mockRestore();
  });

  it('should handle undefined error in getPerformanceChart and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart.mockRejectedValue(undefined);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching performance chart:', undefined);
    loggerSpy.mockRestore();
  });

  it('should handle primitive error in getPerformanceChart and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart.mockRejectedValue('fail');
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching performance chart:', 'fail');
    loggerSpy.mockRestore();
  });

  it('should return 401 if no userId for getAIInsights', async () => {
    const req = {};
    const res = mockRes();
    await dashboardController.getAIInsights(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
  });

  it('should call service and return data for getAIInsights', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getAIInsights = jest.fn().mockResolvedValue({ data: 'insights-data' });
    await dashboardController.getAIInsights(req, res);
    expect(dashboardService.getAIInsights).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalledWith(expect.objectContaining({ success: true }));
  });

  it('should call service and return mock data if result is undefined in getAIInsights and assert response', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    jest.resetModules();
    jest.doMock('../src/services/aiPortfolioOptimizer', () => ({ optimizePortfolio: jest.fn().mockResolvedValue(undefined) }));
    await dashboardController.getAIInsights(req, res);
    expect(res.json).toHaveBeenCalledWith(expect.objectContaining({
      success: true,
      message: 'AI-driven portfolio insights retrieved',
      data: expect.objectContaining({
        portfolioAnalysis: expect.any(Object),
        recommendations: expect.any(Object),
        marketInsights: expect.any(Object)
      })
    }));
  });

  it('should handle error object in getReferralData and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    const error = new Error('fail-referral');
    dashboardService.getReferralData = jest.fn().mockRejectedValue(error);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getReferralData(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching referral data:', error);
    loggerSpy.mockRestore();
  });

  it('should handle null error in getReferralData and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getReferralData = jest.fn().mockRejectedValue(null);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getReferralData(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching referral data:', null);
    loggerSpy.mockRestore();
  });

  it('should handle undefined error in getReferralData and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getReferralData = jest.fn().mockRejectedValue(undefined);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getReferralData(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching referral data:', undefined);
    loggerSpy.mockRestore();
  });

  it('should handle primitive error in getReferralData and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getReferralData = jest.fn().mockRejectedValue('fail');
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getReferralData(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching referral data:', 'fail');
    loggerSpy.mockRestore();
  });

  it('should handle empty object error in getReferralData and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getReferralData = jest.fn().mockRejectedValue({});
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getReferralData(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching referral data:', {});
    loggerSpy.mockRestore();
  });

  it('should handle null error in getAIAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getAIAnalytics = jest.fn().mockRejectedValue(null);
    await dashboardController.getAIAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

  it('should call service and return data for getAIAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getAIAnalytics = jest.fn().mockResolvedValue('ai-analytics-data');
    await dashboardController.getAIAnalytics(req, res);
    expect(dashboardService.getAIAnalytics).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });

  it('should handle error in getAIAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getAIAnalytics = jest.fn().mockRejectedValue(new Error('fail-ai-analytics'));
    await dashboardController.getAIAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });

    await dashboardController.getAIAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle undefined error in getAIAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getAIAnalytics = jest.fn().mockRejectedValue(undefined);
    await dashboardController.getAIAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle primitive error in getAIAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getAIAnalytics = jest.fn().mockRejectedValue('fail');
    await dashboardController.getAIAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle falsy service result in getAIAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getAIAnalytics = jest.fn().mockResolvedValue(undefined);
    await dashboardController.getAIAnalytics(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getAIAnalytics = jest.fn().mockResolvedValue(null);
    await dashboardController.getAIAnalytics(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getAIAnalytics = jest.fn().mockResolvedValue(false);
    await dashboardController.getAIAnalytics(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getAIAnalytics = jest.fn().mockResolvedValue({});
    await dashboardController.getAIAnalytics(req, res);
    expect(res.json).toHaveBeenCalled();
  });

  it('should call service and return data for getPortfolioAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPortfolioAnalytics = jest.fn().mockResolvedValue('portfolio-analytics-data');
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(dashboardService.getPortfolioAnalytics).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle error in getPortfolioAnalytics and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    const error = new Error('fail-portfolio-analytics');
    dashboardService.getPortfolioAnalytics = jest.fn().mockRejectedValue(error);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching portfolio analytics:', error);
    loggerSpy.mockRestore();
  });
  it('should handle null error in getPortfolioAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPortfolioAnalytics = jest.fn().mockRejectedValue(null);
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle undefined error in getPortfolioAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPortfolioAnalytics = jest.fn().mockRejectedValue(undefined);
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle primitive error in getPortfolioAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPortfolioAnalytics = jest.fn().mockRejectedValue('fail');
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle falsy service result in getPortfolioAnalytics', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPortfolioAnalytics = jest.fn().mockResolvedValue(undefined);
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPortfolioAnalytics = jest.fn().mockResolvedValue(null);
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPortfolioAnalytics = jest.fn().mockResolvedValue(false);
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPortfolioAnalytics = jest.fn().mockResolvedValue({});
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.json).toHaveBeenCalled();
  });

  it('should return 401 if no userId for getProfile', async () => {
    const req = {};
    const res = mockRes();
    await dashboardController.getProfile(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalled();
  });

  it('should call service and return data for getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart = jest.fn().mockResolvedValue('performance-chart-data');
    await dashboardController.getPerformanceChart(req, res);
    expect(dashboardService.getPerformanceChart).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle error in getPerformanceChart and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    const error = new Error('fail-performance-chart');
    dashboardService.getPerformanceChart = jest.fn().mockRejectedValue(error);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching performance chart:', error);
    loggerSpy.mockRestore();
  });
  it('should handle null error in getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart = jest.fn().mockRejectedValue(null);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle undefined error in getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart = jest.fn().mockRejectedValue(undefined);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle primitive error in getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart = jest.fn().mockRejectedValue('fail');
    await dashboardController.getPerformanceChart(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle falsy service result in getPerformanceChart', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getPerformanceChart = jest.fn().mockResolvedValue(undefined);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPerformanceChart = jest.fn().mockResolvedValue(null);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPerformanceChart = jest.fn().mockResolvedValue(false);
    await dashboardController.getPerformanceChart(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getPerformanceChart = jest.fn().mockResolvedValue({});
    await dashboardController.getPerformanceChart(req, res);
    expect(res.json).toHaveBeenCalled();
  });

  it('should call service and return data for getProfile', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getProfile = jest.fn().mockResolvedValue('profile-data');
    await dashboardController.getProfile(req, res);
    expect(dashboardService.getProfile).toHaveBeenCalledWith('u1');
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle error in getProfile and log error', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    const error = new Error('fail-profile');
    dashboardService.getProfile = jest.fn().mockRejectedValue(error);
    const loggerSpy = jest.spyOn(logger, 'error');
    await dashboardController.getProfile(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
    expect(loggerSpy).toHaveBeenCalledWith('Error fetching profile:', error);
    loggerSpy.mockRestore();
  });
  it('should handle null error in getProfile', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getProfile = jest.fn().mockRejectedValue(null);
    await dashboardController.getProfile(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle undefined error in getProfile', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getProfile = jest.fn().mockRejectedValue(undefined);
    await dashboardController.getProfile(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle primitive error in getProfile', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getProfile = jest.fn().mockRejectedValue('fail');
    await dashboardController.getProfile(req, res);
    expect(res.status).toHaveBeenCalledWith(500);
    expect(res.json).toHaveBeenCalled();
  });
  it('should handle falsy service result in getProfile', async () => {
    const req = { userId: 'u1' };
    const res = mockRes();
    dashboardService.getProfile = jest.fn().mockResolvedValue(undefined);
    await dashboardController.getProfile(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getProfile = jest.fn().mockResolvedValue(null);
    await dashboardController.getProfile(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getProfile = jest.fn().mockResolvedValue(false);
    await dashboardController.getProfile(req, res);
    expect(res.json).toHaveBeenCalled();
    dashboardService.getProfile = jest.fn().mockResolvedValue({});
    await dashboardController.getProfile(req, res);
    expect(res.json).toHaveBeenCalled();
  });

  it('should return 401 if no userId for getPortfolioAnalytics', async () => {
    const req = {};
    const res = mockRes();
    await dashboardController.getPortfolioAnalytics(req, res);
    expect(res.status).toHaveBeenCalledWith(401);
    expect(res.json).toHaveBeenCalled();
  });

});
