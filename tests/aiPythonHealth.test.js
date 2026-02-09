const pythonMlClient = require('../src/utils/pythonMlClient');

jest.mock('../src/utils/pythonMlClient');

describe('Python ML service health', () => {
  test('returns health status when service responds', async () => {
    pythonMlClient.health.mockResolvedValue({ status: 'ok' });

    const health = await pythonMlClient.health();

    expect(health).toEqual({ status: 'ok' });
    expect(pythonMlClient.health).toHaveBeenCalledTimes(1);
  });
});
