const transactionController = require('../src/controllers/transactionController');

function mockRes() {
  const res = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  return res;
}

describe('transactionController', () => {
  it('should return stub for getTransactionLogs', async () => {
    const req = {};
    const res = mockRes();
    await transactionController.getTransactionLogs(req, res);
    expect(res.status).toHaveBeenCalledWith(200);
    expect(res.json).toHaveBeenCalledWith({ message: 'Stub: getTransactionLogs' });
  });

  it('should return stub for getPendingTransactions', async () => {
    const req = {};
    const res = mockRes();
    await transactionController.getPendingTransactions(req, res);
    expect(res.status).toHaveBeenCalledWith(200);
    expect(res.json).toHaveBeenCalledWith({ message: 'Stub: getPendingTransactions' });
  });
});
