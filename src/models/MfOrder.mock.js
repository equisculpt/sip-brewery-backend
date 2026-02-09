module.exports = {
  find: jest.fn(),
  findOne: jest.fn(),
  findById: jest.fn(),
  create: jest.fn(),
  findByIdempotencyKey: jest.fn(),
  findPendingOrders: jest.fn(),
  findOrdersForReconciliation: jest.fn()
};
