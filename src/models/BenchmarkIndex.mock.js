// Jest mock for BenchmarkIndex model
const mockModel = {
  find: jest.fn(),
  findOne: jest.fn(),
  findById: jest.fn(),
  create: jest.fn(),
  updateOne: jest.fn(),
  deleteOne: jest.fn(),
  aggregate: jest.fn(),
  save: jest.fn(),
  countDocuments: jest.fn(),
  // Add more static/instance methods as needed for your tests
};

module.exports = mockModel;
