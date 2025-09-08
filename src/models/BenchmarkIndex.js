const mongoose = require('mongoose');

const BenchmarkIndexSchema = new mongoose.Schema({
  indexId: { type: String, required: true }, // e.g., 'NIFTY50'
  name: { type: String, required: true },
  data: [
    {
      date: { type: String, required: true }, // 'YYYY-MM-DD'
      close: { type: Number, required: true }
    }
  ],
  lastUpdated: { type: Date, default: Date.now }
});

if (process.env.NODE_ENV === 'test') {
  const mockBenchmarkIndexModel = function (data) {
    return {
      ...data,
      save: jest.fn().mockResolvedValue({ _id: 'mockBenchmarkIndexId', ...data }),
    };
  };
  mockBenchmarkIndexModel.find = jest.fn().mockResolvedValue([]);
  mockBenchmarkIndexModel.findOne = jest.fn().mockResolvedValue(null);
  mockBenchmarkIndexModel.findById = jest.fn().mockResolvedValue(null);
  mockBenchmarkIndexModel.findByIdAndDelete = jest.fn().mockResolvedValue(null);
  mockBenchmarkIndexModel.create = jest.fn().mockResolvedValue({ _id: 'mockBenchmarkIndexId' });
  module.exports = mockBenchmarkIndexModel;
} else {
  module.exports = mongoose.model('BenchmarkIndex', BenchmarkIndexSchema);
}