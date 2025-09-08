// Test setup and global configurations
const mongoose = require('mongoose');

// Setup test database
beforeAll(async () => {
  // Connect to test database
  if (process.env.NODE_ENV !== 'test') {
    process.env.NODE_ENV = 'test';
  }
});

afterAll(async () => {
  // Cleanup after tests
  if (mongoose.connection.readyState === 1) {
    await mongoose.connection.close();
  }
});

// Global test utilities
global.testUtils = {
  // Add common test utilities here
};
