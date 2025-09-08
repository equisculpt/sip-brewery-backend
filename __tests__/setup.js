/**
 * 🧪 TEST SETUP CONFIGURATION
 * 
 * Global test setup and utilities
 */

// Global test timeout
jest.setTimeout(30000);

// Mock environment variables
process.env.NODE_ENV = 'test';
process.env.JWT_SECRET = 'test-jwt-secret-key-for-testing-only';
process.env.MONGODB_URI = 'mongodb://localhost:27017/sip-brewery-test';

// Global test utilities
global.testUtils = {
  // Mock user data
  mockUser: {
    _id: '507f1f77bcf86cd799439011',
    email: 'test@example.com',
    role: 'user',
    subscription: 'premium'
  },
  
  // Mock request/response
  mockReq: (overrides = {}) => ({
    body: {},
    params: {},
    query: {},
    headers: {},
    user: global.testUtils.mockUser,
    ...overrides
  }),
  
  mockRes: () => {
    const res = {};
    res.status = jest.fn().mockReturnValue(res);
    res.json = jest.fn().mockReturnValue(res);
    res.send = jest.fn().mockReturnValue(res);
    return res;
  },
  
  // Mock next function
  mockNext: jest.fn()
};

// Setup and teardown
beforeAll(async () => {
  console.log('🧪 Setting up test environment...');
});

afterAll(async () => {
  console.log('🧹 Cleaning up test environment...');
});

beforeEach(() => {
  // Clear all mocks
  jest.clearAllMocks();
});
