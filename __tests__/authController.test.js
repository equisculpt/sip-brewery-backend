const authController = require('../src/controllers/authController');
const User = require('../src/models/User');
jest.mock('../src/models/User');

function mockRes() {
  const res = {};
  res.status = jest.fn().mockReturnValue(res);
  res.json = jest.fn().mockReturnValue(res);
  return res;
}

describe('authController', () => {
  afterEach(() => jest.clearAllMocks());

  describe('checkAuth', () => {
    it('should return 401 if no user', async () => {
      const req = {};
      const res = mockRes();
      await authController.checkAuth(req, res);
      expect(res.status).toHaveBeenCalledWith(401);
    });
  });

  describe('getKYCStatus', () => {
    it('should return 401 if no user', async () => {
      const req = {};
      const res = mockRes();
      await authController.getKYCStatus(req, res);
      expect(res.status).toHaveBeenCalledWith(401);
    });
  });

  describe('updateKYCStatus', () => {
    it('should return 401 if no user', async () => {
      const req = {};
      const res = mockRes();
      await authController.updateKYCStatus(req, res);
      expect(res.status).toHaveBeenCalledWith(401);
    });
  });

  describe('getUserProfile', () => {
    it('should return 401 if no user', async () => {
      const req = {};
      const res = mockRes();
      await authController.getUserProfile(req, res);
      expect(res.status).toHaveBeenCalledWith(401);
    });
  });

  describe('updateUserProfile', () => {
    it('should return 401 if no user', async () => {
      const req = {};
      const res = mockRes();
      await authController.updateUserProfile(req, res);
      expect(res.status).toHaveBeenCalledWith(401);
    });
  });

  // Additional tests for register, login, sendOtp, verifyOtp, forgotPassword, resetPassword, verifyEmail can be added similarly with mocks
});
