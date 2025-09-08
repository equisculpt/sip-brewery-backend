const logger = require('../utils/logger');

module.exports = {
  getKYCStatus: async (req, res, next) => {
    try {
      logger.info('KYC status checked', { userId: req.params.userId, action: 'getKYCStatus' });
      res.status(200).json({ message: 'Stub: getKYCStatus' });
    } catch (err) {
      logger.error('KYC status error', { error: err.message, userId: req.params.userId });
      next(err);
    }
  },
  retriggerKYC: async (req, res, next) => {
    try {
      logger.info('KYC retrigger requested', { userId: req.params.userId, action: 'retriggerKYC' });
      res.status(200).json({ message: 'Stub: retriggerKYC' });
    } catch (err) {
      logger.error('KYC retrigger error', { error: err.message, userId: req.params.userId });
      next(err);
    }
  },
  getKYCLogs: async (req, res, next) => {
    try {
      logger.info('KYC logs requested', { userId: req.params.userId, action: 'getKYCLogs' });
      res.status(200).json({ message: 'Stub: getKYCLogs' });
    } catch (err) {
      logger.error('KYC logs error', { error: err.message, userId: req.params.userId });
      next(err);
    }
  }
};
