const express = require('express');
const router = express.Router();
const logger = require('../utils/logger');

// Investment routes are deprecated - use /api/bse-star-mf routes instead
router.use((req, res) => {
  logger.warn('Deprecated investment route accessed', { 
    path: req.path, 
    method: req.method 
  });
  
  res.status(410).json({
    success: false,
    message: 'This endpoint is deprecated. Please use /api/bse-star-mf routes instead.',
    migration: {
      '/investment/lumpsum': '/api/bse-star-mf/order/lumpsum',
      '/investment/sip': '/api/bse-star-mf/order/sip'
    }
  });
});

module.exports = router;
