// Universal input sanitization middleware for Express
const { sanitize } = require('express-validator');

module.exports = function sanitizeInput(req, res, next) {
  // Recursively sanitize all string fields in req.body, req.query, req.params
  function deepSanitize(obj) {
    if (typeof obj !== 'object' || obj === null) return obj;
    for (const key of Object.keys(obj)) {
      if (typeof obj[key] === 'string') {
        obj[key] = sanitize(obj[key]).trim().escape();
      } else if (typeof obj[key] === 'object') {
        obj[key] = deepSanitize(obj[key]);
      }
    }
    return obj;
  }
  req.body = deepSanitize(req.body);
  req.query = deepSanitize(req.query);
  req.params = deepSanitize(req.params);
  next();
};
