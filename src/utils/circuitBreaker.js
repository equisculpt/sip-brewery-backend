// Circuit breaker utility using opossum
const CircuitBreaker = require('opossum');

function createBreaker(fn, options = {}) {
  const defaultOptions = {
    timeout: 10000,
    errorThresholdPercentage: 50,
    resetTimeout: 30000,
    ...options
  };
  return new CircuitBreaker(fn, defaultOptions);
}

module.exports = { createBreaker };
