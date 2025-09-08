/**
 * Response utility functions for consistent API responses
 */

/**
 * Send success response
 */
const successResponse = (res, message = 'Success', data = null, statusCode = 200) => {
  const response = {
    success: true,
    message,
    timestamp: new Date().toISOString()
  };

  if (data !== null) {
    response.data = data;
  }

  return res.status(statusCode).json(response);
};

/**
 * Send error response
 */
const errorResponse = (res, message = 'Error', data = null, statusCode = 500) => {
  const response = {
    success: false,
    message,
    timestamp: new Date().toISOString(),
    data // Always include data field
  };

  // If error details are provided separately, add as error
  if (data && data.error) {
    response.error = data.error;
  }

  return res.status(statusCode).json(response);
};

module.exports = {
  successResponse,
  errorResponse
}; 