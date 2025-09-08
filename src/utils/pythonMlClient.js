// pythonMlClient.js
// Utility to call the Python analytics/ML microservice from Node.js
const axios = require('axios');

const PYTHON_SERVICE_URL = process.env.PYTHON_SERVICE_URL || 'http://localhost:5001';

module.exports = {
  /**
   * Call mean endpoint
   * @param {Array<number>} data
   */
  async mean(data) {
    const res = await axios.post(`${PYTHON_SERVICE_URL}/statistical/mean`, { data });
    return res.data.mean;
  },

  /**
   * Call value-at-risk endpoint
   * @param {Array<number>} data
   * @param {number} confidenceLevel
   */
  async valueAtRisk(data, confidenceLevel) {
    const res = await axios.post(`${PYTHON_SERVICE_URL}/risk/var`, { data, confidence_level: confidenceLevel });
    return res.data.var;
  },

  /**
   * Call LSTM prediction endpoint
   * @param {Array<number>} series
   */
  async lstmPredict(series) {
    const res = await axios.post(`${PYTHON_SERVICE_URL}/ml/lstm`, { series });
    return res.data.prediction;
  },

  /**
   * Call SHAP explainability endpoint
   * @param {Array<number>} features
   */
  async shapExplain(features) {
    const res = await axios.post(`${PYTHON_SERVICE_URL}/explain/shap`, { features });
    return res.data.shap_values;
  },

  // Add more methods for other endpoints as needed
};
