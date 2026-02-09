// mfApiClient.js
// Fetches Indian mutual fund scheme detail, nav returns, and nav history from custom API endpoints
const axios = require('axios');
const appConfig = require('../config/app');
const bseStarMFService = require('../services/bseStarMFService');
const demoBSEStarMFService = require('../services/demoBSEStarMFService');

const BASE_URL = 'https://nsunjzvf57.execute-api.ap-south-1.amazonaws.com/prod/api/v1';

/**
 * Fetch scheme detail (name, category, fund size, etc.)
 * @param {string|number} scheme_code
 * @returns {Promise<Object>} Scheme detail
 */
async function fetchSchemeDetail(scheme_code) {
  const url = `${BASE_URL}/scheme/detail`;
  const res = await axios.post(url, { scheme_code });
  return res.data.data;
}

/**
 * Fetch NAV returns (1D, 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y)
 * @param {string|number} scheme_code
 * @returns {Promise<Object>} NAV returns
 */
async function fetchNavReturns(scheme_code) {
  const url = `${BASE_URL}/nav/returns`;
  const res = await axios.post(url, { scheme_code });
  return res.data.data;
}

/**
 * Fetch NAV history for a date range
 * @param {string|number} scheme_code
 * @param {string} start_date (DD-MM-YYYY)
 * @param {string} end_date (DD-MM-YYYY)
 * @returns {Promise<Array>} NAV history
 */
async function fetchNavHistory(scheme_code, start_date, end_date) {
  const url = `${BASE_URL}/nav/history`;
  const res = await axios.post(url, { scheme_code, start_date, end_date });
  return res.data.data;
}

async function placeLumpsumOrder(orderData) {
  const service = appConfig.DEMO_MODE ? demoBSEStarMFService : bseStarMFService;
  return service.placeLumpsumOrder(orderData);
}

module.exports = {
  fetchSchemeDetail,
  fetchNavReturns,
  fetchNavHistory,
  placeLumpsumOrder
};
