// mutualFundDataFetcher.js
// Fetches Indian mutual fund NAVs (current and historical) using https://api.mfapi.in/mf
const axios = require('axios');

const BASE_URL = 'https://api.mfapi.in/mf';

/**
 * Fetch all mutual fund schemes metadata
 * @returns {Promise<Array>} List of all schemes (schemeCode, schemeName, etc.)
 */
async function fetchAllSchemes() {
  const url = BASE_URL;
  const res = await axios.get(url);
  return res.data;
}

/**
 * Fetch scheme details and full NAV history
 * @param {string} schemeCode
 * @returns {Promise<Object>} Scheme details + full NAV history
 */
async function fetchSchemeHistory(schemeCode) {
  const url = `${BASE_URL}/${schemeCode}`;
  const res = await axios.get(url);
  return res.data;
}

/**
 * Fetch current/latest NAV for a scheme
 * @param {string} schemeCode
 * @returns {Promise<Object>} Latest NAV object
 */
async function fetchLatestNav(schemeCode) {
  const data = await fetchSchemeHistory(schemeCode);
  if (data && data.data && data.data.length > 0) {
    return data.data[0]; // Latest NAV is first
  }
  return null;
}

module.exports = {
  fetchAllSchemes,
  fetchSchemeHistory,
  fetchLatestNav
};
