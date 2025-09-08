// testNiftyLiveFetch.js
// Quick test to fetch live Nifty 50 price using NSE public API
const axios = require('axios');

async function fetchLiveNifty50() {
  try {
    const headers = {
      'User-Agent': 'Mozilla/5.0',
      'Referer': 'https://www.nseindia.com/',
      'Accept': 'application/json'
    };
    // This endpoint returns index data for NIFTY 50
    const url = 'https://www.nseindia.com/api/equity-stockIndices?index=NIFTY%2050';
    const res = await axios.get(url, { headers });
    const data = res.data.data;
    if (data && data.length > 0) {
      const last = data[0];
      console.log(`Nifty 50 Live Price: ${last.last}`);
      return last.last;
    } else {
      console.log('Could not fetch live Nifty 50 price.');
      return null;
    }
  } catch (err) {
    console.error('Error fetching live Nifty 50 price:', err.message);
    return null;
  }
}

fetchLiveNifty50();
