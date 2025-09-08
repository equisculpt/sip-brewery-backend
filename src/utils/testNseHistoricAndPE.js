// testNseHistoricAndPE.js
// Test fetching historical NSE data and P/E, P/B ratios
const { fetchLatestNseBhavcopy, fetchNseIndexRatios } = require('./nseBseBulkDownloader');
const path = require('path');

(async () => {
  const destDir = path.join(__dirname, 'data');
  console.log('Fetching latest NSE Bhavcopy (historical EOD)...');
  const bhavData = await fetchLatestNseBhavcopy(destDir);
  if (Array.isArray(bhavData)) {
    console.log(`Sample NSE Bhavcopy row:`, bhavData[0]);
  } else {
    console.error('Bhavcopy fetch error:', bhavData);
  }

  console.log('\nFetching latest NSE index ratios (P/E, P/B, Div Yield)...');
  const ratios = await fetchNseIndexRatios();
  if (Array.isArray(ratios)) {
    const niftyRow = ratios.find(r => r['Index Name'] && r['Index Name'].includes('NIFTY 50'));
    console.log('NIFTY 50 Index Ratios:', niftyRow);
  } else {
    console.error('Index ratios fetch error:', ratios);
  }
})();
