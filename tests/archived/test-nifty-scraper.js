const { fetchNifty50HistoricalData } = require('./src/utils/niftyScraper');

(async () => {
  try {
    const data = await fetchNifty50HistoricalData();
    console.log('Fetched NIFTY 50 records:', data.length);
    console.log('First 5 records:', data.slice(0, 5));
  } catch (err) {
    console.error('Scraper error:', err);
  }
})(); 