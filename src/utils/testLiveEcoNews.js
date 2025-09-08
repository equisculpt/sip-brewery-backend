// testLiveEcoNews.js
// Test fetching live economics news and government decisions from NewsAPI and data.gov.in
const { fetchNews } = require('./marketDataFetcher');
const axios = require('axios');

(async () => {
  // Test NewsAPI (if key available)
  try {
    const news = await fetchNews('india economy');
    if (news.articles && news.articles.length > 0) {
      console.log('Sample NewsAPI headline:', news.articles[0].title);
    } else {
      console.log('No articles found from NewsAPI.');
    }
  } catch (err) {
    console.error('NewsAPI error:', err.message);
  }

  // Test scraping data.gov.in for recent government economic datasets
  try {
    const url = 'https://data.gov.in/catalog/union-budget'; // Example: Union Budget datasets
    const res = await axios.get(url);
    const match = res.data.match(/<a [^>]*href="([^"]+\.csv)"/i);
    if (match) {
      console.log('Sample data.gov.in CSV link:', match[1]);
    } else {
      console.log('No CSV links found on data.gov.in Union Budget page.');
    }
  } catch (err) {
    console.error('data.gov.in scrape error:', err.message);
  }
})();
