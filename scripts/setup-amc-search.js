#!/usr/bin/env node

/**
 * 🚀 AMC SEARCH SETUP SCRIPT
 * 
 * Sets up and initializes the AMC search integration
 * Indexes all AMC data and optionally crawls websites
 * 
 * Usage: node scripts/setup-amc-search.js [--crawl]
 */

const { AMCSearchIntegration } = require('../src/finance_crawler/amc-search-integration');
const logger = require('../src/utils/logger');

async function setupAMCSearch() {
  try {
    console.log('🏦 Setting up AMC Search Integration...\n');
    
    // Initialize AMC Search Integration
    const amcSearch = new AMCSearchIntegration({
      indexName: 'amc-data',
      batchSize: 50,
      requestDelay: 2000
    });
    
    // Initialize and index AMC data
    console.log('📊 Initializing AMC data...');
    await amcSearch.initialize();
    
    // Check if crawling is requested
    const shouldCrawl = process.argv.includes('--crawl');
    
    if (shouldCrawl) {
      console.log('\n🕷️ Starting AMC website crawling...');
      console.log('⚠️  This may take several minutes...\n');
      
      // Crawl top 10 AMCs first
      const topAMCs = [
        'ICICI Prudential Asset Management Company Limited',
        'SBI Funds Management Ltd',
        'HDFC Asset Management Company Limited',
        'Nippon Life India Asset Management Ltd',
        'Kotak Mahindra Asset Management Co Ltd',
        'Aditya Birla Sun Life AMC Ltd',
        'UTI Asset Management Company Ltd',
        'Axis Asset Management Company Limited',
        'Mirae Asset Investment Managers (India) Private Limited',
        'Tata Asset Management Limited'
      ];
      
      await amcSearch.crawlAMCWebsites(topAMCs);
      console.log('✅ Top 10 AMC websites crawled successfully\n');
    }
    
    // Display setup statistics
    const stats = amcSearch.getSearchStats();
    console.log('📈 AMC Search Setup Complete!');
    console.log('================================');
    console.log(`Total AMCs indexed: ${stats.totalAMCs}`);
    console.log(`Total websites: ${stats.totalWebsites}`);
    console.log(`Total search keywords: ${stats.totalKeywords}`);
    console.log(`Elasticsearch index: ${stats.indexName}`);
    
    if (shouldCrawl) {
      console.log(`Crawled websites: ${stats.crawlResults}`);
    }
    
    console.log('\n🎯 AMC Search Integration is ready!');
    console.log('You can now search for AMC data using your financial search engine.\n');
    
    // Test search
    console.log('🔍 Testing AMC search...');
    const testResults = await amcSearch.searchAMCs('hdfc mutual fund', { limit: 3 });
    console.log(`Found ${testResults.total} results for "hdfc mutual fund"`);
    
    if (testResults.results.length > 0) {
      console.log('\nTop result:');
      const topResult = testResults.results[0];
      console.log(`- ${topResult.name}`);
      console.log(`- AUM: ₹${topResult.aum_june_2025} Crores`);
      console.log(`- Growth: ${topResult.change_percent}%`);
      console.log(`- Rank: ${topResult.rank}`);
    }
    
    console.log('\n✅ Setup completed successfully!');
    
  } catch (error) {
    console.error('❌ AMC Search setup failed:', error);
    process.exit(1);
  }
}

// Run setup
if (require.main === module) {
  setupAMCSearch().then(() => {
    process.exit(0);
  }).catch(error => {
    console.error('❌ Setup failed:', error);
    process.exit(1);
  });
}

module.exports = { setupAMCSearch };
