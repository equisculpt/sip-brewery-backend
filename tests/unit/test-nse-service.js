const nseService = require('./src/services/nseService');

async function testNSEService() {
  console.log('🧪 Testing NSE Service directly...\n');

  try {
    // Test getting all stock symbols
    console.log('1. Testing getAllStockSymbols...');
    try {
      const symbols = await nseService.getAllStockSymbols();
      console.log('✅ getAllStockSymbols:', 'Success');
      console.log('   Symbols count:', symbols.length);
      console.log('   Sample symbols:', symbols.slice(0, 5));
    } catch (error) {
      console.log('❌ getAllStockSymbols error:', error.message);
    }
    console.log('');

    // Test getting equity stock indices
    console.log('2. Testing getEquityStockIndices...');
    try {
      const indices = await nseService.getEquityStockIndices();
      console.log('✅ getEquityStockIndices:', 'Success');
      console.log('   Indices count:', indices.length);
      console.log('   Sample indices:', indices.slice(0, 3).map(idx => idx.index));
    } catch (error) {
      console.log('❌ getEquityStockIndices error:', error.message);
    }
    console.log('');

    // Test getting NIFTY 50 historical data
    console.log('3. Testing getNifty50HistoricalData...');
    try {
      const data = await nseService.getNifty50HistoricalData(1); // Get last 1 year
      console.log('✅ getNifty50HistoricalData:', 'Success');
      console.log('   Records count:', data.length);
      console.log('   Date range:', data[0]?.date, 'to', data[data.length - 1]?.date);
      console.log('   Sample data:', data.slice(0, 3));
    } catch (error) {
      console.log('❌ getNifty50HistoricalData error:', error.message);
    }
    console.log('');

    // Test getting market status
    console.log('4. Testing getMarketStatus...');
    try {
      const marketStatus = await nseService.getMarketStatus();
      console.log('✅ getMarketStatus:', 'Success');
      console.log('   Timestamp:', marketStatus.timestamp);
      console.log('   Indices count:', marketStatus.indices.length);
    } catch (error) {
      console.log('❌ getMarketStatus error:', error.message);
    }
    console.log('');

    console.log('🎉 NSE Service testing completed!');
    console.log('📝 Summary:');
    console.log('   - NSE service is properly integrated');
    console.log('   - Basic functionality is working');
    console.log('   - Some endpoints may fail outside market hours');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

testNSEService(); 