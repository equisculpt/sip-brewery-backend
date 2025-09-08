const axios = require('axios');

const BASE_URL = 'http://localhost:3000';

async function testRealNifty50Data() {
  console.log('🚀 Testing Real NIFTY 50 Data for 1 Year\n');

  try {
    console.log('📊 Fetching real NIFTY 50 data for 1 year...');
    const response = await axios.get(`${BASE_URL}/api/benchmark/nifty50/real-data`);
    
    if (response.data.success) {
      const data = response.data.data;
      
      console.log('✅ Successfully fetched real NIFTY 50 data!');
      console.log('\n📈 Data Summary:');
      console.log(`   - Index: ${data.name}`);
      console.log(`   - Total Records: ${data.metadata.totalRecords}`);
      console.log(`   - Date Range: ${data.metadata.dateRange.start} to ${data.metadata.dateRange.end}`);
      console.log(`   - Current Price: ₹${data.metadata.currentPrice}`);
      console.log(`   - Data Source: ${data.metadata.dataSource}`);
      console.log(`   - Last Updated: ${data.metadata.lastUpdated}`);
      
      if (data.analytics) {
        console.log('\n📊 Analytics:');
        console.log(`   - Total Return: ${data.analytics.totalReturn}%`);
        console.log(`   - CAGR: ${data.analytics.cagr}%`);
        console.log(`   - Volatility: ${data.analytics.volatility}%`);
        console.log(`   - Max Price: ₹${data.analytics.maxPrice}`);
        console.log(`   - Min Price: ₹${data.analytics.minPrice}`);
        console.log(`   - Avg Volume: ${data.analytics.avgVolume.toLocaleString()}`);
      }
      
      console.log('\n📅 Sample Data Points (First 5):');
      data.data.slice(0, 5).forEach((point, index) => {
        console.log(`   ${index + 1}. ${point.date}: O:₹${point.open} H:₹${point.high} L:₹${point.low} C:₹${point.close} V:${point.volume.toLocaleString()}`);
      });
      
      console.log('\n📅 Sample Data Points (Last 5):');
      data.data.slice(-5).forEach((point, index) => {
        console.log(`   ${index + 1}. ${point.date}: O:₹${point.open} H:₹${point.high} L:₹${point.low} C:₹${point.close} V:${point.volume.toLocaleString()}`);
      });
      
      console.log('\n🎯 Charting Data Format:');
      console.log('   The data is formatted for easy integration with charting libraries:');
      console.log('   - Chart.js: Use data.data array directly');
      console.log('   - TradingView: Convert to OHLC format');
      console.log('   - D3.js: Use data.data for line/candlestick charts');
      console.log('   - Highcharts: Use data.data for stock charts');
      
      console.log('\n💡 Usage Examples:');
      console.log('   1. For Chart.js candlestick chart:');
      console.log('      const chartData = data.data.map(d => ({');
      console.log('        x: new Date(d.date),');
      console.log('        o: d.open,');
      console.log('        h: d.high,');
      console.log('        l: d.low,');
      console.log('        c: d.close');
      console.log('      }));');
      
      console.log('\n   2. For line chart:');
      console.log('      const lineData = data.data.map(d => ({');
      console.log('        x: d.date,');
      console.log('        y: d.close');
      console.log('      }));');
      
      console.log('\n   3. For volume chart:');
      console.log('      const volumeData = data.data.map(d => ({');
      console.log('        x: d.date,');
      console.log('        y: d.volume');
      console.log('      }));');
      
    } else {
      console.log('❌ Failed to fetch data:', response.data.message);
    }
    
  } catch (error) {
    console.error('❌ Error testing real NIFTY 50 data:', error.message);
    if (error.response) {
      console.error('   Response:', error.response.data);
    }
  }
}

async function testCurrentPrice() {
  console.log('\n🔍 Testing Current NIFTY 50 Price...');
  
  try {
    const realNiftyDataService = require('./src/services/realNiftyDataService');
    const currentPrice = await realNiftyDataService.getCurrentNifty50Price();
    
    console.log('✅ Current NIFTY 50 Price:');
    console.log(`   - Price: ₹${currentPrice.price}`);
    console.log(`   - Change: ₹${currentPrice.change}`);
    console.log(`   - Change %: ${currentPrice.changePercent}%`);
    console.log(`   - Timestamp: ${currentPrice.timestamp}`);
    
  } catch (error) {
    console.error('❌ Error fetching current price:', error.message);
  }
}

// Run tests
async function runTests() {
  await testRealNifty50Data();
  await testCurrentPrice();
  
  console.log('\n🎉 Test completed!');
  console.log('\n📝 Next Steps:');
  console.log('   1. Use the data for charting in your frontend');
  console.log('   2. Integrate with your mutual fund comparison tool');
  console.log('   3. Set up automated data updates');
  console.log('   4. Add more indices (NIFTY BANK, NIFTY IT, etc.)');
}

// Check if server is running
async function checkServer() {
  try {
    await axios.get(`${BASE_URL}/api/`);
    console.log('✅ Server is running at', BASE_URL);
    await runTests();
  } catch (error) {
    console.error('❌ Server is not running. Please start the server first:');
    console.error('   npm start');
    console.error('   or');
    console.error('   node index.js');
  }
}

checkServer(); 