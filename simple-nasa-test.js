/**
 * Simple NASA API test
 */

require('dotenv').config();

async function simpleNASATest() {
  console.log('🛰️ Simple NASA API Test\n');
  
  // Check environment variables
  console.log('1. Checking environment variables...');
  console.log(`   EARTHDATA_USERNAME: ${process.env.EARTHDATA_USERNAME ? '✅ Set' : '❌ Missing'}`);
  console.log(`   EARTHDATA_TOKEN: ${process.env.EARTHDATA_TOKEN ? '✅ Set (' + process.env.EARTHDATA_TOKEN.substring(0, 20) + '...)' : '❌ Missing'}`);
  
  if (!process.env.EARTHDATA_USERNAME || !process.env.EARTHDATA_TOKEN) {
    console.log('\n❌ Missing credentials in .env file');
    return false;
  }
  
  // Test direct API call
  console.log('\n2. Testing direct NASA CMR API call...');
  
  try {
    const searchUrl = 'https://cmr.earthdata.nasa.gov/search/granules.json?collection_concept_id=C1000000240-LPDAAC_ECS&bounding_box=72.8,18.9,72.9,19.0&page_size=5';
    
    const response = await fetch(searchUrl, {
      headers: {
        'Authorization': `Bearer ${process.env.EARTHDATA_TOKEN}`,
        'User-Agent': 'ASI-Supply-Chain/1.0'
      }
    });
    
    console.log(`   Response status: ${response.status} ${response.statusText}`);
    
    if (response.ok) {
      const data = await response.json();
      const granuleCount = data.feed?.entry?.length || 0;
      console.log(`   ✅ Success! Found ${granuleCount} granules`);
      console.log(`   Total results: ${data.feed?.opensearch$totalResults?.value || 0}`);
      
      if (granuleCount > 0) {
        const firstGranule = data.feed.entry[0];
        console.log(`   First granule: ${firstGranule.title}`);
        console.log(`   Time: ${firstGranule.time_start}`);
      }
      
      return true;
    } else {
      const errorText = await response.text();
      console.log(`   ❌ API Error: ${errorText}`);
      return false;
    }
    
  } catch (error) {
    console.log(`   ❌ Network Error: ${error.message}`);
    return false;
  }
}

// Test without authentication (public data)
async function testPublicAPI() {
  console.log('\n3. Testing public NASA API (no auth)...');
  
  try {
    const publicUrl = 'https://firms.modaps.eosdis.nasa.gov/api/country/csv/VIIRS_SNPP_NRT/IND/1';
    
    const response = await fetch(publicUrl);
    console.log(`   Response status: ${response.status} ${response.statusText}`);
    
    if (response.ok) {
      const csvData = await response.text();
      const lines = csvData.split('\n').filter(line => line.trim());
      console.log(`   ✅ Success! Got ${lines.length - 1} fire detection records`);
      
      if (lines.length > 1) {
        console.log(`   Sample data: ${lines[1].split(',').slice(0, 3).join(', ')}`);
      }
      
      return true;
    } else {
      console.log(`   ❌ Public API failed`);
      return false;
    }
    
  } catch (error) {
    console.log(`   ❌ Public API Error: ${error.message}`);
    return false;
  }
}

// Run tests
async function runAllTests() {
  console.log('🧪 NASA Earthdata Connection Tests\n');
  
  const results = {
    credentials: await simpleNASATest(),
    publicAPI: await testPublicAPI()
  };
  
  console.log('\n📊 Test Results:');
  console.log(`   Credentials & Auth: ${results.credentials ? '✅ Pass' : '❌ Fail'}`);
  console.log(`   Public API Access: ${results.publicAPI ? '✅ Pass' : '❌ Fail'}`);
  
  if (results.publicAPI && !results.credentials) {
    console.log('\n💡 Diagnosis: Public API works, but authentication failed.');
    console.log('   This suggests a token format or permission issue.');
    console.log('   Try regenerating your NASA Earthdata token.');
  } else if (results.credentials && results.publicAPI) {
    console.log('\n🎉 All tests passed! NASA satellite integration is ready.');
  } else if (!results.publicAPI) {
    console.log('\n🌐 Network connectivity issue. Check internet connection.');
  }
  
  return results.credentials;
}

if (require.main === module) {
  runAllTests()
    .then(success => process.exit(success ? 0 : 1))
    .catch(error => {
      console.error('❌ Test error:', error);
      process.exit(1);
    });
}

module.exports = runAllTests;
