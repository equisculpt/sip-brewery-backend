const axios = require('axios');
const baseURL = 'http://localhost:3000';

const endpoints = [
  { method: 'get', url: '/health', expect: 200 },
  { method: 'get', url: '/status', expect: 200 },
  // Modular endpoints (basic connectivity)
  { method: 'get', url: '/api/dashboard', expect: 200 },
  { method: 'get', url: '/api/leaderboard', expect: 200 },
  { method: 'get', url: '/api/rewards', expect: 200 },
  { method: 'get', url: '/api/smart-sip', expect: 200 },
  { method: 'get', url: '/api/whatsapp', expect: 200 },
  { method: 'get', url: '/api/ai', expect: 200 },
  { method: 'get', url: '/api/admin', expect: 200 },
  { method: 'get', url: '/api/benchmark', expect: 200 },
  { method: 'get', url: '/api/pdf', expect: 200 },
  { method: 'get', url: '/api/ollama', expect: 200 },
  // Universe-class endpoints
  { method: 'get', url: '/api/universe/realtime/status', expect: 200 },
  { method: 'post', url: '/api/universe/ai/analyze', expect: 200, data: { portfolioData: {}, marketData: {} } },
  { method: 'post', url: '/api/universe/tax/optimize', expect: 200, data: { portfolioData: {}, userProfile: {} } },
  { method: 'get', url: '/api/universe/social/feed', expect: 200 },
  { method: 'get', url: '/api/universe/gamification/profile/testuser', expect: 200 },
  { method: 'post', url: '/api/universe/quantum/optimize', expect: 200, data: { portfolioData: {} } },
  { method: 'post', url: '/api/universe/esg/analyze', expect: 200, data: { portfolioData: {}, userProfile: {} } },
  { method: 'get', url: '/api/universe/architecture/status', expect: 200 },
  { method: 'post', url: '/api/universe/security/mfa/setup', expect: 200, data: { userId: 'testuser', methods: ['sms'] } },
  { method: 'get', url: '/api/universe/scalability/status', expect: 200 },
];

const results = [];

async function runTest(endpoint) {
  try {
    let res;
    if (endpoint.method === 'get') {
      res = await axios.get(baseURL + endpoint.url);
    } else if (endpoint.method === 'post') {
      res = await axios.post(baseURL + endpoint.url, endpoint.data || {});
    }
    if (res.status === endpoint.expect) {
      results.push({ url: endpoint.url, method: endpoint.method, status: 'PASS', code: res.status });
      console.log(`✅ [${endpoint.method.toUpperCase()}] ${endpoint.url} - ${res.status}`);
    } else {
      results.push({ url: endpoint.url, method: endpoint.method, status: 'FAIL', code: res.status });
      console.log(`❌ [${endpoint.method.toUpperCase()}] ${endpoint.url} - ${res.status}`);
    }
  } catch (e) {
    if (e.response) {
      results.push({ url: endpoint.url, method: endpoint.method, status: 'FAIL', code: e.response.status });
      console.log(`❌ [${endpoint.method.toUpperCase()}] ${endpoint.url} - ${e.response.status}`);
    } else {
      results.push({ url: endpoint.url, method: endpoint.method, status: 'FAIL', code: 'NO_RESPONSE' });
      console.log(`❌ [${endpoint.method.toUpperCase()}] ${endpoint.url} - NO RESPONSE`);
    }
  }
}

(async () => {
  console.log('🔍 Deep Backend Test Starting...\n');
  for (const endpoint of endpoints) {
    await runTest(endpoint);
  }
  // Summary
  console.log('\n' + '='.repeat(50));
  console.log('📊 DEEP TEST SUMMARY');
  console.log('='.repeat(50));
  let pass = 0, fail = 0;
  results.forEach(r => r.status === 'PASS' ? pass++ : fail++);
  results.forEach(r => {
    console.log(`${r.status === 'PASS' ? '✅' : '❌'} [${r.method.toUpperCase()}] ${r.url} - ${r.code}`);
  });
  console.log(`\n✅ PASSED: ${pass}`);
  console.log(`❌ FAILED: ${fail}`);
  console.log('='.repeat(50));
})(); 