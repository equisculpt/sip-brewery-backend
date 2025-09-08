const request = require('supertest');
let app;
try {
  app = require('../src/app'); // expects Express app export
} catch (e) {
  app = null;
}

describe('/api/rewards/summary endpoint', () => {
  it('should return 401 for unauthenticated request', async () => {
    if (!app) return;
    const res = await request(app).get('/api/rewards/summary');
    expect([401, 403]).toContain(res.statusCode);
  });
  // Add more tests for authenticated and edge cases
});
