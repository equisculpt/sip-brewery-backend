const request = require('supertest');
const { UniverseClassMutualFundPlatform } = require('../src/app');
let app;
beforeAll(async () => {
  const platform = new UniverseClassMutualFundPlatform();
  await platform.initialize();
  app = platform.app;
});

// You may need to adjust these endpoints and payloads to match your actual API

describe('BrewBot End-to-End User Flow', () => {
  let authToken;
  let userId;
  let investmentId;

  const testUser = {
    name: 'Test User',
    email: `brewbot_${Date.now()}@test.com`,
    password: 'TestPass123!',
    mobile: '9999999999'
  };

  it('should sign up a new user', async () => {
    const res = await request(app)
      .post('/api/auth/register')
      .send(testUser)
      .expect(201);
    expect(res.body.success).toBe(true);
    expect(res.body.data.email).toBe(testUser.email);
    userId = res.body.data._id;
  });

  it('should login and get JWT token', async () => {
    const res = await request(app)
      .post('/api/auth/login')
      .send({ email: testUser.email, password: testUser.password })
      .expect(200);
    expect(res.body.success).toBe(true);
    expect(res.body.token).toBeDefined();
    authToken = res.body.token;
  });

  it('should update profile', async () => {
    const res = await request(app)
      .put('/api/auth/profile')
      .set('Authorization', `Bearer ${authToken}`)
      .send({ name: 'BrewBot User', mobile: '8888888888' })
      .expect(200);
    expect(res.body.success).toBe(true);
    expect(res.body.data.name).toBe('BrewBot User');
  });

  it('should make a sample investment', async () => {
    const payload = {
      userId,
      schemeCode: '120503',
      amount: 1000,
      pan: 'ABCDE1234F',
      paymentMode: 'ONLINE'
    };
    const res = await request(app)
      .post('/api/investment')
      .set('Authorization', `Bearer ${authToken}`)
      .send(payload)
      .expect(201);
    expect(res.body.success).toBe(true);
    investmentId = res.body.data._id;
  });

  it('should fetch portfolio', async () => {
    const res = await request(app)
      .get(`/api/portfolio/${userId}`)
      .set('Authorization', `Bearer ${authToken}`)
      .expect(200);
    expect(res.body.success).toBe(true);
    expect(Array.isArray(res.body.data.holdings)).toBe(true);
  });

  it('should get BrewBot insights', async () => {
    const res = await request(app)
      .get(`/api/agi/insights?userId=${userId}`)
      .set('Authorization', `Bearer ${authToken}`)
      .expect(200);
    expect(res.body.success).toBe(true);
    expect(res.body.message).toMatch(/BrewBot|brewbot/i);
  });
});
