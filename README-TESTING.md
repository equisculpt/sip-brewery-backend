# 🧪 SipBrewery Backend Testing Framework

A comprehensive testing suite for the SipBrewery mutual fund investment platform backend, covering unit tests, integration tests, API tests, and performance testing.

## 📋 Table of Contents

- [Overview](#overview)
- [Test Structure](#test-structure)
- [Setup](#setup)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Coverage](#coverage)
- [CI/CD Integration](#cicd-integration)
- [Performance Testing](#performance-testing)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This testing framework provides:

- **Unit Tests**: Individual service and function testing
- **Integration Tests**: API endpoint testing with authentication
- **WhatsApp Bot Tests**: Message processing and AI integration
- **Performance Tests**: Load testing with Artillery
- **Security Tests**: Vulnerability scanning and validation
- **90%+ Code Coverage**: Comprehensive test coverage
- **CI/CD Integration**: Automated testing on every commit

## 📁 Test Structure

```
__tests__/
├── setup.js                 # Jest global setup
├── env-setup.js            # Test environment configuration
├── __mocks__/
│   └── testData.js         # Mock data for all test scenarios
├── services/
│   ├── portfolioAnalyticsService.test.js
│   ├── smartSipService.test.js
│   ├── rewardsService.test.js
│   ├── leaderboardService.test.js
│   └── aiService.test.js
├── api/
│   ├── dashboard.test.js
│   ├── auth.test.js
│   ├── smartSip.test.js
│   ├── leaderboard.test.js
│   └── rewards.test.js
├── whatsapp/
│   └── whatsappBot.test.js
├── integration/
│   ├── userWorkflow.test.js
│   ├── portfolioWorkflow.test.js
│   └── sipWorkflow.test.js
└── performance/
    ├── load-test.yml
    └── processors.js
```

## 🚀 Setup

### 1. Install Dependencies

```bash
npm install
```

### 2. Create Test Environment File

```bash
cp .env.example .env.test
```

Add the following to `.env.test`:

```env
NODE_ENV=test
MONGODB_URI=mongodb://localhost:27017/sipbrewery-test
JWT_SECRET=test-jwt-secret-key
DISABLE_EXTERNAL_APIS=true
DISABLE_WHATSAPP=true
DISABLE_AI_SERVICES=true
LOG_LEVEL=error
```

### 3. Verify Jest Configuration

The `jest.config.js` file is already configured with:
- Test environment setup
- Coverage thresholds (90%)
- Mock data integration
- Performance monitoring

## 🏃‍♂️ Running Tests

### Quick Start

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run with coverage report
npm run test:coverage
```

### Specific Test Categories

```bash
# Unit tests only
npm run test:unit

# API integration tests
npm run test:api

# WhatsApp bot tests
npm run test:whatsapp

# Integration tests
npm run test:integration

# Performance tests
npm run test:performance
```

### CI/CD Mode

```bash
# Run tests for continuous integration
npm run test:ci
```

## 🧪 Test Categories

### 1. Unit Tests (`__tests__/services/`)

Test individual service functions in isolation:

```javascript
describe('Portfolio Analytics Service', () => {
  it('should calculate XIRR correctly', async () => {
    const xirr = await portfolioAnalyticsService.calculateXIRR(userId, '1Y');
    expect(xirr).toBeGreaterThan(-100);
    expect(xirr).toBeLessThan(1000);
  });
});
```

**Coverage**: Business logic, calculations, data transformations

### 2. API Integration Tests (`__tests__/api/`)

Test complete API endpoints with authentication:

```javascript
describe('Dashboard API Endpoints', () => {
  it('should return portfolio summary', async () => {
    const response = await request(app)
      .get('/api/dashboard/summary')
      .set('Authorization', `Bearer ${authToken}`)
      .expect(200);
    
    expect(response.body.data).toHaveProperty('portfolio');
  });
});
```

**Coverage**: HTTP endpoints, authentication, request/response validation

### 3. WhatsApp Bot Tests (`__tests__/whatsapp/`)

Test WhatsApp message processing and AI integration:

```javascript
describe('WhatsApp Bot Integration', () => {
  it('should process SIP order message', async () => {
    const response = await request(app)
      .post('/api/whatsapp/webhook')
      .send(sipOrderPayload)
      .expect(200);
    
    expect(response.body.success).toBe(true);
  });
});
```

**Coverage**: Message processing, intent detection, AI responses

### 4. Integration Tests (`__tests__/integration/`)

Test complete user workflows:

```javascript
describe('User Investment Workflow', () => {
  it('should complete full investment cycle', async () => {
    // 1. User registration
    // 2. Portfolio creation
    // 3. SIP setup
    // 4. Transaction processing
    // 5. Performance tracking
  });
});
```

**Coverage**: End-to-end user journeys, data consistency

## 📊 Coverage

### Coverage Requirements

- **Global Coverage**: 90% minimum
- **Branches**: 90% minimum
- **Functions**: 90% minimum
- **Lines**: 90% minimum

### Coverage Report

```bash
npm run test:coverage
```

Generates coverage report in:
- `coverage/lcov-report/index.html` (HTML report)
- `coverage/lcov.info` (LCOV format for CI)

### Coverage Exclusions

The following are excluded from coverage:
- Database configuration files
- Environment setup files
- Test files themselves
- External service integrations (mocked)

## 🔄 CI/CD Integration

### GitHub Actions Workflow

Located at `.github/workflows/test.yml`:

- **Triggers**: Push to main/dev, Pull requests
- **Node.js Versions**: 18.x, 20.x
- **Test Categories**: Unit, API, WhatsApp, Integration
- **Coverage**: Uploaded to Codecov
- **Security**: npm audit
- **Performance**: Artillery load tests

### Pre-commit Hooks

```bash
# Install pre-commit hooks
npm install husky --save-dev
npx husky install
npx husky add .husky/pre-commit "npm run test:ci"
```

### Branch Protection Rules

Configure in GitHub repository settings:
- Require status checks to pass
- Require coverage threshold (90%)
- Require security audit to pass

## ⚡ Performance Testing

### Artillery Load Tests

```bash
# Run performance tests
npm run test:performance
```

### Test Scenarios

1. **Dashboard API Load Test** (40% weight)
   - Portfolio summary
   - Performance metrics
   - Recent transactions

2. **WhatsApp Bot Load Test** (30% weight)
   - Message processing
   - AI response generation

3. **Smart SIP API Load Test** (20% weight)
   - SIP analysis
   - SIP creation

4. **Other APIs** (10% weight)
   - Leaderboard
   - AI services
   - Authentication

### Performance Metrics

- **Response Time**: < 500ms (95th percentile)
- **Error Rate**: < 1%
- **Throughput**: 1000+ requests/second
- **Concurrent Users**: 1000+

### Load Test Phases

1. **Warm-up** (60s): 10 req/s
2. **Sustained Load** (120s): 50 req/s
3. **Peak Load** (60s): 100 req/s
4. **Cool-down** (30s): 10 req/s

## 🛡️ Security Testing

### Automated Security Checks

```bash
# Run security audit
npm audit

# Run with specific level
npm audit --audit-level=moderate
```

### Security Test Categories

1. **Input Validation**
   - SQL injection prevention
   - XSS protection
   - Input sanitization

2. **Authentication**
   - JWT token validation
   - Role-based access control
   - Session management

3. **API Security**
   - Rate limiting
   - CORS configuration
   - Request validation

4. **Data Protection**
   - Sensitive data encryption
   - Audit logging
   - Privacy compliance

## 📝 Best Practices

### Writing Tests

1. **Test Structure**
   ```javascript
   describe('Service Name', () => {
     beforeEach(async () => {
       // Setup test data
     });
     
     describe('Method Name', () => {
       it('should do something specific', async () => {
         // Test implementation
       });
     });
   });
   ```

2. **Naming Conventions**
   - Test files: `*.test.js`
   - Describe blocks: Service/Component name
   - Test cases: "should [expected behavior]"

3. **Mocking Strategy**
   - Mock external APIs
   - Mock database connections
   - Use realistic test data

4. **Assertions**
   - Test one thing per test case
   - Use descriptive assertion messages
   - Test both success and failure scenarios

### Test Data Management

1. **Mock Data**
   - Use `__tests__/__mocks__/testData.js`
   - Keep data realistic and diverse
   - Update data when models change

2. **Database Isolation**
   - Use in-memory MongoDB for tests
   - Clean database between tests
   - Don't rely on external data

3. **Environment Isolation**
   - Use separate test environment
   - Disable external services
   - Use test-specific configuration

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Ensure MongoDB is running
   mongod --dbpath ./data/db
   
   # Or use Docker
   docker run -d -p 27017:27017 mongo:latest
   ```

2. **Test Timeout Errors**
   ```javascript
   // Increase timeout for slow tests
   jest.setTimeout(30000);
   ```

3. **Coverage Below Threshold**
   ```bash
   # Check which files are missing coverage
   npm run test:coverage
   
   # Add tests for uncovered code
   ```

4. **Performance Test Failures**
   ```bash
   # Check if server is running
   npm start
   
   # Run with lower load first
   artillery run --config.overrides.phases[0].arrivalRate=1 load-test.yml
   ```

### Debug Mode

```bash
# Run tests with debug output
DEBUG=* npm test

# Run specific test with debug
DEBUG=* npm test -- --testNamePattern="should calculate XIRR"
```

### Test Logs

```bash
# View test logs
npm test > test-output.log 2>&1

# View performance test results
artillery run load-test.yml --output artillery-report.json
```

## 📚 Additional Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [Supertest Documentation](https://github.com/visionmedia/supertest)
- [Artillery Documentation](https://www.artillery.io/docs)
- [MongoDB Memory Server](https://github.com/nodkz/mongodb-memory-server)

## 🤝 Contributing

When adding new features:

1. Write tests first (TDD approach)
2. Ensure 90%+ coverage
3. Update mock data if needed
4. Add performance tests for new APIs
5. Update this documentation

## 📞 Support

For testing framework issues:
1. Check troubleshooting section
2. Review test logs
3. Verify environment setup
4. Create issue with detailed error information 