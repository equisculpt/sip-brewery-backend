# 🚀 PRODUCTION DEPLOYMENT CHECKLIST

**Platform:** SIP Brewery - ML/AI Infrastructure  
**Version:** Phase 1 Complete  
**Date:** February 9, 2026

---

## ✅ PRE-DEPLOYMENT CHECKLIST

### 1. Environment Setup

#### Python Environment
```bash
# Create virtual environment
python -m venv ml_env
source ml_env/bin/activate  # Linux/Mac
# or
ml_env\Scripts\activate  # Windows

# Install dependencies
cd ml
pip install -r requirements.txt

# Verify installations
python -c "import torch; print(torch.__version__)"
python -c "import torch_geometric; print('PyG OK')"
python -c "import transformers; print('Transformers OK')"
```

- [ ] Python 3.9+ installed
- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] GPU support verified (optional but recommended)

#### Node.js Environment
```bash
# Install Node.js dependencies
npm install

# Verify ML service dependencies
npm list ioredis
npm list express-validator
```

- [ ] Node.js 16+ installed
- [ ] All npm packages installed
- [ ] Redis client configured

#### Database Setup
```bash
# MongoDB
mongod --dbpath /data/db

# Redis
redis-server

# PostgreSQL (for MLflow)
createdb mlflow
```

- [ ] MongoDB running
- [ ] Redis running
- [ ] PostgreSQL running (for MLflow)

---

### 2. MLflow Setup

```bash
# Start MLflow tracking server
mlflow server \
  --backend-store-uri postgresql://mlflow:password@localhost/mlflow \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

# Or for development (SQLite)
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000
```

- [ ] MLflow server running on port 5000
- [ ] UI accessible at http://localhost:5000
- [ ] Backend store configured
- [ ] Artifact store accessible

**Verify:**
```bash
curl http://localhost:5000/health
```

---

### 3. Feast Feature Store Setup

```bash
cd ml/feature_store/feature_repo

# Initialize Feast
feast apply

# Verify feature views
feast feature-views list

# Test feature retrieval
feast materialize-incremental $(date +%Y-%m-%d)
```

- [ ] Feast initialized
- [ ] Feature views registered
- [ ] Online store configured (SQLite/Redis)
- [ ] Offline store configured (Parquet files)

**Verify:**
```python
from feast import FeatureStore
store = FeatureStore(repo_path="ml/feature_store/feature_repo")
print(store.list_feature_views())
```

---

### 4. Model Deployment

#### Create Inference Scripts

**File:** `ml/models/inference/portfolio_optimizer_inference.py`
```python
import sys
import json
import torch
from rl_portfolio_optimizer import RLPortfolioOptimizer, PortfolioEnvironment

def main():
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())
    
    # Load model
    model = RLPortfolioOptimizer(state_dim=30, action_dim=10)
    model.load_model('models/rl_portfolio_model.pth')
    
    # Prepare state
    state = prepare_state(input_data)
    
    # Predict
    result = model.optimize_portfolio(state)
    
    # Output JSON
    print(json.dumps(result))

if __name__ == '__main__':
    main()
```

- [ ] Created inference script for RL Portfolio Optimizer
- [ ] Created inference script for GNN Fund Predictor
- [ ] Created inference script for LSTM Risk Predictor
- [ ] Created inference script for Behavioral Predictor
- [ ] Created inference script for Market Regime Detector

#### Download Pre-trained Models

```bash
# Create models directory
mkdir -p ml/models/trained

# Download or train models
cd ml/models
python rl_portfolio_optimizer.py  # Train and save
python gnn_fund_predictor.py      # Train and save
python lstm_risk_predictor.py     # Train and save
python behavioral_predictor.py    # Train and save
python market_regime_detector.py  # Train and save
```

- [ ] All 5 models trained
- [ ] Model weights saved to `ml/models/trained/`
- [ ] Model versions registered in MLflow

---

### 5. API Integration

#### Update app.js

**File:** `src/app.js`
```javascript
// Add ML routes
const mlRoutes = require('./routes/ml');
app.use('/api/ml', mlRoutes);
```

- [ ] ML routes added to app.js
- [ ] Routes tested with Postman/curl
- [ ] Authentication working
- [ ] Validation working

#### Test Endpoints

```bash
# Health check
curl http://localhost:3000/api/ml/health

# Portfolio optimization (requires auth token)
curl -X POST http://localhost:3000/api/ml/portfolio/optimize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolioData": {
      "holdings": [...]
    }
  }'
```

- [ ] Health endpoint responding
- [ ] Portfolio optimization working
- [ ] Fund prediction working
- [ ] Risk prediction working
- [ ] Behavioral prediction working
- [ ] Market regime detection working

---

### 6. Caching Configuration

```javascript
// Environment variables
REDIS_HOST=localhost
REDIS_PORT=6379
ML_CACHE_ENABLED=true
```

- [ ] Redis connection configured
- [ ] Cache TTL set appropriately
- [ ] Cache keys namespaced (`ml:*`)
- [ ] Cache invalidation working

**Test caching:**
```bash
# First request (cache miss)
time curl -X GET http://localhost:3000/api/ml/market/regime

# Second request (cache hit - should be faster)
time curl -X GET http://localhost:3000/api/ml/market/regime
```

---

### 7. Monitoring Setup

#### Prometheus Metrics

**File:** `src/middleware/mlMetrics.js`
```javascript
const prometheus = require('prom-client');

const mlInferenceCounter = new prometheus.Counter({
  name: 'ml_inference_total',
  help: 'Total ML inference requests',
  labelNames: ['model', 'status']
});

const mlInferenceLatency = new prometheus.Histogram({
  name: 'ml_inference_duration_seconds',
  help: 'ML inference latency',
  labelNames: ['model']
});
```

- [ ] Prometheus client installed
- [ ] Metrics endpoint created (`/metrics`)
- [ ] Custom ML metrics defined
- [ ] Grafana dashboard created

#### Logging

```javascript
// Configure Winston logger for ML
logger.info('ML inference request', {
  model: 'portfolio_optimizer',
  userId: userId,
  latency: latencyMs,
  cached: isCached
});
```

- [ ] ML-specific log levels configured
- [ ] Structured logging implemented
- [ ] Log aggregation configured (ELK/Loki)

---

### 8. Security Hardening

#### API Security
- [ ] Rate limiting configured (100 req/min per user)
- [ ] Input validation on all endpoints
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] CORS configured properly

#### Model Security
- [ ] Model files encrypted at rest
- [ ] Secure model loading (verify checksums)
- [ ] No sensitive data in model artifacts
- [ ] Access control on model endpoints

#### Data Privacy
- [ ] No PII in training data
- [ ] User data anonymized
- [ ] GDPR compliance verified
- [ ] Data retention policies implemented

---

### 9. Performance Testing

#### Load Testing

```bash
# Install Artillery
npm install -g artillery

# Run load test
artillery quick --count 100 --num 10 \
  http://localhost:3000/api/ml/market/regime
```

**Performance Targets:**
- [ ] API latency < 500ms (p95)
- [ ] Throughput > 100 req/s
- [ ] Cache hit rate > 70%
- [ ] Error rate < 1%

#### Stress Testing
```bash
# Gradual ramp-up
artillery run load-test-config.yml
```

- [ ] System stable under 2x normal load
- [ ] Graceful degradation under 5x load
- [ ] Auto-scaling triggers working

---

### 10. Documentation

- [ ] API documentation complete (Swagger/OpenAPI)
- [ ] Model documentation complete
- [ ] Deployment guide written
- [ ] Troubleshooting guide created
- [ ] Runbook for on-call engineers

---

## 🚀 DEPLOYMENT STEPS

### Step 1: Pre-deployment Verification

```bash
# Run all tests
npm test

# Check code quality
npm run lint

# Verify environment
node scripts/verify-environment.js
```

- [ ] All tests passing
- [ ] No linting errors
- [ ] Environment verified

### Step 2: Database Migrations

```bash
# Run migrations
npm run migrate

# Verify schema
npm run migrate:verify
```

- [ ] Migrations completed
- [ ] Schema verified
- [ ] Rollback tested

### Step 3: Deploy ML Services

```bash
# Start MLflow
pm2 start mlflow-server.js --name mlflow

# Start ML inference workers
pm2 start ml-worker.js --name ml-worker -i 4

# Verify
pm2 status
```

- [ ] MLflow server running
- [ ] ML workers running
- [ ] Health checks passing

### Step 4: Deploy API Server

```bash
# Build application
npm run build

# Start with PM2
pm2 start ecosystem.config.js --env production

# Verify
pm2 logs
curl http://localhost:3000/health
```

- [ ] Application built
- [ ] Server running
- [ ] Health check passing

### Step 5: Smoke Tests

```bash
# Run smoke tests
npm run test:smoke

# Test critical paths
./scripts/smoke-test.sh
```

- [ ] All smoke tests passing
- [ ] Critical paths working
- [ ] No errors in logs

### Step 6: Enable Monitoring

```bash
# Start Prometheus
docker run -d -p 9090:9090 \
  -v ./prometheus.yml:/etc/prometheus/prometheus.yml \
  prom/prometheus

# Start Grafana
docker run -d -p 3001:3000 grafana/grafana
```

- [ ] Prometheus scraping metrics
- [ ] Grafana dashboards loaded
- [ ] Alerts configured

### Step 7: Gradual Rollout

```bash
# Route 10% traffic to new version
# (using load balancer or feature flags)
```

- [ ] 10% traffic routed
- [ ] Metrics looking good
- [ ] No errors detected
- [ ] Increase to 50%
- [ ] Increase to 100%

---

## 🔍 POST-DEPLOYMENT VERIFICATION

### Immediate Checks (0-1 hour)

- [ ] All services running
- [ ] No error spikes in logs
- [ ] API latency within SLA
- [ ] Cache hit rate normal
- [ ] Database connections stable

### Short-term Monitoring (1-24 hours)

- [ ] Model predictions accurate
- [ ] No memory leaks
- [ ] CPU usage normal
- [ ] Disk usage stable
- [ ] No security alerts

### Long-term Monitoring (1-7 days)

- [ ] Model performance stable
- [ ] User feedback positive
- [ ] Business metrics improving
- [ ] No degradation over time

---

## 🚨 ROLLBACK PLAN

### Trigger Conditions

Rollback if:
- Error rate > 5%
- API latency > 2x baseline
- Model accuracy drops > 10%
- Critical security issue
- Data corruption detected

### Rollback Steps

```bash
# Stop new version
pm2 stop all

# Restore previous version
git checkout v1.0.0
npm install
npm run build

# Restart
pm2 start ecosystem.config.js

# Verify
curl http://localhost:3000/health
```

- [ ] Rollback procedure tested
- [ ] Rollback time < 5 minutes
- [ ] Data integrity maintained

---

## 📊 SUCCESS METRICS

### Technical Metrics
- API Uptime: > 99.9%
- API Latency (p95): < 500ms
- Error Rate: < 1%
- Cache Hit Rate: > 70%
- Model Accuracy: Within 5% of baseline

### Business Metrics
- User Engagement: +20%
- Portfolio Returns: +15%
- Churn Rate: -30%
- Support Tickets: -20%

---

## 📞 SUPPORT CONTACTS

### On-Call Engineers
- Primary: [Name] - [Phone]
- Secondary: [Name] - [Phone]

### Escalation
- ML Team Lead: [Name]
- CTO: [Name]
- DevOps: [Name]

### External Support
- MLflow: [Support Link]
- PyTorch: [Community Forum]
- AWS/Cloud Provider: [Support Ticket]

---

## 📝 DEPLOYMENT SIGN-OFF

- [ ] Development Team Lead: _________________ Date: _______
- [ ] QA Lead: _________________ Date: _______
- [ ] DevOps Lead: _________________ Date: _______
- [ ] Security Lead: _________________ Date: _______
- [ ] Product Manager: _________________ Date: _______
- [ ] CTO: _________________ Date: _______

---

**Deployment Status:** ⚪ Not Started | 🟡 In Progress | ✅ Complete

**Last Updated:** February 9, 2026  
**Next Review:** After Phase 2 Implementation
