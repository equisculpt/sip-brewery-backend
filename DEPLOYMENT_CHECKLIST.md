# SIP Brewery ASI WhatsApp Deployment Checklist

## Pre-Deployment Setup

### 1. Environment Configuration
- [ ] Copy .env.template to .env
- [ ] Update WHATSAPP_VERIFY_TOKEN with your webhook verification token
- [ ] Update WHATSAPP_ACCESS_TOKEN with your WhatsApp Business API token
- [ ] Update WHATSAPP_PHONE_NUMBER_ID with your phone number ID
- [ ] Update GEMINI_API_KEY with your Google Gemini AI API key
- [ ] Update MONGODB_URI with your MongoDB connection string
- [ ] Update JWT_SECRET with a secure random string
- [ ] Configure ALLOWED_ORIGINS for CORS

### 2. WhatsApp Business API Setup
- [ ] Create WhatsApp Business Account
- [ ] Set up Meta Developer Account
- [ ] Create WhatsApp Business App
- [ ] Configure webhook URL: https://your-domain.com/api/asi-whatsapp/webhook
- [ ] Set webhook verification token
- [ ] Subscribe to webhook events: messages
- [ ] Test webhook verification

### 3. Database Setup
- [ ] Install and configure MongoDB
- [ ] Create database: sip-brewery
- [ ] Set up database indexes for performance
- [ ] Configure database backup strategy

### 4. SSL Certificate (Production)
- [ ] Obtain SSL certificate for your domain
- [ ] Configure HTTPS (WhatsApp requires HTTPS for webhooks)
- [ ] Update webhook URL to use HTTPS

### 5. Dependencies Installation
- [ ] Run: npm install
- [ ] Verify all dependencies are installed correctly
- [ ] Check for any security vulnerabilities: npm audit

## Deployment Steps

### 1. Code Deployment
- [ ] Clone/update repository on server
- [ ] Copy .env file with production values
- [ ] Install dependencies: npm install --production

### 2. Service Configuration
- [ ] Configure process manager (PM2 recommended)
- [ ] Set up log rotation
- [ ] Configure monitoring and alerting

### 3. Testing
- [ ] Run integration tests: npm run test:integration
- [ ] Test webhook endpoint manually
- [ ] Send test WhatsApp message
- [ ] Verify ASI integration
- [ ] Test report generation

### 4. Go Live
- [ ] Start the application: npm start
- [ ] Verify health check: /health endpoint
- [ ] Monitor logs for errors
- [ ] Test with real WhatsApp messages

## Post-Deployment Verification

### 1. Functionality Tests
- [ ] WhatsApp message processing
- [ ] Intent detection working
- [ ] User onboarding flow
- [ ] Investment operations
- [ ] ASI analysis
- [ ] Report generation
- [ ] Error handling

### 2. Performance Tests
- [ ] Response time < 2 seconds
- [ ] Memory usage within limits
- [ ] Database query performance
- [ ] Rate limiting working

### 3. Security Tests
- [ ] Webhook verification working
- [ ] HTTPS enforced
- [ ] Rate limiting active
- [ ] Input validation working
- [ ] Error messages don't expose sensitive data

## Monitoring Setup

### 1. Application Monitoring
- [ ] Set up application performance monitoring
- [ ] Configure error tracking
- [ ] Set up uptime monitoring
- [ ] Configure log aggregation

### 2. Business Metrics
- [ ] Track message volume
- [ ] Monitor response times
- [ ] Track user engagement
- [ ] Monitor report generation

### 3. Alerts
- [ ] High error rate alerts
- [ ] Performance degradation alerts
- [ ] Webhook failure alerts
- [ ] Database connection alerts

## Maintenance

### 1. Regular Tasks
- [ ] Monitor logs daily
- [ ] Review performance metrics
- [ ] Update dependencies monthly
- [ ] Backup database regularly

### 2. Security Updates
- [ ] Apply security patches promptly
- [ ] Review access logs
- [ ] Rotate API keys quarterly
- [ ] Update SSL certificates before expiry

## Rollback Plan

### 1. Preparation
- [ ] Document current version
- [ ] Create database backup
- [ ] Prepare rollback script

### 2. Rollback Steps
- [ ] Stop current application
- [ ] Restore previous version
- [ ] Restore database if needed
- [ ] Restart application
- [ ] Verify functionality

## Support Information

- **Documentation**: ASI_WHATSAPP_INTEGRATION.md
- **Health Check**: /health
- **Service Status**: /api/asi-whatsapp/status
- **Test Suite**: npm run test:integration
- **Logs Location**: ./logs/app.log

## Emergency Contacts

- **Development Team**: [Your contact info]
- **WhatsApp Business Support**: [Support contact]
- **Infrastructure Team**: [Infrastructure contact]

---

**Deployment Date**: _______________
**Deployed By**: _______________
**Version**: 2.0.0
**Environment**: _______________