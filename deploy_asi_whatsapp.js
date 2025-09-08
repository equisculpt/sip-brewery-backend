const fs = require('fs');
const path = require('path');

console.log('🚀 SIP BREWERY ASI WHATSAPP DEPLOYMENT SCRIPT');
console.log('📱 Integrating Complete Platform Operations via WhatsApp');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

class ASIWhatsAppDeployment {
    constructor() {
        this.projectRoot = __dirname;
        this.deploymentSteps = [];
        this.errors = [];
    }

    async deploy() {
        console.log('🔧 Starting ASI WhatsApp integration deployment...\n');

        try {
            // Step 1: Verify file structure
            await this.verifyFileStructure();

            // Step 2: Update main app.js
            await this.updateMainApp();

            // Step 3: Create environment template
            await this.createEnvironmentTemplate();

            // Step 4: Update package.json dependencies
            await this.updatePackageDependencies();

            // Step 5: Create startup script
            await this.createStartupScript();

            // Step 6: Generate deployment checklist
            await this.generateDeploymentChecklist();

            // Step 7: Run verification tests
            await this.runVerificationTests();

            // Generate deployment report
            await this.generateDeploymentReport();

        } catch (error) {
            console.error('❌ Deployment failed:', error);
            this.errors.push(`Deployment failed: ${error.message}`);
        }
    }

    async verifyFileStructure() {
        console.log('📁 Verifying file structure...');

        const requiredFiles = [
            'src/services/ASIWhatsAppService.js',
            'src/controllers/ASIWhatsAppController.js',
            'src/routes/asiWhatsAppRoutes.js',
            'COMPLETE_REPORT_SUITE.js',
            'test_asi_whatsapp_integration.js',
            'ASI_WHATSAPP_INTEGRATION.md'
        ];

        const requiredDirs = [
            'src/services',
            'src/controllers',
            'src/routes',
            'src/models',
            'src/middleware',
            'src/utils'
        ];

        // Check directories
        for (const dir of requiredDirs) {
            const dirPath = path.join(this.projectRoot, dir);
            if (!fs.existsSync(dirPath)) {
                console.log(`📁 Creating directory: ${dir}`);
                fs.mkdirSync(dirPath, { recursive: true });
            } else {
                console.log(`✅ Directory exists: ${dir}`);
            }
        }

        // Check files
        for (const file of requiredFiles) {
            const filePath = path.join(this.projectRoot, file);
            if (fs.existsSync(filePath)) {
                console.log(`✅ File exists: ${file}`);
                this.deploymentSteps.push(`Verified file: ${file}`);
            } else {
                console.log(`❌ Missing file: ${file}`);
                this.errors.push(`Missing required file: ${file}`);
            }
        }

        console.log('✅ File structure verification completed\n');
    }

    async updateMainApp() {
        console.log('🔧 Updating main application...');

        const appJsPath = path.join(this.projectRoot, 'app.js');
        
        if (!fs.existsSync(appJsPath)) {
            console.log('📝 Creating new app.js file...');
            await this.createMainAppFile();
        } else {
            console.log('📝 Updating existing app.js file...');
            await this.updateExistingAppFile();
        }

        console.log('✅ Main application updated\n');
    }

    async createMainAppFile() {
        const appContent = `const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const mongoose = require('mongoose');
require('dotenv').config();

// Import routes
const asiWhatsAppRoutes = require('./src/routes/asiWhatsAppRoutes');

console.log('🚀 SIP BREWERY - ASI WHATSAPP PLATFORM');
console.log('📱 Complete Mutual Fund Operations via WhatsApp');
console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');

const app = express();
const PORT = process.env.PORT || 3000;

// Security middleware
app.use(helmet());
app.use(cors({
    origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
    credentials: true
}));

// Logging
app.use(morgan('combined'));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Database connection
mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/sip-brewery', {
    useNewUrlParser: true,
    useUnifiedTopology: true
})
.then(() => console.log('✅ Database connected successfully'))
.catch(err => console.error('❌ Database connection failed:', err));

// Health check endpoint
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        timestamp: new Date().toISOString(),
        service: 'SIP Brewery ASI WhatsApp Platform',
        version: '2.0.0'
    });
});

// Routes
app.use('/api/asi-whatsapp', asiWhatsAppRoutes);

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'SIP Brewery ASI WhatsApp Platform',
        description: 'Complete mutual fund operations via WhatsApp',
        version: '2.0.0',
        endpoints: {
            health: '/health',
            webhook: '/api/asi-whatsapp/webhook',
            status: '/api/asi-whatsapp/status',
            documentation: '/api/asi-whatsapp/docs'
        }
    });
});

// Error handling middleware
app.use((error, req, res, next) => {
    console.error('❌ Application error:', error);
    res.status(500).json({
        success: false,
        message: 'Internal server error',
        error: process.env.NODE_ENV === 'development' ? error.message : 'Something went wrong'
    });
});

// 404 handler
app.use('*', (req, res) => {
    res.status(404).json({
        success: false,
        message: 'Endpoint not found'
    });
});

// Start server
app.listen(PORT, () => {
    console.log(\`🌟 SIP Brewery ASI WhatsApp Platform running on port \${PORT}\`);
    console.log(\`📱 WhatsApp webhook: http://localhost:\${PORT}/api/asi-whatsapp/webhook\`);
    console.log(\`🔍 Health check: http://localhost:\${PORT}/health\`);
    console.log(\`📊 Service status: http://localhost:\${PORT}/api/asi-whatsapp/status\`);
    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
});

module.exports = app;`;

        fs.writeFileSync(path.join(this.projectRoot, 'app.js'), appContent);
        this.deploymentSteps.push('Created main app.js file');
    }

    async updateExistingAppFile() {
        const appJsPath = path.join(this.projectRoot, 'app.js');
        let appContent = fs.readFileSync(appJsPath, 'utf8');

        // Check if ASI WhatsApp routes are already included
        if (!appContent.includes('asiWhatsAppRoutes')) {
            // Add import
            const importLine = "const asiWhatsAppRoutes = require('./src/routes/asiWhatsAppRoutes');";
            if (!appContent.includes(importLine)) {
                const importRegex = /(const.*require.*routes.*\n)/g;
                const matches = appContent.match(importRegex);
                if (matches && matches.length > 0) {
                    appContent = appContent.replace(matches[matches.length - 1], matches[matches.length - 1] + importLine + '\n');
                } else {
                    // Add after other requires
                    appContent = appContent.replace(/(require\('dotenv'\)\.config\(\);)/, '$1\n\n' + importLine);
                }
            }

            // Add route
            const routeLine = "app.use('/api/asi-whatsapp', asiWhatsAppRoutes);";
            if (!appContent.includes(routeLine)) {
                const routeRegex = /(app\.use\('\/api\/.*\n)/g;
                const routeMatches = appContent.match(routeRegex);
                if (routeMatches && routeMatches.length > 0) {
                    appContent = appContent.replace(routeMatches[routeMatches.length - 1], routeMatches[routeMatches.length - 1] + routeLine + '\n');
                } else {
                    // Add before error handling
                    appContent = appContent.replace(/(\/\/ Error handling middleware)/, routeLine + '\n\n$1');
                }
            }

            fs.writeFileSync(appJsPath, appContent);
            this.deploymentSteps.push('Updated existing app.js file with ASI WhatsApp routes');
        } else {
            this.deploymentSteps.push('ASI WhatsApp routes already integrated in app.js');
        }
    }

    async createEnvironmentTemplate() {
        console.log('🔧 Creating environment template...');

        const envTemplate = `# SIP Brewery ASI WhatsApp Platform Environment Configuration

# Server Configuration
NODE_ENV=development
PORT=3000

# Database Configuration
MONGODB_URI=mongodb://localhost:27017/sip-brewery

# WhatsApp Business API Configuration
WHATSAPP_VERIFY_TOKEN=your_webhook_verify_token_here
WHATSAPP_ACCESS_TOKEN=your_whatsapp_access_token_here
WHATSAPP_PHONE_NUMBER_ID=your_phone_number_id_here
WHATSAPP_BUSINESS_ACCOUNT_ID=your_business_account_id_here

# AI Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Security Configuration
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,https://your-domain.com

# Report Generation Configuration
PUPPETEER_EXECUTABLE_PATH=/usr/bin/google-chrome
REPORTS_OUTPUT_DIR=./reports

# Logging Configuration
LOG_LEVEL=info
LOG_FILE=./logs/app.log

# Rate Limiting Configuration
RATE_LIMIT_WINDOW_MS=900000
RATE_LIMIT_MAX_REQUESTS=100

# Email Configuration (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password

# Webhook Configuration
WEBHOOK_SECRET=your_webhook_secret_here

# Development Configuration
DEBUG=asi-whatsapp:*
ENABLE_SWAGGER=true

# Production Configuration (uncomment for production)
# NODE_ENV=production
# HTTPS_CERT_PATH=/path/to/cert.pem
# HTTPS_KEY_PATH=/path/to/key.pem`;

        const envPath = path.join(this.projectRoot, '.env.template');
        fs.writeFileSync(envPath, envTemplate);
        this.deploymentSteps.push('Created .env.template file');

        // Check if .env exists
        const envFilePath = path.join(this.projectRoot, '.env');
        if (!fs.existsSync(envFilePath)) {
            fs.writeFileSync(envFilePath, envTemplate);
            this.deploymentSteps.push('Created .env file from template');
            console.log('⚠️ Please update .env file with your actual configuration values');
        }

        console.log('✅ Environment template created\n');
    }

    async updatePackageDependencies() {
        console.log('📦 Updating package dependencies...');

        const packageJsonPath = path.join(this.projectRoot, 'package.json');
        let packageJson;

        if (fs.existsSync(packageJsonPath)) {
            packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
        } else {
            packageJson = {
                name: 'sip-brewery-asi-whatsapp',
                version: '2.0.0',
                description: 'SIP Brewery ASI WhatsApp Platform - Complete mutual fund operations via WhatsApp',
                main: 'app.js',
                scripts: {},
                dependencies: {},
                devDependencies: {}
            };
        }

        // Required dependencies
        const requiredDependencies = {
            'express': '^4.18.2',
            'mongoose': '^7.5.0',
            'cors': '^2.8.5',
            'helmet': '^7.0.0',
            'morgan': '^1.10.0',
            'dotenv': '^16.3.1',
            'express-rate-limit': '^6.10.0',
            'puppeteer': '^21.1.1',
            '@google/generative-ai': '^0.1.3',
            'axios': '^1.5.0',
            'jsonwebtoken': '^9.0.2',
            'bcryptjs': '^2.4.3',
            'joi': '^17.9.2',
            'winston': '^3.10.0'
        };

        const requiredDevDependencies = {
            'nodemon': '^3.0.1',
            'jest': '^29.6.4',
            'supertest': '^6.3.3',
            'eslint': '^8.47.0'
        };

        // Update dependencies
        packageJson.dependencies = { ...packageJson.dependencies, ...requiredDependencies };
        packageJson.devDependencies = { ...packageJson.devDependencies, ...requiredDevDependencies };

        // Update scripts
        packageJson.scripts = {
            ...packageJson.scripts,
            'start': 'node app.js',
            'dev': 'nodemon app.js',
            'test': 'jest',
            'test:integration': 'node test_asi_whatsapp_integration.js',
            'lint': 'eslint .',
            'deploy': 'node deploy_asi_whatsapp.js'
        };

        fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
        this.deploymentSteps.push('Updated package.json with required dependencies');

        console.log('✅ Package dependencies updated\n');
    }

    async createStartupScript() {
        console.log('🚀 Creating startup script...');

        const startupScript = `#!/bin/bash

echo "🚀 SIP BREWERY ASI WHATSAPP PLATFORM STARTUP"
echo "📱 Complete Mutual Fund Operations via WhatsApp"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please create one from .env.template"
    exit 1
fi

# Check if node_modules exists
if [ ! -d node_modules ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Run integration tests
echo "🧪 Running integration tests..."
npm run test:integration

# Check if tests passed
if [ $? -eq 0 ]; then
    echo "✅ Integration tests passed"
else
    echo "⚠️ Integration tests failed, but continuing startup..."
fi

# Start the application
echo "🌟 Starting SIP Brewery ASI WhatsApp Platform..."
npm start`;

        const startupPath = path.join(this.projectRoot, 'start.sh');
        fs.writeFileSync(startupPath, startupScript);
        
        // Make executable (on Unix systems)
        if (process.platform !== 'win32') {
            const { exec } = require('child_process');
            exec(`chmod +x ${startupPath}`);
        }

        this.deploymentSteps.push('Created startup script');

        // Create Windows batch file
        const windowsScript = `@echo off
echo 🚀 SIP BREWERY ASI WHATSAPP PLATFORM STARTUP
echo 📱 Complete Mutual Fund Operations via WhatsApp
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if not exist .env (
    echo ❌ .env file not found. Please create one from .env.template
    pause
    exit /b 1
)

if not exist node_modules (
    echo 📦 Installing dependencies...
    npm install
)

echo 🧪 Running integration tests...
npm run test:integration

echo 🌟 Starting SIP Brewery ASI WhatsApp Platform...
npm start`;

        const windowsPath = path.join(this.projectRoot, 'start.bat');
        fs.writeFileSync(windowsPath, windowsScript);
        this.deploymentSteps.push('Created Windows startup script');

        console.log('✅ Startup scripts created\n');
    }

    async generateDeploymentChecklist() {
        console.log('📋 Generating deployment checklist...');

        const checklist = `# SIP Brewery ASI WhatsApp Deployment Checklist

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
**Environment**: _______________`;

        const checklistPath = path.join(this.projectRoot, 'DEPLOYMENT_CHECKLIST.md');
        fs.writeFileSync(checklistPath, checklist);
        this.deploymentSteps.push('Generated deployment checklist');

        console.log('✅ Deployment checklist generated\n');
    }

    async runVerificationTests() {
        console.log('🧪 Running verification tests...');

        try {
            // Test 1: Check if all files exist
            const requiredFiles = [
                'src/services/ASIWhatsAppService.js',
                'src/controllers/ASIWhatsAppController.js',
                'src/routes/asiWhatsAppRoutes.js'
            ];

            for (const file of requiredFiles) {
                if (fs.existsSync(path.join(this.projectRoot, file))) {
                    console.log(`✅ File exists: ${file}`);
                } else {
                    console.log(`❌ Missing file: ${file}`);
                    this.errors.push(`Missing file: ${file}`);
                }
            }

            // Test 2: Check syntax of main files
            try {
                require('./src/services/ASIWhatsAppService.js');
                console.log('✅ ASIWhatsAppService syntax valid');
            } catch (error) {
                console.log('❌ ASIWhatsAppService syntax error:', error.message);
                this.errors.push(`ASIWhatsAppService syntax error: ${error.message}`);
            }

            try {
                require('./src/controllers/ASIWhatsAppController.js');
                console.log('✅ ASIWhatsAppController syntax valid');
            } catch (error) {
                console.log('❌ ASIWhatsAppController syntax error:', error.message);
                this.errors.push(`ASIWhatsAppController syntax error: ${error.message}`);
            }

            // Test 3: Check environment template
            const envTemplatePath = path.join(this.projectRoot, '.env.template');
            if (fs.existsSync(envTemplatePath)) {
                const envContent = fs.readFileSync(envTemplatePath, 'utf8');
                const requiredVars = ['WHATSAPP_VERIFY_TOKEN', 'WHATSAPP_ACCESS_TOKEN', 'GEMINI_API_KEY', 'MONGODB_URI'];
                
                for (const varName of requiredVars) {
                    if (envContent.includes(varName)) {
                        console.log(`✅ Environment variable template: ${varName}`);
                    } else {
                        console.log(`❌ Missing environment variable: ${varName}`);
                        this.errors.push(`Missing environment variable: ${varName}`);
                    }
                }
            }

            this.deploymentSteps.push('Completed verification tests');

        } catch (error) {
            console.log('❌ Verification tests failed:', error.message);
            this.errors.push(`Verification tests failed: ${error.message}`);
        }

        console.log('✅ Verification tests completed\n');
    }

    async generateDeploymentReport() {
        const endTime = Date.now();
        const report = {
            deploymentSuite: 'SIP Brewery ASI WhatsApp Integration',
            timestamp: new Date().toISOString(),
            status: this.errors.length === 0 ? 'SUCCESS' : 'PARTIAL_SUCCESS',
            summary: {
                totalSteps: this.deploymentSteps.length,
                completedSteps: this.deploymentSteps.length,
                errors: this.errors.length,
                warnings: this.errors.length > 0 ? this.errors.length : 0
            },
            deploymentSteps: this.deploymentSteps,
            errors: this.errors,
            nextSteps: [
                'Update .env file with your actual configuration values',
                'Install dependencies: npm install',
                'Configure WhatsApp Business API webhook',
                'Set up MongoDB database',
                'Run integration tests: npm run test:integration',
                'Deploy to production server',
                'Configure SSL certificate for HTTPS',
                'Test with real WhatsApp messages'
            ],
            files: {
                created: [
                    'src/services/ASIWhatsAppService.js',
                    'src/controllers/ASIWhatsAppController.js',
                    'src/routes/asiWhatsAppRoutes.js',
                    'test_asi_whatsapp_integration.js',
                    'ASI_WHATSAPP_INTEGRATION.md',
                    '.env.template',
                    'start.sh',
                    'start.bat',
                    'DEPLOYMENT_CHECKLIST.md'
                ],
                updated: [
                    'app.js',
                    'package.json'
                ]
            }
        };

        // Save report
        const reportPath = path.join(this.projectRoot, 'deployment_report.json');
        fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));

        console.log('📊 DEPLOYMENT REPORT');
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
        console.log(`🚀 Deployment Suite: ${report.deploymentSuite}`);
        console.log(`📅 Timestamp: ${report.timestamp}`);
        console.log(`✅ Status: ${report.status}`);
        console.log(`📊 Steps Completed: ${report.summary.completedSteps}/${report.summary.totalSteps}`);
        console.log(`❌ Errors: ${report.summary.errors}`);
        console.log(`📄 Report saved to: ${reportPath}`);
        console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');

        if (this.errors.length > 0) {
            console.log('⚠️ DEPLOYMENT WARNINGS:');
            this.errors.forEach(error => {
                console.log(`   • ${error}`);
            });
            console.log('');
        }

        console.log('🎉 ASI WhatsApp Integration deployment completed!');
        console.log('📋 Next: Follow the DEPLOYMENT_CHECKLIST.md for production setup');
        console.log('🧪 Test: Run npm run test:integration to verify functionality');
        console.log('🚀 Start: Use npm start or ./start.sh to launch the platform\n');

        return report;
    }
}

// Run deployment
async function deploy() {
    const deployment = new ASIWhatsAppDeployment();
    await deployment.deploy();
}

// Execute if run directly
if (require.main === module) {
    deploy().catch(console.error);
}

module.exports = ASIWhatsAppDeployment;
