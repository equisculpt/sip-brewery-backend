@echo off
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
npm start