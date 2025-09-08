#!/bin/bash

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
npm start