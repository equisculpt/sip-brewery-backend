/**
 * 🚀 SIMPLE TEST SERVER
 * Basic server to test if our backend is working
 */

const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    message: 'SIP Brewery Backend is running!',
    timestamp: new Date().toISOString(),
    version: '3.0.0'
  });
});

// Status endpoint
app.get('/status', (req, res) => {
  res.json({
    platform: 'SIP Brewery Backend',
    version: '3.0.0',
    environment: process.env.NODE_ENV || 'development',
    features: [
      '🚀 Unified ASI API',
      '🧠 Python AI Integration',
      '📊 Real-time Analytics',
      '💰 Mutual Fund Analysis',
      '🤖 WhatsApp Integration'
    ],
    endpoints: {
      health: '/health',
      status: '/status',
      asi: '/api/asi/*'
    }
  });
});

// Test ASI endpoint (mock for now)
app.post('/api/asi/process', (req, res) => {
  res.json({
    success: true,
    message: 'ASI endpoint is working!',
    data: {
      request: req.body,
      processing: 'Mock ASI processing',
      result: 'Backend is ready for frontend integration'
    },
    timestamp: new Date().toISOString()
  });
});

// Test fund analysis endpoint
app.get('/api/asi/fund/:fundCode/analyze', (req, res) => {
  const { fundCode } = req.params;
  res.json({
    success: true,
    message: 'Fund analysis endpoint working',
    data: {
      fundCode,
      analysis: 'Mock fund analysis',
      recommendation: 'Backend ready for integration'
    },
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Route not found',
    message: 'SIP Brewery Backend Test Server',
    availableEndpoints: [
      'GET /health - Health check',
      'GET /status - Server status',
      'POST /api/asi/process - Test ASI endpoint',
      'GET /api/asi/fund/:fundCode/analyze - Test fund analysis'
    ]
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 SIP Brewery Backend Test Server running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`📈 Status: http://localhost:${PORT}/status`);
  console.log(`🧠 ASI Test: http://localhost:${PORT}/api/asi/process`);
  console.log(`💰 Fund Test: http://localhost:${PORT}/api/asi/fund/TEST001/analyze`);
});

module.exports = app;
