/**
 * Simple SIP Brewery Backend
 * Basic working version for frontend development
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');

const app = express();
const PORT = process.env.PORT || 3000;

// Basic middleware
app.use(helmet());
app.use(cors({
  origin: ['http://localhost:3001', 'http://localhost:3000'],
  credentials: true
}));
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Basic logging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.path}`);
  next();
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    success: true,
    message: 'SIP Brewery Backend is running',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

// Status endpoint
app.get('/status', (req, res) => {
  res.json({
    success: true,
    message: 'Backend operational',
    data: {
      uptime: process.uptime(),
      memory: process.memoryUsage(),
      timestamp: new Date().toISOString()
    }
  });
});

// Mock mutual fund chart data endpoint
app.get('/api/mutual-funds/chart/:schemeCode', (req, res) => {
  const { schemeCode } = req.params;
  const { period = '1Y' } = req.query;
  
  // Generate mock chart data
  const now = Date.now();
  const oneDay = 24 * 60 * 60 * 1000;
  const days = period === '1M' ? 30 : period === '3M' ? 90 : period === '6M' ? 180 : 365;
  const mockCandles = [];
  
  for (let i = days - 1; i >= 0; i--) {
    const time = now - (i * oneDay);
    const basePrice = 100 + Math.sin(i * 0.1) * 10;
    const open = basePrice + (Math.random() - 0.5) * 2;
    const close = open + (Math.random() - 0.5) * 3;
    const high = Math.max(open, close) + Math.random() * 2;
    const low = Math.min(open, close) - Math.random() * 2;
    
    mockCandles.push({
      time: Math.floor(time / 1000),
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Math.floor(Math.random() * 1000000),
      nav: Number(close.toFixed(2))
    });
  }
  
  const mockData = {
    scheme: {
      code: schemeCode,
      name: `Mock Fund ${schemeCode}`,
      category: 'Large Cap',
      aum: '₹1,000 Cr',
      expense_ratio: 1.2
    },
    period,
    candles: mockCandles,
    indicators: {
      moving_averages: {
        sma_20: mockCandles.map((_, i) => ({ index: i, value: 100 + Math.random() * 10 })),
        sma_50: mockCandles.map((_, i) => ({ index: i, value: 95 + Math.random() * 10 })),
        sma_100: mockCandles.map((_, i) => ({ index: i, value: 90 + Math.random() * 10 })),
        sma_200: mockCandles.map((_, i) => ({ index: i, value: 85 + Math.random() * 10 })),
        ema_12: mockCandles.map((_, i) => ({ index: i, value: 102 + Math.random() * 8 })),
        ema_26: mockCandles.map((_, i) => ({ index: i, value: 98 + Math.random() * 8 })),
        ema_50: mockCandles.map((_, i) => ({ index: i, value: 94 + Math.random() * 8 }))
      },
      oscillators: {
        rsi: mockCandles.map((_, i) => ({ index: i, value: 30 + Math.random() * 40 })),
        macd: {
          macd: mockCandles.map((_, i) => ({ index: i, value: (Math.random() - 0.5) * 2 })),
          signal: mockCandles.map((_, i) => ({ index: i, value: (Math.random() - 0.5) * 1.5 })),
          histogram: mockCandles.map((_, i) => ({ index: i, value: (Math.random() - 0.5) * 1 }))
        },
        stochastic: {
          k: mockCandles.map((_, i) => ({ index: i, value: 20 + Math.random() * 60 })),
          d: mockCandles.map((_, i) => ({ index: i, value: 25 + Math.random() * 50 }))
        }
      },
      volatility: {
        bollinger_bands: {
          upper: mockCandles.map((_, i) => ({ index: i, value: 105 + Math.random() * 5 })),
          middle: mockCandles.map((_, i) => ({ index: i, value: 100 + Math.random() * 3 })),
          lower: mockCandles.map((_, i) => ({ index: i, value: 95 + Math.random() * 5 }))
        }
      },
      volume: {
        volume_sma: mockCandles.map((_, i) => ({ index: i, value: 500000 + Math.random() * 200000 }))
      }
    },
    statistics: {
      total_return: 12.5,
      annualized_return: 15.2,
      volatility: 18.3,
      sharpe_ratio: 1.2,
      max_drawdown: -8.5,
      current_nav: mockCandles[mockCandles.length - 1].nav
    },
    metadata: {
      total_records: mockCandles.length,
      start_date: new Date(now - days * oneDay).toISOString().split('T')[0],
      end_date: new Date(now).toISOString().split('T')[0],
      last_updated: new Date().toISOString()
    }
  };
  
  res.json({
    success: true,
    message: 'Chart data retrieved successfully',
    data: mockData
  });
});

// Mock ASI endpoints
app.post('/api/asi/process', (req, res) => {
  res.json({
    success: true,
    message: 'ASI processing completed',
    data: {
      result: 'Mock ASI response',
      analysis: 'This is a mock response for development',
      timestamp: new Date().toISOString()
    }
  });
});

// Catch-all for API routes
app.use('/api/*', (req, res) => {
  res.json({
    success: false,
    message: 'API endpoint not implemented yet',
    endpoint: req.path,
    method: req.method
  });
});

// Error handling middleware
app.use((err, req, res, next) => {
  console.error('Error:', err);
  res.status(500).json({
    success: false,
    message: 'Internal server error',
    error: process.env.NODE_ENV === 'development' ? err.message : 'Something went wrong'
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    message: 'Route not found',
    path: req.originalUrl
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 SIP Brewery Backend running on port ${PORT}`);
  console.log(`📊 Health check: http://localhost:${PORT}/health`);
  console.log(`📈 API Base: http://localhost:${PORT}/api`);
  console.log(`⚡ Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = app;
