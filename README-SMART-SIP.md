# 🧠 Smart SIP (Dynamic SIP) Module

A complete backend module for AI-powered Smart SIP (Systematic Investment Plan) with market-timed adjustments, built with Node.js, Express, and MongoDB.

## 🎯 Overview

Smart SIP automatically adjusts your monthly investment amount based on market conditions using AI-powered analysis. Instead of investing a fixed amount every month, Smart SIP invests more when markets are down (buying low) and less when markets are expensive (avoiding overvaluation).

## ✨ Features

### 🧠 AI-Powered Market Analysis
- **Market Score Calculation**: Analyzes market conditions from -1 (overheated) to +1 (bottomed)
- **Multi-Factor Analysis**: P/E ratio, RSI, sentiment, Fear & Greed Index, breakout indicators
- **Dynamic SIP Adjustment**: Automatically calculates optimal investment amount
- **AI Commentary**: Placeholder for Gemini/OpenAI integration

### 📊 Smart SIP Logic
- **Static SIP**: Fixed amount per month (e.g., ₹10,000/month)
- **Smart SIP**: Variable amount based on market conditions
  - Average SIP: ₹20,000/month
  - Min SIP: ₹16,000 (80% of average)
  - Max SIP: ₹24,000 (120% of average)

### 🔄 Automated Execution
- **Daily Market Analysis**: Updates market conditions at 9:00 AM IST
- **Daily SIP Execution**: Processes eligible SIPs at 9:30 AM IST
- **Weekly Analytics**: Generates performance reports on Sundays
- **Manual Triggers**: API endpoints for immediate execution

### 📈 Performance Tracking
- **SIP History**: Complete record of all investments with market analysis
- **Performance Metrics**: Total invested, average amount, best/worst SIPs
- **Market Timing Efficiency**: Comparison with static SIP performance
- **Analytics Dashboard**: Comprehensive performance insights

## 🏗️ Architecture

```
Smart SIP Module
├── Models/
│   └── SmartSip.js          # MongoDB schema with business logic
├── Services/
│   ├── marketScoreService.js # AI market analysis engine
│   ├── smartSipService.js   # Core business logic
│   └── cronService.js       # Automated execution
├── Controllers/
│   └── smartSipController.js # API request handling
└── Routes/
    └── smartSip.js          # RESTful API endpoints
```

## 📋 API Endpoints

### 🔐 Authentication Required
All endpoints require valid Supabase JWT token in Authorization header:
```
Authorization: Bearer <jwt-token>
```

### 🚀 SIP Management

#### `POST /api/sip/start`
Start a new SIP (static or smart)

**Request Body:**
```json
{
  "sipType": "SMART",
  "averageSip": 20000,
  "fundSelection": [
    {
      "schemeCode": "HDFC001",
      "schemeName": "HDFC Flexicap Fund",
      "allocation": 60
    },
    {
      "schemeCode": "PARAG001",
      "schemeName": "Parag Parikh Flexicap Fund",
      "allocation": 40
    }
  ],
  "sipDay": 1,
  "preferences": {
    "riskTolerance": "MODERATE",
    "marketTiming": true,
    "aiEnabled": true,
    "notifications": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "sipType": "SMART",
    "minSip": 16000,
    "maxSip": 24000,
    "nextSIPDate": "2025-08-01T00:00:00.000Z",
    "message": "SIP started successfully"
  }
}
```

#### `GET /api/sip/recommendation`
Get current SIP recommendation based on market conditions

**Response:**
```json
{
  "success": true,
  "data": {
    "date": "2025-07-15",
    "marketScore": 0.65,
    "reason": "Market appears oversold with P/E at 18.2, RSI at 35.1, and bearish sentiment. Good opportunity to increase SIP.",
    "recommendedSIP": 23800,
    "fundSplit": {
      "HDFC Flexicap Fund": 60,
      "Parag Parikh Flexicap Fund": 40
    },
    "indicators": {
      "peRatio": 18.2,
      "rsi": 35.1,
      "breakout": false,
      "sentiment": "BEARISH",
      "fearGreedIndex": 25
    },
    "sipType": "SMART",
    "nextSIPDate": "2025-08-01T00:00:00.000Z"
  }
}
```

#### `GET /api/sip/details`
Get user's SIP details and current status

**Response:**
```json
{
  "success": true,
  "data": {
    "sipType": "SMART",
    "averageSip": 20000,
    "minSip": 16000,
    "maxSip": 24000,
    "fundSelection": [...],
    "status": "ACTIVE",
    "nextSIPDate": "2025-08-01T00:00:00.000Z",
    "lastSIPAmount": 23800,
    "performance": {
      "totalInvested": 23800,
      "totalSIPs": 1,
      "averageAmount": 23800,
      "bestSIPAmount": 23800,
      "worstSIPAmount": 23800
    },
    "currentRecommendation": {...},
    "recentHistory": [...],
    "preferences": {...}
  }
}
```

### ⚙️ SIP Configuration

#### `PUT /api/sip/preferences`
Update SIP preferences

**Request Body:**
```json
{
  "riskTolerance": "AGGRESSIVE",
  "marketTiming": true,
  "aiEnabled": false,
  "notifications": true
}
```

#### `PUT /api/sip/status`
Update SIP status (pause/resume/stop)

**Request Body:**
```json
{
  "status": "PAUSED"
}
```

### 📊 Analytics & History

#### `GET /api/sip/analytics`
Get SIP analytics and performance metrics

**Response:**
```json
{
  "success": true,
  "data": {
    "totalSIPs": 12,
    "totalInvested": 245600,
    "averageAmount": 20467,
    "bestAmount": 24000,
    "worstAmount": 16000,
    "marketTimingEfficiency": 2.34,
    "sipType": "SMART",
    "performance": {...}
  }
}
```

#### `GET /api/sip/history?limit=10`
Get SIP history with optional limit

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "date": "2025-07-01T00:00:00.000Z",
      "amount": 23800,
      "marketScore": 0.65,
      "marketReason": "Market appears oversold...",
      "fundSplit": {...},
      "executed": true,
      "transactionId": "txn_123"
    }
  ]
}
```

### 🧠 Market Analysis

#### `GET /api/sip/market-analysis`
Get current market analysis for frontend display

**Response:**
```json
{
  "success": true,
  "data": {
    "score": 0.65,
    "reason": "Market appears oversold...",
    "indicators": {
      "peRatio": 18.2,
      "rsi": 35.1,
      "breakout": false,
      "sentiment": "BEARISH",
      "fearGreedIndex": 25,
      "macd": -0.3,
      "volume": 0.7
    },
    "timestamp": "2025-07-15T09:00:00.000Z"
  }
}
```

### 🔄 Manual Execution

#### `POST /api/sip/execute`
Execute SIP manually (for testing or immediate execution)

**Response:**
```json
{
  "success": true,
  "data": {
    "amount": 23800,
    "marketScore": 0.65,
    "reason": "Market appears oversold...",
    "nextSIPDate": "2025-08-01T00:00:00.000Z",
    "message": "SIP executed successfully"
  }
}
```

## 🧠 Market Analysis Algorithm

### Market Score Calculation
The system calculates a market score from -1 to +1 using weighted factors:

1. **P/E Ratio (30% weight)**
   - < 15: Very attractive (0.8 to 1.0)
   - 15-18: Attractive (0.4 to 0.8)
   - 18-22: Fair value (-0.2 to 0.4)
   - 22-25: Expensive (-0.6 to -0.2)
   - > 25: Very expensive (-1.0 to -0.6)

2. **RSI (25% weight)**
   - < 30: Oversold (0.6 to 1.0)
   - 30-45: Slightly oversold (0.2 to 0.6)
   - 45-55: Neutral (-0.2 to 0.2)
   - 55-70: Slightly overbought (-0.6 to -0.2)
   - > 70: Overbought (-1.0 to -0.6)

3. **Sentiment (20% weight)**
   - BEARISH: 0.6 (good buying opportunity)
   - NEUTRAL: 0.0
   - BULLISH: -0.6 (expensive)

4. **Fear & Greed Index (15% weight)**
   - 0-25: Extreme Fear (0.8 to 1.0)
   - 25-45: Fear (0.4 to 0.8)
   - 45-55: Neutral (-0.2 to 0.4)
   - 55-75: Greed (-0.6 to -0.2)
   - 75-100: Extreme Greed (-1.0 to -0.6)

5. **Breakout Indicator (10% weight)**
   - True: -0.3 (market breaking out)
   - False: 0.3 (no breakout)

### SIP Amount Calculation
Based on market score:
- **Score > 0.5**: Invest maximum amount (₹24,000)
- **Score < -0.5**: Invest minimum amount (₹16,000)
- **Score -0.5 to 0.5**: Interpolate between min and max

## ⏰ Cron Jobs

### Daily Jobs (IST Timezone)
- **9:00 AM**: Market analysis update for all active SIPs
- **9:30 AM**: SIP execution for eligible users

### Weekly Jobs
- **Sunday 10:00 AM**: Weekly analytics and report generation

### Manual Triggers
```javascript
const cronService = require('./src/services/cronService');

// Trigger market analysis update
await cronService.triggerMarketAnalysis();

// Trigger SIP execution
await cronService.triggerSIPExecution();

// Get job status
const status = cronService.getJobStatus();
```

## 🗄️ Database Schema

### SmartSip Collection
```javascript
{
  userId: String,                    // Required, indexed
  sipType: "STATIC" | "SMART",      // Required
  averageSip: Number,               // Required (1000-100000)
  minSip: Number,                   // Auto-calculated (80% of average)
  maxSip: Number,                   // Auto-calculated (120% of average)
  fundSelection: [{
    schemeCode: String,             // Required
    schemeName: String,             // Required
    allocation: Number              // Required (0-100)
  }],
  lastSIPAmount: Number,            // Default: 0
  nextSIPDate: Date,                // Required
  sipDay: Number,                   // Required (1-31)
  status: "ACTIVE" | "PAUSED" | "STOPPED",
  sipHistory: [{
    date: Date,                     // Required
    amount: Number,                 // Required
    marketScore: Number,            // -1 to 1
    marketReason: String,
    fundSplit: Map,
    executed: Boolean,              // Default: false
    transactionId: String
  }],
  marketAnalysis: {
    lastUpdated: Date,
    currentScore: Number,           // -1 to 1
    currentReason: String,
    recommendedAmount: Number,
    indicators: {
      peRatio: Number,
      rsi: Number,
      breakout: Boolean,
      sentiment: String,
      fearGreedIndex: Number
    }
  },
  preferences: {
    riskTolerance: "CONSERVATIVE" | "MODERATE" | "AGGRESSIVE",
    marketTiming: Boolean,          // Default: true
    aiEnabled: Boolean,             // Default: true
    notifications: Boolean          // Default: true
  },
  performance: {
    totalInvested: Number,          // Default: 0
    totalSIPs: Number,              // Default: 0
    averageAmount: Number,          // Default: 0
    bestSIPAmount: Number,          // Default: 0
    worstSIPAmount: Number          // Default: 0
  },
  isActive: Boolean,                // Default: true
  createdAt: Date,
  updatedAt: Date
}
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
npm install node-cron
```

### 2. Environment Variables
Add to your `.env` file:
```env
ENABLE_CRON=true  # Enable cron jobs in development
```

### 3. Run Tests
```bash
node test-smart-sip.js
```

### 4. Start Server
```bash
npm start
```

### 5. Test API Endpoints
```bash
# Start a Smart SIP
curl -X POST http://localhost:3000/api/sip/start \
  -H "Authorization: Bearer <your-jwt-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "sipType": "SMART",
    "averageSip": 20000,
    "fundSelection": [
      {
        "schemeCode": "HDFC001",
        "schemeName": "HDFC Flexicap Fund",
        "allocation": 60
      },
      {
        "schemeCode": "PARAG001",
        "schemeName": "Parag Parikh Flexicap Fund",
        "allocation": 40
      }
    ]
  }'

# Get current recommendation
curl -X GET http://localhost:3000/api/sip/recommendation \
  -H "Authorization: Bearer <your-jwt-token>"
```

## 🔧 Frontend Integration

### React Component Example
```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const SmartSIPSetup = () => {
  const [sipData, setSipData] = useState({
    sipType: 'SMART',
    averageSip: 20000,
    fundSelection: []
  });
  const [recommendation, setRecommendation] = useState(null);

  const startSIP = async () => {
    try {
      const response = await axios.post('/api/sip/start', sipData, {
        headers: { Authorization: `Bearer ${token}` }
      });
      console.log('SIP started:', response.data);
    } catch (error) {
      console.error('Error starting SIP:', error);
    }
  };

  const getRecommendation = async () => {
    try {
      const response = await axios.get('/api/sip/recommendation', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setRecommendation(response.data.data);
    } catch (error) {
      console.error('Error getting recommendation:', error);
    }
  };

  return (
    <div>
      <h2>Smart SIP Setup</h2>
      
      {/* SIP Type Selection */}
      <div>
        <label>
          <input
            type="radio"
            value="STATIC"
            checked={sipData.sipType === 'STATIC'}
            onChange={(e) => setSipData({...sipData, sipType: e.target.value})}
          />
          Static SIP (Fixed Amount)
        </label>
        <label>
          <input
            type="radio"
            value="SMART"
            checked={sipData.sipType === 'SMART'}
            onChange={(e) => setSipData({...sipData, sipType: e.target.value})}
          />
          Smart SIP (Market-Timed)
        </label>
      </div>

      {/* Average SIP Amount */}
      <div>
        <label>Average SIP Amount (₹)</label>
        <input
          type="number"
          value={sipData.averageSip}
          onChange={(e) => setSipData({...sipData, averageSip: parseInt(e.target.value)})}
          min="1000"
          max="100000"
        />
      </div>

      {/* Smart SIP Range Display */}
      {sipData.sipType === 'SMART' && (
        <div>
          <p>Min SIP: ₹{Math.round(sipData.averageSip * 0.8)}</p>
          <p>Max SIP: ₹{Math.round(sipData.averageSip * 1.2)}</p>
        </div>
      )}

      {/* Current Recommendation */}
      {recommendation && (
        <div>
          <h3>Current Recommendation</h3>
          <p>Market Score: {recommendation.marketScore}</p>
          <p>Reason: {recommendation.reason}</p>
          <p>Recommended Amount: ₹{recommendation.recommendedSIP}</p>
        </div>
      )}

      <button onClick={getRecommendation}>Get Recommendation</button>
      <button onClick={startSIP}>Start SIP</button>
    </div>
  );
};

export default SmartSIPSetup;
```

## 🔮 Future Enhancements

### 1. AI Integration
- **Gemini/OpenAI Integration**: Real AI-powered market analysis
- **Sentiment Analysis**: Social media and news sentiment
- **Predictive Models**: Machine learning for market prediction

### 2. Advanced Features
- **Portfolio Rebalancing**: Automatic fund allocation adjustment
- **Tax Optimization**: Tax-loss harvesting and optimization
- **Goal-Based Investing**: Target-based SIP adjustments

### 3. Integration
- **BSE Star MF API**: Real SIP execution
- **Notification Service**: Email/SMS alerts
- **Reporting Service**: Advanced analytics and reports

### 4. Mobile App
- **Push Notifications**: Real-time SIP updates
- **Offline Support**: Basic functionality without internet
- **Biometric Auth**: Secure authentication

## 📊 Performance Metrics

### Market Timing Efficiency
```
Efficiency = ((Smart SIP Total - Static SIP Total) / Static SIP Total) × 100
```

### Example Calculation
- Static SIP: ₹20,000 × 12 months = ₹2,40,000
- Smart SIP: ₹2,45,600 (varied amounts)
- Efficiency: ((2,45,600 - 2,40,000) / 2,40,000) × 100 = 2.34%

## 🛡️ Security & Compliance

- **Authentication**: Supabase JWT token validation
- **Authorization**: User-specific data access
- **Rate Limiting**: API request throttling
- **Data Encryption**: Sensitive data encryption
- **Audit Trail**: Complete SIP history tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Built with ❤️ for SIP Brewery - Making Smart Investing Accessible** 