# 🚀 SIP Brewery - Complete Mutual Fund Dashboard Backend

A comprehensive Node.js + Express + MongoDB backend for the SIP Brewery mutual fund platform with Supabase authentication, KYC management, and advanced portfolio analytics.

## 🎯 Features Implemented

### ✅ Authentication & Security
- **Supabase JWT Integration**: Secure token verification
- **Auto User Creation**: Automatic user profile creation on first login
- **KYC Status Management**: Complete KYC workflow support
- **Rate Limiting**: API protection against abuse
- **CORS Configuration**: Secure cross-origin requests

### ✅ Complete Dashboard Modules

#### 📊 Holdings Management
- Fund holdings with real-time NAV
- SIP status tracking
- Performance calculations (XIRR, returns)
- Category-wise portfolio breakdown

#### 🧠 Smart SIP Center
- AI-powered SIP recommendations
- Market-timed investment strategies
- Custom SIP amount ranges
- Next SIP date calculations

#### 📈 Transaction History
- Complete transaction tracking
- SIP, Lumpsum, Redemption support
- Transaction status management
- Historical data analysis

#### 📋 Statements & Reports
- Tax statements generation
- P&L statements
- Transaction statements
- PDF generation support

#### 🎁 Rewards & Cashback
- Multi-tier reward system
- Referral bonuses
- Loyalty rewards
- Cashback on transactions

#### 🔗 Referral Program
- Unique referral codes
- Referral tracking
- Bonus calculations
- Anonymous user protection

#### 🤖 AI Portfolio Analytics
- XIRR calculations
- Risk assessment
- Performance percentile ranking
- Smart recommendations
- Market context analysis

#### 📊 Peer Comparison
- User vs category average
- User vs all users comparison
- Strong/weak fund identification
- Portfolio diversification metrics

#### 📈 Performance Charts
- Multi-period performance data
- Benchmark comparisons
- Chart-ready data format
- Historical trend analysis

#### 👤 User Profile Management
- Complete profile information
- Risk profile assessment
- Investment preferences
- Notification settings

## 🏗️ Architecture

### Database Collections
```
users          - User profiles and KYC status
holdings       - Fund holdings and SIP data
transactions   - All transaction history
rewards        - Rewards and cashback data
ai_insights    - AI-generated analytics
```

### API Structure
```
/api/auth/*    - Authentication and KYC
/api/dashboard/* - Dashboard data endpoints
/api/ai/*      - AI analysis endpoints
/api/benchmark/* - Market data endpoints
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Install dependencies
npm install

# Set up environment variables
cp env.example .env
```

### 2. Environment Variables
```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017/sip-brewery

# Supabase (optional for development)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Server
PORT=3000
NODE_ENV=development
```

### 3. Start Server
```bash
npm start
```

### 4. Seed Sample Data
```bash
node test-dashboard-complete.js
```

## 📚 API Documentation

### Authentication Endpoints

#### Check Authentication
```http
GET /api/auth/check
Authorization: Bearer <supabase_token>

Response:
{
  "success": true,
  "data": {
    "userId": "dummy-uid-123",
    "email": "test@sipbrewery.com",
    "name": "Milin Raijada",
    "kycStatus": "SUCCESS",
    "isActive": true
  }
}
```

#### Get KYC Status
```http
GET /api/auth/kyc/status
Authorization: Bearer <supabase_token>

Response:
{
  "success": true,
  "data": {
    "status": "SUCCESS",
    "isCompleted": true,
    "profile": {
      "name": "Milin Raijada",
      "email": "test@sipbrewery.com",
      "mobile": "+91-9876543210",
      "pan": "ABCDE1234F",
      "riskProfile": "Moderate",
      "investorSince": "2023-12-01T00:00:00.000Z"
    }
  }
}
```

### Dashboard Endpoints

#### Complete Dashboard Data
```http
GET /api/dashboard
Authorization: Bearer <supabase_token>

Response:
{
  "success": true,
  "data": {
    "holdings": [...],
    "smartSIPCenter": {...},
    "transactions": [...],
    "statements": {...},
    "rewards": {...},
    "referral": {...},
    "aiAnalytics": {...},
    "portfolioAnalytics": {...},
    "performanceChart": {...},
    "profile": {...}
  }
}
```

#### Individual Module Endpoints
```http
GET /api/dashboard/holdings          # User holdings
GET /api/dashboard/smart-sip         # Smart SIP data
GET /api/dashboard/transactions      # Transaction history
GET /api/dashboard/statements        # Statement links
GET /api/dashboard/rewards           # Rewards data
GET /api/dashboard/referral          # Referral data
GET /api/dashboard/ai-analytics      # AI insights
GET /api/dashboard/portfolio-analytics # Peer comparison
GET /api/dashboard/performance-chart # Chart data
GET /api/dashboard/profile           # User profile
```

## 💡 Frontend Integration Examples

### 1. Authentication Flow
```javascript
// After Supabase login
const token = supabase.auth.session()?.access_token;

// Verify with backend
const response = await fetch('/api/auth/check', {
  headers: { 'Authorization': `Bearer ${token}` }
});

const userData = await response.json();
if (userData.success) {
  // User authenticated, check KYC
  const kycResponse = await fetch('/api/auth/kyc/status', {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  
  const kycData = await kycResponse.json();
  if (kycData.data.isCompleted) {
    // Show dashboard
    loadDashboard(token);
  } else {
    // Show KYC form
    showKYCForm();
  }
}
```

### 2. Dashboard Loading
```javascript
async function loadDashboard(token) {
  const response = await fetch('/api/dashboard', {
    headers: { 'Authorization': `Bearer ${token}` }
  });
  
  const dashboardData = await response.json();
  
  // Update UI components
  updateHoldings(dashboardData.data.holdings);
  updateTransactions(dashboardData.data.transactions);
  updateRewards(dashboardData.data.rewards);
  updateAnalytics(dashboardData.data.aiAnalytics);
  updatePerformanceChart(dashboardData.data.performanceChart);
}
```

### 3. Individual Module Loading
```javascript
// Load only holdings
const holdingsResponse = await fetch('/api/dashboard/holdings', {
  headers: { 'Authorization': `Bearer ${token}` }
});

// Load only transactions
const transactionsResponse = await fetch('/api/dashboard/transactions?limit=20', {
  headers: { 'Authorization': `Bearer ${token}` }
});
```

## 🔧 Data Models

### User Model
```javascript
{
  uid: String,              // Supabase user ID
  email: String,            // User email
  name: String,             // Full name
  mobile: String,           // Phone number
  pan: String,              // PAN number
  kycStatus: String,        // PENDING/SUCCESS/FAILED/REJECTED
  riskProfile: String,      // Conservative/Moderate/Aggressive
  investorSince: Date,      // Investment start date
  referralCode: String,     // Unique referral code
  profile: Object,          // Extended profile data
  preferences: Object       // User preferences
}
```

### Holding Model
```javascript
{
  userId: String,           // User ID
  schemeCode: String,       // Fund scheme code
  schemeName: String,       // Fund name
  folio: String,            // Folio number
  units: Number,            // Number of units
  currentNav: Number,       // Current NAV
  value: Number,            // Current value
  invested: Number,         // Total invested
  returns: Number,          // Absolute returns
  returnsPercentage: Number, // Percentage returns
  sipStatus: String,        // ACTIVE/PAUSED/STOPPED
  category: String,         // Fund category
  fundHouse: String,        // Fund house name
  riskLevel: String         // Low/Moderate/High
}
```

### Transaction Model
```javascript
{
  userId: String,           // User ID
  transactionId: String,    // Unique transaction ID
  type: String,             // SIP/LUMPSUM/REDEMPTION
  schemeCode: String,       // Fund scheme code
  amount: Number,           // Transaction amount
  units: Number,            // Units bought/sold
  nav: Number,              // NAV at transaction
  date: Date,               // Transaction date
  status: String,           // PENDING/SUCCESS/FAILED
  orderType: String         // BUY/SELL
}
```

## 🎨 Sample Dashboard Response

```json
{
  "success": true,
  "data": {
    "holdings": [
      {
        "schemeName": "HDFC Flexi Cap Fund",
        "folio": "123456789",
        "units": 100.5,
        "currentNav": 85.12,
        "value": 8555.56,
        "invested": 7000,
        "returns": 1555.56,
        "returnsPercentage": 22.22,
        "sipStatus": "ACTIVE",
        "category": "Flexi Cap",
        "fundHouse": "HDFC Mutual Fund"
      }
    ],
    "smartSIPCenter": {
      "customSip": {
        "minAmount": 1000,
        "maxAmount": 10000,
        "strategy": "market-timed",
        "aiEnabled": true
      },
      "activeSIPs": 3,
      "totalSIPAmount": 10000,
      "nextSIPDate": "2024-08-01"
    },
    "transactions": [
      {
        "type": "SIP",
        "date": "2024-07-01",
        "fund": "HDFC Flexi Cap Fund",
        "amount": 5000,
        "units": 58.73,
        "nav": 85.12,
        "status": "SUCCESS"
      }
    ],
    "rewards": {
      "totalPoints": 1240,
      "totalAmount": 1240,
      "referralBonus": 200,
      "loyalty": 300,
      "cashback": 740,
      "history": [...]
    },
    "aiAnalytics": {
      "xirr": 14.5,
      "percentile": "Top 25%",
      "insight": "Your portfolio is beating inflation by 6% annually",
      "nextBestAction": "Increase SIP in smallcap fund",
      "riskScore": "Medium"
    },
    "portfolioAnalytics": {
      "userXirr": 14.2,
      "avgXirrSameCategory": 11.8,
      "avgXirrAllUsers": 12.5,
      "strongContributors": ["HDFC Midcap", "Parag Parikh Flexi Cap"],
      "weakContributors": ["SBI Large Cap"]
    },
    "performanceChart": {
      "periods": ["1M", "3M", "6M", "1Y", "3Y", "5Y"],
      "values": [101000, 104000, 108000, 120000, 145000, 190000],
      "benchmark": [100000, 102500, 107000, 114000, 130000, 165000]
    }
  }
}
```

## 🚀 Deployment

### Production Setup
1. Set `NODE_ENV=production`
2. Configure MongoDB Atlas connection
3. Set up Supabase production credentials
4. Configure environment variables
5. Set up PM2 or similar process manager

### Docker Deployment
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
EXPOSE 3000
CMD ["npm", "start"]
```

## 🔮 Future Enhancements

- **Real-time NAV Updates**: WebSocket integration for live NAV
- **Advanced AI Analytics**: Integration with Gemini/OpenAI
- **PDF Generation**: Automated statement generation
- **Email Notifications**: Transaction and performance alerts
- **Mobile App API**: Optimized endpoints for mobile
- **Multi-currency Support**: International fund support
- **Tax Optimization**: Tax-loss harvesting suggestions

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Contact: milin@sipbrewery.com
- Documentation: [API Docs](https://docs.sipbrewery.com)

---

**Built with ❤️ for SIP Brewery** 