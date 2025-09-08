# 🏆 SIP Brewery Leaderboard System

A complete, production-grade leaderboard and portfolio copying system for mutual fund investors. Features anonymous user display, XIRR-based rankings, and automated portfolio copying functionality.

## 🚀 Features

### Core Features
- **🏆 Multi-duration Leaderboards** (1M, 3M, 6M, 1Y, 3Y)
- **🔐 Anonymous User Display** using secret codes (e.g., `SBX2-91U`)
- **📊 Portfolio Allocation-based Rankings** (never shows ₹ amounts)
- **🧠 Copy Portfolio + Copy SIP** functionality
- **🔄 Daily Auto-update System** with MongoDB
- **📈 XIRR Calculation** for accurate return measurement

### Security & Privacy
- **Anonymous Display**: Users identified only by secret codes
- **No Amount Exposure**: Only allocation percentages shown
- **Audit Trail**: Complete copy history tracking
- **Authentication Required**: All copy operations require login

### Automation
- **Daily XIRR Updates** (8:00 AM IST)
- **Daily Leaderboard Generation** (8:30 AM IST)
- **Weekly Cleanup** (Sunday 9:00 AM IST)
- **Real-time Statistics** calculation

## 🏗️ Architecture

### Database Models

#### User Model
```javascript
{
  _id: ObjectId,
  supabaseId: String,
  email: String,
  phone: String,
  secretCode: String, // SBX2-91U format
  name: String,
  kycStatus: String,
  preferences: Object,
  isActive: Boolean
}
```

#### UserPortfolio Model
```javascript
{
  userId: ObjectId,
  funds: [{
    schemeCode: String,
    schemeName: String,
    investedValue: Number,
    currentValue: Number,
    units: Number,
    startDate: Date,
    lastNav: Number,
    lastNavDate: Date
  }],
  xirr1M: Number,
  xirr3M: Number,
  xirr6M: Number,
  xirr1Y: Number,
  xirr3Y: Number,
  totalInvested: Number,
  totalCurrentValue: Number,
  allocation: Map,
  transactions: Array,
  leaderboardHistory: Array
}
```

#### Leaderboard Model
```javascript
{
  duration: String, // 1M, 3M, 6M, 1Y, 3Y
  leaders: [{
    secretCode: String,
    returnPercent: Number,
    allocation: Map,
    rank: Number,
    userId: ObjectId,
    portfolioId: ObjectId
  }],
  generatedAt: Date,
  totalParticipants: Number,
  averageReturn: Number,
  medianReturn: Number
}
```

#### PortfolioCopy Model
```javascript
{
  sourceSecretCode: String,
  sourceUserId: ObjectId,
  targetUserId: ObjectId,
  investmentType: String, // SIP or LUMPSUM
  averageSip: Number,
  copiedAllocation: Map,
  sourceReturnPercent: Number,
  duration: String,
  status: String,
  executionDetails: Object,
  metadata: Object
}
```

## 📡 API Endpoints

### Public Endpoints

#### Get Leaderboard
```http
GET /api/leaderboard/:duration
```
**Response:**
```json
{
  "success": true,
  "data": {
    "duration": "3M",
    "leaders": [
      {
        "secretCode": "SBX2-91U",
        "returnPercent": 17.4,
        "rank": 1,
        "allocation": {
          "Parag Parikh Flexicap": 55,
          "Quant Tax Saver": 45
        }
      }
    ],
    "generatedAt": "2024-01-15T08:30:00Z",
    "totalParticipants": 1250,
    "averageReturn": 12.3,
    "medianReturn": 11.8
  }
}
```

#### Get All Leaderboards
```http
GET /api/leaderboard
```

#### Get Leaderboard Statistics
```http
GET /api/leaderboard/:duration/stats
```

### Protected Endpoints (Authentication Required)

#### Copy Portfolio
```http
POST /api/leaderboard/portfolio/copy
Content-Type: application/json
Authorization: Bearer <jwt_token>

{
  "sourceSecretCode": "SBX2-91U",
  "investmentType": "SIP",
  "averageSip": 5000
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "copyId": "507f1f77bcf86cd799439011",
    "message": "Portfolio copied successfully. Your SIP setup has been created based on the leader's allocation.",
    "allocation": {
      "Parag Parikh Flexicap": 55,
      "Quant Tax Saver": 45
    }
  }
}
```

#### Get User Leaderboard History
```http
GET /api/leaderboard/user/history
Authorization: Bearer <jwt_token>
```

#### Get Portfolio Copy History
```http
GET /api/leaderboard/portfolio/copy/history
Authorization: Bearer <jwt_token>
```

#### Get User Rank
```http
GET /api/leaderboard/:duration/user/rank
Authorization: Bearer <jwt_token>
```

### Admin Endpoints

#### Generate Leaderboards
```http
POST /api/leaderboard/generate
Authorization: Bearer <jwt_token>
```

#### Update XIRR
```http
POST /api/leaderboard/update-xirr
Authorization: Bearer <jwt_token>
```

## 🧮 XIRR Calculation

The system uses a custom XIRR implementation based on the Newton-Raphson method:

```javascript
calculateXIRR(cashFlows, guess = 0.1) {
  const tolerance = 0.0001;
  const maxIterations = 100;
  let rate = guess;
  
  for (let i = 0; i < maxIterations; i++) {
    const npv = this.calculateNPV(cashFlows, rate);
    const derivative = this.calculateNPVDerivative(cashFlows, rate);
    
    if (Math.abs(derivative) < tolerance) break;
    
    const newRate = rate - npv / derivative;
    if (Math.abs(newRate - rate) < tolerance) {
      rate = newRate;
      break;
    }
    rate = newRate;
  }
  
  return (Math.pow(1 + rate, 12) - 1) * 100;
}
```

## ⏰ Cron Jobs

### Daily Jobs
- **8:00 AM IST**: Update XIRR for all portfolios
- **8:30 AM IST**: Generate leaderboards for all durations

### Weekly Jobs
- **Sunday 9:00 AM IST**: Clean up old leaderboard data

### Development Jobs
- **Every 10 minutes**: Test job for development

## 🔧 Installation & Setup

### 1. Install Dependencies
```bash
npm install node-cron
```

### 2. Environment Variables
```env
# Add to your .env file
ENABLE_CRON=true
NODE_ENV=production
```

### 3. Database Setup
The system automatically creates necessary indexes:
- User: `secretCode`, `email`, `phone`
- UserPortfolio: `userId`, `xirr1M`, `xirr3M`, etc.
- Leaderboard: `duration`, `generatedAt`
- PortfolioCopy: `sourceSecretCode`, `targetUserId`

### 4. Initialize System
```javascript
// In your app.js
const leaderboardCronService = require('./services/leaderboardCronService');
leaderboardCronService.init();
```

## 🧪 Testing

### Run Complete Test Suite
```bash
node test-leaderboard.js
```

### Test Individual Components
```javascript
const { 
  testXIRRCalculation,
  testLeaderboardGeneration,
  testPortfolioCopying,
  testUserHistory,
  testCronService,
  testDatabaseOperations 
} = require('./test-leaderboard');

// Test XIRR calculation
await testXIRRCalculation();

// Test leaderboard generation
await testLeaderboardGeneration();

// Test portfolio copying
await testPortfolioCopying();
```

### Seed Test Data
```javascript
const { seedLeaderboardData } = require('./src/utils/leaderboardSeeder');
await seedLeaderboardData();
```

## 📊 Sample Data

### Sample Leaderboard Response
```json
{
  "duration": "1Y",
  "leaders": [
    {
      "secretCode": "SBX2-91U",
      "returnPercent": 18.7,
      "rank": 1,
      "allocation": {
        "Parag Parikh Flexicap": 40,
        "SBI Smallcap": 35,
        "Quant Tax Saver": 25
      }
    },
    {
      "secretCode": "SBY3-MN1",
      "returnPercent": 17.2,
      "rank": 2,
      "allocation": {
        "HDFC Flexicap": 50,
        "Mirae Asset Emerging Bluechip": 30,
        "Axis Bluechip": 20
      }
    }
  ],
  "totalParticipants": 1250,
  "averageReturn": 12.3,
  "medianReturn": 11.8
}
```

### Sample Portfolio Copy Response
```json
{
  "success": true,
  "copyId": "507f1f77bcf86cd799439011",
  "message": "Portfolio copied successfully. Your SIP setup has been created based on the leader's allocation.",
  "allocation": {
    "Parag Parikh Flexicap": 40,
    "SBI Smallcap": 35,
    "Quant Tax Saver": 25
  }
}
```

## 🔒 Security Features

### Anonymous Display
- Users identified only by secret codes (e.g., `SBX2-91U`)
- No personal information exposed in leaderboards
- Secret codes generated automatically on user creation

### Data Protection
- No actual investment amounts shown
- Only allocation percentages displayed
- Complete audit trail for all copy operations

### Authentication
- All copy operations require valid JWT token
- User verification before portfolio copying
- Rate limiting on copy operations

## 🚀 Production Deployment

### 1. Database Optimization
```javascript
// Ensure indexes are created
db.users.createIndex({ "secretCode": 1 });
db.userportfolios.createIndex({ "xirr1Y": -1 });
db.leaderboards.createIndex({ "duration": 1, "generatedAt": -1 });
```

### 2. Cron Job Management
```javascript
// Monitor cron job status
const status = leaderboardCronService.getJobStatus();
console.log('Cron jobs:', status);

// Manual trigger if needed
await leaderboardCronService.triggerXIRRUpdate();
await leaderboardCronService.triggerLeaderboardGeneration();
```

### 3. Performance Monitoring
- Monitor XIRR calculation performance
- Track leaderboard generation time
- Monitor database query performance
- Set up alerts for failed cron jobs

### 4. Scaling Considerations
- Use Redis for caching leaderboard data
- Implement database sharding for large datasets
- Use message queues for async processing
- Consider CDN for static leaderboard data

## 🔮 Future Enhancements

### Planned Features
- **Real-time Updates**: WebSocket-based live leaderboard updates
- **Advanced Analytics**: Risk-adjusted returns, Sharpe ratio
- **Social Features**: Follow top performers, share achievements
- **Mobile App**: Native mobile leaderboard experience
- **AI Insights**: Portfolio recommendations based on leader analysis

### Integration Opportunities
- **BSE Star MF API**: Real mutual fund execution
- **Notification Service**: Copy alerts and updates
- **Analytics Platform**: Advanced performance tracking
- **Payment Gateway**: Direct investment execution

## 📞 Support

For questions or issues:
1. Check the test suite: `node test-leaderboard.js`
2. Review the API documentation above
3. Check server logs for detailed error messages
4. Verify database connectivity and indexes

## 📄 License

This leaderboard system is part of the SIP Brewery backend and follows the same licensing terms.

---

**🏆 Ready for Production**: The leaderboard system is fully implemented, tested, and ready to power a world-class mutual fund investment platform with anonymous leaderboards and portfolio copying functionality. 