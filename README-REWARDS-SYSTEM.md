# SEBI-Compliant Rewards System for SIPBrewery

## 🎯 Overview

A comprehensive, SEBI-compliant rewards system for mutual fund investors, built with Node.js, MongoDB, and Supabase Auth. The system provides multiple reward types while ensuring compliance with SEBI regulations and preventing abuse.

## 🏗️ Architecture

### Core Components

- **Models**: MongoDB schemas for rewards, referrals, and user tracking
- **Services**: Business logic for reward calculations and validation
- **Controllers**: API endpoints for user and admin operations
- **Middleware**: Authentication and authorization
- **Routes**: RESTful API structure

### Database Collections

1. **users** - Extended with referral fields
2. **rewards** - SEBI-compliant reward transactions
3. **reward_summaries** - User reward statistics
4. **referrals** - Referral relationships and bonuses

## 🪙 Reward Types

### 1. SIP Loyalty Points
- **Reward**: 1 point per successful SIP installment
- **Eligibility**: KYC verified users with confirmed SIP transactions
- **Compliance**: Post-transaction based, requires BSE confirmation

### 2. Cashback for 12 SIPs
- **Reward**: ₹500 cashback after completing 12 SIPs in one fund
- **Eligibility**: 12 successful SIP installments in the same fund
- **Compliance**: No upfront bonuses, achievement-based

### 3. Referral Bonus
- **Reward**: ₹100 per successful referral
- **Eligibility**: Referred user must complete KYC and start SIP
- **Compliance**: Post-verification, anti-abuse measures

## ⚖️ SEBI Compliance Features

### ✅ Compliance Measures
- **No upfront bonuses** - All rewards are post-transaction
- **KYC verification required** - Only verified users eligible
- **BSE confirmation tracking** - Links to actual transactions
- **Detailed audit trail** - Complete transaction history
- **Anti-abuse protection** - Prevents gaming the system

### 🚫 Anti-Abuse Measures
- **Self-referral prevention** - Users cannot refer themselves
- **Referral limits** - Maximum 50 referrals per year
- **Duplicate prevention** - No duplicate rewards for same transaction
- **SIP cancellation tracking** - Bonuses revoked if SIP cancelled within 3 months
- **IP and device tracking** - Fraud detection capabilities

## 🔧 API Endpoints

### User Endpoints (Require Authentication)

#### Get Reward Summary
```http
GET /api/rewards/summary
Authorization: Bearer <supabase-jwt-token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalPoints": 15,
    "totalCashback": 500,
    "totalReferralBonus": 200,
    "totalSipInstallments": 15,
    "pendingPayout": 700,
    "totalPaidOut": 0,
    "recentTransactions": [...],
    "lastUpdated": "2024-01-15T10:30:00Z"
  }
}
```

#### Get Reward Transactions
```http
GET /api/rewards/transactions?page=1&limit=20&type=SIP_LOYALTY_POINTS
Authorization: Bearer <supabase-jwt-token>
```

#### Get Referral Code
```http
GET /api/rewards/referral-code
Authorization: Bearer <supabase-jwt-token>
```

#### Get Referral Leaderboard
```http
GET /api/rewards/leaderboard?limit=10
Authorization: Bearer <supabase-jwt-token>
```

#### Simulate SIP Reward (Testing)
```http
POST /api/rewards/simulate-sip-reward
Authorization: Bearer <supabase-jwt-token>
Content-Type: application/json

{
  "sipId": "SIP_001",
  "fundName": "HDFC Mid-Cap Opportunities Fund",
  "folioNumber": "HDFC123456"
}
```

### Internal Endpoints (Called by Transaction System)

#### Award SIP Points
```http
POST /api/rewards/award-sip-points
Content-Type: application/json

{
  "userId": "user-supabase-id",
  "sipId": "SIP_001",
  "fundName": "HDFC Mid-Cap Opportunities Fund",
  "folioNumber": "HDFC123456",
  "bseConfirmationId": "BSE_CONFIRM_001"
}
```

#### Validate Referral
```http
POST /api/rewards/validate-referral
Content-Type: application/json

{
  "referredId": "referred-user-supabase-id"
}
```

#### Revoke Referral Bonus
```http
POST /api/rewards/revoke-referral
Content-Type: application/json

{
  "referredId": "referred-user-supabase-id"
}
```

### Admin Endpoints (Require Admin Authentication)

#### Mark Reward as Paid
```http
POST /api/rewards/admin/mark-paid
Authorization: Bearer <admin-jwt-token>
Content-Type: application/json

{
  "rewardId": "reward-mongodb-id"
}
```

#### Get Unpaid Rewards
```http
GET /api/rewards/admin/unpaid
Authorization: Bearer <admin-jwt-token>
```

#### Get User Reward History
```http
GET /api/rewards/admin/user/:userId
Authorization: Bearer <admin-jwt-token>
```

#### Export Unpaid Rewards CSV
```http
GET /api/rewards/admin/export-csv
Authorization: Bearer <admin-jwt-token>
```

## 🗄️ Database Schema

### User Model (Extended)
```javascript
{
  supabaseId: String,           // Supabase user ID
  referralCode: String,         // Unique referral code (REF + 6 chars)
  referredBy: String,           // Referrer's supabaseId
  referralCount: Number,        // Total successful referrals
  totalReferralBonus: Number,   // Total referral bonus earned
  kycStatus: String,            // PENDING, VERIFIED, REJECTED
  // ... other existing fields
}
```

### Reward Model
```javascript
{
  userId: String,               // User's supabaseId
  type: String,                 // REFERRAL_BONUS, SIP_LOYALTY_POINTS, CASHBACK_12_SIPS
  amount: Number,               // Cash amount (0 for points)
  points: Number,               // Loyalty points (0 for cash)
  description: String,          // Human-readable description
  status: String,               // PENDING, CREDITED, REDEEMED, EXPIRED, REVOKED
  isPaid: Boolean,              // Payout status
  paidAt: Date,                 // Payout timestamp
  paidBy: String,               // Admin who marked as paid
  sipId: String,                // Related SIP transaction ID
  referralId: String,           // Related referral ID
  fundName: String,             // Mutual fund name
  folioNumber: String,          // Folio number
  bseConfirmationId: String,    // BSE confirmation ID
  transactionTimestamp: Date,   // SEBI compliance timestamp
  // ... audit fields
}
```

### RewardSummary Model
```javascript
{
  userId: String,               // User's supabaseId
  totalPoints: Number,          // Total loyalty points
  totalCashback: Number,        // Total cashback earned
  totalReferralBonus: Number,   // Total referral bonus
  totalSipInstallments: Number, // Total SIP installments
  pendingPayout: Number,        // Amount pending payout
  totalPaidOut: Number,         // Amount already paid
  lastUpdated: Date,            // Last summary update
  // ... statistics fields
}
```

### Referral Model
```javascript
{
  referrerId: String,           // Referrer's supabaseId
  referredId: String,           // Referred user's supabaseId
  referralCode: String,         // Referral code used
  status: String,               // PENDING, KYC_COMPLETED, SIP_STARTED, BONUS_PAID
  bonusAmount: Number,          // Bonus amount (default: 100)
  bonusPaid: Boolean,           // Whether bonus was paid
  kycCompletedAt: Date,         // KYC completion timestamp
  sipStartedAt: Date,           // First SIP timestamp
  sipCancelledAt: Date,         // SIP cancellation timestamp (if applicable)
  // ... anti-abuse fields
}
```

## 🔐 Security & Authentication

### User Authentication
- **Supabase JWT** verification for all user endpoints
- **KYC verification** required for reward access
- **Active account** check before operations

### Admin Authentication
- **JWT-based** admin authentication
- **Role-based** access control
- **Audit logging** for all admin actions

### Anti-Abuse Protection
- **IP tracking** for referral validation
- **Device fingerprinting** for fraud detection
- **Rate limiting** on sensitive operations
- **Referral limits** (50 per year per user)

## 🚀 Installation & Setup

### Prerequisites
- Node.js 16+
- MongoDB 5+
- Supabase project with Auth enabled

### Environment Variables
```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017/sip-brewery

# Supabase
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# JWT (for admin)
JWT_SECRET=your-jwt-secret

# Optional
NODE_ENV=production
ENABLE_CRON=true
```

### Installation
```bash
# Install dependencies
npm install

# Install Supabase client
npm install @supabase/supabase-js

# Start the server
npm start

# Run tests
node test-rewards-simple.js
```

## 🧪 Testing

### Simple Logic Test
```bash
node test-rewards-simple.js
```

### Full System Test (Requires MongoDB)
```bash
node test-rewards-system.js
```

### Test Coverage
- ✅ SIP loyalty points logic
- ✅ Cashback calculation (12 SIPs)
- ✅ Referral bonus validation
- ✅ Anti-abuse measures
- ✅ Reward summary generation
- ✅ Admin functions

## 📊 Business Logic

### SIP Loyalty Points
1. Verify user KYC status
2. Check for duplicate SIP reward
3. Award 1 point per successful SIP
4. Update reward summary
5. Check for cashback eligibility

### Cashback for 12 SIPs
1. Count SIP installments for specific fund
2. Award ₹500 when count reaches 12
3. Prevent duplicate cashback
4. Update summary and pending payout

### Referral Bonus
1. Validate referral relationship
2. Check anti-abuse measures
3. Verify referred user KYC and SIP
4. Award ₹100 to referrer
5. Create referral record
6. Update statistics

## 🔄 Integration Points

### Transaction System Integration
- **SIP confirmation** triggers loyalty points
- **BSE confirmation** required for compliance
- **Fund and folio** tracking for cashback

### Admin Dashboard Integration
- **Unpaid rewards** export for manual processing
- **User reward history** for support
- **Referral leaderboard** for analytics

### Frontend Integration
- **Real-time updates** via WebSocket (future)
- **Push notifications** for new rewards
- **Gamification** elements

## 📈 Analytics & Reporting

### User Analytics
- Reward earning patterns
- Referral effectiveness
- SIP completion rates

### Business Analytics
- Total rewards distributed
- Referral conversion rates
- Cost per acquisition

### Compliance Reporting
- SEBI audit trails
- Transaction verification
- Anti-abuse statistics

## 🔮 Future Enhancements

### Planned Features
- **WebSocket** real-time updates
- **Push notifications** for rewards
- **Gamification** leaderboards
- **Advanced analytics** dashboard
- **Automated payouts** integration

### Scalability Considerations
- **Redis caching** for performance
- **Database indexing** optimization
- **Microservices** architecture
- **Event-driven** processing

## 📝 Compliance Documentation

### SEBI Guidelines Adherence
- ✅ No upfront bonuses or signup rewards
- ✅ Post-transaction based rewards only
- ✅ KYC verification mandatory
- ✅ Complete audit trail maintained
- ✅ Anti-abuse measures implemented

### Audit Requirements
- All reward transactions logged
- BSE confirmation IDs tracked
- User verification status recorded
- Admin actions audited
- Export capabilities for compliance

## 🆘 Support & Troubleshooting

### Common Issues
1. **MongoDB connection** - Check connection string
2. **Supabase auth** - Verify JWT token
3. **KYC verification** - Ensure user status is VERIFIED
4. **Duplicate rewards** - Check SIP ID uniqueness

### Debug Mode
```bash
NODE_ENV=development DEBUG=rewards:* npm start
```

### Logs
- All operations logged to `logs/combined.log`
- Errors logged to `logs/error.log`
- Audit trail in MongoDB

## 📄 License

This project is proprietary to SIPBrewery. All rights reserved.

---

**Built with ❤️ for SEBI-compliant mutual fund rewards** 