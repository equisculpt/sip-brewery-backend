# WhatsApp Chatbot System for SIPBrewery

A world-class, SEBI-compliant WhatsApp chatbot backend for mutual fund operations, built with Node.js, MongoDB, and AI integration.

## 🚀 Features

### Core Functionalities
- **User Onboarding**: Complete KYC-compliant user registration via WhatsApp
- **Portfolio Management**: View holdings, track performance, and get insights
- **SIP Operations**: Create, stop, and monitor SIP investments
- **AI-Powered Analysis**: Get intelligent fund analysis using Gemini AI
- **Rewards & Referrals**: Track loyalty points, cashback, and referral bonuses
- **Leaderboard**: View top performers and copy successful portfolios
- **Statement Generation**: Get tax P&L, portfolio summaries, and transaction reports

### Technical Features
- **Intent Recognition**: Advanced NLP-based message parsing
- **Session Management**: Persistent conversation state tracking
- **Rate Limiting**: Anti-abuse protection (3 messages/second per user)
- **SEBI Compliance**: Built-in disclaimers and regulatory safeguards
- **Multi-Provider Support**: Twilio integration with fallback simulation
- **Real-time Analytics**: Message tracking and performance monitoring

## 🛠 Tech Stack

- **Backend**: Node.js + Express
- **Database**: MongoDB (Mongoose)
- **Authentication**: Supabase JWT
- **WhatsApp**: Twilio API
- **AI**: Google Gemini API
- **Security**: Helmet, CORS, Rate Limiting
- **Logging**: Winston Logger

## 📁 Project Structure

```
src/
├── ai/
│   └── geminiClient.js          # AI integration with Gemini
├── whatsapp/
│   └── whatsappClient.js        # WhatsApp message handling
├── models/
│   ├── WhatsAppSession.js       # User session management
│   ├── SipOrder.js             # SIP order tracking
│   └── WhatsAppMessage.js      # Message logging
├── services/
│   └── whatsAppService.js      # Core business logic
├── controllers/
│   └── whatsAppController.js   # API endpoints
├── routes/
│   └── whatsapp.js             # Route definitions
├── utils/
│   └── parseMessage.js         # Intent recognition
└── middleware/
    ├── authenticateUser.js     # JWT authentication
    └── validation.js           # Request validation
```

## 🚀 Quick Start

### 1. Environment Setup

Create a `.env` file with the following variables:

```env
# MongoDB
MONGODB_URI=mongodb://localhost:27017/sipbrewery

# WhatsApp (Twilio)
TWILIO_ACCOUNT_SID=your_twilio_account_sid
TWILIO_AUTH_TOKEN=your_twilio_auth_token
TWILIO_PHONE_NUMBER=+1234567890

# AI (Gemini)
GEMINI_API_KEY=your_gemini_api_key

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Test Configuration
TEST_PHONE_NUMBER=+919876543210
```

### 2. Install Dependencies

```bash
npm install
```

### 3. Run the Application

```bash
# Development
npm run dev

# Production
npm start
```

### 4. Test the System

```bash
# Run comprehensive tests
node test-whatsapp-chatbot.js
```

## 📱 WhatsApp Integration

### Webhook Setup

1. **Twilio Configuration**:
   - Create a Twilio account
   - Set up WhatsApp Business API
   - Configure webhook URL: `https://your-domain.com/api/whatsapp/webhook`

2. **Webhook Endpoint**:
   ```
   POST /api/whatsapp/webhook
   ```

### Message Flow

1. **Incoming Message** → Webhook receives from Twilio
2. **Intent Detection** → Parse message using NLP patterns
3. **Session Management** → Track user state and context
4. **Business Logic** → Process based on intent
5. **AI Integration** → Generate responses using Gemini
6. **Response** → Send back via WhatsApp
7. **Logging** → Store for analytics and compliance

## 🤖 AI Integration

### Gemini AI Features

- **Fund Analysis**: Intelligent mutual fund insights
- **Response Generation**: Context-aware conversations
- **SEBI Compliance**: Automatic disclaimer injection
- **Fallback Mode**: Works without API key

### Example AI Prompts

```javascript
// Fund Analysis
const analysis = await geminiClient.analyzeFund('HDFC Flexicap');

// Response Generation
const response = await geminiClient.generateResponse('Hello, how are you?');
```

## 💬 Supported Intents

### User Interactions

| Intent | Example Message | Response |
|--------|----------------|----------|
| `GREETING` | "Hi", "Hello" | Welcome message |
| `ONBOARDING` | "My name is John" | Collect user data |
| `PORTFOLIO_VIEW` | "Show my portfolio" | Portfolio summary |
| `SIP_CREATE` | "Invest ₹5000 in HDFC Flexicap" | SIP confirmation |
| `AI_ANALYSIS` | "Analyse HDFC Flexicap" | Fund analysis |
| `REWARDS` | "My rewards" | Rewards summary |
| `REFERRAL` | "Refer a friend" | Referral link |
| `LEADERBOARD` | "Leaderboard" | Top performers |
| `HELP` | "Help" | Available commands |

### Advanced Features

- **Confirmation Handling**: Yes/No responses for actions
- **Context Awareness**: Remembers conversation state
- **Error Recovery**: Graceful handling of invalid inputs
- **Multi-language Support**: Extensible for different languages

## 📊 API Endpoints

### Public Endpoints

```javascript
// WhatsApp webhook (Twilio)
POST /api/whatsapp/webhook

// Health check
GET /api/whatsapp/health
```

### Admin Endpoints (Require Authentication)

```javascript
// Send test message
POST /api/whatsapp/admin/test-message
{
  "phoneNumber": "+919876543210",
  "message": "Test message"
}

// Get client status
GET /api/whatsapp/admin/client-status

// Test AI analysis
POST /api/whatsapp/admin/test-ai
{
  "fundName": "HDFC Flexicap"
}

// Get session statistics
GET /api/whatsapp/admin/stats

// Get recent sessions
GET /api/whatsapp/admin/sessions?limit=10

// Get session details
GET /api/whatsapp/admin/sessions/:phoneNumber

// Get message analytics
GET /api/whatsapp/admin/analytics?days=7
```

## 🔐 Security & Compliance

### SEBI Compliance

- **No Investment Advice**: All AI responses include disclaimers
- **KYC Verification**: Required before investment operations
- **Transaction Logging**: Complete audit trail for all actions
- **Data Protection**: Secure handling of sensitive information

### Security Features

- **Rate Limiting**: 3 messages/second per user
- **JWT Authentication**: Secure admin access
- **Input Validation**: Sanitize all user inputs
- **Error Handling**: No sensitive data in error messages

## 📈 Analytics & Monitoring

### Session Analytics

```javascript
// Get session statistics
{
  "sessions": {
    "total": 150,
    "completed": 120,
    "completionRate": "80.00"
  },
  "messages": {
    "total": 2500,
    "inbound": 1200,
    "outbound": 1300,
    "aiGenerated": 800
  }
}
```

### Message Analytics

```javascript
// Daily intent distribution
{
  "analytics": [
    {
      "_id": "2024-01-15",
      "intents": [
        { "intent": "GREETING", "count": 45 },
        { "intent": "PORTFOLIO_VIEW", "count": 32 }
      ],
      "totalCount": 77
    }
  ]
}
```

## 🧪 Testing

### Comprehensive Test Suite

```bash
# Run all tests
node test-whatsapp-chatbot.js
```

### Test Coverage

- ✅ Message Parsing (Intent Detection)
- ✅ WhatsApp Client (Message Sending)
- ✅ AI Integration (Gemini API)
- ✅ Session Management (State Tracking)
- ✅ Intent Handling (Business Logic)
- ✅ SIP Order Creation (Investment Flow)
- ✅ End-to-End Conversations (User Journey)
- ✅ Rate Limiting (Anti-abuse)
- ✅ Error Handling (Edge Cases)
- ✅ Performance (Response Times)

### Test Results Example

```
📊 Test Results Summary:
========================
✅ Message Parsing: 7/7 (100.0%)
✅ WhatsApp Client: 3/3 (100.0%)
✅ AI Integration: 3/3 (100.0%)
✅ Session Management: 3/3 (100.0%)
✅ Intent Handling: 3/3 (100.0%)
✅ SIP Order Creation: 2/2 (100.0%)
✅ End-to-End Conversation: 4/4 (100.0%)
✅ Rate Limiting: 1/1 (100.0%)
✅ Error Handling: 2/2 (100.0%)
✅ Performance: 1/1 (100.0%)

========================
Overall: 29/29 tests passed (100.0%)
🎉 Excellent! WhatsApp chatbot system is working well!
```

## 🚀 Deployment

### Production Setup

1. **Environment Variables**:
   ```bash
   NODE_ENV=production
   MONGODB_URI=mongodb+srv://...
   TWILIO_ACCOUNT_SID=...
   GEMINI_API_KEY=...
   ```

2. **Database Setup**:
   ```bash
   # Create indexes for performance
   db.whatsappsessions.createIndex({ "phoneNumber": 1 })
   db.whatsappmessages.createIndex({ "timestamp": -1 })
   ```

3. **Monitoring**:
   - Set up health checks
   - Configure error alerting
   - Monitor response times

### Scaling Considerations

- **Horizontal Scaling**: Stateless design supports multiple instances
- **Database**: Use MongoDB Atlas for managed scaling
- **Caching**: Redis for session data (optional)
- **Load Balancing**: Nginx or cloud load balancer

## 🔧 Configuration

### WhatsApp Provider

```javascript
// Environment variables
WHATSAPP_PROVIDER=TWILIO  // or 'SIMULATED' for development
TWILIO_ACCOUNT_SID=your_sid
TWILIO_AUTH_TOKEN=your_token
TWILIO_PHONE_NUMBER=+1234567890
```

### AI Configuration

```javascript
// Gemini AI settings
GEMINI_API_KEY=your_api_key
AI_FALLBACK_ENABLED=true
AI_RESPONSE_TIMEOUT=10000
```

### Rate Limiting

```javascript
// Rate limiting configuration
RATE_LIMIT_WINDOW=1000    // 1 second
RATE_LIMIT_MAX_MESSAGES=3 // 3 messages per second
```

## 📝 Logging

### Log Levels

- **INFO**: Normal operations
- **WARN**: Non-critical issues
- **ERROR**: Critical errors
- **DEBUG**: Detailed debugging (development only)

### Log Format

```javascript
{
  "timestamp": "2024-01-15T10:30:00.000Z",
  "level": "info",
  "message": "WhatsApp message processed",
  "phoneNumber": "+919876543210",
  "intent": "PORTFOLIO_VIEW",
  "processingTime": 245
}
```

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/new-feature`
3. **Make changes**: Follow coding standards
4. **Add tests**: Ensure test coverage
5. **Submit PR**: Include description and test results

### Code Standards

- **ESLint**: Follow project linting rules
- **JSDoc**: Document all functions
- **Error Handling**: Proper try-catch blocks
- **Logging**: Use structured logging

## 📞 Support

### Common Issues

1. **Webhook Not Receiving Messages**:
   - Check Twilio configuration
   - Verify webhook URL is accessible
   - Check server logs for errors

2. **AI Responses Not Working**:
   - Verify Gemini API key
   - Check API quota limits
   - Review fallback mode

3. **Rate Limiting Issues**:
   - Check rate limit configuration
   - Monitor user message frequency
   - Review rate limit logs

### Getting Help

- **Documentation**: Check this README
- **Issues**: Create GitHub issue
- **Logs**: Check application logs
- **Testing**: Run test suite for diagnostics

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Twilio**: WhatsApp Business API
- **Google**: Gemini AI API
- **MongoDB**: Database solution
- **Supabase**: Authentication service

---

**Built with ❤️ for SIPBrewery** 