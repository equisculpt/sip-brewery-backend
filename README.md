# SIP Brewery Backend

A comprehensive backend API for mutual fund analysis and benchmarking with AI-powered insights.

## 🚀 Features

- **Mutual Fund Analysis**: Comprehensive analysis of mutual funds using NAV data
- **AI-Powered Insights**: Integration with Google Gemini AI for intelligent fund recommendations
- **Benchmark Comparison**: Compare mutual funds against NIFTY 50 and other benchmarks
- **Real-time Data**: Fetch live NAV data from mutual fund APIs
- **Performance Metrics**: Calculate CAGR, XIRR, volatility, drawdown, and more
- **Chart Generation**: Generate comparison charts for fund vs benchmark analysis

## 🛠️ Recent Fixes

The following issues have been resolved:

1. **Express Version Compatibility**: Downgraded from Express 5.1.0 to 4.18.2 to fix path-to-regexp compatibility issues
2. **Missing Dependencies**: Added missing packages (helmet, compression, express-rate-limit)
3. **Route Ordering**: Fixed benchmark routes ordering to prevent parameter conflicts
4. **Missing Middleware**: Created errorHandler middleware
5. **Missing Functions**: Added fetchNAVData function to aiService
6. **File Structure**: Consolidated Express setup in src/app.js for better organization

## 📋 Prerequisites

- Node.js (v16 or higher)
- MongoDB (local or cloud)
- Google Gemini API key (for AI features)

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd sip-brewery-backend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   # Server Configuration
   PORT=3000
   NODE_ENV=development

   # AI Configuration
   GEMINI_API_KEY=your_gemini_api_key_here

   # Database Configuration
   MONGODB_URI=mongodb+srv://your_connection_string

   # API Configuration
   API_VERSION=v1
   API_PREFIX=/api
   ```

4. **Start the server**
   ```bash
   npm start
   # or for development
   npm run dev
   ```

5. **Test the API**
   ```bash
   node test-api.js
   ```

## 📡 API Endpoints

### Core Endpoints
- `GET /` - Welcome message and API information
- `GET /health` - Health check endpoint

### AI Analysis
- `GET /api/ai/health` - AI service health status
- `POST /api/ai/analyze` - Analyze mutual funds with AI
- `GET /api/ai/test/:schemeCode` - Test mutual fund data fetching

### Benchmark
- `GET /api/benchmark/:indexId` - Get benchmark data
- `GET /api/benchmark/compare/:fundId` - Compare fund with benchmark
- `GET /api/benchmark/insights/:fundId` - Generate AI insights
- `POST /api/benchmark/update-nifty` - Update NIFTY 50 data

## 🔧 Usage Examples

### Analyze Mutual Funds
```bash
curl -X POST http://localhost:3000/api/ai/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "schemeCodes": ["120010", "120011"],
    "query": "Compare these funds and recommend the better one"
  }'
```

### Get Benchmark Data
```bash
curl http://localhost:3000/api/benchmark/NIFTY50
```

### Update NIFTY 50 Data
```bash
curl -X POST http://localhost:3000/api/benchmark/update-nifty
```

## 📊 Data Sources

- **Mutual Fund NAV**: [MF API](https://api.mfapi.in/)
- **NIFTY 50 Data**: [NIFTY Indices](https://www.niftyindices.com/)
- **AI Analysis**: Google Gemini API

## 🏗️ Project Structure

```
sip-brewery-backend/
├── src/
│   ├── app.js                 # Main Express application
│   ├── config/
│   │   └── database.js        # Database configuration
│   ├── controllers/
│   │   ├── aiController.js    # AI analysis controller
│   │   └── benchmarkController.js # Benchmark controller
│   ├── middleware/
│   │   ├── errorHandler.js    # Error handling middleware
│   │   └── validation.js      # Request validation
│   ├── models/
│   │   └── BenchmarkIndex.js  # MongoDB schema
│   ├── routes/
│   │   ├── ai.js             # AI routes
│   │   └── benchmarkRoutes.js # Benchmark routes
│   ├── services/
│   │   ├── aiService.js      # AI analysis service
│   │   └── benchmarkService.js # Benchmark service
│   └── utils/
│       ├── logger.js         # Winston logger
│       ├── niftyScraper.js   # NIFTY data scraper
│       └── response.js       # Response utilities
├── index.js                  # Application entry point
├── package.json
└── README.md
```

## 🔍 Testing

Run the comprehensive API test:
```bash
node test-api.js
```

This will test all major endpoints and provide a status report.

## 🐛 Troubleshooting

### Common Issues

1. **MongoDB Connection Error**
   - Ensure MongoDB is running
   - Check MONGODB_URI in .env file
   - Verify network connectivity

2. **AI Analysis Fails**
   - Set GEMINI_API_KEY in .env file
   - Verify API key is valid
   - Check internet connectivity

3. **NIFTY Data Scraping Fails**
   - The scraper may fail due to website changes
   - Check logs for specific error messages
   - Consider using alternative data sources

4. **Port Already in Use**
   - Change PORT in .env file
   - Kill existing process on port 3000

## 📝 Logs

Logs are stored in the `logs/` directory:
- `combined.log` - All logs
- `error.log` - Error logs only

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the ISC License.

## 🆘 Support

For issues and questions:
1. Check the logs in `logs/` directory
2. Review the troubleshooting section
3. Create an issue with detailed error information 