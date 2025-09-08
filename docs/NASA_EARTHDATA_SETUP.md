# 🛰️ NASA EARTHDATA SETUP GUIDE

## ✅ **You're Already Set Up!**

I can see you've created your NASA Earthdata account (`equisculpt`) and are on the user tokens page. Here's how to complete the setup:

## 🔑 **Step 1: Generate Your Token**

### **On the User Tokens Page:**
```
https://urs.earthdata.nasa.gov/users/equisculpt/user_tokens
```

1. **Click "Generate Token"**
2. **Give it a name**: `ASI-Supply-Chain-Intelligence`
3. **Set expiration**: Choose "Never expires" or 1 year
4. **Copy the token** - it will look like: `EDL-u-equisculpt-1234567890abcdef...`

### **⚠️ Important: Save Your Token Immediately**
- You can only see the full token once
- Copy it to a secure location
- Don't share it publicly

## 🔧 **Step 2: Configure Your Environment**

### **Add to .env file:**
```bash
# NASA Earthdata credentials
EARTHDATA_USERNAME=equisculpt
EARTHDATA_TOKEN=EDL-u-equisculpt-your-actual-token-here

# Alternative: You can also use password instead of token
EARTHDATA_PASSWORD=your_password_if_preferred
```

### **Update your nasa-earthdata-client.js:**
```javascript
class NASAEarthdataClient extends EventEmitter {
  constructor(options = {}) {
    super();
    
    this.config = {
      baseUrl: 'https://cmr.earthdata.nasa.gov',
      searchUrl: 'https://cmr.earthdata.nasa.gov/search',
      
      // Use token authentication (preferred)
      username: process.env.EARTHDATA_USERNAME || 'equisculpt',
      token: process.env.EARTHDATA_TOKEN,
      
      // Fallback to password if no token
      password: process.env.EARTHDATA_PASSWORD,
      
      ...options
    };
  }

  async makeRequest(url, options = {}) {
    const requestOptions = {
      method: 'GET',
      timeout: this.config.timeout,
      ...options
    };
    
    // Use token authentication (preferred method)
    if (this.config.token) {
      requestOptions.headers = {
        'Authorization': `Bearer ${this.config.token}`,
        ...requestOptions.headers
      };
    }
    // Fallback to basic auth with username/password
    else if (this.config.username && this.config.password) {
      const auth = Buffer.from(`${this.config.username}:${this.config.password}`).toString('base64');
      requestOptions.headers = {
        'Authorization': `Basic ${auth}`,
        ...requestOptions.headers
      };
    }
    
    // Rest of the method...
  }
}
```

## 🚀 **Step 3: Test Your Setup**

### **Quick Test Script:**
```javascript
// test-nasa-earthdata.js
const NASAEarthdataClient = require('./src/finance_crawler/nasa-earthdata-client');

async function testNASAConnection() {
  try {
    console.log('🛰️ Testing NASA Earthdata connection...');
    
    const client = new NASAEarthdataClient({
      username: 'equisculpt',
      token: process.env.EARTHDATA_TOKEN
    });
    
    await client.initialize();
    
    // Test search for Mumbai port
    console.log('🔍 Searching for satellite data over Mumbai port...');
    const mumbaiData = await client.searchPortActivity('Mumbai', 7);
    
    console.log('✅ Success! Found data:');
    console.log(`- Optical granules: ${mumbaiData.opticalGranules.length}`);
    console.log(`- Time range: ${mumbaiData.timeRange.start} to ${mumbaiData.timeRange.end}`);
    console.log(`- Analysis: ${mumbaiData.analysis.dataAvailability}`);
    
    return true;
    
  } catch (error) {
    console.error('❌ NASA Earthdata test failed:', error.message);
    return false;
  }
}

// Run the test
testNASAConnection()
  .then(success => {
    if (success) {
      console.log('🎉 NASA Earthdata integration is working!');
    } else {
      console.log('🔧 Please check your credentials and try again.');
    }
  });
```

### **Run the test:**
```bash
# Set your token first
export EARTHDATA_TOKEN="your-actual-token-here"

# Run the test
node test-nasa-earthdata.js
```

## 📊 **Step 4: Integration with Supply Chain System**

### **Update your supply-chain-main.js:**
```javascript
// Add NASA Earthdata to your supply chain system
const NASAEarthdataClient = require('./nasa-earthdata-client');

class SupplyChainMain {
  async initialize() {
    try {
      // ... existing initialization code ...
      
      // Initialize NASA Earthdata client
      console.log('🛰️ Initializing NASA Earthdata integration...');
      this.nasaClient = new NASAEarthdataClient({
        username: 'equisculpt',
        token: process.env.EARTHDATA_TOKEN
      });
      
      await this.nasaClient.initialize();
      
      // Add satellite data monitoring
      this.startSatelliteMonitoring();
      
      console.log('✅ NASA Earthdata integration active');
      
    } catch (error) {
      console.error('❌ Failed to initialize NASA Earthdata:', error);
    }
  }

  async startSatelliteMonitoring() {
    // Monitor major Indian ports daily
    const ports = ['Mumbai', 'Chennai', 'Kolkata', 'Visakhapatnam'];
    
    for (const port of ports) {
      try {
        const portData = await this.nasaClient.searchPortActivity(port, 7);
        
        // Integrate with supply chain intelligence
        this.system.handleSupplyChainEvent('satellite_port_data', {
          port,
          dataAvailability: portData.analysis.dataAvailability,
          granulesFound: portData.opticalGranules.length,
          lastUpdate: new Date().toISOString()
        });
        
        console.log(`📡 ${port} port: ${portData.opticalGranules.length} satellite images available`);
        
      } catch (error) {
        console.error(`❌ Failed to get satellite data for ${port}:`, error.message);
      }
    }
    
    // Schedule regular updates
    setInterval(() => this.startSatelliteMonitoring(), 24 * 60 * 60 * 1000); // Daily
  }
}
```

## 🎯 **Available Data for Your Supply Chain Intelligence**

### **What You Can Monitor Now:**

#### **🚢 Port Activity (Daily Updates)**
```javascript
// Mumbai Port - Ship traffic and congestion
const mumbaiPort = await nasaClient.searchPortActivity('Mumbai', 7);

// Chennai Port - Container throughput
const chennaiPort = await nasaClient.searchPortActivity('Chennai', 7);
```

#### **🏭 Industrial Activity (Weekly Updates)**
```javascript
// Mumbai-Pune Industrial Belt - Factory utilization
const industrialData = await nasaClient.searchIndustrialActivity('Mumbai-Pune', 30);

// Thermal signatures indicate production levels
```

#### **🌾 Agricultural Monitoring (Bi-weekly Updates)**
```javascript
// Punjab Wheat Belt - Crop health assessment
const cropData = await nasaClient.searchAgriculturalActivity('Punjab', 60);

// NDVI data for yield prediction
```

#### **🔥 Environmental Risks (Real-time)**
```javascript
// National fire monitoring
const riskData = await nasaClient.searchEnvironmentalRisks('national', 7);

// Early warning for supply chain disruptions
```

## 📈 **Investment Intelligence Integration**

### **Satellite Data → Investment Signals:**
```javascript
// Example: Port congestion affects logistics stocks
if (mumbaiPort.analysis.dataAvailability === 'good' && 
    mumbaiPort.opticalGranules.length > 10) {
  
  // High data availability = good port operations
  // → Positive signal for logistics companies
  this.generateInvestmentSignal({
    type: 'logistics_positive',
    source: 'satellite_data',
    confidence: 'high',
    affectedStocks: ['CONCOR', 'GATI', 'MAHLOG'],
    reasoning: 'Satellite data shows normal port operations'
  });
}
```

## 🔒 **Security Best Practices**

### **Token Management:**
```bash
# Store in environment variables, not in code
export EARTHDATA_TOKEN="your-token"

# Or in .env file (add to .gitignore)
echo "EARTHDATA_TOKEN=your-token" >> .env
echo ".env" >> .gitignore
```

### **Token Rotation:**
- Regenerate tokens every 6-12 months
- Monitor token usage on NASA dashboard
- Use different tokens for dev/prod environments

## 🚀 **Next Steps**

1. **Generate your token** on the NASA page you're already on
2. **Add it to your .env file**
3. **Run the test script** to verify connection
4. **Start the supply chain system** with satellite integration

```bash
# Start your enhanced supply chain system
node src/finance_crawler/supply-chain-main.js start
```

## 📞 **Support**

### **If you encounter issues:**
- **Token problems**: Regenerate on NASA URS page
- **API errors**: Check NASA Earthdata status page
- **Data access**: Verify your account has data access permissions

### **NASA Resources:**
- **Documentation**: https://earthdata.nasa.gov/learn/use-data
- **API Guide**: https://cmr.earthdata.nasa.gov/search/site/docs/search/api.html
- **Support Forum**: https://forum.earthdata.nasa.gov/

---

**You're almost there! Just generate the token and you'll have real-time satellite intelligence for your supply chain system.**
