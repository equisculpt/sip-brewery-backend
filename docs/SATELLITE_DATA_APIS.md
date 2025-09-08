# 🛰️ SATELLITE DATA APIs - COMPLETE REFERENCE

## ✅ **NASA APIs (FREE)**

### **1. NASA FIRMS (Fire Information for Resource Management System)**
```javascript
// COMPLETELY FREE - No API key required for basic access
const firmsAPI = 'https://firms.modaps.eosdis.nasa.gov/api/country/csv/';

// Get fire data for India (last 24 hours)
const indiaFires = `${firmsAPI}VIIRS_SNPP_NRT/IND/1`;

// Example request
fetch(indiaFires)
  .then(response => response.text())
  .then(data => {
    // CSV data with fire locations, confidence, brightness
    console.log('Fire data:', data);
  });
```

### **2. NASA Giovanni API**
```javascript
// Atmospheric data API
const giovanniAPI = 'https://giovanni.gsfc.nasa.gov/giovanni/';

// Example: Get atmospheric data
const atmosphericRequest = {
  service: 'ArAvTs',
  version: '1.02',
  format: 'json',
  starttime: '2024-01-01',
  endtime: '2024-01-31',
  bbox: '68,8,97,37', // India bounding box
  data: 'MERRA2_400.tavgM_2d_aer_Nx:TOTEXTTAU'
};
```

### **3. NASA Earthdata Search API**
```javascript
// Search and download satellite data
const earthdataAPI = 'https://cmr.earthdata.nasa.gov/search/';

// Search for MODIS data over India
const modisSearch = `${earthdataAPI}granules.json?` +
  'collection_concept_id=C1000000240-LPDAAC_ECS&' +
  'bounding_box=68,8,97,37&' +
  'temporal=2024-01-01T00:00:00Z,2024-01-31T23:59:59Z';

// Requires free Earthdata account
const headers = {
  'Authorization': 'Bearer YOUR_EARTHDATA_TOKEN'
};
```

## ✅ **ESA COPERNICUS APIs (FREE)**

### **1. Copernicus Open Access Hub API**
```javascript
// Free registration required: https://scihub.copernicus.eu/
const copernicusAPI = 'https://scihub.copernicus.eu/dhus/search?';

// Search for Sentinel-2 data over Mumbai
const sentinelSearch = copernicusAPI +
  'q=platformname:Sentinel-2 AND ' +
  'footprint:"Intersects(POLYGON((72.8 18.9,72.9 18.9,72.9 19.0,72.8 19.0,72.8 18.9)))" AND ' +
  'beginPosition:[2024-01-01T00:00:00.000Z TO 2024-01-31T23:59:59.999Z]&' +
  'format=json&rows=100';

// Authentication with username:password
const auth = btoa('username:password');
fetch(sentinelSearch, {
  headers: {
    'Authorization': `Basic ${auth}`
  }
});
```

### **2. Sentinel Hub APIs**
```javascript
// Free tier: 1000 requests/month
const sentinelHubAPI = 'https://services.sentinel-hub.com/';

// Process API - Get processed satellite imagery
const processRequest = {
  input: {
    bounds: {
      bbox: [72.8, 18.9, 72.9, 19.0] // Mumbai port area
    },
    data: [{
      type: "sentinel-2-l2a",
      dataFilter: {
        timeRange: {
          from: "2024-01-01T00:00:00Z",
          to: "2024-01-31T23:59:59Z"
        },
        maxCloudCoverage: 20
      }
    }]
  },
  output: {
    width: 512,
    height: 512,
    responses: [{
      identifier: "default",
      format: {
        type: "image/jpeg"
      }
    }]
  },
  evalscript: `
    //VERSION=3
    function setup() {
      return {
        input: ["B02", "B03", "B04"],
        output: { bands: 3 }
      };
    }
    function evaluatePixel(sample) {
      return [sample.B04, sample.B03, sample.B02];
    }
  `
};

// Make request with OAuth token
fetch(`${sentinelHubAPI}api/v1/process`, {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer YOUR_OAUTH_TOKEN',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify(processRequest)
});
```

### **3. Copernicus Climate Data Store API**
```javascript
// Climate data API
const cdsAPI = 'https://cds.climate.copernicus.eu/api/v2/';

// Example: ERA5 weather data
const weatherRequest = {
  product_type: 'reanalysis',
  variable: ['2m_temperature', 'total_precipitation'],
  year: '2024',
  month: '01',
  day: ['01', '02', '03'],
  time: ['00:00', '12:00'],
  area: [37, 68, 8, 97], // India bounding box
  format: 'netcdf'
};
```

## ✅ **USGS APIs (FREE)**

### **1. USGS Earth Explorer Machine-to-Machine (M2M) API**
```javascript
// Free registration required: https://ers.cr.usgs.gov/
const usgsAPI = 'https://m2m.cr.usgs.gov/api/api/json/stable/';

// Login to get API key
const loginRequest = {
  username: 'your_username',
  password: 'your_password'
};

fetch(`${usgsAPI}login`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(loginRequest)
})
.then(response => response.json())
.then(data => {
  const apiKey = data.data;
  
  // Search for Landsat data
  const searchRequest = {
    datasetName: 'landsat_ot_c2_l2',
    spatialFilter: {
      filterType: 'mbr',
      lowerLeft: { latitude: 18.9, longitude: 72.8 },
      upperRight: { latitude: 19.0, longitude: 72.9 }
    },
    temporalFilter: {
      startDate: '2024-01-01',
      endDate: '2024-01-31'
    },
    maxResults: 100
  };
  
  return fetch(`${usgsAPI}scene-search`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Auth-Token': apiKey
    },
    body: JSON.stringify(searchRequest)
  });
});
```

### **2. USGS TNM (The National Map) API**
```javascript
// Elevation and geographic data
const tnmAPI = 'https://tnmaccess.nationalmap.gov/api/v1/';

// Get elevation data
const elevationRequest = `${tnmAPI}products?` +
  'datasets=National Elevation Dataset (NED) 1/3 arc-second&' +
  'bbox=72.8,18.9,72.9,19.0&' +
  'outputFormat=JSON';

fetch(elevationRequest)
  .then(response => response.json())
  .then(data => {
    console.log('Elevation data:', data);
  });
```

## 🔧 **PRACTICAL IMPLEMENTATION**

### **Complete API Integration Example**
```javascript
class SatelliteAPIClient {
  constructor() {
    this.apis = {
      nasa: {
        firms: 'https://firms.modaps.eosdis.nasa.gov/api/',
        earthdata: 'https://cmr.earthdata.nasa.gov/search/'
      },
      esa: {
        copernicus: 'https://scihub.copernicus.eu/dhus/',
        sentinelHub: 'https://services.sentinel-hub.com/'
      },
      usgs: {
        m2m: 'https://m2m.cr.usgs.gov/api/api/json/stable/',
        tnm: 'https://tnmaccess.nationalmap.gov/api/v1/'
      }
    };
    
    this.credentials = {
      earthdata: { token: process.env.EARTHDATA_TOKEN },
      copernicus: { 
        username: process.env.COPERNICUS_USER,
        password: process.env.COPERNICUS_PASS 
      },
      sentinelHub: { token: process.env.SENTINEL_HUB_TOKEN },
      usgs: {
        username: process.env.USGS_USER,
        password: process.env.USGS_PASS
      }
    };
  }

  // Get fire data from NASA FIRMS (no auth required)
  async getFireData(country = 'IND', days = 1) {
    const url = `${this.apis.nasa.firms}country/csv/VIIRS_SNPP_NRT/${country}/${days}`;
    
    try {
      const response = await fetch(url);
      const csvData = await response.text();
      return this.parseCSV(csvData);
    } catch (error) {
      console.error('Error fetching fire data:', error);
      return null;
    }
  }

  // Search Sentinel data from Copernicus
  async searchSentinelData(bbox, startDate, endDate, platform = 'Sentinel-2') {
    const polygon = this.bboxToPolygon(bbox);
    const query = `platformname:${platform} AND ` +
      `footprint:"Intersects(${polygon})" AND ` +
      `beginPosition:[${startDate} TO ${endDate}]`;
    
    const url = `${this.apis.esa.copernicus}search?q=${encodeURIComponent(query)}&format=json&rows=100`;
    
    const auth = btoa(`${this.credentials.copernicus.username}:${this.credentials.copernicus.password}`);
    
    try {
      const response = await fetch(url, {
        headers: { 'Authorization': `Basic ${auth}` }
      });
      return await response.json();
    } catch (error) {
      console.error('Error searching Sentinel data:', error);
      return null;
    }
  }

  // Get processed imagery from Sentinel Hub
  async getProcessedImagery(bbox, date, evalscript) {
    const request = {
      input: {
        bounds: { bbox },
        data: [{
          type: "sentinel-2-l2a",
          dataFilter: {
            timeRange: {
              from: `${date}T00:00:00Z`,
              to: `${date}T23:59:59Z`
            }
          }
        }]
      },
      output: {
        width: 512,
        height: 512,
        responses: [{ identifier: "default", format: { type: "image/jpeg" } }]
      },
      evalscript
    };

    try {
      const response = await fetch(`${this.apis.esa.sentinelHub}api/v1/process`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.credentials.sentinelHub.token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(request)
      });
      
      return await response.blob();
    } catch (error) {
      console.error('Error getting processed imagery:', error);
      return null;
    }
  }

  // Search Landsat data from USGS
  async searchLandsatData(bbox, startDate, endDate) {
    // First login to get API key
    const loginResponse = await fetch(`${this.apis.usgs.m2m}login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        username: this.credentials.usgs.username,
        password: this.credentials.usgs.password
      })
    });
    
    const loginData = await loginResponse.json();
    const apiKey = loginData.data;

    // Search for scenes
    const searchRequest = {
      datasetName: 'landsat_ot_c2_l2',
      spatialFilter: {
        filterType: 'mbr',
        lowerLeft: { latitude: bbox[1], longitude: bbox[0] },
        upperRight: { latitude: bbox[3], longitude: bbox[2] }
      },
      temporalFilter: { startDate, endDate },
      maxResults: 100
    };

    try {
      const response = await fetch(`${this.apis.usgs.m2m}scene-search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Auth-Token': apiKey
        },
        body: JSON.stringify(searchRequest)
      });
      
      return await response.json();
    } catch (error) {
      console.error('Error searching Landsat data:', error);
      return null;
    }
  }

  // Helper methods
  bboxToPolygon(bbox) {
    const [minX, minY, maxX, maxY] = bbox;
    return `POLYGON((${minX} ${minY},${maxX} ${minY},${maxX} ${maxY},${minX} ${maxY},${minX} ${minY}))`;
  }

  parseCSV(csvText) {
    const lines = csvText.split('\n');
    const headers = lines[0].split(',');
    
    return lines.slice(1).map(line => {
      const values = line.split(',');
      const obj = {};
      headers.forEach((header, index) => {
        obj[header.trim()] = values[index]?.trim();
      });
      return obj;
    }).filter(obj => Object.keys(obj).length > 1);
  }
}

// Usage example
const satelliteAPI = new SatelliteAPIClient();

// Get fire data for India
const fires = await satelliteAPI.getFireData('IND', 7);

// Search for Sentinel-2 data over Mumbai port
const mumbaiPort = [72.8, 18.9, 72.9, 19.0];
const sentinelData = await satelliteAPI.searchSentinelData(
  mumbaiPort, 
  '2024-01-01T00:00:00.000Z', 
  '2024-01-31T23:59:59.999Z'
);

// Get processed NDVI imagery
const ndviScript = `
  //VERSION=3
  function setup() {
    return {
      input: ["B04", "B08"],
      output: { bands: 3 }
    };
  }
  function evaluatePixel(sample) {
    let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
    return [ndvi, ndvi, ndvi];
  }
`;

const ndviImage = await satelliteAPI.getProcessedImagery(
  mumbaiPort, 
  '2024-01-15', 
  ndviScript
);
```

## 🔑 **API AUTHENTICATION SETUP**

### **Environment Variables**
```bash
# .env file
EARTHDATA_TOKEN=your_nasa_token
COPERNICUS_USER=your_copernicus_username
COPERNICUS_PASS=your_copernicus_password
SENTINEL_HUB_TOKEN=your_sentinel_hub_oauth_token
USGS_USER=your_usgs_username
USGS_PASS=your_usgs_password
```

### **Getting API Credentials**

1. **NASA Earthdata**: 
   - Register: https://urs.earthdata.nasa.gov/
   - Generate token: https://urs.earthdata.nasa.gov/profile

2. **ESA Copernicus**: 
   - Register: https://scihub.copernicus.eu/dhus/#/self-registration
   - Use username/password for basic auth

3. **Sentinel Hub**: 
   - Register: https://www.sentinel-hub.com/
   - Create OAuth app: https://apps.sentinel-hub.com/dashboard/

4. **USGS**: 
   - Register: https://ers.cr.usgs.gov/register/
   - Use username/password for M2M API

## 📊 **API RATE LIMITS**

| Service | Free Tier Limit | Cost for More |
|---------|-----------------|---------------|
| NASA FIRMS | Unlimited | Free |
| NASA Earthdata | Unlimited | Free |
| Copernicus Hub | Unlimited | Free |
| Sentinel Hub | 1,000 requests/month | $0.01-0.05 per request |
| USGS M2M | Unlimited | Free |

## 🚀 **Quick Start Integration**

```javascript
// Add to your existing supply chain system
const SatelliteAPIClient = require('./satellite-api-client');

class EnhancedSupplyChainIntelligence {
  constructor() {
    this.satelliteAPI = new SatelliteAPIClient();
  }

  async monitorPortActivity(portName, bbox) {
    // Get recent Sentinel-1 SAR data for ship detection
    const sarData = await this.satelliteAPI.searchSentinelData(
      bbox, 
      this.getDateDaysAgo(7), 
      new Date().toISOString(),
      'Sentinel-1'
    );

    // Analyze ship count and port congestion
    return this.analyzePortCongestion(sarData);
  }

  async monitorIndustrialActivity(facilityName, bbox) {
    // Get thermal imagery for facility utilization
    const thermalScript = `
      //VERSION=3
      function setup() {
        return { input: ["B11"], output: { bands: 1 } };
      }
      function evaluatePixel(sample) {
        return [sample.B11]; // Thermal infrared
      }
    `;

    const thermalImage = await this.satelliteAPI.getProcessedImagery(
      bbox, 
      this.getDateDaysAgo(1), 
      thermalScript
    );

    return this.analyzeFacilityUtilization(thermalImage);
  }

  async monitorEnvironmentalRisks(region, bbox) {
    // Get fire data
    const fires = await this.satelliteAPI.getFireData('IND', 7);
    
    // Filter fires in region
    const regionFires = fires.filter(fire => 
      this.isPointInBbox([fire.longitude, fire.latitude], bbox)
    );

    return this.assessEnvironmentalRisk(regionFires);
  }
}
```

All three agencies provide robust, free APIs that can give you real-time satellite intelligence for your supply chain monitoring system!
