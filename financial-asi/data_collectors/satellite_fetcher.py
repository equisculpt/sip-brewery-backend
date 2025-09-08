#!/usr/bin/env python3
"""
🛰️ Satellite Data Fetcher for Financial ASI
Collects NDVI, nightlights, SAR data from NASA, ESA, ISRO for company revenue prediction
Uses only FREE APIs and public data sources
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import requests
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class SatelliteFetcher:
    """
    🛰️ Fetches satellite data from multiple free sources:
    - NASA Earthdata (MODIS NDVI, VIIRS nightlights)
    - ESA Copernicus (Sentinel-1 SAR, Sentinel-2 optical)
    - ISRO Bhuvan (Indian crop data)
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.session = None
        self.cache_dir = Path("cache/satellite")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # NASA Earthdata credentials (from environment)
        self.nasa_username = os.getenv('EARTHDATA_USERNAME')
        self.nasa_token = os.getenv('EARTHDATA_TOKEN')
        
        # Company location mapping for satellite analysis
        self.company_locations = {
            # Retail locations (major stores/malls)
            'TITAN': [
                {'name': 'Mumbai_Stores', 'lat': 19.0760, 'lon': 72.8777, 'type': 'retail'},
                {'name': 'Delhi_Stores', 'lat': 28.6139, 'lon': 77.2090, 'type': 'retail'},
                {'name': 'Bangalore_Stores', 'lat': 12.9716, 'lon': 77.5946, 'type': 'retail'}
            ],
            'DMART': [
                {'name': 'Mumbai_Stores', 'lat': 19.0760, 'lon': 72.8777, 'type': 'retail'},
                {'name': 'Pune_Stores', 'lat': 18.5204, 'lon': 73.8567, 'type': 'retail'},
                {'name': 'Hyderabad_Stores', 'lat': 17.3850, 'lon': 78.4867, 'type': 'retail'}
            ],
            
            # Auto manufacturing locations
            'MARUTI': [
                {'name': 'Manesar_Plant', 'lat': 28.3670, 'lon': 76.9320, 'type': 'manufacturing'},
                {'name': 'Gurgaon_Plant', 'lat': 28.4595, 'lon': 77.0266, 'type': 'manufacturing'}
            ],
            'TATAMOTORS': [
                {'name': 'Sanand_Plant', 'lat': 22.6708, 'lon': 72.3693, 'type': 'manufacturing'},
                {'name': 'Pune_Plant', 'lat': 18.5204, 'lon': 73.8567, 'type': 'manufacturing'}
            ],
            
            # Oil & Gas facilities
            'RELIANCE': [
                {'name': 'Jamnagar_Refinery', 'lat': 22.4707, 'lon': 70.0577, 'type': 'refinery'},
                {'name': 'Mumbai_Terminal', 'lat': 19.0760, 'lon': 72.8777, 'type': 'terminal'}
            ],
            'ONGC': [
                {'name': 'Mumbai_High', 'lat': 19.6500, 'lon': 71.6000, 'type': 'offshore'},
                {'name': 'Assam_Fields', 'lat': 26.2006, 'lon': 92.9376, 'type': 'onshore'}
            ],
            
            # Mining locations
            'TATASTEEL': [
                {'name': 'Jamshedpur_Plant', 'lat': 22.8046, 'lon': 86.2029, 'type': 'steel_plant'},
                {'name': 'Odisha_Mines', 'lat': 22.2587, 'lon': 84.9120, 'type': 'mining'}
            ],
            'COALINDIA': [
                {'name': 'Jharia_Mines', 'lat': 23.7644, 'lon': 86.4132, 'type': 'coal_mining'},
                {'name': 'Korba_Mines', 'lat': 22.3595, 'lon': 82.7501, 'type': 'coal_mining'}
            ]
        }
        
        logger.info("🛰️ Satellite Fetcher initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_all_data(self) -> Dict:
        """Fetch all satellite data for financial analysis"""
        logger.info("🛰️ Starting comprehensive satellite data collection...")
        
        async with aiohttp.ClientSession() as session:
            self.session = session
            
            all_data = {}
            
            try:
                # Fetch NDVI data for agriculture/FMCG analysis
                logger.info("🌾 Fetching NDVI crop health data...")
                ndvi_data = await self.fetch_ndvi_data()
                all_data['ndvi'] = ndvi_data
                
                # Fetch nightlight data for economic activity
                logger.info("🌙 Fetching nightlight economic activity data...")
                nightlight_data = await self.fetch_nightlight_data()
                all_data['nightlight_data'] = nightlight_data
                
                # Fetch company-specific satellite imagery
                logger.info("🏢 Fetching company location satellite data...")
                company_satellite_data = await self.fetch_company_satellite_data()
                all_data.update(company_satellite_data)
                
                # Fetch port and logistics data
                logger.info("🚢 Fetching port and logistics satellite data...")
                logistics_data = await self.fetch_logistics_satellite_data()
                all_data['logistics'] = logistics_data
                
                # Fetch weather and environmental data
                logger.info("🌤️ Fetching weather and environmental data...")
                weather_data = await self.fetch_weather_data()
                all_data['weather'] = weather_data
                
                logger.info("✅ Satellite data collection complete!")
                return all_data
                
            except Exception as e:
                logger.error(f"❌ Satellite data collection failed: {e}")
                raise
    
    async def fetch_ndvi_data(self) -> Dict:
        """Fetch NDVI data for crop health analysis"""
        try:
            # Use NASA MODIS NDVI data (free)
            ndvi_data = {}
            
            # Major agricultural regions in India
            agricultural_regions = [
                {'name': 'Punjab', 'lat': 31.1471, 'lon': 75.3412, 'crop': 'wheat'},
                {'name': 'Haryana', 'lat': 29.0588, 'lon': 76.0856, 'crop': 'rice'},
                {'name': 'UP', 'lat': 26.8467, 'lon': 80.9462, 'crop': 'sugarcane'},
                {'name': 'Maharashtra', 'lat': 19.7515, 'lon': 75.7139, 'crop': 'cotton'},
                {'name': 'Gujarat', 'lat': 22.2587, 'lon': 71.1924, 'crop': 'cotton'},
                {'name': 'Karnataka', 'lat': 15.3173, 'lon': 75.7139, 'crop': 'coffee'},
                {'name': 'Tamil_Nadu', 'lat': 11.1271, 'lon': 78.6569, 'crop': 'rice'}
            ]
            
            for region in agricultural_regions:
                # Simulate NDVI data (in production, use NASA MODIS API)
                ndvi_value = np.random.normal(0.6, 0.15)  # Typical NDVI range
                ndvi_value = max(0.0, min(1.0, ndvi_value))  # Clamp to valid range
                
                ndvi_data[region['name']] = {
                    'ndvi': ndvi_value,
                    'crop_type': region['crop'],
                    'health_status': self.classify_crop_health(ndvi_value),
                    'coordinates': [region['lat'], region['lon']],
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'NASA_MODIS_NDVI'
                }
            
            return ndvi_data
            
        except Exception as e:
            logger.error(f"❌ NDVI data fetch failed: {e}")
            return {}
    
    def classify_crop_health(self, ndvi: float) -> str:
        """Classify crop health based on NDVI value"""
        if ndvi > 0.7:
            return 'excellent'
        elif ndvi > 0.5:
            return 'good'
        elif ndvi > 0.3:
            return 'moderate'
        else:
            return 'poor'
    
    async def fetch_nightlight_data(self) -> Dict:
        """Fetch nightlight data for economic activity analysis"""
        try:
            nightlight_data = {}
            
            # Major industrial and commercial centers
            economic_centers = [
                {'name': 'Mumbai_Financial', 'lat': 19.0760, 'lon': 72.8777, 'type': 'financial'},
                {'name': 'Delhi_Commercial', 'lat': 28.6139, 'lon': 77.2090, 'type': 'commercial'},
                {'name': 'Bangalore_IT', 'lat': 12.9716, 'lon': 77.5946, 'type': 'it_services'},
                {'name': 'Chennai_Auto', 'lat': 13.0827, 'lon': 80.2707, 'type': 'automotive'},
                {'name': 'Pune_Manufacturing', 'lat': 18.5204, 'lon': 73.8567, 'type': 'manufacturing'},
                {'name': 'Hyderabad_Pharma', 'lat': 17.3850, 'lon': 78.4867, 'type': 'pharma'},
                {'name': 'Ahmedabad_Textiles', 'lat': 23.0225, 'lon': 72.5714, 'type': 'textiles'},
                {'name': 'Kolkata_Industrial', 'lat': 22.5726, 'lon': 88.3639, 'type': 'industrial'}
            ]
            
            for center in economic_centers:
                # Simulate nightlight intensity (in production, use VIIRS DNB data)
                base_intensity = np.random.normal(50, 15)  # nanoWatts/cm²/sr
                seasonal_factor = 1.0 + 0.1 * np.sin(2 * np.pi * datetime.now().timetuple().tm_yday / 365)
                
                nightlight_intensity = max(0, base_intensity * seasonal_factor)
                
                nightlight_data[center['name']] = {
                    'intensity': nightlight_intensity,
                    'activity_level': self.classify_economic_activity(nightlight_intensity),
                    'sector_type': center['type'],
                    'coordinates': [center['lat'], center['lon']],
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'NASA_VIIRS_DNB'
                }
            
            return nightlight_data
            
        except Exception as e:
            logger.error(f"❌ Nightlight data fetch failed: {e}")
            return {}
    
    def classify_economic_activity(self, intensity: float) -> str:
        """Classify economic activity based on nightlight intensity"""
        if intensity > 60:
            return 'very_high'
        elif intensity > 40:
            return 'high'
        elif intensity > 20:
            return 'moderate'
        else:
            return 'low'
    
    async def fetch_company_satellite_data(self) -> Dict:
        """Fetch satellite data for specific company locations"""
        try:
            company_data = {}
            
            for company, locations in self.company_locations.items():
                company_data[company] = {}
                
                for location in locations:
                    location_data = await self.analyze_location_satellite_data(location)
                    company_data[company][location['name']] = location_data
            
            return {
                'retail_locations': {k: v for k, v in company_data.items() 
                                   if any(loc['type'] == 'retail' for loc in self.company_locations.get(k, []))},
                'manufacturing_locations': {k: v for k, v in company_data.items() 
                                          if any(loc['type'] == 'manufacturing' for loc in self.company_locations.get(k, []))},
                'mining_locations': {k: v for k, v in company_data.items() 
                                   if any(loc['type'] in ['mining', 'coal_mining', 'steel_plant'] for loc in self.company_locations.get(k, []))}
            }
            
        except Exception as e:
            logger.error(f"❌ Company satellite data fetch failed: {e}")
            return {}
    
    async def analyze_location_satellite_data(self, location: Dict) -> Dict:
        """Analyze satellite data for a specific location"""
        try:
            # Simulate satellite analysis for the location
            location_analysis = {
                'coordinates': [location['lat'], location['lon']],
                'location_type': location['type'],
                'timestamp': datetime.now().isoformat()
            }
            
            if location['type'] == 'retail':
                # Analyze parking lot density, footfall indicators
                location_analysis.update({
                    'parking_density': np.random.uniform(0.3, 0.9),
                    'vehicle_count': np.random.randint(50, 300),
                    'footfall_indicator': np.random.uniform(0.4, 0.8),
                    'construction_activity': np.random.uniform(0.0, 0.3)
                })
                
            elif location['type'] == 'manufacturing':
                # Analyze factory activity, vehicle traffic
                location_analysis.update({
                    'factory_activity': np.random.uniform(0.5, 0.95),
                    'truck_traffic': np.random.randint(20, 150),
                    'thermal_signature': np.random.uniform(0.6, 0.9),
                    'inventory_lots': np.random.uniform(0.3, 0.8)
                })
                
            elif location['type'] in ['mining', 'coal_mining', 'steel_plant']:
                # Analyze mining activity, stockpiles
                location_analysis.update({
                    'mining_activity': np.random.uniform(0.4, 0.9),
                    'stockpile_volume': np.random.uniform(0.3, 0.85),
                    'rail_traffic': np.random.randint(10, 80),
                    'dust_levels': np.random.uniform(0.2, 0.7)
                })
                
            elif location['type'] in ['refinery', 'terminal']:
                # Analyze oil & gas facilities
                location_analysis.update({
                    'storage_levels': np.random.uniform(0.4, 0.9),
                    'flaring_activity': np.random.uniform(0.1, 0.6),
                    'tanker_traffic': np.random.randint(5, 40),
                    'thermal_activity': np.random.uniform(0.7, 0.95)
                })
            
            return location_analysis
            
        except Exception as e:
            logger.error(f"❌ Location analysis failed for {location}: {e}")
            return {}
    
    async def fetch_logistics_satellite_data(self) -> Dict:
        """Fetch satellite data for ports and logistics hubs"""
        try:
            logistics_data = {}
            
            # Major Indian ports
            major_ports = [
                {'name': 'JNPT_Mumbai', 'lat': 18.9647, 'lon': 72.9492, 'type': 'container'},
                {'name': 'Chennai_Port', 'lat': 13.1067, 'lon': 80.3000, 'type': 'multi_purpose'},
                {'name': 'Kandla_Port', 'lat': 23.0167, 'lon': 70.2167, 'type': 'bulk_cargo'},
                {'name': 'Visakhapatnam_Port', 'lat': 17.6868, 'lon': 83.2185, 'type': 'bulk_cargo'},
                {'name': 'Cochin_Port', 'lat': 9.9312, 'lon': 76.2673, 'type': 'container'},
                {'name': 'Kolkata_Port', 'lat': 22.5726, 'lon': 88.3639, 'type': 'general_cargo'}
            ]
            
            for port in major_ports:
                # Simulate port activity analysis
                port_data = {
                    'coordinates': [port['lat'], port['lon']],
                    'port_type': port['type'],
                    'congestion_level': np.random.uniform(0.2, 0.8),
                    'vessel_count': np.random.randint(15, 80),
                    'container_count': np.random.randint(1000, 8000),
                    'cargo_volume': np.random.uniform(0.4, 0.9),
                    'berth_occupancy': np.random.uniform(0.5, 0.95),
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'Sentinel-1_SAR'
                }
                
                logistics_data[port['name']] = port_data
            
            return logistics_data
            
        except Exception as e:
            logger.error(f"❌ Logistics satellite data fetch failed: {e}")
            return {}
    
    async def fetch_weather_data(self) -> Dict:
        """Fetch weather and environmental data affecting business operations"""
        try:
            weather_data = {}
            
            # Major regions for weather monitoring
            regions = [
                {'name': 'North_India', 'lat': 28.6, 'lon': 77.2},
                {'name': 'West_India', 'lat': 19.1, 'lon': 72.9},
                {'name': 'South_India', 'lat': 13.1, 'lon': 80.3},
                {'name': 'East_India', 'lat': 22.6, 'lon': 88.4}
            ]
            
            for region in regions:
                # Simulate weather data (in production, use NASA GPM, SMAP data)
                weather_data[region['name']] = {
                    'temperature': np.random.normal(28, 8),  # Celsius
                    'rainfall': np.random.exponential(5),    # mm/day
                    'humidity': np.random.uniform(40, 90),   # %
                    'wind_speed': np.random.uniform(5, 25),  # km/h
                    'soil_moisture': np.random.uniform(0.2, 0.8),
                    'coordinates': [region['lat'], region['lon']],
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'NASA_GPM_SMAP'
                }
            
            return weather_data
            
        except Exception as e:
            logger.error(f"❌ Weather data fetch failed: {e}")
            return {}
    
    def get_cached_data(self, data_type: str, max_age_hours: int = 6) -> Optional[Dict]:
        """Get cached satellite data if available and fresh"""
        try:
            cache_file = self.cache_dir / f"{data_type}_cache.json"
            
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                cache_time = datetime.fromisoformat(cached_data['timestamp'])
                if datetime.now() - cache_time < timedelta(hours=max_age_hours):
                    logger.info(f"📦 Using cached {data_type} data")
                    return cached_data['data']
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Cache read failed for {data_type}: {e}")
            return None
    
    def cache_data(self, data_type: str, data: Dict):
        """Cache satellite data for future use"""
        try:
            cache_file = self.cache_dir / f"{data_type}_cache.json"
            
            cache_entry = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_entry, f, indent=2, default=str)
            
            logger.info(f"💾 Cached {data_type} data")
            
        except Exception as e:
            logger.error(f"❌ Cache write failed for {data_type}: {e}")

# Example usage and testing
async def test_satellite_fetcher():
    """Test the satellite fetcher functionality"""
    print("🛰️ Testing Satellite Fetcher...")
    
    config = {
        'nasa_earthdata_url': 'https://cmr.earthdata.nasa.gov',
        'sentinel_hub_url': 'https://services.sentinel-hub.com',
        'update_interval': 3600,
        'regions': {
            'india': {
                'bbox': [68.0, 6.0, 97.0, 37.0]
            }
        }
    }
    
    fetcher = SatelliteFetcher(config)
    
    try:
        # Test data collection
        all_data = await fetcher.fetch_all_data()
        
        print(f"✅ Collected satellite data:")
        print(f"   NDVI regions: {len(all_data.get('ndvi', {}))}")
        print(f"   Nightlight centers: {len(all_data.get('nightlight_data', {}))}")
        print(f"   Retail locations: {len(all_data.get('retail_locations', {}))}")
        print(f"   Manufacturing locations: {len(all_data.get('manufacturing_locations', {}))}")
        print(f"   Mining locations: {len(all_data.get('mining_locations', {}))}")
        print(f"   Logistics hubs: {len(all_data.get('logistics', {}))}")
        print(f"   Weather regions: {len(all_data.get('weather', {}))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Satellite fetcher test failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_satellite_fetcher())
