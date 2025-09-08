#!/usr/bin/env python3
"""
🏢 Company Location Mapper for Financial ASI
Maps companies to their geographic locations for satellite analysis
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)

class CompanyMapper:
    """🏢 Maps companies to geographic locations"""
    
    def __init__(self, companies_config: Dict):
        self.companies_config = companies_config
        
        # Pre-defined company locations
        self.company_locations = {
            'TITAN': [
                {'name': 'Mumbai_Stores', 'lat': 19.0760, 'lon': 72.8777, 'type': 'retail'},
                {'name': 'Delhi_Stores', 'lat': 28.6139, 'lon': 77.2090, 'type': 'retail'},
                {'name': 'Bangalore_Stores', 'lat': 12.9716, 'lon': 77.5946, 'type': 'retail'}
            ],
            'MARUTI': [
                {'name': 'Manesar_Plant', 'lat': 28.3670, 'lon': 76.9320, 'type': 'manufacturing'},
                {'name': 'Gurgaon_Plant', 'lat': 28.4595, 'lon': 77.0266, 'type': 'manufacturing'}
            ],
            'RELIANCE': [
                {'name': 'Jamnagar_Refinery', 'lat': 22.4707, 'lon': 70.0577, 'type': 'refinery'},
                {'name': 'Mumbai_Terminal', 'lat': 19.0760, 'lon': 72.8777, 'type': 'terminal'}
            ],
            'TATASTEEL': [
                {'name': 'Jamshedpur_Plant', 'lat': 22.8046, 'lon': 86.2029, 'type': 'steel_plant'},
                {'name': 'Odisha_Mines', 'lat': 22.2587, 'lon': 84.9120, 'type': 'mining'}
            ],
            'HUL': [
                {'name': 'Mumbai_Factory', 'lat': 19.0760, 'lon': 72.8777, 'type': 'manufacturing'},
                {'name': 'Bangalore_Factory', 'lat': 12.9716, 'lon': 77.5946, 'type': 'manufacturing'}
            ]
        }
        
        logger.info("🏢 Company Mapper initialized")
    
    async def get_company_locations(self) -> Dict:
        """Get all company locations for satellite analysis"""
        logger.info("🏢 Mapping company locations...")
        
        mapped_locations = {}
        
        for company, locations in self.company_locations.items():
            mapped_locations[company] = {
                'locations': locations,
                'total_locations': len(locations),
                'primary_type': locations[0]['type'] if locations else 'unknown',
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"✅ Mapped {len(mapped_locations)} companies to locations")
        return mapped_locations
