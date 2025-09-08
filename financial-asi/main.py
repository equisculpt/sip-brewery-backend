#!/usr/bin/env python3
"""
🧠💼 ULTIMATE FINANCIAL ASI - Main Controller
Financial Artificial Superintelligence for accurate company revenue & market forecasting
Using only FREE satellite data and public APIs
"""

import asyncio
import logging
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_collectors.satellite_fetcher import SatelliteFetcher
from data_collectors.traffic_scraper import TrafficScraper
from data_collectors.macro_scraper import MacroScraper
from data_collectors.company_mapper import CompanyMapper
from image_models.yolo_vehicle_count import VehicleCounter
from image_models.nightlight_analyzer import NightlightAnalyzer
from image_models.mine_stockpile_volume import StockpileAnalyzer
from ml_models.macro_predictor import MacroPredictor
from ml_models.sector_forecaster import SectorForecaster
from ml_models.company_eps_estimator import CompanyEPSEstimator
from ml_models.market_movement_fuser import MarketMovementFuser
from output.report_generator import ReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_asi.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinancialASI:
    """
    🧠 Financial Artificial Superintelligence System
    Predicts market movements and company earnings using satellite data + AI
    """
    
    def __init__(self):
        self.config = self.load_config()
        self.data_collectors = {}
        self.image_models = {}
        self.ml_models = {}
        self.predictions = {}
        self.last_update = None
        
        # Create output directory
        Path("output").mkdir(exist_ok=True)
        
        logger.info("🧠 Financial ASI initialized")
    
    def load_config(self):
        """Load configuration for the ASI system"""
        return {
            'satellite': {
                'nasa_earthdata_url': 'https://cmr.earthdata.nasa.gov',
                'sentinel_hub_url': 'https://services.sentinel-hub.com',
                'update_interval': 3600,  # 1 hour
                'regions': {
                    'india': {
                        'bbox': [68.0, 6.0, 97.0, 37.0],  # [min_lon, min_lat, max_lon, max_lat]
                        'major_cities': ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad'],
                        'industrial_zones': ['manesar', 'sanand', 'surat', 'coimbatore', 'pune']
                    }
                }
            },
            'companies': {
                'retail': ['TITAN', 'TRENT', 'DMART', 'RELIANCE', 'SHOPERSTOP'],
                'auto': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO'],
                'oil_gas': ['RELIANCE', 'ONGC', 'BPCL', 'IOC', 'GAIL'],
                'mining': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'COALINDIA', 'VEDL'],
                'fmcg': ['HUL', 'ITC', 'NESTLEIND', 'BRITANNIA', 'DABUR'],
                'manufacturing': ['TATASTEEL', 'L&T', 'BHEL', 'SAIL', 'HINDALCO']
            },
            'prediction_horizon': 30,  # days
            'confidence_threshold': 0.7,
            'update_frequency': 'daily'
        }
    
    async def initialize(self):
        """Initialize all components of the Financial ASI"""
        logger.info("🚀 Initializing Financial ASI components...")
        
        try:
            # Initialize data collectors
            logger.info("📡 Initializing data collectors...")
            self.data_collectors['satellite'] = SatelliteFetcher(self.config['satellite'])
            self.data_collectors['traffic'] = TrafficScraper()
            self.data_collectors['macro'] = MacroScraper()
            self.data_collectors['company_mapper'] = CompanyMapper(self.config['companies'])
            
            # Initialize image analysis models
            logger.info("🖼️ Initializing image analysis models...")
            self.image_models['vehicle_counter'] = VehicleCounter()
            self.image_models['nightlight_analyzer'] = NightlightAnalyzer()
            self.image_models['stockpile_analyzer'] = StockpileAnalyzer()
            
            # Initialize ML prediction models
            logger.info("🤖 Initializing ML prediction models...")
            self.ml_models['macro_predictor'] = MacroPredictor()
            self.ml_models['sector_forecaster'] = SectorForecaster()
            self.ml_models['company_eps_estimator'] = CompanyEPSEstimator(self.config['companies'])
            self.ml_models['market_fuser'] = MarketMovementFuser()
            
            # Initialize report generator
            self.report_generator = ReportGenerator()
            
            logger.info("✅ Financial ASI initialization complete!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Financial ASI: {e}")
            raise
    
    async def collect_all_data(self):
        """Collect data from all sources"""
        logger.info("📊 Starting comprehensive data collection...")
        
        collected_data = {}
        
        try:
            # Collect satellite data
            logger.info("🛰️ Collecting satellite data...")
            satellite_data = await self.data_collectors['satellite'].fetch_all_data()
            collected_data['satellite'] = satellite_data
            
            # Collect traffic and logistics data
            logger.info("🚚 Collecting traffic and logistics data...")
            traffic_data = await self.data_collectors['traffic'].fetch_all_data()
            collected_data['traffic'] = traffic_data
            
            # Collect macro economic data
            logger.info("📈 Collecting macro economic data...")
            macro_data = await self.data_collectors['macro'].fetch_all_data()
            collected_data['macro'] = macro_data
            
            # Map companies to geographic locations
            logger.info("🏢 Mapping company locations...")
            company_locations = await self.data_collectors['company_mapper'].get_company_locations()
            collected_data['company_locations'] = company_locations
            
            logger.info("✅ Data collection complete!")
            return collected_data
            
        except Exception as e:
            logger.error(f"❌ Data collection failed: {e}")
            raise
    
    async def analyze_satellite_images(self, satellite_data):
        """Analyze satellite images using computer vision models"""
        logger.info("🔍 Analyzing satellite images with AI models...")
        
        analysis_results = {}
        
        try:
            # Analyze vehicle counts at retail locations
            if 'retail_locations' in satellite_data:
                logger.info("🚗 Counting vehicles at retail locations...")
                vehicle_counts = await self.image_models['vehicle_counter'].analyze_locations(
                    satellite_data['retail_locations']
                )
                analysis_results['vehicle_counts'] = vehicle_counts
            
            # Analyze nightlight intensity for economic activity
            if 'nightlight_data' in satellite_data:
                logger.info("🌙 Analyzing nightlight economic activity...")
                nightlight_analysis = await self.image_models['nightlight_analyzer'].analyze_activity(
                    satellite_data['nightlight_data']
                )
                analysis_results['nightlight_activity'] = nightlight_analysis
            
            # Analyze mining stockpiles
            if 'mining_locations' in satellite_data:
                logger.info("⛏️ Analyzing mining stockpiles...")
                stockpile_volumes = await self.image_models['stockpile_analyzer'].analyze_stockpiles(
                    satellite_data['mining_locations']
                )
                analysis_results['stockpile_volumes'] = stockpile_volumes
            
            logger.info("✅ Satellite image analysis complete!")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Satellite image analysis failed: {e}")
            raise
    
    async def generate_predictions(self, all_data, image_analysis):
        """Generate comprehensive market and company predictions"""
        logger.info("🔮 Generating AI predictions...")
        
        predictions = {}
        
        try:
            # Generate macro economic predictions
            logger.info("📊 Predicting macro economic trends...")
            macro_predictions = await self.ml_models['macro_predictor'].predict(
                all_data['macro'], all_data['traffic']
            )
            predictions['macro'] = macro_predictions
            
            # Generate sector-wise forecasts
            logger.info("🏭 Forecasting sector performance...")
            sector_forecasts = await self.ml_models['sector_forecaster'].predict(
                all_data, image_analysis, macro_predictions
            )
            predictions['sectors'] = sector_forecasts
            
            # Generate company-specific EPS estimates
            logger.info("🏢 Estimating company EPS and revenue...")
            company_predictions = await self.ml_models['company_eps_estimator'].predict(
                all_data, image_analysis, sector_forecasts
            )
            predictions['companies'] = company_predictions
            
            # Fuse all predictions for market movement
            logger.info("🎯 Fusing predictions for market movement...")
            market_movement = await self.ml_models['market_fuser'].predict(
                macro_predictions, sector_forecasts, company_predictions
            )
            predictions['market_movement'] = market_movement
            
            logger.info("✅ AI predictions generated successfully!")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed: {e}")
            raise
    
    async def run_full_analysis(self):
        """Run complete Financial ASI analysis cycle"""
        logger.info("🧠 Starting complete Financial ASI analysis...")
        
        try:
            start_time = datetime.now()
            
            # Step 1: Collect all data
            all_data = await self.collect_all_data()
            
            # Step 2: Analyze satellite images
            image_analysis = await self.analyze_satellite_images(all_data['satellite'])
            
            # Step 3: Generate predictions
            predictions = await self.generate_predictions(all_data, image_analysis)
            
            # Step 4: Generate comprehensive report
            logger.info("📄 Generating comprehensive report...")
            report_path = await self.report_generator.generate_report(
                all_data, image_analysis, predictions
            )
            
            # Store results
            self.predictions = predictions
            self.last_update = datetime.now()
            
            # Save predictions to JSON
            with open('output/latest_predictions.json', 'w') as f:
                json.dump({
                    'timestamp': self.last_update.isoformat(),
                    'predictions': predictions,
                    'data_summary': {
                        'satellite_sources': len(all_data.get('satellite', {})),
                        'companies_analyzed': len(predictions.get('companies', {})),
                        'sectors_analyzed': len(predictions.get('sectors', {}))
                    }
                }, f, indent=2, default=str)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info(f"🎉 Financial ASI analysis complete! Duration: {duration:.1f}s")
            logger.info(f"📊 Report generated: {report_path}")
            
            # Display key insights
            self.display_key_insights(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Financial ASI analysis failed: {e}")
            raise
    
    def display_key_insights(self, predictions):
        """Display key insights from predictions"""
        print("\n" + "="*80)
        print("🧠💼 FINANCIAL ASI - KEY INSIGHTS")
        print("="*80)
        
        # Market Movement
        if 'market_movement' in predictions:
            market = predictions['market_movement']
            print(f"\n📈 MARKET OUTLOOK:")
            print(f"   Nifty Prediction: {market.get('nifty_direction', 'N/A')} ({market.get('nifty_confidence', 0):.1%} confidence)")
            print(f"   Sensex Prediction: {market.get('sensex_direction', 'N/A')} ({market.get('sensex_confidence', 0):.1%} confidence)")
            print(f"   Time Horizon: {market.get('time_horizon', 'N/A')} days")
        
        # Top Company Predictions
        if 'companies' in predictions:
            companies = predictions['companies']
            print(f"\n🏢 TOP COMPANY PREDICTIONS:")
            
            # Sort by confidence and show top 5
            sorted_companies = sorted(
                companies.items(), 
                key=lambda x: x[1].get('confidence', 0), 
                reverse=True
            )[:5]
            
            for symbol, pred in sorted_companies:
                revenue_growth = pred.get('revenue_growth', 0)
                eps_growth = pred.get('eps_growth', 0)
                confidence = pred.get('confidence', 0)
                
                print(f"   {symbol}: Revenue {revenue_growth:+.1%}, EPS {eps_growth:+.1%} ({confidence:.1%} confidence)")
        
        # Sector Performance
        if 'sectors' in predictions:
            sectors = predictions['sectors']
            print(f"\n🏭 SECTOR OUTLOOK:")
            for sector, pred in sectors.items():
                outlook = pred.get('outlook', 'N/A')
                confidence = pred.get('confidence', 0)
                print(f"   {sector.title()}: {outlook} ({confidence:.1%} confidence)")
        
        print("\n" + "="*80)
        print(f"📊 Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("📄 Detailed report available in output/prediction_report.pdf")
        print("="*80 + "\n")
    
    async def start_continuous_monitoring(self):
        """Start continuous monitoring and prediction updates"""
        logger.info("🔄 Starting continuous Financial ASI monitoring...")
        
        while True:
            try:
                await self.run_full_analysis()
                
                # Wait for next update cycle
                update_interval = self.config['satellite']['update_interval']
                logger.info(f"⏰ Next update in {update_interval/3600:.1f} hours...")
                await asyncio.sleep(update_interval)
                
            except KeyboardInterrupt:
                logger.info("🛑 Continuous monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Error in continuous monitoring: {e}")
                logger.info("⏰ Retrying in 5 minutes...")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

async def main():
    """Main entry point for Financial ASI"""
    print("🧠💼 ULTIMATE FINANCIAL ASI - Starting...")
    print("Using FREE satellite data + AI for accurate market & company predictions")
    print("="*80)
    
    # Create Financial ASI instance
    asi = FinancialASI()
    
    try:
        # Initialize the system
        await asi.initialize()
        
        # Run single analysis or continuous monitoring
        import sys
        if '--continuous' in sys.argv:
            await asi.start_continuous_monitoring()
        else:
            predictions = await asi.run_full_analysis()
            
            # Show summary
            print("\n🎉 Financial ASI analysis complete!")
            print("📊 Check output/prediction_report.pdf for detailed insights")
            print("💡 Run with --continuous flag for real-time monitoring")
        
    except KeyboardInterrupt:
        print("\n🛑 Financial ASI stopped by user")
    except Exception as e:
        logger.error(f"❌ Financial ASI failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
