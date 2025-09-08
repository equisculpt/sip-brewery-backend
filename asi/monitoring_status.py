#!/usr/bin/env python3
"""
Live Monitoring System Status Check
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MonitoringStatus:
    """Check status of live monitoring system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.market_data_dir = self.base_dir / "market_data"
        
    def check_deployment_status(self):
        """Check if system is properly deployed"""
        logger.info("LIVE MONITORING SYSTEM STATUS")
        logger.info("=" * 60)
        
        # Check required files
        required_files = [
            "live_corporate_actions_crawler.py",
            "real_time_corporate_monitor.py",
            "corporate_actions_dashboard.py"
        ]
        
        logger.info("\nCore Components:")
        for file in required_files:
            file_path = self.base_dir / file
            if file_path.exists():
                logger.info(f"  ✓ {file}")
            else:
                logger.info(f"  ✗ {file} - MISSING")
        
        # Check directories
        required_dirs = ["market_data", "logs", "config"]
        logger.info("\nDirectories:")
        for dir_name in required_dirs:
            dir_path = self.base_dir / dir_name
            if dir_path.exists():
                logger.info(f"  ✓ {dir_name}/")
            else:
                logger.info(f"  ✗ {dir_name}/ - MISSING")
        
        # Check startup scripts
        startup_scripts = ["start_monitoring.bat"]
        logger.info("\nStartup Scripts:")
        for script in startup_scripts:
            script_path = self.base_dir / script
            if script_path.exists():
                logger.info(f"  ✓ {script}")
            else:
                logger.info(f"  ✗ {script} - MISSING")
    
    def check_data_status(self):
        """Check data collection status"""
        logger.info("\nData Collection Status:")
        
        # Check live corporate actions data
        actions_file = self.market_data_dir / "live_corporate_actions.json"
        if actions_file.exists():
            try:
                with open(actions_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                metadata = data.get('metadata', {})
                actions = data.get('corporate_actions', [])
                
                logger.info(f"  ✓ Corporate Actions Data:")
                logger.info(f"    - Total Actions: {metadata.get('total_actions', 0)}")
                logger.info(f"    - Last Updated: {metadata.get('crawling_timestamp', 'Unknown')}")
                logger.info(f"    - Data Sources: {', '.join(metadata.get('data_sources', []))}")
                logger.info(f"    - High Impact: {metadata.get('high_impact_count', 0)}")
                logger.info(f"    - Urgent: {metadata.get('urgent_count', 0)}")
                
                # Exchange distribution
                exchange_dist = metadata.get('exchange_distribution', {})
                if exchange_dist:
                    logger.info(f"    - Exchange Distribution:")
                    for exchange, count in exchange_dist.items():
                        logger.info(f"      • {exchange}: {count} actions")
                
            except Exception as e:
                logger.error(f"  ✗ Error reading corporate actions data: {e}")
        else:
            logger.info(f"  ✗ No corporate actions data found")
        
        # Check monitoring state
        state_file = self.market_data_dir / "real_time_monitoring_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state_data = json.load(f)
                
                logger.info(f"  ✓ Monitoring State:")
                logger.info(f"    - Last Check: {state_data.get('last_check_time', 'Unknown')}")
                logger.info(f"    - Total Alerts: {len(state_data.get('alerts', []))}")
                logger.info(f"    - Tracked Items: {len(state_data.get('last_check', {}))}")
                
            except Exception as e:
                logger.error(f"  ✗ Error reading monitoring state: {e}")
        else:
            logger.info(f"  ✗ No monitoring state found")
    
    def check_system_health(self):
        """Check overall system health"""
        logger.info("\nSystem Health Check:")
        
        try:
            # Test component imports
            from live_corporate_actions_crawler import LiveCorporateActionsCrawler
            from real_time_corporate_monitor import RealTimeCorporateMonitor
            from corporate_actions_dashboard import CorporateActionsDashboard
            
            logger.info("  ✓ All components can be imported")
            
            # Test crawler initialization
            crawler = LiveCorporateActionsCrawler()
            logger.info("  ✓ Corporate Actions Crawler - OK")
            
            # Test monitor initialization
            monitor = RealTimeCorporateMonitor()
            logger.info("  ✓ Real-Time Monitor - OK")
            
            # Test dashboard initialization
            dashboard = CorporateActionsDashboard()
            logger.info("  ✓ Dashboard - OK")
            
            logger.info("  ✓ System Health: EXCELLENT")
            
        except Exception as e:
            logger.error(f"  ✗ System Health Check Failed: {e}")
    
    def show_usage_instructions(self):
        """Show usage instructions"""
        logger.info("\nUsage Instructions:")
        logger.info("=" * 30)
        
        logger.info("\n1. Start Real-Time Monitoring:")
        logger.info("   • Windows: Double-click 'start_monitoring.bat'")
        logger.info("   • Command: python real_time_corporate_monitor.py")
        
        logger.info("\n2. View Live Dashboard:")
        logger.info("   • Command: python corporate_actions_dashboard.py")
        
        logger.info("\n3. Run Fresh Data Collection:")
        logger.info("   • Command: python live_corporate_actions_crawler.py")
        
        logger.info("\n4. Demo Monitoring (5 minutes):")
        logger.info("   • The monitor will run automatically and show live alerts")
        logger.info("   • Press Ctrl+C to stop monitoring")
        
        logger.info("\n5. Monitoring Features:")
        logger.info("   • Real-time AGM announcements")
        logger.info("   • Quarterly results tracking")
        logger.info("   • Dividend declarations")
        logger.info("   • Board meeting notifications")
        logger.info("   • Corporate action alerts")
        logger.info("   • Live dashboard analytics")
    
    def show_data_sources(self):
        """Show configured data sources"""
        logger.info("\nConfigured Data Sources:")
        logger.info("=" * 35)
        
        sources = {
            "NSE Live API": "https://www.nseindia.com/api/corporates-corporateActions",
            "BSE Announcements": "https://www.bseindia.com/corporates/ann.html",
            "NSE Results": "https://www.nseindia.com/api/corporates-financial-results",
            "BSE Events": "https://www.bseindia.com/corporates/Forthcoming_Events.html",
            "SEBI Filings": "https://www.sebi.gov.in/filings/exchange-filings"
        }
        
        for name, url in sources.items():
            logger.info(f"  • {name}")
            logger.info(f"    {url}")
    
    def run_status_check(self):
        """Run complete status check"""
        self.check_deployment_status()
        self.check_data_status()
        self.check_system_health()
        self.show_data_sources()
        self.show_usage_instructions()
        
        logger.info("\n" + "=" * 60)
        logger.info("STATUS CHECK COMPLETE")
        logger.info("=" * 60)

def main():
    """Main function"""
    status_checker = MonitoringStatus()
    status_checker.run_status_check()

if __name__ == "__main__":
    main()
