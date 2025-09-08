#!/usr/bin/env python3
"""
Simple Live Monitoring Deployment Script
"""

import os
import sys
import subprocess
import logging
import json
import time
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_dependencies():
    """Install required packages"""
    logger.info("Installing dependencies...")
    
    packages = [
        "requests",
        "beautifulsoup4", 
        "pandas",
        "schedule",
        "lxml"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            logger.info(f"Installed: {package}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")
            return False
    
    return True

def setup_directories():
    """Create required directories"""
    logger.info("Setting up directories...")
    
    base_dir = Path(__file__).parent
    
    # Create directories
    dirs = ["market_data", "logs", "config"]
    for dir_name in dirs:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"Created directory: {dir_name}")
    
    return True

def test_components():
    """Test monitoring components"""
    logger.info("Testing components...")
    
    try:
        # Test imports
        from live_corporate_actions_crawler import LiveCorporateActionsCrawler
        from real_time_corporate_monitor import RealTimeCorporateMonitor
        from corporate_actions_dashboard import CorporateActionsDashboard
        
        logger.info("All components imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"Component import failed: {e}")
        return False

def create_startup_script():
    """Create Windows startup script"""
    logger.info("Creating startup script...")
    
    base_dir = Path(__file__).parent
    script_path = base_dir / "start_monitoring.bat"
    
    with open(script_path, 'w') as f:
        f.write(f"""@echo off
echo Starting Live Corporate Actions Monitoring System...
cd /d "{base_dir}"
python real_time_corporate_monitor.py
pause
""")
    
    logger.info(f"Created startup script: {script_path}")
    return True

def run_initial_crawl():
    """Run initial data collection"""
    logger.info("Running initial data collection...")
    
    try:
        from live_corporate_actions_crawler import LiveCorporateActionsCrawler
        
        crawler = LiveCorporateActionsCrawler()
        actions = crawler.crawl_all_corporate_actions()
        
        logger.info(f"Collected {len(actions)} corporate actions")
        
        # Save data
        crawler.save_corporate_actions(actions)
        logger.info("Data saved successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"Initial crawl failed: {e}")
        return False

def main():
    """Main deployment function"""
    logger.info("Starting Live Monitoring System Deployment")
    logger.info("=" * 60)
    
    steps = [
        ("Installing Dependencies", install_dependencies),
        ("Setting up Directories", setup_directories),
        ("Testing Components", test_components),
        ("Creating Startup Script", create_startup_script),
        ("Running Initial Data Collection", run_initial_crawl)
    ]
    
    for step_name, step_func in steps:
        logger.info(f"\nStep: {step_name}")
        if not step_func():
            logger.error(f"Deployment failed at step: {step_name}")
            return False
        logger.info(f"Step completed: {step_name}")
    
    logger.info("\n" + "=" * 60)
    logger.info("DEPLOYMENT SUCCESSFUL!")
    logger.info("=" * 60)
    
    logger.info("\nHow to start monitoring:")
    logger.info("1. Double-click 'start_monitoring.bat'")
    logger.info("2. Or run: python real_time_corporate_monitor.py")
    
    logger.info("\nHow to view dashboard:")
    logger.info("Run: python corporate_actions_dashboard.py")
    
    # Ask if user wants to start monitoring now
    try:
        response = input("\nStart monitoring now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            logger.info("Starting live monitoring...")
            
            from real_time_corporate_monitor import RealTimeCorporateMonitor
            
            monitor = RealTimeCorporateMonitor()
            
            # Add simple alert handler
            def alert_handler(alert):
                print(f"ALERT: {alert.company_name} - {alert.alert_type}")
            
            monitor.add_alert_callback(alert_handler)
            
            # Run demo monitoring for 5 minutes
            monitor.run_demo_monitoring(duration_minutes=5)
            
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    
    return True

if __name__ == "__main__":
    main()
