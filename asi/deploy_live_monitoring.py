#!/usr/bin/env python3
"""
Live Corporate Actions Monitoring System Deployment
Complete deployment and startup script for real-time monitoring
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

class LiveMonitoringDeployment:
    """Deployment manager for live monitoring system"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.market_data_dir = self.base_dir / "market_data"
        self.required_packages = [
            "requests",
            "beautifulsoup4", 
            "pandas",
            "schedule",
            "lxml",
            "selenium"
        ]
        
        self.monitoring_components = {
            "crawler": "live_corporate_actions_crawler.py",
            "monitor": "real_time_corporate_monitor.py", 
            "dashboard": "corporate_actions_dashboard.py"
        }
        
        self.deployment_config = {
            "monitoring_interval": 5,  # minutes
            "alert_threshold": 24,     # hours
            "max_alerts_per_run": 50,
            "enable_email_alerts": False,
            "enable_webhook_alerts": False,
            "log_level": "INFO"
        }

    def check_dependencies(self):
        """Check and install required dependencies"""
        logger.info("🔍 Checking dependencies...")
        
        missing_packages = []
        for package in self.required_packages:
            try:
                __import__(package.replace("-", "_"))
                logger.info(f"   ✅ {package} - installed")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"   ❌ {package} - missing")
        
        if missing_packages:
            logger.info(f"📦 Installing missing packages: {missing_packages}")
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    logger.info(f"   ✅ {package} - installed successfully")
                except subprocess.CalledProcessError as e:
                    logger.error(f"   ❌ Failed to install {package}: {e}")
                    return False
        
        return True

    def setup_directories(self):
        """Setup required directories"""
        logger.info("📁 Setting up directories...")
        
        # Create market_data directory
        self.market_data_dir.mkdir(exist_ok=True)
        logger.info(f"   ✅ Created: {self.market_data_dir}")
        
        # Create logs directory
        logs_dir = self.base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        logger.info(f"   ✅ Created: {logs_dir}")
        
        # Create config directory
        config_dir = self.base_dir / "config"
        config_dir.mkdir(exist_ok=True)
        logger.info(f"   ✅ Created: {config_dir}")
        
        return True

    def validate_components(self):
        """Validate monitoring components exist"""
        logger.info("🔧 Validating monitoring components...")
        
        for name, filename in self.monitoring_components.items():
            filepath = self.base_dir / filename
            if filepath.exists():
                logger.info(f"   ✅ {name}: {filename}")
            else:
                logger.error(f"   ❌ {name}: {filename} - NOT FOUND")
                return False
        
        return True

    def create_deployment_config(self):
        """Create deployment configuration file"""
        logger.info("⚙️ Creating deployment configuration...")
        
        config_file = self.base_dir / "config" / "monitoring_config.json"
        
        config = {
            "deployment": {
                "deployed_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "environment": "production"
            },
            "monitoring": self.deployment_config,
            "data_sources": {
                "NSE_LIVE": {
                    "enabled": True,
                    "url": "https://www.nseindia.com/api/corporates-corporateActions",
                    "timeout": 30
                },
                "BSE_LIVE": {
                    "enabled": True,
                    "url": "https://www.bseindia.com/corporates/ann.html",
                    "timeout": 30
                },
                "SEBI_FILINGS": {
                    "enabled": True,
                    "url": "https://www.sebi.gov.in/filings/exchange-filings",
                    "timeout": 30
                }
            },
            "alerts": {
                "email": {
                    "enabled": False,
                    "smtp_server": "",
                    "smtp_port": 587,
                    "username": "",
                    "password": "",
                    "recipients": []
                },
                "webhook": {
                    "enabled": False,
                    "url": "",
                    "headers": {}
                }
            }
        }
        
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"   ✅ Config saved: {config_file}")
        return True

    def test_monitoring_components(self):
        """Test monitoring components"""
        logger.info("🧪 Testing monitoring components...")
        
        # Test crawler
        try:
            logger.info("   Testing corporate actions crawler...")
            from live_corporate_actions_crawler import LiveCorporateActionsCrawler
            crawler = LiveCorporateActionsCrawler()
            logger.info("   ✅ Crawler initialized successfully")
        except Exception as e:
            logger.error(f"   ❌ Crawler test failed: {e}")
            return False
        
        # Test monitor
        try:
            logger.info("   Testing real-time monitor...")
            from real_time_corporate_monitor import RealTimeCorporateMonitor
            monitor = RealTimeCorporateMonitor()
            logger.info("   ✅ Monitor initialized successfully")
        except Exception as e:
            logger.error(f"   ❌ Monitor test failed: {e}")
            return False
        
        # Test dashboard
        try:
            logger.info("   Testing dashboard...")
            from corporate_actions_dashboard import CorporateActionsDashboard
            dashboard = CorporateActionsDashboard()
            logger.info("   ✅ Dashboard initialized successfully")
        except Exception as e:
            logger.error(f"   ❌ Dashboard test failed: {e}")
            return False
        
        return True

    def create_startup_scripts(self):
        """Create startup scripts for different platforms"""
        logger.info("📜 Creating startup scripts...")
        
        # Windows batch script
        windows_script = self.base_dir / "start_monitoring.bat"
        with open(windows_script, 'w') as f:
            f.write(f"""@echo off
echo 🚀 Starting Live Corporate Actions Monitoring System...
cd /d "{self.base_dir}"
python real_time_corporate_monitor.py
pause
""")
        logger.info(f"   ✅ Windows script: {windows_script}")
        
        # Linux/Mac shell script
        unix_script = self.base_dir / "start_monitoring.sh"
        with open(unix_script, 'w') as f:
            f.write(f"""#!/bin/bash
echo "🚀 Starting Live Corporate Actions Monitoring System..."
cd "{self.base_dir}"
python3 real_time_corporate_monitor.py
""")
        
        # Make shell script executable
        try:
            os.chmod(unix_script, 0o755)
        except:
            pass  # Windows doesn't support chmod
        
        logger.info(f"   ✅ Unix script: {unix_script}")
        
        # Dashboard startup script
        dashboard_script = self.base_dir / "start_dashboard.bat"
        with open(dashboard_script, 'w') as f:
            f.write(f"""@echo off
echo 📊 Starting Corporate Actions Dashboard...
cd /d "{self.base_dir}"
python corporate_actions_dashboard.py
pause
""")
        logger.info(f"   ✅ Dashboard script: {dashboard_script}")
        
        return True

    def create_service_files(self):
        """Create service files for system deployment"""
        logger.info("🔧 Creating service files...")
        
        # Systemd service file (Linux)
        service_content = f"""[Unit]
Description=Live Corporate Actions Monitoring System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory={self.base_dir}
ExecStart=/usr/bin/python3 {self.base_dir}/real_time_corporate_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = self.base_dir / "live-monitoring.service"
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        logger.info(f"   ✅ Systemd service: {service_file}")
        
        # Docker compose file
        docker_compose = f"""version: '3.8'

services:
  live-monitoring:
    build: .
    container_name: live-corporate-monitoring
    restart: unless-stopped
    volumes:
      - ./market_data:/app/market_data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONUNBUFFERED=1
      - LOG_LEVEL=INFO
    networks:
      - monitoring-network

  dashboard:
    build: .
    container_name: corporate-dashboard
    restart: unless-stopped
    ports:
      - "8080:8080"
    volumes:
      - ./market_data:/app/market_data
      - ./logs:/app/logs
    command: python corporate_actions_dashboard.py
    depends_on:
      - live-monitoring
    networks:
      - monitoring-network

networks:
  monitoring-network:
    driver: bridge
"""
        
        docker_file = self.base_dir / "docker-compose.yml"
        with open(docker_file, 'w') as f:
            f.write(docker_compose)
        
        logger.info(f"   ✅ Docker Compose: {docker_file}")
        
        return True

    def run_initial_data_collection(self):
        """Run initial data collection"""
        logger.info("📊 Running initial data collection...")
        
        try:
            from live_corporate_actions_crawler import LiveCorporateActionsCrawler
            crawler = LiveCorporateActionsCrawler()
            
            logger.info("   Fetching latest corporate actions...")
            actions = crawler.crawl_all_corporate_actions()
            
            logger.info(f"   ✅ Collected {len(actions)} corporate actions")
            
            # Save data
            crawler.save_corporate_actions(actions)
            logger.info("   ✅ Data saved successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"   ❌ Initial data collection failed: {e}")
            return False

    def deploy(self):
        """Run complete deployment"""
        logger.info("🚀 Starting Live Monitoring System Deployment...")
        logger.info("="*80)
        
        steps = [
            ("Dependencies", self.check_dependencies),
            ("Directories", self.setup_directories),
            ("Components", self.validate_components),
            ("Configuration", self.create_deployment_config),
            ("Testing", self.test_monitoring_components),
            ("Startup Scripts", self.create_startup_scripts),
            ("Service Files", self.create_service_files),
            ("Initial Data", self.run_initial_data_collection)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n📋 Step: {step_name}")
            if not step_func():
                logger.error(f"❌ Deployment failed at step: {step_name}")
                return False
            logger.info(f"✅ Step completed: {step_name}")
        
        logger.info("\n" + "="*80)
        logger.info("🎉 DEPLOYMENT SUCCESSFUL!")
        logger.info("="*80)
        
        self.show_deployment_summary()
        return True

    def show_deployment_summary(self):
        """Show deployment summary and next steps"""
        logger.info("\n📋 DEPLOYMENT SUMMARY:")
        logger.info("="*50)
        
        logger.info("✅ Components Deployed:")
        for name, filename in self.monitoring_components.items():
            logger.info(f"   • {name.title()}: {filename}")
        
        logger.info("\n🚀 How to Start Monitoring:")
        logger.info("   Windows: Double-click 'start_monitoring.bat'")
        logger.info("   Linux/Mac: ./start_monitoring.sh")
        logger.info("   Python: python real_time_corporate_monitor.py")
        
        logger.info("\n📊 How to View Dashboard:")
        logger.info("   Windows: Double-click 'start_dashboard.bat'")
        logger.info("   Python: python corporate_actions_dashboard.py")
        
        logger.info("\n🔧 Configuration:")
        logger.info(f"   Config File: config/monitoring_config.json")
        logger.info(f"   Data Directory: market_data/")
        logger.info(f"   Logs Directory: logs/")
        
        logger.info("\n⚙️ System Service (Linux):")
        logger.info("   sudo cp live-monitoring.service /etc/systemd/system/")
        logger.info("   sudo systemctl enable live-monitoring")
        logger.info("   sudo systemctl start live-monitoring")
        
        logger.info("\n🐳 Docker Deployment:")
        logger.info("   docker-compose up -d")
        
        logger.info("\n🎯 Monitoring Features:")
        logger.info("   • Real-time AGM announcements")
        logger.info("   • Quarterly results tracking")
        logger.info("   • Dividend declarations")
        logger.info("   • Board meeting notifications")
        logger.info("   • Corporate action alerts")
        logger.info("   • Live dashboard analytics")
        
        logger.info("\n" + "="*50)

def main():
    """Main deployment function"""
    deployment = LiveMonitoringDeployment()
    
    try:
        success = deployment.deploy()
        if success:
            logger.info("\n🎉 Live monitoring system is ready for production!")
            
            # Ask user if they want to start monitoring immediately
            response = input("\n🚀 Start monitoring now? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                logger.info("Starting live monitoring...")
                from real_time_corporate_monitor import RealTimeCorporateMonitor
                
                monitor = RealTimeCorporateMonitor()
                
                # Add simple alert handler
                def alert_handler(alert):
                    print(f"🔔 ALERT: {alert.company_name} - {alert.alert_type}")
                
                monitor.add_alert_callback(alert_handler)
                
                # Start monitoring
                monitor.start_monitoring()
        else:
            logger.error("❌ Deployment failed. Please check the logs above.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ Deployment interrupted by user")
    except Exception as e:
        logger.error(f"❌ Deployment error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
