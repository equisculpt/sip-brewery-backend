#!/usr/bin/env python3
"""
Real-Time Corporate Actions Monitor
Live monitoring with alerts for AGM, Results, Dividends, and Corporate Announcements
"""

import json
import logging
import requests
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, asdict
import os
import schedule

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LiveAlert:
    """Live alert data structure"""
    alert_id: str
    company_name: str
    symbol: str
    exchange: str
    alert_type: str  # NEW_RESULT, DIVIDEND_ANNOUNCED, AGM_SCHEDULED, etc.
    priority: str  # CRITICAL, HIGH, MEDIUM, LOW
    message: str
    event_date: Optional[str] = None
    amount: Optional[float] = None
    document_url: Optional[str] = None
    created_at: str = ""
    notified: bool = False

class RealTimeCorporateMonitor:
    """Real-time corporate actions monitoring system"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        self.monitoring = False
        self.alerts = []
        self.last_check = {}
        self.alert_callbacks = []
        
        # Monitoring configuration
        self.config = {
            "check_interval_minutes": 5,  # Check every 5 minutes
            "alert_threshold_hours": 24,  # Alert for events within 24 hours
            "max_alerts_per_run": 50,
            "priority_keywords": {
                "CRITICAL": ["bonus", "split", "merger", "acquisition", "delisting"],
                "HIGH": ["dividend", "result", "buyback", "rights"],
                "MEDIUM": ["agm", "board meeting", "announcement"],
                "LOW": ["circular", "notice", "clarification"]
            }
        }
        
        # Live data sources
        self.live_sources = {
            "NSE_LIVE": {
                "name": "NSE Live Feed",
                "url": "https://www.nseindia.com/api/corporates-corporateActions?index=equities",
                "type": "API",
                "enabled": True
            },
            "BSE_LIVE": {
                "name": "BSE Live Announcements",
                "url": "https://www.bseindia.com/corporates/ann.html",
                "type": "WEB",
                "enabled": True
            },
            "NSE_RESULTS": {
                "name": "NSE Financial Results",
                "url": "https://www.nseindia.com/api/corporates-financial-results?index=equities",
                "type": "API",
                "enabled": True
            },
            "BSE_EVENTS": {
                "name": "BSE Forthcoming Events",
                "url": "https://www.bseindia.com/corporates/Forthcoming_Events.html",
                "type": "WEB",
                "enabled": True
            }
        }
    
    def add_alert_callback(self, callback: Callable[[LiveAlert], None]):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def trigger_alert(self, alert: LiveAlert):
        """Trigger alert and notify callbacks"""
        self.alerts.append(alert)
        alert.notified = True
        
        logger.info(f"🚨 ALERT [{alert.priority}]: {alert.company_name} - {alert.alert_type}")
        logger.info(f"   Message: {alert.message}")
        
        # Call registered callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def determine_priority(self, text: str, action_type: str) -> str:
        """Determine alert priority"""
        text_lower = text.lower()
        
        for priority, keywords in self.config["priority_keywords"].items():
            for keyword in keywords:
                if keyword in text_lower:
                    return priority
        
        # Default priorities by action type
        if action_type in ["RESULT", "DIVIDEND", "BONUS", "SPLIT"]:
            return "HIGH"
        elif action_type in ["AGM", "BOARD_MEETING"]:
            return "MEDIUM"
        else:
            return "LOW"
    
    def check_nse_live_feed(self) -> List[LiveAlert]:
        """Check NSE live corporate actions"""
        alerts = []
        
        try:
            # Get NSE session
            self.session.get("https://www.nseindia.com")
            
            # Check corporate actions
            response = self.session.get(
                "https://www.nseindia.com/api/corporates-corporateActions?index=equities",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                current_time = datetime.now()
                
                for item in data.get('data', [])[:20]:  # Check latest 20
                    company_name = item.get('companyName', '')
                    symbol = item.get('symbol', '')
                    subject = item.get('subject', '')
                    
                    # Create unique key for tracking
                    item_key = f"NSE_{symbol}_{subject}_{item.get('anmDt', '')}"
                    
                    # Check if this is a new item
                    if item_key not in self.last_check:
                        action_type = self.classify_action_type(subject)
                        priority = self.determine_priority(subject, action_type)
                        
                        alert = LiveAlert(
                            alert_id=f"NSE_{current_time.timestamp()}_{symbol}",
                            company_name=company_name,
                            symbol=symbol,
                            exchange="NSE",
                            alert_type=f"NEW_{action_type}",
                            priority=priority,
                            message=f"New {action_type.lower()}: {subject}",
                            event_date=item.get('bcStDt'),
                            document_url=item.get('attchmntFile'),
                            created_at=current_time.isoformat()
                        )
                        
                        alerts.append(alert)
                        self.last_check[item_key] = current_time.isoformat()
                
                logger.info(f"✅ NSE Live: {len(alerts)} new alerts")
            
        except Exception as e:
            logger.warning(f"NSE live feed check failed: {e}")
        
        return alerts
    
    def check_bse_live_feed(self) -> List[LiveAlert]:
        """Check BSE live announcements"""
        alerts = []
        
        try:
            response = self.session.get(
                "https://www.bseindia.com/corporates/ann.html",
                timeout=10
            )
            
            if response.status_code == 200:
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(response.content, 'html.parser')
                current_time = datetime.now()
                
                # Find announcement tables
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    
                    for row in rows[1:11]:  # Check latest 10
                        cells = row.find_all(['td', 'th'])
                        
                        if len(cells) >= 4:
                            symbol = cells[0].get_text(strip=True)
                            company_name = cells[1].get_text(strip=True)
                            subject = cells[2].get_text(strip=True)
                            date = cells[3].get_text(strip=True)
                            
                            # Create unique key
                            item_key = f"BSE_{symbol}_{subject}_{date}"
                            
                            if item_key not in self.last_check and company_name and subject:
                                action_type = self.classify_action_type(subject)
                                priority = self.determine_priority(subject, action_type)
                                
                                alert = LiveAlert(
                                    alert_id=f"BSE_{current_time.timestamp()}_{symbol}",
                                    company_name=company_name,
                                    symbol=symbol,
                                    exchange="BSE",
                                    alert_type=f"NEW_{action_type}",
                                    priority=priority,
                                    message=f"New {action_type.lower()}: {subject}",
                                    created_at=current_time.isoformat()
                                )
                                
                                alerts.append(alert)
                                self.last_check[item_key] = current_time.isoformat()
                
                logger.info(f"✅ BSE Live: {len(alerts)} new alerts")
            
        except Exception as e:
            logger.warning(f"BSE live feed check failed: {e}")
        
        return alerts
    
    def classify_action_type(self, text: str) -> str:
        """Classify action type from text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["result", "quarterly", "q1", "q2", "q3", "q4"]):
            return "RESULT"
        elif any(word in text_lower for word in ["dividend", "interim dividend", "final dividend"]):
            return "DIVIDEND"
        elif any(word in text_lower for word in ["bonus", "bonus shares"]):
            return "BONUS"
        elif any(word in text_lower for word in ["split", "subdivision"]):
            return "SPLIT"
        elif any(word in text_lower for word in ["agm", "annual general meeting"]):
            return "AGM"
        elif any(word in text_lower for word in ["board meeting", "board"]):
            return "BOARD_MEETING"
        elif any(word in text_lower for word in ["buyback", "repurchase"]):
            return "BUYBACK"
        elif any(word in text_lower for word in ["rights", "rights issue"]):
            return "RIGHTS"
        else:
            return "ANNOUNCEMENT"
    
    def run_monitoring_cycle(self):
        """Run one monitoring cycle"""
        logger.info("🔍 Running monitoring cycle...")
        
        all_alerts = []
        
        # Check NSE live feed
        if self.live_sources["NSE_LIVE"]["enabled"]:
            nse_alerts = self.check_nse_live_feed()
            all_alerts.extend(nse_alerts)
            time.sleep(2)
        
        # Check BSE live feed
        if self.live_sources["BSE_LIVE"]["enabled"]:
            bse_alerts = self.check_bse_live_feed()
            all_alerts.extend(bse_alerts)
            time.sleep(2)
        
        # Trigger alerts
        for alert in all_alerts:
            self.trigger_alert(alert)
        
        # Save monitoring state
        self.save_monitoring_state()
        
        logger.info(f"✅ Monitoring cycle complete: {len(all_alerts)} new alerts")
    
    def save_monitoring_state(self):
        """Save monitoring state and alerts"""
        output_file = "market_data/real_time_monitoring_state.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert alerts to dict
        alerts_dict = []
        for alert in self.alerts[-100:]:  # Keep last 100 alerts
            alerts_dict.append(asdict(alert))
        
        state = {
            "monitoring_active": self.monitoring,
            "last_update": datetime.now().isoformat(),
            "total_alerts": len(self.alerts),
            "recent_alerts": alerts_dict,
            "last_check_items": dict(list(self.last_check.items())[-500:]),  # Keep last 500 items
            "config": self.config,
            "sources_status": self.live_sources
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    
    def load_monitoring_state(self):
        """Load previous monitoring state"""
        try:
            with open("market_data/real_time_monitoring_state.json", 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.last_check = state.get('last_check_items', {})
            logger.info(f"✅ Loaded monitoring state: {len(self.last_check)} tracked items")
            
        except FileNotFoundError:
            logger.info("No previous monitoring state found, starting fresh")
    
    def start_monitoring(self):
        """Start real-time monitoring"""
        logger.info("🚀 Starting Real-Time Corporate Actions Monitoring...")
        
        # Load previous state
        self.load_monitoring_state()
        
        self.monitoring = True
        
        # Schedule monitoring cycles
        schedule.every(self.config["check_interval_minutes"]).minutes.do(self.run_monitoring_cycle)
        
        logger.info(f"✅ Monitoring started:")
        logger.info(f"   Check Interval: {self.config['check_interval_minutes']} minutes")
        logger.info(f"   Sources: {len([s for s in self.live_sources.values() if s['enabled']])} enabled")
        
        # Run initial cycle
        self.run_monitoring_cycle()
        
        # Start monitoring loop
        while self.monitoring:
            try:
                schedule.run_pending()
                time.sleep(30)  # Check every 30 seconds for scheduled tasks
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("🛑 Monitoring stopped")
    
    def get_live_summary(self) -> Dict:
        """Get live monitoring summary"""
        current_time = datetime.now()
        recent_alerts = [a for a in self.alerts if 
                        (current_time - datetime.fromisoformat(a.created_at)).total_seconds() < 3600]  # Last hour
        
        summary = {
            "monitoring_status": "ACTIVE" if self.monitoring else "STOPPED",
            "total_alerts": len(self.alerts),
            "recent_alerts_1h": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.priority == "CRITICAL"]),
            "high_priority_alerts": len([a for a in recent_alerts if a.priority == "HIGH"]),
            "last_update": current_time.isoformat(),
            "sources_enabled": len([s for s in self.live_sources.values() if s["enabled"]]),
            "check_interval": f"{self.config['check_interval_minutes']} minutes"
        }
        
        return summary
    
    def run_demo_monitoring(self, duration_minutes: int = 10):
        """Run demo monitoring for specified duration"""
        logger.info(f"🚀 Starting Demo Real-Time Monitoring ({duration_minutes} minutes)...")
        
        # Load state
        self.load_monitoring_state()
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        cycle_count = 0
        
        while datetime.now() < end_time:
            cycle_count += 1
            logger.info(f"\n--- Monitoring Cycle {cycle_count} ---")
            
            self.run_monitoring_cycle()
            
            # Show summary
            summary = self.get_live_summary()
            logger.info(f"📊 Live Summary:")
            logger.info(f"   Recent Alerts (1h): {summary['recent_alerts_1h']}")
            logger.info(f"   Critical: {summary['critical_alerts']}")
            logger.info(f"   High Priority: {summary['high_priority_alerts']}")
            
            # Wait for next cycle
            wait_time = self.config["check_interval_minutes"] * 60  # Convert to seconds
            logger.info(f"⏰ Waiting {self.config['check_interval_minutes']} minutes for next cycle...")
            time.sleep(min(wait_time, 60))  # Max 1 minute wait for demo
        
        logger.info(f"\n🎉 Demo monitoring complete!")
        logger.info(f"   Duration: {duration_minutes} minutes")
        logger.info(f"   Cycles: {cycle_count}")
        logger.info(f"   Total Alerts: {len(self.alerts)}")

def alert_handler(alert: LiveAlert):
    """Example alert handler"""
    print(f"🔔 ALERT: {alert.company_name} ({alert.symbol}) - {alert.alert_type}")
    print(f"   Priority: {alert.priority}")
    print(f"   Message: {alert.message}")

def main():
    """Main function"""
    monitor = RealTimeCorporateMonitor()
    
    # Add alert handler
    monitor.add_alert_callback(alert_handler)
    
    # Run demo monitoring
    monitor.run_demo_monitoring(duration_minutes=5)

if __name__ == "__main__":
    main()
