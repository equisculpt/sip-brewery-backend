"""
ASI Market Data Auto-Updater
Autonomous System Intelligence for maintaining comprehensive market data
Automatically detects new listings, delistings, and updates database
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any
from pathlib import Path
import schedule
import time
from dataclasses import dataclass
import aiohttp

from comprehensive_market_data_fetcher import ComprehensiveMarketDataFetcher, CompanyData, MutualFundData

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asi_market_updater.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class MarketChange:
    """Represents a change in market data"""
    change_type: str  # NEW_LISTING, DELISTING, UPDATE, STATUS_CHANGE
    instrument_type: str  # COMPANY, MUTUAL_FUND
    symbol_or_code: str
    name: str
    details: Dict[str, Any]
    detected_at: str

class ASIMarketDataUpdater:
    """Autonomous System Intelligence for market data updates"""
    
    def __init__(self):
        self.fetcher = None
        self.data_dir = Path("market_data")
        self.changes_log = self.data_dir / "changes_log.json"
        self.asi_config = self.data_dir / "asi_config.json"
        
        # ASI configuration
        self.config = {
            "update_frequency_hours": 6,  # Update every 6 hours
            "quick_check_frequency_minutes": 30,  # Quick checks every 30 minutes
            "max_retries": 3,
            "alert_on_changes": True,
            "auto_integrate_changes": True,
            "backup_before_update": True
        }
        
        self.load_config()
        
        # Change tracking
        self.detected_changes: List[MarketChange] = []
        self.last_full_update = None
        self.last_quick_check = None

    def load_config(self):
        """Load ASI configuration"""
        try:
            if self.asi_config.exists():
                with open(self.asi_config, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
        except Exception as e:
            logger.error(f"Error loading ASI config: {e}")

    def save_config(self):
        """Save ASI configuration"""
        try:
            with open(self.asi_config, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving ASI config: {e}")

    def load_changes_log(self) -> List[MarketChange]:
        """Load historical changes log"""
        try:
            if self.changes_log.exists():
                with open(self.changes_log, 'r') as f:
                    data = json.load(f)
                    return [MarketChange(**change) for change in data.get("changes", [])]
        except Exception as e:
            logger.error(f"Error loading changes log: {e}")
        return []

    def save_changes_log(self, changes: List[MarketChange]):
        """Save changes log"""
        try:
            changes_data = {
                "last_updated": datetime.now().isoformat(),
                "total_changes": len(changes),
                "changes": [change.__dict__ for change in changes]
            }
            
            with open(self.changes_log, 'w') as f:
                json.dump(changes_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving changes log: {e}")

    async def detect_new_listings(self, current_companies: List[CompanyData], 
                                 previous_companies: List[CompanyData]) -> List[MarketChange]:
        """Detect new company listings"""
        changes = []
        
        current_symbols = {c.symbol for c in current_companies}
        previous_symbols = {c.symbol for c in previous_companies}
        
        new_symbols = current_symbols - previous_symbols
        
        for symbol in new_symbols:
            company = next(c for c in current_companies if c.symbol == symbol)
            change = MarketChange(
                change_type="NEW_LISTING",
                instrument_type="COMPANY",
                symbol_or_code=symbol,
                name=company.name,
                details={
                    "exchange": company.exchange,
                    "sector": company.sector,
                    "industry": company.industry,
                    "listing_date": company.listing_date
                },
                detected_at=datetime.now().isoformat()
            )
            changes.append(change)
            logger.info(f"🆕 NEW LISTING DETECTED: {symbol} - {company.name} on {company.exchange}")
        
        return changes

    async def detect_delistings(self, current_companies: List[CompanyData], 
                               previous_companies: List[CompanyData]) -> List[MarketChange]:
        """Detect company delistings"""
        changes = []
        
        current_symbols = {c.symbol for c in current_companies}
        previous_symbols = {c.symbol for c in previous_companies}
        
        delisted_symbols = previous_symbols - current_symbols
        
        for symbol in delisted_symbols:
            company = next(c for c in previous_companies if c.symbol == symbol)
            change = MarketChange(
                change_type="DELISTING",
                instrument_type="COMPANY",
                symbol_or_code=symbol,
                name=company.name,
                details={
                    "exchange": company.exchange,
                    "sector": company.sector,
                    "last_seen": company.last_updated
                },
                detected_at=datetime.now().isoformat()
            )
            changes.append(change)
            logger.warning(f"🚫 DELISTING DETECTED: {symbol} - {company.name}")
        
        return changes

    async def detect_mutual_fund_changes(self, current_funds: List[MutualFundData], 
                                        previous_funds: List[MutualFundData]) -> List[MarketChange]:
        """Detect mutual fund changes"""
        changes = []
        
        current_codes = {f.scheme_code for f in current_funds}
        previous_codes = {f.scheme_code for f in previous_funds}
        
        # New funds
        new_codes = current_codes - previous_codes
        for code in new_codes:
            fund = next(f for f in current_funds if f.scheme_code == code)
            change = MarketChange(
                change_type="NEW_LISTING",
                instrument_type="MUTUAL_FUND",
                symbol_or_code=code,
                name=fund.scheme_name,
                details={
                    "amc_name": fund.amc_name,
                    "category": fund.category,
                    "sub_category": fund.sub_category,
                    "nav": fund.nav
                },
                detected_at=datetime.now().isoformat()
            )
            changes.append(change)
            logger.info(f"🆕 NEW MUTUAL FUND: {code} - {fund.scheme_name}")
        
        # Closed funds
        closed_codes = previous_codes - current_codes
        for code in closed_codes:
            fund = next(f for f in previous_funds if f.scheme_code == code)
            change = MarketChange(
                change_type="DELISTING",
                instrument_type="MUTUAL_FUND",
                symbol_or_code=code,
                name=fund.scheme_name,
                details={
                    "amc_name": fund.amc_name,
                    "category": fund.category,
                    "last_nav": fund.nav
                },
                detected_at=datetime.now().isoformat()
            )
            changes.append(change)
            logger.warning(f"🚫 MUTUAL FUND CLOSED: {code} - {fund.scheme_name}")
        
        return changes

    async def perform_full_update(self) -> Dict[str, Any]:
        """Perform full database update with change detection"""
        logger.info("🔄 ASI: Starting full market data update...")
        
        try:
            # Load previous data
            previous_companies, previous_funds = self.fetcher.load_existing_data()
            
            # Backup current data if configured
            if self.config["backup_before_update"]:
                await self.create_backup()
            
            # Fetch new data
            result = await self.fetcher.update_database(force_refresh=True)
            
            if result["status"] == "success":
                # Load new data
                current_companies, current_funds = self.fetcher.load_existing_data()
                
                # Detect changes
                changes = []
                
                if previous_companies:
                    changes.extend(await self.detect_new_listings(current_companies, previous_companies))
                    changes.extend(await self.detect_delistings(current_companies, previous_companies))
                
                if previous_funds:
                    changes.extend(await self.detect_mutual_fund_changes(current_funds, previous_funds))
                
                # Log changes
                if changes:
                    self.detected_changes.extend(changes)
                    self.save_changes_log(self.detected_changes)
                    
                    if self.config["alert_on_changes"]:
                        await self.send_change_alerts(changes)
                
                self.last_full_update = datetime.now()
                
                logger.info(f"✅ ASI: Full update complete. {len(changes)} changes detected.")
                
                return {
                    "status": "success",
                    "changes_detected": len(changes),
                    "companies_count": len(current_companies),
                    "funds_count": len(current_funds),
                    "update_time": self.last_full_update.isoformat()
                }
            else:
                logger.error("❌ ASI: Full update failed")
                return {"status": "failed", "reason": "Data fetch failed"}
                
        except Exception as e:
            logger.error(f"❌ ASI: Full update error: {e}")
            return {"status": "error", "error": str(e)}

    async def perform_quick_check(self) -> Dict[str, Any]:
        """Perform quick check for critical changes"""
        logger.info("⚡ ASI: Performing quick market check...")
        
        try:
            # Quick check using lightweight APIs
            async with aiohttp.ClientSession() as session:
                # Check NSE for any major announcements
                nse_url = "https://www.nseindia.com/api/corporate-announcements"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                try:
                    async with session.get(nse_url, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            # Process announcements for listings/delistings
                            announcements = data.get("data", [])
                            
                            critical_announcements = []
                            for announcement in announcements[:10]:  # Check recent 10
                                subject = announcement.get("subject", "").lower()
                                if any(keyword in subject for keyword in ["listing", "delisting", "suspension"]):
                                    critical_announcements.append(announcement)
                            
                            if critical_announcements:
                                logger.info(f"⚠️ ASI: {len(critical_announcements)} critical announcements found")
                                # Trigger full update if critical changes detected
                                return await self.perform_full_update()
                                
                except Exception as e:
                    logger.warning(f"Quick check API error: {e}")
            
            self.last_quick_check = datetime.now()
            return {"status": "no_changes", "check_time": self.last_quick_check.isoformat()}
            
        except Exception as e:
            logger.error(f"❌ ASI: Quick check error: {e}")
            return {"status": "error", "error": str(e)}

    async def create_backup(self):
        """Create backup of current data"""
        try:
            backup_dir = self.data_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup companies
            companies_file = self.data_dir / "all_companies.json"
            if companies_file.exists():
                backup_companies = backup_dir / f"companies_backup_{timestamp}.json"
                with open(companies_file, 'r') as src, open(backup_companies, 'w') as dst:
                    dst.write(src.read())
            
            # Backup mutual funds
            funds_file = self.data_dir / "all_mutual_funds.json"
            if funds_file.exists():
                backup_funds = backup_dir / f"funds_backup_{timestamp}.json"
                with open(funds_file, 'r') as src, open(backup_funds, 'w') as dst:
                    dst.write(src.read())
            
            logger.info(f"📦 ASI: Backup created with timestamp {timestamp}")
            
        except Exception as e:
            logger.error(f"❌ ASI: Backup creation failed: {e}")

    async def send_change_alerts(self, changes: List[MarketChange]):
        """Send alerts for detected changes"""
        try:
            alert_summary = {
                "timestamp": datetime.now().isoformat(),
                "total_changes": len(changes),
                "new_listings": len([c for c in changes if c.change_type == "NEW_LISTING"]),
                "delistings": len([c for c in changes if c.change_type == "DELISTING"]),
                "companies": len([c for c in changes if c.instrument_type == "COMPANY"]),
                "mutual_funds": len([c for c in changes if c.instrument_type == "MUTUAL_FUND"])
            }
            
            # Save alert to file
            alerts_file = self.data_dir / "asi_alerts.json"
            alerts = []
            
            if alerts_file.exists():
                with open(alerts_file, 'r') as f:
                    alerts = json.load(f)
            
            alerts.append(alert_summary)
            
            # Keep only last 100 alerts
            alerts = alerts[-100:]
            
            with open(alerts_file, 'w') as f:
                json.dump(alerts, f, indent=2)
            
            logger.info(f"🚨 ASI: Alert sent - {alert_summary}")
            
        except Exception as e:
            logger.error(f"❌ ASI: Alert sending failed: {e}")

    async def start_autonomous_monitoring(self):
        """Start autonomous monitoring system"""
        logger.info("🤖 ASI: Starting autonomous market data monitoring...")
        
        # Initialize fetcher
        self.fetcher = ComprehensiveMarketDataFetcher()
        
        # Schedule tasks
        schedule.every(self.config["update_frequency_hours"]).hours.do(
            lambda: asyncio.create_task(self.perform_full_update())
        )
        
        schedule.every(self.config["quick_check_frequency_minutes"]).minutes.do(
            lambda: asyncio.create_task(self.perform_quick_check())
        )
        
        # Perform initial update
        async with self.fetcher:
            await self.perform_full_update()
        
        # Start monitoring loop
        while True:
            try:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                logger.info("🛑 ASI: Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ ASI: Monitoring error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

    def get_status_report(self) -> Dict[str, Any]:
        """Get current ASI status report"""
        return {
            "asi_status": "active",
            "last_full_update": self.last_full_update.isoformat() if self.last_full_update else None,
            "last_quick_check": self.last_quick_check.isoformat() if self.last_quick_check else None,
            "total_changes_detected": len(self.detected_changes),
            "recent_changes": len([c for c in self.detected_changes 
                                 if datetime.fromisoformat(c.detected_at) > datetime.now() - timedelta(days=7)]),
            "config": self.config,
            "next_full_update": (self.last_full_update + timedelta(hours=self.config["update_frequency_hours"])).isoformat() 
                               if self.last_full_update else "pending"
        }

async def main():
    """Main function to start ASI monitoring"""
    updater = ASIMarketDataUpdater()
    await updater.start_autonomous_monitoring()

if __name__ == "__main__":
    asyncio.run(main())
