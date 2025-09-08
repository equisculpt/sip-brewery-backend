#!/usr/bin/env python3
"""
Live Corporate Actions & Submissions Crawler
Real-time monitoring of AGM, Results, Board Meetings, and Corporate Announcements
NSE + BSE + SEBI Live Submissions
"""

import json
import logging
import requests
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup
import pandas as pd
import re
from urllib.parse import urljoin, urlparse
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class CorporateAction:
    """Corporate action/submission data structure"""
    company_name: str
    symbol: str
    exchange: str
    action_type: str  # AGM, RESULT, BOARD_MEETING, DIVIDEND, BONUS, SPLIT, etc.
    announcement_date: str
    record_date: Optional[str] = None
    ex_date: Optional[str] = None
    event_date: Optional[str] = None
    description: str = ""
    amount: Optional[float] = None
    percentage: Optional[float] = None
    document_url: Optional[str] = None
    filing_category: str = ""
    urgency: str = "NORMAL"  # HIGH, NORMAL, LOW
    market_impact: str = "MEDIUM"  # HIGH, MEDIUM, LOW
    data_source: str = ""
    last_updated: str = ""
    raw_data: Dict = None

class LiveCorporateActionsCrawler:
    """Live crawler for corporate actions and submissions"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/html, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Corporate action sources
        self.corporate_sources = {
            "NSE_CORPORATE_ACTIONS": {
                "name": "NSE Corporate Actions",
                "base_url": "https://www.nseindia.com",
                "endpoints": [
                    "/api/corporates-corporateActions?index=equities",
                    "/api/corporates-board-meetings?index=equities", 
                    "/api/corporates-financial-results?index=equities",
                    "/api/event-calendar"
                ],
                "type": "LIVE_API"
            },
            "BSE_CORPORATE_ACTIONS": {
                "name": "BSE Corporate Actions",
                "base_url": "https://www.bseindia.com",
                "endpoints": [
                    "/corporates/ann.html",
                    "/corporates/Forthcoming_Events.html",
                    "/corporates/Results.html",
                    "/xml-data/corpfiling/AttachLive/AttachLive.xml"
                ],
                "type": "WEB_SCRAPING"
            },
            "SEBI_FILINGS": {
                "name": "SEBI Live Filings",
                "base_url": "https://www.sebi.gov.in",
                "endpoints": [
                    "/filings/exchange-filings",
                    "/sebiweb/other/OtherAction.do?doRecognised=yes",
                    "/enforcement/enforcement-actions"
                ],
                "type": "REGULATORY"
            },
            "NSE_ANNOUNCEMENTS": {
                "name": "NSE Live Announcements",
                "base_url": "https://www.nseindia.com",
                "endpoints": [
                    "/companies-listing/corporate-filings-announcements",
                    "/products-services/equity-derivatives-list",
                    "/market-data/live-equity-market"
                ],
                "type": "ANNOUNCEMENTS"
            },
            "BSE_ANNOUNCEMENTS": {
                "name": "BSE Live Announcements", 
                "base_url": "https://www.bseindia.com",
                "endpoints": [
                    "/corporates/ann.html",
                    "/corporates/List_Scrips.html",
                    "/markets/MarketInfo/DispNewNoticesCirculars.html"
                ],
                "type": "ANNOUNCEMENTS"
            }
        }
        
        self.corporate_actions = []
        self.action_keywords = {
            "AGM": ["annual general meeting", "agm", "general meeting"],
            "RESULT": ["quarterly results", "financial results", "q1", "q2", "q3", "q4", "annual results"],
            "DIVIDEND": ["dividend", "interim dividend", "final dividend"],
            "BONUS": ["bonus shares", "bonus issue", "bonus"],
            "SPLIT": ["stock split", "share split", "subdivision"],
            "BOARD_MEETING": ["board meeting", "board", "meeting"],
            "BUYBACK": ["buyback", "share buyback", "repurchase"],
            "RIGHTS": ["rights issue", "rights shares"],
            "MERGER": ["merger", "amalgamation", "acquisition"],
            "DELISTING": ["delisting", "voluntary delisting"],
            "IPO": ["initial public offering", "ipo", "public issue"],
            "LISTING": ["listing", "new listing"]
        }
    
    def get_nse_cookies(self) -> Dict[str, str]:
        """Get NSE session cookies"""
        try:
            response = self.session.get("https://www.nseindia.com", timeout=10)
            return dict(response.cookies)
        except:
            return {}
    
    def safe_request(self, url: str, timeout: int = 15, retries: int = 3) -> Optional[requests.Response]:
        """Make safe HTTP request with retries"""
        for attempt in range(retries):
            try:
                # Add random delay
                time.sleep(random.uniform(0.5, 1.5))
                
                response = self.session.get(url, timeout=timeout)
                response.raise_for_status()
                return response
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
                if attempt < retries - 1:
                    time.sleep(random.uniform(2, 4))
                continue
        
        return None
    
    def classify_action_type(self, text: str) -> str:
        """Classify corporate action type from text"""
        text_lower = text.lower()
        
        for action_type, keywords in self.action_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return action_type
        
        return "OTHER"
    
    def determine_urgency(self, action_type: str, event_date: str) -> str:
        """Determine urgency based on action type and timing"""
        try:
            if event_date:
                event_dt = datetime.strptime(event_date, "%Y-%m-%d")
                days_until = (event_dt - datetime.now()).days
                
                if days_until <= 2:
                    return "HIGH"
                elif days_until <= 7:
                    return "MEDIUM"
        except:
            pass
        
        # High urgency actions
        if action_type in ["RESULT", "DIVIDEND", "BONUS", "SPLIT"]:
            return "HIGH"
        elif action_type in ["AGM", "BOARD_MEETING"]:
            return "MEDIUM"
        else:
            return "NORMAL"
    
    def determine_market_impact(self, action_type: str, amount: Optional[float] = None) -> str:
        """Determine potential market impact"""
        high_impact_actions = ["RESULT", "DIVIDEND", "BONUS", "SPLIT", "MERGER", "BUYBACK"]
        medium_impact_actions = ["AGM", "BOARD_MEETING", "RIGHTS"]
        
        if action_type in high_impact_actions:
            return "HIGH"
        elif action_type in medium_impact_actions:
            return "MEDIUM"
        else:
            return "LOW"
    
    def crawl_nse_corporate_actions(self) -> List[CorporateAction]:
        """Crawl NSE corporate actions"""
        logger.info("🔍 Crawling NSE Corporate Actions...")
        actions = []
        
        try:
            # Get NSE cookies first
            cookies = self.get_nse_cookies()
            if cookies:
                self.session.cookies.update(cookies)
            
            # NSE Corporate Actions API
            nse_api_url = "https://www.nseindia.com/api/corporates-corporateActions?index=equities"
            response = self.safe_request(nse_api_url)
            
            if response:
                try:
                    data = response.json()
                    
                    for item in data.get('data', [])[:50]:  # Limit to 50 recent actions
                        action = CorporateAction(
                            company_name=item.get('companyName', ''),
                            symbol=item.get('symbol', ''),
                            exchange="NSE",
                            action_type=self.classify_action_type(item.get('subject', '')),
                            announcement_date=item.get('anmDt', ''),
                            record_date=item.get('recDt'),
                            ex_date=item.get('exDt'),
                            event_date=item.get('bcStDt'),
                            description=item.get('subject', ''),
                            document_url=item.get('attchmntFile'),
                            filing_category="CORPORATE_ACTION",
                            data_source="NSE_API",
                            last_updated=datetime.now().isoformat(),
                            raw_data=item
                        )
                        
                        action.urgency = self.determine_urgency(action.action_type, action.event_date)
                        action.market_impact = self.determine_market_impact(action.action_type)
                        
                        actions.append(action)
                
                except json.JSONDecodeError:
                    logger.warning("Failed to parse NSE corporate actions JSON")
            
            # NSE Board Meetings
            board_meetings_url = "https://www.nseindia.com/api/corporates-board-meetings?index=equities"
            response = self.safe_request(board_meetings_url)
            
            if response:
                try:
                    data = response.json()
                    
                    for item in data.get('data', [])[:30]:  # Limit to 30 recent meetings
                        action = CorporateAction(
                            company_name=item.get('companyName', ''),
                            symbol=item.get('symbol', ''),
                            exchange="NSE",
                            action_type="BOARD_MEETING",
                            announcement_date=item.get('anmDt', ''),
                            event_date=item.get('meetingDt'),
                            description=item.get('meetingPurpose', ''),
                            filing_category="BOARD_MEETING",
                            urgency="MEDIUM",
                            market_impact="MEDIUM",
                            data_source="NSE_API",
                            last_updated=datetime.now().isoformat(),
                            raw_data=item
                        )
                        
                        actions.append(action)
                
                except json.JSONDecodeError:
                    logger.warning("Failed to parse NSE board meetings JSON")
            
            logger.info(f"✅ NSE: Found {len(actions)} corporate actions")
            
        except Exception as e:
            logger.error(f"❌ NSE corporate actions crawling failed: {e}")
        
        return actions
    
    def crawl_bse_corporate_actions(self) -> List[CorporateAction]:
        """Crawl BSE corporate actions"""
        logger.info("🔍 Crawling BSE Corporate Actions...")
        actions = []
        
        try:
            # BSE Corporate Announcements
            bse_url = "https://www.bseindia.com/corporates/ann.html"
            response = self.safe_request(bse_url)
            
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find announcement tables
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    
                    for row in rows[1:21]:  # Limit to 20 recent announcements
                        cells = row.find_all(['td', 'th'])
                        
                        if len(cells) >= 4:
                            company_name = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                            symbol = cells[0].get_text(strip=True) if len(cells) > 0 else ''
                            subject = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                            date = cells[3].get_text(strip=True) if len(cells) > 3 else ''
                            
                            if company_name and subject:
                                action = CorporateAction(
                                    company_name=company_name,
                                    symbol=symbol,
                                    exchange="BSE",
                                    action_type=self.classify_action_type(subject),
                                    announcement_date=date,
                                    description=subject,
                                    filing_category="CORPORATE_ANNOUNCEMENT",
                                    data_source="BSE_WEB",
                                    last_updated=datetime.now().isoformat()
                                )
                                
                                action.urgency = self.determine_urgency(action.action_type, action.event_date)
                                action.market_impact = self.determine_market_impact(action.action_type)
                                
                                actions.append(action)
            
            # BSE Forthcoming Events
            events_url = "https://www.bseindia.com/corporates/Forthcoming_Events.html"
            response = self.safe_request(events_url)
            
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    
                    for row in rows[1:16]:  # Limit to 15 events
                        cells = row.find_all(['td', 'th'])
                        
                        if len(cells) >= 3:
                            company_name = cells[0].get_text(strip=True) if len(cells) > 0 else ''
                            event_type = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                            event_date = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                            
                            if company_name and event_type:
                                action = CorporateAction(
                                    company_name=company_name,
                                    symbol="",
                                    exchange="BSE",
                                    action_type=self.classify_action_type(event_type),
                                    announcement_date=datetime.now().strftime("%Y-%m-%d"),
                                    event_date=event_date,
                                    description=event_type,
                                    filing_category="FORTHCOMING_EVENT",
                                    data_source="BSE_WEB",
                                    last_updated=datetime.now().isoformat()
                                )
                                
                                action.urgency = self.determine_urgency(action.action_type, action.event_date)
                                action.market_impact = self.determine_market_impact(action.action_type)
                                
                                actions.append(action)
            
            logger.info(f"✅ BSE: Found {len(actions)} corporate actions")
            
        except Exception as e:
            logger.error(f"❌ BSE corporate actions crawling failed: {e}")
        
        return actions
    
    def crawl_sebi_filings(self) -> List[CorporateAction]:
        """Crawl SEBI live filings"""
        logger.info("🔍 Crawling SEBI Live Filings...")
        actions = []
        
        try:
            # SEBI Exchange Filings
            sebi_url = "https://www.sebi.gov.in/filings/exchange-filings"
            response = self.safe_request(sebi_url)
            
            if response:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find filing tables
                tables = soup.find_all('table')
                for table in tables:
                    rows = table.find_all('tr')
                    
                    for row in rows[1:11]:  # Limit to 10 recent filings
                        cells = row.find_all(['td', 'th'])
                        
                        if len(cells) >= 3:
                            company_name = cells[0].get_text(strip=True) if len(cells) > 0 else ''
                            filing_type = cells[1].get_text(strip=True) if len(cells) > 1 else ''
                            filing_date = cells[2].get_text(strip=True) if len(cells) > 2 else ''
                            
                            if company_name and filing_type:
                                action = CorporateAction(
                                    company_name=company_name,
                                    symbol="",
                                    exchange="SEBI",
                                    action_type=self.classify_action_type(filing_type),
                                    announcement_date=filing_date,
                                    description=filing_type,
                                    filing_category="SEBI_FILING",
                                    urgency="HIGH",  # SEBI filings are usually important
                                    market_impact="HIGH",
                                    data_source="SEBI_WEB",
                                    last_updated=datetime.now().isoformat()
                                )
                                
                                actions.append(action)
            
            logger.info(f"✅ SEBI: Found {len(actions)} filings")
            
        except Exception as e:
            logger.error(f"❌ SEBI filings crawling failed: {e}")
        
        return actions
    
    def deduplicate_actions(self, actions: List[CorporateAction]) -> List[CorporateAction]:
        """Remove duplicate corporate actions"""
        seen_actions = set()
        unique_actions = []
        
        for action in actions:
            # Create unique key
            key = (
                action.company_name.lower().strip(),
                action.symbol.lower().strip(),
                action.action_type,
                action.announcement_date,
                action.description.lower().strip()[:50]  # First 50 chars
            )
            
            if key not in seen_actions:
                seen_actions.add(key)
                unique_actions.append(action)
        
        logger.info(f"🔄 Deduplication: {len(actions)} → {len(unique_actions)} actions")
        return unique_actions
    
    def crawl_all_corporate_actions(self) -> List[CorporateAction]:
        """Crawl all corporate action sources"""
        logger.info("🚀 Starting Live Corporate Actions Crawling...")
        
        all_actions = []
        
        # Crawl NSE
        nse_actions = self.crawl_nse_corporate_actions()
        all_actions.extend(nse_actions)
        time.sleep(random.uniform(2, 4))
        
        # Crawl BSE
        bse_actions = self.crawl_bse_corporate_actions()
        all_actions.extend(bse_actions)
        time.sleep(random.uniform(2, 4))
        
        # Crawl SEBI
        sebi_actions = self.crawl_sebi_filings()
        all_actions.extend(sebi_actions)
        
        # Deduplicate
        unique_actions = self.deduplicate_actions(all_actions)
        
        # Sort by urgency and date
        unique_actions.sort(key=lambda x: (
            {"HIGH": 0, "MEDIUM": 1, "NORMAL": 2, "LOW": 3}.get(x.urgency, 3),
            x.announcement_date
        ), reverse=True)
        
        self.corporate_actions = unique_actions
        return unique_actions
    
    def analyze_market_impact(self, actions: List[CorporateAction]) -> Dict:
        """Analyze potential market impact of corporate actions"""
        analysis = {
            "high_impact_actions": [],
            "urgent_actions": [],
            "dividend_announcements": [],
            "result_announcements": [],
            "agm_meetings": [],
            "action_summary": {},
            "exchange_summary": {},
            "urgency_summary": {}
        }
        
        for action in actions:
            # High impact actions
            if action.market_impact == "HIGH":
                analysis["high_impact_actions"].append(action)
            
            # Urgent actions
            if action.urgency == "HIGH":
                analysis["urgent_actions"].append(action)
            
            # Specific action types
            if action.action_type == "DIVIDEND":
                analysis["dividend_announcements"].append(action)
            elif action.action_type == "RESULT":
                analysis["result_announcements"].append(action)
            elif action.action_type == "AGM":
                analysis["agm_meetings"].append(action)
            
            # Summaries
            analysis["action_summary"][action.action_type] = analysis["action_summary"].get(action.action_type, 0) + 1
            analysis["exchange_summary"][action.exchange] = analysis["exchange_summary"].get(action.exchange, 0) + 1
            analysis["urgency_summary"][action.urgency] = analysis["urgency_summary"].get(action.urgency, 0) + 1
        
        return analysis
    
    def save_corporate_actions(self, actions: List[CorporateAction]) -> str:
        """Save corporate actions to JSON file"""
        output_file = "market_data/live_corporate_actions.json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Convert to dictionaries
        actions_dict = []
        for action in actions:
            action_dict = asdict(action)
            if action_dict['raw_data'] is None:
                action_dict['raw_data'] = {}
            actions_dict.append(action_dict)
        
        # Analyze market impact
        analysis = self.analyze_market_impact(actions)
        
        # Create metadata
        metadata = {
            "total_actions": len(actions),
            "crawling_timestamp": datetime.now().isoformat(),
            "data_sources": ["NSE_API", "BSE_WEB", "SEBI_WEB"],
            "action_types_covered": list(self.action_keywords.keys()),
            "high_impact_count": len(analysis["high_impact_actions"]),
            "urgent_count": len(analysis["urgent_actions"]),
            "dividend_count": len(analysis["dividend_announcements"]),
            "result_count": len(analysis["result_announcements"]),
            "agm_count": len(analysis["agm_meetings"]),
            "exchange_distribution": analysis["exchange_summary"],
            "urgency_distribution": analysis["urgency_summary"],
            "action_type_distribution": analysis["action_summary"]
        }
        
        data = {
            "metadata": metadata,
            "corporate_actions": actions_dict,
            "market_impact_analysis": analysis
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved corporate actions to {output_file}")
        return output_file
    
    def display_live_analysis(self, actions: List[CorporateAction]):
        """Display live analysis of corporate actions"""
        analysis = self.analyze_market_impact(actions)
        
        logger.info(f"\n📊 Live Corporate Actions Analysis:")
        logger.info(f"   Total Actions: {len(actions)}")
        logger.info(f"   High Impact: {len(analysis['high_impact_actions'])}")
        logger.info(f"   Urgent: {len(analysis['urgent_actions'])}")
        
        logger.info(f"\n🎯 Action Type Distribution:")
        for action_type, count in sorted(analysis["action_summary"].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {action_type}: {count}")
        
        logger.info(f"\n🏢 Exchange Distribution:")
        for exchange, count in sorted(analysis["exchange_summary"].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {exchange}: {count}")
        
        logger.info(f"\n⚡ Urgency Distribution:")
        for urgency, count in sorted(analysis["urgency_summary"].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {urgency}: {count}")
        
        # Show top urgent actions
        if analysis["urgent_actions"]:
            logger.info(f"\n🚨 Top Urgent Actions:")
            for i, action in enumerate(analysis["urgent_actions"][:5], 1):
                logger.info(f"   {i}. {action.company_name} ({action.symbol}) - {action.action_type}")
                logger.info(f"      {action.description}")
                logger.info(f"      Exchange: {action.exchange} | Date: {action.announcement_date}")
        
        # Show recent results
        if analysis["result_announcements"]:
            logger.info(f"\n📈 Recent Result Announcements:")
            for i, action in enumerate(analysis["result_announcements"][:3], 1):
                logger.info(f"   {i}. {action.company_name} ({action.symbol})")
                logger.info(f"      {action.description}")
        
        # Show upcoming AGMs
        if analysis["agm_meetings"]:
            logger.info(f"\n🏛️ Upcoming AGM Meetings:")
            for i, action in enumerate(analysis["agm_meetings"][:3], 1):
                logger.info(f"   {i}. {action.company_name} ({action.symbol})")
                logger.info(f"      Date: {action.event_date}")
    
    def run_live_analysis(self) -> str:
        """Run complete live corporate actions analysis"""
        logger.info("🚀 Starting Live Corporate Actions Analysis...")
        
        # Crawl all sources
        actions = self.crawl_all_corporate_actions()
        
        # Display analysis
        self.display_live_analysis(actions)
        
        # Save data
        output_file = self.save_corporate_actions(actions)
        
        logger.info(f"\n🎉 LIVE CORPORATE ACTIONS ANALYSIS COMPLETE!")
        logger.info(f"   ✅ {len(actions)} corporate actions analyzed")
        logger.info(f"   ✅ Real-time market impact assessment")
        logger.info(f"   ✅ Multi-source data integration (NSE + BSE + SEBI)")
        logger.info(f"   📁 Data saved to: {output_file}")
        
        return output_file

def main():
    """Main function"""
    crawler = LiveCorporateActionsCrawler()
    crawler.run_live_analysis()

if __name__ == "__main__":
    main()
