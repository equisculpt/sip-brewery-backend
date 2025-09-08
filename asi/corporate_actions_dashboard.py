#!/usr/bin/env python3
"""
Corporate Actions Live Dashboard
Real-time dashboard for viewing AGM, Results, Dividends, and Corporate Announcements
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import os
from live_corporate_actions_crawler import LiveCorporateActionsCrawler, CorporateAction

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CorporateActionsDashboard:
    """Live dashboard for corporate actions"""
    
    def __init__(self):
        self.crawler = LiveCorporateActionsCrawler()
        self.actions_file = "market_data/live_corporate_actions.json"
        self.dashboard_data = {}
    
    def load_latest_actions(self) -> List[CorporateAction]:
        """Load latest corporate actions"""
        try:
            with open(self.actions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            actions = []
            for action_dict in data.get('corporate_actions', []):
                action = CorporateAction(
                    company_name=action_dict['company_name'],
                    symbol=action_dict['symbol'],
                    exchange=action_dict['exchange'],
                    action_type=action_dict['action_type'],
                    announcement_date=action_dict['announcement_date'],
                    record_date=action_dict.get('record_date'),
                    ex_date=action_dict.get('ex_date'),
                    event_date=action_dict.get('event_date'),
                    description=action_dict['description'],
                    amount=action_dict.get('amount'),
                    percentage=action_dict.get('percentage'),
                    document_url=action_dict.get('document_url'),
                    filing_category=action_dict['filing_category'],
                    urgency=action_dict['urgency'],
                    market_impact=action_dict['market_impact'],
                    data_source=action_dict['data_source'],
                    last_updated=action_dict['last_updated'],
                    raw_data=action_dict.get('raw_data', {})
                )
                actions.append(action)
            
            return actions
            
        except FileNotFoundError:
            logger.warning("No corporate actions data found. Running fresh crawl...")
            return self.crawler.crawl_all_corporate_actions()
    
    def create_dashboard_summary(self, actions: List[CorporateAction]) -> Dict:
        """Create dashboard summary"""
        now = datetime.now()
        today = now.date()
        
        summary = {
            "total_actions": len(actions),
            "last_updated": now.isoformat(),
            "today_actions": 0,
            "urgent_actions": 0,
            "high_impact_actions": 0,
            "upcoming_events": 0,
            "recent_results": 0,
            "dividend_announcements": 0,
            "agm_meetings": 0
        }
        
        for action in actions:
            # Today's actions
            try:
                action_date = datetime.strptime(action.announcement_date, "%Y-%m-%d").date()
                if action_date == today:
                    summary["today_actions"] += 1
            except:
                pass
            
            # Urgent actions
            if action.urgency == "HIGH":
                summary["urgent_actions"] += 1
            
            # High impact actions
            if action.market_impact == "HIGH":
                summary["high_impact_actions"] += 1
            
            # Upcoming events (within 7 days)
            if action.event_date:
                try:
                    event_date = datetime.strptime(action.event_date, "%Y-%m-%d").date()
                    if today <= event_date <= today + timedelta(days=7):
                        summary["upcoming_events"] += 1
                except:
                    pass
            
            # Action type counts
            if action.action_type == "RESULT":
                summary["recent_results"] += 1
            elif action.action_type == "DIVIDEND":
                summary["dividend_announcements"] += 1
            elif action.action_type == "AGM":
                summary["agm_meetings"] += 1
        
        return summary
    
    def get_urgent_actions(self, actions: List[CorporateAction]) -> List[CorporateAction]:
        """Get urgent actions"""
        urgent = [action for action in actions if action.urgency == "HIGH"]
        return sorted(urgent, key=lambda x: x.announcement_date, reverse=True)[:10]
    
    def get_todays_actions(self, actions: List[CorporateAction]) -> List[CorporateAction]:
        """Get today's actions"""
        today = datetime.now().date()
        todays = []
        
        for action in actions:
            try:
                action_date = datetime.strptime(action.announcement_date, "%Y-%m-%d").date()
                if action_date == today:
                    todays.append(action)
            except:
                pass
        
        return sorted(todays, key=lambda x: x.last_updated, reverse=True)
    
    def get_upcoming_events(self, actions: List[CorporateAction]) -> List[CorporateAction]:
        """Get upcoming events (next 7 days)"""
        today = datetime.now().date()
        upcoming = []
        
        for action in actions:
            if action.event_date:
                try:
                    event_date = datetime.strptime(action.event_date, "%Y-%m-%d").date()
                    if today <= event_date <= today + timedelta(days=7):
                        upcoming.append(action)
                except:
                    pass
        
        return sorted(upcoming, key=lambda x: x.event_date)[:15]
    
    def get_recent_results(self, actions: List[CorporateAction]) -> List[CorporateAction]:
        """Get recent quarterly results"""
        results = [action for action in actions if action.action_type == "RESULT"]
        return sorted(results, key=lambda x: x.announcement_date, reverse=True)[:10]
    
    def get_dividend_announcements(self, actions: List[CorporateAction]) -> List[CorporateAction]:
        """Get recent dividend announcements"""
        dividends = [action for action in actions if action.action_type == "DIVIDEND"]
        return sorted(dividends, key=lambda x: x.announcement_date, reverse=True)[:10]
    
    def get_exchange_breakdown(self, actions: List[CorporateAction]) -> Dict:
        """Get breakdown by exchange"""
        breakdown = {}
        
        for action in actions:
            exchange = action.exchange
            if exchange not in breakdown:
                breakdown[exchange] = {
                    "total": 0,
                    "urgent": 0,
                    "high_impact": 0,
                    "action_types": {}
                }
            
            breakdown[exchange]["total"] += 1
            
            if action.urgency == "HIGH":
                breakdown[exchange]["urgent"] += 1
            
            if action.market_impact == "HIGH":
                breakdown[exchange]["high_impact"] += 1
            
            action_type = action.action_type
            if action_type not in breakdown[exchange]["action_types"]:
                breakdown[exchange]["action_types"][action_type] = 0
            breakdown[exchange]["action_types"][action_type] += 1
        
        return breakdown
    
    def display_dashboard(self):
        """Display comprehensive dashboard"""
        logger.info("🚀 Loading Corporate Actions Dashboard...")
        
        # Load latest actions
        actions = self.load_latest_actions()
        
        # Create summary
        summary = self.create_dashboard_summary(actions)
        
        # Display header
        logger.info(f"\n" + "="*80)
        logger.info(f"📊 CORPORATE ACTIONS LIVE DASHBOARD")
        logger.info(f"   Last Updated: {summary['last_updated']}")
        logger.info(f"="*80)
        
        # Display summary
        logger.info(f"\n📈 SUMMARY:")
        logger.info(f"   Total Actions: {summary['total_actions']:,}")
        logger.info(f"   Today's Actions: {summary['today_actions']}")
        logger.info(f"   Urgent Actions: {summary['urgent_actions']}")
        logger.info(f"   High Impact: {summary['high_impact_actions']}")
        logger.info(f"   Upcoming Events (7 days): {summary['upcoming_events']}")
        logger.info(f"   Recent Results: {summary['recent_results']}")
        logger.info(f"   Dividend Announcements: {summary['dividend_announcements']}")
        logger.info(f"   AGM Meetings: {summary['agm_meetings']}")
        
        # Exchange breakdown
        exchange_breakdown = self.get_exchange_breakdown(actions)
        logger.info(f"\n🏢 EXCHANGE BREAKDOWN:")
        for exchange, data in sorted(exchange_breakdown.items()):
            logger.info(f"   {exchange}: {data['total']} actions ({data['urgent']} urgent, {data['high_impact']} high impact)")
        
        # Today's actions
        todays_actions = self.get_todays_actions(actions)
        if todays_actions:
            logger.info(f"\n🔥 TODAY'S ACTIONS ({len(todays_actions)}):")
            for i, action in enumerate(todays_actions[:5], 1):
                logger.info(f"   {i}. {action.company_name} ({action.symbol}) - {action.action_type}")
                logger.info(f"      {action.description}")
                logger.info(f"      Exchange: {action.exchange} | Urgency: {action.urgency}")
        
        # Urgent actions
        urgent_actions = self.get_urgent_actions(actions)
        if urgent_actions:
            logger.info(f"\n🚨 URGENT ACTIONS ({len(urgent_actions)}):")
            for i, action in enumerate(urgent_actions[:5], 1):
                logger.info(f"   {i}. {action.company_name} ({action.symbol}) - {action.action_type}")
                logger.info(f"      {action.description}")
                logger.info(f"      Date: {action.announcement_date} | Impact: {action.market_impact}")
        
        # Upcoming events
        upcoming_events = self.get_upcoming_events(actions)
        if upcoming_events:
            logger.info(f"\n📅 UPCOMING EVENTS (Next 7 Days):")
            for i, action in enumerate(upcoming_events[:5], 1):
                logger.info(f"   {i}. {action.company_name} ({action.symbol}) - {action.action_type}")
                logger.info(f"      Event Date: {action.event_date}")
                logger.info(f"      Description: {action.description}")
        
        # Recent results
        recent_results = self.get_recent_results(actions)
        if recent_results:
            logger.info(f"\n📊 RECENT QUARTERLY RESULTS:")
            for i, action in enumerate(recent_results[:5], 1):
                logger.info(f"   {i}. {action.company_name} ({action.symbol})")
                logger.info(f"      Date: {action.announcement_date}")
                logger.info(f"      Description: {action.description}")
        
        # Dividend announcements
        dividend_announcements = self.get_dividend_announcements(actions)
        if dividend_announcements:
            logger.info(f"\n💰 RECENT DIVIDEND ANNOUNCEMENTS:")
            for i, action in enumerate(dividend_announcements[:5], 1):
                logger.info(f"   {i}. {action.company_name} ({action.symbol})")
                logger.info(f"      Date: {action.announcement_date}")
                logger.info(f"      Description: {action.description}")
                if action.record_date:
                    logger.info(f"      Record Date: {action.record_date}")
        
        logger.info(f"\n" + "="*80)
        logger.info(f"🎯 Dashboard Complete - {summary['total_actions']} actions analyzed")
        logger.info(f"="*80)
    
    def refresh_and_display(self):
        """Refresh data and display dashboard"""
        logger.info("🔄 Refreshing corporate actions data...")
        
        # Run fresh crawl
        fresh_actions = self.crawler.crawl_all_corporate_actions()
        
        # Save fresh data
        self.crawler.save_corporate_actions(fresh_actions)
        
        # Display dashboard
        self.display_dashboard()
    
    def search_actions(self, query: str) -> List[CorporateAction]:
        """Search corporate actions"""
        actions = self.load_latest_actions()
        query_lower = query.lower()
        
        matching_actions = []
        for action in actions:
            if (query_lower in action.company_name.lower() or
                query_lower in action.symbol.lower() or
                query_lower in action.description.lower() or
                query_lower in action.action_type.lower()):
                matching_actions.append(action)
        
        return matching_actions
    
    def display_search_results(self, query: str):
        """Display search results"""
        results = self.search_actions(query)
        
        logger.info(f"\n🔍 Search Results for '{query}': {len(results)} found")
        
        if results:
            for i, action in enumerate(results[:10], 1):
                logger.info(f"\n   {i}. {action.company_name} ({action.symbol}) - {action.action_type}")
                logger.info(f"      Exchange: {action.exchange}")
                logger.info(f"      Date: {action.announcement_date}")
                logger.info(f"      Description: {action.description}")
                logger.info(f"      Urgency: {action.urgency} | Impact: {action.market_impact}")
                if action.event_date:
                    logger.info(f"      Event Date: {action.event_date}")
        else:
            logger.info("   No matching actions found.")

def main():
    """Main function"""
    dashboard = CorporateActionsDashboard()
    
    # Display dashboard
    dashboard.refresh_and_display()
    
    # Example searches
    logger.info(f"\n" + "="*80)
    logger.info(f"🔍 EXAMPLE SEARCHES:")
    logger.info(f"="*80)
    
    # Search for specific companies
    search_queries = ["RELIANCE", "TCS", "HDFC", "DIVIDEND", "RESULT", "AGM"]
    
    for query in search_queries:
        dashboard.display_search_results(query)

if __name__ == "__main__":
    main()
