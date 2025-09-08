"""
Specialized Data Sources for BSE SME and NSE Emerge Platforms
Handles unique data requirements for small and emerging companies
"""
import aiohttp
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import re
from bs4 import BeautifulSoup
from dataclasses import dataclass

@dataclass
class SMECompanyData:
    symbol: str
    name: str
    exchange: str
    price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    book_value: Optional[float] = None
    face_value: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class NSEEmergeDataSource:
    """Data source for NSE Emerge platform companies"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com"
        self.emerge_api_url = f"{self.base_url}/api/emerge-quote-equity"
        self.emerge_list_url = f"{self.base_url}/api/emerge-equity-list"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for NSE requests"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.nseindia.com/',
            'X-Requested-With': 'XMLHttpRequest'
        }
    
    async def get_emerge_companies_list(self) -> List[Dict[str, Any]]:
        """Get list of all NSE Emerge companies"""
        try:
            headers = self.get_headers()
            
            async with self.session.get(self.emerge_list_url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('data', [])
                else:
                    logging.error(f"Failed to fetch NSE Emerge list: {response.status}")
                    return []
                    
        except Exception as e:
            logging.error(f"Error fetching NSE Emerge companies: {e}")
            return []
    
    async def get_emerge_company_data(self, symbol: str) -> Optional[SMECompanyData]:
        """Get detailed data for NSE Emerge company"""
        try:
            headers = self.get_headers()
            url = f"{self.emerge_api_url}?symbol={symbol}"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_emerge_data(symbol, data)
                else:
                    logging.warning(f"Failed to fetch data for {symbol}: {response.status}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _parse_emerge_data(self, symbol: str, data: Dict[str, Any]) -> SMECompanyData:
        """Parse NSE Emerge API response"""
        price_info = data.get('priceInfo', {})
        security_info = data.get('securityInfo', {})
        
        return SMECompanyData(
            symbol=symbol,
            name=security_info.get('companyName', ''),
            exchange="NSE_EMERGE",
            price=float(price_info.get('lastPrice', 0)) if price_info.get('lastPrice') else None,
            change=float(price_info.get('change', 0)) if price_info.get('change') else None,
            change_percent=float(price_info.get('pChange', 0)) if price_info.get('pChange') else None,
            volume=int(data.get('securityWiseDP', {}).get('quantityTraded', 0)) if data.get('securityWiseDP', {}).get('quantityTraded') else None,
            pe_ratio=float(price_info.get('pe', 0)) if price_info.get('pe') else None,
            face_value=float(security_info.get('faceValue', 0)) if security_info.get('faceValue') else None,
            high_52w=float(price_info.get('weekHighLow', {}).get('max', 0)) if price_info.get('weekHighLow', {}).get('max') else None,
            low_52w=float(price_info.get('weekHighLow', {}).get('min', 0)) if price_info.get('weekHighLow', {}).get('min') else None,
            timestamp=datetime.now(),
            metadata={
                'listing_date': security_info.get('listingDate'),
                'industry': security_info.get('industry'),
                'isin': security_info.get('isin'),
                'series': security_info.get('series')
            }
        )

class BSESMEDataSource:
    """Data source for BSE SME platform companies"""
    
    def __init__(self):
        self.base_url = "https://www.bseindia.com"
        self.sme_api_url = f"{self.base_url}/xml-data/corpfiling/AttachLive/SME_Live_quote.xml"
        self.sme_list_url = f"{self.base_url}/corporates/List_Scrips.aspx?expandable=7"
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def get_headers(self) -> Dict[str, str]:
        """Get headers for BSE requests"""
        return {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.bseindia.com/'
        }
    
    async def get_sme_companies_list(self) -> List[Dict[str, Any]]:
        """Get list of all BSE SME companies"""
        try:
            headers = self.get_headers()
            
            async with self.session.get(self.sme_list_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_sme_list(html)
                else:
                    logging.error(f"Failed to fetch BSE SME list: {response.status}")
                    return []
                    
        except Exception as e:
            logging.error(f"Error fetching BSE SME companies: {e}")
            return []
    
    def _parse_sme_list(self, html: str) -> List[Dict[str, Any]]:
        """Parse BSE SME companies list from HTML"""
        companies = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find the SME section table
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                
                for row in rows[1:]:  # Skip header row
                    cells = row.find_all('td')
                    
                    if len(cells) >= 3:
                        companies.append({
                            'scrip_code': cells[0].get_text(strip=True),
                            'company_name': cells[1].get_text(strip=True),
                            'symbol': cells[2].get_text(strip=True) if len(cells) > 2 else None
                        })
                        
        except Exception as e:
            logging.error(f"Error parsing BSE SME list: {e}")
            
        return companies
    
    async def get_sme_company_data(self, scrip_code: str) -> Optional[SMECompanyData]:
        """Get detailed data for BSE SME company"""
        try:
            headers = self.get_headers()
            url = f"{self.base_url}/stock-share-price/{scrip_code}/live-bse-stock-price"
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_sme_data(scrip_code, html)
                else:
                    logging.warning(f"Failed to fetch data for {scrip_code}: {response.status}")
                    return None
                    
        except Exception as e:
            logging.error(f"Error fetching data for {scrip_code}: {e}")
            return None
    
    def _parse_sme_data(self, scrip_code: str, html: str) -> Optional[SMECompanyData]:
        """Parse BSE SME company data from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract company name
            name_element = soup.find('h1')
            company_name = name_element.get_text(strip=True) if name_element else ''
            
            # Extract price data
            price_data = {}
            
            # Look for price information in various formats
            price_elements = soup.find_all(['span', 'div'], class_=re.compile(r'price|value|amount'))
            
            for element in price_elements:
                text = element.get_text(strip=True)
                
                # Try to extract numeric values
                numbers = re.findall(r'[\d,]+\.?\d*', text)
                if numbers:
                    try:
                        value = float(numbers[0].replace(',', ''))
                        
                        if 'price' in element.get('class', []):
                            price_data['price'] = value
                        elif 'change' in element.get('class', []):
                            price_data['change'] = value
                        elif 'volume' in element.get('class', []):
                            price_data['volume'] = int(value)
                            
                    except ValueError:
                        continue
            
            return SMECompanyData(
                symbol=scrip_code,
                name=company_name,
                exchange="BSE_SME",
                price=price_data.get('price'),
                change=price_data.get('change'),
                volume=price_data.get('volume'),
                timestamp=datetime.now(),
                metadata={
                    'scrip_code': scrip_code,
                    'source_url': f"{self.base_url}/stock-share-price/{scrip_code}/live-bse-stock-price"
                }
            )
            
        except Exception as e:
            logging.error(f"Error parsing BSE SME data for {scrip_code}: {e}")
            return None

class SMENewsSource:
    """Specialized news source for SME and emerging companies"""
    
    def __init__(self):
        self.news_sources = {
            'sme_world': 'https://www.smeworld.org',
            'sme_times': 'https://www.smetimes.in',
            'business_standard_sme': 'https://www.business-standard.com/topic/sme'
        }
        self.session = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_sme_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get news specific to SME companies"""
        news_items = []
        
        for source_name, base_url in self.news_sources.items():
            try:
                items = await self._fetch_news_from_source(source_name, base_url, symbol)
                news_items.extend(items)
            except Exception as e:
                logging.error(f"Error fetching news from {source_name}: {e}")
                
        return news_items
    
    async def _fetch_news_from_source(self, source_name: str, base_url: str, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news from a specific source"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Construct search URL (this would be customized per source)
        search_url = f"{base_url}/search?q={symbol}"
        
        try:
            async with self.session.get(search_url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    return self._parse_news_html(html, source_name, symbol)
                else:
                    logging.warning(f"Failed to fetch news from {source_name}: {response.status}")
                    return []
                    
        except Exception as e:
            logging.error(f"Error fetching from {source_name}: {e}")
            return []
    
    def _parse_news_html(self, html: str, source_name: str, symbol: str) -> List[Dict[str, Any]]:
        """Parse news HTML and extract articles"""
        news_items = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Look for article elements (this would be customized per source)
            articles = soup.find_all(['article', 'div'], class_=re.compile(r'news|article|story'))
            
            for article in articles[:10]:  # Limit to 10 articles
                title_element = article.find(['h1', 'h2', 'h3', 'h4'])
                link_element = article.find('a')
                
                if title_element and link_element:
                    title = title_element.get_text(strip=True)
                    link = link_element.get('href', '')
                    
                    # Make link absolute if needed
                    if link.startswith('/'):
                        link = f"{self.news_sources[source_name]}{link}"
                    
                    news_items.append({
                        'title': title,
                        'url': link,
                        'source': source_name,
                        'symbol': symbol,
                        'timestamp': datetime.now().isoformat(),
                        'content': title  # Summary for now
                    })
                    
        except Exception as e:
            logging.error(f"Error parsing news HTML from {source_name}: {e}")
            
        return news_items

class SMEEmergeDataAggregator:
    """Aggregator for all SME and Emerge data sources"""
    
    def __init__(self):
        self.nse_emerge = NSEEmergeDataSource()
        self.bse_sme = BSESMEDataSource()
        self.sme_news = SMENewsSource()
        
    async def __aenter__(self):
        """Initialize all data sources"""
        await self.nse_emerge.__aenter__()
        await self.bse_sme.__aenter__()
        await self.sme_news.__aenter__()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup all data sources"""
        await self.nse_emerge.__aexit__(exc_type, exc_val, exc_tb)
        await self.bse_sme.__aexit__(exc_type, exc_val, exc_tb)
        await self.sme_news.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_comprehensive_sme_data(self, symbol: str, exchange: str = None) -> Dict[str, Any]:
        """Get comprehensive data for SME/Emerge company"""
        result = {
            'symbol': symbol,
            'exchange': exchange,
            'price_data': None,
            'news': [],
            'metadata': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get price data based on exchange
            if exchange == 'NSE_EMERGE' or exchange is None:
                nse_data = await self.nse_emerge.get_emerge_company_data(symbol)
                if nse_data:
                    result['price_data'] = nse_data
                    result['exchange'] = 'NSE_EMERGE'
            
            if exchange == 'BSE_SME' or (exchange is None and result['price_data'] is None):
                bse_data = await self.bse_sme.get_sme_company_data(symbol)
                if bse_data:
                    result['price_data'] = bse_data
                    result['exchange'] = 'BSE_SME'
            
            # Get news data
            news_items = await self.sme_news.get_sme_news(symbol)
            result['news'] = news_items
            
            # Add metadata
            result['metadata'] = {
                'data_sources': ['NSE_EMERGE', 'BSE_SME', 'SME_NEWS'],
                'last_updated': datetime.now().isoformat(),
                'data_quality': 'high' if result['price_data'] else 'partial'
            }
            
        except Exception as e:
            logging.error(f"Error aggregating data for {symbol}: {e}")
            result['metadata']['error'] = str(e)
        
        return result
    
    async def get_all_sme_emerge_companies(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get list of all SME and Emerge companies"""
        result = {
            'nse_emerge': [],
            'bse_sme': [],
            'total_count': 0
        }
        
        try:
            # Get NSE Emerge companies
            nse_companies = await self.nse_emerge.get_emerge_companies_list()
            result['nse_emerge'] = nse_companies
            
            # Get BSE SME companies
            bse_companies = await self.bse_sme.get_sme_companies_list()
            result['bse_sme'] = bse_companies
            
            result['total_count'] = len(nse_companies) + len(bse_companies)
            
        except Exception as e:
            logging.error(f"Error fetching SME/Emerge companies list: {e}")
            result['error'] = str(e)
        
        return result

# Global instance
sme_emerge_aggregator = SMEEmergeDataAggregator()
