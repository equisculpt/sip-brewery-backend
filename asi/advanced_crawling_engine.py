"""
Advanced Crawling Engine for ASI Finance Search
Intelligent rate limiting, anti-bot bypass, parallel crawling, content deduplication
"""
import asyncio
import aiohttp
import hashlib
import time
import random
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import json
from collections import defaultdict, deque
import heapq

@dataclass
class CrawlTask:
    url: str
    priority: int
    symbol: str
    source_type: str
    retry_count: int = 0
    scheduled_time: float = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __lt__(self, other):
        return self.priority < other.priority

@dataclass
class CrawlResult:
    url: str
    content: str
    status_code: int
    headers: Dict[str, str]
    timestamp: datetime
    content_hash: str
    source_type: str
    symbol: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class RateLimiter:
    """Intelligent rate limiter with domain-specific limits"""
    
    def __init__(self):
        self.domain_limits = {
            'nseindia.com': {'requests_per_second': 0.5, 'burst': 2},
            'bseindia.com': {'requests_per_second': 0.8, 'burst': 3},
            'moneycontrol.com': {'requests_per_second': 1.0, 'burst': 5},
            'economictimes.indiatimes.com': {'requests_per_second': 1.2, 'burst': 4},
            'reuters.com': {'requests_per_second': 0.8, 'burst': 3},
            'bloomberg.com': {'requests_per_second': 0.6, 'burst': 2},
            'yahoo.com': {'requests_per_second': 1.5, 'burst': 6},
            'default': {'requests_per_second': 1.0, 'burst': 3}
        }
        self.domain_queues = defaultdict(deque)
        self.last_request_time = defaultdict(float)
        
    async def acquire(self, url: str) -> None:
        """Acquire permission to make request with intelligent rate limiting"""
        domain = urlparse(url).netloc
        limits = self.domain_limits.get(domain, self.domain_limits['default'])
        
        current_time = time.time()
        last_time = self.last_request_time[domain]
        
        # Calculate required delay
        min_interval = 1.0 / limits['requests_per_second']
        time_since_last = current_time - last_time
        
        if time_since_last < min_interval:
            delay = min_interval - time_since_last
            # Add jitter to avoid thundering herd
            delay += random.uniform(0, 0.5)
            await asyncio.sleep(delay)
            
        self.last_request_time[domain] = time.time()

class UserAgentRotator:
    """Rotate user agents to avoid detection"""
    
    def __init__(self):
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        self.current_index = 0
        
    def get_headers(self, url: str) -> Dict[str, str]:
        """Get headers with rotated user agent and domain-specific customization"""
        domain = urlparse(url).netloc
        
        headers = {
            'User-Agent': self.user_agents[self.current_index],
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Domain-specific headers
        if 'nseindia.com' in domain:
            headers.update({
                'Referer': 'https://www.nseindia.com/',
                'X-Requested-With': 'XMLHttpRequest'
            })
        elif 'moneycontrol.com' in domain:
            headers.update({
                'Referer': 'https://www.moneycontrol.com/'
            })
            
        self.current_index = (self.current_index + 1) % len(self.user_agents)
        return headers

class ContentDeduplicator:
    """Deduplicate content using hashing and similarity detection"""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.content_hashes: Set[str] = set()
        self.similarity_threshold = similarity_threshold
        self.content_cache: Dict[str, str] = {}
        
    def is_duplicate(self, content: str, url: str) -> bool:
        """Check if content is duplicate"""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if content_hash in self.content_hashes:
            return True
            
        # Check for near-duplicates using content length and first/last N chars
        content_signature = self._generate_signature(content)
        
        for existing_hash, existing_sig in self.content_cache.items():
            if self._calculate_similarity(content_signature, existing_sig) > self.similarity_threshold:
                return True
                
        # Store new content
        self.content_hashes.add(content_hash)
        self.content_cache[content_hash] = content_signature
        
        return False
    
    def _generate_signature(self, content: str) -> str:
        """Generate content signature for similarity comparison"""
        # Remove whitespace and normalize
        normalized = ' '.join(content.split())
        
        # Take first and last 200 chars + length
        if len(normalized) > 400:
            signature = normalized[:200] + str(len(normalized)) + normalized[-200:]
        else:
            signature = normalized
            
        return signature
    
    def _calculate_similarity(self, sig1: str, sig2: str) -> float:
        """Calculate similarity between two signatures"""
        if len(sig1) == 0 or len(sig2) == 0:
            return 0.0
            
        # Simple similarity based on common characters
        common = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return common / max(len(sig1), len(sig2))

class QualityScorer:
    """Score content quality for ranking and filtering"""
    
    def __init__(self):
        self.quality_indicators = {
            'financial_keywords': ['revenue', 'profit', 'earnings', 'dividend', 'market cap', 'pe ratio'],
            'news_indicators': ['announced', 'reported', 'said', 'according to', 'sources'],
            'data_indicators': ['₹', '%', 'crore', 'lakh', 'million', 'billion'],
            'spam_indicators': ['click here', 'subscribe now', 'advertisement', 'sponsored']
        }
        
    def score_content(self, content: str, url: str, source_type: str) -> float:
        """Score content quality from 0.0 to 1.0"""
        score = 0.5  # Base score
        
        content_lower = content.lower()
        
        # Financial relevance
        financial_count = sum(1 for keyword in self.quality_indicators['financial_keywords'] 
                            if keyword in content_lower)
        score += min(financial_count * 0.1, 0.3)
        
        # News quality indicators
        news_count = sum(1 for indicator in self.quality_indicators['news_indicators']
                        if indicator in content_lower)
        score += min(news_count * 0.05, 0.2)
        
        # Data presence
        data_count = sum(1 for indicator in self.quality_indicators['data_indicators']
                        if indicator in content)
        score += min(data_count * 0.05, 0.15)
        
        # Spam detection (negative scoring)
        spam_count = sum(1 for spam in self.quality_indicators['spam_indicators']
                        if spam in content_lower)
        score -= spam_count * 0.1
        
        # Content length scoring
        if 100 < len(content) < 10000:
            score += 0.1
        elif len(content) < 50:
            score -= 0.2
            
        # Source type bonus
        source_bonuses = {
            'official': 0.2,
            'news': 0.1,
            'regulatory': 0.15,
            'research': 0.1
        }
        score += source_bonuses.get(source_type, 0)
        
        return max(0.0, min(1.0, score))

class AdvancedCrawlingEngine:
    """Main crawling engine with all advanced features"""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter()
        self.user_agent_rotator = UserAgentRotator()
        self.content_deduplicator = ContentDeduplicator()
        self.quality_scorer = QualityScorer()
        
        # Task management
        self.task_queue = []
        self.active_tasks: Set[str] = set()
        self.completed_tasks: Dict[str, CrawlResult] = {}
        self.failed_tasks: Dict[str, int] = defaultdict(int)
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'duplicate_content': 0,
            'start_time': time.time()
        }
        
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            
    def add_crawl_task(self, url: str, symbol: str, source_type: str, priority: int = 5, metadata: Dict[str, Any] = None):
        """Add a crawl task to the queue"""
        task = CrawlTask(
            url=url,
            priority=priority,
            symbol=symbol,
            source_type=source_type,
            metadata=metadata or {}
        )
        heapq.heappush(self.task_queue, task)
        
    async def crawl_all(self) -> Dict[str, List[CrawlResult]]:
        """Execute all crawl tasks with intelligent scheduling"""
        results = defaultdict(list)
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # Process all tasks
        tasks = []
        while self.task_queue:
            task = heapq.heappop(self.task_queue)
            if task.url not in self.active_tasks:
                self.active_tasks.add(task.url)
                tasks.append(self._crawl_single_task(task, semaphore))
                
        # Execute all tasks concurrently
        if tasks:
            crawl_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in crawl_results:
                if isinstance(result, CrawlResult):
                    results[result.symbol].append(result)
                elif isinstance(result, Exception):
                    logging.error(f"Crawl task failed: {result}")
                    
        return dict(results)
    
    async def _crawl_single_task(self, task: CrawlTask, semaphore: asyncio.Semaphore) -> Optional[CrawlResult]:
        """Execute a single crawl task"""
        async with semaphore:
            try:
                # Rate limiting
                await self.rate_limiter.acquire(task.url)
                
                # Get headers
                headers = self.user_agent_rotator.get_headers(task.url)
                
                # Make request
                self.stats['total_requests'] += 1
                
                async with self.session.get(task.url, headers=headers) as response:
                    content = await response.text()
                    
                    # Check for duplicates
                    if self.content_deduplicator.is_duplicate(content, task.url):
                        self.stats['duplicate_content'] += 1
                        return None
                        
                    # Score content quality
                    quality_score = self.quality_scorer.score_content(content, task.url, task.source_type)
                    
                    # Create result
                    result = CrawlResult(
                        url=task.url,
                        content=content,
                        status_code=response.status,
                        headers=dict(response.headers),
                        timestamp=datetime.now(),
                        content_hash=hashlib.md5(content.encode()).hexdigest(),
                        source_type=task.source_type,
                        symbol=task.symbol,
                        metadata={**task.metadata, 'quality_score': quality_score}
                    )
                    
                    self.stats['successful_requests'] += 1
                    self.active_tasks.discard(task.url)
                    return result
                    
            except Exception as e:
                self.stats['failed_requests'] += 1
                self.failed_tasks[task.url] += 1
                self.active_tasks.discard(task.url)
                
                # Retry logic
                if task.retry_count < 3 and self.failed_tasks[task.url] < 5:
                    task.retry_count += 1
                    task.scheduled_time = time.time() + (2 ** task.retry_count)  # Exponential backoff
                    heapq.heappush(self.task_queue, task)
                    
                logging.error(f"Failed to crawl {task.url}: {e}")
                return None
                
    def get_statistics(self) -> Dict[str, Any]:
        """Get crawling statistics"""
        runtime = time.time() - self.stats['start_time']
        return {
            **self.stats,
            'runtime_seconds': runtime,
            'requests_per_second': self.stats['total_requests'] / max(runtime, 1),
            'success_rate': self.stats['successful_requests'] / max(self.stats['total_requests'], 1),
            'active_tasks': len(self.active_tasks),
            'pending_tasks': len(self.task_queue)
        }

# Factory function for easy usage
async def create_crawling_engine(max_concurrent: int = 10) -> AdvancedCrawlingEngine:
    """Create and initialize crawling engine"""
    engine = AdvancedCrawlingEngine(max_concurrent)
    await engine.__aenter__()
    return engine
