"""
Comprehensive Indian Companies Database for ASI Finance Search Engine
Contains NSE/BSE symbols, company names, sectors, and variations
"""
import json
import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
from dataclasses import dataclass

@dataclass
class CompanyInfo:
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap_category: str  # Large, Mid, Small
    exchange: str  # NSE, BSE
    aliases: List[str]
    keywords: List[str]

class IndianCompaniesDatabase:
    def __init__(self):
        self.companies = self._load_companies_data()
        self.symbol_map = self._build_symbol_map()
        self.sector_map = self._build_sector_map()
        
    def _load_companies_data(self) -> Dict[str, CompanyInfo]:
        """Load comprehensive Indian companies database"""
        companies = {
            # IT Sector - Top Companies
            "TCS": CompanyInfo(
                symbol="TCS",
                name="Tata Consultancy Services",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["tcs", "tata consultancy", "tata consultancy services", "tata consulting"],
                keywords=["it", "software", "consulting", "tata group", "technology"]
            ),
            "INFY": CompanyInfo(
                symbol="INFY",
                name="Infosys Limited",
                sector="Information Technology", 
                industry="IT Services",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["infosys", "infy", "infosys limited"],
                keywords=["it", "software", "consulting", "technology", "bangalore"]
            ),
            "WIPRO": CompanyInfo(
                symbol="WIPRO",
                name="Wipro Limited",
                sector="Information Technology",
                industry="IT Services", 
                market_cap_category="Large",
                exchange="NSE",
                aliases=["wipro", "wipro limited"],
                keywords=["it", "software", "consulting", "technology"]
            ),
            "HCLTECH": CompanyInfo(
                symbol="HCLTECH",
                name="HCL Technologies",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category="Large", 
                exchange="NSE",
                aliases=["hcl", "hcl tech", "hcl technologies", "hcltech"],
                keywords=["it", "software", "consulting", "technology"]
            ),
            "TECHM": CompanyInfo(
                symbol="TECHM",
                name="Tech Mahindra",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["tech mahindra", "techm", "mahindra tech"],
                keywords=["it", "software", "consulting", "technology", "mahindra"]
            ),
            
            # Banking Sector
            "HDFCBANK": CompanyInfo(
                symbol="HDFCBANK", 
                name="HDFC Bank Limited",
                sector="Financial Services",
                industry="Private Bank",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["hdfc bank", "hdfc", "hdfcbank"],
                keywords=["bank", "banking", "finance", "private bank"]
            ),
            "ICICIBANK": CompanyInfo(
                symbol="ICICIBANK",
                name="ICICI Bank Limited", 
                sector="Financial Services",
                industry="Private Bank",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["icici bank", "icici", "icicibank"],
                keywords=["bank", "banking", "finance", "private bank"]
            ),
            "AXISBANK": CompanyInfo(
                symbol="AXISBANK",
                name="Axis Bank Limited",
                sector="Financial Services", 
                industry="Private Bank",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["axis bank", "axis", "axisbank"],
                keywords=["bank", "banking", "finance", "private bank"]
            ),
            "SBIN": CompanyInfo(
                symbol="SBIN",
                name="State Bank of India",
                sector="Financial Services",
                industry="Public Bank", 
                market_cap_category="Large",
                exchange="NSE",
                aliases=["sbi", "state bank", "state bank of india", "sbin"],
                keywords=["bank", "banking", "finance", "public bank", "government"]
            ),
            
            # Oil & Gas
            "RELIANCE": CompanyInfo(
                symbol="RELIANCE",
                name="Reliance Industries Limited",
                sector="Oil & Gas",
                industry="Integrated Oil & Gas",
                market_cap_category="Large", 
                exchange="NSE",
                aliases=["reliance", "ril", "reliance industries", "reliance limited"],
                keywords=["oil", "gas", "petrochemicals", "telecom", "jio", "ambani"]
            ),
            "ONGC": CompanyInfo(
                symbol="ONGC", 
                name="Oil and Natural Gas Corporation",
                sector="Oil & Gas",
                industry="Oil Exploration",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["ongc", "oil and natural gas", "oil natural gas corporation"],
                keywords=["oil", "gas", "exploration", "government", "psu"]
            ),
            
            # Pharmaceuticals
            "SUNPHARMA": CompanyInfo(
                symbol="SUNPHARMA",
                name="Sun Pharmaceutical Industries",
                sector="Healthcare",
                industry="Pharmaceuticals",
                market_cap_category="Large",
                exchange="NSE", 
                aliases=["sun pharma", "sunpharma", "sun pharmaceutical"],
                keywords=["pharma", "pharmaceutical", "healthcare", "medicine"]
            ),
            "DRREDDY": CompanyInfo(
                symbol="DRREDDY",
                name="Dr. Reddy's Laboratories", 
                sector="Healthcare",
                industry="Pharmaceuticals",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["dr reddy", "dr reddys", "drreddy", "reddy labs"],
                keywords=["pharma", "pharmaceutical", "healthcare", "medicine"]
            ),
            
            # FMCG
            "HINDUNILVR": CompanyInfo(
                symbol="HINDUNILVR",
                name="Hindustan Unilever Limited",
                sector="FMCG", 
                industry="Personal Care",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["hul", "hindustan unilever", "hindunilvr", "unilever"],
                keywords=["fmcg", "consumer goods", "personal care", "soap", "shampoo"]
            ),
            "ITC": CompanyInfo(
                symbol="ITC",
                name="ITC Limited",
                sector="FMCG",
                industry="Diversified FMCG", 
                market_cap_category="Large",
                exchange="NSE",
                aliases=["itc", "itc limited"],
                keywords=["fmcg", "tobacco", "cigarettes", "hotels", "consumer goods"]
            ),
            
            # Metals & Mining
            "TATASTEEL": CompanyInfo(
                symbol="TATASTEEL", 
                name="Tata Steel Limited",
                sector="Metals & Mining",
                industry="Steel",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["tata steel", "tatasteel", "tata steel limited"],
                keywords=["steel", "metals", "mining", "tata group"]
            ),
            "HINDALCO": CompanyInfo(
                symbol="HINDALCO",
                name="Hindalco Industries Limited",
                sector="Metals & Mining",
                industry="Aluminium",
                market_cap_category="Large", 
                exchange="NSE",
                aliases=["hindalco", "hindalco industries"],
                keywords=["aluminium", "metals", "mining", "aditya birla"]
            ),
            
            # Auto Sector
            "MARUTI": CompanyInfo(
                symbol="MARUTI",
                name="Maruti Suzuki India Limited", 
                sector="Automobile",
                industry="Passenger Cars",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["maruti", "maruti suzuki", "maruti suzuki india"],
                keywords=["auto", "car", "automobile", "suzuki", "passenger vehicle"]
            ),
            "TATAMOTORS": CompanyInfo(
                symbol="TATAMOTORS",
                name="Tata Motors Limited",
                sector="Automobile",
                industry="Commercial Vehicles",
                market_cap_category="Large",
                exchange="NSE", 
                aliases=["tata motors", "tatamotors", "tata motor"],
                keywords=["auto", "car", "automobile", "commercial vehicle", "tata group"]
            ),
            
            # Telecom
            "BHARTIARTL": CompanyInfo(
                symbol="BHARTIARTL",
                name="Bharti Airtel Limited",
                sector="Telecom",
                industry="Telecom Services",
                market_cap_category="Large",
                exchange="NSE",
                aliases=["airtel", "bharti airtel", "bhartiartl"],
                keywords=["telecom", "mobile", "airtel", "bharti"]
            )
        }
        return companies
    
    def _build_symbol_map(self) -> Dict[str, str]:
        """Build mapping from all possible names/aliases to symbols"""
        symbol_map = {}
        
        for symbol, company in self.companies.items():
            # Add symbol itself
            symbol_map[symbol.lower()] = symbol
            
            # Add company name
            symbol_map[company.name.lower()] = symbol
            
            # Add all aliases
            for alias in company.aliases:
                symbol_map[alias.lower()] = symbol
                
        return symbol_map
    
    def _build_sector_map(self) -> Dict[str, List[str]]:
        """Build mapping from sectors to company symbols"""
        sector_map = {}
        
        for symbol, company in self.companies.items():
            sector = company.sector.lower()
            if sector not in sector_map:
                sector_map[sector] = []
            sector_map[sector].append(symbol)
            
        return sector_map
    
    def resolve_symbol(self, query: str) -> Optional[str]:
        """Resolve natural language query to stock symbol"""
        if not query:
            return None
            
        # Clean and normalize query
        query = self._clean_query(query)
        
        # Try exact match first
        if query in self.symbol_map:
            return self.symbol_map[query]
        
        # Try fuzzy matching
        best_match = self._fuzzy_match(query)
        if best_match:
            return best_match
            
        # Try sector matching
        sector_symbols = self._match_sector(query)
        if sector_symbols:
            return sector_symbols[0]  # Return first match for now
            
        return None
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query"""
        query = query.lower().strip()
        
        # Remove common suffixes
        suffixes = [
            "share price", "stock price", "price", "share", "stock",
            "limited", "ltd", "ltd.", "company", "corp", "corporation",
            "industries", "services", "technologies", "tech"
        ]
        
        for suffix in suffixes:
            if query.endswith(" " + suffix):
                query = query[:-len(suffix)-1].strip()
                
        return query
    
    def _fuzzy_match(self, query: str, threshold: float = 0.6) -> Optional[str]:
        """Find best fuzzy match for the query"""
        best_ratio = 0
        best_symbol = None
        
        for key, symbol in self.symbol_map.items():
            ratio = SequenceMatcher(None, query, key).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_symbol = symbol
                
        return best_symbol
    
    def _match_sector(self, query: str) -> List[str]:
        """Match query against sectors and return symbols"""
        sector_keywords = {
            "it": "information technology",
            "software": "information technology", 
            "technology": "information technology",
            "tech": "information technology",
            "bank": "financial services",
            "banking": "financial services",
            "finance": "financial services",
            "oil": "oil & gas",
            "gas": "oil & gas",
            "pharma": "healthcare",
            "pharmaceutical": "healthcare",
            "medicine": "healthcare",
            "fmcg": "fmcg",
            "consumer": "fmcg",
            "auto": "automobile",
            "car": "automobile",
            "telecom": "telecom",
            "mobile": "telecom"
        }
        
        for keyword, sector in sector_keywords.items():
            if keyword in query:
                return self.sector_map.get(sector, [])
                
        return []
    
    def get_company_info(self, symbol: str) -> Optional[CompanyInfo]:
        """Get complete company information"""
        return self.companies.get(symbol.upper())
    
    def search_companies(self, query: str) -> List[Tuple[str, CompanyInfo, float]]:
        """Search companies and return ranked results with confidence scores"""
        results = []
        query = self._clean_query(query)
        
        for symbol, company in self.companies.items():
            score = self._calculate_relevance_score(query, company)
            if score > 0.3:  # Minimum relevance threshold
                results.append((symbol, company, score))
                
        # Sort by relevance score
        results.sort(key=lambda x: x[2], reverse=True)
        return results
    
    def _calculate_relevance_score(self, query: str, company: CompanyInfo) -> float:
        """Calculate relevance score for a company given a query"""
        score = 0.0
        
        # Check symbol match
        if query == company.symbol.lower():
            score += 1.0
        elif company.symbol.lower() in query:
            score += 0.8
            
        # Check name match
        name_ratio = SequenceMatcher(None, query, company.name.lower()).ratio()
        score += name_ratio * 0.9
        
        # Check aliases
        for alias in company.aliases:
            alias_ratio = SequenceMatcher(None, query, alias).ratio()
            score += alias_ratio * 0.7
            
        # Check keywords
        for keyword in company.keywords:
            if keyword in query:
                score += 0.5
                
        return min(score, 1.0)  # Cap at 1.0

# Global instance
indian_companies_db = IndianCompaniesDatabase()
