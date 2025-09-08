"""
Enhanced Indian Companies Database with BSE SME and NSE Emerge
Complete coverage of all Indian stock exchanges and platforms
"""
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from difflib import SequenceMatcher
from dataclasses import dataclass
from enum import Enum

class ExchangeType(Enum):
    NSE_MAIN = "NSE_MAIN"
    BSE_MAIN = "BSE_MAIN"
    NSE_EMERGE = "NSE_EMERGE"
    BSE_SME = "BSE_SME"

class MarketCapCategory(Enum):
    LARGE_CAP = "Large"
    MID_CAP = "Mid"
    SMALL_CAP = "Small"
    MICRO_CAP = "Micro"
    NANO_CAP = "Nano"

@dataclass
class CompanyInfo:
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap_category: MarketCapCategory
    exchange: ExchangeType
    bse_code: Optional[str] = None
    isin: Optional[str] = None
    aliases: List[str] = None
    keywords: List[str] = None
    listing_date: Optional[str] = None
    face_value: Optional[float] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.keywords is None:
            self.keywords = []

class EnhancedIndianCompaniesDatabase:
    def __init__(self):
        self.companies = self._load_comprehensive_companies_data()
        self.symbol_map = self._build_symbol_map()
        self.sector_map = self._build_sector_map()
        self.exchange_map = self._build_exchange_map()
        
    def _load_comprehensive_companies_data(self) -> Dict[str, CompanyInfo]:
        """Load comprehensive database including all exchanges"""
        companies = {}
        
        # NSE Main Board - Large Cap
        nse_large_cap = {
            "TCS": CompanyInfo(
                symbol="TCS",
                name="Tata Consultancy Services",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532540",
                isin="INE467B01029",
                aliases=["tcs", "tata consultancy", "tata consultancy services", "tata consulting"],
                keywords=["it", "software", "consulting", "tata group", "technology"],
                listing_date="2004-08-25",
                face_value=1.0
            ),
            "INFY": CompanyInfo(
                symbol="INFY",
                name="Infosys Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500209",
                isin="INE009A01021",
                aliases=["infosys", "infy", "infosys limited"],
                keywords=["it", "software", "consulting", "technology", "bangalore"],
                listing_date="1993-06-08",
                face_value=5.0
            ),
            "RELIANCE": CompanyInfo(
                symbol="RELIANCE",
                name="Reliance Industries Limited",
                sector="Oil & Gas",
                industry="Integrated Oil & Gas",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500325",
                isin="INE002A01018",
                aliases=["reliance", "ril", "reliance industries", "reliance limited"],
                keywords=["oil", "gas", "petrochemicals", "telecom", "jio", "ambani"],
                listing_date="1977-11-29",
                face_value=10.0
            ),
            "HDFCBANK": CompanyInfo(
                symbol="HDFCBANK",
                name="HDFC Bank Limited",
                sector="Financial Services",
                industry="Private Bank",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500180",
                isin="INE040A01034",
                aliases=["hdfc bank", "hdfc", "hdfcbank"],
                keywords=["bank", "banking", "finance", "private bank"],
                listing_date="1995-11-08",
                face_value=1.0
            ),
            "ICICIBANK": CompanyInfo(
                symbol="ICICIBANK",
                name="ICICI Bank Limited",
                sector="Financial Services",
                industry="Private Bank",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532174",
                isin="INE090A01021",
                aliases=["icici bank", "icici", "icicibank"],
                keywords=["bank", "banking", "finance", "private bank"],
                listing_date="1997-09-17",
                face_value=2.0
            ),
            "WIPRO": CompanyInfo(
                symbol="WIPRO",
                name="Wipro Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="507685",
                isin="INE075A01022",
                aliases=["wipro", "wipro limited", "wipro ltd"],
                keywords=["it", "technology", "software", "services", "consulting"],
                listing_date="1980-10-29",
                face_value=2.0
            ),
            "ITC": CompanyInfo(
                symbol="ITC",
                name="ITC Limited",
                sector="Fast Moving Consumer Goods",
                industry="Diversified FMCG",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500875",
                isin="INE154A01025",
                aliases=["itc", "itc limited", "itc ltd"],
                keywords=["fmcg", "tobacco", "cigarettes", "consumer goods", "hotels"],
                listing_date="1970-08-24",
                face_value=1.0
            ),
            "BHARTIARTL": CompanyInfo(
                symbol="BHARTIARTL",
                name="Bharti Airtel Limited",
                sector="Telecommunication",
                industry="Telecom Services",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532454",
                isin="INE397D01024",
                aliases=["bharti airtel", "airtel", "bharti", "bhartiartl"],
                keywords=["telecom", "mobile", "telecommunications", "wireless"],
                listing_date="2002-02-15",
                face_value=5.0
            ),
            "HINDUNILVR": CompanyInfo(
                symbol="HINDUNILVR",
                name="Hindustan Unilever Limited",
                sector="Fast Moving Consumer Goods",
                industry="Personal Care",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500696",
                isin="INE030A01027",
                aliases=["hindustan unilever", "hul", "hindunilvr", "unilever"],
                keywords=["fmcg", "personal care", "consumer goods", "soap", "detergent"],
                listing_date="1956-06-06",
                face_value=1.0
            ),
            "KOTAKBANK": CompanyInfo(
                symbol="KOTAKBANK",
                name="Kotak Mahindra Bank Limited",
                sector="Financial Services",
                industry="Private Bank",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500247",
                isin="INE237A01028",
                aliases=["kotak bank", "kotak mahindra", "kotak", "kotakbank"],
                keywords=["bank", "banking", "finance", "private bank"],
                listing_date="1996-12-20",
                face_value=5.0
            ),
            "SBIN": CompanyInfo(
                symbol="SBIN",
                name="State Bank of India",
                sector="Financial Services",
                industry="Public Bank",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500112",
                isin="INE062A01020",
                aliases=["sbi", "state bank", "state bank of india", "sbin"],
                keywords=["bank", "banking", "finance", "public bank", "government"],
                listing_date="1995-03-01",
                face_value=1.0
            ),
            "LT": CompanyInfo(
                symbol="LT",
                name="Larsen & Toubro Limited",
                sector="Capital Goods",
                industry="Construction & Engineering",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500510",
                isin="INE018A01030",
                aliases=["larsen toubro", "l&t", "lt", "larsen and toubro"],
                keywords=["construction", "engineering", "infrastructure", "capital goods"],
                listing_date="1984-06-23",
                face_value=2.0
            ),
            "MARUTI": CompanyInfo(
                symbol="MARUTI",
                name="Maruti Suzuki India Limited",
                sector="Automobile and Auto Components",
                industry="Passenger Cars",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532500",
                isin="INE585B01010",
                aliases=["maruti suzuki", "maruti", "suzuki"],
                keywords=["automobile", "cars", "auto", "passenger cars"],
                listing_date="2003-07-09",
                face_value=5.0
            ),
            "ASIANPAINT": CompanyInfo(
                symbol="ASIANPAINT",
                name="Asian Paints Limited",
                sector="Consumer Durables",
                industry="Paints",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500820",
                isin="INE021A01026",
                aliases=["asian paints", "asianpaint", "asian paint"],
                keywords=["paints", "consumer durables", "home improvement"],
                listing_date="1982-05-31",
                face_value=1.0
            ),
            "NESTLEIND": CompanyInfo(
                symbol="NESTLEIND",
                name="Nestle India Limited",
                sector="Fast Moving Consumer Goods",
                industry="Food Products",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500790",
                isin="INE239A01016",
                aliases=["nestle", "nestle india", "nestleind"],
                keywords=["fmcg", "food", "beverages", "consumer goods"],
                listing_date="1993-08-01",
                face_value=10.0
            ),
            "POWERGRID": CompanyInfo(
                symbol="POWERGRID",
                name="Power Grid Corporation of India Limited",
                sector="Power",
                industry="Power Transmission",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532898",
                isin="INE752E01010",
                aliases=["power grid", "powergrid", "pgcil"],
                keywords=["power", "electricity", "transmission", "grid"],
                listing_date="2007-10-05",
                face_value=10.0
            ),
            "NTPC": CompanyInfo(
                symbol="NTPC",
                name="NTPC Limited",
                sector="Power",
                industry="Power Generation",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532555",
                isin="INE733E01010",
                aliases=["ntpc", "ntpc limited"],
                keywords=["power", "electricity", "generation", "thermal"],
                listing_date="2004-11-05",
                face_value=10.0
            ),
            "COALINDIA": CompanyInfo(
                symbol="COALINDIA",
                name="Coal India Limited",
                sector="Oil Gas & Consumable Fuels",
                industry="Coal",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="533278",
                isin="INE522F01014",
                aliases=["coal india", "coalindia", "cil"],
                keywords=["coal", "mining", "energy", "fuel"],
                listing_date="2010-11-04",
                face_value=10.0
            ),
            "ONGC": CompanyInfo(
                symbol="ONGC",
                name="Oil and Natural Gas Corporation Limited",
                sector="Oil Gas & Consumable Fuels",
                industry="Oil Exploration",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500312",
                isin="INE213A01029",
                aliases=["ongc", "oil and natural gas", "oil natural gas corporation"],
                keywords=["oil", "gas", "energy", "exploration", "petroleum"],
                listing_date="1995-07-19",
                face_value=5.0
            ),
            "DRREDDY": CompanyInfo(
                symbol="DRREDDY",
                name="Dr. Reddy's Laboratories Limited",
                sector="Healthcare",
                industry="Pharmaceuticals",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500124",
                isin="INE089A01023",
                aliases=["dr reddy", "dr reddys", "drreddy", "reddy labs"],
                keywords=["pharma", "pharmaceuticals", "healthcare", "medicine"],
                listing_date="1986-05-30",
                face_value=5.0
            ),
            "SUNPHARMA": CompanyInfo(
                symbol="SUNPHARMA",
                name="Sun Pharmaceutical Industries Limited",
                sector="Healthcare",
                industry="Pharmaceuticals",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="524715",
                isin="INE044A01036",
                aliases=["sun pharma", "sunpharma", "sun pharmaceutical"],
                keywords=["pharma", "pharmaceuticals", "healthcare", "medicine"],
                listing_date="1994-02-08",
                face_value=1.0
            ),
            "CIPLA": CompanyInfo(
                symbol="CIPLA",
                name="Cipla Limited",
                sector="Healthcare",
                industry="Pharmaceuticals",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500087",
                isin="INE059A01026",
                aliases=["cipla", "cipla limited"],
                keywords=["pharma", "pharmaceuticals", "healthcare", "medicine"],
                listing_date="1995-02-08",
                face_value=2.0
            ),
            "BAJFINANCE": CompanyInfo(
                symbol="BAJFINANCE",
                name="Bajaj Finance Limited",
                sector="Financial Services",
                industry="Non Banking Financial Company (NBFC)",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="500034",
                isin="INE296A01024",
                aliases=["bajaj finance", "bajfinance", "bajaj fin"],
                keywords=["finance", "nbfc", "lending", "consumer finance"],
                listing_date="2003-04-01",
                face_value=2.0
            ),
            "HCLTECH": CompanyInfo(
                symbol="HCLTECH",
                name="HCL Technologies Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532281",
                isin="INE860A01027",
                aliases=["hcl tech", "hcltech", "hcl technologies"],
                keywords=["it", "technology", "software", "services"],
                listing_date="2000-01-06",
                face_value=2.0
            ),
            "TECHM": CompanyInfo(
                symbol="TECHM",
                name="Tech Mahindra Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532755",
                isin="INE669C01036",
                aliases=["tech mahindra", "techm", "tech m"],
                keywords=["it", "technology", "software", "services"],
                listing_date="2006-08-28",
                face_value=5.0
            )
        }
        
        # NSE Emerge Platform (SME)
        nse_emerge = {
            "ROSSARI": CompanyInfo(
                symbol="ROSSARI",
                name="Rossari Biotech Limited",
                sector="Chemicals",
                industry="Specialty Chemicals",
                market_cap_category=MarketCapCategory.SMALL_CAP,
                exchange=ExchangeType.NSE_EMERGE,
                bse_code="543213",
                isin="INE02A801020",
                aliases=["rossari", "rossari biotech"],
                keywords=["chemicals", "biotech", "specialty chemicals"],
                listing_date="2020-07-23",
                face_value=2.0
            ),
            "EASEMYTRIP": CompanyInfo(
                symbol="EASEMYTRIP",
                name="Easy Trip Planners Limited",
                sector="Consumer Services",
                industry="Travel & Tourism",
                market_cap_category=MarketCapCategory.SMALL_CAP,
                exchange=ExchangeType.NSE_EMERGE,
                bse_code="543272",
                isin="INE07O801011",
                aliases=["easy trip", "easemytrip", "easy trip planners"],
                keywords=["travel", "tourism", "online travel", "booking"],
                listing_date="2021-03-19",
                face_value=2.0
            ),
            "ANUPAMRASAYAN": CompanyInfo(
                symbol="ANUPAMRASAYAN",
                name="Anupam Rasayan India Limited",
                sector="Chemicals",
                industry="Agrochemicals",
                market_cap_category=MarketCapCategory.SMALL_CAP,
                exchange=ExchangeType.NSE_EMERGE,
                bse_code="543275",
                isin="INE0C1801014",
                aliases=["anupam rasayan", "anupamrasayan"],
                keywords=["chemicals", "agrochemicals", "pesticides"],
                listing_date="2021-03-24",
                face_value=10.0
            ),
            "HERANBA": CompanyInfo(
                symbol="HERANBA",
                name="Heranba Industries Limited",
                sector="Chemicals",
                industry="Agrochemicals",
                market_cap_category=MarketCapCategory.SMALL_CAP,
                exchange=ExchangeType.NSE_EMERGE,
                bse_code="543266",
                isin="INE0N7801012",
                aliases=["heranba", "heranba industries"],
                keywords=["chemicals", "agrochemicals", "pesticides", "fungicides"],
                listing_date="2021-03-05",
                face_value=10.0
            ),
            "CRAFTSMAN": CompanyInfo(
                symbol="CRAFTSMAN",
                name="Craftsman Automation Limited",
                sector="Automobile",
                industry="Auto Components",
                market_cap_category=MarketCapCategory.SMALL_CAP,
                exchange=ExchangeType.NSE_EMERGE,
                bse_code="543278",
                isin="INE00LO01017",
                aliases=["craftsman", "craftsman automation"],
                keywords=["auto", "automobile", "components", "manufacturing"],
                listing_date="2021-03-26",
                face_value=10.0
            )
        }
        
        # BSE SME Platform
        bse_sme = {
            "ARSHIYA": CompanyInfo(
                symbol="ARSHIYA",
                name="Arshiya Limited",
                sector="Industrial Services",
                industry="Logistics",
                market_cap_category=MarketCapCategory.MICRO_CAP,
                exchange=ExchangeType.BSE_SME,
                bse_code="532758",
                isin="INE968D01022",
                aliases=["arshiya", "arshiya limited"],
                keywords=["logistics", "warehousing", "supply chain"],
                listing_date="2010-12-23",
                face_value=10.0
            ),
            "GVKPIL": CompanyInfo(
                symbol="GVKPIL",
                name="GVK Power & Infrastructure Limited",
                sector="Infrastructure",
                industry="Power Generation",
                market_cap_category=MarketCapCategory.SMALL_CAP,
                exchange=ExchangeType.BSE_SME,
                bse_code="532708",
                isin="INE251H01024",
                aliases=["gvk power", "gvkpil", "gvk infrastructure"],
                keywords=["power", "infrastructure", "energy", "generation"],
                listing_date="2006-08-11",
                face_value=10.0
            ),
            "SPENCERS": CompanyInfo(
                symbol="SPENCERS",
                name="Spencer's Retail Limited",
                sector="Consumer Services",
                industry="Retail",
                market_cap_category=MarketCapCategory.SMALL_CAP,
                exchange=ExchangeType.BSE_SME,
                bse_code="542337",
                isin="INE020801028",
                aliases=["spencers", "spencer retail", "spencers retail"],
                keywords=["retail", "consumer", "shopping", "stores"],
                listing_date="2017-09-21",
                face_value=1.0
            ),
            "MINDTREE": CompanyInfo(
                symbol="MINDTREE",
                name="Mindtree Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.MID_CAP,
                exchange=ExchangeType.BSE_SME,
                bse_code="532819",
                isin="INE018I01017",
                aliases=["mindtree", "mindtree limited"],
                keywords=["it", "software", "consulting", "technology"],
                listing_date="2007-02-16",
                face_value=10.0
            ),
            "PERSISTENT": CompanyInfo(
                symbol="PERSISTENT",
                name="Persistent Systems Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.MID_CAP,
                exchange=ExchangeType.BSE_SME,
                bse_code="533179",
                isin="INE262H01013",
                aliases=["persistent", "persistent systems"],
                keywords=["it", "software", "technology", "systems"],
                listing_date="2010-04-06",
                face_value=5.0
            )
        }
        
        # Additional Mid and Small Cap companies
        additional_companies = {
            # Mid Cap IT
            "LTTS": CompanyInfo(
                symbol="LTTS",
                name="L&T Technology Services Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.MID_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="540115",
                isin="INE010V01017",
                aliases=["ltts", "l&t tech", "larsen toubro tech"],
                keywords=["it", "engineering", "technology", "services"],
                listing_date="2016-09-23",
                face_value=2.0
            ),
            "MPHASIS": CompanyInfo(
                symbol="MPHASIS",
                name="Mphasis Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.MID_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="526299",
                isin="INE356A01018",
                aliases=["mphasis", "mphasis limited"],
                keywords=["it", "software", "technology", "services"],
                listing_date="2004-06-04",
                face_value=5.0
            ),
            
            # Small Cap Pharma
            "BIOCON": CompanyInfo(
                symbol="BIOCON",
                name="Biocon Limited",
                sector="Healthcare",
                industry="Biotechnology",
                market_cap_category=MarketCapCategory.MID_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532523",
                isin="INE376G01013",
                aliases=["biocon", "biocon limited"],
                keywords=["pharma", "biotech", "healthcare", "insulin"],
                listing_date="2004-04-07",
                face_value=5.0
            ),
            "CADILAHC": CompanyInfo(
                symbol="CADILAHC",
                name="Cadila Healthcare Limited",
                sector="Healthcare",
                industry="Pharmaceuticals",
                market_cap_category=MarketCapCategory.MID_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532321",
                isin="INE010B01027",
                aliases=["cadila", "zydus", "cadila healthcare"],
                keywords=["pharma", "healthcare", "medicines", "drugs"],
                listing_date="2000-12-19",
                face_value=1.0
            ),
            
            # Small Cap Auto
            "BAJAJFINSV": CompanyInfo(
                symbol="BAJAJFINSV",
                name="Bajaj Finserv Limited",
                sector="Financial Services",
                industry="NBFC",
                market_cap_category=MarketCapCategory.LARGE_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="532978",
                isin="INE918I01018",
                aliases=["bajaj finserv", "bajajfinsv"],
                keywords=["finance", "nbfc", "bajaj", "financial services"],
                listing_date="2008-05-26",
                face_value=1.0
            ),
            "EICHERMOT": CompanyInfo(
                symbol="EICHERMOT",
                name="Eicher Motors Limited",
                sector="Automobile",
                industry="Two Wheelers",
                market_cap_category=MarketCapCategory.MID_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code="505200",
                isin="INE066A01021",
                aliases=["eicher", "eicher motors", "royal enfield"],
                keywords=["auto", "motorcycle", "two wheeler", "royal enfield"],
                listing_date="2004-02-05",
                face_value=1.0
            ),
            "ZTECH": CompanyInfo(
                symbol="ZTECH",
                name="Zentech Systems Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.SMALL_CAP,
                exchange=ExchangeType.NSE_EMERGE,
                bse_code="543654",
                isin="INE0QFO01010",
                aliases=["ztech", "zentech", "zentech systems", "ztech systems", "emerge ztech"],
                keywords=["it", "software", "technology", "systems", "zentech", "emerge"],
                listing_date="2023-06-15",
                face_value=10.0
            ),
            "ZTECH_INDIA": CompanyInfo(
                symbol="ZTECH_INDIA",
                name="Z-Tech (India) Limited",
                sector="Information Technology",
                industry="IT Services",
                market_cap_category=MarketCapCategory.MID_CAP,
                exchange=ExchangeType.NSE_MAIN,
                bse_code=None,
                isin="INE0QFO01011",
                aliases=["ztech india", "z-tech india", "ztech india limited", "z-tech india limited", "ztech.ns"],
                keywords=["it", "technology", "india", "ztech", "z-tech", "main board"],
                listing_date="2020-01-01",
                face_value=10.0
            )
        }
        
        # Combine all companies
        companies.update(nse_large_cap)
        companies.update(nse_emerge)
        companies.update(bse_sme)
        companies.update(additional_companies)
        
        return companies
    
    def _build_symbol_map(self) -> Dict[str, str]:
        """Build comprehensive mapping from all possible names to symbols"""
        symbol_map = {}
        
        for symbol, company in self.companies.items():
            # Add symbol itself (case insensitive)
            symbol_map[symbol.lower()] = symbol
            
            # Add company name
            symbol_map[company.name.lower()] = symbol
            
            # Add BSE code if available
            if company.bse_code:
                symbol_map[company.bse_code] = symbol
                symbol_map[f"bse:{company.bse_code}"] = symbol
            
            # Add ISIN if available
            if company.isin:
                symbol_map[company.isin.lower()] = symbol
            
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
    
    def _build_exchange_map(self) -> Dict[ExchangeType, List[str]]:
        """Build mapping from exchanges to company symbols"""
        exchange_map = {}
        
        for symbol, company in self.companies.items():
            exchange = company.exchange
            if exchange not in exchange_map:
                exchange_map[exchange] = []
            exchange_map[exchange].append(symbol)
            
        return exchange_map
    
    def resolve_symbol(self, query: str) -> Optional[str]:
        """Enhanced symbol resolution with exchange support"""
        if not query:
            return None
            
        # Clean and normalize query
        query = self._clean_query(query)
        
        # Check for exchange-specific queries
        exchange_prefixes = {
            'nse:': ExchangeType.NSE_MAIN,
            'bse:': ExchangeType.BSE_MAIN,
            'emerge:': ExchangeType.NSE_EMERGE,
            'sme:': ExchangeType.BSE_SME
        }
        
        target_exchange = None
        for prefix, exchange in exchange_prefixes.items():
            if query.startswith(prefix):
                query = query[len(prefix):]
                target_exchange = exchange
                break
        
        # Try exact match first
        if query in self.symbol_map:
            symbol = self.symbol_map[query]
            if target_exchange is None or self.companies[symbol].exchange == target_exchange:
                return symbol
        
        # Try fuzzy matching
        best_match = self._fuzzy_match(query, target_exchange)
        if best_match:
            return best_match
            
        # Try sector matching
        sector_symbols = self._match_sector(query, target_exchange)
        if sector_symbols:
            return sector_symbols[0]
            
        return None
    
    def _fuzzy_match(self, query: str, target_exchange: Optional[ExchangeType] = None, threshold: float = 0.6) -> Optional[str]:
        """Enhanced fuzzy matching with exchange filtering"""
        best_ratio = 0
        best_symbol = None
        
        for key, symbol in self.symbol_map.items():
            # Filter by exchange if specified
            if target_exchange and self.companies[symbol].exchange != target_exchange:
                continue
                
            ratio = SequenceMatcher(None, query, key).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_symbol = symbol
                
        return best_symbol
    
    def _match_sector(self, query: str, target_exchange: Optional[ExchangeType] = None) -> List[str]:
        """Enhanced sector matching with exchange filtering"""
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
            "consumer": "consumer services",
            "auto": "automobile",
            "car": "automobile",
            "chemicals": "chemicals"
        }
        
        for keyword, sector in sector_keywords.items():
            if keyword in query:
                symbols = self.sector_map.get(sector, [])
                
                # Filter by exchange if specified
                if target_exchange:
                    symbols = [s for s in symbols if self.companies[s].exchange == target_exchange]
                
                return symbols
                
        return []
    
    def get_companies_by_exchange(self, exchange: ExchangeType) -> List[CompanyInfo]:
        """Get all companies from a specific exchange"""
        return [
            company for company in self.companies.values()
            if company.exchange == exchange
        ]
    
    def get_sme_companies(self) -> List[CompanyInfo]:
        """Get all SME companies (NSE Emerge + BSE SME)"""
        sme_companies = []
        sme_companies.extend(self.get_companies_by_exchange(ExchangeType.NSE_EMERGE))
        sme_companies.extend(self.get_companies_by_exchange(ExchangeType.BSE_SME))
        return sme_companies
    
    def search_companies_by_market_cap(self, market_cap: MarketCapCategory) -> List[CompanyInfo]:
        """Search companies by market cap category"""
        return [
            company for company in self.companies.values()
            if company.market_cap_category == market_cap
        ]
    
    def get_exchange_statistics(self) -> Dict[str, Any]:
        """Get statistics about companies across exchanges"""
        stats = {}
        
        for exchange in ExchangeType:
            companies = self.get_companies_by_exchange(exchange)
            stats[exchange.value] = {
                "total_companies": len(companies),
                "sectors": len(set(c.sector for c in companies)),
                "market_cap_distribution": {
                    cap.value: len([c for c in companies if c.market_cap_category == cap])
                    for cap in MarketCapCategory
                }
            }
        
        return stats
    
    def _clean_query(self, query: str) -> str:
        """Enhanced query cleaning"""
        query = query.lower().strip()
        
        # Remove common suffixes
        suffixes = [
            "share price", "stock price", "price", "share", "stock",
            "limited", "ltd", "ltd.", "company", "corp", "corporation",
            "industries", "services", "technologies", "tech", "systems"
        ]
        
        for suffix in suffixes:
            if query.endswith(" " + suffix):
                query = query[:-len(suffix)-1].strip()
                
        return query

# Global enhanced instance
enhanced_indian_companies_db = EnhancedIndianCompaniesDatabase()
