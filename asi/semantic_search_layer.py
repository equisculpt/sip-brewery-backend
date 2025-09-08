"""
Semantic Search Layer for ASI Finance Search Engine
NLP processing, context-aware ranking, multi-modal search
"""
import re
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json
import logging

# NLP Libraries
try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning("Transformers not available. Using basic NLP processing.")

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    logging.warning("spaCy not available. Using basic entity extraction.")

@dataclass
class SearchQuery:
    raw_query: str
    intent: str  # price, news, analysis, comparison, etc.
    entities: List[str]
    time_filter: Optional[str] = None
    numeric_filters: Dict[str, float] = field(default_factory=dict)
    sector_filter: Optional[str] = None
    sentiment: str = "neutral"  # positive, negative, neutral
    confidence: float = 0.0

@dataclass
class SearchResult:
    content: str
    title: str
    url: str
    symbol: str
    source: str
    timestamp: datetime
    relevance_score: float
    quality_score: float
    sentiment_score: float
    entities: List[str]
    summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)

class QueryProcessor:
    """Process and understand natural language queries"""
    
    def __init__(self):
        self.intent_patterns = {
            'price': [
                r'price', r'cost', r'value', r'worth', r'trading at',
                r'current price', r'share price', r'stock price'
            ],
            'news': [
                r'news', r'latest', r'update', r'announcement', r'report',
                r'what happened', r'recent', r'today'
            ],
            'analysis': [
                r'analysis', r'research', r'recommendation', r'rating',
                r'buy', r'sell', r'hold', r'target price', r'forecast'
            ],
            'comparison': [
                r'vs', r'versus', r'compare', r'comparison', r'better than',
                r'against', r'relative to'
            ],
            'financial': [
                r'revenue', r'profit', r'earnings', r'dividend', r'pe ratio',
                r'market cap', r'debt', r'cash flow', r'balance sheet'
            ],
            'performance': [
                r'performance', r'returns', r'gain', r'loss', r'growth',
                r'decline', r'up', r'down', r'percentage'
            ]
        }
        
        self.time_patterns = {
            'today': r'today|now|current',
            'yesterday': r'yesterday',
            'this_week': r'this week|past week|last week',
            'this_month': r'this month|past month|last month',
            'this_quarter': r'this quarter|q[1-4]|quarter',
            'this_year': r'this year|ytd|year to date',
            'last_year': r'last year|previous year'
        }
        
        self.numeric_patterns = {
            'above': r'above|over|more than|greater than|>\s*',
            'below': r'below|under|less than|<\s*',
            'between': r'between|from\s+\d+\s+to'
        }
        
        # Load spaCy model if available
        if HAS_SPACY:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                self.nlp = None
                logging.warning("spaCy English model not found. Install with: python -m spacy download en_core_web_sm")
        else:
            self.nlp = None
            
    def process_query(self, query: str) -> SearchQuery:
        """Process natural language query and extract structured information"""
        query = query.strip().lower()
        
        # Extract intent
        intent = self._extract_intent(query)
        
        # Extract entities (company names, symbols)
        entities = self._extract_entities(query)
        
        # Extract time filters
        time_filter = self._extract_time_filter(query)
        
        # Extract numeric filters
        numeric_filters = self._extract_numeric_filters(query)
        
        # Extract sector filter
        sector_filter = self._extract_sector_filter(query)
        
        # Analyze sentiment
        sentiment = self._analyze_sentiment(query)
        
        # Calculate confidence
        confidence = self._calculate_confidence(query, intent, entities)
        
        return SearchQuery(
            raw_query=query,
            intent=intent,
            entities=entities,
            time_filter=time_filter,
            numeric_filters=numeric_filters,
            sector_filter=sector_filter,
            sentiment=sentiment,
            confidence=confidence
        )
    
    def _extract_intent(self, query: str) -> str:
        """Extract primary intent from query"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query):
                    score += 1
            intent_scores[intent] = score
            
        # Return intent with highest score, default to 'general'
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        return 'general'
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract company names and symbols from query"""
        entities = []
        
        if self.nlp:
            # Use spaCy for named entity recognition
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ['ORG', 'PERSON', 'PRODUCT']:
                    entities.append(ent.text)
        else:
            # Basic entity extraction using patterns
            # Look for capitalized words that might be company names
            words = query.split()
            for word in words:
                if word.isupper() and len(word) > 1:
                    entities.append(word)
                elif word.istitle() and len(word) > 2:
                    entities.append(word)
                    
        return entities
    
    def _extract_time_filter(self, query: str) -> Optional[str]:
        """Extract time-based filters from query"""
        for time_key, pattern in self.time_patterns.items():
            if re.search(pattern, query):
                return time_key
        return None
    
    def _extract_numeric_filters(self, query: str) -> Dict[str, float]:
        """Extract numeric filters (price ranges, etc.)"""
        filters = {}
        
        # Look for numbers in the query
        numbers = re.findall(r'\d+(?:\.\d+)?', query)
        
        for operator, pattern in self.numeric_patterns.items():
            if re.search(pattern, query):
                if numbers:
                    if operator == 'between' and len(numbers) >= 2:
                        filters['min_value'] = float(numbers[0])
                        filters['max_value'] = float(numbers[1])
                    elif operator in ['above', 'below'] and numbers:
                        filters[f'{operator}_value'] = float(numbers[0])
                        
        return filters
    
    def _extract_sector_filter(self, query: str) -> Optional[str]:
        """Extract sector/industry filters"""
        sector_keywords = {
            'it': ['it', 'software', 'technology', 'tech'],
            'banking': ['bank', 'banking', 'finance', 'financial'],
            'pharma': ['pharma', 'pharmaceutical', 'medicine', 'drug'],
            'auto': ['auto', 'automobile', 'car', 'vehicle'],
            'oil_gas': ['oil', 'gas', 'petroleum', 'energy'],
            'fmcg': ['fmcg', 'consumer', 'goods'],
            'telecom': ['telecom', 'mobile', 'communication']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(keyword in query for keyword in keywords):
                return sector
                
        return None
    
    def _analyze_sentiment(self, query: str) -> str:
        """Analyze sentiment of the query"""
        positive_words = ['good', 'great', 'excellent', 'positive', 'bullish', 'up', 'gain', 'profit']
        negative_words = ['bad', 'poor', 'negative', 'bearish', 'down', 'loss', 'decline', 'fall']
        
        positive_count = sum(1 for word in positive_words if word in query)
        negative_count = sum(1 for word in negative_words if word in query)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        return 'neutral'
    
    def _calculate_confidence(self, query: str, intent: str, entities: List[str]) -> float:
        """Calculate confidence score for query understanding"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence if we found clear intent
        if intent != 'general':
            confidence += 0.2
            
        # Boost confidence if we found entities
        if entities:
            confidence += 0.2
            
        # Boost confidence for longer, more specific queries
        if len(query.split()) > 3:
            confidence += 0.1
            
        return min(confidence, 1.0)

class SemanticEmbedder:
    """Generate semantic embeddings for content and queries"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
        if HAS_TRANSFORMERS:
            try:
                # Use sentence transformers for better semantic understanding
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.has_sentence_model = True
            except Exception as e:
                logging.warning(f"Failed to load sentence transformer: {e}")
                self.has_sentence_model = False
                
                # Fallback to basic transformer
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
                    self.model = AutoModel.from_pretrained('distilbert-base-uncased')
                    self.has_basic_model = True
                except Exception as e:
                    logging.warning(f"Failed to load basic transformer: {e}")
                    self.has_basic_model = False
        else:
            self.has_sentence_model = False
            self.has_basic_model = False
            
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        if self.has_sentence_model:
            return self.sentence_model.encode([text])[0]
        elif self.has_basic_model:
            return self._basic_embed(text)
        else:
            # Fallback to simple TF-IDF-like embedding
            return self._simple_embed(text)
    
    def _basic_embed(self, text: str) -> np.ndarray:
        """Basic embedding using transformer model"""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use mean pooling of last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1)
            
        return embeddings.numpy().flatten()
    
    def _simple_embed(self, text: str) -> np.ndarray:
        """Simple embedding based on word frequency"""
        # This is a very basic fallback
        words = text.lower().split()
        # Create a simple hash-based embedding
        embedding = np.zeros(384)  # Match sentence transformer dimension
        
        for i, word in enumerate(words[:50]):  # Limit to first 50 words
            hash_val = hash(word) % 384
            embedding[hash_val] += 1.0 / (i + 1)  # Position weighting
            
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
            
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

class ContextAwareRanker:
    """Rank search results based on context and relevance"""
    
    def __init__(self):
        self.ranking_weights = {
            'semantic_similarity': 0.3,
            'entity_match': 0.25,
            'intent_match': 0.2,
            'quality_score': 0.15,
            'recency': 0.1
        }
        
    def rank_results(self, query: SearchQuery, results: List[SearchResult], 
                    query_embedding: np.ndarray) -> List[SearchResult]:
        """Rank search results based on multiple factors"""
        
        scored_results = []
        
        for result in results:
            score = self._calculate_relevance_score(query, result, query_embedding)
            result.relevance_score = score
            scored_results.append(result)
            
        # Sort by relevance score
        scored_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return scored_results
    
    def _calculate_relevance_score(self, query: SearchQuery, result: SearchResult, 
                                 query_embedding: np.ndarray) -> float:
        """Calculate overall relevance score for a result"""
        
        scores = {}
        
        # Semantic similarity (if embeddings available)
        if len(query_embedding) > 0:
            # This would require result embeddings to be pre-computed
            scores['semantic_similarity'] = 0.5  # Placeholder
        else:
            scores['semantic_similarity'] = 0.0
            
        # Entity matching
        scores['entity_match'] = self._calculate_entity_match(query, result)
        
        # Intent matching
        scores['intent_match'] = self._calculate_intent_match(query, result)
        
        # Quality score (from crawling engine)
        scores['quality_score'] = result.quality_score
        
        # Recency score
        scores['recency'] = self._calculate_recency_score(result.timestamp)
        
        # Calculate weighted sum
        total_score = sum(
            scores[factor] * weight 
            for factor, weight in self.ranking_weights.items()
        )
        
        return total_score
    
    def _calculate_entity_match(self, query: SearchQuery, result: SearchResult) -> float:
        """Calculate entity matching score"""
        if not query.entities:
            return 0.5  # Neutral score if no entities in query
            
        matches = 0
        for entity in query.entities:
            if entity.lower() in result.content.lower() or entity.lower() in result.title.lower():
                matches += 1
                
        return matches / len(query.entities)
    
    def _calculate_intent_match(self, query: SearchQuery, result: SearchResult) -> float:
        """Calculate intent matching score"""
        intent_keywords = {
            'price': ['price', 'trading', 'value', '₹', 'rs'],
            'news': ['announced', 'reported', 'news', 'update'],
            'analysis': ['analysis', 'recommendation', 'target', 'rating'],
            'financial': ['revenue', 'profit', 'earnings', 'dividend']
        }
        
        if query.intent in intent_keywords:
            keywords = intent_keywords[query.intent]
            content_lower = result.content.lower()
            
            matches = sum(1 for keyword in keywords if keyword in content_lower)
            return min(matches / len(keywords), 1.0)
            
        return 0.5  # Neutral score for unknown intents
    
    def _calculate_recency_score(self, timestamp: datetime) -> float:
        """Calculate recency score (newer content scores higher)"""
        now = datetime.now()
        age_hours = (now - timestamp).total_seconds() / 3600
        
        # Score decreases with age, but slowly
        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 0.9
        elif age_hours < 168:  # 1 week
            return 0.7
        elif age_hours < 720:  # 1 month
            return 0.5
        else:
            return 0.3

class SemanticSearchEngine:
    """Main semantic search engine combining all components"""
    
    def __init__(self):
        self.query_processor = QueryProcessor()
        self.embedder = SemanticEmbedder()
        self.ranker = ContextAwareRanker()
        
    async def search(self, raw_query: str, results: List[SearchResult]) -> Tuple[SearchQuery, List[SearchResult]]:
        """Perform semantic search on results"""
        
        # Process query
        processed_query = self.query_processor.process_query(raw_query)
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(processed_query.raw_query)
        
        # Filter results based on query
        filtered_results = self._filter_results(processed_query, results)
        
        # Rank results
        ranked_results = self.ranker.rank_results(processed_query, filtered_results, query_embedding)
        
        return processed_query, ranked_results
    
    def _filter_results(self, query: SearchQuery, results: List[SearchResult]) -> List[SearchResult]:
        """Filter results based on query constraints"""
        filtered = results
        
        # Time filter
        if query.time_filter:
            filtered = self._apply_time_filter(filtered, query.time_filter)
            
        # Numeric filters
        if query.numeric_filters:
            filtered = self._apply_numeric_filters(filtered, query.numeric_filters)
            
        # Sector filter
        if query.sector_filter:
            filtered = self._apply_sector_filter(filtered, query.sector_filter)
            
        return filtered
    
    def _apply_time_filter(self, results: List[SearchResult], time_filter: str) -> List[SearchResult]:
        """Apply time-based filtering"""
        now = datetime.now()
        
        time_deltas = {
            'today': timedelta(days=1),
            'this_week': timedelta(weeks=1),
            'this_month': timedelta(days=30),
            'this_quarter': timedelta(days=90),
            'this_year': timedelta(days=365)
        }
        
        if time_filter in time_deltas:
            cutoff = now - time_deltas[time_filter]
            return [r for r in results if r.timestamp >= cutoff]
            
        return results
    
    def _apply_numeric_filters(self, results: List[SearchResult], filters: Dict[str, float]) -> List[SearchResult]:
        """Apply numeric filtering (placeholder for price filters, etc.)"""
        # This would require extracting numeric values from content
        # For now, return all results
        return results
    
    def _apply_sector_filter(self, results: List[SearchResult], sector: str) -> List[SearchResult]:
        """Apply sector-based filtering"""
        sector_keywords = {
            'it': ['software', 'technology', 'it services'],
            'banking': ['bank', 'banking', 'financial services'],
            'pharma': ['pharmaceutical', 'drug', 'medicine']
        }
        
        if sector in sector_keywords:
            keywords = sector_keywords[sector]
            return [r for r in results if any(keyword in r.content.lower() for keyword in keywords)]
            
        return results

# Global instance
semantic_search_engine = SemanticSearchEngine()
