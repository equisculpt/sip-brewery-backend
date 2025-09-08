"""
🤖 FINANCIAL LLM ASI
Advanced Financial Language Models for Superhuman Financial Intelligence
Ensemble of state-of-the-art financial models with consensus reasoning

@author 35+ Year Experienced AI Engineer
@version 1.0.0 - Financial LLM Implementation
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    pipeline, BertTokenizer, BertForSequenceClassification
)
import numpy as np
from sentence_transformers import SentenceTransformer
import re
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("financial_llm_asi")

@dataclass
class FinancialAnalysis:
    query: str
    analysis: str
    sentiment: str
    confidence: float
    key_insights: List[str]
    risk_factors: List[str]
    recommendations: List[str]
    model_consensus: Dict[str, Any]
    timestamp: datetime

class FinancialLLMASI:
    """
    Advanced Financial Language Model ASI
    Uses ensemble of specialized financial models for superhuman analysis
    """
    
    def __init__(self, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configurations
        self.model_configs = {
            'finbert': {
                'model_name': 'ProsusAI/finbert',
                'type': 'sentiment',
                'weight': 0.3,
                'loaded': False
            },
            'financial_bert': {
                'model_name': 'ahmedrachid/FinancialBERT-Sentiment-Analysis',
                'type': 'sentiment',
                'weight': 0.25,
                'loaded': False
            },
            'distilbert_financial': {
                'model_name': 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis',
                'type': 'sentiment',
                'weight': 0.25,
                'loaded': False
            },
            'general_llm': {
                'model_name': 'microsoft/DialoGPT-medium',
                'type': 'generation',
                'weight': 0.2,
                'loaded': False
            }
        }
        
        # Model storage
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Sentence transformer for embeddings
        self.sentence_transformer = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Financial domain knowledge
        self.financial_keywords = {
            'bullish': ['growth', 'profit', 'increase', 'rise', 'bull', 'positive', 'gain', 'up', 'strong'],
            'bearish': ['loss', 'decline', 'fall', 'bear', 'negative', 'drop', 'down', 'weak', 'crash'],
            'neutral': ['stable', 'flat', 'unchanged', 'sideways', 'consolidation', 'range-bound'],
            'risk': ['risk', 'volatility', 'uncertainty', 'concern', 'warning', 'caution', 'threat'],
            'opportunity': ['opportunity', 'potential', 'upside', 'benefit', 'advantage', 'favorable']
        }
        
        # Financial ratios and metrics knowledge
        self.financial_metrics = {
            'valuation': ['pe_ratio', 'pb_ratio', 'ev_ebitda', 'price_to_sales', 'dividend_yield'],
            'profitability': ['roe', 'roa', 'profit_margin', 'operating_margin', 'ebitda_margin'],
            'liquidity': ['current_ratio', 'quick_ratio', 'cash_ratio', 'working_capital'],
            'leverage': ['debt_to_equity', 'debt_ratio', 'interest_coverage', 'debt_service_coverage'],
            'efficiency': ['asset_turnover', 'inventory_turnover', 'receivables_turnover']
        }
        
        logger.info(f"🤖 Financial LLM ASI initialized on {self.device}")
    
    async def initialize_models(self):
        """Initialize all financial models"""
        logger.info("🔄 Loading financial language models...")
        
        # Load models in parallel
        tasks = []
        for model_key, config in self.model_configs.items():
            task = asyncio.create_task(self._load_model_async(model_key, config))
            tasks.append(task)
        
        # Wait for all models to load
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Check results
        loaded_count = 0
        for i, (model_key, result) in enumerate(zip(self.model_configs.keys(), results)):
            if isinstance(result, Exception):
                logger.error(f"Failed to load {model_key}: {result}")
                self.model_configs[model_key]['loaded'] = False
            else:
                self.model_configs[model_key]['loaded'] = True
                loaded_count += 1
        
        # Load sentence transformer
        try:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Sentence transformer loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
        
        logger.info(f"✅ Loaded {loaded_count}/{len(self.model_configs)} financial models")
    
    async def _load_model_async(self, model_key: str, config: Dict[str, Any]):
        """Load a single model asynchronously"""
        try:
            model_name = config['model_name']
            model_type = config['type']
            
            logger.info(f"Loading {model_key} ({model_name})...")
            
            if model_type == 'sentiment':
                # Load sentiment analysis model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Create pipeline
                sentiment_pipeline = pipeline(
                    'sentiment-analysis',
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == 'cuda' else -1
                )
                
                self.tokenizers[model_key] = tokenizer
                self.models[model_key] = model
                self.pipelines[model_key] = sentiment_pipeline
                
            elif model_type == 'generation':
                # Load text generation model
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForCausalLM.from_pretrained(model_name)
                
                # Add padding token if not present
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                generation_pipeline = pipeline(
                    'text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device == 'cuda' else -1,
                    max_length=512,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                self.tokenizers[model_key] = tokenizer
                self.models[model_key] = model
                self.pipelines[model_key] = generation_pipeline
            
            logger.info(f"✅ {model_key} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_key}: {e}")
            raise e
    
    async def advanced_financial_reasoning(self, query: str, context: Dict[str, Any] = None) -> FinancialAnalysis:
        """
        Perform advanced financial reasoning using ensemble of models
        """
        logger.info(f"🧠 Analyzing: {query[:100]}...")
        
        # Prepare context
        if context is None:
            context = {}
        
        # Run analysis in parallel
        tasks = [
            self._sentiment_analysis(query),
            self._generate_insights(query, context),
            self._extract_key_concepts(query),
            self._risk_assessment(query),
            self._generate_recommendations(query, context)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        sentiment_result = results[0] if not isinstance(results[0], Exception) else {'sentiment': 'neutral', 'confidence': 0.5}
        insights_result = results[1] if not isinstance(results[1], Exception) else []
        concepts_result = results[2] if not isinstance(results[2], Exception) else []
        risk_result = results[3] if not isinstance(results[3], Exception) else []
        recommendations_result = results[4] if not isinstance(results[4], Exception) else []
        
        # Create consensus analysis
        analysis = self._create_consensus_analysis(
            query, sentiment_result, insights_result, concepts_result, 
            risk_result, recommendations_result
        )
        
        return FinancialAnalysis(
            query=query,
            analysis=analysis['analysis'],
            sentiment=sentiment_result['sentiment'],
            confidence=analysis['confidence'],
            key_insights=insights_result,
            risk_factors=risk_result,
            recommendations=recommendations_result,
            model_consensus=analysis['consensus'],
            timestamp=datetime.now()
        )
    
    async def _sentiment_analysis(self, text: str) -> Dict[str, Any]:
        """Perform ensemble sentiment analysis"""
        sentiment_results = []
        
        # Run sentiment analysis with all available models
        for model_key, config in self.model_configs.items():
            if config['type'] == 'sentiment' and config['loaded']:
                try:
                    pipeline = self.pipelines[model_key]
                    result = pipeline(text)
                    
                    # Normalize result format
                    if isinstance(result, list) and len(result) > 0:
                        result = result[0]
                    
                    sentiment_results.append({
                        'model': model_key,
                        'sentiment': result.get('label', 'NEUTRAL').lower(),
                        'confidence': result.get('score', 0.5),
                        'weight': config['weight']
                    })
                    
                except Exception as e:
                    logger.warning(f"Sentiment analysis failed for {model_key}: {e}")
        
        # Calculate weighted consensus
        if sentiment_results:
            return self._calculate_sentiment_consensus(sentiment_results)
        else:
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def _calculate_sentiment_consensus(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate weighted sentiment consensus"""
        sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_weight = 0
        
        for result in results:
            sentiment = result['sentiment']
            confidence = result['confidence']
            weight = result['weight']
            
            # Map different sentiment labels
            if sentiment in ['positive', 'pos', 'bullish']:
                sentiment = 'positive'
            elif sentiment in ['negative', 'neg', 'bearish']:
                sentiment = 'negative'
            else:
                sentiment = 'neutral'
            
            weighted_score = confidence * weight
            sentiment_scores[sentiment] += weighted_score
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for sentiment in sentiment_scores:
                sentiment_scores[sentiment] /= total_weight
        
        # Determine final sentiment
        final_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        final_confidence = sentiment_scores[final_sentiment]
        
        return {
            'sentiment': final_sentiment,
            'confidence': final_confidence,
            'scores': sentiment_scores
        }
    
    async def _generate_insights(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate financial insights using LLM"""
        insights = []
        
        # Use generation models
        for model_key, config in self.model_configs.items():
            if config['type'] == 'generation' and config['loaded']:
                try:
                    pipeline = self.pipelines[model_key]
                    
                    # Create financial prompt
                    prompt = f"Financial Analysis: {query}\n\nKey insights:"
                    
                    # Generate response
                    response = pipeline(prompt, max_length=200, num_return_sequences=1)
                    
                    if response and len(response) > 0:
                        generated_text = response[0]['generated_text']
                        # Extract insights from generated text
                        insight = self._extract_insight_from_text(generated_text, prompt)
                        if insight:
                            insights.append(insight)
                
                except Exception as e:
                    logger.warning(f"Insight generation failed for {model_key}: {e}")
        
        # Add rule-based insights
        rule_based_insights = self._generate_rule_based_insights(query, context)
        insights.extend(rule_based_insights)
        
        return list(set(insights))  # Remove duplicates
    
    def _extract_insight_from_text(self, text: str, prompt: str) -> Optional[str]:
        """Extract meaningful insight from generated text"""
        # Remove the prompt from the generated text
        insight = text.replace(prompt, '').strip()
        
        # Clean up the insight
        insight = re.sub(r'\n+', ' ', insight)
        insight = re.sub(r'\s+', ' ', insight)
        
        # Filter out low-quality insights
        if len(insight) < 20 or len(insight) > 200:
            return None
        
        return insight
    
    def _generate_rule_based_insights(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate insights using financial domain rules"""
        insights = []
        query_lower = query.lower()
        
        # Market trend insights
        if any(word in query_lower for word in ['market', 'index', 'nifty', 'sensex']):
            insights.append("Market analysis should consider both technical and fundamental factors")
        
        # Stock analysis insights
        if any(word in query_lower for word in ['stock', 'share', 'equity']):
            insights.append("Stock valuation requires analysis of financial ratios and industry comparisons")
        
        # Mutual fund insights
        if any(word in query_lower for word in ['fund', 'mutual', 'sip', 'nav']):
            insights.append("Mutual fund selection should focus on long-term performance and expense ratios")
        
        # Risk-related insights
        if any(word in query_lower for word in self.financial_keywords['risk']):
            insights.append("Risk assessment should include both systematic and unsystematic risk factors")
        
        return insights
    
    async def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key financial concepts from text"""
        concepts = []
        text_lower = text.lower()
        
        # Extract financial keywords
        for category, keywords in self.financial_keywords.items():
            found_keywords = [kw for kw in keywords if kw in text_lower]
            if found_keywords:
                concepts.extend(found_keywords)
        
        # Extract financial metrics
        for category, metrics in self.financial_metrics.items():
            found_metrics = [metric for metric in metrics if metric.replace('_', ' ') in text_lower]
            if found_metrics:
                concepts.extend(found_metrics)
        
        # Use sentence transformer for semantic similarity
        if self.sentence_transformer:
            try:
                # Define financial concept templates
                concept_templates = [
                    "financial performance", "market volatility", "investment strategy",
                    "risk management", "portfolio diversification", "asset allocation",
                    "market sentiment", "economic indicators", "company valuation"
                ]
                
                # Get embeddings
                text_embedding = self.sentence_transformer.encode([text])
                template_embeddings = self.sentence_transformer.encode(concept_templates)
                
                # Calculate similarities
                similarities = np.dot(text_embedding, template_embeddings.T)[0]
                
                # Add highly similar concepts
                for i, similarity in enumerate(similarities):
                    if similarity > 0.5:  # Threshold for relevance
                        concepts.append(concept_templates[i])
                        
            except Exception as e:
                logger.warning(f"Semantic concept extraction failed: {e}")
        
        return list(set(concepts))
    
    async def _risk_assessment(self, query: str) -> List[str]:
        """Assess financial risks mentioned in the query"""
        risks = []
        query_lower = query.lower()
        
        # Market risks
        if any(word in query_lower for word in ['market', 'economy', 'recession', 'inflation']):
            risks.append("Market risk due to economic conditions")
        
        # Credit risks
        if any(word in query_lower for word in ['debt', 'credit', 'default', 'bankruptcy']):
            risks.append("Credit risk from counterparty default")
        
        # Liquidity risks
        if any(word in query_lower for word in ['liquidity', 'cash', 'withdrawal']):
            risks.append("Liquidity risk in market stress scenarios")
        
        # Operational risks
        if any(word in query_lower for word in ['management', 'governance', 'fraud']):
            risks.append("Operational risk from management decisions")
        
        # Currency risks
        if any(word in query_lower for word in ['currency', 'forex', 'exchange']):
            risks.append("Currency risk from exchange rate fluctuations")
        
        return risks
    
    async def _generate_recommendations(self, query: str, context: Dict[str, Any]) -> List[str]:
        """Generate financial recommendations"""
        recommendations = []
        query_lower = query.lower()
        
        # Investment recommendations
        if any(word in query_lower for word in ['invest', 'buy', 'portfolio']):
            recommendations.append("Diversify investments across asset classes")
            recommendations.append("Consider risk tolerance and investment horizon")
        
        # Risk management recommendations
        if any(word in query_lower for word in self.financial_keywords['risk']):
            recommendations.append("Implement proper risk management strategies")
            recommendations.append("Regular portfolio rebalancing is recommended")
        
        # Market timing recommendations
        if any(word in query_lower for word in ['timing', 'entry', 'exit']):
            recommendations.append("Avoid market timing; focus on systematic investing")
        
        # Mutual fund recommendations
        if any(word in query_lower for word in ['fund', 'sip', 'mutual']):
            recommendations.append("Choose funds with consistent long-term performance")
            recommendations.append("Monitor expense ratios and fund manager track record")
        
        return recommendations
    
    def _create_consensus_analysis(self, query: str, sentiment: Dict, insights: List[str],
                                 concepts: List[str], risks: List[str], 
                                 recommendations: List[str]) -> Dict[str, Any]:
        """Create consensus analysis from all model outputs"""
        
        # Calculate overall confidence
        confidence_factors = [
            sentiment.get('confidence', 0.5),
            min(len(insights) / 3, 1.0),  # Normalize insight count
            min(len(concepts) / 5, 1.0),  # Normalize concept count
            min(len(risks) / 3, 1.0),     # Normalize risk count
            min(len(recommendations) / 3, 1.0)  # Normalize recommendation count
        ]
        
        overall_confidence = np.mean(confidence_factors)
        
        # Create comprehensive analysis text
        analysis_parts = []
        
        if insights:
            analysis_parts.append(f"Key insights: {'; '.join(insights[:3])}")
        
        if concepts:
            analysis_parts.append(f"Relevant concepts: {', '.join(concepts[:5])}")
        
        if risks:
            analysis_parts.append(f"Risk factors: {'; '.join(risks[:3])}")
        
        analysis_text = ". ".join(analysis_parts) if analysis_parts else "Analysis completed with available data."
        
        return {
            'analysis': analysis_text,
            'confidence': overall_confidence,
            'consensus': {
                'sentiment': sentiment,
                'insight_count': len(insights),
                'concept_count': len(concepts),
                'risk_count': len(risks),
                'recommendation_count': len(recommendations)
            }
        }
    
    @lru_cache(maxsize=100)
    def get_financial_definition(self, term: str) -> str:
        """Get definition of financial terms (cached)"""
        definitions = {
            'pe_ratio': 'Price-to-Earnings ratio measures stock valuation relative to earnings',
            'nav': 'Net Asset Value represents the per-share value of a mutual fund',
            'sip': 'Systematic Investment Plan allows regular investing in mutual funds',
            'aum': 'Assets Under Management represents total market value of investments',
            'expense_ratio': 'Annual fee charged by mutual funds as percentage of assets',
            'volatility': 'Measure of price fluctuation and investment risk',
            'beta': 'Measure of stock sensitivity to market movements',
            'alpha': 'Measure of investment performance relative to benchmark'
        }
        
        return definitions.get(term.lower(), f"Financial term: {term}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models"""
        status = {}
        
        for model_key, config in self.model_configs.items():
            status[model_key] = {
                'model_name': config['model_name'],
                'type': config['type'],
                'loaded': config['loaded'],
                'weight': config['weight']
            }
        
        status['sentence_transformer'] = {
            'loaded': self.sentence_transformer is not None
        }
        
        return status
    
    async def batch_analysis(self, queries: List[str]) -> List[FinancialAnalysis]:
        """Perform batch analysis of multiple queries"""
        logger.info(f"🔄 Processing batch of {len(queries)} queries")
        
        # Process queries in parallel
        tasks = [self.advanced_financial_reasoning(query) for query in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        logger.info(f"✅ Completed batch analysis: {len(valid_results)}/{len(queries)} successful")
        
        return valid_results

# Example usage
async def main():
    asi = FinancialLLMASI()
    
    try:
        # Initialize models
        await asi.initialize_models()
        
        # Test financial reasoning
        query = "What is the outlook for Indian banking stocks given the recent RBI policy changes?"
        
        analysis = await asi.advanced_financial_reasoning(query)
        
        print(f"Query: {analysis.query}")
        print(f"Sentiment: {analysis.sentiment} (Confidence: {analysis.confidence:.2f})")
        print(f"Analysis: {analysis.analysis}")
        print(f"Key Insights: {analysis.key_insights}")
        print(f"Risk Factors: {analysis.risk_factors}")
        print(f"Recommendations: {analysis.recommendations}")
        
        # Test batch analysis
        queries = [
            "Should I invest in technology mutual funds?",
            "What are the risks of investing in small-cap stocks?",
            "How does inflation affect bond investments?"
        ]
        
        batch_results = await asi.batch_analysis(queries)
        print(f"\nBatch Analysis Results: {len(batch_results)} analyses completed")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main())
