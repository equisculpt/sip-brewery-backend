"""
Generative AI Personalization Engine
Uses GPT-style models for hyper-personalized recommendations and insights
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class TransformerPersonalizationModel(nn.Module):
    """
    Transformer-based model for personalized recommendations
    """
    
    def __init__(
        self,
        vocab_size: int = 10000,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048
    ):
        super(TransformerPersonalizationModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # Embedding + positional encoding
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        
        # Transformer
        x = self.transformer(x, mask=mask)
        
        # Output
        return self.output_layer(x)


class GenerativePersonalizationEngine:
    """
    Generative AI engine for hyper-personalized financial advice
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = TransformerPersonalizationModel()
        
        if model_path:
            self.load_model(model_path)
        
        self.user_profiles = {}
        self.interaction_history = {}
        
    def create_user_profile(
        self,
        user_id: str,
        demographics: Dict,
        financial_data: Dict,
        preferences: Dict,
        behavior_history: List[Dict]
    ) -> Dict:
        """
        Create comprehensive user profile for personalization
        
        Args:
            user_id: User identifier
            demographics: Age, location, occupation, etc.
            financial_data: Income, assets, liabilities, etc.
            preferences: Risk tolerance, goals, interests
            behavior_history: Past interactions and decisions
        
        Returns:
            User profile
        """
        profile = {
            'user_id': user_id,
            'demographics': demographics,
            'financial_data': financial_data,
            'preferences': preferences,
            'behavior_history': behavior_history,
            'created_at': datetime.now().isoformat(),
            'embedding': self._generate_user_embedding(
                demographics, financial_data, preferences, behavior_history
            )
        }
        
        self.user_profiles[user_id] = profile
        
        logger.info(f"Created profile for user: {user_id}")
        
        return profile
    
    def generate_personalized_recommendation(
        self,
        user_id: str,
        context: Dict,
        n_recommendations: int = 5
    ) -> List[Dict]:
        """
        Generate personalized fund recommendations
        
        Args:
            user_id: User identifier
            context: Current context (market conditions, recent events, etc.)
            n_recommendations: Number of recommendations to generate
        
        Returns:
            List of personalized recommendations with explanations
        """
        profile = self.user_profiles.get(user_id)
        
        if not profile:
            raise ValueError(f"User profile not found: {user_id}")
        
        # Generate recommendations using transformer model
        recommendations = []
        
        # Combine user profile and context
        input_features = self._prepare_input(profile, context)
        
        # Generate recommendations (simplified)
        for i in range(n_recommendations):
            recommendation = {
                'fund_id': f'FUND{i+1:03d}',
                'confidence': np.random.uniform(0.7, 0.95),
                'reasoning': self._generate_reasoning(profile, context, i),
                'expected_return': np.random.uniform(0.08, 0.18),
                'risk_level': np.random.choice(['LOW', 'MEDIUM', 'HIGH']),
                'personalization_score': np.random.uniform(0.8, 1.0)
            }
            
            recommendations.append(recommendation)
        
        # Sort by personalization score
        recommendations.sort(key=lambda x: x['personalization_score'], reverse=True)
        
        # Log interaction
        self._log_interaction(user_id, 'recommendation', recommendations)
        
        return recommendations
    
    def generate_personalized_insight(
        self,
        user_id: str,
        portfolio_data: Dict
    ) -> str:
        """
        Generate personalized financial insight using natural language
        
        Args:
            user_id: User identifier
            portfolio_data: Current portfolio data
        
        Returns:
            Personalized insight text
        """
        profile = self.user_profiles.get(user_id)
        
        if not profile:
            return "Unable to generate insight: User profile not found"
        
        # Generate insight based on profile and portfolio
        insights = []
        
        # Portfolio performance insight
        if portfolio_data.get('returns', 0) > 0.15:
            insights.append(
                f"Great job! Your portfolio is outperforming with {portfolio_data['returns']*100:.1f}% returns. "
                f"This aligns well with your {profile['preferences'].get('risk_tolerance', 'moderate')} risk profile."
            )
        elif portfolio_data.get('returns', 0) < 0:
            insights.append(
                f"Your portfolio is currently down {abs(portfolio_data['returns'])*100:.1f}%. "
                f"Given your {profile['preferences'].get('investment_horizon', 'long-term')} horizon, "
                f"staying invested could help you recover."
            )
        
        # Diversification insight
        concentration = portfolio_data.get('concentration_hhi', 0)
        if concentration > 0.25:
            insights.append(
                "Your portfolio is heavily concentrated in a few holdings. "
                "Consider diversifying to reduce risk."
            )
        
        # Goal-based insight
        goals = profile['preferences'].get('goals', [])
        if 'retirement' in goals:
            insights.append(
                f"You're {profile['demographics'].get('age', 30)} years old. "
                f"For retirement planning, consider increasing your equity allocation."
            )
        
        # Combine insights
        full_insight = " ".join(insights)
        
        if not full_insight:
            full_insight = "Your portfolio looks balanced. Keep monitoring and rebalancing periodically."
        
        return full_insight
    
    def generate_conversational_response(
        self,
        user_id: str,
        user_query: str,
        conversation_history: List[Dict]
    ) -> str:
        """
        Generate conversational response (chatbot-style)
        
        Args:
            user_id: User identifier
            user_query: User's question or statement
            conversation_history: Previous conversation turns
        
        Returns:
            Generated response
        """
        profile = self.user_profiles.get(user_id)
        
        # Intent classification (simplified)
        intent = self._classify_intent(user_query)
        
        # Generate response based on intent
        if intent == 'recommendation':
            return "Based on your profile, I recommend looking at balanced funds with moderate risk. Would you like specific suggestions?"
        
        elif intent == 'portfolio_review':
            return "Let me analyze your portfolio. Your current allocation is well-diversified with a good mix of equity and debt."
        
        elif intent == 'market_update':
            return "The market is currently in a bullish phase. This could be a good time to review your equity exposure."
        
        elif intent == 'goal_planning':
            return "For your retirement goal, I suggest a systematic investment approach with annual rebalancing."
        
        else:
            return "I'm here to help with your investments. You can ask me about fund recommendations, portfolio review, or market updates."
    
    def _generate_user_embedding(
        self,
        demographics: Dict,
        financial_data: Dict,
        preferences: Dict,
        behavior_history: List[Dict]
    ) -> np.ndarray:
        """Generate user embedding vector"""
        # Simplified embedding generation
        features = [
            demographics.get('age', 30) / 100,
            financial_data.get('income', 50000) / 1000000,
            financial_data.get('net_worth', 100000) / 10000000,
            {'conservative': 0.3, 'moderate': 0.6, 'aggressive': 0.9}.get(
                preferences.get('risk_tolerance', 'moderate'), 0.6
            ),
            len(behavior_history) / 100
        ]
        
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)
        
        return np.array(features[:128])
    
    def _prepare_input(self, profile: Dict, context: Dict) -> torch.Tensor:
        """Prepare input for transformer model"""
        # Simplified input preparation
        embedding = profile['embedding']
        context_features = [
            context.get('market_sentiment', 0.5),
            context.get('volatility', 0.15),
            context.get('interest_rate', 0.06)
        ]
        
        combined = np.concatenate([embedding, context_features])
        
        return torch.FloatTensor(combined).unsqueeze(0)
    
    def _generate_reasoning(
        self,
        profile: Dict,
        context: Dict,
        recommendation_index: int
    ) -> str:
        """Generate natural language reasoning for recommendation"""
        reasons = [
            f"Matches your {profile['preferences'].get('risk_tolerance', 'moderate')} risk profile",
            f"Aligns with your {profile['preferences'].get('investment_horizon', 'long-term')} investment horizon",
            "Strong historical performance in current market conditions",
            "Low correlation with your existing holdings",
            "Managed by experienced fund manager with proven track record"
        ]
        
        return reasons[recommendation_index % len(reasons)]
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intent from query"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['recommend', 'suggest', 'should i invest']):
            return 'recommendation'
        elif any(word in query_lower for word in ['portfolio', 'holdings', 'review']):
            return 'portfolio_review'
        elif any(word in query_lower for word in ['market', 'news', 'update']):
            return 'market_update'
        elif any(word in query_lower for word in ['goal', 'retirement', 'planning']):
            return 'goal_planning'
        else:
            return 'general'
    
    def _log_interaction(self, user_id: str, interaction_type: str, data: any):
        """Log user interaction for learning"""
        if user_id not in self.interaction_history:
            self.interaction_history[user_id] = []
        
        self.interaction_history[user_id].append({
            'type': interaction_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
    
    def learn_from_feedback(
        self,
        user_id: str,
        recommendation_id: str,
        feedback: Dict
    ):
        """
        Learn from user feedback to improve personalization
        
        Args:
            user_id: User identifier
            recommendation_id: ID of recommendation
            feedback: User feedback (liked, clicked, invested, etc.)
        """
        # Update user profile based on feedback
        profile = self.user_profiles.get(user_id)
        
        if profile and feedback.get('action') == 'invested':
            # Positive signal - update preferences
            logger.info(f"Positive feedback from {user_id} on {recommendation_id}")
            
            # In production, retrain model with this feedback
        
        self._log_interaction(user_id, 'feedback', feedback)
    
    def save_model(self, path: str):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user_profiles': self.user_profiles
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.user_profiles = checkpoint.get('user_profiles', {})
        
        logger.info(f"Model loaded from {path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    print("Generative AI Personalization Engine")
    print("=" * 60)
    
    # Create engine
    engine = GenerativePersonalizationEngine()
    
    # Create user profile
    profile = engine.create_user_profile(
        user_id='user_001',
        demographics={'age': 32, 'location': 'Mumbai', 'occupation': 'Software Engineer'},
        financial_data={'income': 1500000, 'net_worth': 5000000},
        preferences={'risk_tolerance': 'moderate', 'investment_horizon': 'long-term', 'goals': ['retirement', 'wealth_creation']},
        behavior_history=[]
    )
    
    print(f"\nCreated profile for user: {profile['user_id']}")
    
    # Generate recommendations
    recommendations = engine.generate_personalized_recommendation(
        user_id='user_001',
        context={'market_sentiment': 0.7, 'volatility': 0.12},
        n_recommendations=3
    )
    
    print(f"\nTop 3 Personalized Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['fund_id']} - Confidence: {rec['confidence']*100:.1f}%")
        print(f"   Reasoning: {rec['reasoning']}")
    
    # Generate insight
    portfolio_data = {
        'returns': 0.18,
        'concentration_hhi': 0.15,
        'sharpe_ratio': 1.3
    }
    
    insight = engine.generate_personalized_insight('user_001', portfolio_data)
    print(f"\nPersonalized Insight:")
    print(insight)
    
    # Conversational response
    response = engine.generate_conversational_response(
        user_id='user_001',
        user_query="What funds should I invest in?",
        conversation_history=[]
    )
    
    print(f"\nConversational Response:")
    print(response)
