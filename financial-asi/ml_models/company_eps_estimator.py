#!/usr/bin/env python3
"""
🏢💰 Company EPS & Revenue Estimator for Financial ASI
Uses satellite data + AI to predict individual company earnings
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import joblib

logger = logging.getLogger(__name__)

class CompanyEPSEstimator:
    """🏢 AI-powered company EPS and revenue estimator"""
    
    def __init__(self, companies_config: Dict):
        self.companies_config = companies_config
        self.models = {}
        self.scalers = {}
        
        # Model cache directory
        self.model_dir = Path("models/company_eps")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Company-specific configurations
        self.company_features = {
            'TITAN': {
                'primary_features': ['parking_density', 'footfall_indicator', 'nightlight_activity'],
                'sector': 'retail',
                'seasonal_factors': [0.8, 0.9, 1.1, 1.3, 1.2, 0.9, 0.8, 0.9, 1.0, 1.4, 1.5, 1.2]
            },
            'MARUTI': {
                'primary_features': ['factory_activity', 'inventory_lots', 'truck_traffic'],
                'sector': 'auto',
                'seasonal_factors': [0.9, 0.8, 1.0, 1.1, 0.9, 0.8, 0.9, 1.2, 1.3, 1.4, 1.2, 1.0]
            },
            'RELIANCE': {
                'primary_features': ['storage_levels', 'flaring_activity', 'tanker_traffic'],
                'sector': 'oil_gas',
                'seasonal_factors': [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 1.0, 1.0, 1.0, 1.1]
            },
            'TATASTEEL': {
                'primary_features': ['mining_activity', 'stockpile_volume', 'rail_traffic'],
                'sector': 'mining',
                'seasonal_factors': [1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 1.0, 1.1, 1.2, 1.1, 1.0]
            },
            'HUL': {
                'primary_features': ['consumer_activity', 'distribution_network', 'rural_demand'],
                'sector': 'fmcg',
                'seasonal_factors': [1.0, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 1.0, 1.1, 1.3, 1.4, 1.2]
            }
        }
        
        logger.info("🏢 Company EPS Estimator initialized")
    
    async def predict(self, all_data: Dict, image_analysis: Dict, sector_forecasts: Dict) -> Dict:
        """Generate EPS and revenue predictions for all companies"""
        logger.info("🔮 Generating company EPS and revenue predictions...")
        
        company_predictions = {}
        
        try:
            # Get all companies from config
            all_companies = []
            for sector_companies in self.companies_config.values():
                all_companies.extend(sector_companies)
            
            for company in all_companies:
                if company in self.company_features:
                    logger.info(f"📊 Predicting {company} performance...")
                    
                    prediction = await self.predict_company_performance(
                        company, all_data, image_analysis, sector_forecasts
                    )
                    
                    company_predictions[company] = prediction
            
            logger.info(f"✅ Generated predictions for {len(company_predictions)} companies")
            return company_predictions
            
        except Exception as e:
            logger.error(f"❌ Company prediction failed: {e}")
            raise
    
    async def predict_company_performance(self, company: str, all_data: Dict, 
                                        image_analysis: Dict, sector_forecasts: Dict) -> Dict:
        """Predict individual company performance using satellite + AI"""
        try:
            company_config = self.company_features[company]
            
            # Extract features for this company
            features = self.extract_company_features(company, all_data, image_analysis, sector_forecasts)
            
            # Load or train model for this company
            model = await self.get_or_train_model(company, features)
            
            # Generate predictions
            predictions = self.generate_company_predictions(company, model, features, company_config)
            
            return {
                'company': company,
                'sector': company_config['sector'],
                'predictions': predictions,
                'confidence': predictions['confidence'],
                'timestamp': datetime.now().isoformat(),
                'prediction_horizon': '30_days'
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to predict {company} performance: {e}")
            return self.get_fallback_prediction(company)
    
    def extract_company_features(self, company: str, all_data: Dict, 
                               image_analysis: Dict, sector_forecasts: Dict) -> np.ndarray:
        """Extract satellite and macro features for company prediction"""
        try:
            company_config = self.company_features[company]
            features = []
            
            # Satellite-based features
            if company_config['sector'] == 'retail':
                # Retail-specific satellite features
                retail_data = image_analysis.get('vehicle_counts', {}).get(company, {})
                for location, data in retail_data.items():
                    features.extend([
                        data.get('parking_density', 0.5),
                        data.get('vehicle_count', 100) / 300.0,  # Normalize
                        data.get('footfall_indicator', 0.5)
                    ])
                
            elif company_config['sector'] == 'auto':
                # Auto manufacturing satellite features
                manufacturing_data = all_data.get('satellite', {}).get('manufacturing_locations', {}).get(company, {})
                for location, data in manufacturing_data.items():
                    features.extend([
                        data.get('factory_activity', 0.7),
                        data.get('truck_traffic', 50) / 150.0,  # Normalize
                        data.get('inventory_lots', 0.5)
                    ])
            
            # Add sector forecast features
            sector_data = sector_forecasts.get(company_config['sector'], {})
            features.extend([
                sector_data.get('growth_forecast', 0.05),
                sector_data.get('confidence', 0.7)
            ])
            
            # Add seasonal factor
            current_month = datetime.now().month - 1  # 0-indexed
            seasonal_factor = company_config['seasonal_factors'][current_month]
            features.append(seasonal_factor)
            
            # Ensure consistent feature vector size
            while len(features) < 10:
                features.append(0.5)
            
            return np.array(features[:10])
            
        except Exception as e:
            logger.error(f"❌ Feature extraction failed for {company}: {e}")
            return np.array([0.5] * 10)
    
    async def get_or_train_model(self, company: str, features: np.ndarray):
        """Get existing model or train new one for company"""
        try:
            model_path = self.model_dir / f"{company}_model.joblib"
            
            if model_path.exists():
                model = joblib.load(model_path)
                logger.info(f"📦 Loaded existing model for {company}")
            else:
                model = await self.train_company_model(company, features)
                joblib.dump(model, model_path)
                logger.info(f"💾 Saved new model for {company}")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Model loading/training failed for {company}: {e}")
            return self.create_fallback_model()
    
    async def train_company_model(self, company: str, sample_features: np.ndarray):
        """Train AI model for company using synthetic historical data"""
        try:
            logger.info(f"🤖 Training AI model for {company}...")
            
            # Generate synthetic training data
            n_samples = 500
            company_config = self.company_features[company]
            
            X = []
            y_revenue = []
            y_eps = []
            
            for i in range(n_samples):
                # Generate synthetic features with variations
                features = sample_features.copy()
                noise = np.random.normal(0, 0.1, len(features))
                features = np.clip(features + noise, 0, 2)
                
                # Generate target variables based on sector
                if company_config['sector'] == 'retail':
                    base_revenue_growth = features[0] * 0.3 + features[1] * 0.4
                    base_eps_growth = base_revenue_growth * 0.8
                elif company_config['sector'] == 'auto':
                    base_revenue_growth = features[0] * 0.4 + features[1] * 0.3
                    base_eps_growth = base_revenue_growth * 1.2
                else:
                    base_revenue_growth = features[0] * 0.3 + features[1] * 0.3
                    base_eps_growth = base_revenue_growth * 0.9
                
                # Add seasonal and random factors
                seasonal_factor = features[-1]
                revenue_growth = base_revenue_growth * seasonal_factor + np.random.normal(0, 0.05)
                eps_growth = base_eps_growth * seasonal_factor + np.random.normal(0, 0.08)
                
                X.append(features)
                y_revenue.append(revenue_growth)
                y_eps.append(eps_growth)
            
            X = np.array(X)
            y_revenue = np.array(y_revenue)
            y_eps = np.array(y_eps)
            
            # Train models
            revenue_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            eps_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            
            revenue_model.fit(X, y_revenue)
            eps_model.fit(X, y_eps)
            
            model = {
                'revenue_model': revenue_model,
                'eps_model': eps_model,
                'scaler': StandardScaler().fit(X)
            }
            
            logger.info(f"✅ Trained AI model for {company}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Model training failed for {company}: {e}")
            return self.create_fallback_model()
    
    def create_fallback_model(self):
        """Create simple fallback model"""
        return {
            'revenue_model': None,
            'eps_model': None,
            'scaler': None
        }
    
    def generate_company_predictions(self, company: str, model: Dict, 
                                   features: np.ndarray, company_config: Dict) -> Dict:
        """Generate comprehensive predictions for company"""
        try:
            if model['revenue_model'] is not None:
                # Use trained AI model
                features_scaled = model['scaler'].transform(features.reshape(1, -1))
                revenue_growth = model['revenue_model'].predict(features_scaled)[0]
                eps_growth = model['eps_model'].predict(features_scaled)[0]
                confidence = 0.75
            else:
                # Use rule-based fallback
                revenue_growth, eps_growth, confidence = self.generate_fallback_predictions(
                    company, features, company_config
                )
            
            predictions = {
                'revenue_growth': float(revenue_growth),
                'eps_growth': float(eps_growth),
                'confidence': float(confidence),
                'price_target_change': float(eps_growth * 0.8 + revenue_growth * 0.2),
                'recommendation': self.generate_recommendation(revenue_growth, eps_growth, confidence),
                'risk_factors': self.identify_risk_factors(company_config['sector']),
                'catalysts': self.identify_catalysts(company_config['sector'])
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Prediction generation failed for {company}: {e}")
            return {
                'revenue_growth': 0.05,
                'eps_growth': 0.05,
                'confidence': 0.5,
                'price_target_change': 0.05,
                'recommendation': 'HOLD',
                'risk_factors': ['Market volatility'],
                'catalysts': ['Operational improvements']
            }
    
    def generate_fallback_predictions(self, company: str, features: np.ndarray, 
                                    company_config: Dict) -> Tuple[float, float, float]:
        """Generate rule-based predictions as fallback"""
        try:
            avg_feature = np.mean(features)
            seasonal_factor = company_config['seasonal_factors'][datetime.now().month - 1]
            
            if company_config['sector'] == 'retail':
                revenue_growth = (avg_feature - 0.5) * 0.3 * seasonal_factor
                eps_growth = revenue_growth * 0.8
            elif company_config['sector'] == 'auto':
                revenue_growth = (avg_feature - 0.5) * 0.4 * seasonal_factor
                eps_growth = revenue_growth * 1.2
            else:
                revenue_growth = (avg_feature - 0.5) * 0.35 * seasonal_factor
                eps_growth = revenue_growth * 0.9
            
            confidence = 0.6
            return revenue_growth, eps_growth, confidence
            
        except Exception as e:
            logger.error(f"❌ Fallback prediction failed for {company}: {e}")
            return 0.05, 0.05, 0.5
    
    def generate_recommendation(self, revenue_growth: float, eps_growth: float, confidence: float) -> str:
        """Generate investment recommendation"""
        if confidence < 0.6:
            return 'HOLD'
        
        combined_growth = (revenue_growth + eps_growth) / 2
        
        if combined_growth > 0.15 and confidence > 0.8:
            return 'STRONG_BUY'
        elif combined_growth > 0.08 and confidence > 0.7:
            return 'BUY'
        elif combined_growth > 0.02:
            return 'HOLD'
        else:
            return 'SELL'
    
    def identify_risk_factors(self, sector: str) -> List[str]:
        """Identify key risk factors by sector"""
        risk_map = {
            'retail': ['Consumer sentiment risk', 'Competition risk', 'Real estate cost risk'],
            'auto': ['Commodity price risk', 'Regulatory risk', 'Technology disruption'],
            'oil_gas': ['Oil price volatility', 'Regulatory changes', 'Environmental concerns'],
            'mining': ['Commodity price volatility', 'Environmental regulations', 'Labor issues'],
            'fmcg': ['Raw material inflation', 'Rural demand slowdown', 'Competition']
        }
        return risk_map.get(sector, ['Market risk', 'Operational risk', 'Regulatory risk'])
    
    def identify_catalysts(self, sector: str) -> List[str]:
        """Identify positive catalysts by sector"""
        catalyst_map = {
            'retail': ['Festive season demand', 'Store expansion', 'Digital transformation'],
            'auto': ['New model launches', 'Export growth', 'EV transition'],
            'oil_gas': ['Refining margin expansion', 'Petrochemical growth', 'Gas monetization'],
            'mining': ['Infrastructure demand', 'Export opportunities', 'Operational efficiency'],
            'fmcg': ['Rural recovery', 'Premium product mix', 'Market share gains']
        }
        return catalyst_map.get(sector, ['Market expansion', 'Operational improvements', 'Cost optimization'])
    
    def get_fallback_prediction(self, company: str) -> Dict:
        """Get fallback prediction for company"""
        return {
            'company': company,
            'sector': 'unknown',
            'predictions': {
                'revenue_growth': 0.05,
                'eps_growth': 0.05,
                'confidence': 0.5,
                'price_target_change': 0.05,
                'recommendation': 'HOLD',
                'risk_factors': ['Data unavailable'],
                'catalysts': ['Market recovery']
            },
            'confidence': 0.5,
            'timestamp': datetime.now().isoformat(),
            'prediction_horizon': '30_days'
        }

# Test function
async def test_company_eps_estimator():
    """Test the company EPS estimator"""
    print("🏢 Testing Company EPS Estimator...")
    
    companies_config = {
        'retail': ['TITAN', 'DMART'],
        'auto': ['MARUTI'],
        'oil_gas': ['RELIANCE'],
        'mining': ['TATASTEEL'],
        'fmcg': ['HUL']
    }
    
    estimator = CompanyEPSEstimator(companies_config)
    
    # Mock data
    all_data = {'satellite': {'nightlight_data': {}, 'manufacturing_locations': {}}}
    image_analysis = {'vehicle_counts': {}}
    sector_forecasts = {'retail': {'growth_forecast': 0.08, 'confidence': 0.7}}
    
    try:
        predictions = await estimator.predict(all_data, image_analysis, sector_forecasts)
        
        print(f"✅ Generated predictions for {len(predictions)} companies:")
        for company, pred in predictions.items():
            print(f"   {company}: Revenue {pred['predictions']['revenue_growth']:+.1%}, "
                  f"EPS {pred['predictions']['eps_growth']:+.1%}, "
                  f"Rec: {pred['predictions']['recommendation']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_company_eps_estimator())
