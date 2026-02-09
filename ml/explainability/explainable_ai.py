"""
Explainable AI System using SHAP and LIME
Provides interpretable explanations for all ML model predictions
"""

import numpy as np
import pandas as pd
import torch
import shap
from lime import lime_tabular
from typing import Dict, List, Optional
import logging
import matplotlib.pyplot as plt
import json

logger = logging.getLogger(__name__)


class ModelExplainer:
    """
    Unified explainer for all ML models
    Supports SHAP and LIME explanations
    """
    
    def __init__(self, model, model_type: str = 'pytorch'):
        self.model = model
        self.model_type = model_type
        self.shap_explainer = None
        self.lime_explainer = None
        
    def initialize_shap(self, background_data: np.ndarray):
        """Initialize SHAP explainer with background data"""
        if self.model_type == 'pytorch':
            # For PyTorch models
            def model_predict(x):
                with torch.no_grad():
                    x_tensor = torch.FloatTensor(x)
                    output = self.model(x_tensor)
                    if isinstance(output, dict):
                        # Handle dict outputs (e.g., from our models)
                        return output.get('weights', output.get('returns', x_tensor)).numpy()
                    return output.numpy()
            
            self.shap_explainer = shap.KernelExplainer(
                model_predict,
                background_data
            )
        else:
            self.shap_explainer = shap.Explainer(self.model, background_data)
        
        logger.info("SHAP explainer initialized")
    
    def initialize_lime(self, training_data: np.ndarray, feature_names: List[str]):
        """Initialize LIME explainer"""
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            mode='regression',
            verbose=False
        )
        logger.info("LIME explainer initialized")
    
    def explain_prediction_shap(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Generate SHAP explanation for a single prediction
        
        Returns:
            Dictionary with SHAP values and visualization data
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not initialized. Call initialize_shap first.")
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(instance)
        
        # Create explanation dictionary
        explanation = {
            'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'base_value': float(self.shap_explainer.expected_value),
            'feature_importance': {}
        }
        
        # Rank features by importance
        if feature_names:
            importance_dict = {
                feature_names[i]: float(abs(shap_values[i]))
                for i in range(len(shap_values))
            }
            explanation['feature_importance'] = dict(
                sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            )
        
        return explanation
    
    def explain_prediction_lime(
        self,
        instance: np.ndarray,
        num_features: int = 10
    ) -> Dict:
        """
        Generate LIME explanation for a single prediction
        
        Returns:
            Dictionary with LIME explanation
        """
        if self.lime_explainer is None:
            raise ValueError("LIME explainer not initialized. Call initialize_lime first.")
        
        # Generate explanation
        explanation = self.lime_explainer.explain_instance(
            instance,
            self.model.predict if hasattr(self.model, 'predict') else lambda x: self.model(torch.FloatTensor(x)).detach().numpy(),
            num_features=num_features
        )
        
        # Extract feature importance
        feature_importance = dict(explanation.as_list())
        
        return {
            'feature_importance': feature_importance,
            'score': explanation.score,
            'local_pred': explanation.local_pred
        }
    
    def generate_comprehensive_explanation(
        self,
        instance: np.ndarray,
        feature_names: List[str],
        prediction: any,
        model_name: str
    ) -> Dict:
        """
        Generate comprehensive explanation combining multiple methods
        
        Returns:
            Complete explanation with reasoning
        """
        explanation = {
            'model': model_name,
            'prediction': prediction,
            'timestamp': pd.Timestamp.now().isoformat(),
            'explanations': {}
        }
        
        # SHAP explanation
        try:
            shap_exp = self.explain_prediction_shap(instance, feature_names)
            explanation['explanations']['shap'] = shap_exp
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
        
        # LIME explanation
        try:
            lime_exp = self.explain_prediction_lime(instance)
            explanation['explanations']['lime'] = lime_exp
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
        
        # Generate human-readable summary
        explanation['summary'] = self._generate_summary(
            explanation['explanations'],
            feature_names,
            prediction
        )
        
        return explanation
    
    def _generate_summary(
        self,
        explanations: Dict,
        feature_names: List[str],
        prediction: any
    ) -> str:
        """Generate human-readable summary of explanations"""
        
        summary_parts = []
        
        # Get top features from SHAP
        if 'shap' in explanations:
            top_features = list(explanations['shap']['feature_importance'].items())[:3]
            summary_parts.append("Top influential features:")
            for feature, importance in top_features:
                summary_parts.append(f"  - {feature}: {importance:.4f}")
        
        # Add prediction context
        if isinstance(prediction, dict):
            if 'weights' in prediction:
                summary_parts.append(f"\nRecommended portfolio weights based on these factors")
            elif 'returns' in prediction:
                summary_parts.append(f"\nPredicted returns influenced by these features")
        
        return "\n".join(summary_parts)


class PortfolioDecisionExplainer:
    """
    Specialized explainer for portfolio management decisions
    """
    
    def __init__(self):
        self.decision_templates = {
            'BUY': "Recommending to BUY {fund} because {reasons}",
            'SELL': "Recommending to SELL {fund} because {reasons}",
            'HOLD': "Recommending to HOLD current positions because {reasons}",
            'REBALANCE': "Recommending portfolio REBALANCE because {reasons}"
        }
    
    def explain_portfolio_decision(
        self,
        decision: Dict,
        market_context: Dict,
        user_profile: Dict,
        shap_values: Optional[Dict] = None
    ) -> Dict:
        """
        Explain a portfolio management decision
        
        Returns:
            Comprehensive explanation with reasoning chain
        """
        
        explanation = {
            'decision': decision,
            'reasoning_chain': [],
            'confidence_factors': [],
            'risk_factors': [],
            'alternative_actions': []
        }
        
        # Build reasoning chain
        action_type = decision.get('type', 'HOLD')
        
        # Market-based reasoning
        if market_context.get('regime') == 'BULL':
            explanation['reasoning_chain'].append({
                'factor': 'Market Regime',
                'value': 'Bullish',
                'impact': 'Positive',
                'weight': 0.3,
                'explanation': 'Current market conditions favor growth investments'
            })
        elif market_context.get('regime') == 'BEAR':
            explanation['reasoning_chain'].append({
                'factor': 'Market Regime',
                'value': 'Bearish',
                'impact': 'Negative',
                'weight': 0.3,
                'explanation': 'Market downturn suggests defensive positioning'
            })
        
        # Risk-based reasoning
        if user_profile.get('risk_tolerance') == 'conservative':
            explanation['reasoning_chain'].append({
                'factor': 'Risk Profile',
                'value': 'Conservative',
                'impact': 'Constraint',
                'weight': 0.25,
                'explanation': 'User preference limits exposure to volatile assets'
            })
        
        # Portfolio-based reasoning
        if decision.get('concentration_risk', False):
            explanation['risk_factors'].append({
                'risk': 'Concentration',
                'severity': 'High',
                'mitigation': 'Diversification recommended'
            })
        
        # SHAP-based reasoning
        if shap_values:
            top_features = sorted(
                shap_values.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]
            
            for feature, value in top_features:
                explanation['reasoning_chain'].append({
                    'factor': feature,
                    'value': value,
                    'impact': 'Positive' if value > 0 else 'Negative',
                    'weight': abs(value),
                    'explanation': f'{feature} contributed {"positively" if value > 0 else "negatively"} to this decision'
                })
        
        # Confidence factors
        explanation['confidence_factors'] = [
            {
                'factor': 'Model Confidence',
                'value': decision.get('confidence', 0.5),
                'threshold': 0.75
            },
            {
                'factor': 'Data Quality',
                'value': 0.9,  # Would be calculated from actual data
                'threshold': 0.8
            }
        ]
        
        # Alternative actions
        explanation['alternative_actions'] = self._generate_alternatives(
            decision,
            market_context
        )
        
        # Generate natural language summary
        explanation['summary'] = self._generate_decision_summary(
            decision,
            explanation['reasoning_chain']
        )
        
        return explanation
    
    def _generate_alternatives(
        self,
        decision: Dict,
        market_context: Dict
    ) -> List[Dict]:
        """Generate alternative actions with explanations"""
        
        alternatives = []
        action_type = decision.get('type')
        
        if action_type == 'BUY':
            alternatives.append({
                'action': 'HOLD',
                'reason': 'Wait for better entry point',
                'confidence': 0.6
            })
            alternatives.append({
                'action': 'BUY_DIFFERENT',
                'reason': 'Consider alternative fund with similar profile',
                'confidence': 0.5
            })
        elif action_type == 'SELL':
            alternatives.append({
                'action': 'HOLD',
                'reason': 'Market may recover',
                'confidence': 0.4
            })
            alternatives.append({
                'action': 'PARTIAL_SELL',
                'reason': 'Reduce exposure gradually',
                'confidence': 0.7
            })
        
        return alternatives
    
    def _generate_decision_summary(
        self,
        decision: Dict,
        reasoning_chain: List[Dict]
    ) -> str:
        """Generate natural language summary"""
        
        action = decision.get('type', 'HOLD')
        fund = decision.get('fund_id', 'portfolio')
        confidence = decision.get('confidence', 0.5)
        
        summary = f"Decision: {action} {fund}\n"
        summary += f"Confidence: {confidence*100:.1f}%\n\n"
        summary += "Key Factors:\n"
        
        for i, reason in enumerate(reasoning_chain[:3], 1):
            summary += f"{i}. {reason['factor']}: {reason['explanation']}\n"
        
        return summary


class ExplainabilityDashboard:
    """
    Dashboard for visualizing model explanations
    """
    
    def __init__(self):
        self.explanations_history = []
    
    def create_feature_importance_plot(
        self,
        feature_importance: Dict,
        title: str = "Feature Importance"
    ) -> str:
        """
        Create feature importance visualization
        
        Returns:
            Path to saved plot
        """
        features = list(feature_importance.keys())[:10]
        values = [feature_importance[f] for f in features]
        
        plt.figure(figsize=(10, 6))
        plt.barh(features, values)
        plt.xlabel('Importance')
        plt.title(title)
        plt.tight_layout()
        
        plot_path = f'feature_importance_{pd.Timestamp.now().timestamp()}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def create_decision_timeline(
        self,
        decisions: List[Dict]
    ) -> str:
        """Create timeline visualization of decisions"""
        
        timestamps = [pd.Timestamp(d['timestamp']) for d in decisions]
        confidences = [d.get('confidence', 0.5) for d in decisions]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, confidences, marker='o')
        plt.xlabel('Time')
        plt.ylabel('Confidence')
        plt.title('Decision Confidence Over Time')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_path = f'decision_timeline_{pd.Timestamp.now().timestamp()}.png'
        plt.savefig(plot_path)
        plt.close()
        
        return plot_path
    
    def generate_explanation_report(
        self,
        explanation: Dict,
        output_format: str = 'json'
    ) -> str:
        """
        Generate comprehensive explanation report
        
        Args:
            explanation: Explanation dictionary
            output_format: 'json', 'html', or 'pdf'
        
        Returns:
            Path to generated report
        """
        if output_format == 'json':
            report_path = f'explanation_report_{pd.Timestamp.now().timestamp()}.json'
            with open(report_path, 'w') as f:
                json.dump(explanation, f, indent=2)
            return report_path
        
        elif output_format == 'html':
            html_content = self._generate_html_report(explanation)
            report_path = f'explanation_report_{pd.Timestamp.now().timestamp()}.html'
            with open(report_path, 'w') as f:
                f.write(html_content)
            return report_path
        
        return ""
    
    def _generate_html_report(self, explanation: Dict) -> str:
        """Generate HTML report"""
        
        html = f"""
        <html>
        <head>
            <title>Model Explanation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f5f5f5; }}
                .factor {{ margin: 10px 0; padding: 10px; background: white; }}
            </style>
        </head>
        <body>
            <h1>Model Explanation Report</h1>
            <div class="section">
                <h2>Summary</h2>
                <p>{explanation.get('summary', 'No summary available')}</p>
            </div>
            <div class="section">
                <h2>Reasoning Chain</h2>
                {self._format_reasoning_chain(explanation.get('reasoning_chain', []))}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _format_reasoning_chain(self, chain: List[Dict]) -> str:
        """Format reasoning chain as HTML"""
        html = ""
        for reason in chain:
            html += f"""
            <div class="factor">
                <strong>{reason.get('factor', 'Unknown')}</strong>: 
                {reason.get('explanation', 'No explanation')}
                (Impact: {reason.get('impact', 'Unknown')}, 
                Weight: {reason.get('weight', 0):.2f})
            </div>
            """
        return html


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    print("Explainable AI System initialized")
    print("\nCapabilities:")
    print("- SHAP explanations for feature importance")
    print("- LIME explanations for local interpretability")
    print("- Portfolio decision explanations")
    print("- Visualization dashboards")
    print("- Comprehensive reporting")
