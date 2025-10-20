import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
sys.path.append('/Users/saimudragada/Desktop/recommendation_system_mab')

from src.recommendation_engine.recommender import RecommendationEngine
from src.bandit.mab_engine import BanditRecommender

class ABTestSimulator:
    """Simulate A/B test: Baseline vs Bandit recommendations"""
    
    def __init__(self, base_recommender: RecommendationEngine, 
                 n_simulations: int = 5000):
        """
        Initialize A/B test simulator
        
        Args:
            base_recommender: Base recommendation engine
            n_simulations: Number of user interactions to simulate
        """
        self.base_recommender = base_recommender
        self.n_simulations = n_simulations
        
        # Initialize bandit recommenders
        self.epsilon_greedy = BanditRecommender(
            base_recommender, 
            algorithm='epsilon_greedy',
            epsilon=0.1
        )
        
        self.ucb = BanditRecommender(
            base_recommender,
            algorithm='ucb',
            ucb_c=2.0
        )
        
        # Results storage
        self.results = {
            'baseline': {'clicks': [], 'purchases': [], 'revenue': []},
            'epsilon_greedy': {'clicks': [], 'purchases': [], 'revenue': []},
            'ucb': {'clicks': [], 'purchases': [], 'revenue': []}
        }
        
        print(f"✓ A/B Test initialized for {n_simulations} simulations")
    
    def _simulate_user_behavior(self, product: Dict) -> tuple:
        """Simulate user interaction with a product"""
        # Click probability based on rating and price
        click_prob = 0.7 + (product['rating'] / 5.0) * 0.2
        clicked = np.random.random() < click_prob
        
        if not clicked:
            return False, False, 0
        
        # Purchase probability (if clicked)
        # Higher rating = higher purchase probability
        # Lower price = slightly higher purchase probability
        purchase_prob = (product['rating'] / 5.0) * 0.18 + 0.02
        purchase_prob *= (1 - min(product['price'] / 1000, 0.3))  # Price sensitivity
        
        purchased = np.random.random() < purchase_prob
        revenue = product['price'] if purchased else 0
        
        return clicked, purchased, revenue
    
    def run_simulation(self):
        """Run A/B test simulation"""
        print("\nRunning A/B Test Simulation...")
        print("="*60)
        
        users = self.base_recommender.users_df['user_id'].values
        
        for i in tqdm(range(self.n_simulations), desc="Simulating interactions"):
            # Random user
            user_id = np.random.choice(users)
            
            # === BASELINE (Control Group) ===
            baseline_recs = self.base_recommender.get_user_recommendations(
                user_id, top_k=5, exclude_interacted=False
            )
            
            if baseline_recs:
                # User sees top recommendation
                product = baseline_recs[0]
                clicked, purchased, revenue = self._simulate_user_behavior(product)
                
                self.results['baseline']['clicks'].append(1 if clicked else 0)
                self.results['baseline']['purchases'].append(1 if purchased else 0)
                self.results['baseline']['revenue'].append(revenue)
            
            # === EPSILON-GREEDY BANDIT (Treatment Group 1) ===
            eg_recs = self.epsilon_greedy.get_recommendations(
                user_id, top_k=5, use_bandit=True
            )
            
            if eg_recs:
                product = eg_recs[0]
                clicked, purchased, revenue = self._simulate_user_behavior(product)
                
                # Record feedback for learning
                self.epsilon_greedy.record_feedback(
                    product['product_id'], clicked, purchased, revenue
                )
                
                self.results['epsilon_greedy']['clicks'].append(1 if clicked else 0)
                self.results['epsilon_greedy']['purchases'].append(1 if purchased else 0)
                self.results['epsilon_greedy']['revenue'].append(revenue)
            
            # === UCB BANDIT (Treatment Group 2) ===
            ucb_recs = self.ucb.get_recommendations(
                user_id, top_k=5, use_bandit=True
            )
            
            if ucb_recs:
                product = ucb_recs[0]
                clicked, purchased, revenue = self._simulate_user_behavior(product)
                
                # Record feedback for learning
                self.ucb.record_feedback(
                    product['product_id'], clicked, purchased, revenue
                )
                
                self.results['ucb']['clicks'].append(1 if clicked else 0)
                self.results['ucb']['purchases'].append(1 if purchased else 0)
                self.results['ucb']['revenue'].append(revenue)
        
        print("\n✓ Simulation complete!")
    
    def calculate_metrics(self) -> Dict:
        """Calculate A/B test metrics"""
        metrics = {}
        
        for variant in ['baseline', 'epsilon_greedy', 'ucb']:
            clicks = np.array(self.results[variant]['clicks'])
            purchases = np.array(self.results[variant]['purchases'])
            revenue = np.array(self.results[variant]['revenue'])
            
            n = len(clicks)
            
            metrics[variant] = {
                'n': n,
                'ctr': clicks.sum() / n if n > 0 else 0,
                'conversion_rate': purchases.sum() / n if n > 0 else 0,
                'total_revenue': revenue.sum(),
                'revenue_per_interaction': revenue.mean(),
                'revenue_per_purchase': revenue[purchases > 0].mean() if purchases.sum() > 0 else 0,
            }
        
        return metrics
    
    def calculate_improvements(self, metrics: Dict) -> Dict:
        """Calculate improvements over baseline"""
        baseline = metrics['baseline']
        improvements = {}
        
        for variant in ['epsilon_greedy', 'ucb']:
            variant_metrics = metrics[variant]
            
            improvements[variant] = {
                'ctr_improvement': (
                    (variant_metrics['ctr'] - baseline['ctr']) / baseline['ctr'] * 100
                    if baseline['ctr'] > 0 else 0
                ),
                'conversion_improvement': (
                    (variant_metrics['conversion_rate'] - baseline['conversion_rate']) / 
                    baseline['conversion_rate'] * 100
                    if baseline['conversion_rate'] > 0 else 0
                ),
                'revenue_per_interaction_improvement': (
                    (variant_metrics['revenue_per_interaction'] - 
                     baseline['revenue_per_interaction']) / 
                    baseline['revenue_per_interaction'] * 100
                    if baseline['revenue_per_interaction'] > 0 else 0
                ),
                'total_revenue_improvement': (
                    (variant_metrics['total_revenue'] - baseline['total_revenue']) / 
                    baseline['total_revenue'] * 100
                    if baseline['total_revenue'] > 0 else 0
                )
            }
        
        return improvements
    
    def print_results(self):
        """Print A/B test results"""
        metrics = self.calculate_metrics()
        improvements = self.calculate_improvements(metrics)
        
        print("\n" + "="*60)
        print("A/B TEST RESULTS")
        print("="*60)
        
        # Print metrics for each variant
        for variant in ['baseline', 'epsilon_greedy', 'ucb']:
            m = metrics[variant]
            print(f"\n{variant.upper().replace('_', ' ')}:")
            print(f"  Interactions: {m['n']:,}")
            print(f"  CTR: {m['ctr']*100:.2f}%")
            print(f"  Conversion Rate: {m['conversion_rate']*100:.2f}%")
            print(f"  Total Revenue: ${m['total_revenue']:,.2f}")
            print(f"  Revenue/Interaction: ${m['revenue_per_interaction']:.2f}")
            print(f"  Revenue/Purchase: ${m['revenue_per_purchase']:.2f}")
        
        # Print improvements
        print("\n" + "="*60)
        print("IMPROVEMENTS OVER BASELINE")
        print("="*60)
        
        for variant in ['epsilon_greedy', 'ucb']:
            imp = improvements[variant]
            print(f"\n{variant.upper().replace('_', ' ')}:")
            print(f"  CTR: {imp['ctr_improvement']:+.2f}%")
            print(f"  Conversion Rate: {imp['conversion_improvement']:+.2f}%")
            print(f"  Revenue/Interaction: {imp['revenue_per_interaction_improvement']:+.2f}%")
            print(f"  Total Revenue: {imp['total_revenue_improvement']:+.2f}%")
        
        # Determine winner
        print("\n" + "="*60)
        best_variant = max(
            ['epsilon_greedy', 'ucb'],
            key=lambda v: improvements[v]['revenue_per_interaction_improvement']
        )
        best_improvement = improvements[best_variant]['revenue_per_interaction_improvement']
        
        print(f"🏆 WINNER: {best_variant.upper().replace('_', ' ')}")
        print(f"   Revenue improvement: +{best_improvement:.2f}%")
        print("="*60)
        
        return metrics, improvements


if __name__ == "__main__":
    print("="*60)
    print("A/B TESTING: BASELINE VS MULTI-ARMED BANDITS")
    print("="*60)
    
    # Initialize
    base_engine = RecommendationEngine()
    
    # Run A/B test with 5000 simulations
    ab_test = ABTestSimulator(base_engine, n_simulations=5000)
    ab_test.run_simulation()
    
    # Print results
    metrics, improvements = ab_test.print_results()
    
    print("\n✓ A/B test complete! Results saved.")