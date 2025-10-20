import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import defaultdict
import json

class MultiArmedBandit:
    """Multi-Armed Bandit for recommendation optimization"""
    
    def __init__(self, n_arms: int, algorithm: str = 'epsilon_greedy', 
                 epsilon: float = 0.1, ucb_c: float = 2.0):
        """
        Initialize MAB
        
        Args:
            n_arms: Number of products/arms
            algorithm: 'epsilon_greedy' or 'ucb'
            epsilon: Exploration rate for epsilon-greedy
            ucb_c: Exploration parameter for UCB
        """
        self.n_arms = n_arms
        self.algorithm = algorithm
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        
        # Statistics for each arm
        self.arm_counts = np.zeros(n_arms)  # Number of times each arm was pulled
        self.arm_rewards = np.zeros(n_arms)  # Total rewards for each arm
        self.arm_values = np.zeros(n_arms)   # Average reward for each arm
        
        self.total_pulls = 0
        self.total_reward = 0
        
        print(f"✓ Initialized {algorithm} MAB with {n_arms} arms")
    
    def select_arm(self, candidate_indices: List[int]) -> int:
        """Select an arm using the chosen algorithm"""
        
        if self.algorithm == 'epsilon_greedy':
            return self._epsilon_greedy(candidate_indices)
        elif self.algorithm == 'ucb':
            return self._ucb(candidate_indices)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _epsilon_greedy(self, candidate_indices: List[int]) -> int:
        """Epsilon-greedy selection"""
        if np.random.random() < self.epsilon:
            # Explore: random selection
            return np.random.choice(candidate_indices)
        else:
            # Exploit: choose best arm
            candidate_values = self.arm_values[candidate_indices]
            best_idx = candidate_indices[np.argmax(candidate_values)]
            return best_idx
    
    def _ucb(self, candidate_indices: List[int]) -> int:
        """Upper Confidence Bound selection"""
        if self.total_pulls == 0:
            # First pull: random
            return np.random.choice(candidate_indices)
        
        ucb_values = []
        for arm_idx in candidate_indices:
            if self.arm_counts[arm_idx] == 0:
                # Unvisited arms get infinite UCB
                ucb_values.append(float('inf'))
            else:
                # UCB formula: average reward + exploration bonus
                avg_reward = self.arm_values[arm_idx]
                exploration_bonus = self.ucb_c * np.sqrt(
                    np.log(self.total_pulls) / self.arm_counts[arm_idx]
                )
                ucb_values.append(avg_reward + exploration_bonus)
        
        # Select arm with highest UCB
        best_idx = candidate_indices[np.argmax(ucb_values)]
        return best_idx
    
    def update(self, arm_idx: int, reward: float):
        """Update arm statistics after observing reward"""
        self.arm_counts[arm_idx] += 1
        self.arm_rewards[arm_idx] += reward
        self.arm_values[arm_idx] = self.arm_rewards[arm_idx] / self.arm_counts[arm_idx]
        
        self.total_pulls += 1
        self.total_reward += reward
    
    def get_statistics(self) -> Dict:
        """Get current MAB statistics"""
        return {
            'total_pulls': int(self.total_pulls),
            'total_reward': float(self.total_reward),
            'avg_reward': float(self.total_reward / self.total_pulls) if self.total_pulls > 0 else 0,
            'arms_explored': int(np.sum(self.arm_counts > 0)),
            'top_arms': self._get_top_arms(10)
        }
    
    def _get_top_arms(self, k: int = 10) -> List[Dict]:
        """Get top performing arms"""
        top_indices = np.argsort(self.arm_values)[-k:][::-1]
        
        top_arms = []
        for idx in top_indices:
            if self.arm_counts[idx] > 0:
                top_arms.append({
                    'arm_idx': int(idx),
                    'avg_reward': float(self.arm_values[idx]),
                    'pulls': int(self.arm_counts[idx]),
                    'total_reward': float(self.arm_rewards[idx])
                })
        
        return top_arms


class BanditRecommender:
    """Recommendation system with MAB optimization"""
    
    def __init__(self, base_recommender, algorithm: str = 'epsilon_greedy', 
                 epsilon: float = 0.1, ucb_c: float = 2.0):
        """
        Initialize Bandit-powered recommender
        
        Args:
            base_recommender: Base RecommendationEngine instance
            algorithm: MAB algorithm to use
            epsilon: Exploration rate
            ucb_c: UCB exploration parameter
        """
        self.base_recommender = base_recommender
        self.n_products = len(base_recommender.products_df)
        
        # Initialize MAB
        self.mab = MultiArmedBandit(
            n_arms=self.n_products,
            algorithm=algorithm,
            epsilon=epsilon,
            ucb_c=ucb_c
        )
        
        print(f"✓ Bandit Recommender initialized with {algorithm}")
    
    def get_recommendations(self, user_id: str, top_k: int = 10, 
                           use_bandit: bool = True) -> List[Dict]:
        """Get recommendations with optional bandit optimization"""
        
        # Get candidate products from base recommender
        candidates = self.base_recommender.get_user_recommendations(
            user_id, top_k=top_k * 3  # Get more candidates
        )
        
        if not use_bandit or len(candidates) == 0:
            return candidates[:top_k]
        
        # Get candidate indices
        candidate_product_ids = [c['product_id'] for c in candidates]
        candidate_indices = [
            self.base_recommender.products_df[
                self.base_recommender.products_df['product_id'] == pid
            ].index[0]
            for pid in candidate_product_ids
        ]
        
        # Use MAB to rerank/select top K
        selected_recommendations = []
        selected_indices = set()
        
        for _ in range(min(top_k, len(candidates))):
            # Filter out already selected
            available_indices = [idx for idx in candidate_indices if idx not in selected_indices]
            
            if not available_indices:
                break
            
            # Select using bandit
            selected_idx = self.mab.select_arm(available_indices)
            selected_indices.add(selected_idx)
            
            # Get product info
            product = self.base_recommender.products_df.iloc[selected_idx].to_dict()
            
            # Add bandit stats
            product['bandit_pulls'] = int(self.mab.arm_counts[selected_idx])
            product['bandit_avg_reward'] = float(self.mab.arm_values[selected_idx])
            
            selected_recommendations.append(product)
        
        return selected_recommendations
    
    def record_feedback(self, product_id: str, clicked: bool, purchased: bool, revenue: float = 0):
        """Record user feedback and update bandit"""
        # Get product index
        product_idx = self.base_recommender.products_df[
            self.base_recommender.products_df['product_id'] == product_id
        ].index[0]
        
        # Calculate reward (weighted: click=0.1, purchase=1.0, revenue normalized)
        reward = 0
        if clicked:
            reward += 0.1
        if purchased:
            reward += 1.0 + (revenue / 500.0)  # Normalize revenue
        
        # Update bandit
        self.mab.update(product_idx, reward)
    
    def get_statistics(self) -> Dict:
        """Get bandit statistics"""
        return self.mab.get_statistics()


if __name__ == "__main__":
    from src.recommendation_engine.recommender import RecommendationEngine
    
    print("="*50)
    print("MULTI-ARMED BANDIT TESTING")
    print("="*50)
    
    # Initialize base recommender
    base_engine = RecommendationEngine()
    
    # Test 1: Epsilon-Greedy Bandit
    print("\n" + "="*50)
    print("TEST 1: Epsilon-Greedy Bandit")
    print("="*50)
    bandit_recommender = BanditRecommender(
        base_engine, 
        algorithm='epsilon_greedy',
        epsilon=0.15
    )
    
    test_user = base_engine.users_df.iloc[0]['user_id']
    recommendations = bandit_recommender.get_recommendations(test_user, top_k=5)
    
    print(f"\nRecommendations for {test_user}:")
    for i, prod in enumerate(recommendations, 1):
        print(f"{i}. {prod['name']} - ${prod['price']:.2f}")
        print(f"   Bandit pulls: {prod['bandit_pulls']}, Avg reward: {prod['bandit_avg_reward']:.3f}")
    
    # Test 2: UCB Bandit
    print("\n" + "="*50)
    print("TEST 2: UCB Bandit")
    print("="*50)
    ucb_recommender = BanditRecommender(
        base_engine,
        algorithm='ucb',
        ucb_c=2.0
    )
    
    recommendations = ucb_recommender.get_recommendations(test_user, top_k=5)
    print(f"\nUCB Recommendations for {test_user}:")
    for i, prod in enumerate(recommendations, 1):
        print(f"{i}. {prod['name']} - ${prod['price']:.2f}")
    
    # Test 3: Simulate feedback
    print("\n" + "="*50)
    print("TEST 3: Simulating User Feedback")
    print("="*50)
    
    # Simulate 100 interactions
    for i in range(100):
        user = np.random.choice(base_engine.users_df['user_id'])
        recs = bandit_recommender.get_recommendations(user, top_k=3, use_bandit=True)
        
        for rec in recs:
            # Simulate feedback (biased toward higher-rated products)
            clicked = np.random.random() < 0.8
            purchased = clicked and (np.random.random() < (rec['rating'] / 5.0) * 0.2)
            revenue = rec['price'] if purchased else 0
            
            bandit_recommender.record_feedback(
                rec['product_id'], clicked, purchased, revenue
            )
    
    # Show statistics
    stats = bandit_recommender.get_statistics()
    print("\nBandit Statistics after 300 interactions:")
    print(f"Total pulls: {stats['total_pulls']}")
    print(f"Average reward: {stats['avg_reward']:.4f}")
    print(f"Arms explored: {stats['arms_explored']}/{bandit_recommender.n_products}")
    
    print("\nTop 5 performing products:")
    for i, arm in enumerate(stats['top_arms'][:5], 1):
        product = base_engine.products_df.iloc[arm['arm_idx']]
        print(f"{i}. {product['name']}")
        print(f"   Pulls: {arm['pulls']}, Avg Reward: {arm['avg_reward']:.4f}")