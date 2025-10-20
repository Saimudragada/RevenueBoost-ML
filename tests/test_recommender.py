import sys
sys.path.append('/Users/saimudragada/Desktop/recommendation_system_mab')

import pytest
import numpy as np
from src.recommendation_engine.recommender import RecommendationEngine
from src.bandit.mab_engine import BanditRecommender, MultiArmedBandit

def test_recommendation_engine_initialization():
    """Test that recommendation engine initializes correctly"""
    engine = RecommendationEngine()
    assert engine.index.ntotal == 1000
    assert len(engine.products_df) == 1000
    assert len(engine.users_df) == 500

def test_get_similar_products():
    """Test similar product retrieval"""
    engine = RecommendationEngine()
    product_id = engine.products_df.iloc[0]['product_id']
    similar = engine.get_similar_products(product_id, top_k=5)
    
    assert len(similar) == 5
    assert all('similarity_score' in p for p in similar)

def test_user_recommendations():
    """Test personalized recommendations"""
    engine = RecommendationEngine()
    user_id = engine.users_df.iloc[0]['user_id']
    recommendations = engine.get_user_recommendations(user_id, top_k=10)
    
    assert len(recommendations) <= 10
    assert all('final_score' in r for r in recommendations)

def test_multi_armed_bandit():
    """Test MAB initialization and selection"""
    mab = MultiArmedBandit(n_arms=100, algorithm='epsilon_greedy')
    
    # Test arm selection
    arm = mab.select_arm(list(range(20)))
    assert 0 <= arm < 20
    
    # Test update
    mab.update(arm, reward=1.0)
    assert mab.total_pulls == 1
    assert mab.arm_counts[arm] == 1

def test_bandit_recommender():
    """Test bandit-powered recommendations"""
    base_engine = RecommendationEngine()
    bandit = BanditRecommender(base_engine)
    
    user_id = base_engine.users_df.iloc[0]['user_id']
    recommendations = bandit.get_recommendations(user_id, top_k=5)
    
    assert len(recommendations) <= 5
    assert all('bandit_pulls' in r for r in recommendations)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])