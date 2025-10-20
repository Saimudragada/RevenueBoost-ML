import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Tuple
import json

class RecommendationEngine:
    def __init__(self, embeddings_dir='data/embeddings', data_dir='data/raw'):
        """Initialize recommendation engine with FAISS index"""
        print("Initializing Recommendation Engine...")
        
        # Load embeddings
        self.product_embeddings = np.load(f'{embeddings_dir}/product_combined_embeddings.npy')
        self.user_embeddings = np.load(f'{embeddings_dir}/user_embeddings.npy')
        
        # Load data
        self.products_df = pd.read_csv(f'{data_dir}/products.csv')
        self.users_df = pd.read_csv(f'{data_dir}/users.csv')
        self.interactions_df = pd.read_csv(f'{data_dir}/interactions.csv')
        
        # Load metadata
        with open(f'{embeddings_dir}/metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Build FAISS index
        self._build_faiss_index()
        
        print(f"✓ Engine initialized with {len(self.products_df)} products")
        print(f"✓ FAISS index built with dimension {self.product_embeddings.shape[1]}")
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        dimension = self.product_embeddings.shape[1]
        
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.product_embeddings)
        
        # Add to index
        self.index.add(self.product_embeddings)
        
        print(f"✓ FAISS index built: {self.index.ntotal} vectors indexed")
    
    def get_similar_products(self, product_id: str, top_k: int = 10) -> List[Dict]:
        """Get similar products using FAISS"""
        # Get product embedding
        product_idx = self.products_df[self.products_df['product_id'] == product_id].index[0]
        query_embedding = self.product_embeddings[product_idx:product_idx+1]
        
        # Search FAISS index
        distances, indices = self.index.search(query_embedding, top_k + 1)  # +1 to exclude self
        
        # Format results (skip first result which is the query product itself)
        similar_products = []
        for idx, score in zip(indices[0][1:], distances[0][1:]):
            product = self.products_df.iloc[idx].to_dict()
            product['similarity_score'] = float(score)
            similar_products.append(product)
        
        return similar_products
    
    def get_user_recommendations(self, user_id: str, top_k: int = 10, 
                                exclude_interacted: bool = True) -> List[Dict]:
        """Get personalized recommendations for a user"""
        # Get user embedding (384-dim, text only)
        user_idx = self.users_df[self.users_df['user_id'] == user_id].index[0]
        user_embedding_text = self.user_embeddings[user_idx:user_idx+1]
        
        # Pad user embedding to match product embedding dimension (896)
        # Product embeddings = [text(384) + image(512)]
        text_dim = 384
        image_dim = 512
        
        # Create padding for image dimension (zeros or small random values)
        image_padding = np.zeros((1, image_dim), dtype=np.float32)
        user_embedding_full = np.concatenate([user_embedding_text, image_padding], axis=1)
        
        # Normalize user embedding
        faiss.normalize_L2(user_embedding_full)
        
        # Search FAISS index
        distances, indices = self.index.search(user_embedding_full, top_k * 3)  # Get more candidates
        
        # Get user's interaction history
        if exclude_interacted:
            interacted_products = set(
                self.interactions_df[self.interactions_df['user_id'] == user_id]['product_id']
            )
        else:
            interacted_products = set()
        
        # Format results and apply business logic
        recommendations = []
        for idx, score in zip(indices[0], distances[0]):
            product = self.products_df.iloc[idx]
            
            # Skip already interacted products
            if product['product_id'] in interacted_products:
                continue
            
            # Calculate final score with business logic
            final_score = self._calculate_ranking_score(
                similarity_score=float(score),
                popularity=product['popularity_score'],
                rating=product['rating'],
                price=product['price']
            )
            
            rec = product.to_dict()
            rec['similarity_score'] = float(score)
            rec['final_score'] = final_score
            recommendations.append(rec)
            
            if len(recommendations) >= top_k:
                break
        
        # Sort by final score
        recommendations.sort(key=lambda x: x['final_score'], reverse=True)
        
        return recommendations
    
    def _calculate_ranking_score(self, similarity_score: float, popularity: float, 
                                 rating: float, price: float) -> float:
        """Business logic for ranking candidates"""
        # Weighted combination of factors
        score = (
            0.5 * similarity_score +           # Personalization
            0.2 * (popularity / 2.0) +         # Popularity (normalized)
            0.2 * (rating / 5.0) +             # Quality
            0.1 * (1 - min(price / 500, 1))    # Price sensitivity (cheaper = better)
        )
        return score
    
    def get_cold_start_recommendations(self, category: str = None, top_k: int = 10) -> List[Dict]:
        """Get recommendations for cold-start users (no history)"""
        products = self.products_df.copy()
        
        # Filter by category if specified
        if category:
            products = products[products['category'] == category]
        
        # Rank by popularity and rating
        products['cold_start_score'] = (
            0.6 * products['popularity_score'] / products['popularity_score'].max() +
            0.4 * products['rating'] / 5.0
        )
        
        # Sort and return top K
        top_products = products.nlargest(top_k, 'cold_start_score')
        
        recommendations = []
        for _, product in top_products.iterrows():
            rec = product.to_dict()
            rec['final_score'] = rec['cold_start_score']
            recommendations.append(rec)
        
        return recommendations
    
    def calculate_metrics(self) -> Dict:
        """Calculate business metrics"""
        total_interactions = len(self.interactions_df)
        total_purchases = self.interactions_df['purchased'].sum()
        total_revenue = self.interactions_df['revenue'].sum()
        
        metrics = {
            'total_interactions': total_interactions,
            'total_purchases': int(total_purchases),
            'ctr': 1.0,  # All interactions are clicks in our dataset
            'conversion_rate': total_purchases / total_interactions,
            'total_revenue': total_revenue,
            'revenue_per_interaction': total_revenue / total_interactions,
            'revenue_per_purchase': total_revenue / total_purchases if total_purchases > 0 else 0,
            'avg_products_per_user': total_interactions / len(self.users_df),
        }
        
        return metrics

if __name__ == "__main__":
    # Initialize engine
    engine = RecommendationEngine()
    
    # Test 1: Similar products
    print("\n" + "="*50)
    print("TEST 1: Similar Products")
    print("="*50)
    test_product = engine.products_df.iloc[0]['product_id']
    similar = engine.get_similar_products(test_product, top_k=5)
    print(f"\nProducts similar to {test_product}:")
    for i, prod in enumerate(similar, 1):
        print(f"{i}. {prod['name']} (Score: {prod['similarity_score']:.3f})")
    
    # Test 2: User recommendations
    print("\n" + "="*50)
    print("TEST 2: User Recommendations")
    print("="*50)
    test_user = engine.users_df.iloc[0]['user_id']
    recommendations = engine.get_user_recommendations(test_user, top_k=5)
    print(f"\nRecommendations for {test_user}:")
    for i, prod in enumerate(recommendations, 1):
        print(f"{i}. {prod['name']} - ${prod['price']:.2f} (Score: {prod['final_score']:.3f})")
    
    # Test 3: Cold start
    print("\n" + "="*50)
    print("TEST 3: Cold Start Recommendations")
    print("="*50)
    cold_start = engine.get_cold_start_recommendations(category='Electronics', top_k=5)
    print("\nTop Electronics for new users:")
    for i, prod in enumerate(cold_start, 1):
        print(f"{i}. {prod['name']} - ${prod['price']:.2f}")
    
    # Test 4: Business metrics
    print("\n" + "="*50)
    print("BUSINESS METRICS")
    print("="*50)
    metrics = engine.calculate_metrics()
    for key, value in metrics.items():
        if 'rate' in key or 'ctr' in key:
            print(f"{key}: {value*100:.2f}%")
        elif 'revenue' in key:
            print(f"{key}: ${value:,.2f}")
        else:
            print(f"{key}: {value:,.0f}")