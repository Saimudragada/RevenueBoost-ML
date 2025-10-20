import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os

np.random.seed(42)

class DataGenerator:
    def __init__(self, n_users=500, n_products=1000):
        self.n_users = n_users
        self.n_products = n_products
        self.categories = ['Electronics', 'Fashion', 'Books', 'Home', 'Sports']
        
    def generate_products(self):
        """Generate product catalog"""
        products = []
        
        for i in range(self.n_products):
            category = np.random.choice(self.categories)
            
            # Product attributes
            product = {
                'product_id': f'P{i:04d}',
                'category': category,
                'name': f'{category}_Product_{i}',
                'price': np.random.uniform(10, 500),
                'description': self._generate_description(category, i),
                'rating': np.random.uniform(3.0, 5.0),
                'popularity_score': np.random.exponential(0.3)  # Long-tail distribution
            }
            products.append(product)
        
        return pd.DataFrame(products)
    
    def _generate_description(self, category, idx):
        """Generate synthetic product descriptions"""
        templates = {
            'Electronics': [
                'High-performance device with advanced features and sleek design',
                'Cutting-edge technology for modern lifestyle and productivity',
                'Premium quality electronics with warranty and customer support'
            ],
            'Fashion': [
                'Stylish and comfortable apparel for everyday wear',
                'Trendy fashion item with premium fabric and modern design',
                'Elegant clothing piece perfect for any occasion'
            ],
            'Books': [
                'Bestselling book with captivating storyline and great reviews',
                'Educational content with comprehensive coverage and examples',
                'Inspiring read that will change your perspective on life'
            ],
            'Home': [
                'Durable home product designed for comfort and functionality',
                'Elegant home decor item to enhance your living space',
                'Practical household item with modern aesthetic appeal'
            ],
            'Sports': [
                'Professional-grade sports equipment for peak performance',
                'Comfortable athletic gear designed for active lifestyle',
                'High-quality sports product with durability and style'
            ]
        }
        return np.random.choice(templates[category]) + f' (Item #{idx})'
    
    def generate_users(self):
        """Generate user profiles"""
        users = []
        
        for i in range(self.n_users):
            # User preferences (affinity to categories)
            preferences = np.random.dirichlet(np.ones(len(self.categories)) * 2)
            
            user = {
                'user_id': f'U{i:04d}',
                'signup_date': (datetime.now() - timedelta(days=np.random.randint(30, 365))).strftime('%Y-%m-%d'),
                'category_preferences': dict(zip(self.categories, preferences)),
                'avg_session_duration': np.random.uniform(5, 30),  # minutes
                'purchase_frequency': np.random.choice(['low', 'medium', 'high'], p=[0.5, 0.3, 0.2])
            }
            users.append(user)
        
        return pd.DataFrame(users)
    
    def generate_interactions(self, products_df, users_df, n_interactions=10000):
        """Generate user-product interactions"""
        interactions = []
        
        for _ in range(n_interactions):
            user_id = np.random.choice(users_df['user_id'])
            user_prefs = users_df[users_df['user_id'] == user_id]['category_preferences'].values[0]
            
            # Sample product based on user preferences
            category_weights = products_df['category'].map(user_prefs)
            product_idx = np.random.choice(
                len(products_df), 
                p=(category_weights * products_df['popularity_score']).values / 
                  (category_weights * products_df['popularity_score']).sum()
            )
            product = products_df.iloc[product_idx]
            
            # Interaction details
            clicked = True
            purchased = np.random.random() < 0.15  # 15% conversion rate baseline
            
            interaction = {
                'user_id': user_id,
                'product_id': product['product_id'],
                'timestamp': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'clicked': clicked,
                'purchased': purchased,
                'revenue': product['price'] if purchased else 0,
                'session_id': f'S{np.random.randint(0, 5000):05d}'
            }
            interactions.append(interaction)
        
        return pd.DataFrame(interactions)
    
    def save_data(self, output_dir='data/raw'):
        """Generate and save all data"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating products...")
        products_df = self.generate_products()
        products_df.to_csv(f'{output_dir}/products.csv', index=False)
        print(f"✓ Generated {len(products_df)} products")
        
        print("\nGenerating users...")
        users_df = self.generate_users()
        users_df.to_csv(f'{output_dir}/users.csv', index=False)
        print(f"✓ Generated {len(users_df)} users")
        
        print("\nGenerating interactions...")
        interactions_df = self.generate_interactions(products_df, users_df)
        interactions_df.to_csv(f'{output_dir}/interactions.csv', index=False)
        print(f"✓ Generated {len(interactions_df)} interactions")
        
        # Statistics
        print("\n" + "="*50)
        print("DATA GENERATION SUMMARY")
        print("="*50)
        print(f"Products: {len(products_df)}")
        print(f"Users: {len(users_df)}")
        print(f"Interactions: {len(interactions_df)}")
        print(f"CTR: 100%")  # All interactions are clicks in this dataset
        print(f"Conversion Rate: {(interactions_df['purchased'].sum() / len(interactions_df) * 100):.2f}%")
        print(f"Total Revenue: ${interactions_df['revenue'].sum():,.2f}")
        print(f"Avg Revenue per Purchase: ${interactions_df[interactions_df['purchased']]['revenue'].mean():.2f}")
        print("="*50)

if __name__ == "__main__":
    generator = DataGenerator(n_users=500, n_products=1000)
    generator.save_data()