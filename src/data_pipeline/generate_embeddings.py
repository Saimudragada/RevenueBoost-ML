import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import os

class EmbeddingGenerator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """Initialize embedding models"""
        print(f"Loading BERT model: {model_name}...")
        self.text_model = SentenceTransformer(model_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_model.to(self.device)
        print(f"✓ Model loaded on {self.device}")
        
    def generate_text_embeddings(self, products_df):
        """Generate BERT embeddings for product descriptions"""
        print("\nGenerating text embeddings...")
        
        # Combine product info for better embeddings
        texts = []
        for _, row in products_df.iterrows():
            text = f"{row['category']} {row['name']}: {row['description']}"
            texts.append(text)
        
        # Generate embeddings in batches
        embeddings = self.text_model.encode(
            texts, 
            batch_size=32, 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def generate_image_embeddings(self, n_products, embedding_dim=512):
        """Generate simulated image embeddings (ResNet-style)"""
        print("\nGenerating simulated image embeddings...")
        
        # Simulate ResNet-50 output (2048-dim, we'll use 512 for efficiency)
        # In production, you'd use actual ResNet on product images
        embeddings = np.random.randn(n_products, embedding_dim).astype(np.float32)
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        print(f"✓ Generated image embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def generate_user_embeddings(self, users_df, products_df, interactions_df, text_embeddings):
        """Generate user embeddings based on interaction history"""
        print("\nGenerating user embeddings...")
        
        user_embeddings = []
        
        for user_id in tqdm(users_df['user_id'], desc="Processing users"):
            # Get user's interaction history
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            if len(user_interactions) == 0:
                # Cold start: use random embedding
                user_emb = np.random.randn(text_embeddings.shape[1])
            else:
                # Average embeddings of interacted products (weighted by purchase)
                product_ids = user_interactions['product_id'].values
                weights = user_interactions['purchased'].values.astype(float) * 2 + 1  # Purchases weighted 2x
                
                # Get product indices
                product_indices = products_df[products_df['product_id'].isin(product_ids)].index.values
                
                # Weighted average of product embeddings
                product_embs = text_embeddings[product_indices]
                user_emb = np.average(product_embs, axis=0, weights=weights[:len(product_indices)])
            
            # Normalize
            user_emb = user_emb / np.linalg.norm(user_emb)
            user_embeddings.append(user_emb)
        
        user_embeddings = np.array(user_embeddings).astype(np.float32)
        print(f"✓ Generated user embeddings with shape: {user_embeddings.shape}")
        return user_embeddings
    
    def save_embeddings(self, products_df, users_df, interactions_df, output_dir='data/embeddings'):
        """Generate and save all embeddings"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate product text embeddings
        text_embeddings = self.generate_text_embeddings(products_df)
        np.save(f'{output_dir}/product_text_embeddings.npy', text_embeddings)
        
        # Generate product image embeddings (simulated)
        image_embeddings = self.generate_image_embeddings(len(products_df))
        np.save(f'{output_dir}/product_image_embeddings.npy', image_embeddings)
        
        # Combine text + image embeddings
        combined_embeddings = np.concatenate([text_embeddings, image_embeddings], axis=1)
        np.save(f'{output_dir}/product_combined_embeddings.npy', combined_embeddings)
        
        # Generate user embeddings
        user_embeddings = self.generate_user_embeddings(users_df, products_df, interactions_df, text_embeddings)
        np.save(f'{output_dir}/user_embeddings.npy', user_embeddings)
        
        # Save metadata
        embedding_metadata = {
            'text_dim': text_embeddings.shape[1],
            'image_dim': image_embeddings.shape[1],
            'combined_dim': combined_embeddings.shape[1],
            'n_products': len(products_df),
            'n_users': len(users_df)
        }
        
        import json
        with open(f'{output_dir}/metadata.json', 'w') as f:
            json.dump(embedding_metadata, f, indent=2)
        
        print("\n" + "="*50)
        print("EMBEDDING GENERATION SUMMARY")
        print("="*50)
        print(f"Text embeddings: {text_embeddings.shape}")
        print(f"Image embeddings: {image_embeddings.shape}")
        print(f"Combined embeddings: {combined_embeddings.shape}")
        print(f"User embeddings: {user_embeddings.shape}")
        print(f"Output directory: {output_dir}")
        print("="*50)

if __name__ == "__main__":
    # Load data
    products_df = pd.read_csv('data/raw/products.csv')
    users_df = pd.read_csv('data/raw/users.csv')
    interactions_df = pd.read_csv('data/raw/interactions.csv')
    
    # Generate embeddings
    generator = EmbeddingGenerator()
    generator.save_embeddings(products_df, users_df, interactions_df)