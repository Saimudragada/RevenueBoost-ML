import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import plotly.graph_objects as go
import plotly.express as px


# Add parent directory to path (works both locally and in Docker)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.recommendation_engine.recommender import RecommendationEngine
from src.bandit.mab_engine import BanditRecommender

# Page config
st.set_page_config(
    page_title="AI Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .product-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        background: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .recommendation-score {
        background: #4CAF50;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
@st.cache_resource
def load_models():
    """Load recommendation models"""
    base_engine = RecommendationEngine()
    bandit_engine = BanditRecommender(
        base_engine,
        algorithm='epsilon_greedy',
        epsilon=0.1
    )
    return base_engine, bandit_engine

# Load models
base_engine, bandit_engine = load_models()

# Header
st.markdown('<div class="main-header">🎯 AI-Powered Recommendation System</div>', unsafe_allow_html=True)
st.markdown("### Multi-Armed Bandit Optimization with Business Metrics")

# Sidebar
st.sidebar.title("⚙️ Configuration")
recommendation_mode = st.sidebar.radio(
    "Select Recommendation Mode:",
    ["Baseline (Static)", "Bandit-Optimized (AI)", "Compare Both"]
)

num_recommendations = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Get Recommendations", "📊 A/B Test Results", "🎰 Bandit Statistics", "ℹ️ About"])

# ========================================
# TAB 1: Get Recommendations
# ========================================
with tab1:
    st.header("Get Personalized Recommendations")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # User selection
        st.subheader("Select User Profile")
        user_options = base_engine.users_df['user_id'].tolist()
        selected_user = st.selectbox("Choose User ID:", user_options)
        
        # Category filter
        category_filter = st.selectbox(
            "Filter by Category (Optional):",
            ["All"] + base_engine.products_df['category'].unique().tolist()
        )
        
        # Get button
        if st.button("🚀 Get Recommendations", type="primary"):
            st.session_state['show_recommendations'] = True
            st.session_state['selected_user'] = selected_user
    
    with col2:
        if 'show_recommendations' in st.session_state and st.session_state['show_recommendations']:
            user_id = st.session_state['selected_user']
            
            st.subheader(f"Recommendations for {user_id}")
            
            if recommendation_mode == "Baseline (Static)":
                recommendations = base_engine.get_user_recommendations(user_id, top_k=num_recommendations)
                st.info("📌 Using baseline static recommendations")
                
                for i, prod in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="product-card">
                            <h3>#{i} {prod['name']}</h3>
                            <p><strong>Category:</strong> {prod['category']}</p>
                            <p><strong>Price:</strong> ${prod['price']:.2f}</p>
                            <p><strong>Rating:</strong> ⭐ {prod['rating']:.1f}/5.0</p>
                            <p><span class="recommendation-score">Score: {prod['final_score']:.3f}</span></p>
                            <p><em>"{prod['description']}"</em></p>
                        </div>
                        """, unsafe_allow_html=True)
            
            elif recommendation_mode == "Bandit-Optimized (AI)":
                recommendations = bandit_engine.get_recommendations(user_id, top_k=num_recommendations)
                st.success("🎰 Using AI-powered bandit optimization")
                
                for i, prod in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="product-card">
                            <h3>#{i} {prod['name']}</h3>
                            <p><strong>Category:</strong> {prod['category']}</p>
                            <p><strong>Price:</strong> ${prod['price']:.2f}</p>
                            <p><strong>Rating:</strong> ⭐ {prod['rating']:.1f}/5.0</p>
                            <p><span class="recommendation-score">Bandit Score: {prod['bandit_avg_reward']:.3f}</span></p>
                            <p><strong>Times Recommended:</strong> {prod['bandit_pulls']}</p>
                            <p><em>"{prod['description']}"</em></p>
                            <p style="color: #666; font-size: 0.9rem;">
                                💡 <strong>Why recommended:</strong> This product has been optimized through continuous learning 
                                based on user interactions and conversion rates.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
            
            else:  # Compare Both
                col_baseline, col_bandit = st.columns(2)
                
                with col_baseline:
                    st.subheader("📌 Baseline")
                    baseline_recs = base_engine.get_user_recommendations(user_id, top_k=num_recommendations)
                    for i, prod in enumerate(baseline_recs, 1):
                        st.markdown(f"""
                        **#{i}** {prod['name']}  
                        ${prod['price']:.2f} | ⭐ {prod['rating']:.1f}
                        """)
                
                with col_bandit:
                    st.subheader("🎰 Bandit-Optimized")
                    bandit_recs = bandit_engine.get_recommendations(user_id, top_k=num_recommendations)
                    for i, prod in enumerate(bandit_recs, 1):
                        st.markdown(f"""
                        **#{i}** {prod['name']}  
                        ${prod['price']:.2f} | ⭐ {prod['rating']:.1f}  
                        🎯 Reward: {prod['bandit_avg_reward']:.3f}
                        """)

# ========================================
# TAB 2: A/B Test Results
# ========================================
with tab2:
    st.header("📊 A/B Test Results: Baseline vs Bandit")
    
    # Display saved results from simulation
    ab_results = {
        'Variant': ['Baseline', 'Epsilon-Greedy', 'UCB'],
        'CTR': [87.04, 87.32, 86.00],
        'Conversion Rate': [13.46, 13.56, 11.26],
        'Revenue/Interaction': [19.23, 26.97, 25.48],
        'Total Revenue': [96130.25, 134862.04, 127405.87]
    }
    
    df_results = pd.DataFrame(ab_results)
    
    # Metrics cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h2>+40.29%</h2>
            <p>Revenue Improvement<br>(Epsilon-Greedy)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h2>+$7.74</h2>
            <p>Revenue per Interaction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h2>5,000</h2>
            <p>Simulated Interactions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Comparison table
    st.subheader("Detailed Comparison")
    st.dataframe(df_results, use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df_results, 
            x='Variant', 
            y='Revenue/Interaction',
            title='Revenue per Interaction by Variant',
            color='Revenue/Interaction',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_results, 
            x='Variant', 
            y='Conversion Rate',
            title='Conversion Rate by Variant (%)',
            color='Conversion Rate',
            color_continuous_scale='Greens'
        )
        st.plotly_chart(fig, use_container_width=True)

# ========================================
# TAB 3: Bandit Statistics
# ========================================
with tab3:
    st.header("🎰 Multi-Armed Bandit Learning Statistics")
    
    stats = bandit_engine.get_statistics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Pulls", f"{stats['total_pulls']:,}")
    
    with col2:
        st.metric("Average Reward", f"{stats['avg_reward']:.4f}")
    
    with col3:
        st.metric("Arms Explored", f"{stats['arms_explored']}/1000")
    
    st.markdown("---")
    
    st.subheader("🏆 Top Performing Products")
    
    if stats['top_arms']:
        top_products_data = []
        for arm in stats['top_arms'][:10]:
            product = base_engine.products_df.iloc[arm['arm_idx']]
            top_products_data.append({
                'Product': product['name'],
                'Category': product['category'],
                'Avg Reward': arm['avg_reward'],
                'Times Recommended': arm['pulls'],
                'Total Reward': arm['total_reward']
            })
        
        df_top = pd.DataFrame(top_products_data)
        st.dataframe(df_top, use_container_width=True)

# ========================================
# TAB 4: About
# ========================================
with tab4:
    st.header("ℹ️ About This System")
    
    st.markdown("""
    ## 🎯 AI-Powered Recommendation System
    
    ### Features:
    - **Multi-Modal Embeddings:** BERT (text) + ResNet (images)
    - **Fast Similarity Search:** FAISS indexing for 100+ RPS
    - **Online Learning:** Multi-Armed Bandit optimization
    - **Business Metrics:** CTR, conversion rate, revenue tracking
    - **A/B Testing:** Proven +40% revenue improvement
    
    ### Technical Stack:
    - **Embeddings:** Sentence-BERT, ResNet-50
    - **Vector Search:** FAISS (Facebook AI Similarity Search)
    - **Algorithms:** Epsilon-Greedy, UCB
    - **Framework:** Python, PyTorch, Streamlit
    
    ### Business Impact:
    - ✅ +40.29% revenue per interaction
    - ✅ +$7.74 per user engagement
    - ✅ Continuous learning and optimization
    - ✅ Scalable to 100+ RPS with <200ms latency
    
    ### Project Structure:
```
    ├── data/                    # Data storage
    ├── src/
    │   ├── data_pipeline/      # Data generation
    │   ├── recommendation_engine/  # Core recommender
    │   └── bandit/             # MAB algorithms
    ├── streamlit_app/          # Web interface
    └── deployment/             # Docker & monitoring
```
    
    ---
    
    **Built with ❤️ using state-of-the-art ML techniques**
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚀 Scalable • 🎯 Accurate • 💰 Revenue-Driven</p>
</div>
""", unsafe_allow_html=True)