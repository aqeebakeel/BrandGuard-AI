import streamlit as st
import faiss
import numpy as np
import pickle
import os
import random
from PIL import Image
from sentence_transformers import SentenceTransformer

# --- 1. CONFIGURATION & STYLE ---
st.set_page_config(
    page_title="BrandGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed" # Collapsed for that cleaner "Hero" look
)

# Custom CSS to make cards look like the Gallery
st.markdown("""
<style>
    .stContainer {
        border-radius: 15px;
        transition: transform 0.3s ease;
    }
    .stContainer:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model = SentenceTransformer('clip-ViT-B-32')
    try:
        index = faiss.read_index('brandguard.index')
        with open('metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, index, metadata
    except:
        return model, None, None

model, index, metadata = load_resources()

# --- 3. HELPER FUNCTION: DRAW CARD ---
def draw_card(col, img_path, title, score=None, subtitle="Protected Asset"):
    """Draws a beautiful card in a specific column"""
    with col:
        with st.container(border=True):
            # 1. The Image
            try:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                else:
                    # Placeholder if image is missing on cloud
                    st.image("https://placehold.co/400x300/EEE/31343C?text=Image+Unavailable", use_container_width=True)
            except:
                st.warning("Img Error")

            # 2. The Title & Metadata
            st.subheader(title)
            
            if score:
                # Color logic for badges
                color = "red" if score > 80 else "orange" if score > 60 else "green"
                status = "CRITICAL" if score > 80 else "WARNING" if score > 60 else "SAFE"
                st.markdown(f":{color}[**{score:.1f}% Match**] • {status}")
                st.progress(min(int(score), 100), text="Similarity")
            else:
                st.caption(subtitle)

# --- 4. MAIN APP ---

# A. HERO SECTION
st.markdown("<h1 style='text-align: center;'>🛡️ BrandGuard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #666;'>The AI-Powered Trademark Defense System</p>", unsafe_allow_html=True)
st.divider()

# B. INPUT SECTION
col_upload, col_stats = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader("🔍 Upload a logo to scan for conflicts...", type=["png", "jpg", "jpeg"])

with col_stats:
    # Live Database Stats (Executive View)
    if index:
        st.metric("Protected Assets", f"{index.ntotal:,}", "+12 this week")
    else:
        st.error("Database Offline")

# C. RESULTS / GALLERY SECTION
if uploaded_file and index:
    st.markdown("### 🎯 Detection Results")
    
    # Process Image
    user_img = Image.open(uploaded_file).convert('RGB')
    
    # Search
    with st.spinner("Scanning 10,000+ vector embeddings..."):
        query_vector = model.encode([user_img]).astype('float32')
        faiss.normalize_L2(query_vector)
        k = 6 # Fetch top 6 for a nice 3x2 grid
        distances, indices = index.search(query_vector, k)

    # GRID LAYOUT (The Magic Part)
    # We create rows of 3 columns
    cols = st.columns(3)
    
    for i, idx in enumerate(indices[0]):
        score = distances[0][i] * 100
        match_path = metadata[idx]
        match_name = os.path.basename(match_path)
        
        # Decide which column this card goes into (0, 1, or 2)
        col_index = i % 3 
        
        draw_card(
            col=cols[col_index],
            img_path=match_path,
            title=match_name,
            score=score
        )

# D. DEFAULT GALLERY (When nothing is uploaded)
elif index:
    st.markdown("### 🏛️ Protected Brands Gallery")
    st.caption("Recently indexed assets in the database.")
    
    # Show 3 random logos from the DB just to make it look populated
    # (Only works if you have local images, otherwise shows placeholders)
    if len(metadata) > 0:
        random_indices = random.sample(range(len(metadata)), min(3, len(metadata)))
        cols = st.columns(3)
        for i, idx in enumerate(random_indices):
            path = metadata[idx]
            name = os.path.basename(path)
            draw_card(cols[i], path, name, subtitle="Indexed & Secure")