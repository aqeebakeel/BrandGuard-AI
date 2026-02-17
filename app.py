import streamlit as st
import faiss
import numpy as np
import pickle
import os  # <--- This is the key fix
from PIL import Image
from sentence_transformers import SentenceTransformer

# Setup Page
st.set_page_config(page_title="BrandGuard Enterprise", layout="wide")

@st.cache_resource
def load_resources():
    # 1. Load Model
    model = SentenceTransformer('clip-ViT-B-32')
    
    # 2. Load FAISS Index (The Database)
    try:
        index = faiss.read_index('brandguard.index')
        with open('metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, index, metadata
    except Exception as e:
        return model, None, None

model, index, metadata = load_resources()

st.title("🛡️ BrandGuard: Enterprise Edition")
st.markdown("Searching across **10,000+ trademarks** in milliseconds using FAISS Vector Search.")

uploaded_file = st.file_uploader("Upload Logo Candidate", type=["png", "jpg", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    
    # Display User Image
    with col1:
        user_img = Image.open(uploaded_file).convert('RGB')
        # FIXED: Updated deprecated parameter
        st.image(user_img, caption="Your Upload", width=300) 

    if index is None:
        st.error("⚠️ Database not found! Please run 'ingest.py' locally and upload the .index file.")
    else:
        # 1. Vectorize User Image
        with st.spinner("Searching 10,000+ logos..."):
            query_vector = model.encode([user_img])
            
            # FAISS Requirement: Float32 and Normalized
            query_vector = query_vector.astype('float32')
            faiss.normalize_L2(query_vector)

            # 2. Search (k=5 means find top 5 matches)
            k = 5
            distances, indices = index.search(query_vector, k)

        # 3. Display Results
        with col2:
            st.subheader(f"Top {k} Conflicts")
            
            # Iterate through results
            for rank, idx in enumerate(indices[0]):
                score = distances[0][rank] * 100  # Convert 0-1 to percentage
                match_path = metadata[idx] # Get filename from ID
                
                # SAFETY FIX: Handle paths with different OS separators
                # This works for both Windows (\) and Linux (/)
                match_name = os.path.basename(match_path)

                # Conflict Card
                with st.expander(f"#{rank+1}: {match_name} ({score:.2f}%)", expanded=(rank==0)):
                    st.write(f"**Similarity Score:** {score:.2f}%")
                    
                    try:
                        # Attempt to load image if it exists locally
                        if os.path.exists(match_path):
                            st.image(match_path, width=150)
                        else:
                            st.caption(f"Preview unavailable (File not uploaded to cloud): {match_name}")
                    except:
                        st.warning("Image load failed")