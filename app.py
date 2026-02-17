import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer, util
import os

# 1. Setup & Configuration
st.set_page_config(page_title="BrandGuard AI", layout="centered")

# 2. Load the AI Model (Cached so it only runs once!)
@st.cache_resource
def load_model():
    print("Loading CLIP Model... (This might take a moment)")
    return SentenceTransformer('clip-ViT-B-32')

# 3. The "Database" Builder
def build_vector_db(model):
    """
    Scans the 'logos' folder and converts every image into a vector.
    Returns: A list of dictionaries [{'name': 'nike.jpg', 'embedding': vector}, ...]
    """
    db = []
    logos_folder = "logos"
    
    # Create folder if it doesn't exist
    if not os.path.exists(logos_folder):
        os.makedirs(logos_folder)
        st.warning(f"⚠️ Folder '{logos_folder}' not found. I created it for you. Please put some logo images inside!")
        return db

    # Scan for images
    files = [f for f in os.listdir(logos_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not files:
        st.warning(f"⚠️ No logos found in '{logos_folder}'. Please add some images to test against!")
        return db

    # Progress bar for indexing
    progress_text = "Indexing Database..."
    my_bar = st.progress(0, text=progress_text)

    for i, file in enumerate(files):
        path = os.path.join(logos_folder, file)
        try:
            img = Image.open(path)
            # Generate embedding
            embedding = model.encode(img, convert_to_tensor=True)
            db.append({"name": file, "embedding": embedding, "path": path})
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
        
        # Update progress bar
        my_bar.progress((i + 1) / len(files), text=progress_text)

    my_bar.empty() # Clear bar when done
    return db

# --- MAIN APP LOGIC ---

st.title("🛡️ BrandGuard: Trademark Conflict Detector")
st.markdown("Upload a logo sketch to check for conflicts with famous brands.")

# Load Model & DB
model = load_model()
vector_db = build_vector_db(model)

# Sidebar: Show what's in the database
st.sidebar.header(f"📚 Database ({len(vector_db)} Logos)")
for item in vector_db:
    st.sidebar.image(item['path'], caption=item['name'], width=100)

# File Uploader
uploaded_file = st.file_uploader("Upload your logo:", type=["png", "jpg", "jpeg"])

if uploaded_file and vector_db:
    # 1. Display User Image
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Your Upload")
        user_img = Image.open(uploaded_file)
        st.image(user_img, use_column_width=True)
    
    # 2. Convert User Image to Vector
    with st.spinner("Analyzing geometric features..."):
        user_embedding = model.encode(user_img, convert_to_tensor=True)

    # 3. Compare against Database (The Core Logic)
    best_match = None
    highest_score = -1.0

    for entry in vector_db:
        # Calculate Cosine Similarity
        score = util.cos_sim(user_embedding, entry['embedding']).item()
        
        if score > highest_score:
            highest_score = score
            best_match = entry

    # 4. Display Results
    with col2:
        st.subheader("Closest Match")
        if best_match:
            st.image(best_match['path'], caption=f"{best_match['name']}", use_container_width=True)
        
    # 5. The Verdict
    st.divider()
    score_percent = round(highest_score * 100, 2)
    st.metric("Conflict Score", f"{score_percent}%")
    
    if score_percent > 60: # Threshold for "Conflict"
        st.error(f"🚨 **CONFLICT DETECTED!** \n\nThis logo is {score_percent}% similar to **{best_match['name']}**.")
        st.progress(min(highest_score,1.0))
    else:
        st.success(f"✅ **CLEAN!** \n\nLow similarity detected ({score_percent}%). This logo looks unique.")

elif not vector_db:
    st.info("👈 Please add logo images to the 'logos' folder to start detecting conflicts.")