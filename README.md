# 🛡️ BrandGuard: AI-Powered Visual Identity Defense

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brandguard-ai-d7sjqn4bv5fuba3ej2ddwz.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Model](https://img.shields.io/badge/Model-OpenAI_CLIP-green)](https://github.com/openai/CLIP)
[![Vector DB](https://img.shields.io/badge/Vector_DB-FAISS-yellow)](https://github.com/facebookresearch/faiss)

**BrandGuard** is an enterprise-grade semantic search engine designed to protect intellectual property by detecting trademark infringement in logos. Unlike traditional keyword-based searches, BrandGuard uses **Computer Vision** and **Vector Embeddings** to understand the *visual semantics* of a logo, detecting conflicts even if the image is a hand-drawn sketch, a distorted variation, or a completely different color.

## 🔗 [Live Demo: Launch BrandGuard AI](https://brandguard-ai-d7sjqn4bv5fuba3ej2ddwz.streamlit.app/)

**Demo Images** - 
<img width="391" height="476" alt="image" src="https://github.com/user-attachments/assets/8a3fef35-06c0-4893-9dff-87af096fdcbf" />
<img width="400" height="214" alt="image" src="https://github.com/user-attachments/assets/2819715f-2c60-416c-8fb5-654501886413" />


## 📸 Project Gallery

### 1. The Executive Dashboard
*A clean, modern interface managing over 1,400+ protected assets with real-time analytics.*
![Executive Dashboard](demo_dashboard.png)

### 2. High-Precision Conflict Detection
*BrandGuard successfully flagging a "Pink Lace" distortion of the Nike logo with high confidence, despite color and shape changes.*
![Conflict Detection](demo_conflict.png)

---

## 🚀 The "Why" (Problem vs. Solution)

| **Traditional Search** ❌ | **BrandGuard AI** ✅ |
| :--- | :--- |
| **Keyword Dependent:** Can only find "Nike" if the file is named "nike.png". | **Visual Understanding:** Finds "Nike" even if the file is named "photo_123.jpg". |
| **Exact Match Only:** Fails if the logo is rotated, recolored, or sketched. | **Semantic Match:** Detects shape, geometry, and stylistic similarity (e.g., a "pink swoosh"). |
| **Slow Scaling:** Linear search gets slower with every new image. | **Instant Retrieval:** FAISS Vector Index searches millions of images in milliseconds. |

---

## 🛠️ Tech Stack & Architecture

**BrandGuard** is built on a modern MLOps pipeline:

* **Frontend:** [Streamlit](https://streamlit.io/) (for rapid, interactive UI).
* **AI Engine:** [OpenAI CLIP (ViT-B/32)](https://github.com/openai/CLIP) – A multimodal model that connects text and images.
* **Vector Search:** [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss) – For ultra-fast similarity search on high-dimensional vectors.
* **Data Processing:** `Pillow` (Image manipulation) & `NumPy` (Matrix operations).
* **Deployment:** Docker (Containerization) & Streamlit Cloud (CI/CD).

### **How It Works (The Pipeline)**

1.  **Ingestion:** The system scans a protected dataset of trademarks (1,400+ images).
2.  **Embedding:** Each logo is passed through the **CLIP Vision Transformer**, converting it into a **512-dimensional vector** (a mathematical fingerprint).
3.  **Indexing:** These vectors are normalized and stored in a **FAISS Index** (`brandguard.index`) for O(1) retrieval speed.
4.  **Inference:** When a user uploads a new logo:
    * It is converted to a vector on the fly.
    * The system calculates the **Cosine Similarity** against the index.
    * It returns the top matches with a confidence score (0-100%) and visualizes them as critical alerts.

---

## ⚡ Performance
* **Latency:** Retrieval time is **<200ms** for a dataset of 1,400+ images.
* **Accuracy:** Successfully identifies "concept clones" (e.g., a hand-drawn Apple logo) with >85% confidence.
* **Scalability:** The FAISS backend is capable of scaling to **millions of vectors** with minimal RAM usage.

---

## 💻 Installation & Usage

### Option 1: Run Locally (Python)

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/BrandGuard-AI.git](https://github.com/YOUR_USERNAME/BrandGuard-AI.git)
    cd BrandGuard-AI
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the App**
    ```bash
    streamlit run app.py
    ```

### Option 2: Run with Docker 🐳
Build and run the containerized application to ensure environment consistency.

```bash
# Build the image
docker build -t brandguard .

# Run container on port 8501
docker run -p 8501:8501 brandguard
