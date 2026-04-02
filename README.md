# 📰 Real-Time Fake News Detector

An AI-powered hybrid system that verifies news credibility using:

- 🌐 Real-time internet validation (NewsAPI)
- 🤖 Transformer-based ML fallback (DistilBERT)
- 🔍 Semantic similarity scoring (Sentence Transformers)

> This project is NOT just a machine learning model.  
> It first verifies news using real-world sources and only falls back to AI when needed.

---

## 🚀 Features

- ✅ Real-time news verification using live internet data  
- ✅ Semantic similarity matching between claim and articles  
- ✅ Confidence scoring system  
- ✅ ML fallback when no trusted sources are found  
- ✅ FastAPI backend (production-style API)  
- ✅ Interactive frontend UI  
- ✅ Clean modular architecture  

---

## 🧠 How It Works

Traditional systems:
Input → Model → Output ❌  

This system:
Input → Internet Verification → ML Model (if needed) → Output ✅  

---

## 🔄 Workflow

1. User enters news text  
2. System searches real-time news using NewsAPI  
3. Extracts top relevant articles  
4. Converts both:
   - User claim
   - Article content  
   into embeddings using Sentence Transformers  
5. Computes cosine similarity  
6. Decision:
   - If similarity ≥ 0.72 → ✅ Verified (Real)
   - Else → 🤖 ML Model Prediction  

---

## 🏗️ Architecture
User Input
->
NewsAPI Search
->
Semantic Similarity (MiniLM)
->
IF score ≥ threshold → VERIFIED
->
ELSE
->
DistilBERT Classifier
->
Final Output


---

## 🤖 ML Model Details

- Model: DistilBERT (fine-tuned)
- Task: Binary classification (Real / Fake)

### Training:
- Datasets:
  - ISOT Fake News Dataset
  - WELFake Dataset
- Preprocessing:
  - Text cleaning
  - Removing URLs, symbols, noise
  - Combining title + content
- Hyperparameters:
  - Epochs: 5
  - Batch size: 16
  - Learning rate: 2e-5
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score

---


## 🌐 API Example

### Request:

```json
{
  "news": "India wins Asia Cup 2025"
}
```
Response:
```json
{
  "status": "verified_online",
  "verdict": "Real (verified online)",
  "confidence": 92,
  "sources": [
    {
      "title": "...",
      "source": "...",
      "url": "..."
    }
  ]
}
```

🧪 Example Outputs
✅ Real News
Verdict: Real (verified online)
Confidence: 90%
Sources: BBC, Reuters

❌ Fake News
Verdict: Fake (ML-based)
Confidence: 85%


🛠️ Tech Stack
Backend:
Python
FastAPI
PyTorch
Transformers (HuggingFace)
AI/NLP:
DistilBERT
Sentence Transformers
Cosine Similarity
Data Sources:
ISOT Dataset
WELFake Dataset
NewsAPI (real-time)
Frontend:
HTML
CSS
JavaScript


📌 Why This Project Stands Out
Combines real-time verification + AI
Handles real-world uncertainty
Uses fallback mechanism (robust system design)
Shows practical system thinking, not just ML

👉 This is a production-level approach, not just a college project.

🔮 Future Improvements
Integrate Google Fact Check API
Add vector database (FAISS / Pinecone)
Multi-language support
Better threshold tuning
Deploy on AWS (Lambda + API Gateway)

👨‍💻 Author
Aditya Gaur
BTech | Cloud & AI Enthusiast
