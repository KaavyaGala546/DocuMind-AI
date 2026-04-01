# DocuMind AI

> AI-powered research assistant for exploring, clustering, and understanding large document collections.

**Designed to bridge the gap between NLP models and real-world usable systems.**

---

## 🚀 Key Impact

- Process and analyze **multi-document datasets efficiently**
- Reduce manual document exploration time through **semantic similarity and clustering**
- Enable **faster insight discovery** across research papers and technical documents
- Integrates multiple NLP techniques into a **single interactive system**

---

## Preview

<p align="center">
  <a href="assets/screenshots/preview.mp4">
    <img src="assets/screenshots/livepreview.png" alt="DocuMind Demo" width="800"/>
  </a>
</p>

<p align="center">
  ▶️ Click the image to watch full demo
</p>

---

## Live Demo

https://docu-mind-ai-kaavya.streamlit.app/

---

## 🧠 Overview

Unlike typical document chat systems, DocuMind AI focuses on **deep document understanding** — clustering, similarity mapping, and thematic discovery.

It is an interactive **document intelligence system** that allows users to:

- Explore document collections
- Identify relationships between documents
- Extract themes and insights

Instead of isolated NLP experiments, this project builds a **complete end-to-end system** combining multiple techniques into a usable workflow.

---

## ⚙️ System Architecture
```
User Interface (Streamlit)
        │
        ▼
Document Ingestion (PDF / TXT)
        │
        ▼
Preprocessing (cleaning, normalization)
        │
        ├───────────────┐
        ▼               ▼
TF-IDF Pipeline     SBERT Embeddings
        │               │
        └───────┬───────┘
                ▼
        Analysis Layer
   (Clustering, Similarity,
    Topic Modeling, Summary)
                │
                ▼
        Visualization Layer
```

---

## 🔍 Core Capabilities

### 📄 Document Processing
- Multi-file ingestion (PDF + TXT)
- Robust text preprocessing pipeline

### 🧠 Representation
- TF-IDF for interpretable keyword analysis
- SBERT for deep semantic understanding

### 📊 Analysis
- K-Means clustering for grouping documents
- Cosine similarity matrix for comparison
- LDA topic modeling for theme discovery
- Extractive summarization for quick insights

### 🖥 Interface
- Interactive Streamlit dashboard
- Smooth navigation across analysis layers

---

## 📌 Example Workflow

**Input**
- Collection of research papers (PDF/TXT)

**System Processing**
- Preprocessing → Embedding → Clustering → Topic Modeling

**Output**
- Semantic clusters of related documents  
- Topic keywords per cluster  
- Similarity heatmap  
- Extractive summaries  

---

## ⚖️ TF-IDF vs SBERT

| Aspect | TF-IDF | SBERT |
|------|--------|------|
| Type | Lexical | Semantic |
| Strength | Interpretability | Context understanding |
| Speed | Fast | Slower |
| Use Case | Keywords, LDA | Similarity, clustering |

---

## 🛠 Tech Stack

**Core**
- Python
- Streamlit

**NLP / ML**
- scikit-learn
- Sentence Transformers (SBERT)
- TF-IDF Vectorization
- K-Means Clustering
- LDA Topic Modeling

**Visualization**
- Plotly
- Pandas

---

## 🎯 Use Cases

- Research paper clustering  
- Literature review acceleration  
- Technical document comparison  
- Theme discovery  
- Knowledge base exploration  

---

## 📁 Project Structure

```
DocuMind-AI/
├── app.py
├── preprocessing.py
├── modeling.py
├── utils.py
├── requirements.txt
├── research_documents/
└── .github/workflows/
```

---

## Local Setup

```bash
git clone https://github.com/KaavyaGala546/DocuMind-AI.git
cd DocuMind-AI
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

---

## Design Philosophy

This project focuses on:

- Build systems, not isolated models
- Balance interpretability (TF-IDF) with semantic depth (SBERT)
- Focus on real-world usability over theoretical complexity

---

## Limitations

- Summarization is extractive
- Performance depends on dataset quality
- LDA less effective on small datasets

---

## Roadmap

- Persistent document sessions  
- Exportable reports  
- Advanced summarization(abstractive) 
- Retrieval-based querying(RAG integration)
- Migration to full-stack architecture  

---

## Author

**Kaavya Gala**  
AI / Full Stack Developer  

GitHub: https://github.com/KaavyaGala546  

---

## Final Note

This project is not just about NLP models —
it is about building systems that make those models usable in real-world workflows.
