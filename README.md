# DocuMind AI

**Document intelligence for research, analysis, and semantic discovery.**

DocuMind AI is an NLP-powered system that transforms unstructured documents into structured insights through semantic similarity, clustering, topic modeling, and summarization — all within an interactive workspace.

---

## Live Demo

https://docu-mind-ai-kaavya.streamlit.app/

---

## What It Does

DocuMind AI enables users to:

- Upload and analyze **PDF or TXT documents**
- Compare documents using **semantic similarity (SBERT)** or **lexical similarity (TF-IDF)**
- Automatically group related documents using **K-Means clustering**
- Discover underlying themes using **topic modeling (LDA)**
- Generate concise summaries for faster understanding
- Explore documents through an interactive analysis interface

---

## Why This Project

Working with large document collections is slow and difficult to scale manually.

DocuMind AI was built to:
- Reduce time spent reading and comparing documents
- Provide structured insights from raw text
- Bridge the gap between NLP models and usable tools

This project focuses on turning machine learning pipelines into a **usable system**, not just isolated experiments.

---

## System Architecture

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

## Core Components

### Document Processing
- Multi-file ingestion (PDF + TXT)
- Text preprocessing pipeline

### Representation
- TF-IDF vectorization for interpretability
- SBERT embeddings for semantic understanding

### Analysis
- K-Means clustering
- Cosine similarity matrix
- LDA topic modeling
- Extractive summarization

### Interface
- Interactive dashboard built with Streamlit
- Structured navigation across analysis views

---

## TF-IDF vs SBERT

| Aspect | TF-IDF | SBERT |
|------|--------|------|
| Type | Lexical | Semantic |
| Strength | Interpretability | Context understanding |
| Speed | Fast | Slower |
| Use Case | Keywords, LDA | Similarity, clustering |

---

## Example Use Cases

- Research paper clustering  
- Literature review acceleration  
- Technical document comparison  
- Theme discovery across datasets  
- Knowledge base exploration  

---

## Project Structure

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

## Design Decisions

**Streamlit**  
Chosen for rapid prototyping and interactive UI development.

**Dual Representation (TF-IDF + SBERT)**  
Balances interpretability with semantic depth.

**K-Means Clustering**  
Provides fast and intuitive grouping for exploratory analysis.

---

## Limitations

- Summarization is extractive (not generative)
- Performance depends on dataset quality
- LDA effectiveness decreases with small datasets
- Architecture is not yet production-grade

---

## Roadmap

- Persistent document sessions  
- Exportable analysis reports  
- Advanced summarization pipelines  
- Retrieval-based querying  
- Migration to FastAPI + frontend architecture  

---

## Author

**Kaavya Gala**  
AI / Full Stack Developer  

GitHub: https://github.com/KaavyaGala546

---

## Notes

This project focuses on combining **machine learning + system design + usability** into a single cohesive application.
