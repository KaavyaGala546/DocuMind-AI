# DocuMind AI

> **Document intelligence for research, analysis, and semantic discovery.**  
> Upload PDFs or text files, uncover hidden themes, compare semantic similarity, cluster related documents, and generate extractive summaries through an interactive analytics workspace.

<p align="left">
  <a href="https://docu-mind-ai-kaavya.streamlit.app/">
    <img src="https://img.shields.io/badge/Live%20Demo-Open%20App-7C3AED?style=for-the-badge" alt="Live Demo">
  </a>
  <img src="https://img.shields.io/badge/Python-3.10%2B-111827?style=for-the-badge&logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Framework-Streamlit-111827?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%7C%20SBERT-111827?style=for-the-badge" alt="NLP Stack">
  <img src="https://img.shields.io/badge/Visuals-Plotly-111827?style=for-the-badge" alt="Plotly">
</p>

---

## Overview

DocuMind AI is an NLP-powered document analysis platform designed to transform **unstructured text into structured insights**.

It enables users to:
- Analyze large collections of documents efficiently  
- Discover hidden patterns and themes  
- Compare semantic similarity across documents  
- Generate quick summaries for faster understanding  

The project combines **machine learning, NLP, and interactive UI design** into a product-style experience.

---

## Why This Project Exists

Working with large document sets is slow and inefficient when done manually.

DocuMind AI solves this by answering:
- Which documents are similar?
- What themes exist across the dataset?
- How can documents be grouped automatically?
- What are the key takeaways from each document?

Instead of treating NLP as just a notebook experiment, this project turns it into an **interactive system**.

---

## Core Features

### Multi-Format Document Ingestion
- Supports **PDF** and **TXT** files  
- Upload custom datasets or use built-in corpora  
- Handles multiple documents in one session  

### Dual Analysis Engines
- **TF-IDF** → keyword-based, interpretable analysis  
- **SBERT** → semantic understanding and similarity  

### Document Clustering
- K-Means clustering  
- Automatic cluster suggestion  
- Interactive cluster exploration  

### Topic Modeling
- LDA-based thematic extraction  
- Identifies dominant concepts across corpus  

### Similarity Analysis
- Pairwise cosine similarity  
- Interactive heatmap visualization  

### Extractive Summarization
- Key sentence extraction  
- Faster document understanding  

### Interactive UI
- Premium dark-themed dashboard  
- Insight cards and structured views  
- Real-time interaction using Streamlit  

---

## Screenshots

> Add real screenshots after final UI polish.

![Hero](assets/screenshots/hero.png)  
![Clusters](assets/screenshots/clusters.png)  
![Similarity](assets/screenshots/similarity.png)  
![Explorer](assets/screenshots/explorer.png)

---

## Architecture

```
                ┌──────────────────────────────┐
                │        User Interface         │
                │         Streamlit App         │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │      Document Ingestion       │
                │   PDF/TXT Loader + Parsing    │
                └──────────────┬───────────────┘
                               │
                               ▼
                ┌──────────────────────────────┐
                │       Preprocessing Layer     │
                │ cleaning, normalization, NLP  │
                └──────────────┬───────────────┘
                               │
         ┌─────────────────────┴─────────────────────┐
         ▼                                           ▼
┌───────────────────────┐                  ┌───────────────────────┐
│   TF-IDF Pipeline     │                  │   SBERT Embeddings    │
│ keywords, LDA, stats  │                  │ semantic similarity   │
└─────────────┬─────────┘                  └─────────────┬─────────┘
              │                                          │
              └───────────────┬──────────────────────────┘
                              ▼
                ┌──────────────────────────────┐
                │    Analysis & Modeling        │
                │ clustering, similarity, LDA   │
                │ summarization, exploration    │
                └──────────────┬───────────────┘
                               ▼
                ┌──────────────────────────────┐
                │      Visual Insight Layer     │
                │ cards, charts, heatmaps, UI   │
                └──────────────────────────────┘
```

---

## Workflow

1. Load documents (upload or built-in corpus)  
2. Preprocess text  
3. Choose analysis engine (TF-IDF or SBERT)  
4. Run:
   - clustering  
   - similarity  
   - topic modeling  
   - summarization  
5. Explore results visually  

---

## Tech Stack

### Core
- Python  
- Streamlit  

### NLP / ML
- scikit-learn  
- Sentence Transformers (SBERT)  
- TF-IDF Vectorization  
- K-Means Clustering  
- LDA Topic Modeling  

### Visualization
- Plotly  
- Pandas  

### Data Handling
- PDF parsing  
- TXT ingestion  

---

## TF-IDF vs SBERT

| Feature | TF-IDF | SBERT |
|--------|--------|------|
| Focus | Lexical frequency | Semantic meaning |
| Strength | Interpretability | Context understanding |
| Speed | Faster | Slower |
| Usage | Keywords, LDA | Similarity, clustering |

---

## Project Structure

```
DocuMind-AI/
│
├── app.py
├── preprocessing.py
├── modeling.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
├── .pylintrc
├── .github/
│   └── workflows/
└── research_documents/
```

---

## Local Setup

### Clone repository
```bash
git clone https://github.com/KaavyaGala546/DocuMind-AI.git
cd DocuMind-AI
```

### Create environment
```bash
python -m venv venv
```

### Activate

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run app
```bash
streamlit run app.py
```

---

## Key Design Decisions

### Why Streamlit
Fast iteration and interactive UI without heavy frontend setup.

### Why TF-IDF + SBERT
Balances interpretability and semantic understanding.

### Why K-Means
Simple, fast, and effective for document grouping.

---

## Use Cases

- Research paper clustering  
- Literature reviews  
- Technical document comparison  
- Theme discovery  
- Knowledge base exploration  

---

## Limitations

- Extractive summarization only  
- Performance depends on data quality  
- LDA weak on small datasets  
- Not yet full production architecture  

---

## Future Improvements

- Persistent sessions  
- Exportable reports  
- Advanced summarization  
- Retrieval-based querying  
- FastAPI + React architecture  

---

## Live Demo

https://docu-mind-ai-kaavya.streamlit.app/

---

## Author

**Kaavya Gala**  
AI / Full Stack Developer  

GitHub: https://github.com/KaavyaGala546  

---

## Feedback

Open an issue or reach out if you have suggestions or improvements.
