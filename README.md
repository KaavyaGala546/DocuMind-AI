# 🚀 DocuMind-AI

**DocuMind-AI** is an interactive NLP-powered research document analyzer that helps you explore, compare, and extract insights from text documents using both classical and modern NLP techniques.

It allows users to analyze document similarity, cluster related papers, uncover hidden topics, and generate summaries — all through an intuitive Streamlit interface.

---

## 🌐 Live Demo

👉 https://documindai.streamlit.app

---

## ✨ Features

- 📄 Upload and analyze `.txt` or `.pdf` documents  
- 🔍 Document similarity analysis  
- 🧠 Dual vectorization:
  - TF-IDF (Classical NLP)
  - SBERT (Semantic Embeddings)  
- 📊 Clustering of related documents (K-Means)  
- 🧩 Topic modeling for theme discovery  
- ✂️ Extractive text summarization  
- 📈 Interactive visualizations (heatmaps, clusters)  
- 🎛️ Clean and modern Streamlit UI  

---

## 🧠 Why DocuMind-AI?

Traditional NLP methods like TF-IDF focus on keyword matching, while modern approaches like SBERT capture contextual meaning.

This project lets you **compare both approaches side-by-side**, making it a powerful tool for:
- research analysis  
- document comparison  
- NLP learning and experimentation  

---

## 🛠️ Tech Stack

**Language**
- Python  

**Libraries & Frameworks**
- Streamlit  
- Pandas  
- NumPy  
- Scikit-learn  
- NLTK  
- Sentence-Transformers (SBERT)  
- Plotly  

---

## ⚙️ How It Works

### 1. Preprocessing
- Text cleaning  
- Tokenization  
- Stopword removal  
- Lemmatization  

### 2. Vectorization
- TF-IDF representation  
- SBERT embeddings  

### 3. Analysis
- Document similarity  
- K-Means clustering  
- Topic modeling  
- Extractive summarization  

### 4. Visualization
- Similarity heatmaps  
- Cluster plots  
- Interactive dashboard  

---

## 📁 Project Structure

```
DocuMind-AI/
├── app.py
├── preprocessing.py
├── modeling.py
├── utils.py
├── create_corpus.py
├── requirements.txt
├── research_documents/
└── .github/workflows/
```

---

## ⚡ Installation

```bash
git clone https://github.com/KaavyaGala546/DocuMind-AI.git
cd DocuMind-AI
pip install -r requirements.txt
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

---

## 🧪 Usage

1. Open the app locally or via the live demo  
2. Upload your own documents OR use sample data  
3. Select vectorization mode:
   - TF-IDF  
   - SBERT  
4. Explore:
   - similarity  
   - clustering  
   - topics  
   - summaries  

---

## 📊 Key Insights

- SBERT captures **contextual similarity** better than TF-IDF  
- TF-IDF performs well for **keyword-based matching**  
- Clustering improves significantly with semantic embeddings  
- Visualization helps understand document relationships clearly  

---

## ⚠️ Limitations

- Works best with structured or research-style text  
- Extractive summaries may miss deeper context  
- Performance depends on dataset size  

---

## 🔮 Future Improvements

- Abstractive summarization (transformers)  
- Better PDF parsing  
- Larger dataset support  
- Exportable reports  
- Enhanced UI/UX  

---

## 👩‍💻 Author

**Kaavya Gala**

---

## ⭐ If you like this project

Give it a star ⭐ and feel free to fork or contribute!
