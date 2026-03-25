import streamlit as st
import pandas as pd
from preprocessing import execute_preprocessing_pipeline, prepare_text_for_summary
from modeling import *
from utils import (
    generate_extractive_summary,
    render_silhouette_chart,
    render_similarity_heatmap,
    load_corpus_from_directory,
    process_uploaded_files,
)
import plotly.express as px
import plotly.graph_objects as go
import os
import re

# ---------------------------------------------------------------------------
# Custom CSS — Premium Dark Theme
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- Import Google Fonts ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Outfit:wght@400;500;600;700;800&display=swap');

/* ---- Core variables & Neutral palette ---- */
:root {
    --bg-dark: #0A0C10;
    --surface-1: #12151C;
    --surface-2: #1A1F26;
    --accent: #9333ea; /* More vibrant violet */
    --accent-soft: rgba(147, 51, 234, 0.1);
    --border-subtle: rgba(255, 255, 255, 0.04);
    --border-medium: rgba(255, 255, 255, 0.08);
    --text-primary: #F9FAFB;
    --text-secondary: #9CA3AF;
    --text-muted: #6B7280;
    --success: #10B981;
    --font-main: 'Inter', -apple-system, sans-serif;
    --font-heading: 'Outfit', sans-serif;
}

/* ---- Reset & Global Styling ---- */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-dark) !important;
    font-family: var(--font-main);
    color: var(--text-primary);
}

h1, h2, h3, h4, .stHeader {
    font-family: var(--font-heading) !important;
    letter-spacing: -0.02em;
}

/* ---- Hero Section (Luxurious) ---- */
.hero-section {
    padding: 4rem 0 3rem 0;
    text-align: center;
}
.hero-title {
    font-size: 4rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
    background: linear-gradient(to bottom right, #FFFFFF 30%, #9333ea 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.04em;
}
.hero-subtitle {
    font-size: 1.2rem;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ---- Modern KPI Grid (Glassmorphism) ---- */
.kpi-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 3rem;
}
.kpi-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 1.75rem 1.5rem;
    transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    backdrop-filter: blur(8px);
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(147, 51, 234, 0.3);
    background: var(--surface-2);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
}
.kpi-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #FFF;
    margin-bottom: 0.2rem;
    font-family: var(--font-heading);
    letter-spacing: -0.02em;
}
.kpi-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 700;
}

/* ---- Component Cards ---- */
.content-card {
    background: var(--surface-1);
    border: 1px solid var(--border-subtle);
    border-radius: 20px;
    padding: 1.75rem;
    margin-bottom: 1.5rem;
}
.cluster-badge {
    display: inline-flex;
    align-items: center;
    background: var(--accent-soft);
    color: var(--accent);
    padding: 4px 14px;
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 1.25rem;
    border: 1px solid rgba(147, 51, 234, 0.2);
}

/* ---- Refined Chips/Pills ---- */
.chip {
    display: inline-flex;
    align-items: center;
    background: transparent;
    border: 1px solid var(--border-medium);
    color: var(--text-secondary);
    padding: 5px 14px;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 500;
    margin-right: 8px;
    margin-bottom: 8px;
    transition: all 0.25s ease;
}
.chip:hover {
    border-color: var(--accent);
    color: #FFF;
    background: rgba(255, 255, 255, 0.02);
}

/* ---- Modernized Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    padding: 0 !important;
    border-bottom: 1px solid var(--border-subtle);
    margin-bottom: 2rem;
    gap: 2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    padding: 12px 0 !important;
    color: var(--text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: color 0.3s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    border-radius: 0 !important;
}

/* ---- Document Viewer Refinements ---- */
.doc-viewer {
    background: #07090D;
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 1rem;
    line-height: 1.8;
    color: #CBD5E1;
    max-height: 60vh;
    overflow-y: auto;
}
.highlight-summary { 
    background: rgba(16, 185, 129, 0.1) !important; 
    border-bottom: 2px solid var(--success); 
    color: #FFF !important;
    padding: 2px 0;
}
.highlight-keyword { 
    color: var(--accent) !important; 
    font-weight: 700; 
    border-bottom: 1px dashed var(--accent);
}

/* ---- Sidebar Overhaul ---- */
[data-testid="stSidebar"] {
    background-color: #080A0E !important;
    border-right: 1px solid var(--border-subtle);
}
.sidebar-section-header {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-muted);
    font-weight: 800;
    margin: 2.5rem 0 1rem 0;
    opacity: 0.8;
}

/* ---- Clean-up default UI ---- */
.stDeployButton, [data-testid="stToolbar"], #MainMenu, footer {
    display: none !important;
}
div[data-testid="stMetric"] { display: none; }
div[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Outfit', color='#8B949E'),
    title_font=dict(size=20, color='#E6EDF3', family='Outfit'),
    margin=dict(t=60, b=40, l=40, r=40),
    xaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
    yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False),
)


# ---------------------------------------------------------------------------
# Document Viewer Modal
# ---------------------------------------------------------------------------

def _highlight_text(text, keywords, summary_sentences):
    for sent in summary_sentences:
        pattern = re.escape(sent.strip())
        text = re.sub(
            pattern,
            f"<mark style='background-color:#2ea043; color:white; padding: 1px 3px; border-radius: 3px;'>{sent}</mark>",
            text,
            flags=re.IGNORECASE
        )
    for word in keywords:
        pattern = r"\b" + re.escape(word) + r"\b"
        text = re.sub(
            pattern,
            f"<span style='background-color:#da3633; color:white; padding: 1px 3px; border-radius: 3px;'>{word}</span>",
            text,
            flags=re.IGNORECASE
        )
    return text


@st.dialog("Intelligence Synthesis Report", width="large")
def show_document_modal(doc_name, raw_text, cleaned_text, keywords, summary_sentences):
    st.markdown(f"<div style='margin-bottom: 2rem;'><h1 style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{doc_name}</h1>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='display: flex; gap: 0.75rem;'>"
        "<span class='chip' style='background: rgba(16, 185, 129, 0.05); color: #10B981; border-color: rgba(16, 185, 129, 0.2); font-weight: 600;'>Extraction Fragment</span>"
        "<span class='chip' style='background: var(--accent-soft); color: var(--accent); border-color: rgba(147, 51, 234, 0.2); font-weight: 600;'>Semantic Anchor</span>"
        "</div></div>",
        unsafe_allow_html=True
    )

    view_highlighted, view_original = st.tabs(["Neural Highlights", "Raw Sequence"])

    with view_highlighted:
        highlighted = _highlight_text(cleaned_text, keywords, summary_sentences)
        st.markdown(f"<div class='doc-viewer'>{highlighted}</div>", unsafe_allow_html=True)

    with view_original:
        st.markdown(f"<div class='doc-viewer'>{raw_text}</div>", unsafe_allow_html=True)




# ---------------------------------------------------------------------------
# Page Config & CSS Injection
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="DocuMind AI", page_icon="🧠")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class='hero-section'>
        <div class='hero-title'>DocuMind AI</div>
        <div class='hero-subtitle'>
            Document intelligence for research, analysis, and semantic discovery.
            Extract hidden themes and measure similarity with precision.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("<div class='sidebar-section-header'>Analysis Engine</div>", unsafe_allow_html=True)

vectorization_mode = st.sidebar.radio(
    "Vectorization Mode",
    ["TF-IDF (Classical)", "Semantic Embeddings (SBERT)"],
    index=0,
    help="TF-IDF uses lexical word frequencies. Semantic Embeddings use a pre-trained encoder to capture meaning.",
    label_visibility="collapsed"
)

use_semantic = vectorization_mode == "Semantic Embeddings (SBERT)"

preserve_numbers = st.sidebar.toggle(
    "Retain Numerical Data (e.g., statistics, years)",
    value=True,
    disabled=use_semantic
)

st.sidebar.markdown("<div class='sidebar-section-header'>Data Source</div>", unsafe_allow_html=True)


CORPUS_OPTIONS = {
    "Upload My Own Files": None,
    "Primary Research Corpus (Text)": "research_documents",
    "AI Research Papers (PDF)": "research_documents/pdf_papers",
    "Semantic Limitation Demo": "research_documents/semantic_demo",
}

selected_corpus_label = st.sidebar.selectbox(
    "Choose Corpus",
    options=list(CORPUS_OPTIONS.keys()),
    label_visibility="collapsed"
)

selected_corpus_path = CORPUS_OPTIONS[selected_corpus_label]

if selected_corpus_label == "Semantic Limitation Demo":
    st.sidebar.warning(
        "**Why this demo?**\n\n"
        "Three documents describe the *exact same event* — online shopping — "
        "using completely different vocabulary.\n\n"
        "TF-IDF is lexical, so it **cannot** recognise semantic equivalence. "
        "Expect near-zero similarity."
    )
elif selected_corpus_label == "AI Research Papers (PDF)":
    st.sidebar.info(
        "**PDF Corpus**\n\n"
        "Classic CS papers (Attention, BERT, ResNet, MapReduce, GFS) + an outlier (Cricket Rule Book). "
        "Perfect for demonstrating domain-based clustering."
    )

uploaded_files = st.sidebar.file_uploader(
    "Upload .txt or .pdf files",
    accept_multiple_files=True,
    disabled=(selected_corpus_path is not None)
)

# ---------------------------------------------------------------------------
# Load Documents
# ---------------------------------------------------------------------------

raw_docs = []
filenames = []

if selected_corpus_path is not None:
    raw_docs, filenames = load_corpus_from_directory(selected_corpus_path)
    if not raw_docs:
        st.error(f"No documents found in `{selected_corpus_path}/`.")
elif uploaded_files:
    raw_docs, filenames = process_uploaded_files(uploaded_files)

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@st.cache_data
def run_tfidf_pipeline(raw_docs, preserve_numbers):
    processed_docs = [
        execute_preprocessing_pipeline(doc, preserve_numeric=preserve_numbers)
        for doc in raw_docs
    ]
    X, vectorizer = extract_tfidf_features(processed_docs)
    return processed_docs, X, vectorizer

@st.cache_data
def run_semantic_pipeline(raw_docs):
    X_sem = compute_semantic_embeddings(raw_docs)
    return X_sem


if raw_docs:
    # Always run TF-IDF for keywords, summarization, and LDA
    with st.spinner("Processing documents…"):
        processed_docs, X_tfidf, vectorizer = run_tfidf_pipeline(raw_docs, preserve_numbers)

    if use_semantic:
        with st.spinner("Computing semantic embeddings (SBERT)…"):
            X = run_semantic_pipeline(raw_docs)
        feature_count = X.shape[1]
        feature_label = "Embedding Dims"
        engine_label = "SBERT + K-Means"
    else:
        X = X_tfidf
        feature_count = len(vectorizer.get_feature_names_out())
        feature_label = "Vocabulary Terms"
        engine_label = "K-Means++"

    # -- Modern KPI Cards --
    st.markdown(
        f"""
        <div class='kpi-container'>
            <div class='kpi-card'>
                <div class='kpi-value'>{len(raw_docs)}</div>
                <div class='kpi-label'>Documents Loaded</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{feature_count}</div>
                <div class='kpi-label'>{feature_label}</div>
            </div>
            <div class='kpi-card'>
                <div class='kpi-value'>{engine_label}</div>
                <div class='kpi-label'>Analysis Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Clustering Insights", "LDA Topic Modeling", "Similarity Matrix"])

    # ---------------------------------------------------------------
    # TAB 3 — Similarity Matrix
    # ---------------------------------------------------------------
    with tab3:
        st.subheader("Document Similarity (Cosine Matrix)")
        with st.expander("How does Cosine Similarity work?"):
            st.write(
                "Cosine similarity measures the angle between two document vectors in TF-IDF space. "
                "**1.0** = identical term distributions, **0.0** = nothing in common. "
                "It's length-agnostic — a 10-page and 2-page paper on the same topic still score highly."
            )

        if len(raw_docs) >= 2:
            with st.spinner("Computing pairwise similarities…"):
                similarity = calculate_cosine_similarity(X)
                fig_sim = render_similarity_heatmap(similarity, filenames)
                fig_sim.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_sim, width='stretch')
        else:
            st.warning("Upload at least 2 documents to view similarity.")

    # ---------------------------------------------------------------
    # TAB 2 — LDA
    # ---------------------------------------------------------------
    with tab2:
        st.subheader("Topic Extraction via LDA")
        with st.expander("What is LDA?"):
            st.write(
                "Latent Dirichlet Allocation discovers hidden 'topics' — each topic is a distribution "
                "over words, and each document is a mixture of topics. Great for finding dominant themes "
                "without manual labelling."
            )

        if len(raw_docs) >= 2:
            st.markdown("<div class='content-card'>", unsafe_allow_html=True)
            n_topics = st.slider("Target Themes", min_value=2, max_value=min(6, len(raw_docs)), value=3, key="lda_slider")

            with st.spinner("Decoding latent themes…"):
                lda_model = perform_lda_modeling(X_tfidf, n_topics=n_topics)

            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda_model.components_):
                top_features_ind = topic.argsort()[:-10 - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                
                st.markdown(f"<div style='margin-bottom: 1.5rem;'>", unsafe_allow_html=True)
                st.markdown(f"<div class='cluster-badge'>Theme Axis {topic_idx + 1}</div>", unsafe_allow_html=True)
                chips_html = "".join(f"<div class='chip' style='border-color: rgba(255,255,255,0.05);'>{kw}</div>" for kw in top_features)
                st.markdown(chips_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("Upload at least 2 documents for topic modeling.")


    # ---------------------------------------------------------------
    # TAB 1 — Clustering
    # ---------------------------------------------------------------
    with tab1:
        st.subheader("Semantic K-Means Clustering")
        with st.expander("How does K-Means Clustering work?"):
            st.write(
                "K-Means partitions documents into *k* groups by minimising distance to cluster centroids. "
                "We use the **Silhouette Score** (−1 to 1) to auto-suggest the best *k* — higher = better separation."
            )

        if len(raw_docs) >= 2:
            with st.spinner("Evaluating optimal cluster count…"):
                suggested_k, scores_per_k = calculate_optimal_clusters(X)

            # -- Silhouette Chart --
            if len(raw_docs) >= 3 and scores_per_k:
                st.markdown("#### Silhouette Score Evaluation")
                fig_sil = render_silhouette_chart(scores_per_k)
                if fig_sil:
                    fig_sil.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig_sil, width='stretch')
                    st.caption("Higher silhouette score → better-separated clusters. The peak indicates the recommended *k*.")
            elif len(raw_docs) < 3:
                st.info("Need at least 3 documents for silhouette analysis.")

            k = st.slider(
                "Number of Clusters (K-Means)",
                min_value=1,
                max_value=len(raw_docs),
                value=suggested_k
            )

            if k > 0:
                with st.spinner("Running K-Means…"):
                    labels = perform_kmeans_clustering(X, k)
                cluster_df = pd.DataFrame({"Document": filenames, "Cluster": labels})

                if k > 1:
                    coords = apply_dimensionality_reduction(X, n_components=2)
                    cluster_df['PCA1'] = coords[:, 0]
                    cluster_df['PCA2'] = coords[:, 1]
                    cluster_df['Cluster'] = cluster_df['Cluster'].astype(str)

                    fig = px.scatter(
                        cluster_df, x='PCA1', y='PCA2', color='Cluster',
                        hover_name='Document', title="Semantic Topology (Latent Space)",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig.update_traces(
                        marker=dict(size=12, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.1)')),
                        selector=dict(mode='markers')
                    )
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)


                else:
                    st.info("K=1 — All documents in a single cluster. PCA scatter disabled.")

                # -- Cluster-level features --
                cluster_texts = {i: "" for i in range(k)}
                for label, text in zip(labels, raw_docs):
                    cluster_texts[label] += text + " "

                cluster_list = [cluster_texts[i] for i in range(k)]

                with st.spinner("Extracting cluster insights…"):
                    processed_clusters = [execute_preprocessing_pipeline(c, preserve_numeric=preserve_numbers) for c in cluster_list]
                    cluster_X, cluster_vectorizer = extract_tfidf_features(processed_clusters)

                    cluster_vocab_size = len(cluster_vectorizer.get_feature_names_out())
                    dynamic_top_n = max(3, min(10, int(0.1 * cluster_vocab_size)))

                    cluster_keywords = identify_top_keywords(
                        cluster_vectorizer, cluster_X, top_n=dynamic_top_n
                    )

                    # Per-document summaries within each cluster
                    cluster_doc_summaries = {}
                    for cid in range(k):
                        doc_indices = [i for i, lbl in enumerate(labels) if lbl == cid]
                        per_doc = []
                        for idx in doc_indices:
                            readable = prepare_text_for_summary(raw_docs[idx], preserve_numeric=preserve_numbers)
                            sents = generate_extractive_summary(readable, cluster_vectorizer, top_n=2)
                            per_doc.append((filenames[idx], sents))
                        cluster_doc_summaries[cid] = per_doc

                st.markdown("<div style='margin-bottom: 2.5rem;'></div>", unsafe_allow_html=True)
                st.markdown("<div class='sidebar-section-header'>Cluster Insights</div>", unsafe_allow_html=True)


                for cluster_id in range(k):
                    st.markdown(f"<div class='content-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='cluster-badge'>Cluster {cluster_id}</div>", unsafe_allow_html=True)

                    # Keywords as chips
                    st.markdown("<div style='margin-bottom: 0.5rem; font-weight: 600; font-size: 0.9rem; color: #FFF;'>Core Semantic Markers</div>", unsafe_allow_html=True)
                    chips_html = "".join(
                        f"<div class='chip'>{kw}</div>" for kw in cluster_keywords[cluster_id]
                    )
                    st.markdown(chips_html, unsafe_allow_html=True)


                    # Per-document summary
                    st.markdown("<div style='margin-top: 1.5rem; margin-bottom: 0.5rem; font-weight: 600;'>Document Syntheses</div>", unsafe_allow_html=True)
                    for doc_name, sents in cluster_doc_summaries[cluster_id]:
                        if sents:
                            combined = " ".join(sents)
                            st.markdown(
                                f"<div class='summary-item'><strong>{doc_name}:</strong> {combined}</div>",
                                unsafe_allow_html=True
                            )

                    # Document inspection button
                    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                    cluster_docs_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                    doc_count = len(cluster_docs_indices)
                    
                    if st.button(f"Inspect {doc_count} Documents in Cluster {cluster_id}", key=f"btn_cluster_{cluster_id}", use_container_width=True):
                        # Show first doc as default or a simple list? 
                        # The original code showed a button for EACH document. I'll stick to a toggle/expander or just keep the buttons but style them.
                        pass
                    
                    with st.expander("Expand Document List"):
                        for idx in cluster_docs_indices:
                            doc_name = filenames[idx]
                            if st.button(doc_name, key=f"btn_cl_doc_{cluster_id}_{idx}", use_container_width=True):
                                cleaned_doc = prepare_text_for_summary(raw_docs[idx], preserve_numeric=preserve_numbers)
                                doc_sents = [s for dn, ss in cluster_doc_summaries[cluster_id] if dn == doc_name for s in ss]
                                show_document_modal(
                                    doc_name, raw_docs[idx], cleaned_doc,
                                    cluster_keywords[cluster_id], doc_sents
                                )

                    st.markdown("</div>", unsafe_allow_html=True)


            else:
                st.markdown("<div class='sidebar-section-header'>Global Repository</div>", unsafe_allow_html=True)
                global_keywords = identify_top_keywords(vectorizer, X, top_n=8)
                readable_docs = [prepare_text_for_summary(doc, preserve_numeric=preserve_numbers) for doc in raw_docs]
                global_summaries = [generate_extractive_summary(doc, vectorizer, top_n=4) for doc in readable_docs]
                
                for idx, doc_name in enumerate(filenames):
                    st.markdown(f"<div class='content-card' style='padding: 1.25rem 1.5rem; display: flex; justify-content: space-between; align-items: center;'>", unsafe_allow_html=True)
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        st.markdown(f"**{doc_name}**", unsafe_allow_html=True)
                        tags = "".join(f"<span class='chip' style='font-size: 0.65rem; padding: 2px 8px; margin-top: 4px;'>{kw}</span>" for kw in global_keywords[idx][:3])
                        st.markdown(tags, unsafe_allow_html=True)
                    with col2:
                        if st.button("Analyze", key=f"btn_all_{idx}", use_container_width=True):
                            show_document_modal(doc_name, raw_docs[idx], readable_docs[idx], global_keywords[idx], global_summaries[idx])
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Upload at least 2 documents to view clustering.")

            st.markdown("<div class='sidebar-section-header'>Available Documents</div>", unsafe_allow_html=True)
            global_keywords = identify_top_keywords(vectorizer, X, top_n=8)
            readable_docs = [prepare_text_for_summary(doc, preserve_numeric=preserve_numbers) for doc in raw_docs]
            global_summaries = [generate_extractive_summary(doc, vectorizer, top_n=4) for doc in readable_docs]
            for idx, doc_name in enumerate(filenames):
                st.markdown(f"<div class='content-card' style='padding: 1rem 1.25rem;'>", unsafe_allow_html=True)
                st.markdown(f"**{doc_name}**", unsafe_allow_html=True)
                if st.button("View Detailed Report", key=f"btn_single_{idx}", use_container_width=True):
                    show_document_modal(doc_name, raw_docs[idx], readable_docs[idx], global_keywords[idx], global_summaries[idx])
                st.markdown("</div>", unsafe_allow_html=True)

else:
    # --- Ultra-Premium Empty State ---
    st.markdown(
        """
        <div style='text-align:center; padding: 8rem 2rem; border: 1px dashed var(--border-medium); border-radius: 24px; background: rgba(255,255,255,0.01); backdrop-filter: blur(10px);'>
            <div style='font-size: 5rem; margin-bottom: 2rem; filter: saturate(0.5) opacity(0.8);'>🧪</div>
            <h1 style='font-weight: 800; margin-bottom: 0.75rem; font-size: 2.5rem; color: #FFF;'>Intelligence Awaits</h1>
            <p style='color: var(--text-secondary); max-width: 550px; margin: 0 auto 3rem auto; font-size: 1.15rem; font-weight: 400; line-height: 1.7;'>
                DocuMind AI is ready to distill your research. Select a corpus from the sidebar or upload your own neural data to begin the synthesis.
            </p>
            <div style='display: flex; justify-content: center; gap: 1.5rem;'>
                <div style='background: var(--accent-soft); border: 1px solid rgba(147, 51, 234, 0.3); color: var(--accent); padding: 12px 24px; border-radius: 12px; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;'>
                    1. Select Data
                </div>
                <div style='background: rgba(255,255,255,0.03); border: 1px solid var(--border-medium); color: var(--text-secondary); padding: 12px 24px; border-radius: 12px; font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase;'>
                    2. Scale Insights
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )