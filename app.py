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
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b1120;
        --panel: #111827;
        --panel-2: #0f172a;
        --card: rgba(15, 23, 42, 0.82);
        --border: rgba(148, 163, 184, 0.16);
        --border-medium: rgba(148, 163, 184, 0.22);
        --text: #e5e7eb;
        --text-secondary: #94a3b8;
        --muted: #94a3b8;
        --accent: #7c3aed;
        --accent-2: #60a5fa;
        --accent-soft: rgba(124, 58, 237, 0.12);
        --success: #22c55e;
        --warning: #f59e0b;
        --shadow: 0 10px 30px rgba(2, 6, 23, 0.35);
        --radius-xl: 24px;
        --radius-lg: 18px;
        --radius-md: 14px;
    }

    html, body, [class*="css"] {
        font-family: "Inter", sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(124, 58, 237, 0.14), transparent 26%),
            radial-gradient(circle at top right, rgba(96, 165, 250, 0.10), transparent 24%),
            linear-gradient(180deg, #0b1120 0%, #0f172a 100%);
        color: var(--text);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(15,23,42,0.98) 0%, rgba(17,24,39,0.98) 100%);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--text);
    }

    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px dashed rgba(96, 165, 250, 0.35);
        border-radius: 18px;
        padding: 0.6rem;
    }

    .dm-hero {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(124, 58, 237, 0.18), rgba(96, 165, 250, 0.10)),
            rgba(15, 23, 42, 0.78);
        border: 1px solid var(--border);
        border-radius: 28px;
        padding: 2rem 2rem 1.5rem 2rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.4rem;
    }

    .dm-hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.03) 35%, transparent 70%);
        pointer-events: none;
    }

    .dm-kicker {
        display: inline-block;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(96, 165, 250, 0.22);
        background: rgba(96, 165, 250, 0.08);
        color: #bfdbfe;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }

    .dm-title {
        font-size: clamp(2rem, 4vw, 3.4rem);
        line-height: 1.02;
        font-weight: 800;
        color: #f8fafc;
        margin: 0 0 0.65rem 0;
        letter-spacing: -0.03em;
    }

    .dm-subtitle {
        max-width: 920px;
        color: #cbd5e1;
        font-size: 1.02rem;
        line-height: 1.75;
        margin-bottom: 1.15rem;
    }

    .dm-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-top: 0.2rem;
    }

    .dm-chip,
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.52rem 0.82rem;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.88);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #dbeafe;
        font-size: 0.84rem;
        font-weight: 600;
        margin: 0.25rem 0.35rem 0.25rem 0;
    }

    .sidebar-section-header {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #bfdbfe;
        margin: 1rem 0 0.65rem 0;
    }

    .dm-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(17, 24, 39, 0.88));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.2rem 1.2rem 1.05rem;
        box-shadow: var(--shadow);
        min-height: 140px;
    }

    .dm-card-label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
    }

    .dm-card-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }

    .dm-card-meta {
        color: #cbd5e1;
        font-size: 0.92rem;
        line-height: 1.5;
    }

    .kpi-container {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 1rem;
        margin: 1rem 0 1.25rem 0;
    }

    .kpi-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(17, 24, 39, 0.88));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.2rem 1.2rem 1.05rem;
        box-shadow: var(--shadow);
        min-height: 130px;
    }

    .kpi-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 0.45rem;
        letter-spacing: -0.02em;
        word-break: break-word;
    }

    .kpi-label {
        color: var(--text-secondary);
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .content-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.88), rgba(17, 24, 39, 0.88));
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.4rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .cluster-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.45rem 0.8rem;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.14);
        border: 1px solid rgba(124, 58, 237, 0.24);
        color: #e9d5ff;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }

    .doc-viewer {
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.25rem;
        color: var(--text);
        line-height: 1.8;
        white-space: pre-wrap;
        max-height: 70vh;
        overflow-y: auto;
    }

    .dm-divider {
        height: 1px;
        width: 100%;
        background: linear-gradient(90deg, transparent, rgba(148,163,184,0.22), transparent);
        margin: 1.35rem 0 1.1rem 0;
    }

    .dm-empty {
        background: rgba(15, 23, 42, 0.75);
        border: 1px dashed rgba(148, 163, 184, 0.22);
        border-radius: 24px;
        padding: 2.2rem 1.2rem;
        text-align: center;
        color: var(--muted);
        margin-top: 1rem;
    }

    .dm-empty h3 {
        color: #f8fafc;
        margin-bottom: 0.55rem;
    }

    .dm-note {
        background: rgba(96, 165, 250, 0.08);
        border: 1px solid rgba(96, 165, 250, 0.18);
        border-radius: 16px;
        padding: 0.9rem 1rem;
        color: #dbeafe;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
        background: rgba(15, 23, 42, 0.52);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 0.45rem;
        margin-top: 0.4rem;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 14px;
        padding: 0 1rem;
        color: #cbd5e1;
        font-weight: 600;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.22), rgba(96, 165, 250, 0.16)) !important;
        color: #f8fafc !important;
        border: 1px solid rgba(124, 58, 237, 0.18);
    }

    div[data-testid="stMetric"] {
        background: transparent;
        border: none;
        box-shadow: none;
    }

    .stButton > button {
        width: 100%;
        border-radius: 14px;
        border: 1px solid rgba(96, 165, 250, 0.18);
        background: linear-gradient(180deg, rgba(30,41,59,0.96), rgba(15,23,42,0.96));
        color: #eff6ff;
        font-weight: 600;
        padding: 0.65rem 0.9rem;
        transition: all 0.18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(124, 58, 237, 0.34);
        box-shadow: 0 10px 20px rgba(2, 6, 23, 0.25);
    }

    .streamlit-expanderHeader {
        border-radius: 14px;
        color: #f8fafc !important;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid var(--border);
    }

    div[data-testid="stExpander"] {
        border: none !important;
    }

    .stAlert {
        border-radius: 16px;
    }

    .stMarkdown, p, li, label {
        color: var(--text);
    }

    .dm-keyword-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.4rem;
        margin-bottom: 0.6rem;
    }

    .dm-keyword-pill {
        display: inline-block;
        padding: 0.42rem 0.72rem;
        border-radius: 999px;
        background: rgba(124, 58, 237, 0.12);
        border: 1px solid rgba(124, 58, 237, 0.24);
        color: #e9d5ff;
        font-size: 0.8rem;
        font-weight: 600;
    }

    @media (max-width: 1100px) {
        .kpi-container {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 700px) {
        .kpi-container {
            grid-template-columns: 1fr;
        }
    }
</style>
"""
# ---------------------------------------------------------------------------
# Plotly Theme
# ---------------------------------------------------------------------------

PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(family='Inter', color='#8B949E'),
    title_font=dict(size=20, color='#E6EDF3', family='Inter'),
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
    keyword_html = "".join(
        f"<span class='dm-keyword-pill'>{kw}</span>" for kw in keywords[:10]
    )

    summary_text = " ".join(summary_sentences) if summary_sentences else "No summary generated."

    st.markdown(
        f"""
        <div style="margin-bottom: 1.25rem;">
            <h1 style="font-size: 2rem; margin-bottom: 0.5rem; color: #f8fafc;">{doc_name}</h1>
            <div class="dm-keyword-pills">{keyword_html}</div>
            <div class="content-card" style="margin-top: 0.75rem;">
                <div class="dm-card-label">Summary</div>
                <div class="dm-card-meta" style="font-size: 1rem; color: #e2e8f0;">
                    {summary_text}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    view_highlighted, view_original = st.tabs(["Highlighted Analysis", "Original Text"])

    with view_highlighted:
        highlighted = _highlight_text(cleaned_text, keywords, summary_sentences)
        st.markdown(f"<div class='doc-viewer'>{highlighted}</div>", unsafe_allow_html=True)

    with view_original:
        st.markdown(f"<div class='doc-viewer'>{raw_text}</div>", unsafe_allow_html=True)



# ---------------------------------------------------------------------------
# Page Config & CSS Injection
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="DocuMind AI",
    page_icon="🧠"
)

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Hero Banner
st.markdown(
    """
    <section class="dm-hero">
        <div class="dm-kicker">AI Document Intelligence Workspace</div>
        <h1 class="dm-title">DocuMind AI</h1>
        <p class="dm-subtitle">
            Document intelligence for research, analysis, and semantic discovery.
            Upload PDFs or text files, compare semantic similarity, discover hidden themes,
            cluster related documents, and extract readable summaries from complex corpora.
        </p>
        <div class="dm-chip-row">
            <span class="dm-chip">📄 PDF / TXT</span>
            <span class="dm-chip">🧮 TF-IDF</span>
            <span class="dm-chip">🧠 SBERT</span>
            <span class="dm-chip">🗂️ Clustering</span>
            <span class="dm-chip">🧭 Topics</span>
            <span class="dm-chip">✂️ Summaries</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.markdown("<div class='sidebar-section-header'>Data Source</div>", unsafe_allow_html=True)
CORPUS_OPTIONS = {
    "Research Repository (PDF)": "research_documents/pdf_papers",
    "Technical Articles (TXT)": "research_documents",
    "Semantic Equivalence Demo": "research_documents/semantic_demo",
    "Custom Machine Upload": None,
}

selected_corpus_label = st.sidebar.selectbox(
    "Intel Source",
    options=list(CORPUS_OPTIONS.keys()),
    label_visibility="collapsed"
)
selected_corpus_path = CORPUS_OPTIONS[selected_corpus_label]

uploaded_files = st.sidebar.file_uploader(
    "Upload Source Fragments (.txt, .pdf)",
    accept_multiple_files=True,
    disabled=(selected_corpus_path is not None),
    label_visibility="visible"
)

st.sidebar.markdown("<div class='sidebar-section-header'>Vectorization Engine</div>", unsafe_allow_html=True)
vectorization_mode = st.sidebar.radio(
    "Processing Mode",
    ["Neural Embeddings (SBERT)", "Lexical Frequency (TF-IDF)"],
    index=0,
    label_visibility="collapsed"
)
use_semantic = "Neural" in vectorization_mode

st.sidebar.markdown("<div class='sidebar-section-header'>General Settings</div>", unsafe_allow_html=True)
preserve_numbers = st.sidebar.toggle(
    "Preserve Numeric Context",
    value=True,
    disabled=use_semantic
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

    # -- Determine Suggested K for KPI --
    with st.spinner("Analyzing semantic patterns…"):
        suggested_k, _ = calculate_optimal_clusters(X)

    # -- Modern KPI Cards --
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
            <div class='kpi-value'>{suggested_k}</div>
            <div class='kpi-label'>Suggested Clusters</div>
        </div>
        <div class='kpi-card'>
            <div class='kpi-value'>{engine_label}</div>
            <div class='kpi-label'>Active Engine</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

    # ---------------------------------------------------------------------------
    # Information Architecture Restructuring
    # ---------------------------------------------------------------------------
    
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    tab_clusters, tab_topics, tab_similarity, tab_explorer = st.tabs([
    "Clusters", "Topics", "Similarity", "Document Explorer"
])

    # ---------------------------------------------------------------
    # CLUSTERS TAB
    # ---------------------------------------------------------------
    with tab_clusters:
        st.markdown("<div class='sidebar-section-header'>Semantic Mapping</div>", unsafe_allow_html=True)
        if len(raw_docs) >= 2:
            k = st.slider(
                "Cluster Resolution (K-Means)",
                min_value=1,
                max_value=len(raw_docs),
                value=suggested_k,
                help="Adjust the granularity of semantic groupings."
            )

            if k > 0:
                with st.spinner("Synthesizing clusters…"):
                    labels = perform_kmeans_clustering(X, k)
                cluster_df = pd.DataFrame({"Document": filenames, "Cluster": labels})

                if k > 1:
                    coords = apply_dimensionality_reduction(X, n_components=2)
                    cluster_df['PCA1'] = coords[:, 0]
                    cluster_df['PCA2'] = coords[:, 1]
                    cluster_df['Cluster'] = cluster_df['Cluster'].astype(str)

                    fig = px.scatter(
                        cluster_df, x='PCA1', y='PCA2', color='Cluster',
                        hover_name='Document', title="Semantic Topology",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig.update_traces(
                        marker=dict(size=12, opacity=0.8, line=dict(width=1, color='rgba(255,255,255,0.1)')),
                        selector=dict(mode='markers')
                    )
                    fig.update_layout(**PLOTLY_LAYOUT)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("K=1 — All documents in a single cluster.")

                # -- Cluster-level features & Synthesis --
                cluster_texts = {i: "" for i in range(k)}
                for label, text in zip(labels, raw_docs):
                    cluster_texts[label] += text + " "

                cluster_list = [cluster_texts[i] for i in range(k)]

                with st.spinner("Extracting semantic markers…"):
                    processed_clusters = [execute_preprocessing_pipeline(c, preserve_numeric=preserve_numbers) for c in cluster_list]
                    cluster_X, cluster_vectorizer = extract_tfidf_features(processed_clusters)

                    cluster_vocab_size = len(cluster_vectorizer.get_feature_names_out())
                    dynamic_top_n = max(5, min(12, int(0.12 * cluster_vocab_size)))

                    cluster_keywords = identify_top_keywords(
                        cluster_vectorizer, cluster_X, top_n=dynamic_top_n
                    )

                    # Cluster Synthesis
                    cluster_synthesis = {}
                    for cid in range(k):
                        # Join all text for the cluster and get top sentences
                        full_cluster_text = cluster_texts[cid]
                        # Clean it up
                        readable_cluster = prepare_text_for_summary(full_cluster_text, preserve_numeric=preserve_numbers)
                        # Extract top 3 sentences for the WHOLE cluster
                        synthesis_sents = generate_extractive_summary(readable_cluster, cluster_vectorizer, top_n=3)
                        cluster_synthesis[cid] = " ".join(synthesis_sents)

                # -- Display Cluster Cards --
                for cluster_id in range(k):
                    st.markdown(f"<div class='content-card'>", unsafe_allow_html=True)
                    st.markdown(f"<div class='cluster-badge'>Cluster {cluster_id}</div>", unsafe_allow_html=True)

                    # Synthesis
                    st.markdown(f"<p style='font-size: 1.05rem; line-height: 1.6; color: #E2E8F0; margin-bottom: 1.5rem;'>{cluster_synthesis[cluster_id]}</p>", unsafe_allow_html=True)

                    # Keywords
                    chips_html = "".join(f"<div class='chip'>{kw}</div>" for kw in cluster_keywords[cluster_id])
                    st.markdown(chips_html, unsafe_allow_html=True)

                    # Document List toggle
                    cluster_docs_indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
                    with st.expander(f"Inspect {len(cluster_docs_indices)} Neural Sources"):
                        for idx in cluster_docs_indices:
                            doc_name = filenames[idx]
                            if st.button(f"Analyze: {doc_name}", key=f"btn_cl_doc_{cluster_id}_{idx}", use_container_width=True):
                                # Just a quick way to find the pre-computed summary if needed, or just run a quick one
                                sents = generate_extractive_summary(prepare_text_for_summary(raw_docs[idx], preserve_numeric=preserve_numbers), cluster_vectorizer, top_n=3)
                                cleaned_doc = prepare_text_for_summary(raw_docs[idx], preserve_numeric=preserve_numbers)
show_document_modal(doc_name, raw_docs[idx], cleaned_doc, cluster_keywords[cluster_id], sents)

                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Upload at least 2 documents to view clustering.")

    # ---------------------------------------------------------------
    # TOPICS TAB (LDA)
    # ---------------------------------------------------------------
    with tab_topics:
        st.markdown("<div class='sidebar-section-header'>Thematic Discovery</div>", unsafe_allow_html=True)
        if len(raw_docs) >= 2:
            st.markdown("<div class='content-card' style='padding: 2rem;'>", unsafe_allow_html=True)
            st.markdown("<p style='color: var(--text-secondary); margin-bottom: 2rem; font-size: 0.95rem;'>Latent Dirichlet Allocation identifies the fundamental thematic axes across your corpus.</p>", unsafe_allow_html=True)
            
            n_topics = st.slider("Thematic Resolution", min_value=2, max_value=min(6, len(raw_docs)), value=3, key="lda_slider_final")

            with st.spinner("Decoding latent themes…"):
                lda_model = perform_lda_modeling(X_tfidf, n_topics=n_topics)

            feature_names = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda_model.components_):
                top_features_ind = topic.argsort()[:-10 - 1:-1]
                top_features = [feature_names[i] for i in top_features_ind]
                
                st.markdown(f"<div style='margin-bottom: 2.5rem;'>", unsafe_allow_html=True)
                st.markdown(f"<div class='cluster-badge' style='background: rgba(255,255,255,0.03); color: #FFF; border-color: var(--border-medium);'>Theme Axis {topic_idx + 1}</div>", unsafe_allow_html=True)
                chips_html = "".join(f"<div class='chip' style='border-color: rgba(255,255,255,0.05);'>{kw}</div>" for kw in top_features)
                st.markdown(chips_html, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.warning("Need more documents for topic modeling.")


    # ---------------------------------------------------------------
    # SIMILARITY TAB
    # ---------------------------------------------------------------
    with tab_similarity:
        st.markdown("<div class='sidebar-section-header'>Semantic Proximity</div>", unsafe_allow_html=True)
        if len(raw_docs) >= 2:
            st.markdown("<p style='color: var(--text-secondary); margin-bottom: 2rem; font-size: 0.95rem;'>Pairwise cosine similarity matrix mapping lexical and semantic relationships.</p>", unsafe_allow_html=True)
            with st.spinner("Calculating matrix…"):
                similarity = calculate_cosine_similarity(X)
                fig_sim = render_similarity_heatmap(similarity, filenames)
                fig_sim.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_sim, use_container_width=True)
        else:
            st.warning("Upload at least 2 documents.")


    # ---------------------------------------------------------------
    # EXPLORER TAB
    # ---------------------------------------------------------------
    with tab_explorer:
        st.markdown("<div class='sidebar-section-header'>Neural Repository</div>", unsafe_allow_html=True)
        global_keywords = identify_top_keywords(vectorizer, X_tfidf, top_n=10)
        readable_docs = [prepare_text_for_summary(doc, preserve_numeric=preserve_numbers) for doc in raw_docs]
        global_summaries = [generate_extractive_summary(doc, vectorizer, top_n=5) for doc in readable_docs]
        
        # Search filter
        search_query = st.text_input("Filter Neural Repository", placeholder="🔍 Search by name or content fragment...", label_visibility="collapsed")

        
        filtered_indices = [
            i for i, name in enumerate(filenames) 
            if search_query.lower() in name.lower() or search_query.lower() in raw_docs[i].lower()
        ]

        if not filtered_indices:
            st.info("No documents match your query.")
        else:
            for idx in filtered_indices:
                doc_name = filenames[idx]
                st.markdown(f"<div class='content-card' style='padding: 1.5rem;'>", unsafe_allow_html=True)
                col_info, col_action = st.columns([0.8, 0.2])
                with col_info:
                    st.markdown(f"<h4 style='margin-bottom: 0.5rem;'>{doc_name}</h4>", unsafe_allow_html=True)
                    tags = "".join(f"<span class='chip' style='font-size: 0.7rem; padding: 3px 10px;'>{kw}</span>" for kw in global_keywords[idx][:4])
                    st.markdown(tags, unsafe_allow_html=True)
                with col_action:
                    st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                    if st.button("Synthesize", key=f"btn_explore_{idx}", use_container_width=True):
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
