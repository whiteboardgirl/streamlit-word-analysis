import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from nltk import ngrams
from collections import Counter
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize stopwords
stop_words = set(stopwords.words('english'))

# Streamlit App
st.title("Aby Warburg-inspired Word Analysis Tool")

# Step 1: Upload Dataset
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Select the text column for analysis
    text_column = st.selectbox("Select the text column for analysis", df.columns)

    # Step 2: Preprocess and Analyze Text
    if st.button("Analyze Words"):
        # Combine all text into a single string
        all_text = ' '.join(df[text_column].dropna().astype(str))

        # Tokenize the text using split (basic whitespace tokenization)
        words = all_text.lower().split()

        # Remove stopwords and non-alphabetic tokens
        words = [word for word in words if word.isalpha() and word not in stop_words]

        # Word Connection Visualization
        st.subheader("Word Connection Network")
        G = nx.Graph()
        for i in range(len(words) - 1):
            G.add_edge(words[i], words[i + 1])
        plt.figure(figsize=(10, 8))
        nx.draw_networkx(G, with_labels=True, node_size=20, font_size=10, node_color='skyblue', edge_color='gray')
        plt.title('Connections Between Words')
        st.pyplot(plt)

        # Bigram Network Visualization
        st.subheader("Bigram Network")
        bigrams = list(ngrams(words, 2))
        bigram_counts = Counter(bigrams)
        G = nx.Graph()
        for pair, count in bigram_counts.items():
            G.add_edge(pair[0], pair[1], weight=count)
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, k=1.5)
        nx.draw_networkx(G, pos, node_color='skyblue', edge_color='gray', node_size=500, font_size=10, with_labels=True)
        plt.title('Bigram Network')
        st.pyplot(plt)

        # Word Embeddings Visualization with PCA
        st.subheader("Word Embeddings (PCA)")
        model = Word2Vec([words], vector_size=100, window=5, min_count=1, workers=4)
        word_vectors = model.wv[model.wv.key_to_index]
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(word_vectors)
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        for i, word in enumerate(model.wv.key_to_index):
            plt.annotate(word, xy=(pca_result[i, 0], pca_result[i, 1]))
        plt.title('Word Embeddings Visualization (PCA)')
        st.pyplot(plt)
