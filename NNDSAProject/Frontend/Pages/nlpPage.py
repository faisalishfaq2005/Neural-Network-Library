import streamlit as st
from NLP.data_preprocessing import preprocess_text
from NLP.buildgraph import build_graph
from NLP.analyze_context import analyze_context
from NLP.extract_keywords import extract_keywords
from NLP.cluster_words import cluster_words

import streamlit as st
import pandas as pd

# Custom CSS to improve UI styling
st.markdown("""
    <style>
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 16px;
        }
        .stTextInput>label {
            font-size: 18px;
        }
        .stRadio>label {
            font-size: 18px;
        }
        .stTextArea>label {
            font-size: 18px;
        }
        .stMarkdown {
            font-size: 20px;
            font-weight: bold;
        }
        .stSubheader {
            font-size: 20px;
            color: #1f77b4;
        }
        .stBlockquote {
            font-size: 16px;
            font-style: italic;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.title("NLP Text Processing & Analysis")

# Step 1: Task selection
st.subheader("Select an NLP Task:")
task = st.radio("Choose a task", ("Context Analysis", "Keyword Extraction", "Word Clustering"))

# Sidebar for extra features (e.g., show example text)
st.sidebar.title("Example Text")
example_text = """
In the heart of the forest, a young fox named Finn roamed freely, his bright red fur blending with the vibrant autumn leaves.
Finn was known for his curiosity and his ability to navigate the woods with ease. He often spent hours exploring the tall trees,
searching for hidden paths and secret clearings.
"""
if st.sidebar.checkbox("Show Example Text"):
    st.sidebar.text_area("Example Text", value=example_text, height=150)

# Step 2: Take user input text
text_input = st.text_area("Enter the text to analyze:", height=150, placeholder="Type or paste your text here...")

# Optional: Cluster slider (for Word Clustering task)
num_clusters = None
if task == "Word Clustering":
    num_clusters = st.slider("Select number of clusters:", min_value=1, max_value=10, value=2)

# Function to display word frequency table
def display_word_frequencies(word_frequencies):
    df = pd.DataFrame(word_frequencies, columns=["Word", "Frequency"])
    st.table(df)

# Function to display word clusters
def display_clusters(clusters):
    st.subheader(f"Word Clusters (Total: {len(clusters)}):")
    for idx, cluster in enumerate(clusters):
        st.write(f"Cluster {idx+1}: {', '.join(cluster)}")

# Task Execution when "Proceed" is clicked
if text_input:
    # Preprocess the text and build the graph (these are placeholders for actual processing functions)
    words = preprocess_text(text_input)  # Preprocessing step
    graph = build_graph(words)  # Graph construction step

    if task == "Context Analysis":
        word_to_analyze = st.text_input("Enter a word to analyze its context:", "")
        
        # Show Proceed button after input for the word to analyze
        proceed_button = st.button("Proceed")

        if proceed_button and word_to_analyze:
            # Show processing indicator
            with st.spinner("Processing..."):
                context = analyze_context(graph, word_to_analyze)
                st.subheader(f"Context of '{word_to_analyze}':")
                st.write(context)

    elif task == "Keyword Extraction":
        proceed_button = st.button("Proceed")

        if proceed_button:
            # Show processing indicator
            with st.spinner("Processing..."):
                st.subheader("Top 10 Keywords Based on Degree Centrality:")
                keywords = extract_keywords(graph)
                display_word_frequencies(keywords)

    elif task == "Word Clustering":
        proceed_button = st.button("Proceed")

        if proceed_button and num_clusters:
            # Show processing indicator
            with st.spinner("Processing..."):
                clusters = cluster_words(graph, num_clusters)
                display_clusters(clusters)

else:
    st.markdown("Please enter some text above to get started!")
