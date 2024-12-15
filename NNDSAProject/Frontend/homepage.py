import streamlit as st
from streamlit_extras.colored_header import colored_header

# Page Configuration
st.set_page_config(
    page_title="Neural Network Library",
    page_icon="🧠",
    layout="centered"
)

# Title and Subheader
st.title("Neural Network Library 🧠")
st.subheader("Build, train, and visualize neural networks with ease!")

st.write("""
Welcome to the Neural Network Library! This tool allows you to create and train neural networks 
for various tasks, including regression, classification, and image data processing.
""")

# Task Selection Header
colored_header(
    label="Select Your Task",
    description="Click on a task to start building your neural network!",
    color_name="blue-70"
)

# Task Buttons
col1, col2, col3 = st.columns(3, gap="medium")




with col1:
    if st.button("🔢 Regression"):
        st.query_params.update(page="regressionpage")  # Set query parameter
        st.write('<meta http-equiv="refresh" content="0; URL=Pages/regressionpage.py">', unsafe_allow_html=True)

with col2:
    if st.button("🧾 Classification"):
        st.experimental_set_query_params(page="classification")


with col3:
    if st.button("💬 NLP Tasks"):
        st.experimental_set_query_params(page="nlp")

# Footer
st.markdown("---")
st.write("Developed with ❤️ for AI enthusiasts.")

