import streamlit as st

# Set up the main page
st.set_page_config(
    page_title="Neural Network Library",
    page_icon="🧠",
    layout="centered"
)

# Title and description
st.title("Neural Network Library 🧠")
st.subheader("Build, train, and visualize neural networks with ease!")

st.write("""
Welcome to the Neural Network Library! This tool allows you to create and train neural networks 
for various tasks, including regression, classification, and image data processing.
""")

# User task selection
st.header("Select Your Task")
task = st.selectbox(
    "What type of problem do you want to solve?",
    ("Choose a task", "Regression", "Classification", "Image Data")
)

# Navigation buttons to respective pages
if task == "Regression":
    if st.button("Go to Regression"):
        st.experimental_set_query_params(page="regression")

elif task == "Classification":
    if st.button("Go to Classification"):
        st.experimental_set_query_params(page="classification")

elif task == "Image Data":
    if st.button("Go to Image Data"):
        st.experimental_set_query_params(page="image_data")

# Footer
st.markdown("---")
st.write("Developed with ❤️ for AI enthusiasts.")