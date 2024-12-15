import streamlit as st
import Frontend.homepage,Frontend.regressionpage

params = st.experimental_get_query_params()
page = params.get("page", ["home"])[0]  # Default to 'home'

# Display the selected page
if page == "regression":
    Frontend.regressionpage.run()  # Call the regression page
else:
    Frontend.homepage.run() 