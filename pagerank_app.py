#Run this in terminal with Topic3.ipynb and starwars-full-interactions.json in the same directory
#streamlit run pagerank_app.py

import streamlit as st
import pandas as pd
import json
import tempfile

hide_streamlit_style = """
    <style>
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Import the functions from Topic3.ipynb
import functions

# Access functions from Topic3.ipynb
load_data_1 = functions.load_data_1
convert_to_2D_array_1 = functions.convert_to_2D_array_1
convert_to_unweighted_1 = functions.convert_to_unweighted_1
normalize_array_1 = functions.normalize_array_1
get_pagerank_1 = functions.get_pagerank_1
get_top_nodes_1 = functions.get_top_nodes_1

# Preloaded file path
PRELOADED_FILE_PATH = "starwars-full-interactions.json"

# Title and description
st.title("Star Wars PageRank Viewer")
st.write("""
This app calculates and displays the PageRank for characters in a Star Wars interaction network.
You can either upload a JSON file or use a preloaded Star Wars dataset.
""")

# Choose input method
input_choice = st.radio(
    "Select Input Method",
    options=["Upload a File", "Use Preloaded File"]
)

# Initialize temp_file_path
temp_file_path = None

if input_choice == "Upload a File":
    uploaded_file = st.file_uploader("Upload a JSON File", type=["json"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
else:
    temp_file_path = PRELOADED_FILE_PATH

# Load the data if a valid path exists
if temp_file_path:
    try:
        characters, interactions = load_data_1(temp_file_path)
        M = convert_to_2D_array_1(interactions, characters)

        # Sidebar options
        st.sidebar.header("PageRank Options")
        top_k = st.sidebar.number_input("Top K Nodes", min_value=1, value=5, step=1)
        alpha = st.sidebar.slider("Damping Factor (α)", min_value=0.1, max_value=1.0, value=0.85)
        threshold = st.sidebar.number_input("Convergence Threshold", min_value=1e-10, value=1e-6, step=1e-10, format="%.10f")
        max_iter = st.sidebar.number_input("Max Iterations", min_value=1, value=1000, step=1)
        use_weighted = st.sidebar.checkbox("Use Weighted PageRank", value=False)

        # Show the interaction matrix if the checkbox is selected
        if st.checkbox("Show Interaction Matrix"):
            st.write("### Interaction Matrix")
            st.dataframe(pd.DataFrame(M))

        # Compute PageRank
        st.write("### Results")
        if st.button("Run PageRank"):
            # Normalize matrix based on the user's choice
            if use_weighted:
                M_normalized = normalize_array_1(M)  # Weighted matrix normalization
            else:
                M_unweighted = convert_to_unweighted_1(M)
                M_normalized = normalize_array_1(M_unweighted)

            # Calculate PageRank
            pagerank_vector = get_pagerank_1(M_normalized, alpha=alpha, threshold=threshold, max_iter=max_iter)

            # Get top nodes
            top_nodes = get_top_nodes_1(pagerank_vector, characters, k=top_k)

            # Display top nodes
            top_nodes_df = pd.DataFrame([
                {
                    "Character": node["name"],
                    "PageRank": pagerank_vector[node["index"]][0]
                }
                for node in top_nodes
            ])
            st.write(f"Top {top_k} Characters by PageRank:")
            st.table(top_nodes_df)

    except json.JSONDecodeError:
        st.error("The selected file is not a valid JSON file.")
else:
    st.warning("Please upload a file or select the preloaded dataset.")
