import joblib
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(
    page_title="AI & Data Science FAQ Assistant",
    page_icon="🤖",
    layout="centered",
)

# Title & Description
st.title("🤖 AI & Data Science FAQ Assistant")
st.write(
    "Ask any question related to Data Science, Machine Learning, Python, SQL, or Course Details!"
)


# Load Vectorizer and FAQ Dataset
@st.cache_resource
def load_resources():
    try:
        # Load TF-IDF Vectorizer and DataFrame
        vectorizer = joblib.load("vectorizer.pkl")
        faq_df = joblib.load("faq_dataset.pkl")
        return vectorizer, faq_df
    except Exception as e:
        st.error(f"Error loading model or dataset files: {e}")
        return None, None


vectorizer, faq_df = load_resources()

if vectorizer is not None and faq_df is not None:
    # Sidebar Info
    st.sidebar.header("📊 Dataset Overview")
    st.sidebar.write(f"Total FAQs Loaded: **{len(faq_df)}**")

    st.sidebar.subheader("Available Topics")
    topics = [
        "Machine Learning",
        "Artificial Intelligence",
        "Data Science",
        "Python",
        "SQL",
        "Deep Learning",
        "NLP",
        "Computer Vision",
        "Power BI",
        "Tableau",
        "RAG",
        "Course & Fee Details",
    ]
    for topic in topics:
        st.sidebar.write(f"- {topic}")

    # User Input
    st.subheader("❓ Ask Your Question")
    user_query = st.text_input(
        "Type your question here:",
        placeholder="e.g., What is Data Science? or Do you provide certificates?",
    )

    if st.button("Search Answer", type="primary"):
        if user_query.strip():
            # Vectorize questions dataset & user query
            dataset_vectors = vectorizer.transform(faq_df["Question"])
            query_vector = vectorizer.transform([user_query])

            # Calculate similarity
            similarities = cosine_similarity(
                query_vector, dataset_vectors
            ).flatten()
            best_idx = similarities.argmax()
            confidence = similarities[best_idx]

            # Threshold check
            if confidence > 0.2:
                st.success("### Answer Found!")
                st.write(f"**Matched Question:** {faq_df.iloc[best_idx]['Question']}")
                st.info(f"**Answer:** {faq_df.iloc[best_idx]['Answer']}")
                st.caption(f"Match Confidence Score: {confidence:.2%}")
            else:
                st.warning(
                    "Sorry, I couldn't find a close answer to your question. Try rephrasing or asking about our courses, fees, or AI topics!"
                )
        else:
            st.error("Please enter a question first.")

    st.markdown("---")

    # Display full dataset table
    with st.expander("👀 View All Available Questions & Answers"):
        st.dataframe(faq_df, use_container_width=True)