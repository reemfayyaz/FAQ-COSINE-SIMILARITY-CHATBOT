import joblib
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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


# Initialize AI Models
@st.cache_resource
def initialize_ai_model():
    """Initialize Google Gemini or ChatGPT based on available API keys"""
    ai_model = None
    model_type = None
    
    # Try Google Gemini
    gemini_key = os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            ai_model = genai.GenerativeModel("gemini-pro")
            model_type = "Gemini"
            return ai_model, model_type
        except Exception as e:
            st.warning(f"Gemini initialization failed: {e}")
    
    # Try OpenAI ChatGPT
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            model_type = "ChatGPT"
            return None, model_type  # We'll use openai.ChatCompletion directly
        except Exception as e:
            st.warning(f"ChatGPT initialization failed: {e}")
    
    return ai_model, model_type


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


def get_ai_response(question, model_type, ai_model=None):
    """Get response from AI model (Gemini or ChatGPT)"""
    try:
        context = """You are a helpful AI assistant for a Data Science, Machine Learning, and AI course platform. 
        You specialize in answering questions about Data Science, Machine Learning, Python, SQL, Deep Learning, NLP, 
        Computer Vision, Power BI, Tableau, RAG, and course-related queries. 
        Provide clear, concise, and accurate answers. If the question is outside your domain, politely redirect to course topics."""
        
        if model_type == "Gemini":
            response = ai_model.generate_content(f"{context}\n\nQuestion: {question}")
            return response.text
        
        elif model_type == "ChatGPT":
            import openai
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": question}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content
    
    except Exception as e:
        return f"Error getting AI response: {str(e)}"


vectorizer, faq_df = load_resources()
ai_model, model_type = initialize_ai_model()

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

    # Display AI Model Status
    if model_type:
        st.sidebar.success(f"✅ AI Model: {model_type} (Fallback Enabled)")
    else:
        st.sidebar.warning("⚠️ No AI model configured. Only FAQ matching available.")

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
                # Fallback to AI Model if available
                if model_type:
                    st.info("📌 No close match in FAQ database. Using AI Assistant to answer...")
                    with st.spinner(f"Getting response from {model_type}..."):
                        ai_response = get_ai_response(user_query, model_type, ai_model)
                    st.success("### AI Assistant Response")
                    st.info(f"**Answer:** {ai_response}")
                    st.caption(f"Source: {model_type} AI Model")
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
