import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI & Data Science FAQ Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =========================================================
# FILE SETTINGS
# =========================================================

VECTORIZER_FILE = "vectorizer.pkl"
FAQ_FILE = "faq_dataset.pkl"

SIMILARITY_THRESHOLD = 0.25


# =========================================================
# LOAD .ENV FILE
# =========================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# =========================================================
# GET SECRET / ENVIRONMENT VARIABLE
# =========================================================

def get_secret(name):
    """
    Get API key from Streamlit Secrets first,
    then fall back to environment variables.
    """

    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 900px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        text-align: center;
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .subtitle {
        text-align: center;
        opacity: 0.75;
        margin-bottom: 2rem;
        font-size: 1.05rem;
    }

    .answer-card {
        padding: 1.4rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-top: 1rem;
    }

    .source-badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(100,100,255,0.12);
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD FAQ DATA
# =========================================================

@st.cache_resource
def load_resources():
    try:
        vectorizer = joblib.load(VECTORIZER_FILE)
        faq_df = joblib.load(FAQ_FILE)

        # Make sure required columns exist
        if "Question" not in faq_df.columns:
            raise ValueError(
                "FAQ dataset must contain a 'Question' column."
            )

        if "Answer" not in faq_df.columns:
            raise ValueError(
                "FAQ dataset must contain an 'Answer' column."
            )

        return vectorizer, faq_df

    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        return None, None

    except Exception as e:
        st.error(f"Error loading FAQ resources: {e}")
        return None, None


# =========================================================
# SAVE NEW QUESTION AND ANSWER
# =========================================================

def save_new_faq(question, answer, faq_df):
    """
    Add a new AI-generated question/answer to FAQ dataset
    and rebuild the TF-IDF vectorizer.
    """

    try:

        # Avoid duplicates
        existing_questions = (
            faq_df["Question"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        if question.strip().lower() in existing_questions.values:
            return faq_df, False

        # Create new record
        new_row = pd.DataFrame(
            {
                "Question": [question.strip()],
                "Answer": [answer.strip()],
            }
        )

        # If dataset contains extra columns, add blank values
        for column in faq_df.columns:
            if column not in new_row.columns:
                new_row[column] = ""

        new_row = new_row[faq_df.columns]

        # Add new record
        updated_df = pd.concat(
            [faq_df, new_row],
            ignore_index=True
        )

        # Create NEW vectorizer
        new_vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2)
        )

        new_vectorizer.fit(
            updated_df["Question"].astype(str)
        )

        # Save updated files
        joblib.dump(updated_df, FAQ_FILE)
        joblib.dump(new_vectorizer, VECTORIZER_FILE)

        # Clear Streamlit cache
        st.cache_resource.clear()

        return updated_df, True

    except Exception as e:
        st.error(f"Could not save new FAQ: {e}")
        return faq_df, False


# =========================================================
# INITIALIZE GEMINI
# =========================================================

@st.cache_resource
def initialize_gemini():

    api_key = get_secret("GEMINI_API_KEY")

    # Also accept GOOGLE_API_KEY
    if not api_key:
        api_key = get_secret("GOOGLE_API_KEY")

    if not api_key:
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        return client

    except Exception as e:
        st.sidebar.warning(
            f"Gemini could not initialize: {e}"
        )
        return None


# =========================================================
# INITIALIZE OPENAI
# =========================================================

@st.cache_resource
def initialize_openai():

    api_key = get_secret("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        return client

    except Exception as e:
        st.sidebar.warning(
            f"OpenAI could not initialize: {e}"
        )
        return None


# =========================================================
# SYSTEM INSTRUCTIONS
# =========================================================

SYSTEM_PROMPT = """
You are an AI assistant for a Data Science and Artificial
Intelligence learning platform.

You specialize in:

- Data Science
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Python
- SQL
- NLP
- Computer Vision
- Power BI
- Tableau
- RAG
- Data Analytics
- Statistics
- Generative AI
- AI tools
- Programming concepts
- Course-related questions

Your answers should be:

1. Accurate
2. Beginner friendly
3. Clear and concise
4. Professional
5. Easy to understand

When useful, provide simple examples.

If the user asks something outside these subjects, you may still
answer general educational questions, but do not invent information
about course prices, certificates, schedules, or company policies
unless that information was supplied to you.
"""


# =========================================================
# GEMINI RESPONSE
# =========================================================

def ask_gemini(question, client):

    try:
        from google.genai import types

        response = client.models.generate_content(
            model="gemini-3.7-flash",
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=700,
            ),
        )

        return response.text

    except Exception as e:
        return None


# =========================================================
# CHATGPT RESPONSE
# =========================================================

def ask_chatgpt(question, client):

    try:

        response = client.responses.create(
            model="gpt-5.6",
            instructions=SYSTEM_PROMPT,
            input=question,
        )

        return response.output_text

    except Exception as e:
        return None


# =========================================================
# GET AI FALLBACK RESPONSE
# =========================================================

def get_ai_response(question, gemini_client, openai_client):

    # Try Gemini first
    if gemini_client:

        answer = ask_gemini(
            question,
            gemini_client
        )

        if answer:
            return answer, "Google Gemini"

    # If Gemini fails, try ChatGPT
    if openai_client:

        answer = ask_chatgpt(
            question,
            openai_client
        )

        if answer:
            return answer, "OpenAI ChatGPT"

    return None, None


# =========================================================
# SEARCH FAQ DATABASE
# =========================================================

def search_faq(question, vectorizer, faq_df):

    try:

        dataset_vectors = vectorizer.transform(
            faq_df["Question"].astype(str)
        )

        query_vector = vectorizer.transform(
            [question]
        )

        similarities = cosine_similarity(
            query_vector,
            dataset_vectors
        ).flatten()

        best_index = similarities.argmax()

        confidence = float(
            similarities[best_index]
        )

        return best_index, confidence

    except Exception as e:
        st.error(f"FAQ search error: {e}")
        return None, 0


# =========================================================
# LOAD EVERYTHING
# =========================================================

vectorizer, faq_df = load_resources()

gemini_client = initialize_gemini()
openai_client = initialize_openai()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        🤖 AI & Data Science Assistant
    </div>

    <div class="subtitle">
        Intelligent FAQ Search + Gemini + ChatGPT
    </div>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
    Ask questions about **Data Science, AI, Machine Learning,
    Python, SQL, Power BI, Deep Learning, NLP, RAG, courses,
    and more.**
    """
)


# =========================================================
# CHECK FILES
# =========================================================

if vectorizer is None or faq_df is None:

    st.error(
        """
        The application could not load the FAQ files.

        Make sure these files are in the same folder as app.py:

        • vectorizer.pkl
        • faq_dataset.pkl
        """
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 AI Assistant")

    st.markdown("---")

    st.subheader("📚 Knowledge Base")

    st.metric(
        "Total FAQs",
        len(faq_df)
    )

    st.markdown("---")

    st.subheader("🧠 AI Models")

    if gemini_client:
        st.success("✅ Google Gemini")

    else:
        st.caption("⚪ Gemini not configured")

    if openai_client:
        st.success("✅ ChatGPT")

    else:
        st.caption("⚪ ChatGPT not configured")

    if not gemini_client and not openai_client:

        st.warning(
            """
            Add GEMINI_API_KEY or
            OPENAI_API_KEY to enable
            AI fallback.
            """
        )

    st.markdown("---")

    st.subheader("📖 Topics")

    topics = [
        "Artificial Intelligence",
        "Machine Learning",
        "Data Science",
        "Python",
        "SQL",
        "Deep Learning",
        "NLP",
        "Computer Vision",
        "Power BI",
        "Tableau",
        "RAG",
        "Generative AI",
    ]

    for topic in topics:
        st.caption(f"• {topic}")


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("source"):
            st.caption(
                f"Source: {message['source']}"
            )


# =========================================================
# CHAT INPUT
# =========================================================

user_query = st.chat_input(
    "Ask me anything about AI or Data Science..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if user_query:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)


    # -----------------------------------------------------
    # SEARCH EXISTING FAQ
    # -----------------------------------------------------

    best_index, confidence = search_faq(
        user_query,
        vectorizer,
        faq_df
    )


    # -----------------------------------------------------
    # FAQ MATCH FOUND
    # -----------------------------------------------------

    if (
        best_index is not None
        and confidence >= SIMILARITY_THRESHOLD
    ):

        answer = str(
            faq_df.iloc[best_index]["Answer"]
        )

        matched_question = str(
            faq_df.iloc[best_index]["Question"]
        )

        source = "FAQ Knowledge Base"

        with st.chat_message("assistant"):

            st.markdown(answer)

            st.caption(
                f"📚 FAQ Match • "
                f"Confidence: {confidence:.1%}"
            )

            with st.expander(
                "View matched question"
            ):
                st.write(matched_question)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "source": source,
            }
        )


    # -----------------------------------------------------
    # NO FAQ MATCH - ASK AI
    # -----------------------------------------------------

    else:

        if gemini_client or openai_client:

            with st.chat_message("assistant"):

                with st.spinner(
                    "AI is thinking..."
                ):

                    ai_answer, source = get_ai_response(
                        user_query,
                        gemini_client,
                        openai_client,
                    )

                if ai_answer:

                    st.markdown(ai_answer)

                    st.caption(
                        f"✨ Generated by {source}"
                    )

                    # -------------------------------------
                    # SAVE NEW QUESTION AUTOMATICALLY
                    # -------------------------------------

                    updated_df, saved = save_new_faq(
                        user_query,
                        ai_answer,
                        faq_df,
                    )

                    if saved:

                        st.success(
                            "📚 New question added to "
                            "the FAQ knowledge base."
                        )

                        faq_df = updated_df

                    # Save to chat history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": ai_answer,
                            "source": source,
                        }
                    )

                else:

                    error_message = (
                        "I couldn't generate an AI response. "
                        "Please check your API key."
                    )

                    st.error(error_message)

        else:

            with st.chat_message("assistant"):

                st.warning(
                    """
                    I couldn't find this question in the FAQ
                    database and no AI API is configured.

                    Add a Gemini or OpenAI API key to enable
                    intelligent fallback answers.
                    """
                )


# =========================================================
# DATASET VIEWER
# =========================================================

st.markdown("---")

with st.expander(
    "📚 View FAQ Knowledge Base"
):

    st.dataframe(
        faq_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# CLEAR CHAT
# =========================================================

if st.sidebar.button(
    "🗑️ Clear Chat",
    use_container_width=True
):

    st.session_state.messages = []

    st.rerun()


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🤖 AI & Data Science FAQ Assistant • "
    "TF-IDF + Gemini + ChatGPT"
)
