import os
import joblib
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI & Data Science FAQ Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# FILES / SETTINGS
# ============================================================

FAQ_FILE = "faq_dataset.pkl"
VECTORIZER_FILE = "vectorizer.pkl"
SIMILARITY_THRESHOLD = 0.25

OPENAI_MODEL = "gpt-5.6"
GEMINI_MODEL = "gemini-3.7-flash"

# ============================================================
# OPTIONAL .ENV SUPPORT
# ============================================================

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #0f172a, #1e3a8a, #0f766e);
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.20);
        }

        .hero h1 {
            margin: 0;
            font-size: 2.6rem;
            font-weight: 800;
        }

        .hero p {
            margin-top: 0.7rem;
            margin-bottom: 0;
            opacity: 0.92;
            font-size: 1.05rem;
        }

        .status-card {
            border: 1px solid rgba(128,128,128,0.22);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stChatMessage"] {
            border: 1px solid rgba(128,128,128,0.16);
            border-radius: 16px;
            padding: 0.3rem 0.6rem;
            margin-bottom: 0.6rem;
        }

        .small-note {
            opacity: 0.72;
            font-size: 0.88rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# API KEY HELPER
# ============================================================

def get_secret(name: str):
    """
    Read an API key from Streamlit Secrets first,
    then fall back to environment variables / .env.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name)

# ============================================================
# LOAD FAQ DATA
# ============================================================

@st.cache_resource
def load_resources():
    try:
        faq_df = joblib.load(FAQ_FILE)

        if not isinstance(faq_df, pd.DataFrame):
            faq_df = pd.DataFrame(faq_df)

        if "Question" not in faq_df.columns or "Answer" not in faq_df.columns:
            raise ValueError(
                "faq_dataset.pkl must contain columns named 'Question' and 'Answer'."
            )

        faq_df = faq_df.copy()
        faq_df["Question"] = faq_df["Question"].astype(str)
        faq_df["Answer"] = faq_df["Answer"].astype(str)

        # Try to load the existing vectorizer.
        try:
            vectorizer = joblib.load(VECTORIZER_FILE)
        except Exception:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                lowercase=True,
                ngram_range=(1, 2),
            )
            vectorizer.fit(faq_df["Question"])

        return vectorizer, faq_df, None

    except Exception as e:
        return None, None, str(e)

# ============================================================
# SAVE / REBUILD FAQ KNOWLEDGE BASE
# ============================================================

def save_new_faq(question: str, answer: str, faq_df: pd.DataFrame):
    """
    Add a new AI-generated FAQ and rebuild the TF-IDF vectorizer.
    """
    try:
        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            return faq_df, False, "Question or answer is empty."

        existing = faq_df["Question"].astype(str).str.strip().str.lower()

        if question.lower() in existing.values:
            return faq_df, False, "This question already exists in the FAQ dataset."

        new_row = {column: "" for column in faq_df.columns}
        new_row["Question"] = question
        new_row["Answer"] = answer

        updated_df = pd.concat(
            [faq_df, pd.DataFrame([new_row])],
            ignore_index=True,
        )

        new_vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
        )
        new_vectorizer.fit(updated_df["Question"].astype(str))

        joblib.dump(updated_df, FAQ_FILE)
        joblib.dump(new_vectorizer, VECTORIZER_FILE)

        st.cache_resource.clear()
        return updated_df, True, "FAQ saved successfully."

    except Exception as e:
        return faq_df, False, f"Could not save FAQ: {e}"

# ============================================================
# OPENAI SETUP
# ============================================================

@st.cache_resource
def initialize_openai():
    api_key = get_secret("OPENAI_API_KEY")

    if not api_key:
        return None, "OPENAI_API_KEY not found."

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        return client, None
    except Exception as e:
        return None, str(e)

# ============================================================
# GEMINI SETUP
# ============================================================

@st.cache_resource
def initialize_gemini():
    api_key = get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")

    if not api_key:
        return None, "GEMINI_API_KEY / GOOGLE_API_KEY not found."

    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, str(e)

# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are a helpful AI assistant for a Data Science and Artificial
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
- Generative AI
- Statistics
- Data Analytics
- Course-related educational questions

Instructions:
- Give clear and beginner-friendly answers.
- Keep answers professional and practical.
- Use simple examples when helpful.
- Do not invent course prices, schedules, policies, certificates,
  or organization-specific facts that were not provided.
"""

# ============================================================
# ASK OPENAI
# ============================================================

def ask_openai(question: str, client):
    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            instructions=SYSTEM_PROMPT,
            input=question,
        )
        text = getattr(response, "output_text", None)
        if text:
            return text.strip(), None
        return None, "OpenAI returned an empty response."
    except Exception as e:
        return None, str(e)

# ============================================================
# ASK GEMINI
# ============================================================

def ask_gemini(question: str, client):
    try:
        from google.genai import types

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=question,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.4,
                max_output_tokens=700,
            ),
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip(), None

        return None, "Gemini returned an empty response."
    except Exception as e:
        return None, str(e)

# ============================================================
# FAQ SEARCH
# ============================================================

def search_faq(question: str, vectorizer, faq_df: pd.DataFrame):
    try:
        faq_vectors = vectorizer.transform(faq_df["Question"].astype(str))
        query_vector = vectorizer.transform([question])

        similarities = cosine_similarity(
            query_vector,
            faq_vectors,
        ).flatten()

        best_index = int(similarities.argmax())
        confidence = float(similarities[best_index])

        return best_index, confidence, None

    except Exception as e:
        return None, 0.0, str(e)

# ============================================================
# AI FALLBACK
# ============================================================

def get_ai_fallback(question, preferred_provider, openai_client, gemini_client):
    """
    Use the selected provider first, then try the other provider if available.
    """
    providers = []

    if preferred_provider == "ChatGPT":
        providers = [
            ("ChatGPT", openai_client, ask_openai),
            ("Gemini", gemini_client, ask_gemini),
        ]
    elif preferred_provider == "Gemini":
        providers = [
            ("Gemini", gemini_client, ask_gemini),
            ("ChatGPT", openai_client, ask_openai),
        ]
    else:
        providers = [
            ("Gemini", gemini_client, ask_gemini),
            ("ChatGPT", openai_client, ask_openai),
        ]

    errors = []

    for provider_name, client, function in providers:
        if client is None:
            continue

        answer, error = function(question, client)

        if answer:
            return answer, provider_name, None

        if error:
            errors.append(f"{provider_name}: {error}")

    return None, None, " | ".join(errors) if errors else "No AI provider is connected."

# ============================================================
# INITIALIZE APP
# ============================================================

vectorizer, faq_df, resource_error = load_resources()
openai_client, openai_error = initialize_openai()
gemini_client, gemini_error = initialize_gemini()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>🤖 AI & Data Science FAQ Assistant</h1>
        <p>
            Search your FAQ knowledge base first, then use Gemini or ChatGPT
            for questions that are not already covered.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# STOP IF FAQ FILES FAIL
# ============================================================

if resource_error:
    st.error("Could not load your FAQ files.")
    st.code(resource_error)
    st.info(
        "Make sure faq_dataset.pkl and vectorizer.pkl are in the same folder as app.py."
    )
    st.stop()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("⚙️ Control Panel")

    st.subheader("📚 Knowledge Base")
    st.metric("Total FAQs", len(faq_df))

    st.divider()

    st.subheader("🤖 AI Connections")

    if openai_client:
        st.success("✅ ChatGPT Connected")
        st.caption(f"Model: {OPENAI_MODEL}")
    else:
        st.error("❌ ChatGPT Not Connected")
        st.caption(openai_error)

    if gemini_client:
        st.success("✅ Gemini Connected")
        st.caption(f"Model: {GEMINI_MODEL}")
    else:
        st.error("❌ Gemini Not Connected")
        st.caption(gemini_error)

    st.divider()

    available_options = ["Automatic"]
    if openai_client:
        available_options.append("ChatGPT")
    if gemini_client:
        available_options.append("Gemini")

    preferred_provider = st.selectbox(
        "Preferred AI provider",
        options=available_options,
        help="The app uses this provider first when the FAQ database has no strong match.",
    )

    threshold = st.slider(
        "FAQ Match Threshold",
        min_value=0.10,
        max_value=0.80,
        value=SIMILARITY_THRESHOLD,
        step=0.05,
        help="Higher values require a closer FAQ match before the AI fallback is used.",
    )

    st.divider()

    save_mode = st.toggle(
        "Allow saving AI answers",
        value=True,
        help="Shows an Add to FAQ button after an AI-generated answer.",
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_faq = None
        st.rerun()

# ============================================================
# CONNECTION HELP
# ============================================================

if not openai_client and not gemini_client:
    st.warning(
        """
        **No AI provider is connected yet.**

        Add at least one API key:

        - `OPENAI_API_KEY`
        - `GEMINI_API_KEY`

        You can use a local `.env` file or Streamlit Cloud Secrets.
        """
    )

    with st.expander("🔑 Show API key setup instructions"):
        st.markdown(
            """
            **Local `.env` file**
            ```env
            OPENAI_API_KEY=your_openai_key_here
            GEMINI_API_KEY=your_gemini_key_here
            ```

            **Streamlit Cloud → Settings → Secrets**
            ```toml
            OPENAI_API_KEY = "your_openai_key_here"
            GEMINI_API_KEY = "your_gemini_key_here"
            ```

            Never upload your real API keys to GitHub.
            """
        )

# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_faq" not in st.session_state:
    st.session_state.pending_faq = None

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("source"):
            st.caption(message["source"])

# ============================================================
# SAVE PENDING AI ANSWER
# ============================================================

if st.session_state.pending_faq and save_mode:
    pending = st.session_state.pending_faq

    with st.container(border=True):
        st.subheader("📚 Add AI Answer to FAQ?")
        st.caption(
            "Review the AI-generated answer before permanently adding it "
            "to your FAQ knowledge base."
        )

        st.write(f"**Question:** {pending['question']}")
        st.write(f"**Source:** {pending['source']}")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "✅ Add to FAQ",
                type="primary",
                use_container_width=True,
                key="save_pending_faq",
            ):
                updated_df, saved, message = save_new_faq(
                    pending["question"],
                    pending["answer"],
                    faq_df,
                )

                if saved:
                    faq_df = updated_df
                    st.session_state.pending_faq = None
                    st.success("New question and answer added to your FAQ knowledge base.")
                    st.rerun()
                else:
                    st.warning(message)

        with col2:
            if st.button(
                "❌ Don't Save",
                use_container_width=True,
                key="discard_pending_faq",
            ):
                st.session_state.pending_faq = None
                st.rerun()

# ============================================================
# USER CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask a question about Data Science, AI, Python, SQL, Power BI, courses, and more..."
)

if user_query:
    user_query = user_query.strip()

    if user_query:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_query,
            }
        )

        with st.chat_message("user"):
            st.markdown(user_query)

        # Search local FAQ knowledge base first
        best_index, confidence, search_error = search_faq(
            user_query,
            vectorizer,
            faq_df,
        )

        if search_error:
            st.error(f"FAQ search error: {search_error}")

        # ====================================================
        # FAQ MATCH FOUND
        # ====================================================

        if (
            best_index is not None
            and confidence >= threshold
        ):
            matched_question = str(
                faq_df.iloc[best_index]["Question"]
            )

            answer = str(
                faq_df.iloc[best_index]["Answer"]
            )

            with st.chat_message("assistant"):
                st.markdown(answer)
                st.caption(
                    f"📚 FAQ Knowledge Base • Match confidence: {confidence:.1%}"
                )

                with st.expander("Matched FAQ"):
                    st.write(matched_question)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "source": (
                        f"📚 FAQ Knowledge Base • "
                        f"Match confidence: {confidence:.1%}"
                    ),
                }
            )

            st.session_state.pending_faq = None

        # ====================================================
        # NO STRONG MATCH → ASK AI
        # ====================================================

        else:
            if openai_client or gemini_client:
                with st.chat_message("assistant"):
                    with st.spinner("Searching the AI assistant..."):
                        answer, source, ai_error = get_ai_fallback(
                            user_query,
                            preferred_provider,
                            openai_client,
                            gemini_client,
                        )

                    if answer:
                        st.markdown(answer)
                        st.caption(f"✨ AI fallback: {source}")

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "source": f"✨ AI fallback: {source}",
                            }
                        )

                        if save_mode:
                            st.session_state.pending_faq = {
                                "question": user_query,
                                "answer": answer,
                                "source": source,
                            }

                            st.info(
                                "This answer is new. Use **Add to FAQ** above "
                                "if you want to save it permanently."
                            )

                    else:
                        st.error("The connected AI provider could not answer.")
                        if ai_error:
                            st.code(ai_error)

            else:
                with st.chat_message("assistant"):
                    message = (
                        "I couldn't find a close FAQ match, and ChatGPT/Gemini "
                        "is not connected. Add an API key in `.env` or "
                        "Streamlit Secrets to enable AI fallback."
                    )

                    st.warning(message)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": message,
                            "source": "AI provider not connected",
                        }
                    )

# ============================================================
# KNOWLEDGE BASE VIEWER
# ============================================================

st.divider()

with st.expander("👀 View All FAQ Questions & Answers"):
    st.dataframe(
        faq_df,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.caption(
    "AI & Data Science FAQ Assistant • TF-IDF FAQ Search + Gemini + ChatGPT"
)
