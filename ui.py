
import streamlit as st

import browse
from rag import AnimeOracle

st.set_page_config(page_title="Anime AI Assistant", page_icon="🍥", layout="wide")

# --- Custom CSS ---------------------------------------------------------------
st.markdown("""
<style>

/* PINK VERTICAL LINE */
[data-testid="stSidebar"] {
    border-right: 2px solid #ff1493 !important;
}


/* CLEAR CONVERSATION BUTTON */
[data-testid="stSidebar"] button:last-of-type {
    background-color: #ff1493 !important;
    color: white !important;
    border: 1px solid #ff1493 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stSidebar"] button:last-of-type:hover {
    background-color: #e6007e !important;
    border-color: #e6007e !important;
    color: white !important;
}


/* CHAT SEND BUTTON */
[data-testid="stChatInput"] button {
    background-color: #ff1493 !important;
    color: white !important;
    border: 1px solid #ff1493 !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

[data-testid="stChatInput"] button:hover {
    background-color: #e6007e !important;
    border-color: #e6007e !important;
}

[data-testid="stChatInput"] button svg {
    color: white !important;
    fill: white !important;
}

</style>
""", unsafe_allow_html=True)
# --- Cached resources -------------------------------------------------------
# st.cache_resource keeps ONE instance alive across reruns. Streamlit reruns
# the whole script on every interaction, so without this, the vector store
# and OpenAI clients would reload every single time you send a message --
# slow, wasteful, and unnecessary load on a 4GB machine.
@st.cache_resource
def load_oracle() -> AnimeOracle:
    return AnimeOracle()


@st.cache_data
def load_summary() -> dict:
    return browse.dataset_summary()


oracle = load_oracle()
summary = load_summary()


# --- Sidebar: browse the dataset directly (no LLM, no retrieval) -----------
with st.sidebar:
    st.header("Explore Animes")
    st.caption(f"{summary['total_anime']} anime · {summary['year_range']}")

    with st.expander("📅 Browse by year", expanded=False):
        years = summary["year_range"].split("-")
        min_year, max_year = int(years[0]), int(years[1])
        year_range = st.slider(
            "Year range", min_year, max_year, (min_year, max_year), key="year_slider"
        )
        year_titles = browse.list_titles_by_year(year_range[0], year_range[1])
        st.write(f"{len(year_titles)} anime found")
        st.dataframe(year_titles, use_container_width=True, hide_index=True)

    with st.expander("🎭 Browse by genre", expanded=False):
        genre = st.selectbox("Genre", summary["available_genres"], key="genre_select")
        genre_titles = browse.list_titles_by_genre(genre)
        st.write(f"{len(genre_titles)} anime found")
        st.dataframe(genre_titles, use_container_width=True, hide_index=True)

    st.divider()

    st.header("Session")
    session_id = st.text_input(
        "Your session ID",
        value=st.session_state.get("session_id", "guest"),
        help="Use the same ID next time you visit to pick up where you left off.",
    )
    st.session_state["session_id"] = session_id

    if st.button("Clear this conversation"):
        oracle.memory.clear_session(session_id)
        st.session_state.pop("messages", None)
        st.rerun()


# --- Main chat area ----------------------------------------------------------
st.title("🍥 A³")
st.caption("I am an AI assistant, ask me anything about anime from 2015-2025. I'll cite my sources.")

# Load this session's history from SQLite into st.session_state ONCE per
# session_id (not on every rerun) so returning users see prior messages,
# and so we're not re-hitting the DB unnecessarily on every keystroke.
if "messages" not in st.session_state or st.session_state.get("loaded_session") != session_id:
    history = oracle.memory.get_recent_history(session_id, max_messages=10, max_age_days=7)
    st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in history]
    st.session_state.loaded_session = session_id

# Render existing conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        avatar = "🔮"
    else:
        avatar = "🍥"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Handle new input
if question := st.chat_input("Ask about an anime..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🔮"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🍥"):
        with st.spinner("Searching the archives..."):
            result = oracle.ask(question, session_id=session_id)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption("Sources: " + ", ".join(result["sources"]))

    # oracle.ask() already saved this turn to SQLite -- this just updates
    # what's shown on screen for the rest of the session
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})