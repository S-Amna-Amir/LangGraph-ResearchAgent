import streamlit as st
from research_agent import ask_agent, ingest_file, reset_index, index_size

st.set_page_config(page_title="RAG Research Agent", page_icon="📄")

st.title("📄 RAG Research Agent")

# ── Sidebar: document management ─────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Upload Documents")
    st.caption("Supported: PDF, DOCX, MD, TXT")

    uploaded_files = st.file_uploader(
        "Choose files",
        type=["pdf", "docx", "md", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        # Track which filenames are already ingested so we don't re-process on rerun
        already_ingested = st.session_state.get("ingested_files", set())
        new_files = [f for f in uploaded_files if f.name not in already_ingested]

        if new_files:
            with st.spinner(f"Ingesting {len(new_files)} file(s)..."):
                for f in new_files:
                    try:
                        n_chunks = ingest_file(f.read(), f.name)
                        already_ingested.add(f.name)
                        st.success(f"✅ {f.name} — {n_chunks} chunks")
                    except Exception as e:
                        st.error(f"❌ {f.name}: {e}")
            st.session_state["ingested_files"] = already_ingested

    # Show index status
    n = index_size()
    if n > 0:
        st.info(f"🗂️ {n} chunks indexed from {len(st.session_state.get('ingested_files', set()))} file(s)")
    else:
        st.warning("No documents uploaded yet.")

    # Clear button
    if st.button("🗑️ Clear all documents"):
        reset_index()
        st.session_state["ingested_files"] = set()
        st.session_state["memory"] = []
        st.session_state["chat_history"] = []
        st.rerun()

# ── Session state init ────────────────────────────────────────────────────────

if "memory" not in st.session_state:
    st.session_state.memory = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ── Chat history display ──────────────────────────────────────────────────────

for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)

# ── Chat input ────────────────────────────────────────────────────────────────

user_input = st.chat_input("Ask a question about your documents...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, updated_memory, decision_log = ask_agent(
                user_input,
                st.session_state.memory,
            )

        st.session_state.memory = updated_memory

        with st.expander("🔍 Agent reasoning"):
            st.text(decision_log)

        st.markdown(answer)

    st.session_state.chat_history.append(("assistant", answer))