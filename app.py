import streamlit as st
from research_agent import ask_agent

st.set_page_config(
    page_title="AI Research Agent",
    page_icon="🔎"
)

st.title("AI Research Agent")

if "memory" not in st.session_state:
    st.session_state.memory = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display history
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(message)

# Chat input
user_input = st.chat_input("Ask a question...")

if user_input:

    st.session_state.chat_history.append(("user", user_input))

    with st.chat_message("user"):
        st.write(user_input)

    answer, updated_memory = ask_agent(
        user_input,
        st.session_state.memory
    )

    st.session_state.memory = updated_memory

    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.chat_history.append(("assistant", answer))