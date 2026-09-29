#!/usr/bin/env python3
import streamlit as st
from requests import RequestException

from agent_core import run_prompt

# Page Configuration
st.set_page_config(
    page_title="Sarcastic British ReAct Agent", page_icon="🇬🇧", layout="centered"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .stChatMessage {
        border-radius: 15px;
    }
    .main {
        background-color: #f5f7f9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🇬🇧 Sarcastic British Agent")
st.caption("Powered by LangGraph, Ollama, and a heavy dose of sarcasm.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me something (if you must)..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Agent Response
    with (
        st.chat_message("assistant"),
        st.spinner("Thinking... (and judging your question)"),
    ):
        try:
            # Call the core logic from agent_core.py
            response = run_prompt(prompt)
            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except (RuntimeError, RequestException, ValueError) as e:
            error_msg = f"Oops, something went wrong: {e}"
            st.error(error_msg)
            st.session_state.messages.append(
                {"role": "assistant", "content": error_msg}
            )
