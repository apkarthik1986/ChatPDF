import os
import tempfile
import streamlit as st
from streamlit_chat import message
from pdfquery import PDFQuery
import re

st.set_page_config(page_title="ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, msg in enumerate(st.session_state["messages"]):
        if isinstance(msg, dict):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            prefix = "You:" if role == "user" else "Assistant:"
            st.markdown(f"**{prefix}**\n{content}", unsafe_allow_html=False)
        elif isinstance(msg, tuple) and len(msg) == 2:
            # Legacy format: (text, is_user)
            prefix = "You:" if msg[1] else "Assistant:"
            st.markdown(f"**{prefix}**\n{msg[0]}", unsafe_allow_html=False)

    # placeholder for the thinking spinner area (used elsewhere)
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    cleaned_reply = ""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()

        # show the user's message immediately in the chat (dict format)
        st.session_state["messages"].append({"role": "user", "content": user_text})

        # determine whether to include history in the prompt
        include_history = st.session_state.get("include_history", True)
        history_for_prompt = st.session_state["messages"] if include_history else []

        pdfquery = st.session_state.get("pdfquery")
        if pdfquery is None:
            # Inform the user that the PDFQuery helper isn't ready (no API key or not initialized)
            cleaned_reply = (
                "The assistant is not available. Make sure you set a valid Google API key "
                "and upload at least one PDF."
            )
        else:
            with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
                try:
                    cleaned_reply = pdfquery.ask(user_text, chat_history=history_for_prompt)
                except Exception as e:
                    cleaned_reply = f"An error occurred while contacting the model: {e}"
                    print(f"Error in process_input calling ask: {e}")
            # Format reply for markdown rendering
            cleaned_reply = str(cleaned_reply).replace("\r\n", "\n").replace("\r", "\n")

    # Append assistant reply as dict
    st.session_state["messages"].append({"role": "assistant", "content": cleaned_reply})
    # Force Streamlit rerun to update UI
    st.session_state["last_message_id"] = len(st.session_state["messages"])

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["pdfquery"].ingest(file_path)
        os.remove(file_path)


def is_google_api_key_set() -> bool:
    return len(st.session_state["GOOGLE_API_KEY"]) > 0


def main():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "")
        if is_google_api_key_set():
            st.session_state["pdfquery"] = PDFQuery(st.session_state["GOOGLE_API_KEY"])
        else:
            st.session_state["pdfquery"] = None

    # initialize user input and clear flag if missing
    if "user_input" not in st.session_state:
        st.session_state["user_input"] = ""
    if "clear_user_input" not in st.session_state:
        st.session_state["clear_user_input"] = False

    st.header("ChatPDF")

    if st.text_input("Google API Key", value=st.session_state["GOOGLE_API_KEY"], key="input_GOOGLE_API_KEY", type="password"):
        if (
            len(st.session_state["input_GOOGLE_API_KEY"]) > 0
            and st.session_state["input_GOOGLE_API_KEY"] != st.session_state["GOOGLE_API_KEY"]
        ):
            st.session_state["GOOGLE_API_KEY"] = st.session_state["input_GOOGLE_API_KEY"]
            if st.session_state["pdfquery"] is not None:
                st.warning("Please, upload the files again.")
            st.session_state["messages"] = []
            st.session_state["clear_user_input"] = True
            st.session_state["pdfquery"] = PDFQuery(st.session_state["GOOGLE_API_KEY"])

    st.subheader("Upload a document")
    # Option to allow multiple PDF uploads
    if "allow_multiple" not in st.session_state:
        st.session_state["allow_multiple"] = True
    st.session_state["allow_multiple"] = st.checkbox(
        "Allow multiple PDF uploads", value=st.session_state["allow_multiple"]
    )
    uploaded_files = st.file_uploader(
        "Upload document",
        type=["pdf"],
        key="file_uploader",
        label_visibility="collapsed",
        accept_multiple_files=st.session_state["allow_multiple"],
        disabled=not is_google_api_key_set(),
    )

    st.session_state["ingestion_spinner"] = st.empty()

    # Ingest PDFs immediately after upload
    if uploaded_files:
        st.session_state["pdfquery"] = PDFQuery(st.session_state["GOOGLE_API_KEY"])
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name
            with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
                st.session_state["pdfquery"].ingest(file_path)
            os.remove(file_path)

    # Clear input before widget is created
    if st.session_state.get("clear_user_input"):
        st.session_state["user_input"] = ""
        st.session_state["clear_user_input"] = False
    with st.form(key="input_form"):
        col1, col2 = st.columns([0.85, 0.15])
        with col1:
            st.text_input(
                "Message",
                key="user_input",
                disabled=not is_google_api_key_set(),
            )
        with col2:
            submit = st.form_submit_button("Send", disabled=not is_google_api_key_set())
        if submit:
            process_input()
            st.session_state["clear_user_input"] = True

    display_messages()

    # (debug panel removed) keep UI clean

    st.divider()
    st.markdown("Source code: [Github](https://github.com/Anil-matcha/ChatPDF)")


if __name__ == "__main__":
    main()
