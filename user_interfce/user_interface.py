import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/chat"

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "summary" not in st.session_state:
    st.session_state["summary"] = "The user asked you about Ebla and you answered about its history"

st.title("🗨️ AI Chatbot")

session_id = st.sidebar.text_input("Session ID", value="1")
st.sidebar.text_area("Chat Summary", value=st.session_state["summary"], key="summary_display", height=150)

for role, text in st.session_state["messages"]:
    with st.chat_message(role):
        st.write(text)

if user_input := st.chat_input("Type your message..."):
    st.session_state["messages"].append(("user", user_input))
    
    with st.chat_message("user"):
        st.write(user_input)

    payload = {"session_id": session_id, "message": user_input, "summary": st.session_state["summary"]}

    with st.spinner("Thinking..."):
        response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        response_data = response.json()
        bot_reply = response_data["message"]
        st.session_state["summary"] = response_data["summary"]  
        st.session_state["messages"].append(("assistant", bot_reply))
        with st.chat_message("assistant"):
            st.write(bot_reply)
    else:
        st.error("Error: " + response.text)