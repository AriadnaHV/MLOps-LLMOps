import streamlit as st


import requests

def gemini_api_call(messages):
    # Get the last user message from the messages list
    user_message = [msg["content"] for msg in messages if msg["role"] == "user"][-1]
    
    # URL encode the query parameter
    import urllib.parse
    encoded_query = urllib.parse.quote(user_message)
    
    # URL of the FastAPI endpoint
    #url = f"https://test-fastapi-464252643117.europe-west4.run.app/gemini"
    url = "https://bd16-gemini-api-git-685419628669.europe-west4.run.app/gemini-v2"
    
    # Parameters for the GET request
    params = {
        "query": user_message
    }

    try:
        # Send GET request with query parameters
        response = requests.get(url, params=params)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: {response.status_code}"
            
    except requests.exceptions.RequestException as e:
        return f"Error making request: {str(e)}"


st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    print("user message", st.session_state.messages)
    response = gemini_api_call(st.session_state.messages)
    msg = response
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)