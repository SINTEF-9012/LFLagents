import os
import streamlit as st
from streamlit_chat import message
from streamlit_extras.stylable_container import stylable_container
from agents.retriever_agent import RetrieverAgent, Neo4jCypherTool, Neo4jVectorTool

# Set up the page configuration
st.set_page_config(page_title="SINTED Internship Project", layout="wide")
retrieverAgent = RetrieverAgent()

# Container for chat history
response_container = st.container()
# Container for text box
textcontainer = st.container()

def main():
    with st.sidebar:
        st.write("###")
        if st.button("New Chat"):
            st.session_state['requests'] = []
            st.session_state['responses'] = []
            st.session_state['chat_history'] = []

        new_folder = st.text_input("New Folder", placeholder="Create a new folder")
        if new_folder:
            st.session_state["folder_name"] = new_folder

        st.write("###")
        st.markdown("<hr>", unsafe_allow_html=True)
    
    with textcontainer:
            
        query = st.text_input("Type your question here...", key="text")
        if st.button("Send"):
            with st.spinner("typing..."):
                # Compute vector and Cypher results
                vector_results = retrieverAgent.vector_tool._run(query)
                cypher_results = retrieverAgent.cypher_tool._run(query)
                
                # Generate response using RetrieverAgent
                response_text = retrieverAgent.conversation.predict(input=f"Vector Result:\n {vector_results}  \n\n Cypher Result:\n {cypher_results}  \n\n Query:\n{query}")

                st.session_state.setdefault('requests', []).append(query)
                st.session_state.setdefault('responses', []).append(response_text)
                st.session_state.setdefault('chat_history', []).append((query, response_text))
                folder_name = st.session_state.get("folder_name", "chat_history")
                retrieverAgent.save_chat_history(st.session_state['chat_history'], folder_name, "chat_history.txt")
                

    with response_container:
        if st.session_state.get('responses'):
            for i, response in enumerate(st.session_state['responses']):
                message(response, key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')

if __name__ == "__main__":
    main()