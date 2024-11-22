import streamlit as st
import json
import os
from dataclasses import dataclass
from typing import Literal
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
import streamlit.components.v1 as components

# Initialize the LLM
llm = Ollama(model="llama3.2")

# Load the Coimbatore-related PDF for personalized recommendations
loader = PyPDFLoader("Transcript.pdf")
docs = loader.load()

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
documents = text_splitter.split_documents(docs)

# Create embeddings for the documents and store them in a vector database
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma.from_documents(documents, embeddings)

# Function to determine whether a query is related to Coimbatore
def is_coimbatore_query(query):
    return "meeting" in query.lower()

# File to save chat history
HISTORY_FILE = "chat_history.json"

@dataclass
class Message:
    origin: Literal["human", "ai"]
    message: str

# Load chat history
def load_chat_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []

# Save chat history (converting non-serializable objects like Document to strings)
def save_chat_history(history):
    history_to_save = []
    for msg in history:
        if isinstance(msg['message'], dict):  # Assuming Documents are dict-like
            msg['message'] = str(msg['message'])  # Convert to string
        history_to_save.append(msg)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history_to_save, f)

# Clear chat history
def clear_chat_history():
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

# Initialize session state
def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = load_chat_history()

    if "conversation_chain" not in st.session_state:
        # Coimbatore-specific prompt
        coimbatore_prompt = """You are an intelligent meeting assistant called RUBY for helping employees and clarify their queries regarding the virtual meet.
        Use the following pieces of retrieved context to answer the question in detail:{context}.\
        Greet if the user greets you. \
        If you don't know the answer, just say that you don't know 
        Only answer relevant content and Not anything extra.\
        Dont return the prompt in the answer. \
        Don't respond irrelevant or anything outside the context. \
        ___________
        {context}"""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", coimbatore_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        retriever = db.as_retriever()  # Ensure retriever is initialized here
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        st.session_state.retrieval_chain = rag_chain  # Store the correct chain

        # General prompt
        general_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a highly knowledgeable virtual meeting assistant named RUBY. You help users summarize the meeting,provide semantic analysis,clear doubts with the meeting ,answer who said what,classify opening statements ,closing statements,takeaway point. Always ask follow-up questions to refine the meeting details."),
            ("user", "Question: {input}")
        ])
        st.session_state.general_chain = general_prompt | llm | StrOutputParser()

# Handle user input and update the conversation chain
def on_click_callback():
    """Handle user input and update chat."""
    user_input = st.session_state.user_input
    print(f"User query: {user_input}")
    print(f"Is Coimbatore query: {is_coimbatore_query(user_input)}")

    # Invoke the chain with conversation history
    context = "\n".join([f"{msg['origin']}: {msg['message']}" for msg in st.session_state.history])
    if is_coimbatore_query(user_input):
        response = st.session_state.retrieval_chain.invoke({"input": context + "\n" + user_input})
        answer = response['answer']
    else:
        response = st.session_state.general_chain.invoke({"input": context + "\n" + user_input})
        answer = response

    # Store the conversation history in session state
    st.session_state.history.append({"origin": "human", "message": user_input})
    st.session_state.history.append({"origin": "ai", "message": answer})

    # Save the updated history to a file
    save_chat_history(st.session_state.history)

# Add a button to clear the chat and start a new conversation
if st.sidebar.button("New Chat"):
    st.session_state.history = []  # Clear session state history
    clear_chat_history()  # Clear the stored history

# Initialize session state (load chat history if available)
initialize_session_state()

# Frontend: Display the chat interface
def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

load_css()
#logo_path=
 # Adjust as needed
#st.image(logo_path, width=200, caption="RUBY, THE INTELLIGENT MEETING ASSISTANT")
st.title("RUBY, INTELLIGENT MEETING BOT 🤖")

# Chat placeholders and input form
chat_placeholder = st.container()
prompt_placeholder = st.form("chat-form")

# Display chat history
with chat_placeholder:
    for chat in st.session_state.history:
        div = f"""
        <div class="chat-row {'row-reverse' if chat['origin'] == 'human' else ''}">
            <div class="chat-bubble {'human-bubble' if chat['origin'] == 'human' else 'ai-bubble'}">
                {chat['message']}
            </div>
        </div>
        """
        st.markdown(div, unsafe_allow_html=True)

# Input form for user queries
with prompt_placeholder:
    st.markdown("*Ask RUBY about your meeting !*")
    cols = st.columns((6, 1))
    cols[0].text_input("Chat", key="user_input", label_visibility="collapsed")
    cols[1].form_submit_button("Submit", on_click=on_click_callback)

components.html("""
<script>
const streamlitDoc = window.parent.document;
const buttons = Array.from(streamlitDoc.querySelectorAll('.stButton > button'));
const submitButton = buttons.find(el => el.innerText === 'Submit');
streamlitDoc.addEventListener('keydown', function(e) {
    if (e.key === 'Enter') {
        submitButton.click();
    }
});
</script>
""", height=0, width=0)