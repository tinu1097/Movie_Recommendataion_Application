# Import necessary libraries for embeddings, vector stores, chatbot UI, and conversational chains
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # For generating embeddings using Google's models
from langchain_community.vectorstores import FAISS  # Vector store for efficient similarity search
import streamlit as st  # Framework for building interactive web apps
from langchain.chains import ConversationalRetrievalChain  # Chain for managing conversational retrieval logic
from langchain.memory import ConversationBufferMemory  # Memory to store conversation history
from langchain_groq import ChatGroq  # LLM model from Groq for generating responses
from langchain.prompts import PromptTemplate  # Template for customizing prompts
from dotenv import load_dotenv  # Load environment variables from a .env file
load_dotenv()  # Load environment variables (e.g., API keys)

# Configure the Streamlit page with a title, favicon, and layout
st.set_page_config(
    page_title="CineQuery AI",  # Title of the web page
    page_icon="images.jpg",  # You can use an emoji as a favicon
    layout="centered"
)

# Initialize session state variables for storing chat messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Add a centered title and a horizontal line for better UI
st.markdown("<h1 style='text-align: center;'>CineQuery AI 🎬</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid black; margin-top: -10px;'>", unsafe_allow_html=True)
 
# Display existing chat messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load pre-trained embeddings and the vector store (FAISS index) for movie data
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = FAISS.load_local(
    "faiss_index", 
    embeddings, 
    allow_dangerous_deserialization=True
)

# Initialize the LLM (Large Language Model) for generating responses
# Using Groq's ChatGroq model with specified parameters
llm = ChatGroq(
    temperature=0.3, 
    model_name="llama-3.3-70b-versatile", 
    max_tokens=8000   # Maximum number of tokens in the response
)

# Create a conversational retrieval chain that integrates the LLM, retriever, and memory
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 10}),
    memory=st.session_state.chat_memory,
     
)

# Define a custom prompt template to guide the LLM in generating responses
template = """
You are a movie identification assistant. Use the following conversation history and dataset to answer:

### Conversation History:
{chat_history}

### Dataset Fields:
- name: Movie title
- year: Release year
- rating: IMDb rating (0-10)
- certificate: Content rating
- duration: Runtime (minutes)
- genre: Genre(s)
- votes: IMDb votes
- gross_income: Gross income
- directors_name: Director(s)
- stars_name: Main cast
- description: Plot summary

### Instructions:
1. Analyze the query and conversation history for context
2. Use the dataset to find matching movies
3. Return detailed metadata for results
4. If unknown, state: "Information not available"
5. Maintain context for follow-up questions

Note: Do not give any note in your final response. 
If the user asks an unrelated question to the movie dataset, please do not answer this question.
### Current Query:
{question}
"""

# Create a PromptTemplate object with input variables and the custom template
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=template
)

# Set the custom prompt template for the LLM chain
retrieval_chain.combine_docs_chain.llm_chain.prompt = prompt_template

# Handle user input and generate responses
if question := st.chat_input("Ask. Discover. Watch.:"):
    with st.chat_message("user"):
        st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
    
    # Generate a response using the conversational retrieval chain
    response = retrieval_chain({"question": question})
    
    # Extract and display the assistant's response
    with st.chat_message("assistant"):
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})