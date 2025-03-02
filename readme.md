# Syrian-Heritage-Conversatinal-RAG-Bot

## Overview

This project is an AI-powered conversational chatbot designed to answer queries about Syrian heritage. It leverages state-of-the-art natural language processing techniques, retrieval-augmented generation (RAG), and real-time web search. Built with modular architecture, it processes user queries intelligently, providing context-aware responses. The chatbot is integrated with FastAPI for backend services and Streamlit for a user interface.

## Project Structure

```
AI_AGENT_FINAL_VERSION/
│── API/
│   │── __init__.py
│   │── app.py
│   │── schemas.py
│── config/
│   │── __init__.py
│   │── constants.py
│   │── prompts.py
│   │── settings.py
│── core/
│   │── __init__.py
│   │── graph.py
│   │── routers.py
│── data_scripts/
│   │── dataset_generation/
│   │── vector_db_generation/
│   │── __init__.py
│── models/
│   │── __init__.py
│   │── schemas.py
│── notebooks/
│   │── train_embeddings_intent.ipynb
│── services/
│   │── __init__.py
│   │── generator.py
│   │── grader.py
│   │── intent_classifier.py
│   │── retriever.py
│   │── search.py
│   │── summarizer.py
│── user_interface/
│   │── __init__.py
│   │── user_interface.py
│── utils/
│   │── __init__.py
│   │── utils.py
│── .env

```

## Prerequisites

Before running the system, make sure to install all the required dependencies. You can use `pip` to install them:

1.  Install dependencies:
    
    ```
    pip install -r requirements.txt
    
    ```
    
2.  Create a `.env` file in the `AI_AGENT_FINAL_VERSION` directory with the following API keys:
    
    ```
    export GROQ_API_KEY="your_api_key"
    export TAVILY_API_KEY="your_api_key"
    
    ```
    

## Running the Application

### 1. Start the FastAPI Server

In one terminal window, navigate to the `AI_AGENT_FINAL_VERSION` directory and run the FastAPI server with the following command:

```bash
python -m uvicorn AI_Agent_final_version.API.app:app --host 0.0.0.0 --port 8000 --reload

```

This will start the FastAPI server, allowing backend requests to be handled.

### 2. Launch the User Interface

In another terminal window, navigate to the `AI_AGENT_FINAL_VERSION/user_interface` directory and run the Streamlit user interface with:

```bash
python3 -m streamlit run AI_Agent_final_version/user_interface/user_interface.py

```

This will launch the user interface where users can interact with the chatbot.

## System Architecture

The system is built around a directed acyclic graph managed by LangGraph. It consists of the following main components:

-   **Intent Classifier**: Determines the nature of the user query (whether it needs to be answered directly or requires further retrieval).
-   **Retriever (RAG)**: Retrieves relevant documents from a pre-built FAISS vector database using embeddings.
-   **Grader**: Evaluates the relevance of retrieved documents to the user's query.
-   **Web Search**: Gathers real-time information via the Tavily and DuckDuckGo APIs.
-   **Generator**: Forms the final response based on retrieved documents, web results, and conversation history.
-   **Summarizer**: Updates the conversation memory to maintain context between interactions.

## How It Works

1.  **User Input**: The user submits a query to the chatbot through the Streamlit interface.
2.  **Intent Classification**: The query is processed by the intent classifier, which determines if the query requires a retrieval-augmented generation process.
3.  **Document Retrieval**: If necessary, the retriever fetches relevant documents from the vector database.
4.  **Grading**: The grader evaluates the relevance of the retrieved documents, and if they are insufficient, it triggers a web search.
5.  **Web Search**: External search results are fetched in real-time to enhance the response.
6.  **Response Generation**: The final response is generated by combining the query, relevant documents, and any real-time web data.
7.  **Summarization**: The conversation summary is updated to maintain context for future interactions.

## Environment Setup

1.  **.env File**: Create the `.env` file in the `AI_AGENT_FINAL_VERSION` directory with the required API keys:
    
    ```text
    export GROQ_API_KEY="your_api_key"
    export TAVILY_API_KEY="your_api_key"
    
    ```
    
2.  **Dependencies**: Install the dependencies listed in `requirements.txt` using pip.
    

## Requirements

-   Python 3.7+
-   FastAPI
-   Uvicorn
-   Streamlit
-   Hugging Face
-   FAISS
-   Tavily API (for real-time search)

## Conclusion

This project provides an AI-powered chatbot that educates and engages users with information about Syrian heritage. By utilizing modern NLP techniques, retrieval-augmented generation, and real-time web searches, the system offers accurate, context-aware, and dynamic responses.
