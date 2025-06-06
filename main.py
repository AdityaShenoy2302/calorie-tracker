import os
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from groq import Groq
from typing import List, Dict, Optional

from dotenv import load_dotenv
load_dotenv()

# Hardcoded API key (In production, use environment variables or secrets management)
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY)


class DocumentProcessor:
    @staticmethod
    def get_pdf_text(pdf_docs) -> str:
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    @staticmethod
    def get_csv_text(csv_file) -> str:
        try:
            df = pd.read_csv(csv_file)
            return df.to_markdown()
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return ""

    @staticmethod
    def get_text_chunks(text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=3000, chunk_overlap=500)
        return splitter.split_text(text)

    @staticmethod
    def get_vector_store(chunks: List[str]):
        embeddings = HuggingFaceEmbeddings()
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")


class KnowledgeBase:
    @staticmethod
    @staticmethod
    def search_context(user_question: str) -> str:
        try:
            embeddings = HuggingFaceEmbeddings()
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question, k=3)

            if not docs:
                return "No relevant information found in the uploaded files."

            return "\n".join([doc.page_content for doc in docs])

        except Exception as e:
            return f"Error retrieving context: {str(e)}"


class AgentManager:
    AGENT_PROFILES = {
        "nutrition_expert": {
            "system_prompt": """You are a certified nutritionist. Answer questions STRICTLY based on the provided 
            context from uploaded documents. If the answer isn't in the context, say "This information is not available 
            in my knowledge base." Never speculate or use outside knowledge for nutrition questions."""
        },
        "data_analyst": {
            "system_prompt": """You are a data analysis expert. Analyze and interpret nutritional data, 
            spot trends in calorie intake, and provide insights from datasets. Use tables and charts when helpful."""
        },
        "general_assistant": {
            "system_prompt": """You are a helpful assistant. Answer general questions politely and concisely. 
            For nutrition-specific questions without context available, suggest consulting the nutrition expert 
            after uploading relevant documents."""
        }
    }

    @staticmethod
    @staticmethod
    def route_question(question: str, has_knowledge_base: bool = False) -> str:
        """Determine which agent should handle the question"""
        question_lower = question.lower()

        # General greetings
        general_keywords = ['weather', 'time', 'date', 'hello', 'hi', 'hey']
        if any(keyword in question_lower for keyword in general_keywords):
            return "general_assistant"

        # Nutrition-based keywords
        nutrition_keywords = ['calorie', 'nutrition', 'diet', 'food', 'eat', 'meal', 'protein', 'carb', 'fat']
        data_keywords = ['analyze', 'trend', 'statistic', 'graph', 'chart', 'data', 'pattern']

        if any(keyword in question_lower for keyword in nutrition_keywords):
            return "nutrition_expert" if has_knowledge_base else "general_assistant"

        elif any(keyword in question_lower for keyword in data_keywords):
            return "data_analyst"

        return "general_assistant"

    @staticmethod
    @staticmethod
    def query_agent(agent: str, question: str, context: str = "") -> str:
        """Query the appropriate agent with the question"""
        profile = AgentManager.AGENT_PROFILES.get(agent, AgentManager.AGENT_PROFILES["general_assistant"])

        # Ensure the nutrition expert only responds if context is available
        if agent == "nutrition_expert" and not context.strip():
            return "This information is not available in my knowledge base. Please upload relevant documents."

        messages = [
            {"role": "system", "content": profile["system_prompt"]},
        ]

        if context:
            messages.append({
                "role": "system",
                "content": f"Relevant context from knowledge base:\n{context}\n\nUse this information if relevant."
            })

        messages.append({"role": "user", "content": question})

        try:
            response = client.chat.completions.create(
                messages=messages,
                model="llama-3.3-70b-versatile",
                max_tokens=1000,
                temperature=0.7 if agent == "general_assistant" else 0.3  # More creative for general questions
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying agent: {str(e)}"

class SessionState:
    @staticmethod
    def init_session():
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {"role": "assistant",
                 "content": "Welcome to CalorieGroq! Upload nutrition documents or ask me anything about food and health."}
            ]
        if "processed_files" not in st.session_state:
            st.session_state["processed_files"] = False

    @staticmethod
    def add_message(role: str, content: str):
        st.session_state["messages"].append({"role": role, "content": content})


def main():
    st.set_page_config(page_title="CalorieTracking AI", page_icon="🍏")
    SessionState.init_session()

    # Sidebar for file upload and processing
    with st.sidebar:
        st.title("Knowledge Setup")
        uploaded_files = st.file_uploader(
            "Upload nutrition PDFs or CSV data files",
            accept_multiple_files=True,
            type=["pdf", "csv"]
        )

        if st.button("Process Files"):
            if not uploaded_files:
                st.warning("Please upload files first")
            else:
                with st.spinner("Processing files..."):
                    all_text = ""
                    for file in uploaded_files:
                        if file.name.endswith(".pdf"):
                            all_text += DocumentProcessor.get_pdf_text([file])
                        elif file.name.endswith(".csv"):
                            all_text += DocumentProcessor.get_csv_text(file)

                    if all_text.strip():
                        text_chunks = DocumentProcessor.get_text_chunks(all_text)
                        DocumentProcessor.get_vector_store(text_chunks)
                        st.session_state["processed_files"] = True
                        st.success("Files processed successfully!")
                    else:
                        st.error("No readable text found in the files")

    # Main chat interface
    st.title("🍏 CalorieTracking AI Assistant")
    st.caption("Ask about nutrition, analyze food data, or get health advice")

    # Display chat messages
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about nutrition or upload files to analyze..."):
        SessionState.add_message("user", prompt)

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Ensure nutrition expert only answers if knowledge base is available
                has_knowledge_base = st.session_state.get("processed_files", False)
                context = KnowledgeBase.search_context(prompt) if has_knowledge_base else ""

                # Route question based on knowledge base availability
                agent_type = AgentManager.route_question(prompt, has_knowledge_base)

                response = AgentManager.query_agent(agent_type, prompt, context)

                st.write(response)
                st.caption(f"*Answered by {agent_type.replace('_', ' ')}*")

                SessionState.add_message("assistant", response)


if __name__ == "__main__":
    main()




















