
import os
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Core imports - these are stable
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# These are the correct modern imports (from langchain 0.3+ / 1.x)
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

from pinecone import Pinecone

# Logger fallback
try:
    from logger import log_info, log_error, log_success
except ImportError:
    def log_info(msg, color=None): print(f"[INFO] {msg}")
    def log_error(msg, color=None): print(f"[ERROR] {msg}")
    def log_success(msg, color=None): print(f"[SUCCESS] {msg}")

load_dotenv()

# --------------------- Prompts ---------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, "
               "reformulate the question to be a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an elite, motivational MMA coach that specializes in Boxing, Judo, BJJ, and Krav Maga. 
    Your goal is to help athletes, fighters, parents, and enthusiasts improve their skills, fitness, and mindset.
    Use the provided context to give **practical, actionable, and technique-focused** advice.
    Be encouraging, clear, and specific (e.g., mention stance, footwork, breathing, common mistakes, progressions).
    If the context does not contain enough information, honestly say so and offer a related training tip instead.

    Context: {context}"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# --------------------- Components ---------------------
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def get_vector_store():
    embeddings = get_embeddings()
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = os.getenv("PINECONE_INDEX_NAME")
    return PineconeVectorStore(index=pc.Index(index_name), embedding=embeddings)

def get_llm(temperature: float = 0.7):
    return ChatXAI(
        model="grok-4",
        temperature=temperature,
        xai_api_key=os.getenv("XAI_API_KEY"),
    )

# --------------------- Main run_llm ---------------------
def run_llm(
    query: str, 
    chat_history: Optional[List] = None, 
    temperature: float = 0.7, 
    k: int = 6
) -> Dict[str, Any]:
    try:
        log_info(f"Processing query: {query[:80]}...", "purple")
        
        vector_store = get_vector_store()
        llm = get_llm(temperature)
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # Convert chat history
        history = []
        if chat_history:
            for human, ai in chat_history:
                history.extend([HumanMessage(content=human), AIMessage(content=ai)])
        
        # History-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm=llm, 
            retriever=retriever, 
            prompt=contextualize_q_prompt
        )
        
        # QA chain
        question_answer_chain = qa_prompt | llm | StrOutputParser()
        
        # Full RAG chain
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
        
        result = rag_chain.invoke({
            "input": query,
            "chat_history": history
        })
        
        answer = result.get("answer", "Sorry, I couldn't generate a response.")
        source_docs = result.get("context", [])

        sources: list[str] = []
        all_images: list[str] = []

        for doc in source_docs: 
            source = doc.metadata.get("source", "Unknown Source")
            sources.append(source)

            # extract images that we scraped with BeautifulSoup and stored in metadata
            images : list[str] = doc.metadata.get("images", [])
            for img in images: 
                if img and img.startswith("http"): # basic validation to ensure it's a URL
                    all_images.append(img) # limit to 4 images to avoid overwhelming the user

        # Removing duplicate image URLs while preserving order
        unique_images: list[str] = list(Dict.fromkeys(all_images))
        # ask grok when we use lower case list and dict vs when we use List and Dict from typing - does it affect the output in any way?
        
        log_success(f"Response generated with {len(sources)} sources and {len(unique_images)} images.")
        
        return {
            "answer": answer,
            "sources": sources,
            "images": unique_images,
            "source_documents": source_docs
        }
        
    except Exception as e:
        log_error(f"Error in run_llm: {str(e)}")
        return {
            "answer": "Sorry coach, technical issue. Please check API keys and Pinecone connection.",
            "sources": [],
            "images": [],
            "error": str(e)
        }


# Quick test
def test_run_llm():
    print("\n🧪 Testing MMA Coach...\n")
    result = run_llm("Describe a typical boxing bootcamp workout with a personal trainer.")
    print("ANSWER:\n", result["answer"])
    print("\nSOURCES:")
    for i, s in enumerate(result["sources"], 1):
        print(f"{i}. {s}")

if __name__ == "__main__":
    test_run_llm()











