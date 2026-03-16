"""
rag_chain.py — LangChain LCEL-based RAG chain.

Chain structure:
  {question, context}
    → ChatPromptTemplate
    → OllamaLLM
    → StrOutputParser

The chain also captures source documents for citation display.

Usage:
    from rag_chain import build_rag_chain
    chain = build_rag_chain()
    result = chain.invoke("What are the main risks described in the reports?")
    print(result["answer"])
    print(result["source_documents"])
"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

from app.config import settings
from app.retriever import get_retriever
from app.utils import setup_logging

logger = setup_logging()


# ── Prompt template ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful, precise research assistant.

Answer the user's question **using only the context below**.  
If the context does not contain enough information, say:
  "I could not find a clear answer in the provided documents."

For every factual claim, cite the source document name and page number
using the format: [source.pdf, page N].

Context:
{context}
"""

HUMAN_PROMPT = "{question}"

CONDENSE_QUESTION_PROMPT = """\
Given the following conversation history and a follow-up question, \
rephrase the follow-up question to be a standalone question.
If the follow-up question is already standalone, return it as is.

Chat History:
{chat_history}
Follow-up Question: {question}
Standalone question:"""


def _format_docs(docs: List[Document]) -> str:
    """Flatten retrieved chunks into a single context string."""
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        
        # Convert 0-indexed page to 1-indexed for the LLM
        raw_page = doc.metadata.get("page", 0)
        try:
            page = int(raw_page) + 1
        except (ValueError, TypeError):
            page = raw_page
            
        parts.append(f"[{source}, page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ── Chain builder ──────────────────────────────────────────────────────────

def build_rag_chain(
    ollama_model: Optional[str] = None,
    k: Optional[int] = None,
):
    """
    Build and return the full RAG chain.

    Returns a callable that accepts {"question": str} and returns:
        {
            "answer":           str,
            "source_documents": List[Document],
        }
    """
    model_name = ollama_model or settings.OLLAMA_MODEL
    # Determine model to display in logs
    active_model = ollama_model or (settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.OLLAMA_MODEL)
    logger.info("Building RAG chain  (Provider: %s, Model: %s, with Memory)", settings.LLM_PROVIDER, active_model)

    # 1. Retriever
    retriever = get_retriever(k=k)

    # 2. LLM Provider
    # If the user explicitly asks for gemini, or the config defaults to gemini
    use_gemini = settings.LLM_PROVIDER.lower() == "gemini" or (model_name and "gemini" in model_name.lower())
    if use_gemini:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            logger.error("langchain-google-genai is not installed. Run: pip install langchain-google-genai")
            raise SystemExit(1)
            
        api_key = settings.GEMINI_API_KEY
        if not api_key:
             import os
             api_key = os.environ.get("GEMINI_API_KEY")
             
        if not api_key:
            logger.error("GEMINI_API_KEY is not set.")
            raise SystemExit(1)
            
        model = model_name if model_name and "gemini" in model_name else settings.GEMINI_MODEL
        # LangChain expectation for model name
        if model.startswith("models/"):
            model = model.replace("models/", "")
            
        logger.info("Using LLM: Google Gemini (%s) via LangChain", model)
        llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=settings.GEMINI_TEMPERATURE
        )

        # 1. Chain for question contextualization
        condense_prompt = ChatPromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)
        condense_chain = condense_prompt | llm | StrOutputParser()

        # 2. Main answer chain
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        llm_runnable = answer_prompt | llm | StrOutputParser()
    else:
        from langchain_ollama import OllamaLLM
        model = model_name if model_name else settings.OLLAMA_MODEL
        logger.info("Using LLM: Ollama (%s)", model)
        llm = OllamaLLM(
            model=model,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=settings.OLLAMA_TEMPERATURE,
        )
        # 1. Chain for question contextualization
        condense_prompt = ChatPromptTemplate.from_template(CONDENSE_QUESTION_PROMPT)
        condense_chain = condense_prompt | llm | StrOutputParser()

        # 2. Main answer chain
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ])
        llm_runnable = answer_prompt | llm | StrOutputParser()

    # 4. Sub-chain: retrieve docs and keep them for citation
    retrieve_and_pass = RunnableParallel(
        {
            "source_documents": retriever,            # retrieved docs list
            "question": RunnablePassthrough(),         # original question
        }
    )

    # 5. Assembly
    def _run(inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]
        chat_history = inputs.get("chat_history", "")
        
        # Step A: Contextualize the question if there's history
        if chat_history:
            logger.info("Contextualizing follow-up question...")
            standalone_question = condense_chain.invoke(
                {"chat_history": chat_history, "question": question}
            )
            logger.debug("Standalone question: %s", standalone_question)
        else:
            standalone_question = question
            
        # Step B: Retrieve based on the standalone question
        retrieved = retrieve_and_pass.invoke(standalone_question)
        context = _format_docs(retrieved["source_documents"])
        
        answer = llm_runnable.invoke(
            {"question": standalone_question, "context": context}
        )
        
        return {
            "answer": answer,
            "source_documents": retrieved["source_documents"],
        }

    return RunnableLambda(_run)
