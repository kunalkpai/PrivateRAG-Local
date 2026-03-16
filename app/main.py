"""
main.py — Interactive CLI for the Multi-Document RAG system.

Usage:
    python main.py              # interactive Q&A (DB must already be built)
    python main.py --ingest     # ingest PDFs first, then start Q&A
    python main.py --ingest-only  # ingest and exit (no Q&A)
    python main.py --model mistral  # override Ollama model
"""

import argparse
import sys
from typing import List, Optional

import warnings
# Silence Pydantic V1 deprecation warning from LangChain before it imports
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_core")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from app.utils import format_sources, print_banner, setup_logging


logger = setup_logging()


# ── Argument parsing ───────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Document RAG — Ask questions across your PDF library."
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion before starting the Q&A session.",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run ingestion and exit without starting Q&A.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the Ollama model (e.g. mistral, phi3, gemma2).",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of chunks to retrieve per query (default: from config).",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Run in silent mode, suppressing all internal INFO logs.",
    )
    return parser.parse_args()


# ── Ingestion step ─────────────────────────────────────────────────────────

def _do_ingest() -> None:
    from app.ingest import run_ingestion  # lazy import (heavy deps)
    added = run_ingestion()
    if added == 0:
        logger.info("No new chunks added — proceeding to Q&A.")


def _format_chat_history(history: List[tuple]) -> str:
    """Convert history list to a string block for the LLM."""
    return "\n".join([f"Human: {q}\nAssistant: {a}" for q, a in history])


# ── Q&A loop ───────────────────────────────────────────────────────────────

def _qa_loop(model: Optional[str], k: Optional[int]) -> None:
    from app.rag_chain import build_rag_chain  # lazy import (loads Chroma + LLM)

    print_banner()
    print("Type your question and press Enter. Type 'quit', 'exit', or 'bye' to stop.\n")

    try:
        chain = build_rag_chain(ollama_model=model, k=k)
    except SystemExit:
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to build RAG chain: %s", exc)
        sys.exit(1)

    chat_history: List[tuple] = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q", "bye"}:
            print("\nGoodbye! 👋")
            break

        print("\n⏳ Thinking…\n")
        try:
            history_str = _format_chat_history(chat_history)
            result = chain.invoke({"question": question, "chat_history": history_str})
            answer = result["answer"]
        except Exception as exc:
            logger.error("Chain error: %s", exc)
            print("  ⚠️  An error occurred. Make sure Ollama is running (`ollama serve`).\n")
            continue

        print(f"🤖 Answer:\n{answer}\n")
        
        # Store in history
        chat_history.append((question, answer))
        # Keep only last 5 turns to prevent context overflow
        if len(chat_history) > 5:
            chat_history.pop(0)

        sources = result.get("source_documents", [])
        not_found_msg = "I could not find a clear answer in the provided documents."
        if sources and not_found_msg not in answer:
            print("📚 Sources:")
            print(format_sources(sources))

        print("\n" + "─" * 60 + "\n")


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.silent:
        import logging
        logging.getLogger("rag").setLevel(logging.WARNING)

    if args.ingest or args.ingest_only:
        _do_ingest()

    if args.ingest_only:
        print("\n✅ Ingestion complete. Run `python main.py` to start chatting.")
        return

    _qa_loop(model=args.model, k=args.k)


if __name__ == "__main__":
    main()
