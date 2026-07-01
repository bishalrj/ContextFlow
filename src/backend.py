from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Tuple, Optional, Union, Generator
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import FAISS

# Import database module functions safely
try:
    from src.database import InMemoryDatabaseEngine
except ImportError:
    from database import InMemoryDatabaseEngine

# Ensure .env keys are cleanly parsed at startup
load_dotenv()

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


class OrchestrationBrain:
    """
    Orchestration Brain for multi-tenant, session-isolated RAG under Antigravity Production Patterns.
    Executes hierarchical context swap retrieval and strict grounded token streaming via Mistral Small.
    """

    def __init__(self) -> None:
        # Execution Profile: mistral-small-latest for high throughput & low latency
        self.llm = ChatMistralAI(
            model="mistral-small-latest",
            temperature=0.2,
        )

        # Strict Grounding Guardrails
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert domain assistant. Answer the user's question based strictly "
                "and exclusively on the provided context. If the answer cannot be deduced from "
                "the context, you must state 'I cannot find the answer in the provided official documentation.' "
                "Do not under any circumstances hallucinate or use outside knowledge."
            )),
            ("user", (
                "Context:\n{context}\n\n"
                "Question:\n{question}"
            ))
        ])

    def stream_question(
        self,
        user_query: str,
        vector_store: FAISS,
        parent_map: Dict[str, Dict[str, Any]],
        k: int = 3
    ) -> Tuple[Generator[str, None, None], List[Document]]:
        """
        Streaming query method that runs hierarchical context lookup and executes the RAG chain
        using the .stream() generator pattern, returning both the token generator and retrieved source records.
        """
        if not vector_store or not parent_map:
            raise RuntimeError("Cannot execute transaction: active vector store or parent map is missing.")

        print(f"[STREAMING] Executing dynamic context swap retrieval (k={k})...")
        context_string, retrieved_parents = InMemoryDatabaseEngine.retrieve_dynamic_context(
            query=user_query,
            vector_store=vector_store,
            parent_map=parent_map,
            k=k
        )

        print("[STREAMING] Invoking ChatMistralAI token stream...")
        chain = self.prompt_template | self.llm
        raw_stream = chain.stream({
            "context": context_string,
            "question": user_query
        })

        def string_stream_generator() -> Generator[str, None, None]:
            for chunk in raw_stream:
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if isinstance(content, str):
                    yield content
                else:
                    yield str(content)

        return string_stream_generator(), retrieved_parents

    def stream_dynamic_context(
        self,
        user_query: str,
        vector_store: FAISS,
        parent_map: Dict[str, Dict[str, Any]],
        k: int = 3
    ) -> Tuple[Generator[str, None, None], List[Document]]:
        """Alias for stream_question for backward compatibility."""
        return self.stream_question(user_query, vector_store, parent_map, k=k)

    def execute_transaction(
        self,
        user_question: str,
        vector_store: FAISS,
        parent_map: Dict[str, Dict[str, Any]],
        k: int = 3
    ) -> Dict[str, Any]:
        """
        Synchronous transaction wrapper that invokes Mistral model and delivers dual-signature payload.
        """
        if not vector_store or not parent_map:
            raise RuntimeError("Cannot execute transaction: active vector store or parent map is missing.")

        print(f"[TRANSACTION] Executing dynamic context swap retrieval (k={k})...")
        context_string, retrieved_parents = InMemoryDatabaseEngine.retrieve_dynamic_context(
            query=user_question,
            vector_store=vector_store,
            parent_map=parent_map,
            k=k
        )

        print("[TRANSACTION] Invoking ChatMistralAI grounded generation...")
        chain = self.prompt_template | self.llm
        llm_response = chain.invoke({
            "context": context_string,
            "question": user_question
        })

        answer_text = llm_response.content if hasattr(llm_response, "content") else str(llm_response)
        if not isinstance(answer_text, str):
            answer_text = str(answer_text)

        return {
            "answer": answer_text,
            "sources": retrieved_parents
        }
