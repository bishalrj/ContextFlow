from __future__ import annotations

import io
import os
import sys
import uuid
from typing import List, Dict, Any, Tuple, Optional, Union, Generator
import pypdf
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


class InMemoryDatabaseEngine:
    """
    An in-memory, multi-tenant isolated RAG database engine under Antigravity Production Patterns.
    Eliminates all disk write dependencies by directly parsing byte streams, stitching continuous text,
    generating hierarchical Parent-Child chunks, and vectorizing child chunks in transient memory.
    """

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        parent_chunk_overlap: int = 300,
        child_chunk_size: int = 300,
        child_chunk_overlap: int = 50
    ) -> None:
        self.embeddings = MistralAIEmbeddings(model="mistral-embed")
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap
        )

    def process_and_index_stream(
        self, file_stream: io.BytesIO, file_name: str
    ) -> Tuple[FAISS, Dict[str, Dict[str, Any]]]:
        """
        Accepts an in-memory file stream, extracts text via pypdf, concatenates continuous text,
        generates hierarchical Parent Blocks and Child Chunks, and builds a transient FAISS vector store.
        Returns:
            Tuple[FAISS, Dict[str, Dict[str, Any]]]: (vector_store, parent_map)
        """
        if not file_stream:
            raise ValueError("File stream cannot be empty.")

        print(f"[INGEST] Reading in-memory byte stream for '{file_name}'...")
        reader = pypdf.PdfReader(file_stream)
        page_texts: List[str] = []

        for idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                page_texts.append(text.strip())

        if not page_texts:
            raise ValueError(f"No extractable text found in '{file_name}'.")

        # Continuous Text Stitching: Concatenate extracted text from all pages into one continuous string
        continuous_text = " ".join(page_texts)
        print(f"[STITCH] Concatenated {len(page_texts)} pages into continuous string ({len(continuous_text)} chars).")

        # Hierarchical Sliding Windows: Generate overlapping Parent Blocks
        parent_blocks = self.parent_splitter.create_documents(
            [continuous_text],
            metadatas=[{"file_name": file_name, "source": file_name}]
        )

        parent_map: Dict[str, Dict[str, Any]] = {}
        child_chunks: List[Document] = []

        # Child Vector Compilation & UUID tracking
        for p_idx, block in enumerate(parent_blocks, start=1):
            parent_id = str(uuid.uuid4())
            block_meta = dict(block.metadata)
            block_meta["parent_id"] = parent_id
            block_meta["parent_block_index"] = p_idx
            block_meta["page_index"] = p_idx

            parent_map[parent_id] = {
                "page_content": block.page_content,
                "metadata": block_meta
            }

            children = self.child_splitter.create_documents(
                [block.page_content],
                metadatas=[block_meta]
            )

            for c_idx, child in enumerate(children, start=1):
                child.metadata["parent_id"] = parent_id
                child.metadata["file_name"] = file_name
                child.metadata["source"] = file_name
                child.metadata["page_index"] = p_idx
                child.metadata["child_chunk_index"] = c_idx
                child_chunks.append(child)

        print(f"[INDEX] Vectorizing {len(child_chunks)} child chunks across {len(parent_blocks)} parent blocks in memory...")
        vector_store = FAISS.from_documents(child_chunks, self.embeddings)
        print("[SUCCESS] Transient FAISS vector store and parent map compiled in memory.")

        return vector_store, parent_map

    @staticmethod
    def retrieve_dynamic_context(
        query: str,
        vector_store: FAISS,
        parent_map: Dict[str, Dict[str, Any]],
        k: int = 3
    ) -> Tuple[str, List[Document]]:
        """
        Executes similarity matching against child vectors, gathers associated parent UUIDs,
        extracts complete un-fragmented parent text strings, and returns combined context string + source array.
        """
        if not vector_store or not parent_map:
            raise RuntimeError("Vector store or parent map is uninitialized.")

        child_hits = vector_store.similarity_search(query, k=k)
        retrieved_parents: List[Document] = []
        seen_parent_ids = set()

        for hit in child_hits:
            p_id = hit.metadata.get("parent_id")
            if p_id and p_id in parent_map and p_id not in seen_parent_ids:
                seen_parent_ids.add(p_id)
                parent_info = parent_map[p_id]
                retrieved_parents.append(Document(
                    page_content=parent_info["page_content"],
                    metadata=dict(parent_info["metadata"])
                ))

        combined_context_string = "\n\n---\n\n".join([doc.page_content for doc in retrieved_parents])
        return combined_context_string, retrieved_parents


retrieve_dynamic_context = InMemoryDatabaseEngine.retrieve_dynamic_context
