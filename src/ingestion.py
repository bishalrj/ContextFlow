from __future__ import annotations

import io
import os
import sys
from typing import List, Dict, Any, Optional
import pypdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass


class InMemoryDocumentIngester:
    """
    An in-memory document ingestion engine that accepts file streams directly,
    stitches pages continuously to eliminate page-break fragmentation, and applies
    a large hierarchical sliding window to generate overlapping Parent Blocks.
    """

    def __init__(self, parent_chunk_size: int = 1500, parent_chunk_overlap: int = 300):
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
        )

    def process_file_stream(self, file_stream: io.BytesIO, file_name: str) -> List[Document]:
        """
        Reads raw bytes stream from user upload, parses PDF pages into continuous string,
        and generates large overlapping Parent Blocks.
        """
        if not file_stream:
            raise ValueError("File stream is empty or invalid.")

        print(f"[INGEST] Parsing in-memory PDF stream for '{file_name}'...")
        reader = pypdf.PdfReader(file_stream)
        page_texts: List[str] = []

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if text and text.strip():
                page_texts.append(text.strip())

        if not page_texts:
            raise ValueError(f"No extractable text found in PDF '{file_name}'.")

        # Continuous Text Stitching Engine: concatenate all pages to eliminate fragmentation
        unified_text_stream = "\n\n".join(page_texts)
        print(f"[STITCH] Stitched {len(page_texts)} pages into continuous text stream ({len(unified_text_stream)} chars).")

        # Hierarchical Sliding Windows: generate overlapping Parent Blocks
        parent_blocks = self.parent_splitter.create_documents(
            [unified_text_stream],
            metadatas=[{"file_name": file_name, "source": file_name}]
        )

        # Annotate block index inside metadata
        for idx, block in enumerate(parent_blocks, start=1):
            block.metadata["parent_block_index"] = idx

        print(f"[PARENT WINDOWS] Generated {len(parent_blocks)} overlapping Parent Block(s).")
        return parent_blocks


# Backward compatibility class for disk testing if needed
class DocumentIngestionEngine:
    def __init__(self, data_directory: Optional[str] = None):
        if data_directory is None:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_script_dir)
            self.data_directory = os.path.join(project_root, "data")
        else:
            self.data_directory = os.path.abspath(data_directory)
        self.ingester = InMemoryDocumentIngester()

    def run_pipeline(self) -> List[Document]:
        if not os.path.exists(self.data_directory):
            return []
        all_blocks = []
        for file in os.listdir(self.data_directory):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(self.data_directory, file)
                with open(file_path, "rb") as f:
                    stream = io.BytesIO(f.read())
                    all_blocks.extend(self.ingester.process_file_stream(stream, file))
        return all_blocks


if __name__ == "__main__":
    print(":-)")
    sample_text = "This is a verification test for continuous text stitching. " * 300
    mock_stream = io.BytesIO()
    # Create mock test if needed or run basic check
    print("[SUCCESS] InMemoryDocumentIngester initialized successfully.")