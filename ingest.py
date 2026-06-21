"""
Document ingestion script for mental health RAG.
Run this once before starting the server: python ingest.py
"""

import os
import uuid
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "mental_health_docs"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


def main():
    print("=" * 50)
    print("Mental Health RAG — Document Ingestion")
    print("=" * 50)

    # Load embedding model
    print("\n[1/4] Loading embedding model (first run downloads ~90MB)...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("      ✓ Model ready")

    # Initialize ChromaDB
    print("\n[2/4] Setting up ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Clear existing collection
    try:
        client.delete_collection(COLLECTION_NAME)
        print("      ✓ Cleared previous index")
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)
    print("      ✓ Fresh collection created")

    # Load and chunk documents
    print(f"\n[3/4] Loading documents from {DOCS_DIR}/...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    all_ids = []
    all_metadatas = []
    total_files = 0

    supported_extensions = {".txt", ".md", ".pdf"}

    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.startswith("."):
            continue

        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue

        filepath = os.path.join(DOCS_DIR, filename)
        print(f"      Processing: {filename}")

        try:
            if ext == ".pdf":
                try:
                    from pypdf import PdfReader
                    reader = PdfReader(filepath)
                    text = "\n".join(
                        page.extract_text() or "" for page in reader.pages
                    )
                except ImportError:
                    print(f"        ⚠ pypdf not installed, skipping PDF")
                    continue
            else:
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

            if not text.strip():
                print(f"        ⚠ Empty file, skipping")
                continue

            chunks = splitter.split_text(text)
            print(f"        → {len(chunks)} chunks")

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                all_chunks.append(chunk)
                all_ids.append(str(uuid.uuid4()))
                all_metadatas.append({"source": filename, "chunk_index": i})

            total_files += 1

        except Exception as e:
            print(f"        ✗ Error: {e}")

    if not all_chunks:
        print("\n✗ No documents found to ingest. Make sure the docs/ folder has .txt or .md files.")
        return

    print(f"\n      Total: {len(all_chunks)} chunks from {total_files} files")

    # Embed and store
    print(f"\n[4/4] Embedding and storing (this may take a minute)...")
    embeddings = model.encode(
        all_chunks,
        show_progress_bar=True,
        batch_size=32,
    ).tolist()

    # Store in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:end],
            embeddings=embeddings[i:end],
            ids=all_ids[i:end],
            metadatas=all_metadatas[i:end],
        )

    print("\n" + "=" * 50)
    print(f"✅  Ingestion complete!")
    print(f"    Files processed : {total_files}")
    print(f"    Chunks stored   : {len(all_chunks)}")
    print(f"    Database path   : {CHROMA_DIR}/")
    print("=" * 50)
    print("\nNext step: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
