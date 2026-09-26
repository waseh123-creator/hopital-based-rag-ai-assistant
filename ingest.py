
import os
import pickle
import faiss

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ==========================================
# CONFIGURATION
# ==========================================

PDF_FOLDER = "hospital_knowledge_base"
FAISS_FOLDER = "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ==========================================
# LOAD PDFS
# ==========================================

def load_pdfs():

    documents = []

    if not os.path.exists(PDF_FOLDER):

        print(f"Folder not found: {PDF_FOLDER}")
        return documents

    pdf_files = [
        file for file in os.listdir(PDF_FOLDER)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:

        print("No PDF files found.")
        return documents

    for filename in pdf_files:

        filepath = os.path.join(
            PDF_FOLDER,
            filename
        )

        print(f"Reading: {filename}")

        try:

            reader = PdfReader(filepath)

            for page_number, page in enumerate(reader.pages):

                text = page.extract_text()

                if text and text.strip():

                    documents.append({
                        "text": text.strip(),
                        "source": filename,
                        "page": page_number + 1
                    })

        except Exception as e:

            print(f"Error reading {filename}: {e}")

    return documents


# ==========================================
# SPLIT INTO CHUNKS
# ==========================================

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=800,

        chunk_overlap=150,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = []

    for document in documents:

        split_texts = splitter.split_text(
            document["text"]
        )

        for text in split_texts:

            if text.strip():

                chunks.append({

                    "text": text.strip(),

                    "source": document["source"],

                    "page": document["page"]

                })

    return chunks


# ==========================================
# CREATE FAISS INDEX
# ==========================================

def create_faiss_index(chunks):

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print(
        f"Creating embeddings for "
        f"{len(texts)} chunks..."
    )

    embeddings = model.encode(

        texts,

        show_progress_bar=True,

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    embedding_dimension = embeddings.shape[1]

    # Inner Product is suitable with
    # normalized embeddings
    index = faiss.IndexFlatIP(
        embedding_dimension
    )

    index.add(
        embeddings.astype("float32")
    )

    os.makedirs(
        FAISS_FOLDER,
        exist_ok=True
    )

    # Save FAISS index
    faiss.write_index(

        index,

        os.path.join(
            FAISS_FOLDER,
            "index.faiss"
        )
    )

    # Save chunk metadata
    with open(

        os.path.join(
            FAISS_FOLDER,
            "chunks.pkl"
        ),

        "wb"
    ) as file:

        pickle.dump(
            chunks,
            file
        )

    print("\n================================")
    print("FAISS INDEX CREATED SUCCESSFULLY")
    print("================================")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding dimension: {embedding_dimension}")

    print("\nSaved files:")

    print(
        os.path.join(
            FAISS_FOLDER,
            "index.faiss"
        )
    )

    print(
        os.path.join(
            FAISS_FOLDER,
            "chunks.pkl"
        )
    )


# ==========================================
# MAIN
# ==========================================

def main():

    print("========================================")
    print("Hospital Knowledge Base Ingestion")
    print("========================================")

    documents = load_pdfs()

    if not documents:

        print(
            "\nNo readable PDF content found."
        )

        return

    print(
        f"\nTotal pages loaded: "
        f"{len(documents)}"
    )

    chunks = create_chunks(
        documents
    )

    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )

    if not chunks:

        print(
            "No chunks were created."
        )

        return

    create_faiss_index(
        chunks
    )

    print("\nIngestion completed successfully!")


if __name__ == "__main__":

    main()
