import json
import os

import numpy as np

from vxdf.reader import VXDFReader
from vxdf.writer import VXDFWriter

# --- Configuration ---
FILE_PATH = "sample.vxdf"
EMBEDDING_DIM = 128

def create_sample_vxdf_file():
    """Creates a sample VXDF file with dummy data."""
    print(f"Creating sample VXDF file at '{FILE_PATH}'...")

    # 1. Define sample data
    sample_data = [
        {
            "id": "doc_001",
            "text": "The first rule of VXDF is you do not talk about VXDF.",
            "meta": {"source": "Internal Memo", "author": "Tyler"},
            "vector": np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()
        },
        {
            "id": "doc_002",
            "text": "The second rule of VXDF is you DO NOT talk about VXDF.",
            "meta": {"source": "Internal Memo", "author": "Tyler"},
            "vector": np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()
        },
        {
            "id": "doc_003",
            "text": "With VXDF, you can store text, metadata, and vectors in one file.",
            "meta": {"source": "Documentation", "author": "Cascade"},
            "vector": np.random.rand(EMBEDDING_DIM).astype(np.float32).tolist()
        }
    ]

    # 2. Use VXDFWriter to create the file
    with VXDFWriter(FILE_PATH, embedding_dim=EMBEDDING_DIM, compression="zstd") as writer:
        for chunk in sample_data:
            writer.add_chunk(chunk)
    
    return sample_data

def read_and_verify_vxdf_file(original_data):
    """Reads the created VXDF file and verifies its content."""
    print(f"\nReading and verifying VXDF file '{FILE_PATH}'...")

    # 3. Use VXDFReader to open the file
    with VXDFReader(FILE_PATH) as reader:
        print(f"VXDF Version: {reader.vxdf_version}")
        print(f"Embedding Dimension: {reader.embedding_dim}")

        # 4. Test random access (get a specific chunk)
        print("\n--- Testing Get Chunk by ID ---")
        doc_id_to_get = "doc_002"
        chunk = reader.get_chunk(doc_id_to_get)
        print(f"Successfully retrieved chunk for ID '{doc_id_to_get}':")
        print(json.dumps(chunk, indent=2))

        # 5. Test iteration (stream all chunks)
        print("\n--- Testing Iterate All Chunks ---")
        read_data = []
        for i, read_chunk in enumerate(reader.iter_chunks()):
            print(f"Chunk {i+1}: ID = {read_chunk['id']}")
            read_data.append(read_chunk)

        # 6. Verify data integrity
        print("\n--- Verifying Data ---")
        # Note: Comparing floats requires tolerance. We'll compare IDs for simplicity here.
        original_ids = sorted([d['id'] for d in original_data])
        read_ids = sorted([d['id'] for d in read_data])

        assert original_ids == read_ids, "Mismatch between original and read data!"
        print("Verification successful: All original document IDs were found in the file.")


if __name__ == "__main__":
    # Clean up previous run if file exists
    if os.path.exists(FILE_PATH):
        os.remove(FILE_PATH)

    # Run the example
    original_data = create_sample_vxdf_file()
    read_and_verify_vxdf_file(original_data)

    # Clean up the created file
    # os.remove(FILE_PATH)
    print(f"\nExample finished. The file '{FILE_PATH}' was left for inspection.")
