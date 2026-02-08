from __future__ import annotations

import chromadb
import json

from pathlib import Path
from typing import List, Dict
from chromadb.utils import embedding_functions


class ClinicalVectorStore:
    """
    RAG engine that indexes physician examples and knowledge base documents.
    Enables semantic search for similar clinical cases during recommendation generation.
    """

    def __init__(self, persist_directory: str = "./data/vector_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=str(self.persist_directory))

        # Use local embedding model (no internet required)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )

        # Create or get collections
        self.examples_collection = self.client.get_or_create_collection(
            name="physician_examples",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

        self.knowledge_collection = self.client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_physician_examples(self, examples_json_path: str) -> int:
        """
        Ingest physician examples from JSON file into vector store.

        Format expected:
        [
          {
            "patient_context": {"age": 52, "sex": "Male", "symptoms": [...], "labs": {...}},
            "physician_recommendation": {"assessment": "...", "recommendations": [...], ...}
          },
          ...
        ]

        Returns:
            Number of examples successfully ingested
        """
        with open(examples_json_path, 'r', encoding='utf-8') as f:
            examples = json.load(f)

        if not isinstance(examples, list):
            raise ValueError("Examples file must contain a JSON array")

        documents = []
        metadatas = []
        ids = []

        for i, ex in enumerate(examples):
            # Create searchable text from patient context
            patient_context = ex.get('patient_context', {})
            labs = patient_context.get('labs', {})

            doc_text = f"""
Patient: {patient_context.get('age', 'N/A')} year old {patient_context.get('sex', 'N/A')}
Symptoms: {', '.join(patient_context.get('symptoms', []))}
Labs: Glucose {labs.get('glucose_mg_dl', 'N/A')} mg/dL, 
      HbA1c {labs.get('hba1c_percent', 'N/A')}%, 
      Creatinine {labs.get('creatinine_mg_dl', 'N/A')} mg/dL
Recommendation: {json.dumps(ex.get('physician_recommendation', {}))}
            """.strip()

            documents.append(doc_text)
            metadatas.append({
                "example_id": ex.get("example_id", f"ex_{i}"),
                "age": str(patient_context.get("age", "")),
                "sex": patient_context.get("sex", ""),
                "urgency": ex.get("physician_recommendation", {}).get("urgency_level", "medium")
            })
            ids.append(f"example_{i}")

        # Add to vector store
        self.examples_collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

        return len(examples)

    def ingest_knowledge_documents(self, documents: List[str], doc_names: List[str]) -> int:
        """
        Ingest knowledge base documents (guidelines, protocols).

        Args:
            documents: List of document text content
            doc_names: List of document names/identifiers

        Returns:
            Number of documents ingested
        """
        self.knowledge_collection.add(
            documents=documents,
            metadatas=[{"source": name} for name in doc_names],
            ids=[f"kb_{i}" for i in range(len(documents))]
        )
        return len(documents)

    def semantic_search_examples(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Find similar physician examples using semantic search.

        Returns:
            List of similar examples with similarity scores
        """
        results = self.examples_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity": 1.0 - results["distances"][0][i]  # Convert distance to similarity
            })
        return matches

    def semantic_search_knowledge(self, query: str, top_k: int = 2) -> List[Dict]:
        """
        Search knowledge base for relevant guidelines/protocols.
        """
        results = self.knowledge_collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas"]
        )

        matches = []
        for i in range(len(results["ids"][0])):
            matches.append({
                "document": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown")
            })
        return matches
