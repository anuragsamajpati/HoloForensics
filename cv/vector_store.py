"""
Vector Store System for HoloForensics
Implements ChromaDB-based vector storage for forensic analysis results
"""

import chromadb
import numpy as np
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path
import hashlib
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ForensicDocument:
    """Represents a forensic document for vector storage"""
    doc_id: str
    case_id: str
    document_type: str  # 'scene_analysis', 'event', 'evidence', 'trajectory', 'report'
    content: str
    metadata: Dict[str, Any]
    timestamp: datetime
    confidence: float
    source_system: str
    hash_signature: str

@dataclass
class QueryResult:
    """Represents a query result from vector store"""
    doc_id: str
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    document_type: str
    case_id: str
    timestamp: datetime

class ForensicEmbeddingModel:
    """Handles embedding generation for forensic documents"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model {self.model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a simpler model
            try:
                self.model = SentenceTransformer("all-mpnet-base-v2", device=self.device)
                logger.info("Loaded fallback embedding model")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                raise e2
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate embeddings for text
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector as numpy array
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            # Return zero vector as fallback
            return np.zeros(384)  # Default dimension for MiniLM
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for batch of texts
        
        Args:
            texts: List of texts to encode
            
        Returns:
            Array of embedding vectors
        """
        if not self.model:
            raise RuntimeError("Embedding model not loaded")
        
        try:
            # Clean and preprocess texts
            cleaned_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding batch: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), 384))
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding
        
        Args:
            text: Raw text input
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Basic cleaning
        cleaned = text.strip()
        
        # Remove excessive whitespace
        cleaned = " ".join(cleaned.split())
        
        # Truncate if too long (models have token limits)
        max_length = 500  # Conservative limit
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."
        
        return cleaned

class ForensicVectorStore:
    """Main vector store class for forensic data"""
    
    def __init__(self, 
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "holoforensics_collection"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self.embedding_model = None
        
        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_client()
        self._initialize_embedding_model()
        self._initialize_collection()
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            logger.info(f"Initialized ChromaDB client at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB client: {e}")
            raise e
    
    def _initialize_embedding_model(self):
        """Initialize embedding model"""
        try:
            self.embedding_model = ForensicEmbeddingModel()
            logger.info("Initialized embedding model")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise e
    
    def _initialize_collection(self):
        """Initialize or get ChromaDB collection"""
        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"Retrieved existing collection: {self.collection_name}")
            except Exception:
                # Create new collection
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "HoloForensics forensic analysis data"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise e
    
    def add_document(self, document: ForensicDocument) -> bool:
        """
        Add a forensic document to the vector store
        
        Args:
            document: ForensicDocument to add
            
        Returns:
            Success status
        """
        try:
            # Generate embedding
            embedding = self.embedding_model.encode_text(document.content)
            
            # Prepare metadata
            metadata = {
                "case_id": document.case_id,
                "document_type": document.document_type,
                "timestamp": document.timestamp.isoformat(),
                "confidence": document.confidence,
                "source_system": document.source_system,
                "hash_signature": document.hash_signature,
                **document.metadata
            }
            
            # Add to collection
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[document.content],
                metadatas=[metadata],
                ids=[document.doc_id]
            )
            
            logger.info(f"Added document {document.doc_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add document {document.doc_id}: {e}")
            return False
    
    def add_documents_batch(self, documents: List[ForensicDocument]) -> int:
        """
        Add multiple documents in batch
        
        Args:
            documents: List of ForensicDocument objects
            
        Returns:
            Number of successfully added documents
        """
        if not documents:
            return 0
        
        try:
            # Generate embeddings in batch
            contents = [doc.content for doc in documents]
            embeddings = self.embedding_model.encode_batch(contents)
            
            # Prepare data for batch insert
            ids = [doc.doc_id for doc in documents]
            metadatas = []
            
            for doc in documents:
                metadata = {
                    "case_id": doc.case_id,
                    "document_type": doc.document_type,
                    "timestamp": doc.timestamp.isoformat(),
                    "confidence": doc.confidence,
                    "source_system": doc.source_system,
                    "hash_signature": doc.hash_signature,
                    **doc.metadata
                }
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=contents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to add batch documents: {e}")
            return 0
    
    def query(self, 
              query_text: str, 
              n_results: int = 10,
              case_id: Optional[str] = None,
              document_type: Optional[str] = None,
              min_confidence: Optional[float] = None) -> List[QueryResult]:
        """
        Query the vector store
        
        Args:
            query_text: Text to search for
            n_results: Number of results to return
            case_id: Filter by case ID
            document_type: Filter by document type
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of QueryResult objects
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode_text(query_text)
            
            # Build where clause for filtering
            where_clause = {}
            if case_id:
                where_clause["case_id"] = case_id
            if document_type:
                where_clause["document_type"] = document_type
            if min_confidence:
                where_clause["confidence"] = {"$gte": min_confidence}
            
            # Execute query
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                where=where_clause if where_clause else None
            )
            
            # Process results
            query_results = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    metadata = results["metadatas"][0][i]
                    
                    query_result = QueryResult(
                        doc_id=results["ids"][0][i],
                        content=results["documents"][0][i],
                        metadata=metadata,
                        similarity_score=1.0 - results["distances"][0][i],  # Convert distance to similarity
                        document_type=metadata.get("document_type", "unknown"),
                        case_id=metadata.get("case_id", "unknown"),
                        timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
                    )
                    query_results.append(query_result)
            
            logger.info(f"Query returned {len(query_results)} results")
            return query_results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[QueryResult]:
        """
        Get a specific document by ID
        
        Args:
            doc_id: Document ID to retrieve
            
        Returns:
            QueryResult if found, None otherwise
        """
        try:
            results = self.collection.get(ids=[doc_id])
            
            if results["documents"] and len(results["documents"]) > 0:
                metadata = results["metadatas"][0]
                
                return QueryResult(
                    doc_id=doc_id,
                    content=results["documents"][0],
                    metadata=metadata,
                    similarity_score=1.0,
                    document_type=metadata.get("document_type", "unknown"),
                    case_id=metadata.get("case_id", "unknown"),
                    timestamp=datetime.fromisoformat(metadata.get("timestamp", datetime.now().isoformat()))
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            Success status
        """
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Deleted document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False
    
    def delete_case(self, case_id: str) -> int:
        """
        Delete all documents for a case
        
        Args:
            case_id: Case ID to delete
            
        Returns:
            Number of deleted documents
        """
        try:
            # Get all documents for the case
            results = self.collection.get(where={"case_id": case_id})
            
            if results["ids"]:
                # Delete all documents
                self.collection.delete(ids=results["ids"])
                deleted_count = len(results["ids"])
                logger.info(f"Deleted {deleted_count} documents for case {case_id}")
                return deleted_count
            
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete case {case_id}: {e}")
            return 0
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze types
            sample_results = self.collection.get(limit=100)
            
            doc_types = {}
            case_ids = set()
            
            if sample_results["metadatas"]:
                for metadata in sample_results["metadatas"]:
                    doc_type = metadata.get("document_type", "unknown")
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                    
                    case_id = metadata.get("case_id")
                    if case_id:
                        case_ids.add(case_id)
            
            return {
                "total_documents": count,
                "document_types": doc_types,
                "unique_cases": len(case_ids),
                "collection_name": self.collection_name,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

# Utility functions for creating forensic documents
def create_scene_analysis_document(analysis_result, case_id: str) -> ForensicDocument:
    """Create a document from scene analysis results"""
    
    # Create content summary
    content_parts = [
        f"Scene Analysis for Case {case_id}",
        f"Total Events: {len(analysis_result.detected_events)}",
        f"Total Evidence: {len(analysis_result.evidence_items)}",
        f"Quality Score: {analysis_result.quality_metrics.get('overall_score', 0.0):.2f}"
    ]
    
    # Add event descriptions
    for event in analysis_result.detected_events[:5]:  # Top 5 events
        content_parts.append(f"Event: {event.description} (Confidence: {event.confidence:.2f})")
    
    content = "\n".join(content_parts)
    
    # Create hash signature
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    return ForensicDocument(
        doc_id=f"scene_analysis_{analysis_result.analysis_id}",
        case_id=case_id,
        document_type="scene_analysis",
        content=content,
        metadata={
            "analysis_id": analysis_result.analysis_id,
            "operator_id": analysis_result.operator_id,
            "total_events": len(analysis_result.detected_events),
            "total_evidence": len(analysis_result.evidence_items),
            "quality_score": analysis_result.quality_metrics.get('overall_score', 0.0)
        },
        timestamp=analysis_result.timestamp,
        confidence=analysis_result.quality_metrics.get('overall_score', 0.0),
        source_system="holoforensics_scene_analysis",
        hash_signature=content_hash
    )

def create_event_document(event, case_id: str) -> ForensicDocument:
    """Create a document from a detected event"""
    
    content = f"Event Type: {event.event_type.value}\n"
    content += f"Severity: {event.severity.value}\n"
    content += f"Description: {event.description}\n"
    content += f"Location: {event.location}\n"
    content += f"Time: {event.start_time}s - {event.end_time}s\n"
    content += f"Involved Objects: {', '.join(event.involved_objects)}"
    
    # Create hash signature
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    
    return ForensicDocument(
        doc_id=f"event_{event.event_id}",
        case_id=case_id,
        document_type="event",
        content=content,
        metadata={
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "severity": event.severity.value,
            "start_time": event.start_time,
            "end_time": event.end_time,
            "location": event.location,
            "involved_objects": event.involved_objects
        },
        timestamp=datetime.now(),
        confidence=event.confidence,
        source_system="holoforensics_event_detection",
        hash_signature=content_hash
    )

# Example usage
if __name__ == "__main__":
    # Initialize vector store
    vector_store = ForensicVectorStore()
    
    # Create sample document
    sample_doc = ForensicDocument(
        doc_id=str(uuid.uuid4()),
        case_id="CASE_2024_001",
        document_type="test",
        content="This is a test forensic document for vector storage testing.",
        metadata={"test": True},
        timestamp=datetime.now(),
        confidence=0.95,
        source_system="test_system",
        hash_signature=hashlib.sha256(b"test").hexdigest()
    )
    
    # Add document
    success = vector_store.add_document(sample_doc)
    print(f"Document added: {success}")
    
    # Query documents
    results = vector_store.query("forensic document test")
    print(f"Query results: {len(results)}")
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"Collection stats: {stats}")
