#!/usr/bin/env python3
"""
Test script for HoloForensics RAG System
Demonstrates vector store and RAG functionality with sample forensic data
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add the cv directory to the Python path
sys.path.append('/Users/anuragsamajpati/Desktop/holoforensics/cv')

from vector_store import ForensicVectorStore, ForensicDocument
from rag_system import ForensicRAGSystem

def create_sample_forensic_data():
    """Create sample forensic data for testing"""
    
    # Sample scene analysis document
    scene_doc = ForensicDocument(
        doc_id="scene_analysis_001",
        case_id="CASE_2024_001",
        document_type="scene_analysis",
        content="""Scene Analysis Results for Multi-Camera Crime Scene
Analysis ID: SA_001
Operator: Detective Smith
Total Events Detected: 15
Total Evidence Items: 8
Quality Score: 0.892

Detected Events:
- Person Entry: Suspect enters frame at 12.5s with high confidence
- Vehicle Movement: Black sedan moves through intersection at 18.2s
- Object Interaction: Person picks up object from ground at 25.7s
- Suspicious Activity: Loitering behavior detected from 30s to 45s
- Person Exit: Suspect exits frame at 52.3s

Evidence Items:
- Dropped object at coordinates (245, 180)
- Tire tracks visible in parking area
- Person of interest identified with 87% confidence
- License plate partially visible: ABC-123X
""",
        metadata={
            'analysis_id': 'SA_001',
            'operator_id': 'detective_smith',
            'total_events': 15,
            'total_evidence': 8,
            'quality_score': 0.892,
            'location': 'intersection_main_elm'
        },
        timestamp=datetime.now(),
        confidence=0.892,
        source_system="holoforensics_scene_analysis",
        hash_signature="abc123def456"
    )
    
    # Sample event documents
    event_docs = [
        ForensicDocument(
            doc_id="event_001",
            case_id="CASE_2024_001",
            document_type="event",
            content="""Event Type: Person Entry
Severity: Medium
Description: Suspect individual enters surveillance area from north entrance
Time Range: 12.5s - 13.2s
Location: North entrance, coordinates (120, 85)
Confidence: 0.91
Involved Objects: person_001
Additional Notes: Individual wearing dark clothing, baseball cap obscuring face""",
            metadata={
                'event_id': 'evt_001',
                'event_type': 'person_entry',
                'severity': 'medium',
                'start_time': 12.5,
                'end_time': 13.2,
                'location': 'north_entrance',
                'involved_objects': ['person_001']
            },
            timestamp=datetime.now(),
            confidence=0.91,
            source_system="holoforensics_event_detection",
            hash_signature="evt001hash"
        ),
        
        ForensicDocument(
            doc_id="event_002",
            case_id="CASE_2024_001",
            document_type="event",
            content="""Event Type: Vehicle Movement
Severity: High
Description: Black sedan with partially obscured license plate moves through intersection
Time Range: 18.2s - 22.8s
Location: Main intersection, coordinates (300, 200)
Confidence: 0.95
Involved Objects: vehicle_001
Additional Notes: License plate ABC-123X, possible getaway vehicle""",
            metadata={
                'event_id': 'evt_002',
                'event_type': 'vehicle_movement',
                'severity': 'high',
                'start_time': 18.2,
                'end_time': 22.8,
                'location': 'main_intersection',
                'involved_objects': ['vehicle_001']
            },
            timestamp=datetime.now(),
            confidence=0.95,
            source_system="holoforensics_event_detection",
            hash_signature="evt002hash"
        ),
        
        ForensicDocument(
            doc_id="event_003",
            case_id="CASE_2024_001",
            document_type="event",
            content="""Event Type: Object Interaction
Severity: Critical
Description: Person picks up object from ground, possible evidence tampering
Time Range: 25.7s - 26.3s
Location: Parking area, coordinates (245, 180)
Confidence: 0.88
Involved Objects: person_001, object_001
Additional Notes: Object appears to be small package or container""",
            metadata={
                'event_id': 'evt_003',
                'event_type': 'object_interaction',
                'severity': 'critical',
                'start_time': 25.7,
                'end_time': 26.3,
                'location': 'parking_area',
                'involved_objects': ['person_001', 'object_001']
            },
            timestamp=datetime.now(),
            confidence=0.88,
            source_system="holoforensics_event_detection",
            hash_signature="evt003hash"
        )
    ]
    
    # Sample physics prediction document
    physics_doc = ForensicDocument(
        doc_id="physics_prediction_001",
        case_id="CASE_2024_001",
        document_type="physics_prediction",
        content="""Physics-Informed Trajectory Prediction
Job ID: PP_001
Prediction Method: Kalman Filter + Social Forces
Number of Trajectories: 3
Confidence Score: 0.847

Predicted Trajectories:
- Trajectory 1: Person movement prediction showing likely exit through south door
- Trajectory 2: Vehicle path prediction indicating turn towards highway
- Trajectory 3: Object trajectory showing ballistic path consistent with throwing motion

Social Force Analysis:
- Avoidance behavior detected between person and vehicle
- Attraction force towards exit points
- Repulsion from obstacle areas""",
        metadata={
            'job_id': 'PP_001',
            'prediction_type': 'kalman_social_forces',
            'num_trajectories': 3,
            'confidence_score': 0.847
        },
        timestamp=datetime.now(),
        confidence=0.847,
        source_system="holoforensics_physics_prediction",
        hash_signature="pp001hash"
    )
    
    # Sample video inpainting document
    inpainting_doc = ForensicDocument(
        doc_id="video_inpainting_001",
        case_id="CASE_2024_001",
        document_type="video_inpainting",
        content="""Video Inpainting Analysis
Job ID: VI_001
Method: E2FGVI (Enhanced Flow-Guided Video Inpainting)
Processed Frames: 1250
Quality Score: 0.923
Processing Time: 45.7s
Status: Completed Successfully

Inpainting Results:
- Successfully reconstructed 15 frames with missing data
- Maintained temporal consistency across sequence
- Preserved forensic integrity with chain of custody
- Generated confidence masks for reconstructed regions""",
        metadata={
            'job_id': 'VI_001',
            'inpainting_method': 'E2FGVI',
            'processed_frames': 1250,
            'quality_score': 0.923
        },
        timestamp=datetime.now(),
        confidence=0.923,
        source_system="holoforensics_video_inpainting",
        hash_signature="vi001hash"
    )
    
    return [scene_doc] + event_docs + [physics_doc, inpainting_doc]

def test_vector_store():
    """Test vector store functionality"""
    print("🔧 Testing Vector Store...")
    
    # Initialize vector store
    vector_store = ForensicVectorStore(persist_directory="./test_vector_store")
    
    # Create sample data
    documents = create_sample_forensic_data()
    
    # Add documents
    print(f"📄 Adding {len(documents)} documents to vector store...")
    added_count = vector_store.add_documents_batch(documents)
    print(f"✅ Successfully added {added_count} documents")
    
    # Test queries
    test_queries = [
        "What events occurred with high confidence?",
        "Show me vehicle movements",
        "Find critical severity events",
        "What happened between 20 and 30 seconds?",
        "Show me physics predictions"
    ]
    
    print("\n🔍 Testing vector store queries...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = vector_store.query(query_text=query, n_results=3)
        
        for i, result in enumerate(results):
            print(f"  {i+1}. {result.document_type} (similarity: {result.similarity_score:.3f})")
            print(f"     {result.content[:100]}...")
    
    # Get stats
    stats = vector_store.get_collection_stats()
    print(f"\n📊 Vector Store Stats:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Document types: {stats['document_types']}")
    print(f"  Cases: {stats['cases']}")
    
    return vector_store

def test_rag_system(vector_store):
    """Test RAG system functionality"""
    print("\n🤖 Testing RAG System...")
    
    # Initialize RAG system
    rag_system = ForensicRAGSystem(vector_store=vector_store)
    
    # Test queries
    test_queries = [
        "What events occurred between 10 and 20 seconds?",
        "Show me high confidence evidence from case CASE_2024_001",
        "Find all person-vehicle interactions",
        "What happened at the main intersection?",
        "Show critical events with confidence above 0.8",
        "What physics predictions were made?",
        "Show me video inpainting results",
        "Find events involving person_001"
    ]
    
    print("\n🔍 Testing RAG queries...")
    for query_text in test_queries:
        print(f"\n" + "="*60)
        print(f"Query: {query_text}")
        print("="*60)
        
        response = rag_system.process_query(
            query_text=query_text,
            case_id="CASE_2024_001",
            user_id="test_user"
        )
        
        print(f"Response: {response.response_text}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Processing time: {response.processing_time:.3f}s")
        print(f"Sources: {', '.join(response.sources)}")
        
        if response.retrieved_documents:
            print(f"Retrieved {len(response.retrieved_documents)} documents:")
            for doc in response.retrieved_documents[:3]:
                print(f"  - {doc.document_type}: {doc.similarity_score:.3f}")
    
    # Test system stats
    stats = rag_system.get_system_stats()
    print(f"\n📊 RAG System Stats:")
    print(f"  Query history: {stats['query_history_count']}")
    print(f"  Recent queries: {stats['recent_queries']}")
    print(f"  Vector store: {stats['vector_store']['total_documents']} documents")
    
    return rag_system

def test_query_analysis():
    """Test query analysis functionality"""
    print("\n🔬 Testing Query Analysis...")
    
    from rag_system import ForensicQueryProcessor
    
    processor = ForensicQueryProcessor()
    
    test_queries = [
        "What events occurred between 10 and 20 seconds with confidence above 0.8?",
        "Show me critical severity incidents at the main intersection",
        "Find all person-vehicle interactions in case CASE_2024_001",
        "What happened during the first 30 seconds?",
        "Show me high confidence evidence from the parking area"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        analysis = processor.analyze_query(query)
        
        print(f"  Query type: {analysis['query_type']}")
        print(f"  Entities: {analysis['entities']}")
        print(f"  Filters: {analysis['filters']}")
        
        enhanced = processor.enhance_query(query, analysis)
        print(f"  Enhanced: {enhanced}")

def main():
    """Main test function"""
    print("🚀 HoloForensics RAG System Test Suite")
    print("="*50)
    
    try:
        # Test vector store
        vector_store = test_vector_store()
        
        # Test RAG system
        rag_system = test_rag_system(vector_store)
        
        # Test query analysis
        test_query_analysis()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Summary:")
        print("  - Vector store: ✅ Working")
        print("  - RAG system: ✅ Working") 
        print("  - Query analysis: ✅ Working")
        print("  - Document indexing: ✅ Working")
        print("  - Semantic search: ✅ Working")
        
        print(f"\n🎯 Ready for integration with Django web application!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
