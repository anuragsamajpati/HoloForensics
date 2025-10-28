"""
Django REST API views for RAG (Retrieval-Augmented Generation) System
"""

import json
import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Any

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.conf import settings
from django.utils.decorators import method_decorator
from django.views import View

# Import our RAG system modules
import sys
sys.path.append('/Users/anuragsamajpati/Desktop/holoforensics/cv')

from rag_system import ForensicRAGSystem, RAGQuery, RAGResponse
from vector_store import ForensicVectorStore, ForensicDocument, create_scene_analysis_document, create_event_document
from forensic_scene_analysis import ForensicAnalysisResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system = None

def get_rag_system():
    """Get or initialize the global RAG system"""
    global rag_system
    if rag_system is None:
        try:
            # Initialize with persistent storage
            vector_store_path = os.path.join(settings.MEDIA_ROOT, 'vector_store')
            os.makedirs(vector_store_path, exist_ok=True)
            
            vector_store = ForensicVectorStore(persist_directory=vector_store_path)
            rag_system = ForensicRAGSystem(vector_store=vector_store)
            
            logger.info("Initialized RAG system")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise e
    
    return rag_system

@method_decorator([login_required, csrf_exempt], name='dispatch')
class QueryRAGView(View):
    """Process forensic queries using RAG system"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            
            # Validate required fields
            query_text = data.get('query_text', '').strip()
            if not query_text:
                return JsonResponse({
                    'success': False,
                    'error': 'query_text is required'
                }, status=400)
            
            # Get optional parameters
            case_id = data.get('case_id')
            max_results = data.get('max_results', 10)
            
            # Get RAG system
            rag = get_rag_system()
            
            # Process query
            response = rag.process_query(
                query_text=query_text,
                case_id=case_id,
                user_id=request.user.username,
                max_results=max_results
            )
            
            # Format response
            response_data = {
                'success': True,
                'query_id': response.query_id,
                'response_text': response.response_text,
                'confidence_score': response.confidence_score,
                'processing_time': response.processing_time,
                'sources': response.sources,
                'retrieved_documents': [
                    {
                        'doc_id': doc.doc_id,
                        'document_type': doc.document_type,
                        'case_id': doc.case_id,
                        'similarity_score': doc.similarity_score,
                        'content_preview': doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        'timestamp': doc.timestamp.isoformat(),
                        'metadata': {k: v for k, v in doc.metadata.items() if k not in ['content', 'raw_data']}
                    }
                    for doc in response.retrieved_documents[:5]  # Top 5 for response
                ],
                'metadata': response.metadata
            }
            
            logger.info(f"Processed RAG query for user {request.user.username}")
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required, csrf_exempt], name='dispatch')
class IndexForensicDataView(View):
    """Index forensic analysis results into vector store"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            
            # Validate required fields
            analysis_data = data.get('analysis_data')
            case_id = data.get('case_id')
            
            if not analysis_data or not case_id:
                return JsonResponse({
                    'success': False,
                    'error': 'analysis_data and case_id are required'
                }, status=400)
            
            # Get RAG system
            rag = get_rag_system()
            
            # Create documents from analysis data
            documents = []
            
            # Handle different types of analysis data
            if 'scene_analysis' in analysis_data:
                # Scene analysis results
                scene_data = analysis_data['scene_analysis']
                
                # Create main scene analysis document
                scene_doc = ForensicDocument(
                    doc_id=f"scene_analysis_{scene_data.get('analysis_id', uuid.uuid4())}",
                    case_id=case_id,
                    document_type="scene_analysis",
                    content=self._create_scene_analysis_content(scene_data),
                    metadata={
                        'analysis_id': scene_data.get('analysis_id'),
                        'operator_id': scene_data.get('operator_id'),
                        'total_events': len(scene_data.get('detected_events', [])),
                        'total_evidence': len(scene_data.get('evidence_items', [])),
                        'quality_score': scene_data.get('quality_metrics', {}).get('overall_score', 0.0)
                    },
                    timestamp=datetime.fromisoformat(scene_data.get('timestamp', datetime.now().isoformat())),
                    confidence=scene_data.get('quality_metrics', {}).get('overall_score', 0.0),
                    source_system="holoforensics_scene_analysis",
                    hash_signature=self._calculate_hash(scene_data)
                )
                documents.append(scene_doc)
                
                # Create individual event documents
                for event in scene_data.get('detected_events', [])[:10]:  # Limit to top 10 events
                    event_doc = ForensicDocument(
                        doc_id=f"event_{event.get('event_id', uuid.uuid4())}",
                        case_id=case_id,
                        document_type="event",
                        content=self._create_event_content(event),
                        metadata={
                            'event_id': event.get('event_id'),
                            'event_type': event.get('event_type'),
                            'severity': event.get('severity'),
                            'start_time': event.get('start_time'),
                            'end_time': event.get('end_time'),
                            'location': event.get('location'),
                            'involved_objects': event.get('involved_objects', [])
                        },
                        timestamp=datetime.now(),
                        confidence=event.get('confidence', 0.0),
                        source_system="holoforensics_event_detection",
                        hash_signature=self._calculate_hash(event)
                    )
                    documents.append(event_doc)
            
            # Handle physics prediction results
            if 'physics_prediction' in analysis_data:
                physics_data = analysis_data['physics_prediction']
                
                physics_doc = ForensicDocument(
                    doc_id=f"physics_prediction_{physics_data.get('job_id', uuid.uuid4())}",
                    case_id=case_id,
                    document_type="physics_prediction",
                    content=self._create_physics_content(physics_data),
                    metadata={
                        'job_id': physics_data.get('job_id'),
                        'prediction_type': physics_data.get('prediction_type'),
                        'num_trajectories': len(physics_data.get('predicted_trajectories', [])),
                        'confidence_score': physics_data.get('confidence_score', 0.0)
                    },
                    timestamp=datetime.fromisoformat(physics_data.get('timestamp', datetime.now().isoformat())),
                    confidence=physics_data.get('confidence_score', 0.0),
                    source_system="holoforensics_physics_prediction",
                    hash_signature=self._calculate_hash(physics_data)
                )
                documents.append(physics_doc)
            
            # Handle video inpainting results
            if 'video_inpainting' in analysis_data:
                inpainting_data = analysis_data['video_inpainting']
                
                inpainting_doc = ForensicDocument(
                    doc_id=f"video_inpainting_{inpainting_data.get('job_id', uuid.uuid4())}",
                    case_id=case_id,
                    document_type="video_inpainting",
                    content=self._create_inpainting_content(inpainting_data),
                    metadata={
                        'job_id': inpainting_data.get('job_id'),
                        'inpainting_method': inpainting_data.get('method'),
                        'processed_frames': inpainting_data.get('processed_frames', 0),
                        'quality_score': inpainting_data.get('quality_score', 0.0)
                    },
                    timestamp=datetime.fromisoformat(inpainting_data.get('timestamp', datetime.now().isoformat())),
                    confidence=inpainting_data.get('quality_score', 0.0),
                    source_system="holoforensics_video_inpainting",
                    hash_signature=self._calculate_hash(inpainting_data)
                )
                documents.append(inpainting_doc)
            
            # Add documents to vector store
            indexed_count = rag.add_forensic_data(documents)
            
            logger.info(f"Indexed {indexed_count} documents for case {case_id}")
            
            return JsonResponse({
                'success': True,
                'indexed_documents': indexed_count,
                'case_id': case_id,
                'document_types': list(set(doc.document_type for doc in documents)),
                'message': f'Successfully indexed {indexed_count} forensic documents'
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            logger.error(f"Error indexing forensic data: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
    
    def _create_scene_analysis_content(self, scene_data: Dict[str, Any]) -> str:
        """Create content text for scene analysis document"""
        content_parts = [
            f"Scene Analysis Results",
            f"Analysis ID: {scene_data.get('analysis_id')}",
            f"Operator: {scene_data.get('operator_id')}",
            f"Total Events Detected: {len(scene_data.get('detected_events', []))}",
            f"Total Evidence Items: {len(scene_data.get('evidence_items', []))}",
            f"Quality Score: {scene_data.get('quality_metrics', {}).get('overall_score', 0.0):.3f}"
        ]
        
        # Add event summaries
        events = scene_data.get('detected_events', [])
        if events:
            content_parts.append("\nDetected Events:")
            for event in events[:5]:  # Top 5 events
                content_parts.append(f"- {event.get('event_type', 'Unknown')}: {event.get('description', 'No description')}")
        
        return "\n".join(content_parts)
    
    def _create_event_content(self, event: Dict[str, Any]) -> str:
        """Create content text for event document"""
        return f"""Event Type: {event.get('event_type', 'Unknown')}
Severity: {event.get('severity', 'Unknown')}
Description: {event.get('description', 'No description')}
Time Range: {event.get('start_time', 0)}s - {event.get('end_time', 0)}s
Location: {event.get('location', 'Unknown')}
Confidence: {event.get('confidence', 0.0):.3f}
Involved Objects: {', '.join(event.get('involved_objects', []))}"""
    
    def _create_physics_content(self, physics_data: Dict[str, Any]) -> str:
        """Create content text for physics prediction document"""
        content_parts = [
            f"Physics-Informed Trajectory Prediction",
            f"Job ID: {physics_data.get('job_id')}",
            f"Prediction Method: {physics_data.get('prediction_type', 'Kalman + Social Forces')}",
            f"Number of Trajectories: {len(physics_data.get('predicted_trajectories', []))}",
            f"Confidence Score: {physics_data.get('confidence_score', 0.0):.3f}"
        ]
        
        trajectories = physics_data.get('predicted_trajectories', [])
        if trajectories:
            content_parts.append("\nPredicted Trajectories:")
            for i, traj in enumerate(trajectories[:3]):  # Top 3 trajectories
                content_parts.append(f"- Trajectory {i+1}: {len(traj.get('points', []))} points")
        
        return "\n".join(content_parts)
    
    def _create_inpainting_content(self, inpainting_data: Dict[str, Any]) -> str:
        """Create content text for video inpainting document"""
        return f"""Video Inpainting Analysis
Job ID: {inpainting_data.get('job_id')}
Method: {inpainting_data.get('method', 'E2FGVI')}
Processed Frames: {inpainting_data.get('processed_frames', 0)}
Quality Score: {inpainting_data.get('quality_score', 0.0):.3f}
Processing Time: {inpainting_data.get('processing_time', 0.0):.2f}s
Status: {inpainting_data.get('status', 'Unknown')}"""
    
    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate hash signature for data"""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

@method_decorator([login_required], name='dispatch')
class RAGSystemStatsView(View):
    """Get RAG system statistics"""
    
    def get(self, request):
        try:
            rag = get_rag_system()
            stats = rag.get_system_stats()
            
            # Add user-specific stats
            user_history = rag.get_query_history(user_id=request.user.username)
            
            response_data = {
                'success': True,
                'system_stats': stats,
                'user_stats': {
                    'total_queries': len(user_history),
                    'recent_queries': len([q for q, r in user_history 
                                         if (datetime.now() - q.timestamp).days < 7]),
                    'avg_confidence': sum(r.confidence_score for q, r in user_history) / len(user_history) if user_history else 0.0
                }
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Error getting RAG stats: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required], name='dispatch')
class QueryHistoryView(View):
    """Get user's query history"""
    
    def get(self, request):
        try:
            rag = get_rag_system()
            user_history = rag.get_query_history(user_id=request.user.username)
            
            # Format history for response
            history_data = []
            for query, response in user_history[-20:]:  # Last 20 queries
                history_data.append({
                    'query_id': query.query_id,
                    'query_text': query.query_text,
                    'query_type': query.query_type,
                    'case_id': query.case_id,
                    'timestamp': query.timestamp.isoformat(),
                    'response_confidence': response.confidence_score,
                    'processing_time': response.processing_time,
                    'num_results': len(response.retrieved_documents)
                })
            
            return JsonResponse({
                'success': True,
                'query_history': history_data,
                'total_queries': len(user_history)
            })
            
        except Exception as e:
            logger.error(f"Error getting query history: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required, csrf_exempt], name='dispatch')
class SuggestQueriesView(View):
    """Suggest queries based on available data"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            case_id = data.get('case_id')
            
            rag = get_rag_system()
            
            # Get sample documents to generate suggestions
            sample_docs = rag.vector_store.collection.get(
                where={"case_id": case_id} if case_id else None,
                limit=10
            )
            
            suggestions = []
            
            if sample_docs and sample_docs['metadatas']:
                # Analyze available data types
                doc_types = set()
                event_types = set()
                
                for metadata in sample_docs['metadatas']:
                    doc_types.add(metadata.get('document_type', 'unknown'))
                    if metadata.get('event_type'):
                        event_types.add(metadata.get('event_type'))
                
                # Generate suggestions based on available data
                if 'event' in doc_types:
                    suggestions.extend([
                        "What events occurred with high confidence?",
                        "Show me all critical severity events",
                        "Find events that happened in the first 30 seconds"
                    ])
                
                if 'scene_analysis' in doc_types:
                    suggestions.extend([
                        "What is the overall quality of the scene analysis?",
                        "Show me the scene analysis summary",
                        "Find high-confidence scene analysis results"
                    ])
                
                if 'physics_prediction' in doc_types:
                    suggestions.extend([
                        "What trajectory predictions were made?",
                        "Show me physics prediction results",
                        "Find predicted movements with high confidence"
                    ])
                
                if event_types:
                    for event_type in list(event_types)[:3]:
                        suggestions.append(f"Show me all {event_type} events")
            
            # Default suggestions if no specific data
            if not suggestions:
                suggestions = [
                    "What forensic analysis has been performed?",
                    "Show me all available evidence",
                    "Find high-confidence results",
                    "What events were detected?",
                    "Show me the analysis timeline"
                ]
            
            return JsonResponse({
                'success': True,
                'suggested_queries': suggestions[:8],  # Limit to 8 suggestions
                'case_id': case_id
            })
            
        except Exception as e:
            logger.error(f"Error generating query suggestions: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
