"""
Django REST API views for LLM Agent Integration
Advanced query processing with tool calling and multi-step reasoning
"""

import json
import os
import asyncio
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

# Import our LLM agent modules
import sys
sys.path.append('/Users/anuragsamajpati/Desktop/holoforensics/cv')

from intelligent_forensic_agent import IntelligentForensicAgent, ForensicInvestigation
from rag_system import ForensicRAGSystem
from vector_store import ForensicVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global agent instance
intelligent_agent = None

def get_intelligent_agent():
    """Get or initialize the global intelligent agent"""
    global intelligent_agent
    if intelligent_agent is None:
        try:
            # Initialize with persistent storage
            vector_store_path = os.path.join(settings.MEDIA_ROOT, 'vector_store')
            os.makedirs(vector_store_path, exist_ok=True)
            
            vector_store = ForensicVectorStore(persist_directory=vector_store_path)
            rag_system = ForensicRAGSystem(vector_store=vector_store)
            intelligent_agent = IntelligentForensicAgent(rag_system=rag_system)
            
            logger.info("Initialized Intelligent Forensic Agent")
        except Exception as e:
            logger.error(f"Failed to initialize intelligent agent: {e}")
            raise e
    
    return intelligent_agent

@method_decorator([login_required, csrf_exempt], name='dispatch')
class IntelligentQueryView(View):
    """Process queries using the intelligent forensic agent"""
    
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
            enable_tools = data.get('enable_tools', True)
            reasoning_depth = data.get('reasoning_depth', 'standard')  # minimal, standard, deep
            
            # Get intelligent agent
            agent = get_intelligent_agent()
            
            # Process query with intelligent reasoning
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                investigation = loop.run_until_complete(
                    agent.investigate(
                        query=query_text,
                        case_id=case_id,
                        user_id=request.user.username
                    )
                )
            finally:
                loop.close()
            
            # Format response
            response_data = {
                'success': True,
                'investigation_id': investigation.investigation_id,
                'query': investigation.query,
                'intent': {
                    'type': investigation.intent.intent_type.value,
                    'confidence': investigation.intent.confidence,
                    'entities': investigation.intent.entities,
                    'required_tools': [tool.value for tool in investigation.intent.required_tools]
                },
                'response_text': investigation.final_response,
                'confidence_score': investigation.confidence_score,
                'status': investigation.status,
                'reasoning_trace': [
                    {
                        'step_id': trace.step_id,
                        'step_type': trace.step_type.value,
                        'reasoning': trace.reasoning,
                        'confidence': trace.confidence,
                        'timestamp': trace.timestamp.isoformat()
                    }
                    for trace in investigation.reasoning_trace
                ] if reasoning_depth in ['standard', 'deep'] else [],
                'intermediate_results': investigation.intermediate_results if reasoning_depth == 'deep' else {},
                'metadata': {
                    'reasoning_steps': len(investigation.reasoning_trace),
                    'tools_executed': len([r for r in investigation.intermediate_results.get('tool_results', [])]),
                    'processing_time': sum(trace.confidence for trace in investigation.reasoning_trace) / len(investigation.reasoning_trace) if investigation.reasoning_trace else 0
                }
            }
            
            logger.info(f"Processed intelligent query for user {request.user.username}")
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            }, status=400)
        except Exception as e:
            logger.error(f"Error processing intelligent query: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required], name='dispatch')
class InvestigationStatusView(View):
    """Get status of an ongoing investigation"""
    
    def get(self, request, investigation_id):
        try:
            agent = get_intelligent_agent()
            
            # Check active investigations
            if investigation_id in agent.active_investigations:
                investigation = agent.active_investigations[investigation_id]
                status = "in_progress"
            else:
                # Check history
                investigation = None
                for hist_inv in agent.investigation_history:
                    if hist_inv.investigation_id == investigation_id:
                        investigation = hist_inv
                        break
                
                if not investigation:
                    return JsonResponse({
                        'success': False,
                        'error': 'Investigation not found'
                    }, status=404)
                
                status = investigation.status
            
            response_data = {
                'success': True,
                'investigation_id': investigation_id,
                'status': status,
                'progress': {
                    'reasoning_steps': len(investigation.reasoning_trace),
                    'current_step': investigation.reasoning_trace[-1].step_type.value if investigation.reasoning_trace else 'initializing',
                    'confidence': investigation.confidence_score
                },
                'estimated_completion': None  # Could be calculated based on remaining steps
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            logger.error(f"Error getting investigation status: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required], name='dispatch')
class InvestigationHistoryView(View):
    """Get user's investigation history"""
    
    def get(self, request):
        try:
            agent = get_intelligent_agent()
            
            # Filter by user
            user_investigations = [
                inv for inv in agent.investigation_history
                if any(trace.input_data.get('user_id') == request.user.username 
                      for trace in inv.reasoning_trace)
            ]
            
            # Format history for response
            history_data = []
            for investigation in user_investigations[-20:]:  # Last 20 investigations
                history_data.append({
                    'investigation_id': investigation.investigation_id,
                    'query': investigation.query,
                    'case_id': investigation.case_id,
                    'intent_type': investigation.intent.intent_type.value,
                    'status': investigation.status,
                    'confidence': investigation.confidence_score,
                    'reasoning_steps': len(investigation.reasoning_trace),
                    'timestamp': investigation.reasoning_trace[0].timestamp.isoformat() if investigation.reasoning_trace else None
                })
            
            return JsonResponse({
                'success': True,
                'investigation_history': history_data,
                'total_investigations': len(user_investigations)
            })
            
        except Exception as e:
            logger.error(f"Error getting investigation history: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required, csrf_exempt], name='dispatch')
class AgentCapabilitiesView(View):
    """Get information about agent capabilities and available tools"""
    
    def get(self, request):
        try:
            agent = get_intelligent_agent()
            
            capabilities = {
                'intent_types': [
                    {
                        'type': 'query_data',
                        'description': 'Query and retrieve forensic data using natural language',
                        'examples': ['What events occurred?', 'Show me evidence from the scene']
                    },
                    {
                        'type': 'analyze_scene',
                        'description': 'Perform comprehensive scene analysis and event detection',
                        'examples': ['Analyze the scene', 'Detect all events and objects']
                    },
                    {
                        'type': 'predict_trajectory',
                        'description': 'Predict object trajectories using physics-informed models',
                        'examples': ['Predict where person_001 will go', 'Show trajectory predictions']
                    },
                    {
                        'type': 'timeline_analysis',
                        'description': 'Generate chronological timelines and temporal analysis',
                        'examples': ['Create a timeline', 'Show sequence of events']
                    },
                    {
                        'type': 'inpaint_video',
                        'description': 'Reconstruct missing or corrupted video segments',
                        'examples': ['Inpaint the video', 'Reconstruct missing frames']
                    },
                    {
                        'type': 'generate_report',
                        'description': 'Generate comprehensive forensic analysis reports',
                        'examples': ['Generate a report', 'Summarize findings']
                    }
                ],
                'available_tools': [
                    {
                        'name': 'Scene Analysis',
                        'description': 'Analyze video scenes to detect events and generate scene graphs',
                        'estimated_time': '30-60 seconds'
                    },
                    {
                        'name': 'Physics Prediction',
                        'description': 'Predict object trajectories using Kalman filters and social forces',
                        'estimated_time': '15-30 seconds'
                    },
                    {
                        'name': 'Video Inpainting',
                        'description': 'Reconstruct missing video segments using E2FGVI',
                        'estimated_time': '1-3 minutes'
                    },
                    {
                        'name': 'RAG Query',
                        'description': 'Query forensic database using semantic search',
                        'estimated_time': '1-3 seconds'
                    },
                    {
                        'name': 'Timeline Generator',
                        'description': 'Generate chronological event timelines',
                        'estimated_time': '5-10 seconds'
                    },
                    {
                        'name': 'Report Generator',
                        'description': 'Generate comprehensive forensic reports',
                        'estimated_time': '10-20 seconds'
                    }
                ],
                'reasoning_capabilities': [
                    'Multi-step logical reasoning',
                    'Entity extraction and relationship analysis',
                    'Temporal and spatial reasoning',
                    'Confidence assessment and uncertainty quantification',
                    'Tool selection and execution planning',
                    'Result synthesis and interpretation'
                ],
                'supported_entities': [
                    'Time ranges and temporal references',
                    'Spatial coordinates and locations',
                    'Object identifiers (persons, vehicles, objects)',
                    'Confidence thresholds and quality metrics',
                    'Case IDs and evidence references',
                    'Event types and severity levels'
                ]
            }
            
            return JsonResponse({
                'success': True,
                'capabilities': capabilities,
                'agent_version': '1.0.0',
                'last_updated': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting agent capabilities: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required, csrf_exempt], name='dispatch')
class QuerySuggestionsView(View):
    """Get intelligent query suggestions based on available data and context"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            case_id = data.get('case_id')
            context = data.get('context', {})
            
            agent = get_intelligent_agent()
            
            # Get basic suggestions from RAG system
            rag_response = agent.rag_system.vector_store.collection.get(
                where={"case_id": case_id} if case_id else None,
                limit=10
            )
            
            suggestions = []
            
            if rag_response and rag_response['metadatas']:
                # Analyze available data types
                doc_types = set()
                event_types = set()
                has_temporal_data = False
                
                for metadata in rag_response['metadatas']:
                    doc_types.add(metadata.get('document_type', 'unknown'))
                    if metadata.get('event_type'):
                        event_types.add(metadata.get('event_type'))
                    if metadata.get('start_time') or metadata.get('timestamp'):
                        has_temporal_data = True
                
                # Generate intelligent suggestions based on available data
                if 'scene_analysis' in doc_types:
                    suggestions.extend([
                        "What events were detected in the scene analysis?",
                        "Show me the scene graph and object relationships",
                        "Analyze the quality and confidence of scene detection"
                    ])
                
                if 'event' in doc_types:
                    suggestions.extend([
                        "What are the highest confidence events?",
                        "Show me all critical severity events",
                        "Find events involving specific objects or persons"
                    ])
                
                if has_temporal_data:
                    suggestions.extend([
                        "Generate a chronological timeline of all events",
                        "What happened in the first 30 seconds?",
                        "Show me events between specific time ranges"
                    ])
                
                if 'physics_prediction' in doc_types:
                    suggestions.extend([
                        "What trajectory predictions were made?",
                        "Show me physics-based movement analysis",
                        "Predict future positions of detected objects"
                    ])
                
                if event_types:
                    for event_type in list(event_types)[:3]:
                        suggestions.append(f"Analyze all {event_type} events in detail")
                
                # Add advanced reasoning suggestions
                suggestions.extend([
                    "Perform comprehensive scene analysis and generate insights",
                    "Identify patterns and anomalies in the forensic data",
                    "Generate a detailed forensic analysis report"
                ])
            
            # Default suggestions if no specific data
            if not suggestions:
                suggestions = [
                    "What forensic analysis has been performed?",
                    "Show me all available evidence and findings",
                    "Analyze the scene and detect all events",
                    "Generate a comprehensive timeline of activities",
                    "What are the highest confidence results?",
                    "Perform multi-step forensic investigation"
                ]
            
            return JsonResponse({
                'success': True,
                'suggestions': suggestions[:8],  # Limit to 8 suggestions
                'case_id': case_id,
                'suggestion_context': {
                    'available_doc_types': list(doc_types) if 'doc_types' in locals() else [],
                    'has_temporal_data': has_temporal_data if 'has_temporal_data' in locals() else False,
                    'num_documents': len(rag_response['metadatas']) if rag_response and rag_response['metadatas'] else 0
                }
            })
            
        except Exception as e:
            logger.error(f"Error generating intelligent suggestions: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)

@method_decorator([login_required], name='dispatch')
class AgentStatsView(View):
    """Get comprehensive agent statistics and performance metrics"""
    
    def get(self, request):
        try:
            agent = get_intelligent_agent()
            
            # Calculate statistics
            total_investigations = len(agent.investigation_history)
            user_investigations = [
                inv for inv in agent.investigation_history
                if any(trace.input_data.get('user_id') == request.user.username 
                      for trace in inv.reasoning_trace)
            ]
            
            # Performance metrics
            successful_investigations = [inv for inv in agent.investigation_history if inv.status == 'completed']
            avg_confidence = sum(inv.confidence_score for inv in successful_investigations) / len(successful_investigations) if successful_investigations else 0
            
            # Intent distribution
            intent_distribution = {}
            for inv in agent.investigation_history:
                intent_type = inv.intent.intent_type.value
                intent_distribution[intent_type] = intent_distribution.get(intent_type, 0) + 1
            
            # Tool usage statistics
            tool_usage = {}
            for inv in agent.investigation_history:
                for tool_type in inv.intent.required_tools:
                    tool_name = tool_type.value
                    tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1
            
            stats = {
                'system_stats': {
                    'total_investigations': total_investigations,
                    'successful_investigations': len(successful_investigations),
                    'success_rate': len(successful_investigations) / total_investigations if total_investigations > 0 else 0,
                    'average_confidence': avg_confidence,
                    'active_investigations': len(agent.active_investigations)
                },
                'user_stats': {
                    'user_investigations': len(user_investigations),
                    'user_success_rate': len([inv for inv in user_investigations if inv.status == 'completed']) / len(user_investigations) if user_investigations else 0,
                    'avg_reasoning_steps': sum(len(inv.reasoning_trace) for inv in user_investigations) / len(user_investigations) if user_investigations else 0
                },
                'intent_distribution': intent_distribution,
                'tool_usage': tool_usage,
                'performance_metrics': {
                    'avg_reasoning_steps': sum(len(inv.reasoning_trace) for inv in agent.investigation_history) / total_investigations if total_investigations > 0 else 0,
                    'most_common_intent': max(intent_distribution.keys(), key=intent_distribution.get) if intent_distribution else None,
                    'most_used_tool': max(tool_usage.keys(), key=tool_usage.get) if tool_usage else None
                }
            }
            
            return JsonResponse({
                'success': True,
                'agent_stats': stats,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error getting agent stats: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            }, status=500)
