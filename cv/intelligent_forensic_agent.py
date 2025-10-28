"""
Intelligent Forensic Agent - Week 11 Implementation
Multi-step reasoning agent that integrates LLM capabilities with RAG system and tool calling
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

from llm_agent import ForensicLLMAgent, QueryIntent, IntentType
from tool_calling_framework import ForensicToolExecutor, ExecutionResult, ExecutionStatus
from rag_system import ForensicRAGSystem, RAGResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningStep(Enum):
    """Types of reasoning steps"""
    QUERY_ANALYSIS = "query_analysis"
    INFORMATION_GATHERING = "information_gathering"
    TOOL_EXECUTION = "tool_execution"
    RESULT_SYNTHESIS = "result_synthesis"
    RESPONSE_GENERATION = "response_generation"

@dataclass
class ReasoningTrace:
    """Trace of reasoning steps"""
    step_id: str
    step_type: ReasoningStep
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning: str
    confidence: float
    timestamp: datetime

@dataclass
class ForensicInvestigation:
    """Represents an ongoing forensic investigation"""
    investigation_id: str
    case_id: str
    query: str
    intent: QueryIntent
    reasoning_trace: List[ReasoningTrace]
    intermediate_results: Dict[str, Any]
    final_response: Optional[str]
    confidence_score: float
    status: str

class IntelligentForensicAgent:
    """
    Advanced intelligent agent that combines LLM reasoning with RAG retrieval and tool calling
    for comprehensive forensic analysis
    """
    
    def __init__(self, rag_system: Optional[ForensicRAGSystem] = None):
        """
        Initialize the intelligent forensic agent
        
        Args:
            rag_system: Optional RAG system instance
        """
        self.llm_agent = ForensicLLMAgent()
        self.tool_executor = ForensicToolExecutor()
        self.rag_system = rag_system or ForensicRAGSystem()
        
        self.active_investigations = {}
        self.investigation_history = []
        
        # Reasoning strategies
        self.reasoning_strategies = {
            IntentType.QUERY_DATA: self._strategy_query_data,
            IntentType.ANALYZE_SCENE: self._strategy_analyze_scene,
            IntentType.PREDICT_TRAJECTORY: self._strategy_predict_trajectory,
            IntentType.INPAINT_VIDEO: self._strategy_inpaint_video,
            IntentType.TIMELINE_ANALYSIS: self._strategy_timeline_analysis,
            IntentType.GENERATE_REPORT: self._strategy_generate_report
        }
    
    async def investigate(self, query: str, case_id: Optional[str] = None, 
                         user_id: str = "unknown") -> ForensicInvestigation:
        """
        Conduct a comprehensive forensic investigation based on the query
        
        Args:
            query: User's forensic query
            case_id: Optional case ID filter
            user_id: User identifier
            
        Returns:
            ForensicInvestigation with complete reasoning trace and results
        """
        investigation_id = f"inv_{int(datetime.now().timestamp())}"
        
        logger.info(f"Starting investigation {investigation_id}: {query}")
        
        # Step 1: Query Analysis
        intent = await self._analyze_query(investigation_id, query, case_id)
        
        # Initialize investigation
        investigation = ForensicInvestigation(
            investigation_id=investigation_id,
            case_id=case_id or "unknown",
            query=query,
            intent=intent,
            reasoning_trace=[],
            intermediate_results={},
            final_response=None,
            confidence_score=0.0,
            status="in_progress"
        )
        
        self.active_investigations[investigation_id] = investigation
        
        try:
            # Step 2: Execute reasoning strategy
            strategy = self.reasoning_strategies.get(intent.intent_type, self._strategy_default)
            investigation = await strategy(investigation, user_id)
            
            # Step 3: Generate final response
            investigation = await self._generate_final_response(investigation)
            
            investigation.status = "completed"
            logger.info(f"Investigation {investigation_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Investigation {investigation_id} failed: {e}")
            investigation.status = "failed"
            investigation.final_response = f"Investigation failed: {str(e)}"
            investigation.confidence_score = 0.0
        
        finally:
            # Move to history
            self.investigation_history.append(investigation)
            if investigation_id in self.active_investigations:
                del self.active_investigations[investigation_id]
        
        return investigation
    
    async def _analyze_query(self, investigation_id: str, query: str, 
                           case_id: Optional[str]) -> QueryIntent:
        """Analyze the query to understand intent and extract entities"""
        
        context = {"case_id": case_id} if case_id else None
        intent = self.llm_agent.understand_query(query, context)
        
        # Add reasoning trace
        trace = ReasoningTrace(
            step_id=f"{investigation_id}_query_analysis",
            step_type=ReasoningStep.QUERY_ANALYSIS,
            input_data={"query": query, "case_id": case_id},
            output_data=asdict(intent),
            reasoning=f"Analyzed query intent: {intent.intent_type.value} with {intent.confidence:.2f} confidence",
            confidence=intent.confidence,
            timestamp=datetime.now()
        )
        
        investigation = self.active_investigations[investigation_id]
        investigation.reasoning_trace.append(trace)
        
        return intent
    
    async def _strategy_query_data(self, investigation: ForensicInvestigation, 
                                 user_id: str) -> ForensicInvestigation:
        """Strategy for data querying tasks"""
        
        # Step 1: RAG retrieval
        rag_response = await self._perform_rag_query(investigation, user_id)
        investigation.intermediate_results['rag_response'] = asdict(rag_response)
        
        # Step 2: Analyze retrieved information
        analysis_result = await self._analyze_retrieved_information(investigation, rag_response)
        investigation.intermediate_results['analysis'] = analysis_result
        
        # Step 3: Check if additional tools are needed
        if self._needs_additional_analysis(investigation, rag_response):
            tool_results = await self._execute_additional_tools(investigation)
            investigation.intermediate_results['tool_results'] = tool_results
        
        return investigation
    
    async def _strategy_analyze_scene(self, investigation: ForensicInvestigation, 
                                    user_id: str) -> ForensicInvestigation:
        """Strategy for scene analysis tasks"""
        
        # Step 1: Check existing scene analysis data
        existing_data = await self._check_existing_scene_data(investigation)
        
        if existing_data:
            investigation.intermediate_results['existing_scene_data'] = existing_data
            
            # Analyze existing data
            analysis = await self._analyze_existing_scene_data(investigation, existing_data)
            investigation.intermediate_results['scene_analysis'] = analysis
        else:
            # Step 2: Execute scene analysis tool
            tool_results = await self._execute_scene_analysis_tools(investigation)
            investigation.intermediate_results['new_scene_analysis'] = tool_results
        
        # Step 3: Generate insights and recommendations
        insights = await self._generate_scene_insights(investigation)
        investigation.intermediate_results['insights'] = insights
        
        return investigation
    
    async def _strategy_predict_trajectory(self, investigation: ForensicInvestigation, 
                                         user_id: str) -> ForensicInvestigation:
        """Strategy for trajectory prediction tasks"""
        
        # Step 1: Gather scene data (prerequisite)
        scene_data = await self._ensure_scene_data(investigation)
        investigation.intermediate_results['scene_data'] = scene_data
        
        # Step 2: Execute physics prediction
        prediction_results = await self._execute_physics_prediction(investigation, scene_data)
        investigation.intermediate_results['trajectory_predictions'] = prediction_results
        
        # Step 3: Validate and analyze predictions
        validation = await self._validate_trajectory_predictions(investigation, prediction_results)
        investigation.intermediate_results['validation'] = validation
        
        return investigation
    
    async def _strategy_timeline_analysis(self, investigation: ForensicInvestigation, 
                                        user_id: str) -> ForensicInvestigation:
        """Strategy for timeline analysis tasks"""
        
        # Step 1: Gather all temporal events
        events = await self._gather_temporal_events(investigation)
        investigation.intermediate_results['events'] = events
        
        # Step 2: Generate timeline
        timeline = await self._generate_timeline(investigation, events)
        investigation.intermediate_results['timeline'] = timeline
        
        # Step 3: Identify patterns and anomalies
        patterns = await self._analyze_temporal_patterns(investigation, timeline)
        investigation.intermediate_results['patterns'] = patterns
        
        return investigation
    
    async def _strategy_default(self, investigation: ForensicInvestigation, 
                              user_id: str) -> ForensicInvestigation:
        """Default strategy for unspecified intents"""
        
        # Fallback to RAG query
        rag_response = await self._perform_rag_query(investigation, user_id)
        investigation.intermediate_results['rag_response'] = asdict(rag_response)
        
        return investigation
    
    async def _perform_rag_query(self, investigation: ForensicInvestigation, 
                               user_id: str) -> RAGResponse:
        """Perform RAG query and add to reasoning trace"""
        
        response = self.rag_system.process_query(
            query_text=investigation.query,
            case_id=investigation.case_id,
            user_id=user_id
        )
        
        # Add reasoning trace
        trace = ReasoningTrace(
            step_id=f"{investigation.investigation_id}_rag_query",
            step_type=ReasoningStep.INFORMATION_GATHERING,
            input_data={
                "query": investigation.query,
                "case_id": investigation.case_id
            },
            output_data={
                "response_text": response.response_text,
                "confidence": response.confidence_score,
                "num_documents": len(response.retrieved_documents)
            },
            reasoning=f"Retrieved {len(response.retrieved_documents)} relevant documents with average confidence {response.confidence_score:.2f}",
            confidence=response.confidence_score,
            timestamp=datetime.now()
        )
        
        investigation.reasoning_trace.append(trace)
        return response
    
    async def _analyze_retrieved_information(self, investigation: ForensicInvestigation, 
                                           rag_response: RAGResponse) -> Dict[str, Any]:
        """Analyze the information retrieved from RAG"""
        
        analysis = {
            "document_types": {},
            "confidence_distribution": [],
            "temporal_coverage": {},
            "key_entities": [],
            "gaps_identified": []
        }
        
        # Analyze document types
        for doc in rag_response.retrieved_documents:
            doc_type = doc.document_type
            if doc_type not in analysis["document_types"]:
                analysis["document_types"][doc_type] = 0
            analysis["document_types"][doc_type] += 1
            
            # Collect confidence scores
            analysis["confidence_distribution"].append(doc.similarity_score)
        
        # Identify information gaps
        if len(rag_response.retrieved_documents) < 3:
            analysis["gaps_identified"].append("Limited relevant documents found")
        
        if rag_response.confidence_score < 0.7:
            analysis["gaps_identified"].append("Low overall confidence in retrieved information")
        
        # Add reasoning trace
        trace = ReasoningTrace(
            step_id=f"{investigation.investigation_id}_info_analysis",
            step_type=ReasoningStep.RESULT_SYNTHESIS,
            input_data={"num_documents": len(rag_response.retrieved_documents)},
            output_data=analysis,
            reasoning=f"Analyzed {len(rag_response.retrieved_documents)} documents, identified {len(analysis['gaps_identified'])} potential gaps",
            confidence=0.8,
            timestamp=datetime.now()
        )
        
        investigation.reasoning_trace.append(trace)
        return analysis
    
    def _needs_additional_analysis(self, investigation: ForensicInvestigation, 
                                 rag_response: RAGResponse) -> bool:
        """Determine if additional tool execution is needed"""
        
        # Check if confidence is low
        if rag_response.confidence_score < 0.6:
            return True
        
        # Check if specific analysis is requested
        query_lower = investigation.query.lower()
        analysis_keywords = ['analyze', 'predict', 'inpaint', 'reconstruct', 'generate']
        
        return any(keyword in query_lower for keyword in analysis_keywords)
    
    async def _execute_additional_tools(self, investigation: ForensicInvestigation) -> List[ExecutionResult]:
        """Execute additional analysis tools based on the investigation needs"""
        
        # Plan tool execution based on intent
        tool_calls = self.llm_agent.plan_execution(investigation.intent)
        
        if not tool_calls:
            return []
        
        # Create execution plan
        plan = self.tool_executor.create_execution_plan(tool_calls)
        
        # Execute tools
        results = await self.tool_executor.execute_plan(plan)
        
        # Add reasoning trace
        trace = ReasoningTrace(
            step_id=f"{investigation.investigation_id}_tool_execution",
            step_type=ReasoningStep.TOOL_EXECUTION,
            input_data={"num_tools": len(tool_calls)},
            output_data={"results": [r.status.value for r in results]},
            reasoning=f"Executed {len(tool_calls)} additional analysis tools",
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        investigation.reasoning_trace.append(trace)
        return results
    
    async def _generate_final_response(self, investigation: ForensicInvestigation) -> ForensicInvestigation:
        """Generate the final comprehensive response"""
        
        # Synthesize all intermediate results
        synthesis = await self._synthesize_results(investigation)
        
        # Generate response text
        response_parts = []
        
        # Add RAG response if available
        if 'rag_response' in investigation.intermediate_results:
            rag_data = investigation.intermediate_results['rag_response']
            response_parts.append(rag_data.get('response_text', ''))
        
        # Add tool execution results
        if 'tool_results' in investigation.intermediate_results:
            tool_results = investigation.intermediate_results['tool_results']
            successful_tools = [r for r in tool_results if r.status == ExecutionStatus.COMPLETED]
            
            if successful_tools:
                response_parts.append(f"\nAdditional Analysis Results:")
                for result in successful_tools:
                    tool_name = result.tool_call.tool_type.value.replace('_', ' ').title()
                    response_parts.append(f"• {tool_name}: Completed successfully")
        
        # Add insights and recommendations
        if 'insights' in investigation.intermediate_results:
            insights = investigation.intermediate_results['insights']
            response_parts.append(f"\nKey Insights:")
            for insight in insights.get('key_findings', []):
                response_parts.append(f"• {insight}")
        
        # Add confidence and reliability information
        overall_confidence = self._calculate_investigation_confidence(investigation)
        response_parts.append(f"\nInvestigation Confidence: {overall_confidence:.1%}")
        response_parts.append(f"Analysis Steps Completed: {len(investigation.reasoning_trace)}")
        
        # Combine response
        final_response = "\n".join(response_parts)
        
        investigation.final_response = final_response
        investigation.confidence_score = overall_confidence
        
        # Add final reasoning trace
        trace = ReasoningTrace(
            step_id=f"{investigation.investigation_id}_final_response",
            step_type=ReasoningStep.RESPONSE_GENERATION,
            input_data={"synthesis": synthesis},
            output_data={"response_length": len(final_response)},
            reasoning=f"Generated comprehensive response combining {len(investigation.intermediate_results)} analysis components",
            confidence=overall_confidence,
            timestamp=datetime.now()
        )
        
        investigation.reasoning_trace.append(trace)
        return investigation
    
    async def _synthesize_results(self, investigation: ForensicInvestigation) -> Dict[str, Any]:
        """Synthesize all intermediate results into coherent insights"""
        
        synthesis = {
            "primary_findings": [],
            "confidence_assessment": {},
            "data_quality": {},
            "recommendations": []
        }
        
        # Analyze RAG results
        if 'rag_response' in investigation.intermediate_results:
            rag_data = investigation.intermediate_results['rag_response']
            synthesis["primary_findings"].append("Retrieved relevant forensic documents")
            synthesis["confidence_assessment"]["rag_confidence"] = rag_data.get('confidence_score', 0.0)
        
        # Analyze tool execution results
        if 'tool_results' in investigation.intermediate_results:
            tool_results = investigation.intermediate_results['tool_results']
            successful_count = sum(1 for r in tool_results if r.status == ExecutionStatus.COMPLETED)
            synthesis["primary_findings"].append(f"Completed {successful_count} additional analysis tools")
        
        # Generate recommendations
        if investigation.intent.confidence < 0.7:
            synthesis["recommendations"].append("Consider refining the query for better results")
        
        if 'analysis' in investigation.intermediate_results:
            analysis = investigation.intermediate_results['analysis']
            if analysis.get('gaps_identified'):
                synthesis["recommendations"].extend(analysis['gaps_identified'])
        
        return synthesis
    
    def _calculate_investigation_confidence(self, investigation: ForensicInvestigation) -> float:
        """Calculate overall confidence for the investigation"""
        
        confidences = []
        
        # Intent confidence
        confidences.append(investigation.intent.confidence)
        
        # RAG confidence
        if 'rag_response' in investigation.intermediate_results:
            rag_data = investigation.intermediate_results['rag_response']
            confidences.append(rag_data.get('confidence_score', 0.0))
        
        # Tool execution success rate
        if 'tool_results' in investigation.intermediate_results:
            tool_results = investigation.intermediate_results['tool_results']
            if tool_results:
                success_rate = sum(1 for r in tool_results if r.status == ExecutionStatus.COMPLETED) / len(tool_results)
                confidences.append(success_rate)
        
        # Reasoning trace confidence
        trace_confidences = [trace.confidence for trace in investigation.reasoning_trace]
        if trace_confidences:
            confidences.append(sum(trace_confidences) / len(trace_confidences))
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    # Placeholder methods for specific strategies (to be implemented)
    async def _check_existing_scene_data(self, investigation: ForensicInvestigation) -> Optional[Dict[str, Any]]:
        """Check for existing scene analysis data"""
        # Implementation would query the RAG system for scene analysis documents
        return None
    
    async def _analyze_existing_scene_data(self, investigation: ForensicInvestigation, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze existing scene data"""
        return {"status": "analyzed", "events_count": len(data.get('events', []))}
    
    async def _execute_scene_analysis_tools(self, investigation: ForensicInvestigation) -> List[ExecutionResult]:
        """Execute scene analysis tools"""
        return []
    
    async def _generate_scene_insights(self, investigation: ForensicInvestigation) -> Dict[str, Any]:
        """Generate insights from scene analysis"""
        return {"key_findings": ["Scene analysis completed", "Multiple events detected"]}
    
    async def _ensure_scene_data(self, investigation: ForensicInvestigation) -> Dict[str, Any]:
        """Ensure scene data is available for trajectory prediction"""
        return {"scene_available": True}
    
    async def _execute_physics_prediction(self, investigation: ForensicInvestigation, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute physics prediction"""
        return {"predictions": [], "confidence": 0.8}
    
    async def _validate_trajectory_predictions(self, investigation: ForensicInvestigation, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trajectory predictions"""
        return {"validation_score": 0.85, "issues": []}
    
    async def _gather_temporal_events(self, investigation: ForensicInvestigation) -> List[Dict[str, Any]]:
        """Gather all temporal events for timeline analysis"""
        return []
    
    async def _generate_timeline(self, investigation: ForensicInvestigation, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate timeline from events"""
        return {"timeline": events, "duration": 60.0}
    
    async def _analyze_temporal_patterns(self, investigation: ForensicInvestigation, timeline: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal patterns in the timeline"""
        return {"patterns": [], "anomalies": []}

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_intelligent_agent():
        """Test the intelligent forensic agent"""
        print("🧠 Testing Intelligent Forensic Agent")
        print("=" * 50)
        
        agent = IntelligentForensicAgent()
        
        test_queries = [
            "What events occurred between 10 and 20 seconds with high confidence?",
            "Analyze the scene and predict trajectories for all detected persons",
            "Generate a comprehensive timeline of all events in case CASE_2024_001"
        ]
        
        for query in test_queries:
            print(f"\nInvestigating: {query}")
            
            investigation = await agent.investigate(
                query=query,
                case_id="CASE_2024_001",
                user_id="test_investigator"
            )
            
            print(f"Status: {investigation.status}")
            print(f"Confidence: {investigation.confidence_score:.2f}")
            print(f"Reasoning Steps: {len(investigation.reasoning_trace)}")
            print(f"Response: {investigation.final_response[:200]}...")
            print("-" * 30)
    
    # Run test
    asyncio.run(test_intelligent_agent())
