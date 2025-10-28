"""
Tool Calling Framework for HoloForensics LLM Agent
Executes forensic analysis tools based on LLM agent decisions
"""

import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import traceback

from llm_agent import ToolType, ToolCall, AgentResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Status of tool execution"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class ExecutionResult:
    """Result of a tool execution"""
    tool_call: ToolCall
    status: ExecutionStatus
    output: Any
    error: Optional[str]
    execution_time: float
    timestamp: datetime

@dataclass
class ExecutionPlan:
    """Plan for executing multiple tools"""
    plan_id: str
    tool_calls: List[ToolCall]
    dependencies: Dict[str, List[str]]
    execution_order: List[int]
    estimated_time: float

class ForensicToolExecutor:
    """
    Executes forensic analysis tools based on LLM agent decisions
    """
    
    def __init__(self):
        self.tool_functions = self._register_tool_functions()
        self.execution_history = []
        self.active_executions = {}
        
    def _register_tool_functions(self) -> Dict[ToolType, Callable]:
        """Register available tool functions"""
        return {
            ToolType.SCENE_ANALYSIS: self._execute_scene_analysis,
            ToolType.PHYSICS_PREDICTION: self._execute_physics_prediction,
            ToolType.VIDEO_INPAINTING: self._execute_video_inpainting,
            ToolType.RAG_QUERY: self._execute_rag_query,
            ToolType.EVIDENCE_SEARCH: self._execute_evidence_search,
            ToolType.TIMELINE_GENERATOR: self._execute_timeline_generator,
            ToolType.REPORT_GENERATOR: self._execute_report_generator
        }
    
    def create_execution_plan(self, tool_calls: List[ToolCall]) -> ExecutionPlan:
        """
        Create an execution plan for the tool calls
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            ExecutionPlan with optimized execution order
        """
        plan_id = f"plan_{int(datetime.now().timestamp())}"
        
        # Build dependency graph
        dependencies = {}
        for i, tool_call in enumerate(tool_calls):
            dependencies[str(i)] = []
            for dep in tool_call.dependencies:
                # Find tools that provide the required dependency
                for j, other_call in enumerate(tool_calls):
                    if other_call.expected_output == dep or other_call.tool_type.value == dep:
                        dependencies[str(i)].append(str(j))
        
        # Calculate execution order using topological sort
        execution_order = self._topological_sort(dependencies)
        
        # Estimate execution time
        estimated_time = sum(self._estimate_tool_time(tc.tool_type) for tc in tool_calls)
        
        return ExecutionPlan(
            plan_id=plan_id,
            tool_calls=tool_calls,
            dependencies=dependencies,
            execution_order=execution_order,
            estimated_time=estimated_time
        )
    
    def _topological_sort(self, dependencies: Dict[str, List[str]]) -> List[int]:
        """Perform topological sort to determine execution order"""
        # Simple topological sort implementation
        in_degree = {node: 0 for node in dependencies}
        for node in dependencies:
            for dep in dependencies[node]:
                if dep in in_degree:
                    in_degree[node] += 1
        
        queue = [int(node) for node, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node)
            
            # Update in-degrees
            for other_node in dependencies:
                if str(node) in dependencies[other_node]:
                    in_degree[other_node] -= 1
                    if in_degree[other_node] == 0:
                        queue.append(int(other_node))
        
        return result
    
    def _estimate_tool_time(self, tool_type: ToolType) -> float:
        """Estimate execution time for a tool"""
        time_estimates = {
            ToolType.RAG_QUERY: 2.0,
            ToolType.EVIDENCE_SEARCH: 3.0,
            ToolType.SCENE_ANALYSIS: 30.0,
            ToolType.PHYSICS_PREDICTION: 15.0,
            ToolType.VIDEO_INPAINTING: 60.0,
            ToolType.TIMELINE_GENERATOR: 5.0,
            ToolType.REPORT_GENERATOR: 10.0
        }
        return time_estimates.get(tool_type, 10.0)
    
    async def execute_plan(self, plan: ExecutionPlan, 
                          progress_callback: Optional[Callable] = None) -> List[ExecutionResult]:
        """
        Execute the tool execution plan
        
        Args:
            plan: ExecutionPlan to execute
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of execution results
        """
        results = []
        completed_tools = set()
        
        logger.info(f"Starting execution of plan {plan.plan_id} with {len(plan.tool_calls)} tools")
        
        for step, tool_index in enumerate(plan.execution_order):
            tool_call = plan.tool_calls[tool_index]
            
            # Check dependencies
            dependencies_met = all(
                dep_index in completed_tools 
                for dep_index in [int(d) for d in plan.dependencies.get(str(tool_index), [])]
            )
            
            if not dependencies_met:
                logger.warning(f"Dependencies not met for tool {tool_call.tool_type.value}")
                result = ExecutionResult(
                    tool_call=tool_call,
                    status=ExecutionStatus.SKIPPED,
                    output=None,
                    error="Dependencies not met",
                    execution_time=0.0,
                    timestamp=datetime.now()
                )
                results.append(result)
                continue
            
            # Execute tool
            if progress_callback:
                progress_callback(step + 1, len(plan.execution_order), tool_call.tool_type.value)
            
            result = await self._execute_tool(tool_call, results)
            results.append(result)
            
            if result.status == ExecutionStatus.COMPLETED:
                completed_tools.add(tool_index)
            
            logger.info(f"Completed tool {tool_call.tool_type.value}: {result.status.value}")
        
        self.execution_history.append({
            'plan_id': plan.plan_id,
            'timestamp': datetime.now(),
            'results': results
        })
        
        return results
    
    async def _execute_tool(self, tool_call: ToolCall, previous_results: List[ExecutionResult]) -> ExecutionResult:
        """Execute a single tool"""
        start_time = datetime.now()
        
        try:
            # Get tool function
            tool_function = self.tool_functions.get(tool_call.tool_type)
            if not tool_function:
                raise ValueError(f"Tool {tool_call.tool_type.value} not registered")
            
            # Prepare parameters with results from previous tools
            enhanced_params = self._enhance_parameters(tool_call.parameters, previous_results)
            
            # Execute tool
            logger.info(f"Executing {tool_call.tool_type.value} with parameters: {list(enhanced_params.keys())}")
            output = await tool_function(enhanced_params)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ExecutionResult(
                tool_call=tool_call,
                status=ExecutionStatus.COMPLETED,
                output=output,
                error=None,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Tool execution failed: {str(e)}"
            logger.error(f"Error executing {tool_call.tool_type.value}: {error_msg}")
            logger.error(traceback.format_exc())
            
            return ExecutionResult(
                tool_call=tool_call,
                status=ExecutionStatus.FAILED,
                output=None,
                error=error_msg,
                execution_time=execution_time,
                timestamp=datetime.now()
            )
    
    def _enhance_parameters(self, parameters: Dict[str, Any], 
                           previous_results: List[ExecutionResult]) -> Dict[str, Any]:
        """Enhance parameters with outputs from previous tool executions"""
        enhanced = parameters.copy()
        
        # Add outputs from previous successful executions
        for result in previous_results:
            if result.status == ExecutionStatus.COMPLETED and result.output:
                tool_name = result.tool_call.tool_type.value
                enhanced[f"{tool_name}_output"] = result.output
                
                # Add specific data based on tool type
                if result.tool_call.tool_type == ToolType.SCENE_ANALYSIS:
                    if isinstance(result.output, dict):
                        enhanced['scene_data'] = result.output
                        enhanced['detected_events'] = result.output.get('detected_events', [])
                        
                elif result.tool_call.tool_type == ToolType.RAG_QUERY:
                    if isinstance(result.output, dict):
                        enhanced['query_results'] = result.output
                        enhanced['retrieved_documents'] = result.output.get('retrieved_documents', [])
        
        return enhanced
    
    # Tool execution functions
    async def _execute_scene_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute scene analysis tool"""
        # Simulate scene analysis execution
        # In real implementation, this would call the actual scene analysis API
        
        scene_data = parameters.get('scene_data', {})
        case_id = parameters.get('case_id', 'unknown')
        
        # Mock scene analysis results
        result = {
            'analysis_id': f"SA_{int(datetime.now().timestamp())}",
            'case_id': case_id,
            'status': 'completed',
            'detected_events': [
                {
                    'event_id': 'evt_001',
                    'event_type': 'person_entry',
                    'start_time': 12.5,
                    'end_time': 13.2,
                    'confidence': 0.91,
                    'location': 'north_entrance'
                },
                {
                    'event_id': 'evt_002',
                    'event_type': 'vehicle_movement',
                    'start_time': 18.2,
                    'end_time': 22.8,
                    'confidence': 0.95,
                    'location': 'main_intersection'
                }
            ],
            'scene_graph': {
                'nodes': ['person_001', 'vehicle_001', 'intersection'],
                'edges': [
                    {'from': 'person_001', 'to': 'intersection', 'relation': 'enters'},
                    {'from': 'vehicle_001', 'to': 'intersection', 'relation': 'passes_through'}
                ]
            },
            'quality_metrics': {
                'overall_score': 0.89,
                'detection_confidence': 0.92,
                'temporal_consistency': 0.87
            }
        }
        
        logger.info(f"Scene analysis completed: {len(result['detected_events'])} events detected")
        return result
    
    async def _execute_physics_prediction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute physics prediction tool"""
        scene_data = parameters.get('scene_data', {})
        time_horizon = parameters.get('time_horizon', [0, 60])
        
        # Mock physics prediction results
        result = {
            'prediction_id': f"PP_{int(datetime.now().timestamp())}",
            'method': 'kalman_social_forces',
            'time_horizon': time_horizon,
            'predicted_trajectories': [
                {
                    'object_id': 'person_001',
                    'trajectory_points': [
                        {'time': 30.0, 'x': 150, 'y': 200, 'confidence': 0.85},
                        {'time': 35.0, 'x': 160, 'y': 210, 'confidence': 0.82},
                        {'time': 40.0, 'x': 170, 'y': 220, 'confidence': 0.78}
                    ]
                }
            ],
            'confidence_score': 0.84,
            'uncertainty_bounds': {
                'position_std': 5.2,
                'velocity_std': 1.8
            }
        }
        
        logger.info(f"Physics prediction completed: {len(result['predicted_trajectories'])} trajectories")
        return result
    
    async def _execute_video_inpainting(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute video inpainting tool"""
        method = parameters.get('method', 'E2FGVI')
        quality_threshold = parameters.get('quality_threshold', 0.8)
        
        # Mock video inpainting results
        result = {
            'job_id': f"VI_{int(datetime.now().timestamp())}",
            'method': method,
            'status': 'completed',
            'processed_frames': 1250,
            'inpainted_frames': 45,
            'quality_score': 0.92,
            'processing_time': 45.7,
            'output_files': [
                'inpainted_video_cam1.mp4',
                'inpainted_video_cam2.mp4'
            ]
        }
        
        logger.info(f"Video inpainting completed: {result['inpainted_frames']} frames inpainted")
        return result
    
    async def _execute_rag_query(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG query tool"""
        query_text = parameters.get('query_text', '')
        case_id = parameters.get('case_id')
        
        # Mock RAG query results
        result = {
            'query_id': f"RQ_{int(datetime.now().timestamp())}",
            'query_text': query_text,
            'response_text': f"Based on the forensic analysis, I found relevant information about your query: '{query_text}'. The analysis shows multiple events with high confidence scores.",
            'confidence_score': 0.87,
            'retrieved_documents': [
                {
                    'doc_id': 'scene_analysis_001',
                    'document_type': 'scene_analysis',
                    'similarity_score': 0.92,
                    'content_preview': 'Scene analysis results showing person entry at 12.5s...'
                },
                {
                    'doc_id': 'event_002',
                    'document_type': 'event',
                    'similarity_score': 0.88,
                    'content_preview': 'Vehicle movement detected at main intersection...'
                }
            ],
            'sources': ['scene_analysis:SA_001', 'event:evt_002']
        }
        
        logger.info(f"RAG query completed: {len(result['retrieved_documents'])} documents retrieved")
        return result
    
    async def _execute_evidence_search(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute evidence search tool"""
        search_criteria = parameters.get('search_criteria', {})
        confidence_threshold = parameters.get('confidence_threshold', 0.7)
        
        # Mock evidence search results
        result = {
            'search_id': f"ES_{int(datetime.now().timestamp())}",
            'criteria': search_criteria,
            'evidence_items': [
                {
                    'evidence_id': 'EV_001',
                    'type': 'physical_object',
                    'description': 'Dropped package at coordinates (245, 180)',
                    'confidence': 0.88,
                    'timestamp': '25.7s',
                    'location': 'parking_area'
                },
                {
                    'evidence_id': 'EV_002',
                    'type': 'license_plate',
                    'description': 'Partial license plate: ABC-123X',
                    'confidence': 0.82,
                    'timestamp': '18.2s',
                    'location': 'main_intersection'
                }
            ],
            'total_found': 2,
            'search_time': 2.3
        }
        
        logger.info(f"Evidence search completed: {result['total_found']} items found")
        return result
    
    async def _execute_timeline_generator(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute timeline generator tool"""
        case_id = parameters.get('case_id')
        time_range = parameters.get('time_range', [0, 60])
        
        # Mock timeline generation results
        result = {
            'timeline_id': f"TL_{int(datetime.now().timestamp())}",
            'case_id': case_id,
            'time_range': time_range,
            'timeline_events': [
                {
                    'timestamp': 12.5,
                    'event': 'Person enters surveillance area',
                    'confidence': 0.91,
                    'source': 'scene_analysis'
                },
                {
                    'timestamp': 18.2,
                    'event': 'Vehicle movement through intersection',
                    'confidence': 0.95,
                    'source': 'scene_analysis'
                },
                {
                    'timestamp': 25.7,
                    'event': 'Object interaction detected',
                    'confidence': 0.88,
                    'source': 'scene_analysis'
                }
            ],
            'total_events': 3,
            'generation_time': 1.8
        }
        
        logger.info(f"Timeline generated: {result['total_events']} events")
        return result
    
    async def _execute_report_generator(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute report generator tool"""
        case_id = parameters.get('case_id')
        query_results = parameters.get('query_results', {})
        
        # Mock report generation results
        result = {
            'report_id': f"RP_{int(datetime.now().timestamp())}",
            'case_id': case_id,
            'report_type': 'forensic_analysis_summary',
            'sections': [
                {
                    'title': 'Executive Summary',
                    'content': 'Comprehensive forensic analysis of surveillance footage revealed multiple significant events...'
                },
                {
                    'title': 'Event Timeline',
                    'content': 'Chronological sequence of detected events with confidence scores...'
                },
                {
                    'title': 'Evidence Summary',
                    'content': 'Physical evidence and digital artifacts identified during analysis...'
                }
            ],
            'confidence_score': 0.89,
            'generation_time': 8.5,
            'file_path': f'reports/forensic_report_{case_id}.pdf'
        }
        
        logger.info(f"Report generated: {len(result['sections'])} sections")
        return result

# Example usage
if __name__ == "__main__":
    import asyncio
    from llm_agent import ForensicLLMAgent
    
    async def test_tool_execution():
        """Test the tool calling framework"""
        print("🔧 Testing Tool Calling Framework")
        print("=" * 50)
        
        # Initialize components
        agent = ForensicLLMAgent()
        executor = ForensicToolExecutor()
        
        # Test query
        query = "Analyze the scene and predict trajectories for all detected persons"
        
        # Generate agent response
        agent_response = agent.generate_response(query)
        print(f"Query: {query}")
        print(f"Agent Response: {agent_response.response_text}")
        print(f"Planned Tools: {[tc.tool_type.value for tc in agent_response.tool_calls]}")
        
        # Create execution plan
        plan = executor.create_execution_plan(agent_response.tool_calls)
        print(f"Execution Order: {plan.execution_order}")
        print(f"Estimated Time: {plan.estimated_time:.1f}s")
        
        # Execute plan
        def progress_callback(current, total, tool_name):
            print(f"Progress: {current}/{total} - Executing {tool_name}")
        
        results = await executor.execute_plan(plan, progress_callback)
        
        # Display results
        print("\nExecution Results:")
        for result in results:
            print(f"- {result.tool_call.tool_type.value}: {result.status.value}")
            if result.error:
                print(f"  Error: {result.error}")
            else:
                print(f"  Time: {result.execution_time:.2f}s")
    
    # Run test
    asyncio.run(test_tool_execution())
