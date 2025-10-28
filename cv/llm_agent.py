"""
LLM Agent for HoloForensics - Week 11 Implementation
Advanced query understanding and tool calling capabilities for forensic analysis
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IntentType(Enum):
    """Types of forensic analysis intents"""
    QUERY_DATA = "query_data"
    ANALYZE_SCENE = "analyze_scene"
    PREDICT_TRAJECTORY = "predict_trajectory"
    INPAINT_VIDEO = "inpaint_video"
    GENERATE_REPORT = "generate_report"
    COMPARE_CASES = "compare_cases"
    DETECT_ANOMALIES = "detect_anomalies"
    TIMELINE_ANALYSIS = "timeline_analysis"

class ToolType(Enum):
    """Available forensic analysis tools"""
    SCENE_ANALYSIS = "scene_analysis"
    PHYSICS_PREDICTION = "physics_prediction"
    VIDEO_INPAINTING = "video_inpainting"
    RAG_QUERY = "rag_query"
    EVIDENCE_SEARCH = "evidence_search"
    TIMELINE_GENERATOR = "timeline_generator"
    REPORT_GENERATOR = "report_generator"

@dataclass
class QueryIntent:
    """Represents the intent extracted from a user query"""
    intent_type: IntentType
    confidence: float
    entities: Dict[str, Any]
    parameters: Dict[str, Any]
    required_tools: List[ToolType]
    reasoning: str

@dataclass
class ToolCall:
    """Represents a call to a forensic analysis tool"""
    tool_type: ToolType
    parameters: Dict[str, Any]
    priority: int
    dependencies: List[str]
    expected_output: str

@dataclass
class AgentResponse:
    """Response from the LLM agent"""
    response_text: str
    intent: QueryIntent
    tool_calls: List[ToolCall]
    confidence: float
    reasoning_steps: List[str]
    follow_up_questions: List[str]

class ForensicLLMAgent:
    """
    Advanced LLM Agent for forensic analysis with tool calling capabilities
    """
    
    def __init__(self):
        self.intent_patterns = self._initialize_intent_patterns()
        self.tool_registry = self._initialize_tool_registry()
        self.conversation_history = []
        
    def _initialize_intent_patterns(self) -> Dict[IntentType, List[str]]:
        """Initialize patterns for intent detection"""
        return {
            IntentType.QUERY_DATA: [
                r"what.*happened", r"show.*me", r"find.*", r"search.*",
                r"when.*did", r"where.*", r"who.*", r"how.*many"
            ],
            IntentType.ANALYZE_SCENE: [
                r"analyze.*scene", r"scene.*analysis", r"detect.*events",
                r"generate.*scene.*graph", r"identify.*objects"
            ],
            IntentType.PREDICT_TRAJECTORY: [
                r"predict.*trajectory", r"predict.*movement", r"physics.*prediction",
                r"where.*will.*go", r"future.*path", r"trajectory.*analysis"
            ],
            IntentType.INPAINT_VIDEO: [
                r"inpaint.*video", r"reconstruct.*video", r"fill.*gaps",
                r"repair.*video", r"video.*inpainting"
            ],
            IntentType.GENERATE_REPORT: [
                r"generate.*report", r"create.*report", r"summarize.*findings",
                r"forensic.*report", r"analysis.*summary"
            ],
            IntentType.TIMELINE_ANALYSIS: [
                r"timeline", r"chronological", r"sequence.*events",
                r"temporal.*analysis", r"when.*sequence"
            ]
        }
    
    def _initialize_tool_registry(self) -> Dict[ToolType, Dict[str, Any]]:
        """Initialize available forensic tools"""
        return {
            ToolType.SCENE_ANALYSIS: {
                "description": "Analyze video scenes to detect events and generate scene graphs",
                "parameters": ["scene_data", "case_id", "analysis_config"],
                "output_type": "scene_analysis_results",
                "dependencies": [],
                "api_endpoint": "/api/scene-analysis/start/"
            },
            ToolType.PHYSICS_PREDICTION: {
                "description": "Predict object trajectories using physics-informed models",
                "parameters": ["trajectory_data", "prediction_config", "time_horizon"],
                "output_type": "trajectory_predictions",
                "dependencies": ["scene_analysis"],
                "api_endpoint": "/api/physics/predict/"
            },
            ToolType.VIDEO_INPAINTING: {
                "description": "Reconstruct missing or corrupted video segments",
                "parameters": ["video_data", "mask_data", "inpainting_config"],
                "output_type": "inpainted_video",
                "dependencies": [],
                "api_endpoint": "/api/inpainting/start/"
            },
            ToolType.RAG_QUERY: {
                "description": "Query forensic data using natural language",
                "parameters": ["query_text", "case_id", "filters"],
                "output_type": "query_results",
                "dependencies": [],
                "api_endpoint": "/api/rag/query/"
            },
            ToolType.EVIDENCE_SEARCH: {
                "description": "Search for specific evidence items across cases",
                "parameters": ["search_criteria", "evidence_type", "confidence_threshold"],
                "output_type": "evidence_results",
                "dependencies": [],
                "api_endpoint": "/api/rag/query/"
            },
            ToolType.TIMELINE_GENERATOR: {
                "description": "Generate chronological timeline of events",
                "parameters": ["case_id", "time_range", "event_types"],
                "output_type": "timeline_data",
                "dependencies": ["scene_analysis"],
                "api_endpoint": "/api/rag/query/"
            }
        }
    
    def understand_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> QueryIntent:
        """
        Analyze user query to understand intent and extract entities
        
        Args:
            query: User's natural language query
            context: Optional context from previous interactions
            
        Returns:
            QueryIntent object with extracted information
        """
        query_lower = query.lower()
        
        # Detect primary intent
        intent_scores = {}
        for intent_type, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            if score > 0:
                intent_scores[intent_type] = score / len(patterns)
        
        # Get highest scoring intent
        if intent_scores:
            primary_intent = max(intent_scores.keys(), key=lambda k: intent_scores[k])
            confidence = intent_scores[primary_intent]
        else:
            primary_intent = IntentType.QUERY_DATA
            confidence = 0.5
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine required tools
        required_tools = self._determine_required_tools(primary_intent, entities, query)
        
        # Generate parameters
        parameters = self._extract_parameters(query, entities, primary_intent)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(query, primary_intent, entities, required_tools)
        
        return QueryIntent(
            intent_type=primary_intent,
            confidence=confidence,
            entities=entities,
            parameters=parameters,
            required_tools=required_tools,
            reasoning=reasoning
        )
    
    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities from the query"""
        entities = {}
        
        # Time references
        time_patterns = [
            (r'(\d+\.?\d*)\s*(?:to|and|-)\s*(\d+\.?\d*)\s*(?:seconds?|s)', 'time_range'),
            (r'(\d+\.?\d*)\s*(?:seconds?|s)', 'time_point'),
            (r'frame\s*(\d+)', 'frame_number'),
            (r'between\s+(\d+)\s+and\s+(\d+)', 'time_range')
        ]
        
        for pattern, entity_type in time_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                if entity_type == 'time_range':
                    entities['time_range'] = [float(m[0]), float(m[1])] if len(matches[0]) == 2 else None
                elif entity_type == 'time_point':
                    entities['time_point'] = float(matches[0])
                elif entity_type == 'frame_number':
                    entities['frame_number'] = int(matches[0])
        
        # Location references
        location_patterns = [
            (r'location\s*\((\d+),\s*(\d+)\)', 'coordinates'),
            (r'coordinates?\s*\((\d+),\s*(\d+)\)', 'coordinates'),
            (r'at\s+(\w+(?:\s+\w+)*)', 'location_name'),
            (r'in\s+the\s+(\w+(?:\s+\w+)*)', 'area')
        ]
        
        for pattern, entity_type in location_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                if entity_type == 'coordinates':
                    entities['coordinates'] = [int(matches[0][0]), int(matches[0][1])]
                else:
                    entities[entity_type] = matches[0]
        
        # Object references
        object_patterns = [
            (r'person(?:_(\d+))?', 'person'),
            (r'vehicle(?:_(\d+))?', 'vehicle'),
            (r'car(?:_(\d+))?', 'vehicle'),
            (r'object(?:_(\d+))?', 'object')
        ]
        
        for pattern, entity_type in object_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                entities[entity_type] = matches
        
        # Confidence thresholds
        confidence_patterns = [
            (r'confidence\s+(?:above|over|greater\s+than)\s+(\d+\.?\d*)', 'min_confidence'),
            (r'high\s+confidence', 'min_confidence_high'),
            (r'low\s+confidence', 'max_confidence_low')
        ]
        
        for pattern, entity_type in confidence_patterns:
            matches = re.findall(pattern, query.lower())
            if matches:
                if entity_type == 'min_confidence':
                    conf = float(matches[0])
                    entities['min_confidence'] = conf / 100 if conf > 1 else conf
                elif entity_type == 'min_confidence_high':
                    entities['min_confidence'] = 0.8
                elif entity_type == 'max_confidence_low':
                    entities['max_confidence'] = 0.6
        
        # Case references
        case_pattern = r'case\s+([A-Z0-9_]+)'
        case_matches = re.findall(case_pattern, query, re.IGNORECASE)
        if case_matches:
            entities['case_id'] = case_matches[0]
        
        return entities
    
    def _determine_required_tools(self, intent: IntentType, entities: Dict[str, Any], query: str) -> List[ToolType]:
        """Determine which tools are needed to fulfill the intent"""
        tools = []
        
        if intent == IntentType.QUERY_DATA:
            tools.append(ToolType.RAG_QUERY)
            
        elif intent == IntentType.ANALYZE_SCENE:
            tools.append(ToolType.SCENE_ANALYSIS)
            
        elif intent == IntentType.PREDICT_TRAJECTORY:
            tools.extend([ToolType.SCENE_ANALYSIS, ToolType.PHYSICS_PREDICTION])
            
        elif intent == IntentType.INPAINT_VIDEO:
            tools.append(ToolType.VIDEO_INPAINTING)
            
        elif intent == IntentType.TIMELINE_ANALYSIS:
            tools.extend([ToolType.RAG_QUERY, ToolType.TIMELINE_GENERATOR])
            
        elif intent == IntentType.GENERATE_REPORT:
            tools.extend([ToolType.RAG_QUERY, ToolType.REPORT_GENERATOR])
        
        # Check for specific tool mentions in query
        query_lower = query.lower()
        if 'inpaint' in query_lower or 'reconstruct' in query_lower:
            if ToolType.VIDEO_INPAINTING not in tools:
                tools.append(ToolType.VIDEO_INPAINTING)
                
        if 'predict' in query_lower or 'trajectory' in query_lower:
            if ToolType.PHYSICS_PREDICTION not in tools:
                tools.append(ToolType.PHYSICS_PREDICTION)
                
        if 'timeline' in query_lower or 'chronological' in query_lower:
            if ToolType.TIMELINE_GENERATOR not in tools:
                tools.append(ToolType.TIMELINE_GENERATOR)
        
        return tools
    
    def _extract_parameters(self, query: str, entities: Dict[str, Any], intent: IntentType) -> Dict[str, Any]:
        """Extract parameters for tool calls"""
        parameters = {}
        
        # Add extracted entities as parameters
        parameters.update(entities)
        
        # Add intent-specific parameters
        if intent == IntentType.ANALYZE_SCENE:
            parameters['analysis_type'] = 'full'
            parameters['detect_events'] = True
            parameters['generate_scene_graph'] = True
            
        elif intent == IntentType.PREDICT_TRAJECTORY:
            parameters['prediction_method'] = 'kalman_social_forces'
            parameters['time_horizon'] = entities.get('time_range', [0, 60])
            
        elif intent == IntentType.INPAINT_VIDEO:
            parameters['method'] = 'E2FGVI'
            parameters['quality_threshold'] = 0.8
            
        # Add query text for RAG queries
        parameters['query_text'] = query
        
        return parameters
    
    def _generate_reasoning(self, query: str, intent: IntentType, entities: Dict[str, Any], tools: List[ToolType]) -> str:
        """Generate reasoning for the analysis plan"""
        reasoning_parts = [
            f"Detected intent: {intent.value}",
            f"Extracted entities: {list(entities.keys())}",
            f"Required tools: {[tool.value for tool in tools]}"
        ]
        
        if entities.get('time_range'):
            reasoning_parts.append(f"Time range specified: {entities['time_range'][0]}s to {entities['time_range'][1]}s")
            
        if entities.get('case_id'):
            reasoning_parts.append(f"Case filter: {entities['case_id']}")
            
        if entities.get('min_confidence'):
            reasoning_parts.append(f"Confidence threshold: {entities['min_confidence']}")
        
        return "; ".join(reasoning_parts)
    
    def plan_execution(self, intent: QueryIntent) -> List[ToolCall]:
        """
        Plan the execution of tools to fulfill the intent
        
        Args:
            intent: The understood query intent
            
        Returns:
            List of tool calls in execution order
        """
        tool_calls = []
        
        for i, tool_type in enumerate(intent.required_tools):
            tool_info = self.tool_registry[tool_type]
            
            # Determine parameters for this tool
            tool_params = {}
            for param in tool_info['parameters']:
                if param in intent.parameters:
                    tool_params[param] = intent.parameters[param]
                elif param == 'case_id' and 'case_id' in intent.entities:
                    tool_params[param] = intent.entities['case_id']
                elif param == 'query_text':
                    tool_params[param] = intent.parameters.get('query_text', '')
            
            # Determine dependencies
            dependencies = []
            for dep_tool in tool_info['dependencies']:
                if any(tc.tool_type.value == dep_tool for tc in tool_calls):
                    dependencies.append(dep_tool)
            
            tool_call = ToolCall(
                tool_type=tool_type,
                parameters=tool_params,
                priority=i,
                dependencies=dependencies,
                expected_output=tool_info['output_type']
            )
            
            tool_calls.append(tool_call)
        
        return tool_calls
    
    def generate_response(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Generate a comprehensive response to a forensic query
        
        Args:
            query: User's natural language query
            context: Optional context from previous interactions
            
        Returns:
            AgentResponse with analysis plan and reasoning
        """
        # Understand the query
        intent = self.understand_query(query, context)
        
        # Plan tool execution
        tool_calls = self.plan_execution(intent)
        
        # Generate reasoning steps
        reasoning_steps = [
            f"1. Analyzed query: '{query}'",
            f"2. Detected intent: {intent.intent_type.value} (confidence: {intent.confidence:.2f})",
            f"3. Extracted entities: {', '.join(intent.entities.keys()) if intent.entities else 'none'}",
            f"4. Planned {len(tool_calls)} tool executions"
        ]
        
        # Generate response text
        response_text = self._generate_response_text(intent, tool_calls)
        
        # Generate follow-up questions
        follow_up_questions = self._generate_follow_up_questions(intent, tool_calls)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(intent, tool_calls)
        
        return AgentResponse(
            response_text=response_text,
            intent=intent,
            tool_calls=tool_calls,
            confidence=overall_confidence,
            reasoning_steps=reasoning_steps,
            follow_up_questions=follow_up_questions
        )
    
    def _generate_response_text(self, intent: QueryIntent, tool_calls: List[ToolCall]) -> str:
        """Generate human-readable response text"""
        if not tool_calls:
            return "I understand your query, but I'm not sure how to help with that specific request."
        
        response_parts = [
            f"I'll help you with your {intent.intent_type.value.replace('_', ' ')} request."
        ]
        
        if len(tool_calls) == 1:
            tool = tool_calls[0]
            tool_name = tool.tool_type.value.replace('_', ' ').title()
            response_parts.append(f"I'll use {tool_name} to process your request.")
        else:
            response_parts.append(f"This will require {len(tool_calls)} analysis steps:")
            for i, tool in enumerate(tool_calls, 1):
                tool_name = tool.tool_type.value.replace('_', ' ').title()
                response_parts.append(f"{i}. {tool_name}")
        
        # Add specific details based on intent
        if intent.entities.get('time_range'):
            tr = intent.entities['time_range']
            response_parts.append(f"Focusing on the time range {tr[0]}s to {tr[1]}s.")
            
        if intent.entities.get('case_id'):
            response_parts.append(f"Analyzing case {intent.entities['case_id']}.")
            
        if intent.entities.get('min_confidence'):
            conf = intent.entities['min_confidence']
            response_parts.append(f"Filtering for results with confidence above {conf:.1%}.")
        
        response_parts.append("Processing your request now...")
        
        return " ".join(response_parts)
    
    def _generate_follow_up_questions(self, intent: QueryIntent, tool_calls: List[ToolCall]) -> List[str]:
        """Generate relevant follow-up questions"""
        questions = []
        
        if intent.intent_type == IntentType.QUERY_DATA:
            questions.extend([
                "Would you like me to analyze any specific time periods?",
                "Should I filter by confidence level?",
                "Are you interested in a particular type of event?"
            ])
            
        elif intent.intent_type == IntentType.ANALYZE_SCENE:
            questions.extend([
                "Would you like me to focus on specific object types?",
                "Should I generate a detailed timeline of events?",
                "Do you want trajectory predictions for detected objects?"
            ])
            
        elif intent.intent_type == IntentType.PREDICT_TRAJECTORY:
            questions.extend([
                "What time horizon should I predict for?",
                "Should I include social force interactions?",
                "Would you like uncertainty estimates?"
            ])
        
        return questions[:3]  # Limit to 3 questions
    
    def _calculate_overall_confidence(self, intent: QueryIntent, tool_calls: List[ToolCall]) -> float:
        """Calculate overall confidence in the response"""
        base_confidence = intent.confidence
        
        # Adjust based on tool availability
        tool_confidence = 1.0 if tool_calls else 0.3
        
        # Adjust based on entity extraction
        entity_confidence = min(1.0, len(intent.entities) * 0.2 + 0.6)
        
        return (base_confidence + tool_confidence + entity_confidence) / 3

# Example usage and testing
if __name__ == "__main__":
    agent = ForensicLLMAgent()
    
    test_queries = [
        "What events occurred between 10 and 20 seconds with high confidence?",
        "Analyze the scene and detect all person-vehicle interactions",
        "Predict the trajectory of person_001 for the next 30 seconds",
        "Inpaint the corrupted video segments in case CASE_2024_001",
        "Generate a timeline of all critical events",
        "Show me evidence from the main intersection with confidence above 0.8"
    ]
    
    print("🤖 Testing Forensic LLM Agent")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = agent.generate_response(query)
        
        print(f"Intent: {response.intent.intent_type.value}")
        print(f"Confidence: {response.confidence:.2f}")
        print(f"Tools: {[tc.tool_type.value for tc in response.tool_calls]}")
        print(f"Response: {response.response_text}")
        print("-" * 30)
