"""
RAG (Retrieval-Augmented Generation) System for HoloForensics
Implements intelligent querying and response generation for forensic analysis
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from pathlib import Path

from vector_store import ForensicVectorStore, QueryResult, ForensicDocument

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RAGQuery:
    """Represents a RAG query"""
    query_id: str
    query_text: str
    case_id: Optional[str]
    query_type: str  # 'general', 'case_specific', 'event_analysis', 'evidence_search'
    filters: Dict[str, Any]
    timestamp: datetime
    user_id: str

@dataclass
class RAGResponse:
    """Represents a RAG response"""
    query_id: str
    response_text: str
    retrieved_documents: List[QueryResult]
    confidence_score: float
    sources: List[str]
    metadata: Dict[str, Any]
    timestamp: datetime
    processing_time: float

class ForensicQueryProcessor:
    """Processes and enhances forensic queries"""
    
    def __init__(self):
        self.forensic_keywords = {
            'temporal': ['when', 'time', 'during', 'before', 'after', 'timeline', 'sequence'],
            'spatial': ['where', 'location', 'position', 'area', 'zone', 'coordinates'],
            'objects': ['person', 'vehicle', 'car', 'truck', 'bike', 'weapon', 'object'],
            'events': ['event', 'incident', 'occurrence', 'happening', 'activity', 'action'],
            'evidence': ['evidence', 'proof', 'clue', 'trace', 'forensic', 'analysis'],
            'movement': ['movement', 'motion', 'trajectory', 'path', 'direction', 'speed'],
            'interaction': ['interaction', 'contact', 'collision', 'meeting', 'encounter']
        }
        
        self.severity_keywords = {
            'critical': ['critical', 'urgent', 'emergency', 'severe', 'dangerous'],
            'high': ['high', 'important', 'significant', 'major'],
            'medium': ['medium', 'moderate', 'notable'],
            'low': ['low', 'minor', 'slight']
        }
    
    def analyze_query(self, query_text: str) -> Dict[str, Any]:
        """
        Analyze query to extract intent and entities
        
        Args:
            query_text: Raw query text
            
        Returns:
            Dictionary with query analysis
        """
        query_lower = query_text.lower()
        
        analysis = {
            'query_type': 'general',
            'entities': {},
            'intent': 'search',
            'temporal_references': [],
            'spatial_references': [],
            'confidence_indicators': [],
            'filters': {}
        }
        
        # Detect query type based on keywords
        for category, keywords in self.forensic_keywords.items():
            matches = [kw for kw in keywords if kw in query_lower]
            if matches:
                analysis['entities'][category] = matches
                
                # Determine primary query type
                if category in ['temporal', 'events'] and len(matches) >= 2:
                    analysis['query_type'] = 'event_analysis'
                elif category == 'evidence':
                    analysis['query_type'] = 'evidence_search'
        
        # Extract temporal references
        time_patterns = [
            r'(\d{1,2}:\d{2})',  # Time format
            r'(\d+\.\d+)s',      # Seconds
            r'frame\s+(\d+)',    # Frame numbers
            r'between\s+(\d+)\s+and\s+(\d+)'  # Time ranges
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, query_text)
            if matches:
                analysis['temporal_references'].extend(matches)
        
        # Extract confidence indicators
        confidence_patterns = [
            r'confidence\s+(?:above|over|greater than)\s+(\d+\.?\d*)',
            r'certainty\s+(?:above|over|greater than)\s+(\d+\.?\d*)',
            r'probability\s+(?:above|over|greater than)\s+(\d+\.?\d*)'
        ]
        
        for pattern in confidence_patterns:
            matches = re.findall(pattern, query_lower)
            if matches:
                try:
                    confidence = float(matches[0])
                    if confidence > 1:  # Assume percentage
                        confidence = confidence / 100
                    analysis['confidence_indicators'].append(confidence)
                    analysis['filters']['min_confidence'] = confidence
                except ValueError:
                    pass
        
        # Detect severity filters
        for severity, keywords in self.severity_keywords.items():
            if any(kw in query_lower for kw in keywords):
                analysis['filters']['severity'] = severity
                break
        
        # Extract case ID references
        case_pattern = r'case\s+([A-Z0-9_]+)'
        case_matches = re.findall(case_pattern, query_text, re.IGNORECASE)
        if case_matches:
            analysis['filters']['case_id'] = case_matches[0]
        
        return analysis
    
    def enhance_query(self, query_text: str, analysis: Dict[str, Any]) -> str:
        """
        Enhance query with forensic context
        
        Args:
            query_text: Original query
            analysis: Query analysis results
            
        Returns:
            Enhanced query text
        """
        enhanced_parts = [query_text]
        
        # Add context based on detected entities
        if 'temporal' in analysis['entities']:
            enhanced_parts.append("Include temporal sequence and timeline information.")
        
        if 'spatial' in analysis['entities']:
            enhanced_parts.append("Include location and spatial relationship details.")
        
        if 'events' in analysis['entities']:
            enhanced_parts.append("Focus on event detection and incident analysis.")
        
        if 'evidence' in analysis['entities']:
            enhanced_parts.append("Emphasize forensic evidence and analysis results.")
        
        # Add confidence context
        if analysis['confidence_indicators']:
            conf = analysis['confidence_indicators'][0]
            enhanced_parts.append(f"Focus on high-confidence results (>{conf:.2f}).")
        
        return " ".join(enhanced_parts)

class ForensicResponseGenerator:
    """Generates forensic analysis responses"""
    
    def __init__(self):
        self.response_templates = {
            'event_analysis': {
                'intro': "Based on the forensic analysis, here are the relevant events:",
                'event_format': "• {event_type} at {time}s (Confidence: {confidence:.2f})\n  {description}",
                'summary': "Analysis Summary: {total_events} events detected with {high_conf_events} high-confidence detections."
            },
            'evidence_search': {
                'intro': "Forensic evidence analysis results:",
                'evidence_format': "• Evidence ID: {evidence_id}\n  Type: {evidence_type}\n  Confidence: {confidence:.2f}\n  Description: {description}",
                'summary': "Evidence Summary: {total_evidence} items found with {quality_score:.2f} overall quality score."
            },
            'general': {
                'intro': "Forensic analysis results:",
                'item_format': "• {content_preview}",
                'summary': "Found {total_results} relevant items across {unique_cases} cases."
            }
        }
    
    def generate_response(self, 
                         query: RAGQuery, 
                         retrieved_docs: List[QueryResult],
                         analysis: Dict[str, Any]) -> str:
        """
        Generate response based on query and retrieved documents
        
        Args:
            query: Original query
            retrieved_docs: Retrieved documents from vector store
            analysis: Query analysis results
            
        Returns:
            Generated response text
        """
        if not retrieved_docs:
            return self._generate_no_results_response(query, analysis)
        
        query_type = analysis.get('query_type', 'general')
        template = self.response_templates.get(query_type, self.response_templates['general'])
        
        response_parts = [template['intro']]
        
        # Process documents based on query type
        if query_type == 'event_analysis':
            response_parts.extend(self._format_event_results(retrieved_docs, template))
        elif query_type == 'evidence_search':
            response_parts.extend(self._format_evidence_results(retrieved_docs, template))
        else:
            response_parts.extend(self._format_general_results(retrieved_docs, template))
        
        # Add summary
        summary = self._generate_summary(retrieved_docs, query_type, template)
        if summary:
            response_parts.append("\n" + summary)
        
        # Add confidence and quality indicators
        quality_info = self._generate_quality_info(retrieved_docs)
        if quality_info:
            response_parts.append("\n" + quality_info)
        
        return "\n\n".join(response_parts)
    
    def _format_event_results(self, docs: List[QueryResult], template: Dict[str, str]) -> List[str]:
        """Format event-specific results"""
        results = []
        
        for doc in docs[:5]:  # Top 5 results
            if doc.document_type == 'event':
                metadata = doc.metadata
                event_text = template['event_format'].format(
                    event_type=metadata.get('event_type', 'Unknown'),
                    time=metadata.get('start_time', 'Unknown'),
                    confidence=metadata.get('confidence', 0.0),
                    description=doc.content.split('\n')[2].replace('Description: ', '') if '\n' in doc.content else doc.content[:100]
                )
                results.append(event_text)
        
        return results
    
    def _format_evidence_results(self, docs: List[QueryResult], template: Dict[str, str]) -> List[str]:
        """Format evidence-specific results"""
        results = []
        
        for doc in docs[:5]:  # Top 5 results
            if doc.document_type in ['evidence', 'scene_analysis']:
                metadata = doc.metadata
                evidence_text = template['evidence_format'].format(
                    evidence_id=doc.doc_id,
                    evidence_type=doc.document_type,
                    confidence=metadata.get('confidence', 0.0),
                    description=doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
                )
                results.append(evidence_text)
        
        return results
    
    def _format_general_results(self, docs: List[QueryResult], template: Dict[str, str]) -> List[str]:
        """Format general results"""
        results = []
        
        for doc in docs[:5]:  # Top 5 results
            content_preview = doc.content[:150] + "..." if len(doc.content) > 150 else doc.content
            item_text = template['item_format'].format(content_preview=content_preview)
            results.append(item_text)
        
        return results
    
    def _generate_summary(self, docs: List[QueryResult], query_type: str, template: Dict[str, str]) -> str:
        """Generate summary based on results"""
        if not docs:
            return ""
        
        if query_type == 'event_analysis':
            total_events = len([d for d in docs if d.document_type == 'event'])
            high_conf_events = len([d for d in docs if d.document_type == 'event' and d.metadata.get('confidence', 0) > 0.8])
            return template['summary'].format(
                total_events=total_events,
                high_conf_events=high_conf_events
            )
        
        elif query_type == 'evidence_search':
            total_evidence = len(docs)
            avg_quality = sum(d.metadata.get('confidence', 0) for d in docs) / len(docs)
            return template['summary'].format(
                total_evidence=total_evidence,
                quality_score=avg_quality
            )
        
        else:
            unique_cases = len(set(d.case_id for d in docs))
            return template['summary'].format(
                total_results=len(docs),
                unique_cases=unique_cases
            )
    
    def _generate_quality_info(self, docs: List[QueryResult]) -> str:
        """Generate quality and confidence information"""
        if not docs:
            return ""
        
        avg_similarity = sum(d.similarity_score for d in docs) / len(docs)
        avg_confidence = sum(d.metadata.get('confidence', 0) for d in docs) / len(docs)
        
        quality_parts = [
            f"Quality Metrics:",
            f"• Average Similarity: {avg_similarity:.3f}",
            f"• Average Confidence: {avg_confidence:.3f}",
            f"• Results Reliability: {'High' if avg_confidence > 0.8 else 'Medium' if avg_confidence > 0.6 else 'Low'}"
        ]
        
        return "\n".join(quality_parts)
    
    def _generate_no_results_response(self, query: RAGQuery, analysis: Dict[str, Any]) -> str:
        """Generate response when no results found"""
        suggestions = []
        
        if query.case_id:
            suggestions.append(f"• Try searching without case filter '{query.case_id}'")
        
        if analysis.get('filters', {}).get('min_confidence'):
            suggestions.append("• Lower the confidence threshold")
        
        if len(query.query_text.split()) > 10:
            suggestions.append("• Try a shorter, more specific query")
        
        suggestions.append("• Check if the case data has been processed and indexed")
        
        response = "No matching forensic data found for your query.\n\nSuggestions:\n" + "\n".join(suggestions)
        
        return response

class ForensicRAGSystem:
    """Main RAG system for forensic analysis"""
    
    def __init__(self, vector_store: Optional[ForensicVectorStore] = None):
        """
        Initialize RAG system
        
        Args:
            vector_store: Optional vector store instance
        """
        self.vector_store = vector_store or ForensicVectorStore()
        self.query_processor = ForensicQueryProcessor()
        self.response_generator = ForensicResponseGenerator()
        self.query_history = []
    
    def process_query(self, 
                     query_text: str,
                     case_id: Optional[str] = None,
                     user_id: str = "unknown",
                     max_results: int = 10) -> RAGResponse:
        """
        Process a forensic query and generate response
        
        Args:
            query_text: User query
            case_id: Optional case filter
            user_id: User identifier
            max_results: Maximum number of results to retrieve
            
        Returns:
            RAG response with generated answer
        """
        start_time = datetime.now()
        
        # Create query object
        query = RAGQuery(
            query_id=f"query_{int(start_time.timestamp())}",
            query_text=query_text,
            case_id=case_id,
            query_type='general',
            filters={},
            timestamp=start_time,
            user_id=user_id
        )
        
        try:
            # Analyze query
            analysis = self.query_processor.analyze_query(query_text)
            query.query_type = analysis['query_type']
            query.filters = analysis['filters']
            
            # Enhance query for better retrieval
            enhanced_query = self.query_processor.enhance_query(query_text, analysis)
            
            # Retrieve relevant documents
            retrieved_docs = self.vector_store.query(
                query_text=enhanced_query,
                n_results=max_results,
                case_id=case_id or analysis['filters'].get('case_id'),
                document_type=None,  # Allow all types
                min_confidence=analysis['filters'].get('min_confidence')
            )
            
            # Generate response
            response_text = self.response_generator.generate_response(
                query, retrieved_docs, analysis
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_response_confidence(retrieved_docs, analysis)
            
            # Extract sources
            sources = [f"{doc.document_type}:{doc.doc_id}" for doc in retrieved_docs[:5]]
            
            # Create response
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            response = RAGResponse(
                query_id=query.query_id,
                response_text=response_text,
                retrieved_documents=retrieved_docs,
                confidence_score=confidence_score,
                sources=sources,
                metadata={
                    'query_analysis': analysis,
                    'enhanced_query': enhanced_query,
                    'num_results': len(retrieved_docs)
                },
                timestamp=end_time,
                processing_time=processing_time
            )
            
            # Store in history
            self.query_history.append((query, response))
            
            logger.info(f"Processed query {query.query_id} in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            
            # Return error response
            return RAGResponse(
                query_id=query.query_id,
                response_text=f"Error processing query: {str(e)}",
                retrieved_documents=[],
                confidence_score=0.0,
                sources=[],
                metadata={'error': str(e)},
                timestamp=datetime.now(),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _calculate_response_confidence(self, 
                                    retrieved_docs: List[QueryResult], 
                                    analysis: Dict[str, Any]) -> float:
        """
        Calculate confidence score for the response
        
        Args:
            retrieved_docs: Retrieved documents
            analysis: Query analysis
            
        Returns:
            Confidence score between 0 and 1
        """
        if not retrieved_docs:
            return 0.0
        
        # Base confidence from similarity scores
        avg_similarity = sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs)
        
        # Adjust based on document confidence
        avg_doc_confidence = sum(doc.metadata.get('confidence', 0.5) for doc in retrieved_docs) / len(retrieved_docs)
        
        # Adjust based on number of results
        result_factor = min(len(retrieved_docs) / 5, 1.0)  # Optimal around 5 results
        
        # Combine factors
        confidence = (avg_similarity * 0.4 + avg_doc_confidence * 0.4 + result_factor * 0.2)
        
        return min(confidence, 1.0)
    
    def add_forensic_data(self, documents: List[ForensicDocument]) -> int:
        """
        Add forensic data to the vector store
        
        Args:
            documents: List of forensic documents
            
        Returns:
            Number of successfully added documents
        """
        return self.vector_store.add_documents_batch(documents)
    
    def get_query_history(self, user_id: Optional[str] = None) -> List[Tuple[RAGQuery, RAGResponse]]:
        """
        Get query history
        
        Args:
            user_id: Optional user filter
            
        Returns:
            List of query-response pairs
        """
        if user_id:
            return [(q, r) for q, r in self.query_history if q.user_id == user_id]
        return self.query_history
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics
        
        Returns:
            Dictionary with system stats
        """
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            'vector_store': vector_stats,
            'query_history_count': len(self.query_history),
            'recent_queries': len([q for q, r in self.query_history 
                                 if (datetime.now() - q.timestamp).days < 1])
        }

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    rag_system = ForensicRAGSystem()
    
    # Example queries
    test_queries = [
        "What events occurred between 10 and 20 seconds?",
        "Show me high confidence evidence from case CASE_2024_001",
        "Find all person-vehicle interactions",
        "What happened at location (150, 200)?",
        "Show critical events with confidence above 0.8"
    ]
    
    for query_text in test_queries:
        print(f"\nQuery: {query_text}")
        response = rag_system.process_query(query_text)
        print(f"Response: {response.response_text}")
        print(f"Confidence: {response.confidence_score:.3f}")
        print(f"Processing time: {response.processing_time:.3f}s")
        print("-" * 50)
