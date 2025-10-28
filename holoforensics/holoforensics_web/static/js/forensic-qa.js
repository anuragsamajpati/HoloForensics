/**
 * Forensic Q&A System JavaScript
 * Enhanced with LLM Agent and intelligent multi-step reasoning
 */

// Global variables
let currentQuery = null;
let queryHistory = [];
let systemStats = {};
let currentInvestigation = null;
let agentCapabilities = {};
let reasoningMode = 'standard'; // minimal, standard, deep

// Initialize Forensic Q&A system
document.addEventListener('DOMContentLoaded', function() {
    // Check if we should open the forensic Q&A modal
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('tool') === 'forensic-qa') {
        openForensicQAModal();
    }
    
    // Load system stats and agent capabilities
    loadSystemStats();
    loadAgentCapabilities();
    
    // Set up keyboard shortcuts
    setupKeyboardShortcuts();
    
    // Initialize reasoning mode selector
    initializeReasoningMode();
});

/**
 * Open Forensic Q&A Modal
 */
function openForensicQAModal() {
    const modal = document.getElementById('forensicQAModal');
    if (modal) {
        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
        
        // Load initial data
        loadSystemStats();
        loadQueryHistory();
        loadIntelligentSuggestions();
        
        // Focus on query input
        const queryInput = document.getElementById('queryInput');
        if (queryInput) {
            setTimeout(() => queryInput.focus(), 300);
        }
        
        console.log('Forensic Q&A modal opened');
    }
}

/**
 * Close Forensic Q&A Modal
 */
function closeForensicQAModal() {
    const modal = document.getElementById('forensicQAModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
        
        // Clear any active queries
        clearResponse();
        
        console.log('Forensic Q&A modal closed');
    }
}

/**
 * Load system statistics
 */
async function loadSystemStats() {
    try {
        const response = await fetch('/api/rag/stats/', {
            method: 'GET',
            headers: {
                'X-CSRFToken': getCSRFToken(),
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            systemStats = data;
            updateStatsDisplay(data);
        } else {
            console.error('Failed to load system stats');
        }
    } catch (error) {
        console.error('Error loading system stats:', error);
    }
}

/**
 * Update stats display
 */
function updateStatsDisplay(stats) {
    const indexedDocs = document.getElementById('indexedDocs');
    const userQueries = document.getElementById('userQueries');
    const avgConfidence = document.getElementById('avgConfidence');
    
    if (indexedDocs && stats.system_stats && stats.system_stats.vector_store) {
        indexedDocs.textContent = stats.system_stats.vector_store.total_documents || 0;
    }
    
    if (userQueries && stats.user_stats) {
        userQueries.textContent = stats.user_stats.total_queries || 0;
    }
    
    if (avgConfidence && stats.user_stats) {
        const confidence = stats.user_stats.avg_confidence || 0;
        avgConfidence.textContent = (confidence * 100).toFixed(1) + '%';
    }
}

/**
 * Load query suggestions
 */
async function loadQASuggestions() {
    const caseSelect = document.getElementById('qaCaseSelect');
    const caseId = caseSelect ? caseSelect.value : null;
    
    try {
        showLoading('Loading suggestions...');
        
        const response = await fetch('/api/rag/suggestions/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': getCSRFToken(),
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                case_id: caseId
            })
        });
        
        hideLoading();
        
        if (response.ok) {
            const data = await response.json();
            displaySuggestions(data.suggested_queries || []);
        } else {
            showError('Failed to load suggestions');
        }
    } catch (error) {
        hideLoading();
        console.error('Error loading suggestions:', error);
        showError('Error loading suggestions');
    }
}

/**
 * Display query suggestions
 */
function displaySuggestions(suggestions) {
    const suggestionsContainer = document.getElementById('qaSuggestions');
    const suggestionsGrid = document.getElementById('suggestionsGrid');
    
    if (!suggestionsContainer || !suggestionsGrid) return;
    
    suggestionsGrid.innerHTML = '';
    
    suggestions.forEach(suggestion => {
        const suggestionButton = document.createElement('button');
        suggestionButton.className = 'suggestion-item';
        suggestionButton.textContent = suggestion;
        suggestionButton.onclick = () => {
            document.getElementById('queryInput').value = suggestion;
            suggestionsContainer.style.display = 'none';
        };
        
        suggestionsGrid.appendChild(suggestionButton);
    });
    
    suggestionsContainer.style.display = suggestions.length > 0 ? 'block' : 'none';
}

/**
 * Submit Query with Intelligent Agent
 */
function submitQuery() {
    const queryInput = document.getElementById('queryInput');
    const caseSelect = document.getElementById('caseSelect');
    const responseArea = document.getElementById('responseArea');
    const loadingSpinner = document.getElementById('loadingSpinner');
    
    if (!queryInput || !queryInput.value.trim()) {
        showNotification('Please enter a query', 'error');
        return;
    }
    
    const query = queryInput.value.trim();
    const caseId = caseSelect ? caseSelect.value : null;
    
    // Show loading state
    loadingSpinner.style.display = 'flex';
    responseArea.innerHTML = '';
    
    // Store current query
    currentQuery = {
        text: query,
        case_id: caseId,
        timestamp: new Date().toISOString(),
        reasoning_mode: reasoningMode
    };
    
    // Show intelligent processing indicator
    showIntelligentProcessingStatus('Analyzing query intent...');
    
    // Make API request to intelligent agent
    fetch('/api/llm/intelligent-query/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify({
            query_text: query,
            case_id: caseId,
            enable_tools: true,
            reasoning_depth: reasoningMode
        })
    })
    .then(response => response.json())
    .then(data => {
        loadingSpinner.style.display = 'none';
        hideIntelligentProcessingStatus();
        
        if (data.success) {
            currentInvestigation = data;
            displayIntelligentResponse(data);
            addToQueryHistory(query, data);
        } else {
            showNotification(data.error || 'Investigation failed', 'error');
        }
    })
    .catch(error => {
        console.error('Investigation error:', error);
        loadingSpinner.style.display = 'none';
        hideIntelligentProcessingStatus();
        showNotification('Network error occurred', 'error');
    });
}

/**
 * Display intelligent investigation response
 */
function displayIntelligentResponse(data) {
    const responseArea = document.getElementById('qaResponseArea');
    const responseContent = document.getElementById('responseContent');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const processingTime = document.getElementById('processingTime');
    
    if (!responseArea || !responseContent) return;
    
    // Create enhanced response HTML
    let responseHTML = `
        <div class="intelligent-response">
            <div class="investigation-header">
                <h4>🧠 Intelligent Investigation Results</h4>
                <div class="investigation-meta">
                    <span class="investigation-id">ID: ${data.investigation_id}</span>
                    <span class="intent-type">Intent: ${data.intent.type.replace('_', ' ').toUpperCase()}</span>
                    <span class="reasoning-steps">${data.metadata.reasoning_steps} reasoning steps</span>
                </div>
            </div>
            
            <div class="intent-analysis">
                <h5>Query Understanding</h5>
                <div class="intent-details">
                    <div class="confidence-meter">
                        <span>Intent Confidence: </span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${data.intent.confidence * 100}%"></div>
                        </div>
                        <span>${(data.intent.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="extracted-entities">
                        <strong>Entities:</strong> ${JSON.stringify(data.intent.entities, null, 2)}
                    </div>
                    <div class="required-tools">
                        <strong>Tools Required:</strong> ${data.intent.required_tools.join(', ')}
                    </div>
                </div>
            </div>
            
            <div class="investigation-response">
                <h5>Investigation Results</h5>
                <div class="response-text">${data.response_text}</div>
            </div>`;
    
    // Add reasoning trace if available
    if (data.reasoning_trace && data.reasoning_trace.length > 0) {
        responseHTML += `
            <div class="reasoning-trace">
                <h5>Reasoning Process</h5>
                <div class="reasoning-steps">`;
        
        data.reasoning_trace.forEach((step, index) => {
            responseHTML += `
                <div class="reasoning-step">
                    <div class="step-header">
                        <span class="step-number">${index + 1}</span>
                        <span class="step-type">${step.step_type.replace('_', ' ').toUpperCase()}</span>
                        <span class="step-confidence">${(step.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="step-reasoning">${step.reasoning}</div>
                </div>`;
        });
        
        responseHTML += `
                </div>
            </div>`;
    }
    
    responseHTML += `</div>`;
    
    responseContent.innerHTML = responseHTML;
    
    // Update confidence badge
    if (confidenceBadge) {
        confidenceBadge.textContent = `${(data.confidence_score * 100).toFixed(1)}%`;
        confidenceBadge.className = `confidence-badge ${getConfidenceClass(data.confidence_score)}`;
    }
    
    // Update processing info
    if (processingTime) {
        processingTime.textContent = `${data.metadata.reasoning_steps} steps completed`;
    }
    
    // Show response area
    responseArea.style.display = 'block';
    
    // Scroll to response
    responseArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

/**
 * Display query response (legacy function for backward compatibility)
 */
function displayResponse(data) {
    // For legacy RAG responses, convert to intelligent format
    const intelligentData = {
        investigation_id: 'legacy_' + Date.now(),
        query: currentQuery?.text || '',
        intent: {
            type: 'query_data',
            confidence: data.confidence_score || 0.8,
            entities: {},
            required_tools: ['rag_query']
        },
        response_text: data.response_text,
        confidence_score: data.confidence_score || 0.8,
        status: 'completed',
        reasoning_trace: [],
        metadata: {
            reasoning_steps: 1,
            tools_executed: 0,
            processing_time: 0
        }
    };
    
    displayIntelligentResponse(intelligentData);
}

/**
 * Show intelligent processing status
 */
function showIntelligentProcessingStatus(message) {
    const statusElement = document.getElementById('processingStatus');
    if (statusElement) {
        statusElement.innerHTML = `
            <div class="intelligent-processing">
                <div class="processing-spinner"></div>
                <span class="processing-message">${message}</span>
            </div>`;
        statusElement.style.display = 'block';
    }
}

/**
 * Hide intelligent processing status
 */
function hideIntelligentProcessingStatus() {
    const statusElement = document.getElementById('processingStatus');
    if (statusElement) {
        statusElement.style.display = 'none';
    }
}

/**
 * Load agent capabilities
 */
function loadAgentCapabilities() {
    fetch('/api/llm/capabilities/', {
        method: 'GET',
        headers: {
            'X-CSRFToken': getCSRFToken()
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            agentCapabilities = data.capabilities;
            updateCapabilitiesUI();
        }
    })
    .catch(error => {
        console.error('Error loading agent capabilities:', error);
    });
}

/**
 * Load intelligent suggestions
 */
function loadIntelligentSuggestions() {
    const caseSelect = document.getElementById('caseSelect');
    const caseId = caseSelect ? caseSelect.value : null;
    
    fetch('/api/llm/suggestions/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify({
            case_id: caseId,
            context: {}
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displaySuggestions(data.suggestions);
        }
    })
    .catch(error => {
        console.error('Error loading intelligent suggestions:', error);
    });
}

/**
 * Initialize reasoning mode selector
 */
function initializeReasoningMode() {
    const reasoningSelect = document.getElementById('reasoningMode');
    if (reasoningSelect) {
        reasoningSelect.addEventListener('change', function() {
            reasoningMode = this.value;
            console.log('Reasoning mode changed to:', reasoningMode);
        });
    }
}

/**
 * Update capabilities UI
 */
function updateCapabilitiesUI() {
    const capabilitiesContainer = document.getElementById('agentCapabilities');
    if (!capabilitiesContainer || !agentCapabilities.intent_types) return;
    
    let capabilitiesHTML = '<h5>Agent Capabilities</h5><div class="capabilities-grid">';
    
    agentCapabilities.intent_types.forEach(intent => {
        capabilitiesHTML += `
            <div class="capability-card">
                <h6>${intent.type.replace('_', ' ').toUpperCase()}</h6>
                <p>${intent.description}</p>
                <div class="capability-examples">
                    ${intent.examples.map(ex => `<span class="example">"${ex}"</span>`).join('')}
                </div>
            </div>`;
    });
    
    capabilitiesHTML += '</div>';
    capabilitiesContainer.innerHTML = capabilitiesHTML;
}

/**
 * Show agent capabilities
 */
function showAgentCapabilities() {
    const capabilitiesContainer = document.getElementById('agentCapabilities');
    if (capabilitiesContainer) {
        if (capabilitiesContainer.style.display === 'none') {
            capabilitiesContainer.style.display = 'block';
            loadAgentCapabilities();
        } else {
            capabilitiesContainer.style.display = 'none';
        }
    }
}

/**
 * Get confidence class for styling
 */
function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'high';
    if (confidence >= 0.6) return 'medium';
    return 'low';
}

/**
 * Add query to history
 */
function addToQueryHistory(query, response) {
    const historyItem = {
        query: query,
        response: response,
        timestamp: new Date().toISOString(),
        investigation_id: response.investigation_id || 'legacy_' + Date.now()
    };
    
    queryHistory.unshift(historyItem);
    
    // Keep only last 20 queries
    if (queryHistory.length > 20) {
        queryHistory = queryHistory.slice(0, 20);
    }
    
    updateHistoryDisplay();
}

/**
 * Update history display
 */
function updateHistoryDisplay() {
    const historyList = document.getElementById('historyList');
    if (!historyList) return;
    
    historyList.innerHTML = '';
    
    queryHistory.slice(0, 5).forEach(item => {
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const timestamp = new Date(item.timestamp).toLocaleString();
        const confidence = item.response.confidence_score || 0;
        
        historyItem.innerHTML = `
            <div class="history-query">${item.query}</div>
            <div class="history-meta">
                <span class="history-time">${timestamp}</span>
                <span class="history-confidence ${getConfidenceClass(confidence)}">${(confidence * 100).toFixed(1)}%</span>
            </div>
        `;
        
        historyItem.onclick = () => {
            document.getElementById('queryInput').value = item.query;
        };
        
        historyList.appendChild(historyItem);
    });
}

/**
 * Load query history
 */
function loadQueryHistory() {
    fetch('/api/llm/history/', {
        method: 'GET',
        headers: {
            'X-CSRFToken': getCSRFToken()
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            queryHistory = data.investigation_history.map(inv => ({
                query: inv.query,
                response: {
                    investigation_id: inv.investigation_id,
                    confidence_score: inv.confidence,
                    status: inv.status
                },
                timestamp: inv.timestamp,
                investigation_id: inv.investigation_id
            }));
            updateHistoryDisplay();
        }
    })
    .catch(error => {
        console.error('Error loading query history:', error);
    });
}

/**
 * Load system stats
 */
function loadSystemStats() {
    fetch('/api/llm/stats/', {
        method: 'GET',
        headers: {
            'X-CSRFToken': getCSRFToken()
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            systemStats = data.agent_stats;
            updateSystemStatsDisplay();
        }
    })
    .catch(error => {
        console.error('Error loading system stats:', error);
        // Fallback to RAG stats
        loadRAGStats();
    });
}

/**
 * Load RAG stats (fallback)
 */
function loadRAGStats() {
    fetch('/api/rag/stats/', {
        method: 'GET',
        headers: {
            'X-CSRFToken': getCSRFToken()
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            systemStats = {
                system_stats: {
                    total_investigations: data.stats.total_queries || 0,
                    success_rate: 0.95,
                    average_confidence: data.stats.average_confidence || 0.8
                }
            };
            updateSystemStatsDisplay();
        }
    })
    .catch(error => {
        console.error('Error loading RAG stats:', error);
    });
}

/**
 * Update system stats display
 */
function updateSystemStatsDisplay() {
    const statsContainer = document.getElementById('systemStats');
    if (!statsContainer || !systemStats.system_stats) return;
    
    const stats = systemStats.system_stats;
    
    statsContainer.innerHTML = `
        <div class="stats-grid">
            <div class="stat-item">
                <div class="stat-value">${stats.total_investigations || 0}</div>
                <div class="stat-label">Total Investigations</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${((stats.success_rate || 0) * 100).toFixed(1)}%</div>
                <div class="stat-label">Success Rate</div>
            </div>
            <div class="stat-item">
                <div class="stat-value">${((stats.average_confidence || 0) * 100).toFixed(1)}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>
    `;
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit query
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const queryInput = document.getElementById('queryInput');
            if (queryInput && document.activeElement === queryInput) {
                e.preventDefault();
                submitQuery();
            }
        }
        
        // Escape to close modal
        if (e.key === 'Escape') {
            const modal = document.getElementById('forensicQAModal');
            if (modal && modal.style.display === 'block') {
                closeForensicQAModal();
            }
        }
    });
}

/**
 * Close Forensic Q&A Modal
 */
function closeForensicQAModal() {
    const modal = document.getElementById('forensicQAModal');
    if (modal) {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
        console.log('Forensic Q&A modal closed');
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <span>${message}</span>
        <button onclick="this.parentElement.remove()">&times;</button>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

/**
 * Get CSRF token
 */
function getCSRFToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]')?.value || 
           document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
}

/**
 * Format response text with proper styling
 */
function formatResponseText(text) {
    // Convert line breaks to HTML
    let formatted = text.replace(/\n/g, '<br>');
    
    // Format bullet points
    formatted = formatted.replace(/^• (.+)$/gm, '<div class="bullet-point">• $1</div>');
    
    // Format headers (lines ending with :)
    formatted = formatted.replace(/^([^<\n]+:)$/gm, '<h5 class="response-header">$1</h5>');
    
    // Format confidence scores
    formatted = formatted.replace(/Confidence: ([\d.]+)/g, '<span class="confidence-inline">Confidence: $1</span>');
    
    // Format time references
    formatted = formatted.replace(/(\d+\.?\d*)s/g, '<span class="time-ref">$1s</span>');
    
    return formatted;
}

/**
 * Get confidence class for styling
 */
function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'high-confidence';
    if (confidence >= 0.6) return 'medium-confidence';
    return 'low-confidence';
}

/**
 * Load query history
 */
async function loadQueryHistory() {
    try {
        const response = await fetch('/api/rag/history/', {
            method: 'GET',
            headers: {
                'X-CSRFToken': getCSRFToken(),
                'Content-Type': 'application/json'
            }
        });
        
        if (response.ok) {
            const data = await response.json();
            displayQueryHistory(data.query_history || []);
        }
    } catch (error) {
        console.error('Error loading query history:', error);
    }
}

/**
 * Display query history
 */
function displayQueryHistory(history) {
    const historyContainer = document.getElementById('qaHistory');
    const historyList = document.getElementById('historyList');
    
    if (!historyContainer || !historyList) return;
    
    if (history.length === 0) {
        historyContainer.style.display = 'none';
        return;
    }
    
    historyList.innerHTML = '';
    
    history.slice(0, 5).forEach(item => { // Show last 5 queries
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        
        const queryText = document.createElement('div');
        queryText.className = 'history-query';
        queryText.textContent = item.query_text;
        queryText.onclick = () => {
            document.getElementById('queryInput').value = item.query_text;
        };
        
        const queryMeta = document.createElement('div');
        queryMeta.className = 'history-meta';
        const confidence = (item.response_confidence * 100).toFixed(1);
        queryMeta.textContent = `${confidence}% confidence • ${item.processing_time.toFixed(2)}s`;
        
        historyItem.appendChild(queryText);
        historyItem.appendChild(queryMeta);
        historyList.appendChild(historyItem);
    });
    
    historyContainer.style.display = 'block';
}

/**
 * Export current response
 */
function exportResponse() {
    if (!currentQuery) {
        showError('No response to export');
        return;
    }
    
    const exportData = {
        query: document.getElementById('queryInput').value,
        response: currentQuery.response_text,
        confidence_score: currentQuery.confidence_score,
        processing_time: currentQuery.processing_time,
        sources: currentQuery.sources,
        timestamp: new Date().toISOString(),
        retrieved_documents: currentQuery.retrieved_documents
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = `forensic_qa_response_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showSuccess('Response exported successfully');
}

/**
 * Clear current response
 */
function clearResponse() {
    const responseArea = document.getElementById('qaResponseArea');
    const responseContent = document.getElementById('responseContent');
    const responseSources = document.getElementById('responseSources');
    
    if (responseArea) responseArea.style.display = 'none';
    if (responseContent) responseContent.innerHTML = '';
    if (responseSources) responseSources.style.display = 'none';
    
    currentQuery = null;
}

/**
 * Setup keyboard shortcuts
 */
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit query
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const modal = document.getElementById('forensicQAModal');
            if (modal && modal.style.display === 'block') {
                e.preventDefault();
                submitQuery();
            }
        }
        
        // Escape to close modal
        if (e.key === 'Escape') {
            const modal = document.getElementById('forensicQAModal');
            if (modal && modal.style.display === 'block') {
                closeForensicQAModal();
            }
        }
    });
}

/**
 * Show loading indicator
 */
function showLoading(message = 'Processing...') {
    const loading = document.getElementById('qaLoading');
    if (loading) {
        loading.querySelector('p').textContent = message;
        loading.style.display = 'block';
    }
}

/**
 * Hide loading indicator
 */
function hideLoading() {
    const loading = document.getElementById('qaLoading');
    if (loading) {
        loading.style.display = 'none';
    }
}

/**
 * Show error message
 */
function showError(message) {
    // Create or update error display
    let errorDiv = document.getElementById('qaError');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.id = 'qaError';
        errorDiv.className = 'qa-error';
        
        const qaContainer = document.querySelector('.qa-container');
        if (qaContainer) {
            qaContainer.insertBefore(errorDiv, qaContainer.firstChild);
        }
    }
    
    errorDiv.innerHTML = `
        <i class="fas fa-exclamation-triangle"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.style.display='none'">×</button>
    `;
    errorDiv.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        if (errorDiv) errorDiv.style.display = 'none';
    }, 5000);
}

/**
 * Show success message
 */
function showSuccess(message) {
    // Create or update success display
    let successDiv = document.getElementById('qaSuccess');
    if (!successDiv) {
        successDiv = document.createElement('div');
        successDiv.id = 'qaSuccess';
        successDiv.className = 'qa-success';
        
        const qaContainer = document.querySelector('.qa-container');
        if (qaContainer) {
            qaContainer.insertBefore(successDiv, qaContainer.firstChild);
        }
    }
    
    successDiv.innerHTML = `
        <i class="fas fa-check-circle"></i>
        <span>${message}</span>
        <button onclick="this.parentElement.style.display='none'">×</button>
    `;
    successDiv.style.display = 'block';
    
    // Auto-hide after 3 seconds
    setTimeout(() => {
        if (successDiv) successDiv.style.display = 'none';
    }, 3000);
}

/**
 * Get CSRF token for API requests
 */
function getCSRFToken() {
    const token = document.querySelector('[name=csrfmiddlewaretoken]');
    return token ? token.value : '';
}
