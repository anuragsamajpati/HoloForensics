/**
 * Scene Analysis Modal JavaScript
 * Handles scene graph generation and event detection workflow
 */

// Global variables
let currentSceneAnalysisStep = 1;
let sceneAnalysisJobId = null;
let sceneAnalysisInterval = null;
let sceneData = null;
let validationResults = null;

// Modal management
function openSceneAnalysisModal() {
    const modal = document.getElementById('sceneAnalysisModal');
    modal.style.display = 'block';
    resetSceneAnalysisModal();
    initializeSceneAnalysisHandlers();
}

function closeSceneAnalysisModal() {
    const modal = document.getElementById('sceneAnalysisModal');
    modal.style.display = 'none';
    
    // Clean up any active processes
    if (sceneAnalysisInterval) {
        clearInterval(sceneAnalysisInterval);
        sceneAnalysisInterval = null;
    }
    
    resetSceneAnalysisModal();
}

function resetSceneAnalysisModal() {
    currentSceneAnalysisStep = 1;
    sceneAnalysisJobId = null;
    sceneData = null;
    validationResults = null;
    
    // Reset step indicator
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'completed');
    });
    document.querySelector('.step[data-step="1"]').classList.add('active');
    
    // Show only first step
    document.querySelectorAll('.step-content').forEach(content => {
        content.style.display = 'none';
    });
    document.getElementById('sceneAnalysisStep1').style.display = 'block';
    
    // Reset form elements
    document.getElementById('sceneDataFile').value = '';
    document.getElementById('sceneDataManual').value = '';
    document.getElementById('sceneFileInfo').style.display = 'none';
    document.getElementById('sceneValidationResults').style.display = 'none';
    document.getElementById('validateSceneDataBtn').style.display = 'inline-block';
    document.getElementById('nextToConfigBtn').style.display = 'none';
    
    // Reset input method
    document.querySelector('input[name="sceneInputMethod"][value="upload"]').checked = true;
    showSceneInputPanel('upload');
}

function initializeSceneAnalysisHandlers() {
    // Input method selector
    document.querySelectorAll('input[name="sceneInputMethod"]').forEach(radio => {
        radio.addEventListener('change', function() {
            showSceneInputPanel(this.value);
        });
    });
    
    // File input handler
    const fileInput = document.getElementById('sceneDataFile');
    fileInput.addEventListener('change', handleSceneFileSelect);
    
    // Range input handler for confidence threshold
    const confidenceRange = document.getElementById('sceneConfidenceThreshold');
    if (confidenceRange) {
        confidenceRange.addEventListener('input', function() {
            const valueSpan = this.nextElementSibling;
            if (valueSpan) {
                valueSpan.textContent = this.value;
            }
        });
    }
    
    // Check for URL parameters to auto-open
    const urlParams = new URLSearchParams(window.location.search);
    if (urlParams.get('tool') === 'scene-analysis') {
        openSceneAnalysisModal();
    }
}

function showSceneInputPanel(method) {
    // Hide all panels
    document.getElementById('sceneUploadPanel').style.display = 'none';
    document.getElementById('sceneManualPanel').style.display = 'none';
    document.getElementById('sceneExistingPanel').style.display = 'none';
    
    // Show selected panel
    switch(method) {
        case 'upload':
            document.getElementById('sceneUploadPanel').style.display = 'block';
            break;
        case 'manual':
            document.getElementById('sceneManualPanel').style.display = 'block';
            break;
        case 'existing':
            document.getElementById('sceneExistingPanel').style.display = 'block';
            break;
    }
    
    // Reset validation results
    document.getElementById('sceneValidationResults').style.display = 'none';
    document.getElementById('validateSceneDataBtn').style.display = 'inline-block';
    document.getElementById('nextToConfigBtn').style.display = 'none';
}

function handleSceneFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const fileInfo = document.getElementById('sceneFileInfo');
        const fileName = fileInfo.querySelector('.file-name');
        const fileSize = fileInfo.querySelector('.file-size');
        
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        fileInfo.style.display = 'flex';
        
        // Read file content
        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                sceneData = JSON.parse(e.target.result);
                addSceneAnalysisLog('File loaded successfully', 'success');
            } catch (error) {
                addSceneAnalysisLog('Error parsing file: ' + error.message, 'error');
                sceneData = null;
            }
        };
        reader.readAsText(file);
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Step navigation
function nextSceneAnalysisStep() {
    if (currentSceneAnalysisStep < 4) {
        // Hide current step
        document.getElementById(`sceneAnalysisStep${currentSceneAnalysisStep}`).style.display = 'none';
        document.querySelector(`.step[data-step="${currentSceneAnalysisStep}"]`).classList.remove('active');
        document.querySelector(`.step[data-step="${currentSceneAnalysisStep}"]`).classList.add('completed');
        
        // Show next step
        currentSceneAnalysisStep++;
        document.getElementById(`sceneAnalysisStep${currentSceneAnalysisStep}`).style.display = 'block';
        document.querySelector(`.step[data-step="${currentSceneAnalysisStep}"]`).classList.add('active');
    }
}

function prevSceneAnalysisStep() {
    if (currentSceneAnalysisStep > 1) {
        // Hide current step
        document.getElementById(`sceneAnalysisStep${currentSceneAnalysisStep}`).style.display = 'none';
        document.querySelector(`.step[data-step="${currentSceneAnalysisStep}"]`).classList.remove('active');
        
        // Show previous step
        currentSceneAnalysisStep--;
        document.getElementById(`sceneAnalysisStep${currentSceneAnalysisStep}`).style.display = 'block';
        document.querySelector(`.step[data-step="${currentSceneAnalysisStep}"]`).classList.remove('completed');
        document.querySelector(`.step[data-step="${currentSceneAnalysisStep}"]`).classList.add('active');
    }
}

// Data validation
async function validateSceneData() {
    const inputMethod = document.querySelector('input[name="sceneInputMethod"]:checked').value;
    let dataToValidate = null;
    
    try {
        // Get data based on input method
        switch(inputMethod) {
            case 'upload':
                if (!sceneData) {
                    showNotification('Please select a file first', 'error');
                    return;
                }
                dataToValidate = sceneData;
                break;
                
            case 'manual':
                const manualData = document.getElementById('sceneDataManual').value.trim();
                if (!manualData) {
                    showNotification('Please enter scene data', 'error');
                    return;
                }
                try {
                    dataToValidate = JSON.parse(manualData);
                } catch (error) {
                    showNotification('Invalid JSON format: ' + error.message, 'error');
                    return;
                }
                break;
                
            case 'existing':
                const selectedScene = document.getElementById('existingSceneSelect').value;
                if (!selectedScene) {
                    showNotification('Please select an existing scene', 'error');
                    return;
                }
                // For demo purposes, use sample data
                dataToValidate = generateSampleSceneData();
                break;
        }
        
        // Show loading state
        const validateBtn = document.getElementById('validateSceneDataBtn');
        const originalText = validateBtn.textContent;
        validateBtn.textContent = 'Validating...';
        validateBtn.disabled = true;
        
        // Call validation API
        const response = await fetch('/api/scene-analysis/validate/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({
                scene_data: dataToValidate
            })
        });
        
        const result = await response.json();
        
        if (result.success && result.valid) {
            validationResults = result;
            displayValidationResults(result);
            showNotification('Scene data validated successfully', 'success');
            
            // Show next button
            validateBtn.style.display = 'none';
            document.getElementById('nextToConfigBtn').style.display = 'inline-block';
        } else {
            displayValidationResults(result);
            showNotification('Validation failed: ' + (result.error || 'Invalid data format'), 'error');
        }
        
    } catch (error) {
        console.error('Validation error:', error);
        showNotification('Validation failed: ' + error.message, 'error');
    } finally {
        // Reset button state
        const validateBtn = document.getElementById('validateSceneDataBtn');
        validateBtn.textContent = originalText;
        validateBtn.disabled = false;
    }
}

function displayValidationResults(results) {
    const resultsDiv = document.getElementById('sceneValidationResults');
    const stats = results.statistics || {};
    
    // Update statistics
    document.getElementById('totalFrames').textContent = stats.total_frames || 0;
    document.getElementById('totalObjects').textContent = stats.total_objects || 0;
    document.getElementById('objectTypes').textContent = (stats.object_types || []).join(', ') || 'None';
    
    const timeRange = stats.time_range || [0, 0];
    document.getElementById('timeRange').textContent = `${timeRange[0].toFixed(2)}s - ${timeRange[1].toFixed(2)}s`;
    
    // Show issues if any
    const issuesDiv = document.getElementById('validationIssues');
    const issuesList = document.getElementById('issuesList');
    
    if (results.issues && results.issues.length > 0) {
        issuesList.innerHTML = '';
        results.issues.forEach(issue => {
            const li = document.createElement('li');
            li.textContent = issue;
            issuesList.appendChild(li);
        });
        issuesDiv.style.display = 'block';
    } else {
        issuesDiv.style.display = 'none';
    }
    
    resultsDiv.style.display = 'block';
}

function generateSampleSceneData() {
    return [
        {
            frame_id: 1,
            timestamp: 0.0,
            camera_id: "cam_001",
            objects: [
                {
                    object_id: "person_001",
                    object_type: "person",
                    confidence: 0.95,
                    bbox: [100, 100, 200, 300],
                    center: [150, 200],
                    area: 20000
                },
                {
                    object_id: "vehicle_001",
                    object_type: "car",
                    confidence: 0.88,
                    bbox: [300, 150, 500, 250],
                    center: [400, 200],
                    area: 20000
                }
            ]
        },
        {
            frame_id: 2,
            timestamp: 0.033,
            camera_id: "cam_001",
            objects: [
                {
                    object_id: "person_001",
                    object_type: "person",
                    confidence: 0.93,
                    bbox: [105, 105, 205, 305],
                    center: [155, 205],
                    area: 20000
                }
            ]
        }
    ];
}

// Analysis execution
async function startSceneAnalysis() {
    try {
        // Collect configuration
        const config = {
            case_id: document.getElementById('sceneAnalysisCaseId').value || 'CASE_2024_001',
            analysis_type: document.getElementById('sceneAnalysisType').value,
            confidence_threshold: parseFloat(document.getElementById('sceneConfidenceThreshold').value),
            temporal_window: parseInt(document.getElementById('temporalWindow').value),
            proximity_threshold: parseFloat(document.getElementById('spatialThreshold').value),
            enable_scene_graphs: document.getElementById('enableSceneGraphs').checked,
            enable_event_detection: document.getElementById('enableEventDetection').checked,
            enable_chain_of_custody: document.getElementById('enableChainOfCustody').checked,
            generate_visualizations: document.getElementById('generateVisualizations').checked,
            total_frames: validationResults?.statistics?.total_frames || 0
        };
        
        // Start analysis job
        const response = await fetch('/api/scene-analysis/start/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify(config)
        });
        
        const result = await response.json();
        
        if (result.success) {
            sceneAnalysisJobId = result.job_id;
            addSceneAnalysisLog(`Analysis job started: ${sceneAnalysisJobId}`, 'success');
            
            // Move to processing step
            nextSceneAnalysisStep();
            
            // Start processing scene data
            await processSceneData();
            
        } else {
            throw new Error(result.error || 'Failed to start analysis');
        }
        
    } catch (error) {
        console.error('Analysis start error:', error);
        showNotification('Failed to start analysis: ' + error.message, 'error');
    }
}

async function processSceneData() {
    if (!sceneAnalysisJobId) return;
    
    try {
        // Get scene data based on input method
        const inputMethod = document.querySelector('input[name="sceneInputMethod"]:checked').value;
        let dataToProcess = null;
        
        switch(inputMethod) {
            case 'upload':
                dataToProcess = sceneData;
                break;
            case 'manual':
                dataToProcess = JSON.parse(document.getElementById('sceneDataManual').value);
                break;
            case 'existing':
                dataToProcess = generateSampleSceneData();
                break;
        }
        
        if (!dataToProcess || !Array.isArray(dataToProcess)) {
            throw new Error('Invalid scene data format');
        }
        
        // Process each frame
        for (let i = 0; i < dataToProcess.length; i++) {
            const frameData = dataToProcess[i];
            
            try {
                const response = await fetch('/api/scene-analysis/process-frame/', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': getCSRFToken()
                    },
                    body: JSON.stringify({
                        job_id: sceneAnalysisJobId,
                        frame_data: frameData
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // Update progress
                    const progress = ((i + 1) / dataToProcess.length) * 100;
                    updateSceneAnalysisProgress(progress, `Processing frame ${i + 1}/${dataToProcess.length}`);
                    
                    // Update stats
                    updateProcessingStats(result.frame_result);
                    
                    addSceneAnalysisLog(`Processed frame ${frameData.frame_id}: ${result.frame_result.events_detected} events detected`, 'info');
                    
                } else {
                    addSceneAnalysisLog(`Error processing frame ${frameData.frame_id}: ${result.error}`, 'error');
                }
                
            } catch (error) {
                addSceneAnalysisLog(`Error processing frame ${frameData.frame_id}: ${error.message}`, 'error');
            }
            
            // Small delay to prevent overwhelming the server
            await new Promise(resolve => setTimeout(resolve, 100));
        }
        
        // Finalize analysis
        await finalizeSceneAnalysis();
        
    } catch (error) {
        console.error('Processing error:', error);
        addSceneAnalysisLog('Processing failed: ' + error.message, 'error');
        showNotification('Processing failed: ' + error.message, 'error');
    }
}

async function finalizeSceneAnalysis() {
    if (!sceneAnalysisJobId) return;
    
    try {
        updateSceneAnalysisProgress(100, 'Finalizing analysis...');
        
        const response = await fetch('/api/scene-analysis/finalize/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCSRFToken()
            },
            body: JSON.stringify({
                job_id: sceneAnalysisJobId
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            addSceneAnalysisLog('Analysis completed successfully', 'success');
            
            // Display results
            displaySceneAnalysisResults(result);
            
            // Move to results step
            nextSceneAnalysisStep();
            
        } else {
            throw new Error(result.error || 'Failed to finalize analysis');
        }
        
    } catch (error) {
        console.error('Finalization error:', error);
        addSceneAnalysisLog('Finalization failed: ' + error.message, 'error');
        showNotification('Analysis finalization failed: ' + error.message, 'error');
    }
}

function updateSceneAnalysisProgress(percentage, status) {
    const progressFill = document.getElementById('sceneAnalysisProgress');
    const progressText = document.getElementById('sceneAnalysisProgressText');
    const statusText = document.getElementById('sceneAnalysisStatus');
    
    progressFill.style.width = percentage + '%';
    progressText.textContent = Math.round(percentage) + '%';
    statusText.textContent = status;
}

function updateProcessingStats(frameResult) {
    // Update processing stats in real-time
    const processedFrames = document.getElementById('processedFrames');
    const detectedEvents = document.getElementById('detectedEvents');
    const evidenceItems = document.getElementById('evidenceItems');
    const sceneGraphs = document.getElementById('sceneGraphs');
    
    processedFrames.textContent = parseInt(processedFrames.textContent) + 1;
    detectedEvents.textContent = parseInt(detectedEvents.textContent) + (frameResult.events_detected || 0);
    evidenceItems.textContent = parseInt(evidenceItems.textContent) + (frameResult.evidence_created || 0);
    sceneGraphs.textContent = parseInt(sceneGraphs.textContent) + (frameResult.scene_graph_objects > 0 ? 1 : 0);
}

function displaySceneAnalysisResults(results) {
    const summary = results.results_summary || {};
    
    // Update result summary cards
    document.getElementById('resultSceneGraphs').textContent = summary.total_scene_graphs || 0;
    document.getElementById('resultEvents').textContent = summary.total_events || 0;
    document.getElementById('resultEvidence').textContent = summary.total_evidence || 0;
    document.getElementById('resultQuality').textContent = (summary.quality_score || 0).toFixed(2);
    
    // Display critical events
    displayCriticalEvents(results);
}

function displayCriticalEvents(results) {
    const eventsList = document.getElementById('criticalEventsList');
    eventsList.innerHTML = '';
    
    // For demo purposes, create sample critical events
    const sampleEvents = [
        {
            event_type: 'suspicious_movement',
            severity: 'HIGH',
            timestamp: 2.5,
            description: 'Rapid movement detected near restricted area',
            confidence: 0.92
        },
        {
            event_type: 'object_interaction',
            severity: 'MEDIUM',
            timestamp: 5.1,
            description: 'Person-vehicle interaction detected',
            confidence: 0.78
        }
    ];
    
    sampleEvents.forEach(event => {
        const eventDiv = document.createElement('div');
        eventDiv.className = 'event-item';
        eventDiv.innerHTML = `
            <div class="event-header">
                <span class="event-type">${event.event_type.replace('_', ' ').toUpperCase()}</span>
                <span class="event-severity severity-${event.severity.toLowerCase()}">${event.severity}</span>
            </div>
            <div class="event-description">${event.description}</div>
            <div class="event-meta">
                <span>Time: ${event.timestamp}s</span>
                <span>Confidence: ${(event.confidence * 100).toFixed(1)}%</span>
            </div>
        `;
        eventsList.appendChild(eventDiv);
    });
}

// Utility functions
function addSceneAnalysisLog(message, level = 'info') {
    const logsContainer = document.getElementById('sceneAnalysisLogs');
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${level}`;
    
    const timestamp = new Date().toLocaleTimeString();
    logEntry.innerHTML = `
        <span class="log-time">${timestamp}</span>
        <span class="log-message">${message}</span>
    `;
    
    logsContainer.appendChild(logEntry);
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

function cancelSceneAnalysis() {
    if (sceneAnalysisInterval) {
        clearInterval(sceneAnalysisInterval);
        sceneAnalysisInterval = null;
    }
    
    addSceneAnalysisLog('Analysis cancelled by user', 'warning');
    showNotification('Analysis cancelled', 'warning');
    closeSceneAnalysisModal();
}

function startNewSceneAnalysis() {
    resetSceneAnalysisModal();
}

// Export functions
function downloadSceneAnalysisReport() {
    if (!sceneAnalysisJobId) return;
    
    // Create download link for forensic report
    const link = document.createElement('a');
    link.href = `/api/scene-analysis/results/${sceneAnalysisJobId}/`;
    link.download = `scene_analysis_report_${sceneAnalysisJobId}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showNotification('Report download started', 'success');
}

function downloadSceneGraph() {
    showNotification('Scene graph visualization download started', 'success');
}

function downloadEventTimeline() {
    showNotification('Event timeline download started', 'success');
}

function viewDetailedResults() {
    if (!sceneAnalysisJobId) return;
    
    // Open detailed results in new window
    window.open(`/api/scene-analysis/results/${sceneAnalysisJobId}/`, '_blank');
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeSceneAnalysisHandlers();
});
