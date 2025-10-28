// HoloForensics Scenes Page JavaScript

// Load processed scenes on page load
document.addEventListener('DOMContentLoaded', function() {
    loadProcessedScenes();
    
    // Check URL parameters for tool activation
    const urlParams = new URLSearchParams(window.location.search);
    const tool = urlParams.get('tool');
    
    if (tool === 'video-inpainting') {
        openInpaintingModal();
    } else if (tool === 'physics-prediction') {
        openPhysicsModal();
    } else if (tool === 'forensic-qa') {
        openForensicQA();
    } else if (tool === 'scene-analysis') {
        openSceneAnalysisTool();
    }
});

// Load processed scenes from API
async function loadProcessedScenes() {
    try {
        const response = await fetch('/api/scenes/list/');
        const data = await response.json();
        
        if (data.scenes && data.scenes.length > 0) {
            displayProcessedScenes(data.scenes);
            updateSceneStats(data.scenes);
        }
    } catch (error) {
        console.error('Error loading processed scenes:', error);
    }
}

// Display processed scenes in the UI
function displayProcessedScenes(scenes) {
    const scenesContainer = document.querySelector('.scenes-grid') || createScenesGrid();
    
    scenes.forEach(scene => {
        const sceneCard = createSceneCard(scene);
        scenesContainer.appendChild(sceneCard);
    });
}

// Create scenes grid if it doesn't exist
function createScenesGrid() {
    const container = document.createElement('div');
    container.className = 'scenes-grid';
    container.style.cssText = `
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin: 20px 0;
    `;
    
    // Insert after the upload section
    const uploadSection = document.querySelector('.upload-section');
    if (uploadSection) {
        uploadSection.parentNode.insertBefore(container, uploadSection.nextSibling);
    }
    
    return container;
}

// Create scene card
function createSceneCard(scene) {
    const card = document.createElement('div');
    card.className = 'scene-card';
    card.style.cssText = `
        background: #2d3436;
        border: 1px solid #636e72;
        border-radius: 8px;
        padding: 20px;
        color: white;
    `;
    
    const availableData = scene.available_data || [];
    const dataTypes = availableData.join(', ') || 'No data available';
    
    card.innerHTML = `
        <h3 style="color: #ff4757; margin-bottom: 10px;">${scene.scene_id}</h3>
        <p><strong>Status:</strong> <span style="color: ${scene.status === 'completed' ? '#00b894' : '#fdcb6e'}">${scene.status}</span></p>
        <p><strong>Available Data:</strong> ${dataTypes}</p>
        ${scene.metadata ? `
            <p><strong>Cameras:</strong> ${scene.metadata.processing_stats?.total_cameras || 0}</p>
            <p><strong>Keyframes:</strong> ${scene.metadata.processing_stats?.total_keyframes || 0}</p>
        ` : ''}
        <div style="margin-top: 15px;">
            <button onclick="viewScene('${scene.scene_id}')" class="btn btn-primary" style="margin-right: 10px;">View Details</button>
            <button onclick="open3DViewer('${scene.scene_id}')" class="btn btn-secondary">View in 3D</button>
        </div>
    `;
    
    return card;
}

// Update scene statistics
function updateSceneStats(scenes) {
    const completedScenes = scenes.filter(s => s.status === 'completed').length;
    const totalScenes = scenes.length;
    
    // Update stats in the UI if elements exist
    const statsElements = {
        'total-scenes': totalScenes,
        'completed-scenes': completedScenes,
        'processing-scenes': totalScenes - completedScenes
    };
    
    Object.entries(statsElements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
}

// Modal functionality
function openModal() {
    document.getElementById('createSceneModal').style.display = 'block';
}

function closeModal() {
    document.getElementById('createSceneModal').style.display = 'none';
    document.getElementById('physicsModal').style.display = 'none';
    document.getElementById('sceneAnalysisModal').style.display = 'none';
    document.getElementById('forensicQAModal').style.display = 'none';
    document.getElementById('createSceneForm').reset();
}

function closeSceneAnalysisModal() {
    document.getElementById('sceneAnalysisModal').style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('createSceneModal');
    if (event.target === modal) {
        closeModal();
    }
}

// Scene actions
function viewScene(sceneId) {
    // Load scene details
    fetch(`/api/scene/${sceneId}/results/`)
        .then(response => response.json())
        .then(data => {
            console.log('Scene data:', data);
            alert(`Scene ${sceneId} details:\n${JSON.stringify(data.available_data, null, 2)}`);
        })
        .catch(error => {
            console.error('Error loading scene details:', error);
        });
    showNotification(`Opening ${sceneId} scene details...`, 'info');
    setTimeout(() => {
        window.location.href = '/analysis/';
    }, 1500);
}

function viewProgress(sceneId) {
    // Simulate viewing progress
    showNotification(`Loading ${sceneId} progress...`, 'info');
    setTimeout(() => {
        window.location.href = '/analysis/';
    }, 1500);
}

function openQA(sceneId) {
    // Simulate opening Q&A
    showNotification(`Opening ${sceneId} Q&A system...`, 'info');
    setTimeout(() => {
        window.location.href = '/analysis/';
    }, 1500);
}

// 3D Viewer functionality
function open3DViewer(sceneId) {
    showNotification(`Loading 3D scene viewer for ${sceneId}...`, 'info');
    
    // Show the 3D viewer
    const viewer = document.getElementById('scene3DViewer');
    if (viewer) {
        viewer.style.display = 'flex';
        
        // Initialize the 3D viewer with scene data
        if (window.SceneViewerIntegration) {
            window.SceneViewerIntegration.loadScene(sceneId);
        }
    }
}

function close3DViewer() {
    const viewer = document.getElementById('scene3DViewer');
    if (viewer) {
        viewer.style.display = 'none';
        
        // Cleanup 3D viewer resources
        if (window.SceneViewerIntegration) {
            window.SceneViewerIntegration.cleanup();
        }
    }
}

// Form submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('createSceneForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const formData = new FormData(form);
            const sceneName = formData.get('sceneName');
            const sceneType = formData.get('sceneType');
            
            // Simulate scene creation
            showNotification(`Creating scene "${sceneName}"...`, 'success');
            
            setTimeout(() => {
                closeModal();
                showNotification(`Scene "${sceneName}" created successfully!`, 'success');
                
                // Redirect to analysis dashboard after creation
                setTimeout(() => {
                    window.location.href = '/analysis/';
                }, 2000);
            }, 1500);
        });
    }
});

// Notification system
function showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(notification => notification.remove());
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getNotificationIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button class="notification-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Add notification styles
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${getNotificationColor(type)};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        z-index: 3000;
        display: flex;
        align-items: center;
        gap: 15px;
        min-width: 300px;
        animation: slideInRight 0.3s ease-out;
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 4 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        }
    }, 4000);
}

function getNotificationIcon(type) {
    switch(type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        default: return 'info-circle';
    }
}

function getNotificationColor(type) {
    switch(type) {
        case 'success': return '#2ecc71';
        case 'error': return '#e74c3c';
        case 'warning': return '#f39c12';
        default: return '#3498db';
    }
}

// Add CSS animations for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(100%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideOutRight {
        from {
            opacity: 1;
            transform: translateX(0);
        }
        to {
            opacity: 0;
            transform: translateX(100%);
        }
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
    }
    
    .notification-close {
        background: none;
        border: none;
        color: white;
        cursor: pointer;
        padding: 5px;
        border-radius: 4px;
        transition: background 0.3s ease;
    }
    
    .notification-close:hover {
        background: rgba(255, 255, 255, 0.2);
    }
`;
document.head.appendChild(style);

// Scene card hover effects
document.addEventListener('DOMContentLoaded', function() {
    const sceneCards = document.querySelectorAll('.scene-card');
    
    sceneCards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px) scale(1.02)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0) scale(1)';
        });
    });
});

// Smooth scrolling for navigation
document.addEventListener('DOMContentLoaded', function() {
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            
            // Only prevent default for anchor links
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            }
        });
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + N to create new scene
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        createNewScene();
    }
    
    // Escape to close modal
    if (e.key === 'Escape') {
        closeModal();
    }
});

// Loading animation for scene actions
function addLoadingState(button) {
    const originalText = button.innerHTML;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading...';
    button.disabled = true;
    
    return function() {
        button.innerHTML = originalText;
        button.disabled = false;
    };
}

// Video Inpainting Functionality
let currentInpaintingJob = null;

function openInpaintingModal(sceneId) {
    document.getElementById('inpaintingModal').style.display = 'block';
    document.getElementById('sceneSelect').value = sceneId;
    showInpaintingStep(1);
}

function closeInpaintingModal() {
    document.getElementById('inpaintingModal').style.display = 'none';
    resetInpaintingModal();
}

function resetInpaintingModal() {
    // Reset all steps
    document.querySelectorAll('.inpainting-step').forEach(step => {
        step.classList.remove('active');
        step.style.display = 'none';
    });
    
    // Reset form fields
    document.getElementById('caseId').value = '';
    document.getElementById('sceneSelect').value = '';
    document.getElementById('sceneValidation').style.display = 'none';
    document.getElementById('processingResults').style.display = 'none';
    
    // Reset progress
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressText').textContent = 'Initializing...';
    document.getElementById('processingLog').innerHTML = '<div class="log-entry">🔧 Initializing video inpainting pipeline...</div>';
    
    currentInpaintingJob = null;
}

function showInpaintingStep(stepNumber) {
    // Hide all steps
    document.querySelectorAll('.inpainting-step').forEach(step => {
        step.classList.remove('active');
        step.style.display = 'none';
    });
    
    // Show target step
    const targetStep = document.getElementById(`step${stepNumber}`);
    if (targetStep) {
        targetStep.classList.add('active');
        targetStep.style.display = 'block';
    }
}

function previousStep() {
    const activeStep = document.querySelector('.inpainting-step.active');
    if (activeStep) {
        const stepId = activeStep.id;
        const stepNumber = parseInt(stepId.replace('step', ''));
        if (stepNumber > 1) {
            showInpaintingStep(stepNumber - 1);
        }
    }
}

async function validateScene() {
    const caseId = document.getElementById('caseId').value.trim();
    const sceneId = document.getElementById('sceneSelect').value;
    
    if (!caseId || !sceneId) {
        showNotification('Please enter Case ID and select a scene', 'error');
        return;
    }
    
    try {
        showNotification('Validating scene...', 'info');
        
        const response = await fetch('/api/inpainting/validate/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({
                scene_id: sceneId
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.ready_for_inpainting) {
            // Show validation results
            const sceneInfo = data.scene_info;
            document.getElementById('sceneInfo').innerHTML = `
                <div class="scene-validation-info">
                    <p><strong>Scene ID:</strong> ${sceneInfo.scene_id}</p>
                    <p><strong>Video Files:</strong> ${sceneInfo.video_count}</p>
                    <p><strong>Total Size:</strong> ${sceneInfo.total_size_mb} MB</p>
                    <div class="video-list">
                        <h6>Videos:</h6>
                        ${sceneInfo.videos.map(video => 
                            `<div class="video-item">
                                <span>${video.filename}</span>
                                <span class="video-size">${video.size_mb} MB</span>
                            </div>`
                        ).join('')}
                    </div>
                </div>
            `;
            
            document.getElementById('sceneValidation').style.display = 'block';
            showNotification('Scene validated successfully!', 'success');
            
            // Enable next step
            setTimeout(() => {
                showInpaintingStep(2);
            }, 1500);
            
        } else {
            showNotification(data.error || 'Scene is not ready for inpainting', 'error');
        }
        
    } catch (error) {
        console.error('Validation error:', error);
        showNotification('Failed to validate scene', 'error');
    }
}

async function startInpainting() {
    const caseId = document.getElementById('caseId').value.trim();
    const sceneId = document.getElementById('sceneSelect').value;
    
    // Get configuration options
    const config = {
        evidence_preservation: document.getElementById('evidencePreservation').checked,
        quality_validation: document.getElementById('qualityValidation').checked,
        chain_of_custody: document.getElementById('chainOfCustody').checked,
        backup_originals: document.getElementById('backupOriginals').checked
    };
    
    try {
        showInpaintingStep(3);
        updateProcessingLog('🚀 Starting video inpainting job...', 'info');
        updateProgress(10, 'Initializing inpainting pipeline...');
        
        const response = await fetch('/api/inpainting/start/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': getCsrfToken()
            },
            body: JSON.stringify({
                scene_id: sceneId,
                case_id: caseId,
                ...config
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentInpaintingJob = data.job_id;
            updateProcessingLog(`✅ Job started with ID: ${data.job_id}`, 'success');
            
            if (data.status === 'completed') {
                // Job completed immediately
                handleInpaintingCompletion(data.processing_report);
            } else {
                // Monitor job progress
                monitorInpaintingJob(data.job_id);
            }
            
        } else {
            updateProcessingLog(`❌ Failed to start inpainting: ${data.error}`, 'error');
            showNotification('Failed to start inpainting job', 'error');
        }
        
    } catch (error) {
        console.error('Inpainting error:', error);
        updateProcessingLog(`❌ Error: ${error.message}`, 'error');
        showNotification('Failed to start inpainting', 'error');
    }
}

async function monitorInpaintingJob(jobId) {
    const maxAttempts = 60; // 5 minutes with 5-second intervals
    let attempts = 0;
    
    const checkStatus = async () => {
        try {
            const response = await fetch(`/api/inpainting/status/${jobId}/`);
            const data = await response.json();
            
            if (data.success) {
                const jobData = data.job_data;
                
                switch (jobData.status) {
                    case 'started':
                        updateProgress(30, 'Processing videos...');
                        updateProcessingLog('🔧 Processing videos with inpainting...', 'info');
                        break;
                        
                    case 'completed':
                        handleInpaintingCompletion(jobData.processing_report);
                        return;
                        
                    case 'failed':
                        updateProcessingLog(`❌ Job failed: ${jobData.error}`, 'error');
                        showNotification('Inpainting job failed', 'error');
                        return;
                }
                
                // Continue monitoring
                attempts++;
                if (attempts < maxAttempts) {
                    setTimeout(checkStatus, 5000); // Check every 5 seconds
                } else {
                    updateProcessingLog('⏰ Job monitoring timeout', 'warning');
                    showNotification('Job monitoring timeout - check results manually', 'warning');
                }
                
            } else {
                updateProcessingLog(`❌ Status check failed: ${data.error}`, 'error');
            }
            
        } catch (error) {
            console.error('Status check error:', error);
            updateProcessingLog(`❌ Status check error: ${error.message}`, 'error');
        }
    };
    
    // Start monitoring
    setTimeout(checkStatus, 2000); // First check after 2 seconds
}

function handleInpaintingCompletion(report) {
    updateProgress(100, 'Processing completed!');
    updateProcessingLog('🎉 Video inpainting completed successfully!', 'success');
    
    // Show results
    const resultsContent = document.getElementById('resultsContent');
    resultsContent.innerHTML = `
        <div class="inpainting-results">
            <div class="result-summary">
                <h6>Processing Summary</h6>
                <p><strong>Total Videos:</strong> ${report.processed_videos.length}</p>
                <p><strong>Success Rate:</strong> ${report.success_rate * 100}%</p>
                <p><strong>Compliance Status:</strong> <span class="status-${report.compliance_status.toLowerCase()}">${report.compliance_status}</span></p>
            </div>
            
            <div class="processed-videos">
                <h6>Processed Videos</h6>
                ${report.processed_videos.map(video => `
                    <div class="video-result ${video.status}">
                        <span class="video-name">${video.camera_id}</span>
                        <span class="video-status">${video.status}</span>
                        ${video.inpainting_stats ? `<span class="frames-inpainted">${video.inpainting_stats.frames_inpainted} frames inpainted</span>` : ''}
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    document.getElementById('processingResults').style.display = 'block';
    showNotification('Video inpainting completed successfully!', 'success');
}

function updateProgress(percentage, text) {
    document.getElementById('progressFill').style.width = `${percentage}%`;
    document.getElementById('progressText').textContent = text;
}

function updateProcessingLog(message, type = 'info') {
    const logContainer = document.getElementById('processingLog');
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry ${type}`;
    logEntry.textContent = message;
    
    logContainer.appendChild(logEntry);
    logContainer.scrollTop = logContainer.scrollHeight;
}

async function viewResults() {
    if (currentInpaintingJob) {
        try {
            const response = await fetch(`/api/inpainting/results/${currentInpaintingJob}/`);
            const data = await response.json();
            
            if (data.success) {
                // Open results in new tab or show detailed view
                showNotification('Opening detailed results...', 'info');
                // Could redirect to a dedicated results page
                console.log('Inpainting results:', data);
            } else {
                showNotification('Failed to load results', 'error');
            }
        } catch (error) {
            console.error('Results error:', error);
            showNotification('Failed to load results', 'error');
        }
    }
}

function getCsrfToken() {
    const cookies = document.cookie.split(';');
    for (let cookie of cookies) {
        const [name, value] = cookie.trim().split('=');
        if (name === 'csrftoken') {
            return value;
        }
    }
    return '';
}

// Create New Scene Function
function createNewScene() {
    const modal = document.getElementById('createSceneModal');
    if (modal) {
        modal.style.display = 'block';
    } else {
        // Fallback: create a simple form
        const sceneName = prompt('Enter scene name:');
        if (sceneName) {
            showNotification(`Creating scene: ${sceneName}`, 'success');
            // You can add actual scene creation logic here
            setTimeout(() => {
                showNotification('Scene created successfully!', 'success');
            }, 1500);
        }
    }
}

// Close Modal Function
function closeModal() {
    const modal = document.getElementById('createSceneModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

// Show Notification Function
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#00b894' : type === 'error' ? '#e17055' : '#0984e3'};
        color: white;
        padding: 15px 20px;
        border-radius: 5px;
        z-index: 10000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        font-family: Arial, sans-serif;
    `;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    // Remove after 3 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 3000);
}

// Update modal close functionality to include inpainting modal
window.onclick = function(event) {
    const createModal = document.getElementById('createSceneModal');
    const inpaintingModal = document.getElementById('inpaintingModal');
    
    if (event.target === createModal) {
        closeModal();
    } else if (event.target === inpaintingModal) {
        closeInpaintingModal();
    }
}

// Scene Analysis Tool Functions
function openSceneAnalysisTool() {
    const modal = document.getElementById('sceneAnalysisModal');
    if (modal) {
        modal.style.display = 'block';
        showSceneAnalysisStep(1);
    } else {
        showNotification('Opening Scene Analysis tool...', 'info');
        setTimeout(() => {
            window.location.href = '/scenes/?tool=scene-analysis';
        }, 1000);
    }
}

function closeSceneAnalysisModal() {
    const modal = document.getElementById('sceneAnalysisModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function showSceneAnalysisStep(stepNumber) {
    document.querySelectorAll('.scene-analysis-step').forEach(step => {
        step.classList.remove('active');
        step.style.display = 'none';
    });
    
    const targetStep = document.getElementById(`sceneStep${stepNumber}`);
    if (targetStep) {
        targetStep.classList.add('active');
        targetStep.style.display = 'block';
    }
}

// Forensic Q&A Tool Functions
function openForensicQA() {
    const modal = document.getElementById('forensicQAModal');
    if (modal) {
        modal.style.display = 'block';
        initializeForensicQA();
    } else {
        showNotification('Opening Forensic Q&A system...', 'info');
        setTimeout(() => {
            window.location.href = '/scenes/?tool=forensic-qa';
        }, 1000);
    }
}

function closeForensicQAModal() {
    const modal = document.getElementById('forensicQAModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function initializeForensicQA() {
    // Initialize Q&A interface
    const queryInput = document.getElementById('forensicQuery');
    if (queryInput) {
        queryInput.focus();
    }
}

// Physics Prediction Tool Functions
function openPhysicsModal() {
    const modal = document.getElementById('physicsModal');
    if (modal) {
        modal.style.display = 'block';
        showPhysicsStep(1);
    } else {
        showNotification('Opening Physics Prediction tool...', 'info');
        setTimeout(() => {
            window.location.href = '/scenes/?tool=physics';
        }, 1000);
    }
}

function closePhysicsModal() {
    const modal = document.getElementById('physicsModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function showPhysicsStep(stepNumber) {
    document.querySelectorAll('.physics-step').forEach(step => {
        step.classList.remove('active');
        step.style.display = 'none';
    });
    
    const targetStep = document.getElementById(`physicsStep${stepNumber}`);
    if (targetStep) {
        targetStep.classList.add('active');
        targetStep.style.display = 'block';
    }
}

function initializePhysicsInputHandlers() {
    // Initialize physics prediction input handlers
    console.log('Physics input handlers initialized');
}

// Generic tool launcher for missing modals
function launchAnalysisTool(toolName) {
    showNotification(`Launching ${toolName}...`, 'info');
    
    // Map tool names to their respective functions
    const toolMap = {
        'scene-analysis': openSceneAnalysisTool,
        'forensic-qa': openForensicQA,
        'physics': openPhysicsModal,
        'inpainting': openInpaintingModal
    };
    
    const toolFunction = toolMap[toolName];
    if (toolFunction) {
        setTimeout(toolFunction, 500);
    } else {
        setTimeout(() => {
            window.location.href = `/scenes/?tool=${toolName}`;
        }, 1000);
    }
}

// Update keyboard shortcuts to include all modals
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + N to create new scene
    if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        createNewScene();
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        closeModal();
        closeInpaintingModal();
        closePhysicsModal();
        closeSceneAnalysisModal();
        closeForensicQAModal();
    }
});
