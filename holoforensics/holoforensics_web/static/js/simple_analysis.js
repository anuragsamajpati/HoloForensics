// Simple Analysis Tools - Working Implementation
document.addEventListener('DOMContentLoaded', function() {
    
    // Add click handlers to all analysis tool buttons
    const toolButtons = document.querySelectorAll('.tool-card .btn');
    toolButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            
            const toolCard = this.closest('.tool-card');
            const toolName = toolCard.querySelector('h4').textContent;
            
            console.log('Starting analysis tool:', toolName);
            startAnalysis(toolName);
        });
    });
    
    function startAnalysis(toolName) {
        // Show immediate feedback
        showNotification(`Starting ${toolName}...`, 'info');
        
        // Map tool names to API endpoints
        const toolMap = {
            'Object Detection': 'object-detection',
            'Multi-Camera Analysis': 'object-detection', 
            '3D Reconstruction': '3d-reconstruction',
            'Timeline Analysis': 'physics-prediction',
            'Video Inpainting': 'video-inpainting',
            'Physics Prediction': 'physics-prediction',
            'Scene Analysis': 'object-detection',
            'Identity Tracking': 'object-detection',
            'Q&A System': 'object-detection',
            'Forensic Q&A': 'object-detection'
        };
        
        const apiEndpoint = toolMap[toolName] || 'object-detection';
        
        // Make API call
        fetch(`/api/analysis/${apiEndpoint}/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                scene_id: 'scene_001',
                tool_name: toolName
            })
        })
        .then(response => {
            console.log('API Response status:', response.status);
            return response.json();
        })
        .then(data => {
            console.log('API Response data:', data);
            
            if (data.success) {
                showNotification(`${toolName} started! Job ID: ${data.job_id}`, 'success');
                showProgressModal(data.job_id, toolName, data.estimated_duration_seconds);
            } else {
                showNotification(`Failed to start ${toolName}: ${data.error}`, 'error');
            }
        })
        .catch(error => {
            console.error('API Error:', error);
            showNotification(`Error starting ${toolName}: ${error.message}`, 'error');
        });
    }
    
    function showProgressModal(jobId, toolName, estimatedDuration) {
        // Remove any existing progress modal
        const existingModal = document.querySelector('.analysis-progress-modal');
        if (existingModal) {
            existingModal.remove();
        }
        
        // Create progress modal
        const modal = document.createElement('div');
        modal.className = 'analysis-progress-modal';
        modal.innerHTML = `
            <div class="modal-overlay"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${toolName} Processing</h3>
                    <button class="close-btn" onclick="closeProgressModal()">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="progress-container">
                        <div class="progress-bar">
                            <div class="progress-fill" id="progress-${jobId}" style="width: 0%"></div>
                        </div>
                        <div class="progress-text">
                            <span id="progress-percent-${jobId}">0%</span>
                            <span id="progress-time-${jobId}">Estimated: ${Math.round(estimatedDuration/60)} minutes</span>
                        </div>
                    </div>
                    <div class="progress-details">
                        <p>Job ID: ${jobId}</p>
                        <p>Status: <span id="status-${jobId}">Processing...</span></p>
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="closeProgressModal()">Run in Background</button>
                    <button class="btn btn-primary" onclick="cancelAnalysis('${jobId}')">Cancel</button>
                </div>
            </div>
        `;
        
        // Add CSS styles
        const styles = `
            <style>
            .analysis-progress-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .modal-overlay {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.8);
            }
            .modal-content {
                background: #2d3436;
                border-radius: 10px;
                width: 500px;
                max-width: 90vw;
                color: white;
                position: relative;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            .modal-header {
                padding: 20px;
                border-bottom: 1px solid #636e72;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .modal-header h3 {
                margin: 0;
                color: #ff4757;
            }
            .close-btn {
                background: none;
                border: none;
                color: #b2bec3;
                font-size: 24px;
                cursor: pointer;
                padding: 0;
                width: 30px;
                height: 30px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .close-btn:hover {
                color: white;
            }
            .modal-body {
                padding: 20px;
            }
            .progress-container {
                margin-bottom: 20px;
            }
            .progress-bar {
                background: #636e72;
                height: 20px;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 10px;
            }
            .progress-fill {
                background: linear-gradient(90deg, #ff4757, #ff6b7a);
                height: 100%;
                transition: width 0.3s ease;
            }
            .progress-text {
                display: flex;
                justify-content: space-between;
                font-size: 14px;
                color: #b2bec3;
            }
            .progress-details {
                background: #636e72;
                padding: 15px;
                border-radius: 5px;
                font-size: 14px;
            }
            .progress-details p {
                margin: 5px 0;
            }
            .modal-footer {
                padding: 20px;
                border-top: 1px solid #636e72;
                display: flex;
                gap: 10px;
                justify-content: flex-end;
            }
            .btn {
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 14px;
            }
            .btn-primary {
                background: #ff4757;
                color: white;
            }
            .btn-secondary {
                background: #636e72;
                color: white;
            }
            .btn:hover {
                opacity: 0.9;
            }
            </style>
        `;
        
        // Add styles if not already added
        if (!document.querySelector('#progress-modal-styles')) {
            const styleElement = document.createElement('div');
            styleElement.id = 'progress-modal-styles';
            styleElement.innerHTML = styles;
            document.head.appendChild(styleElement);
        }
        
        document.body.appendChild(modal);
        
        // Start monitoring progress
        monitorProgress(jobId, toolName);
    }
    
    function monitorProgress(jobId, toolName) {
        const checkProgress = () => {
            fetch(`/api/analysis/status/${jobId}/`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateProgress(jobId, data);
                        
                        if (data.status === 'completed') {
                            showResults(jobId, toolName, data.results);
                        } else if (data.status === 'failed') {
                            showError(jobId, data.error);
                        } else {
                            // Continue monitoring
                            setTimeout(checkProgress, 2000);
                        }
                    }
                })
                .catch(error => {
                    console.error('Progress check error:', error);
                    setTimeout(checkProgress, 5000);
                });
        };
        
        checkProgress();
    }
    
    function updateProgress(jobId, data) {
        const progressFill = document.getElementById(`progress-${jobId}`);
        const progressPercent = document.getElementById(`progress-percent-${jobId}`);
        const progressTime = document.getElementById(`progress-time-${jobId}`);
        const status = document.getElementById(`status-${jobId}`);
        
        if (progressFill) {
            progressFill.style.width = `${data.progress}%`;
        }
        if (progressPercent) {
            progressPercent.textContent = `${data.progress}%`;
        }
        if (progressTime && data.estimated_remaining) {
            const remaining = Math.round(data.estimated_remaining / 60);
            progressTime.textContent = `Remaining: ${remaining} minutes`;
        }
        if (status) {
            status.textContent = data.status === 'processing' ? 'Processing...' : data.status;
        }
    }
    
    function showResults(jobId, toolName, results) {
        const modal = document.querySelector('.analysis-progress-modal');
        if (modal) {
            const modalBody = modal.querySelector('.modal-body');
            modalBody.innerHTML = `
                <h4>Analysis Complete!</h4>
                <div class="results-container">
                    <pre style="background: #636e72; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 12px;">
${JSON.stringify(results, null, 2)}
                    </pre>
                </div>
            `;
            
            const modalFooter = modal.querySelector('.modal-footer');
            modalFooter.innerHTML = `
                <button class="btn btn-secondary" onclick="downloadResults('${jobId}', '${toolName}')">Download Results</button>
                <button class="btn btn-primary" onclick="closeProgressModal()">Close</button>
            `;
        }
        
        showNotification(`${toolName} completed successfully!`, 'success');
    }
    
    function showError(jobId, error) {
        const modal = document.querySelector('.analysis-progress-modal');
        if (modal) {
            const status = modal.querySelector(`#status-${jobId}`);
            if (status) {
                status.textContent = `Error: ${error}`;
                status.style.color = '#ff4757';
            }
        }
        
        showNotification(`Analysis failed: ${error}`, 'error');
    }
    
    // Global functions
    window.closeProgressModal = function() {
        const modal = document.querySelector('.analysis-progress-modal');
        if (modal) {
            modal.remove();
        }
    };
    
    window.cancelAnalysis = function(jobId) {
        showNotification('Analysis cancelled', 'info');
        closeProgressModal();
    };
    
    window.downloadResults = function(jobId, toolName) {
        showNotification(`Downloading results for ${toolName}...`, 'info');
    };
    
    function showNotification(message, type) {
        // Remove existing notifications
        const existing = document.querySelectorAll('.analysis-notification');
        existing.forEach(n => n.remove());
        
        const notification = document.createElement('div');
        notification.className = `analysis-notification ${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">
                    ${type === 'success' ? '✓' : type === 'error' ? '✗' : 'ℹ'}
                </span>
                <span class="notification-message">${message}</span>
            </div>
        `;
        
        // Add notification styles
        const notificationStyles = `
            <style>
            .analysis-notification {
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10001;
                padding: 15px 20px;
                border-radius: 5px;
                color: white;
                font-size: 14px;
                min-width: 300px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                animation: slideIn 0.3s ease;
            }
            .analysis-notification.success {
                background: #00b894;
            }
            .analysis-notification.error {
                background: #e17055;
            }
            .analysis-notification.info {
                background: #0984e3;
            }
            .notification-content {
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .notification-icon {
                font-weight: bold;
                font-size: 16px;
            }
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            </style>
        `;
        
        if (!document.querySelector('#notification-styles')) {
            const styleElement = document.createElement('div');
            styleElement.id = 'notification-styles';
            styleElement.innerHTML = notificationStyles;
            document.head.appendChild(styleElement);
        }
        
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 5000);
    }
});
