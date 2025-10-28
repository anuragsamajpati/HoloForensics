// Dashboard JavaScript Functionality
document.addEventListener('DOMContentLoaded', function() {
    // File upload functionality
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const startAnalysisBtn = document.querySelector('.start-analysis-btn');
    let uploadedFiles = [];

    // Upload area click handler
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    // Drag and drop handlers
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });

    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        const files = Array.from(e.dataTransfer.files);
        handleFiles(files);
    });

    // File input change handler
    fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleFiles(files);
    });

    // Handle uploaded files
    function handleFiles(files) {
        uploadedFiles = [...uploadedFiles, ...files];
        updateUploadDisplay();
        updateStartButton();
        updateFilesUploadedCounter();
    }

    // Update upload display
    function updateUploadDisplay() {
        if (uploadedFiles.length > 0) {
            const uploadContent = uploadArea.querySelector('.upload-content');
            uploadContent.innerHTML = `
                <i class="fas fa-check-circle upload-icon" style="color: #27ae60;"></i>
                <h4>${uploadedFiles.length} file(s) selected</h4>
                <p>Click to add more files</p>
                <div class="file-list">
                    ${uploadedFiles.map(file => `
                        <div class="file-item">
                            <i class="fas fa-file"></i>
                            <span>${file.name}</span>
                            <span class="file-size">(${formatFileSize(file.size)})</span>
                        </div>
                    `).join('')}
                </div>
            `;
        }
    }

    // Update start analysis button
    function updateStartButton() {
        if (uploadedFiles.length > 0) {
            startAnalysisBtn.disabled = false;
            startAnalysisBtn.innerHTML = `
                <i class="fas fa-play"></i> 
                Start Analysis (${uploadedFiles.length} files)
            `;
        } else {
            startAnalysisBtn.disabled = true;
            startAnalysisBtn.innerHTML = `
                <i class="fas fa-play"></i> 
                Start Analysis
            `;
        }
    }

    // Format file size
    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    // Pretty label for analysis types
    function getAnalysisLabel(value) {
        const map = {
            'full': 'Full Analysis',
            'object_detection': 'Object Detection',
            '3d_reconstruction': '3D Reconstruction',
            'timeline': 'Timeline Analysis',
            'video_inpainting': 'Video Inpainting',
            'physics_prediction': 'Physics Prediction',
            'scene_analysis': 'Scene Graph & Event Detection',
            'forensic_qa': 'Forensic Q&A System'
        };
        return map[value] || value;
    }

    // Start analysis handler
    startAnalysisBtn.addEventListener('click', async function() {
        if (uploadedFiles.length === 0) return;

        const analysisType = document.getElementById('analysisType').value;
        const analysisLabel = getAnalysisLabel(analysisType);
        const priority = document.getElementById('priority').value;

        // Show loading state
        this.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Uploading & Starting...';
        this.disabled = true;

        try {
            // 1) Upload files to backend
            const { sceneId, etaSeconds } = await uploadSceneFiles(analysisType, priority);
            // persist ETA for later display in results
            window.lastEtaSecondsByScene = window.lastEtaSecondsByScene || {};
            window.lastEtaSecondsByScene[sceneId] = etaSeconds;
            showAnalysisStarted(analysisLabel, priority);

            // Create a progress modal with ETA and start a client-side timer
            const progressModal = createProgressModal(sceneId, analysisLabel, etaSeconds);
            document.body.appendChild(progressModal);
            let startTs = Date.now();
            const etaMs = Math.max(1, (etaSeconds || 60)) * 1000;
            let progressTimer = setInterval(() => {
                const elapsed = Date.now() - startTs;
                let pct = Math.min(100, Math.round((elapsed / etaMs) * 100));
                const remaining = Math.max(0, Math.round((etaMs - elapsed) / 1000));
                updateProgressModal(sceneId, { progress: pct, estimated_remaining: remaining });
                if (pct >= 100) {
                    clearInterval(progressTimer);
                    progressTimer = null;
                }
            }, 1000);

            // 2) Poll processing status until complete
            const statusData = await pollSceneStatus(sceneId);

            // 3) Fetch scene results and show
            const results = await fetchSceneResults(sceneId);
            removeProgressModal(sceneId);
            showAnalysisResults(sceneId, analysisLabel, results);

        } catch (err) {
            console.error('Upload/processing error:', err);
            showNotification(`Failed to process upload: ${err.message || err}`, 'error');
        } finally {
            // Reset UI
            resetUploadForm();
            this.innerHTML = '<i class="fas fa-play"></i> Start Analysis';
            this.disabled = false;
        }
    });

    // Upload selected files to /api/upload-scene/
    async function uploadSceneFiles(analysisType, priority) {
        const fd = new FormData();
        fd.append('scene_name', 'Uploaded Scene');
        fd.append('description', `${analysisType} - ${priority}`);
        fd.append('analysis_type', analysisType);
        uploadedFiles.forEach((file) => fd.append('files', file, file.name));

        const response = await fetch('/api/upload-scene/', {
            method: 'POST',
            body: fd
        });

        if (!response.ok) {
            const text = await response.text();
            throw new Error(`Upload failed (${response.status}): ${text}`);
        }
        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || 'Upload failed');
        }
        showNotification(`Uploaded ${data.uploaded_count} file(s). Scene ID: ${data.scene_id}`, 'success');
        return { sceneId: data.scene_id, etaSeconds: data.eta_seconds };
    }

    // Poll /api/status/<scene_id>/ until ready/completed
    async function pollSceneStatus(sceneId) {
        return new Promise((resolve, reject) => {
            const check = async () => {
                try {
                    const resp = await fetch(`/api/status/${sceneId}/`);
                    if (!resp.ok) {
                        setTimeout(check, 2000);
                        return;
                    }
                    const data = await resp.json();
                    const status = (data.status || '').toLowerCase();
                    if (status === 'completed' || status === 'ready') {
                        resolve(data);
                        return;
                    }
                    // keep polling
                    setTimeout(check, 2000);
                } catch (e) {
                    console.warn('Status poll error:', e);
                    setTimeout(check, 3000);
                }
            };
            check();
        });
    }

    // Fetch results from /api/scene/<scene_id>/results/
    async function fetchSceneResults(sceneId) {
        const resp = await fetch(`/api/scene/${sceneId}/results/`);
        if (!resp.ok) {
            throw new Error(`Failed to load results (${resp.status})`);
        }
        return await resp.json();
    }

    // Show analysis started notification
    function showAnalysisStarted(type, priority) {
        const notification = document.createElement('div');
        notification.className = 'notification success';
        notification.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <div>
                <strong>Analysis Started!</strong>
                <p>Your ${type} analysis has been queued with ${priority} priority.</p>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);

        // Update stats
        updateStats();
    }

    // Reset upload form
    function resetUploadForm() {
        uploadedFiles = [];
        fileInput.value = '';
        
        const uploadContent = uploadArea.querySelector('.upload-content');
        uploadContent.innerHTML = `
            <i class="fas fa-file-upload upload-icon"></i>
            <h4>Drag & Drop Files Here</h4>
            <p>or click to browse</p>
            <div class="supported-formats">
                <span>Supported: MP4, AVI, MOV, JPG, PNG, TIFF</span>
            </div>
        `;
        
        updateStartButton();
        updateFilesUploadedCounter();
    }

    // Update files uploaded counter
    function updateFilesUploadedCounter() {
        const filesUploadedCount = document.getElementById('filesUploadedCount');
        if (filesUploadedCount) {
            filesUploadedCount.textContent = uploadedFiles.length;
        }
    }

    // Update dashboard stats (safe, uses IDs present in template)
    function updateStats() {
        const filesUploadedEl = document.getElementById('filesUploadedCount');
        const processingEl = document.getElementById('processingCount');

        const increment = uploadedFiles.length || 0;
        if (filesUploadedEl) {
            const current = parseInt(filesUploadedEl.textContent) || 0;
            filesUploadedEl.textContent = String(current + increment);
        }
        if (processingEl) {
            // Show how many files have entered processing
            const current = parseInt(processingEl.textContent) || 0;
            processingEl.textContent = String(current + increment);
        }
    }

    // Analysis tool handlers
    document.querySelectorAll('.tool-card .btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.preventDefault();
            const toolName = this.closest('.tool-card').querySelector('h4').textContent;
            const toolType = getToolType(toolName);
            startAnalysisTool(toolType, toolName);
        });
    });

    // Map tool names to API endpoints
    function getToolType(toolName) {
        const toolMap = {
            'Object Detection': 'object-detection',
            'Multi-Camera Analysis': 'object-detection',
            '3D Reconstruction': '3d-reconstruction',
            'Timeline Analysis': 'physics-prediction',
            'Video Inpainting': 'video-inpainting',
            'Physics Prediction': 'physics-prediction',
            'Scene Analysis': 'object-detection',
            'Identity Tracking': 'object-detection'
        };
        return toolMap[toolName] || 'object-detection';
    }

    // Start analysis tool
    async function startAnalysisTool(toolType, toolName) {
        try {
            showNotification(`Starting ${toolName}...`, 'info');
            
            // Get CSRF token
            const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || 
                             document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
            
            console.log('Making API call to:', `/api/analysis/${toolType}/`);
            
            const response = await fetch(`/api/analysis/${toolType}/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken
                },
                body: JSON.stringify({
                    scene_id: 'scene_001',
                    tool_name: toolName
                })
            });

            console.log('Response status:', response.status);
            const data = await response.json();
            console.log('Response data:', data);
            
            if (data.success) {
                showNotification(`${toolName} started successfully! Job ID: ${data.job_id}`, 'success');
                monitorAnalysisJob(data.job_id, toolName, data.estimated_duration_seconds);
            } else {
                showNotification(`Failed to start ${toolName}: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error starting analysis:', error);
            showNotification(`Error starting ${toolName}: ${error.message}`, 'error');
        }
    }

    // Monitor analysis job progress
    async function monitorAnalysisJob(jobId, toolName, estimatedDuration) {
        const progressModal = createProgressModal(jobId, toolName, estimatedDuration);
        document.body.appendChild(progressModal);
        
        const checkProgress = async () => {
            try {
                const response = await fetch(`/api/analysis/status/${jobId}/`);
                const data = await response.json();
                
                if (data.success) {
                    updateProgressModal(jobId, data);
                    
                    if (data.status === 'completed') {
                        showAnalysisResults(jobId, toolName, data.results);
                        removeProgressModal(jobId);
                        return;
                    } else if (data.status === 'failed') {
                        showNotification(`${toolName} failed: ${data.error}`, 'error');
                        removeProgressModal(jobId);
                        return;
                    }
                }
                
                // Continue monitoring
                setTimeout(checkProgress, 2000);
                
            } catch (error) {
                console.error('Error checking progress:', error);
                setTimeout(checkProgress, 5000);
            }
        };
        
        checkProgress();
    }

    // Create progress modal
    function createProgressModal(jobId, toolName, estimatedDuration) {
        const modal = document.createElement('div');
        modal.id = `progress-modal-${jobId}`;
        modal.className = 'progress-modal';
        modal.innerHTML = `
            <div class="progress-modal-content">
                <h3>${toolName} Processing</h3>
                <div class="progress-bar-container">
                    <div class="progress-bar" id="progress-bar-${jobId}">
                        <div class="progress-fill" style="width: 0%"></div>
                    </div>
                    <span class="progress-text" id="progress-text-${jobId}">0%</span>
                </div>
                <p class="progress-info" id="progress-info-${jobId}">
                    Estimated time: ${Math.round(estimatedDuration / 60)} minutes
                </p>
                <button onclick="removeProgressModal('${jobId}')" class="btn btn-secondary">
                    Run in Background
                </button>
            </div>
        `;
        
        // Add CSS for progress modal
        if (!document.getElementById('progress-modal-styles')) {
            const styles = document.createElement('style');
            styles.id = 'progress-modal-styles';
            styles.textContent = `
                .progress-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0,0,0,0.8);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 10000;
                }
                .progress-modal-content {
                    background: #2d3436;
                    padding: 30px;
                    border-radius: 10px;
                    color: white;
                    min-width: 400px;
                    text-align: center;
                }
                .progress-bar-container {
                    margin: 20px 0;
                }
                .progress-bar {
                    background: #636e72;
                    height: 20px;
                    border-radius: 10px;
                    overflow: hidden;
                    position: relative;
                }
                .progress-fill {
                    background: linear-gradient(90deg, #ff4757, #ff6b7a);
                    height: 100%;
                    transition: width 0.3s ease;
                }
                .progress-text {
                    display: block;
                    margin-top: 10px;
                    font-weight: bold;
                }
                .progress-info {
                    color: #b2bec3;
                    margin: 15px 0;
                }
            `;
            document.head.appendChild(styles);
        }
        
        return modal;
    }

    // Update progress modal
    function updateProgressModal(jobId, data) {
        const progressFill = document.querySelector(`#progress-modal-${jobId} .progress-fill`);
        const progressText = document.getElementById(`progress-text-${jobId}`);
        const progressInfo = document.getElementById(`progress-info-${jobId}`);
        
        if (progressFill && progressText) {
            progressFill.style.width = `${data.progress}%`;
            progressText.textContent = `${data.progress}%`;
            
            if (progressInfo && data.estimated_remaining) {
                const mins = Math.floor(data.estimated_remaining / 60);
                const secs = data.estimated_remaining % 60;
                const pretty = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
                progressInfo.textContent = `Estimated remaining: ${pretty}`;
            }
        }
    }

    // Remove progress modal
    window.removeProgressModal = function(jobId) {
        const modal = document.getElementById(`progress-modal-${jobId}`);
        if (modal) {
            modal.remove();
        }
    };

    // Show analysis results
    function showAnalysisResults(jobId, toolName, results) {
        showNotification(`${toolName} completed successfully!`, 'success');

        // Build media previews if available
        let mediaHtml = '';
        const media = Array.isArray(results.media) ? results.media : [];
        if (media.length > 0) {
            mediaHtml = `
                <h4 style="margin: 10px 0 6px 0;">Media Previews</h4>
                <div style="display:flex; flex-direction: column; gap: 12px;">
                    ${media.map(m => `
                        <div style="background:#2d3436; padding:10px; border-radius:8px;">
                            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:8px;">
                                <span style="color:#fff; font-size:0.9rem;">${m.folder} / ${m.name}</span>
                                <a href="${m.download_url}${m.download_url.includes('?') ? '&' : '?'}download=1" style="background:#ff4757;color:#fff;padding:6px 10px;border-radius:6px;text-decoration:none;font-size:12px;">Download</a>
                            </div>
                            ${m.name.toLowerCase().match(/\.(mp4|mov|avi|mkv)$/) ? `
                                <video controls preload="metadata" style="width:100%; max-height:320px; border-radius:6px; background:#000;" src="${m.download_url}"></video>
                            ` : ''}
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Build artifacts list (JSON, PLY, TXT) if available
        let artifactsHtml = '';
        const artifacts = Array.isArray(results.artifacts) ? results.artifacts : [];
        if (artifacts.length > 0) {
            artifactsHtml = `
                <h4 style="margin: 18px 0 6px 0;">Generated Files</h4>
                <div style="display:flex; flex-direction: column; gap: 6px;">
                    ${artifacts.map(a => `
                        <div style="display:flex; align-items:center; justify-content:space-between; background:#2d3436; padding:8px 10px; border-radius:6px;">
                            <span style="color:#fff; font-size:0.9rem;">${a.folder} / ${a.name}</span>
                            <a href="${a.download_url}${a.download_url.includes('?') ? '&' : '?'}download=1" style="background:#2d3436;color:#fff;border:1px solid #444;padding:6px 10px;border-radius:6px;text-decoration:none;font-size:12px;">Download</a>
                        </div>
                    `).join('')}
                </div>
            `;
        }

        // Build ETA label
        let etaLabel = '';
        if (window.lastEtaSecondsByScene && window.lastEtaSecondsByScene[jobId]) {
            const secs = Math.max(0, parseInt(window.lastEtaSecondsByScene[jobId]) || 0);
            const mins = Math.max(1, Math.round(secs / 60));
            etaLabel = `<p style="color:#b2bec3; margin: 6px 0 14px 0;">Estimated total time: ~ ${mins} minute${mins>1?'s':''}</p>`;
        }

        // Close any existing modals to avoid confusion between runs
        document.querySelectorAll('.progress-modal').forEach(m => m.remove());

        // Create results modal (video-only UI)
        const resultsModal = document.createElement('div');
        resultsModal.className = 'progress-modal';
        resultsModal.id = `results-modal-${jobId}`;
        resultsModal.innerHTML = `
            <div class="progress-modal-content" style="max-width: 820px; width: 90vw; max-height: 88vh; overflow-y: auto;">
                <h3>${toolName} - Results</h3>
                ${etaLabel}
                <div class="results-content">
                    ${mediaHtml || '<div style="color:#b2bec3">No video previews available.</div>'}
                    ${artifactsHtml}
                </div>
                <div style="margin-top: 20px; display:flex; gap:10px; justify-content:flex-end; flex-wrap: wrap;">
                    <a href="/api/scene/${jobId}/report/" target="_blank" class="btn btn-secondary" style="text-decoration:none;">Open Report</a>
                    <a href="/api/scene/${jobId}/report/?download=1" class="btn btn-secondary" style="text-decoration:none;">Download Report</a>
                    <button onclick="downloadResults('${jobId}', '${toolName}')" class="btn btn-secondary">Download Results</button>
                    <button onclick="this.closest('.progress-modal').remove()" class="btn btn-primary">Close</button>
                </div>
            </div>
        `;

        document.body.appendChild(resultsModal);
    }

    // Download results
    window.downloadResults = function(jobId, toolName) {
        // Trigger server-side ZIP download
        window.location.href = `/api/scene/${jobId}/download/`;
    };

    // Show notification function
    window.showNotification = function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        
        const iconMap = {
            'success': 'fas fa-check-circle',
            'error': 'fas fa-exclamation-circle',
            'warning': 'fas fa-exclamation-triangle',
            'info': 'fas fa-info-circle'
        };
        
        notification.innerHTML = `
            <i class="${iconMap[type] || iconMap.info}"></i>
            <div>
                <strong>${message}</strong>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 4000);
    };

    // Show tool launch notification
    function showToolLaunch(toolName) {
        showNotification(`Launching ${toolName}...`, 'info');
    }

    // Animate status bars
    function animateStatusBars() {
        const statusFills = document.querySelectorAll('.status-fill');
        statusFills.forEach(fill => {
            const width = fill.style.width;
            fill.style.width = '0%';
            setTimeout(() => {
                fill.style.width = width;
            }, 500);
        });
    }

    // Real-time updates simulation
    function simulateRealTimeUpdates() {
        setInterval(() => {
            // Randomly update GPU processing
            const gpuBar = document.querySelector('.status-fill');
            const currentWidth = parseInt(gpuBar.style.width);
            const newWidth = Math.max(20, Math.min(80, currentWidth + (Math.random() - 0.5) * 10));
            gpuBar.style.width = newWidth + '%';
            
            // Update corresponding value
            const gpuValue = gpuBar.parentElement.parentElement.querySelector('.status-value');
            gpuValue.textContent = Math.round(newWidth) + '%';
        }, 5000);
    }

    // Initialize animations
    setTimeout(animateStatusBars, 1000);
    simulateRealTimeUpdates();

    // Recent item click handlers
    const recentItems = document.querySelectorAll('.recent-item');
    recentItems.forEach(item => {
        item.addEventListener('click', function() {
            const title = this.querySelector('h4').textContent;
            showAnalysisDetails(title);
        });
    });

    // Show analysis details
    function showAnalysisDetails(title) {
        const notification = document.createElement('div');
        notification.className = 'notification info';
        notification.innerHTML = `
            <i class="fas fa-eye"></i>
            <div>
                <strong>Opening Analysis</strong>
                <p>Loading details for "${title}"...</p>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
});

// Add notification styles
const notificationStyles = `
    .notification {
        position: fixed;
        top: 100px;
        right: 20px;
        background: #1a1a1a;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        z-index: 1000;
        min-width: 300px;
        animation: slideIn 0.3s ease;
    }
    
    .notification.success {
        border-color: #27ae60;
    }
    
    .notification.info {
        border-color: #3498db;
    }
    
    .notification i {
        font-size: 1.2rem;
    }
    
    .notification.success i {
        color: #27ae60;
    }
    
    .notification.info i {
        color: #3498db;
    }
    
    .notification strong {
        color: #ffffff;
        display: block;
        margin-bottom: 0.25rem;
    }
    
    .notification p {
        color: #b0b0b0;
        font-size: 0.9rem;
        margin: 0;
    }
    
    .file-list {
        margin-top: 1rem;
        text-align: left;
    }
    
    .file-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 5px;
        margin-bottom: 0.5rem;
        font-size: 0.8rem;
    }
    
    .file-item i {
        color: #ff4757;
    }
    
    .file-size {
        color: #666;
        margin-left: auto;
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;

// Inject notification styles
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);
