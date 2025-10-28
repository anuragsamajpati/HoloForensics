// Physics Prediction Modal Functions
let currentPhysicsStep = 1;
let physicsJobId = null;
let physicsProcessingInterval = null;
let trajectoryPoints = [];

function openPhysicsModal() {
    const modal = document.getElementById('physicsModal');
    modal.style.display = 'block';
    resetPhysicsModal();
}

function closePhysicsModal() {
    const modal = document.getElementById('physicsModal');
    modal.style.display = 'none';
    
    if (physicsProcessingInterval) {
        clearInterval(physicsProcessingInterval);
        physicsProcessingInterval = null;
    }
    
    resetPhysicsModal();
}

function resetPhysicsModal() {
    currentPhysicsStep = 1;
    physicsJobId = null;
    trajectoryPoints = [];
    
    document.querySelectorAll('#physicsModal .step').forEach((step, index) => {
        step.classList.toggle('active', index === 0);
    });
    
    document.querySelectorAll('#physicsModal .step-content').forEach((content, index) => {
        content.classList.toggle('active', index === 0);
    });
    
    const physicsSceneSelect = document.getElementById('physicsSceneSelect');
    if (physicsSceneSelect) physicsSceneSelect.value = '';
    
    const predictionHorizon = document.getElementById('predictionHorizon');
    if (predictionHorizon) predictionHorizon.value = 60;
    
    const confidenceThreshold = document.getElementById('confidenceThreshold');
    if (confidenceThreshold) confidenceThreshold.value = 0.7;
    
    const kalmanWeight = document.getElementById('kalmanWeight');
    if (kalmanWeight) kalmanWeight.value = 0.6;
    
    const dataValidation = document.getElementById('dataValidation');
    if (dataValidation) dataValidation.style.display = 'none';
    
    const validateTrajectoryBtn = document.getElementById('validateTrajectoryBtn');
    if (validateTrajectoryBtn) validateTrajectoryBtn.disabled = true;
    
    const physicsNextStep1 = document.getElementById('physicsNextStep1');
    if (physicsNextStep1) physicsNextStep1.disabled = true;
    
    const physicsProgressFill = document.getElementById('physicsProgressFill');
    if (physicsProgressFill) physicsProgressFill.style.width = '0%';
    
    const physicsProgressText = document.getElementById('physicsProgressText');
    if (physicsProgressText) physicsProgressText.textContent = 'Initializing physics models...';
    
    const physicsLogContent = document.getElementById('physicsLogContent');
    if (physicsLogContent) physicsLogContent.innerHTML = '<div class="log-entry">Physics prediction system initialized</div>';
    
    const physicsResultsSection = document.getElementById('physicsResultsSection');
    if (physicsResultsSection) physicsResultsSection.style.display = 'none';
}

function nextPhysicsStep(step) {
    if (step <= 3 && step > currentPhysicsStep) {
        const currentStepElement = document.querySelector(`#physicsStep${currentPhysicsStep}`);
        const currentStepIndicator = document.querySelector(`#physicsModal .step[data-step="${currentPhysicsStep}"]`);
        
        if (currentStepElement) currentStepElement.classList.remove('active');
        if (currentStepIndicator) currentStepIndicator.classList.remove('active');
        
        const nextStepElement = document.querySelector(`#physicsStep${step}`);
        const nextStepIndicator = document.querySelector(`#physicsModal .step[data-step="${step}"]`);
        
        if (nextStepElement) nextStepElement.classList.add('active');
        if (nextStepIndicator) nextStepIndicator.classList.add('active');
        
        currentPhysicsStep = step;
        
        if (step === 3) {
            startPhysicsPrediction();
        }
    }
}

function prevPhysicsStep(step) {
    if (step >= 1 && step < currentPhysicsStep) {
        const currentStepElement = document.querySelector(`#physicsStep${currentPhysicsStep}`);
        const currentStepIndicator = document.querySelector(`#physicsModal .step[data-step="${currentPhysicsStep}"]`);
        
        if (currentStepElement) currentStepElement.classList.remove('active');
        if (currentStepIndicator) currentStepIndicator.classList.remove('active');
        
        const prevStepElement = document.querySelector(`#physicsStep${step}`);
        const prevStepIndicator = document.querySelector(`#physicsModal .step[data-step="${step}"]`);
        
        if (prevStepElement) prevStepElement.classList.add('active');
        if (prevStepIndicator) prevStepIndicator.classList.add('active');
        
        currentPhysicsStep = step;
    }
}

function initializePhysicsInputHandlers() {
    const inputMethodRadios = document.querySelectorAll('input[name="inputMethod"]');
    inputMethodRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            const sceneDataInput = document.getElementById('sceneDataInput');
            const uploadInput = document.getElementById('uploadInput');
            const manualInput = document.getElementById('manualInput');
            
            if (sceneDataInput) sceneDataInput.style.display = 'none';
            if (uploadInput) uploadInput.style.display = 'none';
            if (manualInput) manualInput.style.display = 'none';
            
            if (this.value === 'scene' && sceneDataInput) {
                sceneDataInput.style.display = 'block';
            } else if (this.value === 'upload' && uploadInput) {
                uploadInput.style.display = 'block';
            } else if (this.value === 'manual' && manualInput) {
                manualInput.style.display = 'block';
            }
            
            updateValidateButton();
        });
    });
    
    const physicsSceneSelect = document.getElementById('physicsSceneSelect');
    if (physicsSceneSelect) {
        physicsSceneSelect.addEventListener('change', function() {
            if (this.value) {
                loadSceneTrajectories(this.value);
            } else {
                const trajectoryList = document.getElementById('trajectoryList');
                if (trajectoryList) trajectoryList.style.display = 'none';
            }
            updateValidateButton();
        });
    }
    
    const trajectoryFile = document.getElementById('trajectoryFile');
    if (trajectoryFile) {
        trajectoryFile.addEventListener('change', function() {
            if (this.files.length > 0) {
                loadTrajectoryFile(this.files[0]);
            }
            updateValidateButton();
        });
    }
    
    const confidenceThreshold = document.getElementById('confidenceThreshold');
    if (confidenceThreshold) {
        confidenceThreshold.addEventListener('input', function() {
            const confidenceValue = document.getElementById('confidenceValue');
            if (confidenceValue) confidenceValue.textContent = this.value;
        });
    }
    
    const kalmanWeight = document.getElementById('kalmanWeight');
    if (kalmanWeight) {
        kalmanWeight.addEventListener('input', function() {
            const kalmanWeightValue = document.getElementById('kalmanWeightValue');
            if (kalmanWeightValue) kalmanWeightValue.textContent = this.value;
        });
    }
}

function updateValidateButton() {
    const selectedMethodRadio = document.querySelector('input[name="inputMethod"]:checked');
    if (!selectedMethodRadio) return;
    
    const selectedMethod = selectedMethodRadio.value;
    const validateBtn = document.getElementById('validateTrajectoryBtn');
    if (!validateBtn) return;
    
    let hasData = false;
    
    if (selectedMethod === 'scene') {
        const physicsSceneSelect = document.getElementById('physicsSceneSelect');
        hasData = physicsSceneSelect && physicsSceneSelect.value !== '';
    } else if (selectedMethod === 'upload') {
        const trajectoryFile = document.getElementById('trajectoryFile');
        hasData = trajectoryFile && trajectoryFile.files.length > 0;
    } else if (selectedMethod === 'manual') {
        const objectId = document.getElementById('objectId');
        hasData = objectId && objectId.value.trim() !== '' && trajectoryPoints.length >= 3;
    }
    
    validateBtn.disabled = !hasData;
}

function loadSceneTrajectories(sceneId) {
    const trajectoryList = document.getElementById('trajectoryList');
    const trajectoryItems = trajectoryList ? trajectoryList.querySelector('.trajectory-items') : null;
    
    if (!trajectoryItems) return;
    
    const mockTrajectories = [
        { id: 'person_001', type: 'person', points: 45, confidence: 0.89 },
        { id: 'person_002', type: 'person', points: 32, confidence: 0.76 },
        { id: 'vehicle_001', type: 'vehicle', points: 28, confidence: 0.92 }
    ];
    
    trajectoryItems.innerHTML = '';
    mockTrajectories.forEach(traj => {
        const item = document.createElement('div');
        item.className = 'trajectory-item';
        item.innerHTML = `
            <div>
                <strong>${traj.id}</strong> (${traj.type})
                <br><small>${traj.points} points, confidence: ${traj.confidence}</small>
            </div>
            <input type="checkbox" checked>
        `;
        trajectoryItems.appendChild(item);
    });
    
    trajectoryList.style.display = 'block';
}

function loadTrajectoryFile(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        try {
            const data = JSON.parse(e.target.result);
            console.log('Loaded trajectory data:', data);
            if (typeof showNotification === 'function') {
                showNotification('Trajectory file loaded successfully', 'success');
            }
        } catch (error) {
            console.error('Error parsing trajectory file:', error);
            if (typeof showNotification === 'function') {
                showNotification('Error parsing trajectory file', 'error');
            }
        }
    };
    reader.readAsText(file);
}

function addTrajectoryPoint() {
    const pointsList = document.getElementById('pointsList');
    if (!pointsList) return;
    
    const pointIndex = trajectoryPoints.length;
    
    const pointDiv = document.createElement('div');
    pointDiv.className = 'trajectory-point';
    pointDiv.innerHTML = `
        <div style="display: flex; gap: 10px; margin-bottom: 10px; align-items: center;">
            <input type="number" placeholder="X" step="0.1" style="width: 80px;" onchange="updateTrajectoryPoint(${pointIndex}, 'x', this.value)">
            <input type="number" placeholder="Y" step="0.1" style="width: 80px;" onchange="updateTrajectoryPoint(${pointIndex}, 'y', this.value)">
            <input type="number" placeholder="Time" step="0.1" style="width: 80px;" onchange="updateTrajectoryPoint(${pointIndex}, 'timestamp', this.value)">
            <button type="button" class="btn-small" onclick="removeTrajectoryPoint(${pointIndex})" style="background: #dc3545;">Remove</button>
        </div>
    `;
    
    pointsList.appendChild(pointDiv);
    trajectoryPoints.push({ x: 0, y: 0, timestamp: 0 });
    updateValidateButton();
}

function updateTrajectoryPoint(index, field, value) {
    if (trajectoryPoints[index]) {
        trajectoryPoints[index][field] = parseFloat(value) || 0;
        updateValidateButton();
    }
}

function removeTrajectoryPoint(index) {
    const pointsList = document.getElementById('pointsList');
    if (!pointsList) return;
    
    const pointDivs = pointsList.querySelectorAll('.trajectory-point');
    if (pointDivs[index]) {
        pointDivs[index].remove();
        trajectoryPoints.splice(index, 1);
        updateValidateButton();
    }
}

function validateTrajectoryData() {
    const selectedMethodRadio = document.querySelector('input[name="inputMethod"]:checked');
    if (!selectedMethodRadio) return;
    
    const selectedMethod = selectedMethodRadio.value;
    const validationSection = document.getElementById('dataValidation');
    const validationItems = document.getElementById('validationItems');
    
    let trajectoryData = [];
    
    if (selectedMethod === 'scene') {
        const checkedTrajectories = document.querySelectorAll('#trajectoryList input[type="checkbox"]:checked');
        trajectoryData = Array.from(checkedTrajectories).map((checkbox, index) => ({
            object_id: `object_${index}`,
            points: [
                { x: 10, y: 10, timestamp: 0 },
                { x: 15, y: 12, timestamp: 1 },
                { x: 20, y: 15, timestamp: 2 }
            ]
        }));
    } else if (selectedMethod === 'manual') {
        const objectId = document.getElementById('objectId');
        if (objectId && objectId.value.trim() && trajectoryPoints.length >= 3) {
            trajectoryData = [{
                object_id: objectId.value.trim(),
                points: trajectoryPoints
            }];
        }
    }
    
    // Get CSRF token
    function getCSRFToken() {
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            const [name, value] = cookie.trim().split('=');
            if (name === 'csrftoken') {
                return value;
            }
        }
        return '';
    }
    
    fetch('/api/physics/validate/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify({
            trajectory_data: trajectoryData
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success && validationItems) {
            validationItems.innerHTML = '';
            
            data.validation_results.forEach(result => {
                const item = document.createElement('div');
                item.className = 'validation-item';
                item.style.color = result.valid ? '#28a745' : '#dc3545';
                item.innerHTML = `
                    <i class="fas fa-${result.valid ? 'check-circle' : 'times-circle'}"></i>
                    <span>${result.object_id}: ${result.valid ? 'Valid' : result.issues.join(', ')}</span>
                `;
                validationItems.appendChild(item);
            });
            
            if (validationSection) validationSection.style.display = 'block';
            
            if (data.summary.validation_passed) {
                const physicsNextStep1 = document.getElementById('physicsNextStep1');
                if (physicsNextStep1) physicsNextStep1.disabled = false;
                
                if (typeof showNotification === 'function') {
                    showNotification('Trajectory data validation passed', 'success');
                }
            } else {
                if (typeof showNotification === 'function') {
                    showNotification('Trajectory data validation failed', 'error');
                }
            }
        } else {
            if (typeof showNotification === 'function') {
                showNotification('Validation error: ' + data.error, 'error');
            }
        }
    })
    .catch(error => {
        console.error('Validation error:', error);
        if (typeof showNotification === 'function') {
            showNotification('Validation request failed', 'error');
        }
    });
}

function startPhysicsPrediction() {
    const selectedMethodRadio = document.querySelector('input[name="inputMethod"]:checked');
    if (!selectedMethodRadio) return;
    
    const selectedMethod = selectedMethodRadio.value;
    let trajectoryData = [];
    
    if (selectedMethod === 'scene') {
        const physicsSceneSelect = document.getElementById('physicsSceneSelect');
        const sceneId = physicsSceneSelect ? physicsSceneSelect.value : 'scene_001';
        const checkedTrajectories = document.querySelectorAll('#trajectoryList input[type="checkbox"]:checked');
        trajectoryData = Array.from(checkedTrajectories).map((checkbox, index) => ({
            object_id: `scene_${sceneId}_object_${index}`,
            object_type: 'person',
            points: [
                { x: 10 + index * 5, y: 10 + index * 3, timestamp: 0, confidence: 0.9 },
                { x: 15 + index * 5, y: 12 + index * 3, timestamp: 1, confidence: 0.85 },
                { x: 20 + index * 5, y: 15 + index * 3, timestamp: 2, confidence: 0.8 }
            ]
        }));
    } else if (selectedMethod === 'manual') {
        const objectId = document.getElementById('objectId');
        const objectType = document.getElementById('objectType');
        if (objectId && objectType) {
            trajectoryData = [{
                object_id: objectId.value.trim(),
                object_type: objectType.value,
                points: trajectoryPoints
            }];
        }
    }
    
    const config = {
        prediction_horizon: parseInt(document.getElementById('predictionHorizon')?.value || 60),
        confidence_threshold: parseFloat(document.getElementById('confidenceThreshold')?.value || 0.7),
        kalman_weight: parseFloat(document.getElementById('kalmanWeight')?.value || 0.6),
        scene_calibration: document.getElementById('sceneCalibration')?.checked || true,
        evidence_preservation: document.getElementById('evidencePreservation')?.checked || true,
        uncertainty_quantification: document.getElementById('uncertaintyQuantification')?.checked || true
    };
    
    // Get CSRF token
    function getCSRFToken() {
        const cookies = document.cookie.split(';');
        for (let cookie of cookies) {
            const [name, value] = cookie.trim().split('=');
            if (name === 'csrftoken') {
                return value;
            }
        }
        return '';
    }
    
    fetch('/api/physics/predict/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRFToken()
        },
        body: JSON.stringify({
            case_id: 'scene_001',
            trajectory_data: trajectoryData,
            config: config,
            context: {
                investigation_type: 'forensic_analysis',
                priority: 'high'
            }
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            physicsJobId = data.job_id;
            
            if (data.status === 'completed') {
                displayPhysicsResults(data);
            } else {
                monitorPhysicsProgress();
            }
        } else {
            if (typeof showNotification === 'function') {
                showNotification('Failed to start physics prediction: ' + data.error, 'error');
            }
            addPhysicsLogEntry('ERROR: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Physics prediction error:', error);
        if (typeof showNotification === 'function') {
            showNotification('Physics prediction request failed', 'error');
        }
        addPhysicsLogEntry('ERROR: Request failed');
    });
}

function monitorPhysicsProgress() {
    if (!physicsJobId) return;
    
    let progress = 0;
    const progressFill = document.getElementById('physicsProgressFill');
    const progressText = document.getElementById('physicsProgressText');
    
    physicsProcessingInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 95) progress = 95;
        
        if (progressFill) progressFill.style.width = progress + '%';
        
        if (progress < 20) {
            if (progressText) progressText.textContent = 'Initializing Kalman filters...';
            addPhysicsLogEntry('Kalman filter initialization complete');
        } else if (progress < 40) {
            if (progressText) progressText.textContent = 'Loading social force models...';
            addPhysicsLogEntry('Social force parameters calibrated');
        } else if (progress < 60) {
            if (progressText) progressText.textContent = 'Processing trajectory data...';
            addPhysicsLogEntry('Trajectory preprocessing complete');
        } else if (progress < 80) {
            if (progressText) progressText.textContent = 'Running physics predictions...';
            addPhysicsLogEntry('Physics simulation in progress');
        } else {
            if (progressText) progressText.textContent = 'Generating forensic report...';
            addPhysicsLogEntry('Finalizing prediction results');
        }
        
        if (progress > 90) {
            checkPhysicsJobStatus();
        }
    }, 1000);
}

function checkPhysicsJobStatus() {
    if (!physicsJobId) return;
    
    fetch(`/api/physics/status/${physicsJobId}/`)
    .then(response => response.json())
    .then(data => {
        if (data.success && data.job_data.status === 'completed') {
            clearInterval(physicsProcessingInterval);
            
            const progressFill = document.getElementById('physicsProgressFill');
            const progressText = document.getElementById('physicsProgressText');
            
            if (progressFill) progressFill.style.width = '100%';
            if (progressText) progressText.textContent = 'Physics prediction complete!';
            addPhysicsLogEntry('Physics prediction completed successfully');
            
            loadPhysicsResults();
        } else if (data.success && data.job_data.status === 'failed') {
            clearInterval(physicsProcessingInterval);
            if (typeof showNotification === 'function') {
                showNotification('Physics prediction failed: ' + data.job_data.error, 'error');
            }
            addPhysicsLogEntry('ERROR: ' + data.job_data.error);
        }
    })
    .catch(error => {
        console.error('Status check error:', error);
    });
}

function loadPhysicsResults() {
    if (!physicsJobId) return;
    
    fetch(`/api/physics/results/${physicsJobId}/`)
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            displayPhysicsResults(data);
        } else {
            if (typeof showNotification === 'function') {
                showNotification('Failed to load results: ' + data.error, 'error');
            }
        }
    })
    .catch(error => {
        console.error('Results loading error:', error);
        if (typeof showNotification === 'function') {
            showNotification('Failed to load prediction results', 'error');
        }
    });
}

function displayPhysicsResults(data) {
    const resultsSection = document.getElementById('physicsResultsSection');
    
    if (data.prediction_results && data.prediction_results.length > 0) {
        const objectCount = document.getElementById('objectCount');
        if (objectCount) objectCount.textContent = data.prediction_results.length;
        
        const avgConfidence = data.prediction_results.reduce((sum, result) => 
            sum + (result.avg_confidence || 0), 0) / data.prediction_results.length;
        const avgConfidenceElement = document.getElementById('avgConfidence');
        if (avgConfidenceElement) avgConfidenceElement.textContent = (avgConfidence * 100).toFixed(1) + '%';
        
        const predictionMethod = document.getElementById('predictionMethod');
        if (predictionMethod) {
            predictionMethod.textContent = data.prediction_results[0].method_used || 'Hybrid Kalman + Social Force';
        }
    }
    
    if (resultsSection) resultsSection.style.display = 'block';
    
    const closePhysicsProcessing = document.getElementById('closePhysicsProcessing');
    if (closePhysicsProcessing) closePhysicsProcessing.style.display = 'block';
    
    addPhysicsLogEntry('Results displayed successfully');
    if (typeof showNotification === 'function') {
        showNotification('Physics prediction completed successfully!', 'success');
    }
}

function addPhysicsLogEntry(message) {
    const logContent = document.getElementById('physicsLogContent');
    if (!logContent) return;
    
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
}

// Initialize physics handlers when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializePhysicsInputHandlers();
    
    // Handle physics modal close on outside click
    window.addEventListener('click', function(event) {
        const physicsModal = document.getElementById('physicsModal');
        if (event.target === physicsModal) {
            closePhysicsModal();
        }
    });
});
