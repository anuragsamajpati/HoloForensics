/**
 * 3D Scene Viewer Integration for HoloForensics
 * Integrates Three.js viewer with Django backend and forensic data
 */

class SceneViewerManager {
    constructor() {
        this.viewer = null;
        this.currentSceneId = null;
        this.sceneData = null;
        this.isInitialized = false;
        
        // UI elements
        this.viewerContainer = null;
        this.controlsPanel = null;
        this.timelineSlider = null;
        this.playButton = null;
        this.objectsList = null;
        
        // State
        this.selectedObjects = new Set();
        this.visibleLayers = new Set(['objects', 'trajectories', 'events', 'reconstruction']);
        
        this.init();
    }
    
    init() {
        this.setupUI();
        this.bindEvents();
        console.log('SceneViewerManager initialized');
    }
    
    setupUI() {
        // Create viewer container if it doesn't exist
        this.viewerContainer = document.getElementById('scene3DViewer');
        if (!this.viewerContainer) {
            console.warn('3D viewer container not found');
            return;
        }
        
        // Get UI controls
        this.controlsPanel = document.getElementById('viewer3DControls');
        this.timelineSlider = document.getElementById('timelineSlider');
        this.playButton = document.getElementById('playButton');
        this.objectsList = document.getElementById('objectsList');
    }
    
    bindEvents() {
        // Timeline controls
        if (this.timelineSlider) {
            this.timelineSlider.addEventListener('input', (e) => {
                const time = parseFloat(e.target.value);
                if (this.viewer) {
                    this.viewer.setTime(time);
                }
                this.updateTimeDisplay(time);
            });
        }
        
        // Play/pause button
        if (this.playButton) {
            this.playButton.addEventListener('click', () => {
                this.togglePlayback();
            });
        }
        
        // Layer visibility toggles
        document.querySelectorAll('.layer-toggle').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const layer = e.target.dataset.layer;
                if (e.target.checked) {
                    this.visibleLayers.add(layer);
                } else {
                    this.visibleLayers.delete(layer);
                }
                this.updateLayerVisibility();
            });
        });
        
        // Camera controls
        document.getElementById('resetCamera')?.addEventListener('click', () => {
            if (this.viewer) this.viewer.resetCamera();
        });
        
        document.getElementById('topView')?.addEventListener('click', () => {
            this.setCameraView('top');
        });
        
        document.getElementById('sideView')?.addEventListener('click', () => {
            this.setCameraView('side');
        });
        
        document.getElementById('frontView')?.addEventListener('click', () => {
            this.setCameraView('front');
        });
    }
    
    async loadScene(sceneId) {
        try {
            console.log(`Loading 3D scene: ${sceneId}`);
            
            // Show loading indicator
            this.showLoading(true);
            
            // Fetch scene data from API
            const sceneData = await this.fetchSceneData(sceneId);
            
            if (!sceneData) {
                throw new Error('No scene data received');
            }
            
            // Initialize viewer if not already done
            if (!this.viewer) {
                this.initializeViewer();
            }
            
            // Load data into viewer
            this.viewer.loadSceneData(sceneData);
            
            // Update UI
            this.updateSceneInfo(sceneData);
            this.updateObjectsList(sceneData.objects || []);
            this.updateTimelineControls(sceneData.duration || 60);
            
            this.currentSceneId = sceneId;
            this.sceneData = sceneData;
            
            console.log('3D scene loaded successfully');
            
        } catch (error) {
            console.error('Error loading 3D scene:', error);
            this.showError('Failed to load 3D scene: ' + error.message);
        } finally {
            this.showLoading(false);
        }
    }
    
    async fetchSceneData(sceneId) {
        try {
            const response = await fetch(`/api/3d/scene/${sceneId}/`, {
                headers: { 'X-CSRFToken': this.getCSRFToken() }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            return result.success ? result.data : null;
        } catch (error) {
            console.error('Error fetching scene results:', error);
            return null;
        }
    }
    
    async fetchObjects(sceneId) {
        try {
            const response = await fetch(`/api/3d/scene/${sceneId}/objects/`, {
                headers: { 'X-CSRFToken': this.getCSRFToken() }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            return result.success ? result.objects : [];
        } catch (error) {
            console.error('Error fetching objects:', error);
            return [];
        }
    }
    
    async fetchTrajectories(sceneId) {
        try {
            const response = await fetch(`/api/3d/scene/${sceneId}/trajectories/`, {
                headers: { 'X-CSRFToken': this.getCSRFToken() }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            return result.success ? result.trajectories : [];
        } catch (error) {
            console.error('Error fetching trajectories:', error);
            return [];
        }
    }
    
    async fetchEvents(sceneId) {
        try {
            const response = await fetch(`/api/3d/scene/${sceneId}/events/`, {
                headers: { 'X-CSRFToken': this.getCSRFToken() }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            return result.success ? result.events : [];
        } catch (error) {
            console.error('Error fetching events:', error);
            return [];
        }
    }
    
    async fetchReconstruction(sceneId) {
        try {
            const response = await fetch(`/api/3d/scene/${sceneId}/reconstruction/`, {
                headers: { 'X-CSRFToken': this.getCSRFToken() }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            return result.success ? result.reconstruction : null;
        } catch (error) {
            console.error('Error fetching reconstruction:', error);
            return null;
        }
    }
    
    processSceneObjects(sceneResults) {
        if (!sceneResults || !sceneResults.detections) return [];
        
        return sceneResults.detections.map(detection => ({
            id: detection.id || `obj_${Math.random().toString(36).substr(2, 9)}`,
            type: detection.class_name || 'object',
            position: {
                x: detection.bbox_3d?.center?.x || 0,
                y: detection.bbox_3d?.center?.y || 0,
                z: detection.bbox_3d?.center?.z || 0
            },
            color: this.getObjectColor(detection.class_name),
            confidence: detection.confidence || 0.5,
            timestamp: detection.timestamp || 0,
            metadata: detection
        }));
    }
    
    processTrajectories(trajectoryData) {
        if (!trajectoryData || !trajectoryData.predictions) return [];
        
        return trajectoryData.predictions.map(pred => ({
            id: pred.object_id || `traj_${Math.random().toString(36).substr(2, 9)}`,
            points: pred.trajectory?.map(point => ({
                x: point.x || 0,
                y: point.y || 0,
                z: point.z || 0,
                timestamp: point.timestamp || 0
            })) || [],
            color: this.getTrajectoryColor(pred.object_type),
            confidence: pred.confidence || 0.5,
            animated: true
        }));
    }
    
    processEvents(eventsData) {
        if (!eventsData || !eventsData.events) return [];
        
        return eventsData.events.map(event => ({
            id: event.id || `event_${Math.random().toString(36).substr(2, 9)}`,
            type: event.event_type || 'unknown',
            position: {
                x: event.location?.x || 0,
                y: event.location?.y || 0.5,
                z: event.location?.z || 0
            },
            timestamp: event.timestamp || 0,
            confidence: event.confidence || 0.5,
            severity: event.severity || 'medium',
            description: event.description || ''
        }));
    }
    
    processReconstruction(sceneResults) {
        if (!sceneResults || !sceneResults.reconstruction) return null;
        
        return {
            meshUrl: sceneResults.reconstruction.mesh_url,
            textureUrl: sceneResults.reconstruction.texture_url,
            position: { x: 0, y: 0, z: 0 },
            scale: 1.0
        };
    }
    
    calculateDuration(trajectories, events) {
        let maxTime = 60; // Default
        
        if (trajectories && trajectories.predictions) {
            trajectories.predictions.forEach(pred => {
                if (pred.trajectory) {
                    pred.trajectory.forEach(point => {
                        if (point.timestamp > maxTime) {
                            maxTime = point.timestamp;
                        }
                    });
                }
            });
        }
        
        if (events && events.events) {
            events.events.forEach(event => {
                if (event.timestamp > maxTime) {
                    maxTime = event.timestamp;
                }
            });
        }
        
        return maxTime;
    }
    
    getObjectColor(className) {
        const colors = {
            'person': 0xff4757,
            'car': 0x3742fa,
            'truck': 0x2f3542,
            'bicycle': 0x2ed573,
            'motorcycle': 0xffa502,
            'bus': 0x1e90ff,
            'default': 0x888888
        };
        
        return colors[className] || colors.default;
    }
    
    getTrajectoryColor(objectType) {
        const colors = {
            'person': 0x00ff00,
            'vehicle': 0x0099ff,
            'object': 0xffff00,
            'default': 0x00ff00
        };
        
        return colors[objectType] || colors.default;
    }
    
    initializeViewer() {
        if (this.isInitialized) return;
        
        this.viewer = new ForensicSceneViewer('scene3DViewer', {
            enableControls: true,
            enableStats: true,
            enableGrid: true,
            enableAxes: true
        });
        
        // Set up viewer event handlers
        this.viewer.onObjectClick = (object, intersection) => {
            this.handleObjectClick(object, intersection);
        };
        
        this.viewer.onTimeUpdate = (time) => {
            this.handleTimeUpdate(time);
        };
        
        this.isInitialized = true;
        console.log('3D viewer initialized');
    }
    
    handleObjectClick(object, intersection) {
        const userData = object.userData;
        console.log('Object clicked:', userData);
        
        // Update selection
        if (userData.id) {
            if (this.selectedObjects.has(userData.id)) {
                this.selectedObjects.delete(userData.id);
                this.unhighlightObject(object);
            } else {
                this.selectedObjects.add(userData.id);
                this.highlightObject(object);
            }
            
            this.updateObjectsList();
        }
        
        // Show object details
        this.showObjectDetails(userData);
    }
    
    handleTimeUpdate(time) {
        // Update timeline slider
        if (this.timelineSlider) {
            this.timelineSlider.value = time;
        }
        
        this.updateTimeDisplay(time);
    }
    
    highlightObject(object) {
        if (object.material) {
            object.material.emissive = new THREE.Color(0x444444);
        }
    }
    
    unhighlightObject(object) {
        if (object.material) {
            object.material.emissive = new THREE.Color(0x000000);
        }
    }
    
    updateSceneInfo(sceneData) {
        const infoPanel = document.getElementById('sceneInfo');
        if (!infoPanel) return;
        
        infoPanel.innerHTML = `
            <div class="scene-info-item">
                <strong>Scene ID:</strong> ${sceneData.id}
            </div>
            <div class="scene-info-item">
                <strong>Objects:</strong> ${sceneData.objects?.length || 0}
            </div>
            <div class="scene-info-item">
                <strong>Trajectories:</strong> ${sceneData.trajectories?.length || 0}
            </div>
            <div class="scene-info-item">
                <strong>Events:</strong> ${sceneData.events?.length || 0}
            </div>
            <div class="scene-info-item">
                <strong>Duration:</strong> ${sceneData.duration?.toFixed(1) || 0}s
            </div>
        `;
    }
    
    updateObjectsList(objects = []) {
        if (!this.objectsList) return;
        
        this.objectsList.innerHTML = '';
        
        objects.forEach(obj => {
            const item = document.createElement('div');
            item.className = 'object-list-item';
            item.innerHTML = `
                <div class="object-info">
                    <span class="object-type">${obj.type}</span>
                    <span class="object-id">${obj.id}</span>
                    <span class="object-confidence">${(obj.confidence * 100).toFixed(1)}%</span>
                </div>
                <div class="object-actions">
                    <button onclick="sceneViewerManager.focusObject('${obj.id}')" class="btn-sm">Focus</button>
                    <button onclick="sceneViewerManager.toggleObjectVisibility('${obj.id}')" class="btn-sm">Toggle</button>
                </div>
            `;
            
            if (this.selectedObjects.has(obj.id)) {
                item.classList.add('selected');
            }
            
            this.objectsList.appendChild(item);
        });
    }
    
    updateTimelineControls(duration) {
        if (this.timelineSlider) {
            this.timelineSlider.max = duration;
            this.timelineSlider.step = duration / 1000; // 1000 steps
        }
        
        // Update duration display
        const durationDisplay = document.getElementById('durationDisplay');
        if (durationDisplay) {
            durationDisplay.textContent = `${duration.toFixed(1)}s`;
        }
    }
    
    updateTimeDisplay(time) {
        const timeDisplay = document.getElementById('currentTimeDisplay');
        if (timeDisplay) {
            timeDisplay.textContent = `${time.toFixed(1)}s`;
        }
    }
    
    updateLayerVisibility() {
        if (!this.viewer) return;
        
        // Update object visibility based on layers
        this.viewer.sceneObjects.forEach((obj, id) => {
            if (obj.mesh) {
                obj.mesh.visible = this.visibleLayers.has('objects');
            }
        });
        
        this.viewer.trajectoryLines.forEach((traj, id) => {
            if (traj.line) {
                traj.line.visible = this.visibleLayers.has('trajectories');
            }
        });
        
        this.viewer.eventMarkers.forEach((marker, id) => {
            marker.visible = this.visibleLayers.has('events');
        });
        
        if (this.viewer.reconstructionMesh) {
            this.viewer.reconstructionMesh.visible = this.visibleLayers.has('reconstruction');
        }
    }
    
    togglePlayback() {
        if (!this.viewer) return;
        
        if (this.viewer.isPlaying) {
            this.viewer.pause();
            this.playButton.innerHTML = '<i class="fas fa-play"></i>';
        } else {
            this.viewer.play();
            this.playButton.innerHTML = '<i class="fas fa-pause"></i>';
        }
    }
    
    setCameraView(view) {
        if (!this.viewer) return;
        
        const positions = {
            'top': { x: 0, y: 50, z: 0 },
            'side': { x: 50, y: 10, z: 0 },
            'front': { x: 0, y: 10, z: 50 }
        };
        
        const pos = positions[view];
        if (pos) {
            this.viewer.camera.position.set(pos.x, pos.y, pos.z);
            this.viewer.camera.lookAt(0, 0, 0);
            
            if (this.viewer.controls) {
                this.viewer.controls.target.set(0, 0, 0);
                this.viewer.controls.update();
            }
        }
    }
    
    focusObject(objectId) {
        if (this.viewer) {
            this.viewer.focusOnObject(objectId);
        }
    }
    
    toggleObjectVisibility(objectId) {
        const obj = this.viewer?.sceneObjects.get(objectId);
        if (obj && obj.mesh) {
            obj.mesh.visible = !obj.mesh.visible;
        }
    }
    
    showObjectDetails(userData) {
        const detailsPanel = document.getElementById('objectDetails');
        if (!detailsPanel) return;
        
        detailsPanel.innerHTML = `
            <h4>Object Details</h4>
            <div class="detail-item"><strong>ID:</strong> ${userData.id || 'N/A'}</div>
            <div class="detail-item"><strong>Type:</strong> ${userData.type || 'N/A'}</div>
            <div class="detail-item"><strong>Confidence:</strong> ${((userData.confidence || 0) * 100).toFixed(1)}%</div>
            <div class="detail-item"><strong>Timestamp:</strong> ${(userData.timestamp || 0).toFixed(2)}s</div>
            ${userData.metadata ? `<div class="detail-item"><strong>Metadata:</strong> <pre>${JSON.stringify(userData.metadata, null, 2)}</pre></div>` : ''}
        `;
        
        detailsPanel.style.display = 'block';
    }
    
    showLoading(show) {
        const loadingIndicator = document.getElementById('viewer3DLoading');
        if (loadingIndicator) {
            loadingIndicator.style.display = show ? 'flex' : 'none';
        }
    }
    
    showError(message) {
        const errorContainer = document.getElementById('viewer3DError');
        if (errorContainer) {
            errorContainer.innerHTML = `<div class="error-message">${message}</div>`;
            errorContainer.style.display = 'block';
            
            setTimeout(() => {
                errorContainer.style.display = 'none';
            }, 5000);
        }
    }
    
    getCSRFToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value || 
               document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
    }
    
    destroy() {
        if (this.viewer) {
            this.viewer.destroy();
            this.viewer = null;
        }
        
        this.isInitialized = false;
        console.log('SceneViewerManager destroyed');
    }
}

// Global instance
let sceneViewerManager = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    sceneViewerManager = new SceneViewerManager();
});

// Export for global access
window.sceneViewerManager = sceneViewerManager;
