/**
 * 3D Scene Viewer for HoloForensics
 * Three.js-based interactive 3D visualization of forensic scenes
 */

class ForensicSceneViewer {
    constructor(containerId, options = {}) {
        this.containerId = containerId;
        this.container = document.getElementById(containerId);
        
        // Configuration
        this.config = {
            enableControls: true,
            enableStats: false,
            enableGrid: true,
            enableAxes: true,
            backgroundColor: 0x1a1a1a,
            cameraPosition: { x: 10, y: 10, z: 10 },
            ...options
        };
        
        // Three.js components
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.stats = null;
        
        // Scene objects
        this.sceneObjects = new Map();
        this.trajectoryLines = new Map();
        this.eventMarkers = new Map();
        this.reconstructionMesh = null;
        
        // Animation
        this.animationId = null;
        this.clock = new THREE.Clock();
        this.isPlaying = false;
        this.currentTime = 0;
        this.maxTime = 60; // Default 60 seconds
        
        // Event handlers
        this.onObjectClick = null;
        this.onTimeUpdate = null;
        
        this.init();
    }
    
    init() {
        if (!this.container) {
            console.error(`Container ${this.containerId} not found`);
            return;
        }
        
        this.createScene();
        this.createCamera();
        this.createRenderer();
        this.createControls();
        this.createLights();
        this.createHelpers();
        
        if (this.config.enableStats) {
            this.createStats();
        }
        
        this.setupEventListeners();
        this.animate();
        
        console.log('ForensicSceneViewer initialized');
    }
    
    createScene() {
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(this.config.backgroundColor);
        this.scene.fog = new THREE.Fog(this.config.backgroundColor, 50, 200);
    }
    
    createCamera() {
        const aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
        
        const pos = this.config.cameraPosition;
        this.camera.position.set(pos.x, pos.y, pos.z);
        this.camera.lookAt(0, 0, 0);
    }
    
    createRenderer() {
        this.renderer = new THREE.WebGLRenderer({ 
            antialias: true,
            alpha: true
        });
        
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.0;
        
        this.container.appendChild(this.renderer.domElement);
    }
    
    createControls() {
        if (!this.config.enableControls) return;
        
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = false;
        this.controls.minDistance = 1;
        this.controls.maxDistance = 100;
        this.controls.maxPolarAngle = Math.PI / 2;
    }
    
    createLights() {
        // Ambient light
        const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
        this.scene.add(ambientLight);
        
        // Directional light (sun)
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight.position.set(50, 50, 25);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 2048;
        directionalLight.shadow.mapSize.height = 2048;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 500;
        directionalLight.shadow.camera.left = -50;
        directionalLight.shadow.camera.right = 50;
        directionalLight.shadow.camera.top = 50;
        directionalLight.shadow.camera.bottom = -50;
        this.scene.add(directionalLight);
        
        // Point lights for better illumination
        const pointLight1 = new THREE.PointLight(0xff4757, 0.5, 30);
        pointLight1.position.set(10, 10, 10);
        this.scene.add(pointLight1);
        
        const pointLight2 = new THREE.PointLight(0x3742fa, 0.5, 30);
        pointLight2.position.set(-10, 10, -10);
        this.scene.add(pointLight2);
    }
    
    createHelpers() {
        if (this.config.enableGrid) {
            const gridHelper = new THREE.GridHelper(50, 50, 0x444444, 0x222222);
            this.scene.add(gridHelper);
        }
        
        if (this.config.enableAxes) {
            const axesHelper = new THREE.AxesHelper(5);
            this.scene.add(axesHelper);
        }
    }
    
    createStats() {
        this.stats = new Stats();
        this.stats.showPanel(0);
        this.container.appendChild(this.stats.dom);
    }
    
    setupEventListeners() {
        // Window resize
        window.addEventListener('resize', () => this.onWindowResize(), false);
        
        // Mouse events for object interaction
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        
        this.renderer.domElement.addEventListener('click', (event) => {
            const rect = this.renderer.domElement.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
            
            raycaster.setFromCamera(mouse, this.camera);
            const intersects = raycaster.intersectObjects(this.scene.children, true);
            
            if (intersects.length > 0) {
                const object = intersects[0].object;
                if (this.onObjectClick) {
                    this.onObjectClick(object, intersects[0]);
                }
            }
        });
    }
    
    onWindowResize() {
        const width = this.container.clientWidth;
        const height = this.container.clientHeight;
        
        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(width, height);
    }
    
    animate() {
        this.animationId = requestAnimationFrame(() => this.animate());
        
        if (this.stats) this.stats.begin();
        
        if (this.controls) {
            this.controls.update();
        }
        
        // Update animation time
        if (this.isPlaying) {
            const delta = this.clock.getDelta();
            this.currentTime += delta;
            
            if (this.currentTime > this.maxTime) {
                this.currentTime = 0; // Loop
            }
            
            this.updateAnimations();
            
            if (this.onTimeUpdate) {
                this.onTimeUpdate(this.currentTime);
            }
        }
        
        this.renderer.render(this.scene, this.camera);
        
        if (this.stats) this.stats.end();
    }
    
    updateAnimations() {
        // Update trajectory animations
        this.trajectoryLines.forEach((trajectory, id) => {
            if (trajectory.animated && trajectory.points) {
                this.updateTrajectoryAnimation(trajectory);
            }
        });
        
        // Update object positions based on time
        this.sceneObjects.forEach((obj, id) => {
            if (obj.trajectory && obj.trajectory.length > 0) {
                this.updateObjectPosition(obj);
            }
        });
    }
    
    updateTrajectoryAnimation(trajectory) {
        const progress = (this.currentTime / this.maxTime) * trajectory.points.length;
        const visiblePoints = Math.floor(progress);
        
        if (trajectory.line && trajectory.line.geometry) {
            const positions = trajectory.line.geometry.attributes.position.array;
            const totalPoints = trajectory.points.length;
            
            // Show trajectory progressively
            for (let i = 0; i < totalPoints * 3; i += 3) {
                const pointIndex = i / 3;
                if (pointIndex <= visiblePoints) {
                    // Point is visible
                    positions[i + 1] = trajectory.points[pointIndex].y; // Show actual Y
                } else {
                    // Point is hidden
                    positions[i + 1] = -1000; // Hide below ground
                }
            }
            
            trajectory.line.geometry.attributes.position.needsUpdate = true;
        }
    }
    
    updateObjectPosition(obj) {
        if (!obj.trajectory || obj.trajectory.length === 0) return;
        
        const timeIndex = Math.floor((this.currentTime / this.maxTime) * obj.trajectory.length);
        const clampedIndex = Math.min(timeIndex, obj.trajectory.length - 1);
        
        if (obj.mesh && obj.trajectory[clampedIndex]) {
            const pos = obj.trajectory[clampedIndex];
            obj.mesh.position.set(pos.x, pos.y, pos.z);
        }
        this.animate();
    }
    
    loadReconstruction(reconstructionData) {
        if (!reconstructionData || !reconstructionData.available) {
            console.log('No reconstruction data available');
            return;
        }
        
        console.log(`Loading ${reconstructionData.type} reconstruction...`);
        
        // Load 3D mesh if available
        if (reconstructionData.mesh && reconstructionData.mesh.url) {
            this.loadMesh(reconstructionData.mesh.url, reconstructionData.type);
        }
        
        // Load point cloud if available  
        if (reconstructionData.point_cloud && reconstructionData.point_cloud.url) {
            this.loadPointCloud(reconstructionData.point_cloud.url);
        }
        
        // Set camera positions based on reconstruction cameras
        if (reconstructionData.cameras && reconstructionData.cameras.length > 0) {
            this.setupReconstructionCameras(reconstructionData.cameras);
        }
        
        // Update scene bounds
        if (reconstructionData.bounds) {
            this.updateSceneBounds(reconstructionData.bounds);
        }
    }
    
    loadMesh(meshUrl, reconstructionType = 'nerf') {
        const loader = new THREE.GLTFLoader();
        
        loader.load(
            meshUrl,
            (gltf) => {
                console.log(`${reconstructionType.toUpperCase()} mesh loaded successfully`);
                
                const mesh = gltf.scene;
                mesh.name = 'reconstruction_mesh';
                
                // Apply material enhancements for different reconstruction types
                mesh.traverse((child) => {
                    if (child.isMesh) {
                        if (reconstructionType === 'nerf') {
                            // NeRF-specific material settings
                            child.material.transparent = false;
                            child.material.opacity = 1.0;
                            child.material.side = THREE.FrontSide;
                        } else if (reconstructionType === '3dgs') {
                            // 3D Gaussian Splatting specific settings
                            child.material.transparent = true;
                            child.material.opacity = 0.9;
                            child.material.side = THREE.DoubleSide;
                        }
                        
                        // Enable shadows
                        child.castShadow = true;
                        child.receiveShadow = true;
                    }
                });
                
                // Scale and position the mesh
                const box = new THREE.Box3().setFromObject(mesh);
                const center = box.getCenter(new THREE.Vector3());
                const size = box.getSize(new THREE.Vector3());
                
                // Center the mesh
                mesh.position.sub(center);
                
                // Scale if necessary
                const maxDimension = Math.max(size.x, size.y, size.z);
                if (maxDimension > 20) {
                    const scale = 20 / maxDimension;
                    mesh.scale.setScalar(scale);
                }
                
                this.scene.add(mesh);
                this.reconstructionMesh = mesh;
                
                // Update layer visibility
                this.updateLayerVisibility();
            },
            (progress) => {
                console.log('Mesh loading progress:', (progress.loaded / progress.total * 100) + '%');
            },
            (error) => {
                console.error('Error loading mesh:', error);
                // Fallback to basic geometry if mesh fails to load
                this.createFallbackReconstruction();
            }
        );
    }
    
    createFallbackReconstruction() {
        // Create a simple wireframe box as fallback
        const geometry = new THREE.BoxGeometry(10, 10, 5);
        const material = new THREE.MeshBasicMaterial({
            color: 0x444444,
            wireframe: true,
            transparent: true,
            opacity: 0.3
        });
        
        const fallbackMesh = new THREE.Mesh(geometry, material);
        fallbackMesh.name = 'fallback_reconstruction';
        this.scene.add(fallbackMesh);
        
        console.log('Created fallback reconstruction visualization');
    }
    
    // Public API methods
    
    loadSceneData(sceneData) {
        console.log('Loading scene data:', sceneData);
        
        // Clear existing scene objects
        this.clearScene();
        
        // Load reconstruction mesh if available
        if (sceneData.reconstruction) {
            this.loadReconstruction(sceneData.reconstruction);
        }
        
        // Load detected objects
        if (sceneData.objects) {
            sceneData.objects.forEach(obj => this.addSceneObject(obj));
        }
        
        // Load trajectories
        if (sceneData.trajectories) {
            sceneData.trajectories.forEach(traj => this.addTrajectory(traj));
        }
        
        // Load events
        if (sceneData.events) {
            sceneData.events.forEach(event => this.addEventMarker(event));
        }
        
        // Update time bounds
        if (sceneData.duration) {
            this.maxTime = sceneData.duration;
        }
    }
    
    addSceneObject(objectData) {
        const { id, type, position, color, trajectory } = objectData;
        
        let geometry, material;
        
        // Create geometry based on object type
        switch (type) {
            case 'person':
                geometry = new THREE.CapsuleGeometry(0.3, 1.8, 4, 8);
                material = new THREE.MeshLambertMaterial({ color: color || 0xff4757 });
                break;
            case 'vehicle':
                geometry = new THREE.BoxGeometry(2, 1, 4);
                material = new THREE.MeshLambertMaterial({ color: color || 0x3742fa });
                break;
            case 'object':
                geometry = new THREE.SphereGeometry(0.5, 8, 6);
                material = new THREE.MeshLambertMaterial({ color: color || 0xffc107 });
                break;
            default:
                geometry = new THREE.BoxGeometry(1, 1, 1);
                material = new THREE.MeshLambertMaterial({ color: color || 0x888888 });
        }
        
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { id, type, ...objectData };
        
        if (position) {
            mesh.position.set(position.x, position.y, position.z);
        }
        
        this.scene.add(mesh);
        
        // Store object data
        this.sceneObjects.set(id, {
            mesh,
            trajectory,
            type,
            ...objectData
        });
        
        console.log(`Added ${type} object: ${id}`);
    }
    
    addTrajectory(trajectoryData) {
        const { id, points, color, animated = true } = trajectoryData;
        
        if (!points || points.length === 0) return;
        
        // Create trajectory line
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(points.length * 3);
        
        points.forEach((point, index) => {
            positions[index * 3] = point.x;
            positions[index * 3 + 1] = point.y;
            positions[index * 3 + 2] = point.z;
        });
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        const material = new THREE.LineBasicMaterial({ 
            color: color || 0x00ff00,
            linewidth: 3,
            transparent: true,
            opacity: 0.8
        });
        
        const line = new THREE.Line(geometry, material);
        this.scene.add(line);
        
        // Store trajectory data
        this.trajectoryLines.set(id, {
            line,
            points,
            animated,
            color
        });
        
        console.log(`Added trajectory: ${id} with ${points.length} points`);
    }
    
    addEventMarker(eventData) {
        const { id, position, type, timestamp, confidence } = eventData;
        
        // Create event marker
        const geometry = new THREE.SphereGeometry(0.2, 8, 6);
        const material = new THREE.MeshBasicMaterial({ 
            color: this.getEventColor(type, confidence),
            transparent: true,
            opacity: 0.8
        });
        
        const marker = new THREE.Mesh(geometry, material);
        marker.position.set(position.x, position.y + 0.5, position.z);
        marker.userData = { id, type, timestamp, confidence, ...eventData };
        
        // Add pulsing animation
        const scale = 1 + Math.sin(Date.now() * 0.005) * 0.2;
        marker.scale.setScalar(scale);
        
        this.scene.add(marker);
        this.eventMarkers.set(id, marker);
        
        console.log(`Added event marker: ${id} (${type})`);
    }
    
    getEventColor(type, confidence) {
        const baseColors = {
            'collision': 0xff4757,
            'interaction': 0x3742fa,
            'anomaly': 0xffc107,
            'movement': 0x2ed573,
            'default': 0x888888
        };
        
        const color = baseColors[type] || baseColors.default;
        
        // Adjust brightness based on confidence
        const brightness = 0.5 + (confidence * 0.5);
        return new THREE.Color(color).multiplyScalar(brightness);
    }
    
    loadReconstruction(reconstructionData) {
        // Load NeRF/3DGS reconstruction mesh
        const { meshUrl, textureUrl, position, scale } = reconstructionData;
        
        const loader = new THREE.GLTFLoader();
        loader.load(meshUrl, (gltf) => {
            this.reconstructionMesh = gltf.scene;
            
            if (position) {
                this.reconstructionMesh.position.set(position.x, position.y, position.z);
            }
            
            if (scale) {
                this.reconstructionMesh.scale.setScalar(scale);
            }
            
            this.reconstructionMesh.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                }
            });
            
            this.scene.add(this.reconstructionMesh);
            console.log('Loaded 3D reconstruction mesh');
        });
    }
    
    clearScene() {
        // Remove all scene objects
        this.sceneObjects.forEach((obj, id) => {
            if (obj.mesh) {
                this.scene.remove(obj.mesh);
            }
        });
        this.sceneObjects.clear();
        
        // Remove trajectories
        this.trajectoryLines.forEach((traj, id) => {
            if (traj.line) {
                this.scene.remove(traj.line);
            }
        });
        this.trajectoryLines.clear();
        
        // Remove event markers
        this.eventMarkers.forEach((marker, id) => {
            this.scene.remove(marker);
        });
        this.eventMarkers.clear();
        
        // Remove reconstruction
        if (this.reconstructionMesh) {
            this.scene.remove(this.reconstructionMesh);
            this.reconstructionMesh = null;
        }
    }
    
    // Animation controls
    play() {
        this.isPlaying = true;
        this.clock.start();
        console.log('Animation started');
    }
    
    pause() {
        this.isPlaying = false;
        console.log('Animation paused');
    }
    
    stop() {
        this.isPlaying = false;
        this.currentTime = 0;
        this.updateAnimations();
        console.log('Animation stopped');
    }
    
    setTime(time) {
        this.currentTime = Math.max(0, Math.min(time, this.maxTime));
        this.updateAnimations();
    }
    
    // Camera controls
    focusOnObject(objectId) {
        const obj = this.sceneObjects.get(objectId);
        if (obj && obj.mesh) {
            const position = obj.mesh.position;
            this.camera.lookAt(position);
            
            if (this.controls) {
                this.controls.target.copy(position);
                this.controls.update();
            }
        }
    }
    
    resetCamera() {
        const pos = this.config.cameraPosition;
        this.camera.position.set(pos.x, pos.y, pos.z);
        this.camera.lookAt(0, 0, 0);
        
        if (this.controls) {
            this.controls.target.set(0, 0, 0);
            this.controls.update();
        }
    }
    
    // Cleanup
    destroy() {
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
        }
        
        this.clearScene();
        
        if (this.renderer) {
            this.renderer.dispose();
            this.container.removeChild(this.renderer.domElement);
        }
        
        if (this.stats && this.stats.dom.parentNode) {
            this.container.removeChild(this.stats.dom);
        }
        
        console.log('ForensicSceneViewer destroyed');
    }
}

// Export for use in other modules
window.ForensicSceneViewer = ForensicSceneViewer;
