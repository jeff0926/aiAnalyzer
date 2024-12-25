class GraphVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true });
        this.controls = null;
        this.nodes = new Map();
        this.edges = new Map();
        this.nodeLabels = new Map();
        this.selectedNode = null;
        this.highlightedNode = null;
        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.initScene();
        this.setupEvents();
    }

    initScene() {
        this.scene.background = new THREE.Color(0xf8fafc);
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
        this.container.appendChild(this.renderer.domElement);

        this.camera.position.z = 100;
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;

        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        const pointLight = new THREE.PointLight(0xffffff, 0.8);
        pointLight.position.set(70, 70, 70);
        this.scene.add(ambientLight, pointLight);

        this.addControls();
        this.animate();
    }

    addControls() {
        const controls = document.createElement('div');
        controls.className = 'graph-controls';
        controls.innerHTML = `
            <button class="graph-control-button" id="zoomToFit">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M4 8V4m0 0h4M4 4l5 5m11-2V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4"/>
                </svg>
            </button>
            <button class="graph-control-button" id="resetCamera">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
                </svg>
            </button>
        `;
        this.container.appendChild(controls);

        document.getElementById('zoomToFit').addEventListener('click', () => this.zoomToFit());
        document.getElementById('resetCamera').addEventListener('click', () => this.resetCamera());
    }

    setupEvents() {
        window.addEventListener('resize', () => this.onWindowResize());
        this.container.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.container.addEventListener('click', (e) => this.onClick(e));
        this.container.addEventListener('dblclick', (e) => this.onDoubleClick(e));
    }

    onWindowResize() {
        this.camera.aspect = this.container.clientWidth / this.container.clientHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    }

    onMouseMove(event) {
        const rect = this.container.getBoundingClientRect();
        this.mouse.x = ((event.clientX - rect.left) / this.container.clientWidth) * 2 - 1;
        this.mouse.y = -((event.clientY - rect.top) / this.container.clientHeight) * 2 + 1;

        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(Array.from(this.nodes.values()));

        if (intersects.length > 0) {
            const node = intersects[0].object;
            this.highlightNode(node);
            this.container.style.cursor = 'pointer';
        } else {
            this.unhighlightNode();
            this.container.style.cursor = 'default';
        }
    }

    onClick(event) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(Array.from(this.nodes.values()));

        if (intersects.length > 0) {
            const node = intersects[0].object;
            this.selectNode(node);
        } else {
            this.deselectNode();
        }
    }

    onDoubleClick(event) {
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObjects(Array.from(this.nodes.values()));

        if (intersects.length > 0) {
            const node = intersects[0].object;
            this.zoomToNode(node);
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.updateEdges();
        this.renderer.render(this.scene, this.camera);
    }

    updateGraph(graphData) {
        this.clearGraph();
        this.createNodes(graphData.nodes);
        this.createEdges(graphData.edges);
        this.applyForceLayout();
        this.zoomToFit();
    }

    createNodes(nodes) {
        nodes.forEach(nodeData => {
            const geometry = new THREE.SphereGeometry(nodeData.size || 1, 32, 32);
            const material = new THREE.MeshPhongMaterial({
                color: this.getNodeColor(nodeData.type),
                transparent: true,
                opacity: 0.8
            });

            const nodeMesh = new THREE.Mesh(geometry, material);
            nodeMesh.position.random().multiplyScalar(50);
            nodeMesh.userData = nodeData;

            const label = this.createNodeLabel(nodeData);
            nodeMesh.add(label);

            this.nodes.set(nodeData.id, nodeMesh);
            this.nodeLabels.set(nodeData.id, label);
            this.scene.add(nodeMesh);
        });
    }

    createEdges(edges) {
        edges.forEach(edgeData => {
            const sourceNode = this.nodes.get(edgeData.source);
            const targetNode = this.nodes.get(edgeData.target);

            if (sourceNode && targetNode) {
                const points = [sourceNode.position, targetNode.position];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const material = new THREE.LineBasicMaterial({
                    color: this.getEdgeColor(edgeData.type),
                    opacity: 0.6,
                    transparent: true
                });

                const edge = new THREE.Line(geometry, material);
                edge.userData = edgeData;

                this.edges.set(`${edgeData.source}-${edgeData.target}`, edge);
                this.scene.add(edge);
            }
        });
    }

    updateEdges() {
        this.edges.forEach(edge => {
            const sourceId = edge.userData.source;
            const targetId = edge.userData.target;
            const sourceNode = this.nodes.get(sourceId);
            const targetNode = this.nodes.get(targetId);

            if (sourceNode && targetNode) {
                const points = [sourceNode.position, targetNode.position];
                edge.geometry.setFromPoints(points);
            }
        });
    }

    createNodeLabel(nodeData) {
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        const fontSize = 24;
        context.font = `${fontSize}px Arial`;

        canvas.width = context.measureText(nodeData.name).width + 20;
        canvas.height = fontSize + 10;

        context.fillStyle = 'white';
        context.fillRect(0, 0, canvas.width, canvas.height);
        context.fillStyle = 'black';
        context.font = `${fontSize}px Arial`;
        context.fillText(nodeData.name, 10, fontSize);

        const texture = new THREE.CanvasTexture(canvas);
        const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
        const sprite = new THREE.Sprite(spriteMaterial);
        sprite.position.y = 2;
        sprite.scale.set(canvas.width / 40, canvas.height / 40, 1);

        return sprite;
    }

    getNodeColor(type) {
        const colors = {
            file: 0x4B5563,      // gray-600
            component: 0x3B82F6,  // blue-500
            function: 0x10B981,   // emerald-500
            class: 0x8B5CF6,      // violet-500
            module: 0xF59E0B,     // amber-500
            default: 0x6B7280     // gray-500
        };
        return colors[type] || colors.default;
    }

    getEdgeColor(type) {
        const colors = {
            imports: 0x9CA3AF,    // gray-400
            calls: 0xF59E0B,      // amber-500
            defines: 0x10B981,    // emerald-500
            contains: 0x6366F1,   // indigo-500
            default: 0xD1D5DB     // gray-300
        };
        return colors[type] || colors.default;
    }

    highlightNode(node) {
        if (this.highlightedNode !== node) {
            if (this.highlightedNode) {
                this.highlightedNode.material.emissive.setHex(0x000000);
            }
            this.highlightedNode = node;
            node.material.emissive.setHex(0x666666);
        }
    }

    unhighlightNode() {
        if (this.highlightedNode && this.highlightedNode !== this.selectedNode) {
            this.highlightedNode.material.emissive.setHex(0x000000);
            this.highlightedNode = null;
        }
    }

    selectNode(node) {
        if (this.selectedNode) {
            this.selectedNode.material.emissive.setHex(0x000000);
        }
        this.selectedNode = node;
        node.material.emissive.setHex(0x999999);

        const event = new CustomEvent('nodeSelected', { detail: node.userData });
        this.container.dispatchEvent(event);
    }

    deselectNode() {
        if (this.selectedNode) {
            this.selectedNode.material.emissive.setHex(0x000000);
            this.selectedNode = null;

            const event = new CustomEvent('nodeDeselected');
            this.container.dispatchEvent(event);
        }
    }

    zoomToNode(node) {
        const position = node.position.clone();
        const distance = 10;
        const direction = new THREE.Vector3().subVectors(this.camera.position, position).normalize();
        
        this.camera.position.copy(position.add(direction.multiplyScalar(distance)));
        this.controls.target.copy(node.position);
    }

    zoomToFit() {
        const box = new THREE.Box3();
        this.nodes.forEach(node => box.expandByObject(node));
        
        const size = box.getSize(new THREE.Vector3());
        const center = box.getCenter(new THREE.Vector3());
        
        const maxDim = Math.max(size.x, size.y, size.z);
        const fov = this.camera.fov * (Math.PI / 180);
        const cameraDistance = maxDim / (2 * Math.tan(fov / 2));
        
        const direction = new THREE.Vector3()
            .subVectors(this.camera.position, center)
            .normalize()
            .multiplyScalar(cameraDistance * 1.5);
        
        this.camera.position.copy(center.clone().add(direction));
        this.controls.target.copy(center);
        this.camera.updateProjectionMatrix();
    }

    resetCamera() {
        this.camera.position.set(0, 0, 100);
        this.controls.target.set(0, 0, 0);
    }

    applyForceLayout() {
        const simulation = d3.forceSimulation()
            .nodes(Array.from(this.nodes.values()))
            .force('charge', d3.forceManyBody().strength(-300))
            .force('center', d3.forceCenter())
            .force('link', d3.forceLink(Array.from(this.edges.values()))
                .id(d => d.userData.id)
                .distance(30))
            .on('tick', () => this.updateEdges());

        for (let i = 0; i < 300; ++i) simulation.tick();
        simulation.stop();
    }

    clearGraph() {
        this.nodes.forEach(node => this.scene.remove(node));
        this.edges.forEach(edge => this.scene.remove(edge));
        this.nodes.clear();
        this.edges.clear();
        this.nodeLabels.clear();
    }
}

// Initialize graph visualization
window.graphViz = new GraphVisualizer('graphVisualization');