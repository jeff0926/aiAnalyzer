
class AnalysisDashboard {
    // === Initialization ===
    constructor() {
        this.state = {
            currentRepo: null,
            analysisResults: null,
            theme: 'light',
            analysisDepth: 'standard',
            isAnalyzing: false
        };
        
        this.initializeEventListeners();
        this.initializeTheme();
    }

    initializeEventListeners() {
        // Repository analysis events
        document.getElementById('newAnalysisBtn')
            .addEventListener('click', () => {
                document.getElementById('repoSelection').classList.remove('hidden');
            });

        document.getElementById('repoForm')
            .addEventListener('submit', (e) => {
                e.preventDefault();
                this.startAnalysis(document.getElementById('repoUrl').value);
            });

        // Settings events
        document.getElementById('settingsBtn')
            .addEventListener('click', () => {
                document.getElementById('settingsModal').classList.remove('hidden');
            });

        document.getElementById('closeSettingsBtn')
            .addEventListener('click', () => {
                document.getElementById('settingsModal').classList.add('hidden');
            });

        document.getElementById('themeSelect')
            .addEventListener('change', (e) => {
                this.setTheme(e.target.value);
            });

        document.getElementById('analysisDepthSelect')
            .addEventListener('change', (e) => {
                this.setAnalysisDepth(e.target.value);
            });

        // Tab navigation
        document.querySelectorAll('.tab-button')
            .forEach(button => {
                button.addEventListener('click', () => 
                    this.switchTab(button.dataset.tab)
                );
            });
    }

    // === Core Analysis Methods ===
    async startAnalysis(repoUrl) {
        try {
            this.setState({ isAnalyzing: true });
            document.getElementById('repoSelection').classList.add('hidden');
            this.showLoadingState();

            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    repoUrl,
                    depth: this.state.analysisDepth
                })
            });

            const results = await response.json();
            if (!response.ok) throw new Error(results.error);

            this.setState({ 
                currentRepo: repoUrl,
                analysisResults: results
            });

            this.updateDashboard(results);
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.setState({ isAnalyzing: false });
            this.hideLoadingState();
        }
    }

    updateDashboard(results) {
        // Update metrics
        document.getElementById('filesCount').textContent = results.fileCount;
        document.getElementById('componentsCount').textContent = results.componentCount;
        document.getElementById('issuesCount').textContent = results.issueCount;

        // Update visualizations
        window.graphViz.updateGraph(results.knowledgeGraph);
        this.updateAnalysisResults(results);
        this.updateRecommendations(results.recommendations);
    }

    updateAnalysisResults(results) {
        const sections = {
            overview: this.generateOverviewContent(results),
            architecture: this.generateArchitectureContent(results),
            security: this.generateSecurityContent(results),
            performance: this.generatePerformanceContent(results)
        };

        for (const [tab, content] of Object.entries(sections)) {
            document.getElementById(tab).innerHTML = content;
        }
    }

    // === UI State Management ===
    switchTab(tabId) {
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.toggle('active', button.dataset.tab === tabId);
        });
        
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('hidden', content.id !== tabId);
        });
    }

    showLoadingState() {
        const loadingElements = [
            ...document.querySelectorAll('.metric-value'),
            document.getElementById('graphVisualization'),
            ...document.querySelectorAll('.tab-content')
        ];

        loadingElements.forEach(el => {
            el.classList.add('loading');
        });
    }

    hideLoadingState() {
        const loadingElements = document.querySelectorAll('.loading, .loading-block');
        loadingElements.forEach(el => {
            el.classList.remove('loading', 'loading-block');
        });
    }

    showError(message) {
        console.error('Analysis Error:', message);
        // TODO: Implement user-facing error notification
    }
}

class AnalysisDashboard {
    // === Initialization ===
    constructor() {
        this.state = {
            currentRepo: null,
            analysisResults: null,
            theme: 'light',
            analysisDepth: 'standard',
            isAnalyzing: false
        };
        
        this.initializeEventListeners();
        this.initializeTheme();
    }

    initializeEventListeners() {
        // Repository analysis events
        document.getElementById('newAnalysisBtn')
            .addEventListener('click', () => {
                document.getElementById('repoSelection').classList.remove('hidden');
            });

        document.getElementById('repoForm')
            .addEventListener('submit', (e) => {
                e.preventDefault();
                this.startAnalysis(document.getElementById('repoUrl').value);
            });

        // Settings events
        document.getElementById('settingsBtn')
            .addEventListener('click', () => {
                document.getElementById('settingsModal').classList.remove('hidden');
            });

        document.getElementById('closeSettingsBtn')
            .addEventListener('click', () => {
                document.getElementById('settingsModal').classList.add('hidden');
            });

        document.getElementById('themeSelect')
            .addEventListener('change', (e) => {
                this.setTheme(e.target.value);
            });

        document.getElementById('analysisDepthSelect')
            .addEventListener('change', (e) => {
                this.setAnalysisDepth(e.target.value);
            });

        // Tab navigation
        document.querySelectorAll('.tab-button')
            .forEach(button => {
                button.addEventListener('click', () => 
                    this.switchTab(button.dataset.tab)
                );
            });
    }

    // === Core Analysis Methods ===
    async startAnalysis(repoUrl) {
        try {
            this.setState({ isAnalyzing: true });
            document.getElementById('repoSelection').classList.add('hidden');
            this.showLoadingState();

            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    repoUrl,
                    depth: this.state.analysisDepth
                })
            });

            const results = await response.json();
            if (!response.ok) throw new Error(results.error);

            this.setState({ 
                currentRepo: repoUrl,
                analysisResults: results
            });

            this.updateDashboard(results);
        } catch (error) {
            this.showError(error.message);
        } finally {
            this.setState({ isAnalyzing: false });
            this.hideLoadingState();
        }
    }

    updateDashboard(results) {
        // Update metrics
        document.getElementById('filesCount').textContent = results.fileCount;
        document.getElementById('componentsCount').textContent = results.componentCount;
        document.getElementById('issuesCount').textContent = results.issueCount;

        // Update visualizations
        window.graphViz.updateGraph(results.knowledgeGraph);
        this.updateAnalysisResults(results);
        this.updateRecommendations(results.recommendations);
    }

    updateAnalysisResults(results) {
        const sections = {
            overview: this.generateOverviewContent(results),
            architecture: this.generateArchitectureContent(results),
            security: this.generateSecurityContent(results),
            performance: this.generatePerformanceContent(results)
        };

        for (const [tab, content] of Object.entries(sections)) {
            document.getElementById(tab).innerHTML = content;
        }
    }

    // === UI State Management ===
    switchTab(tabId) {
        document.querySelectorAll('.tab-button').forEach(button => {
            button.classList.toggle('active', button.dataset.tab === tabId);
        });
        
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.toggle('hidden', content.id !== tabId);
        });
    }

    showLoadingState() {
        const loadingElements = [
            ...document.querySelectorAll('.metric-value'),
            document.getElementById('graphVisualization'),
            ...document.querySelectorAll('.tab-content')
        ];

        loadingElements.forEach(el => {
            el.classList.add('loading');
        });
    }

    hideLoadingState() {
        const loadingElements = document.querySelectorAll('.loading, .loading-block');
        loadingElements.forEach(el => {
            el.classList.remove('loading', 'loading-block');
        });
    }

    showError(message) {
        console.error('Analysis Error:', message);
        // TODO: Implement user-facing error notification
    }
}