/**
 * Advanced Dashboard JavaScript
 * Handles dashboard interactions, data updates, and UI controls
 */

class AdvancedDashboard {
    constructor() {
        this.refreshInterval = null;
        this.charts = {};
        this.currentFilter = 'all';
        
        this.init();
    }
    
    init() {
        console.log('Initializing Advanced Dashboard...');
        
        // Initialize charts
        this.initializeCharts();
        
        // Set up auto-refresh
        this.startAutoRefresh();
        
        // Load initial data
        this.loadDashboardData();
        
        // Set up event listeners
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        // Timeline range selector
        const timelineRange = document.getElementById('timelineRange');
        if (timelineRange) {
            timelineRange.addEventListener('change', (e) => {
                this.updateActivityChart(e.target.value);
            });
        }
        
        // Refresh button
        window.refreshDashboard = () => this.refreshDashboard();
        window.exportReport = () => this.exportReport();
        
        // Case management functions
        window.viewCase = (caseId) => this.viewCase(caseId);
        window.open3DViewer = (caseId) => this.open3DViewer(caseId);
        window.exportCase = (caseId) => this.exportCase(caseId);
        window.filterCases = (filter) => this.filterCases(filter);
        
        // Insight actions
        window.reviewAnomaly = (caseId, sector) => this.reviewAnomaly(caseId, sector);
        window.dismissInsight = (insightId) => this.dismissInsight(insightId);
        window.optimizeProcessing = () => this.optimizeProcessing();
        window.viewTrends = () => this.viewTrends();
    }
    
    async loadDashboardData() {
        try {
            // Load metrics
            await this.updateMetrics();
            
            // Load recent cases
            await this.updateCasesTable();
            
            // Update performance indicators
            this.updatePerformanceIndicators();
            
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showNotification('Failed to load dashboard data', 'error');
        }
    }
    
    async updateMetrics() {
        try {
            const response = await fetch('/api/dashboard/metrics/', {
                headers: { 'X-CSRFToken': this.getCSRFToken() }
            });
            
            if (response.ok) {
                const data = await response.json();
                this.updateMetricCards(data);
            } else {
                // Use mock data if API not available
                this.updateMetricCards(this.getMockMetrics());
            }
        } catch (error) {
            console.error('Error fetching metrics:', error);
            this.updateMetricCards(this.getMockMetrics());
        }
    }
    
    updateMetricCards(data) {
        // Update active cases
        const activeCasesEl = document.getElementById('activeCases');
        if (activeCasesEl) {
            this.animateNumber(activeCasesEl, data.activeCases || 12);
        }
        
        // Update objects detected
        const objectsDetectedEl = document.getElementById('objectsDetected');
        if (objectsDetectedEl) {
            this.animateNumber(objectsDetectedEl, data.objectsDetected || 1247);
        }
        
        // Update trajectories analyzed
        const trajectoriesEl = document.getElementById('trajectoriesAnalyzed');
        if (trajectoriesEl) {
            this.animateNumber(trajectoriesEl, data.trajectoriesAnalyzed || 89);
        }
        
        // Update anomalies found
        const anomaliesEl = document.getElementById('anomaliesFound');
        if (anomaliesEl) {
            this.animateNumber(anomaliesEl, data.anomaliesFound || 23);
        }
    }
    
    animateNumber(element, targetValue) {
        const startValue = parseInt(element.textContent) || 0;
        const duration = 1000;
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            
            const currentValue = Math.floor(startValue + (targetValue - startValue) * progress);
            element.textContent = currentValue.toLocaleString();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }
    
    async updateCasesTable() {
        try {
            const response = await fetch('/api/dashboard/cases/', {
                headers: { 'X-CSRFToken': this.getCSRFToken() }
            });
            
            let cases;
            if (response.ok) {
                const data = await response.json();
                cases = data.cases;
            } else {
                cases = this.getMockCases();
            }
            
            this.renderCasesTable(cases);
        } catch (error) {
            console.error('Error fetching cases:', error);
            this.renderCasesTable(this.getMockCases());
        }
    }
    
    renderCasesTable(cases) {
        const tbody = document.getElementById('casesTableBody');
        if (!tbody) return;
        
        tbody.innerHTML = cases.map(case_ => `
            <tr>
                <td>${case_.id}</td>
                <td>${case_.title}</td>
                <td><span class="status-badge ${case_.status.toLowerCase()}">${case_.status}</span></td>
                <td>${case_.objects}</td>
                <td>${case_.confidence}%</td>
                <td>${case_.lastUpdated}</td>
                <td>
                    <button class="action-btn" onclick="viewCase('${case_.id}')">
                        <i class="fas fa-eye"></i>
                    </button>
                    <button class="action-btn" onclick="open3DViewer('${case_.id}')" ${case_.status !== 'Completed' ? 'disabled' : ''}>
                        <i class="fas fa-cube"></i>
                    </button>
                    <button class="action-btn" onclick="exportCase('${case_.id}')" ${case_.status === 'Processing' ? 'disabled' : ''}>
                        <i class="fas fa-download"></i>
                    </button>
                </td>
            </tr>
        `).join('');
    }
    
    updatePerformanceIndicators() {
        // Simulate real-time performance data
        setInterval(() => {
            const cpuBar = document.querySelector('.indicator-item:nth-child(1) .indicator-fill');
            const gpuBar = document.querySelector('.indicator-item:nth-child(2) .indicator-fill');
            const memoryBar = document.querySelector('.indicator-item:nth-child(3) .indicator-fill');
            
            const cpuValue = document.querySelector('.indicator-item:nth-child(1) .indicator-value');
            const gpuValue = document.querySelector('.indicator-item:nth-child(2) .indicator-value');
            const memoryValue = document.querySelector('.indicator-item:nth-child(3) .indicator-value');
            
            if (cpuBar && gpuBar && memoryBar) {
                const cpu = Math.floor(Math.random() * 30) + 50; // 50-80%
                const gpu = Math.floor(Math.random() * 40) + 60; // 60-100%
                const memory = Math.floor(Math.random() * 20) + 30; // 30-50%
                
                cpuBar.style.width = cpu + '%';
                gpuBar.style.width = gpu + '%';
                memoryBar.style.width = memory + '%';
                
                if (cpuValue) cpuValue.textContent = cpu + '%';
                if (gpuValue) gpuValue.textContent = gpu + '%';
                if (memoryValue) memoryValue.textContent = memory + '%';
            }
        }, 3000);
    }
    
    // Case management functions
    viewCase(caseId) {
        this.showNotification(`Opening case ${caseId}...`, 'info');
        // Redirect to case details or open modal
        window.location.href = `/cases/${caseId}/`;
    }
    
    open3DViewer(caseId) {
        this.showNotification(`Loading 3D viewer for ${caseId}...`, 'info');
        // Open 3D viewer - could redirect to scenes page or open modal
        window.location.href = `/scenes/?case=${caseId}&viewer=3d`;
    }
    
    exportCase(caseId) {
        this.showNotification(`Exporting case ${caseId}...`, 'info');
        // Trigger case export
        fetch(`/api/cases/${caseId}/export/`, {
            method: 'POST',
            headers: { 'X-CSRFToken': this.getCSRFToken() }
        }).then(response => {
            if (response.ok) {
                return response.blob();
            }
            throw new Error('Export failed');
        }).then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `${caseId}_report.pdf`;
            a.click();
            window.URL.revokeObjectURL(url);
            this.showNotification('Case exported successfully', 'success');
        }).catch(error => {
            console.error('Export error:', error);
            this.showNotification('Export failed', 'error');
        });
    }
    
    filterCases(filter) {
        this.currentFilter = filter;
        
        // Update button states
        document.querySelectorAll('.table-actions .btn-sm').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
        
        // Filter table rows
        const rows = document.querySelectorAll('#casesTableBody tr');
        rows.forEach(row => {
            const status = row.querySelector('.status-badge').textContent.toLowerCase();
            
            if (filter === 'all' || status === filter) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }
    
    // Insight actions
    reviewAnomaly(caseId, sector) {
        this.showNotification(`Reviewing anomaly in ${caseId} sector ${sector}...`, 'info');
        // Open anomaly review interface
        window.location.href = `/cases/${caseId}/anomaly/${sector}/`;
    }
    
    dismissInsight(insightId) {
        const insightElement = document.querySelector(`.insight-item:nth-child(${insightId})`);
        if (insightElement) {
            insightElement.style.opacity = '0.5';
            insightElement.style.pointerEvents = 'none';
            
            setTimeout(() => {
                insightElement.remove();
            }, 300);
        }
    }
    
    optimizeProcessing() {
        this.showNotification('Applying processing optimizations...', 'info');
        
        // Simulate optimization
        setTimeout(() => {
            this.showNotification('Processing optimized successfully', 'success');
        }, 2000);
    }
    
    viewTrends() {
        this.showNotification('Loading trend analysis...', 'info');
        // Open trends view
        window.location.href = '/analytics/trends/';
    }
    
    // Dashboard controls
    refreshDashboard() {
        this.showNotification('Refreshing dashboard...', 'info');
        this.loadDashboardData();
        
        // Add loading animation to metric cards
        document.querySelectorAll('.metric-card').forEach(card => {
            card.classList.add('loading');
        });
        
        setTimeout(() => {
            document.querySelectorAll('.metric-card').forEach(card => {
                card.classList.remove('loading');
            });
            this.showNotification('Dashboard refreshed', 'success');
        }, 1500);
    }
    
    exportReport() {
        this.showNotification('Generating dashboard report...', 'info');
        
        // Generate and download report
        fetch('/api/dashboard/export/', {
            method: 'POST',
            headers: { 'X-CSRFToken': this.getCSRFToken() }
        }).then(response => {
            if (response.ok) {
                return response.blob();
            }
            throw new Error('Report generation failed');
        }).then(blob => {
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `dashboard_report_${new Date().toISOString().split('T')[0]}.pdf`;
            a.click();
            window.URL.revokeObjectURL(url);
            this.showNotification('Report exported successfully', 'success');
        }).catch(error => {
            console.error('Export error:', error);
            this.showNotification('Report export failed', 'error');
        });
    }
    
    startAutoRefresh() {
        // Refresh every 30 seconds
        this.refreshInterval = setInterval(() => {
            this.updateMetrics();
            this.updatePerformanceIndicators();
        }, 30000);
    }
    
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
    
    // Utility functions
    getCSRFToken() {
        return document.querySelector('[name=csrfmiddlewaretoken]')?.value || 
               document.querySelector('meta[name="csrf-token"]')?.getAttribute('content') || '';
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check' : type === 'error' ? 'exclamation-triangle' : 'info'}"></i>
            <span>${message}</span>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Remove after delay
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
    
    // Mock data for development
    getMockMetrics() {
        return {
            activeCases: 12,
            objectsDetected: 1247,
            trajectoriesAnalyzed: 89,
            anomaliesFound: 23
        };
    }
    
    getMockCases() {
        return [
            {
                id: 'CASE_2024_001',
                title: 'Indoor Incident Analysis',
                status: 'Completed',
                objects: 47,
                confidence: 94.2,
                lastUpdated: '2 hours ago'
            },
            {
                id: 'CASE_2024_002',
                title: 'Outdoor Investigation',
                status: 'Processing',
                objects: 23,
                confidence: 87.6,
                lastUpdated: '1 hour ago'
            },
            {
                id: 'CASE_2024_003',
                title: 'Evidence Timeline Analysis',
                status: 'Ready',
                objects: 156,
                confidence: 91.8,
                lastUpdated: '30 minutes ago'
            }
        ];
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new AdvancedDashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard) {
        window.dashboard.stopAutoRefresh();
    }
});
