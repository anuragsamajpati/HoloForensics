/**
 * Dashboard Charts JavaScript
 * Handles Chart.js initialization and data visualization
 */

class DashboardCharts {
    constructor() {
        this.charts = {};
        this.chartColors = {
            primary: '#ff4757',
            secondary: '#ff6b7a',
            success: '#2ed573',
            warning: '#ffc107',
            info: '#1e90ff',
            light: '#f8f9fa',
            dark: '#343a40'
        };
        
        this.init();
    }
    
    init() {
        // Set Chart.js defaults
        Chart.defaults.color = '#ccc';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
        Chart.defaults.backgroundColor = 'rgba(255, 71, 87, 0.1)';
        
        // Initialize all charts
        this.initActivityChart();
        this.initAccuracyChart();
        this.initDistributionChart();
        
        // Add chart to dashboard charts registry
        if (window.dashboard) {
            window.dashboard.charts = this.charts;
        }
    }
    
    initActivityChart() {
        const ctx = document.getElementById('activityChart');
        if (!ctx) return;
        
        // Generate mock activity data for the last 30 days
        const data = this.generateActivityData(30);
        
        this.charts.activity = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Cases Created',
                    data: data.cases,
                    borderColor: this.chartColors.primary,
                    backgroundColor: 'rgba(255, 71, 87, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Objects Detected',
                    data: data.objects,
                    borderColor: this.chartColors.info,
                    backgroundColor: 'rgba(30, 144, 255, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }, {
                    label: 'Anomalies Found',
                    data: data.anomalies,
                    borderColor: this.chartColors.warning,
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        borderColor: this.chartColors.primary,
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            displayFormats: {
                                day: 'MMM dd'
                            }
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    initAccuracyChart() {
        const ctx = document.getElementById('accuracyChart');
        if (!ctx) return;
        
        // Generate accuracy data over time
        const data = this.generateAccuracyData();
        
        this.charts.accuracy = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['High Confidence (>90%)', 'Medium Confidence (70-90%)', 'Low Confidence (<70%)'],
                datasets: [{
                    data: [72, 23, 5],
                    backgroundColor: [
                        this.chartColors.success,
                        this.chartColors.warning,
                        this.chartColors.primary
                    ],
                    borderWidth: 0,
                    cutout: '70%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed + '%';
                            }
                        }
                    }
                }
            }
        });
    }
    
    initDistributionChart() {
        const ctx = document.getElementById('distributionChart');
        if (!ctx) return;
        
        this.charts.distribution = new Chart(ctx, {
            type: 'polarArea',
            data: {
                labels: ['Person', 'Vehicle', 'Object', 'Animal', 'Other'],
                datasets: [{
                    data: [45, 25, 15, 10, 5],
                    backgroundColor: [
                        'rgba(255, 71, 87, 0.7)',
                        'rgba(30, 144, 255, 0.7)',
                        'rgba(46, 213, 115, 0.7)',
                        'rgba(255, 193, 7, 0.7)',
                        'rgba(156, 39, 176, 0.7)'
                    ],
                    borderColor: [
                        this.chartColors.primary,
                        this.chartColors.info,
                        this.chartColors.success,
                        this.chartColors.warning,
                        '#9c27b0'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#fff',
                        bodyColor: '#fff',
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return context.label + ': ' + context.parsed + ' (' + percentage + '%)';
                            }
                        }
                    }
                },
                scales: {
                    r: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        pointLabels: {
                            color: '#ccc'
                        },
                        ticks: {
                            color: '#ccc',
                            backdropColor: 'transparent'
                        }
                    }
                }
            }
        });
    }
    
    // Update activity chart with new time range
    updateActivityChart(days) {
        if (!this.charts.activity) return;
        
        const data = this.generateActivityData(parseInt(days));
        
        this.charts.activity.data.labels = data.labels;
        this.charts.activity.data.datasets[0].data = data.cases;
        this.charts.activity.data.datasets[1].data = data.objects;
        this.charts.activity.data.datasets[2].data = data.anomalies;
        
        this.charts.activity.update('active');
    }
    
    // Generate mock activity data
    generateActivityData(days) {
        const labels = [];
        const cases = [];
        const objects = [];
        const anomalies = [];
        
        const now = new Date();
        
        for (let i = days - 1; i >= 0; i--) {
            const date = new Date(now);
            date.setDate(date.getDate() - i);
            
            labels.push(date);
            
            // Generate realistic data with some trends
            const baseCase = Math.floor(Math.random() * 3) + 1;
            const baseObjects = Math.floor(Math.random() * 50) + 20;
            const baseAnomalies = Math.floor(Math.random() * 5);
            
            // Add some weekly patterns (more activity on weekdays)
            const dayOfWeek = date.getDay();
            const weekdayMultiplier = (dayOfWeek >= 1 && dayOfWeek <= 5) ? 1.2 : 0.8;
            
            cases.push(Math.floor(baseCase * weekdayMultiplier));
            objects.push(Math.floor(baseObjects * weekdayMultiplier));
            anomalies.push(Math.floor(baseAnomalies * weekdayMultiplier));
        }
        
        return { labels, cases, objects, anomalies };
    }
    
    // Generate accuracy data
    generateAccuracyData() {
        const labels = [];
        const accuracy = [];
        
        // Generate 24 hours of accuracy data
        for (let i = 0; i < 24; i++) {
            labels.push(`${i.toString().padStart(2, '0')}:00`);
            
            // Simulate accuracy fluctuations (generally high with some variation)
            const baseAccuracy = 90 + Math.random() * 8; // 90-98%
            accuracy.push(Math.round(baseAccuracy * 10) / 10);
        }
        
        return { labels, accuracy };
    }
    
    // Real-time data updates
    startRealTimeUpdates() {
        setInterval(() => {
            this.updateChartsWithRealTimeData();
        }, 10000); // Update every 10 seconds
    }
    
    updateChartsWithRealTimeData() {
        // Update accuracy chart with new data point
        if (this.charts.accuracy) {
            const newAccuracy = 90 + Math.random() * 8;
            // In a real implementation, you would fetch this from an API
        }
        
        // Update distribution chart occasionally
        if (Math.random() < 0.1 && this.charts.distribution) { // 10% chance
            const dataset = this.charts.distribution.data.datasets[0];
            dataset.data = dataset.data.map(value => 
                Math.max(1, value + Math.floor(Math.random() * 3) - 1)
            );
            this.charts.distribution.update('none');
        }
    }
    
    // Resize charts when window resizes
    handleResize() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
    
    // Destroy all charts (cleanup)
    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.destroy === 'function') {
                chart.destroy();
            }
        });
        this.charts = {};
    }
}

// Initialize charts when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboardCharts = new DashboardCharts();
    
    // Start real-time updates
    window.dashboardCharts.startRealTimeUpdates();
    
    // Handle window resize
    window.addEventListener('resize', () => {
        window.dashboardCharts.handleResize();
    });
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboardCharts) {
        window.dashboardCharts.destroy();
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DashboardCharts;
}
