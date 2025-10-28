// Test Analysis Tools - Minimal Working Version
console.log('Test analysis script loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, setting up analysis tools');
    
    // Find all analysis tool buttons
    const buttons = document.querySelectorAll('.tool-card .btn');
    console.log('Found buttons:', buttons.length);
    
    buttons.forEach((button, index) => {
        console.log(`Setting up button ${index}:`, button);
        
        button.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('Button clicked!', this);
            
            const toolCard = this.closest('.tool-card');
            const toolName = toolCard ? toolCard.querySelector('h4').textContent : 'Unknown Tool';
            
            console.log('Tool name:', toolName);
            
            // Show immediate feedback
            alert(`Starting ${toolName}!\n\nThis is a test - the tool would normally:\n1. Make API call\n2. Show progress modal\n3. Display results\n\nAPI endpoint would be: /api/analysis/object-detection/`);
            
            // Test API call
            testAPICall(toolName);
        });
    });
    
    function testAPICall(toolName) {
        console.log('Testing API call for:', toolName);
        
        fetch('/api/analysis/object-detection/', {
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
                alert(`SUCCESS!\n\nTool: ${toolName}\nJob ID: ${data.job_id}\nEstimated Duration: ${data.estimated_duration_seconds} seconds\n\nThe API is working! Progress modal would show here.`);
            } else {
                alert(`API Error: ${data.error}`);
            }
        })
        .catch(error => {
            console.error('API Error:', error);
            alert(`Network Error: ${error.message}`);
        });
    }
});
