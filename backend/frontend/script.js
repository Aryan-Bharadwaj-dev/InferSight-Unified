// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Global variables
let fileQueue = [];
let currentResults = [];

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const textInput = document.getElementById('text-input');
const analyzeTextBtn = document.getElementById('analyze-text-btn');
const clearTextBtn = document.getElementById('clear-text-btn');
const charCount = document.getElementById('char-count');
const loading = document.getElementById('loading');
const resultsSection = document.getElementById('results-section');
const resultsContainer = document.getElementById('results-container');
const fileQueueElement = document.getElementById('file-queue');
const queueContainer = document.getElementById('queue-container');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    updateCharCount();
    checkAPIHealth();
});

// Tab switching functionality
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelector(`[onclick="switchTab('${tabName}')"]`).classList.add('active');
    
    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// Initialize event listeners
function initializeEventListeners() {
    // File drag and drop
    dropZone.addEventListener('dragover', handleDragOver);
    dropZone.addEventListener('dragleave', handleDragLeave);
    dropZone.addEventListener('drop', handleDrop);
    dropZone.addEventListener('click', () => fileInput.click());
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Text analysis
    analyzeTextBtn.addEventListener('click', analyzeText);
    clearTextBtn.addEventListener('click', clearText);
    textInput.addEventListener('input', updateCharCount);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
}

// Drag and drop handlers
function handleDragOver(e) {
    e.preventDefault();
    dropZone.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
        processFiles(files);
    }
}

// File selection handler
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
        processFiles(files);
    }
}

// Process selected files
function processFiles(files) {
    // Filter supported files
    const supportedFiles = files.filter(file => isSupportedFile(file));
    
    if (supportedFiles.length === 0) {
        showNotification('No supported files selected. Please select images, videos, audio files, or text files.', 'warning');
        return;
    }
    
    if (supportedFiles.length < files.length) {
        showNotification(`${files.length - supportedFiles.length} unsupported files were skipped.`, 'info');
    }
    
    // Add files to queue and start analysis
    supportedFiles.forEach(file => {
        const fileId = generateFileId();
        const queueItem = {
            id: fileId,
            file: file,
            status: 'queued',
            progress: 0
        };
        
        fileQueue.push(queueItem);
        addToQueue(queueItem);
        analyzeFile(queueItem);
    });
    
    showFileQueue();
}

// Check if file is supported
function isSupportedFile(file) {
    const supportedExtensions = [
        // Images
        'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif',
        // Audio
        'wav', 'mp3', 'flac', 'ogg', 'aac',
        // Video
        'mp4', 'avi', 'mkv', 'mov', 'wmv',
        // Text
        'txt', 'md', 'html', 'xml'
    ];
    
    const extension = file.name.split('.').pop().toLowerCase();
    return supportedExtensions.includes(extension);
}

// Analyze file
async function analyzeFile(queueItem) {
    try {
        updateQueueItem(queueItem.id, 'processing', 10);
        
        const formData = new FormData();
        formData.append('file', queueItem.file);
        
        const response = await fetch(`${API_BASE_URL}/analyze`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        updateQueueItem(queueItem.id, 'completed', 100);
        
        // Add result to display
        addResult(result);
        showResults();
        
    } catch (error) {
        console.error('Analysis error:', error);
        updateQueueItem(queueItem.id, 'failed', 0);
        showNotification(`Failed to analyze ${queueItem.file.name}: ${error.message}`, 'error');
    }
}

// Analyze text
async function analyzeText() {
    const text = textInput.value.trim();
    
    if (!text) {
        showNotification('Please enter some text to analyze.', 'warning');
        return;
    }
    
    showLoading();
    
    try {
        const formData = new FormData();
        formData.append('text_content', text);
        
        const response = await fetch(`${API_BASE_URL}/analyze-text`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        hideLoading();
        
        // Add result to display
        addResult(result);
        showResults();
        
    } catch (error) {
        console.error('Text analysis error:', error);
        hideLoading();
        showNotification(`Failed to analyze text: ${error.message}`, 'error');
    }
}

// Clear text
function clearText() {
    textInput.value = '';
    updateCharCount();
}

// Update character count
function updateCharCount() {
    const count = textInput.value.length;
    charCount.textContent = `${count} characters`;
}

// Add result to display
function addResult(result) {
    currentResults.push(result);
    
    const resultElement = createResultElement(result);
    resultsContainer.appendChild(resultElement);
}

// Create result element
function createResultElement(result) {
    const div = document.createElement('div');
    div.className = 'result-item';
    
    const hasDetections = result.detections && result.detections.length > 0;
    const statusClass = hasDetections ? 'status-suspicious' : 'status-clean';
    const statusText = hasDetections ? 'Suspicious' : 'Clean';
    
    div.innerHTML = `
        <div class="result-header">
            <div class="file-info">
                <div class="file-name">${result.file_name}</div>
                <div class="file-details">
                    Type: ${result.file_type} | Size: ${formatFileSize(result.file_size)} | 
                    Analyzed: ${formatDateTime(result.analysis_time)}
                </div>
            </div>
            <div class="status-badge ${statusClass}">${statusText}</div>
        </div>
        
        ${hasDetections ? createDetectionsHTML(result.detections) : '<p style="color: #28a745; font-weight: 500;"><i class="fas fa-check-circle"></i> No steganography detected</p>'}
    `;
    
    return div;
}

// Create detections HTML
function createDetectionsHTML(detections) {
    let html = '<div class="detections">';
    
    detections.forEach(detection => {
        const confidenceClass = getConfidenceClass(detection.confidence);
        
        html += `
            <div class="detection">
                <div class="detection-method">${detection.method}</div>
                <div class="detection-confidence">Confidence: ${detection.confidence.toFixed(1)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill ${confidenceClass}" style="width: ${detection.confidence}%"></div>
                </div>
                <div class="detection-description">${detection.description}</div>
            </div>
        `;
    });
    
    html += '</div>';
    return html;
}

// Get confidence class
function getConfidenceClass(confidence) {
    if (confidence >= 70) return 'confidence-high';
    if (confidence >= 40) return 'confidence-medium';
    return 'confidence-low';
}

// Queue management functions
function addToQueue(queueItem) {
    const div = document.createElement('div');
    div.className = 'queue-item';
    div.id = `queue-${queueItem.id}`;
    
    div.innerHTML = `
        <i class="fas fa-file"></i>
        <span>${queueItem.file.name}</span>
        <div class="progress">
            <div class="progress-bar" style="width: ${queueItem.progress}%"></div>
        </div>
    `;
    
    queueContainer.appendChild(div);
}

function updateQueueItem(id, status, progress) {
    const queueItem = fileQueue.find(item => item.id === id);
    if (queueItem) {
        queueItem.status = status;
        queueItem.progress = progress;
        
        const element = document.getElementById(`queue-${id}`);
        if (element) {
            const progressBar = element.querySelector('.progress-bar');
            if (progressBar) {
                progressBar.style.width = `${progress}%`;
            }
            
            const icon = element.querySelector('i');
            if (icon) {
                icon.className = getStatusIcon(status);
            }
        }
    }
}

function getStatusIcon(status) {
    switch (status) {
        case 'queued': return 'fas fa-clock';
        case 'processing': return 'fas fa-spinner fa-spin';
        case 'completed': return 'fas fa-check-circle';
        case 'failed': return 'fas fa-exclamation-circle';
        default: return 'fas fa-file';
    }
}

// UI utility functions
function showLoading() {
    loading.classList.remove('hidden');
}

function hideLoading() {
    loading.classList.add('hidden');
}

function showResults() {
    resultsSection.classList.remove('hidden');
}

function showFileQueue() {
    fileQueueElement.classList.remove('hidden');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas ${getNotificationIcon(type)}"></i>
        <span>${message}</span>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'fa-check-circle';
        case 'warning': return 'fa-exclamation-triangle';
        case 'error': return 'fa-exclamation-circle';
        default: return 'fa-info-circle';
    }
}

// Utility functions
function generateFileId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDateTime(isoString) {
    if (!isoString) return 'Unknown';
    
    const date = new Date(isoString);
    return date.toLocaleString();
}

// Keyboard shortcuts
function handleKeyboard(e) {
    // Ctrl+Enter or Cmd+Enter to analyze text
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (document.activeElement === textInput) {
            analyzeText();
        }
    }
    
    // Escape to clear results
    if (e.key === 'Escape') {
        clearResults();
    }
}

// Clear results
function clearResults() {
    currentResults = [];
    resultsContainer.innerHTML = '';
    resultsSection.classList.add('hidden');
    
    fileQueue = [];
    queueContainer.innerHTML = '';
    fileQueueElement.classList.add('hidden');
}

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (!response.ok) {
            throw new Error('API not responding');
        }
        console.log('API is healthy');
    } catch (error) {
        console.error('API health check failed:', error);
        showNotification('Warning: Cannot connect to the analysis server. Please make sure it is running on localhost:8000', 'warning');
    }
}

// Add notification styles dynamically
const notificationStyles = `
.notification {
    position: fixed;
    top: 20px;
    right: 20px;
    padding: 15px 20px;
    border-radius: 8px;
    color: white;
    font-weight: 500;
    z-index: 1000;
    display: flex;
    align-items: center;
    gap: 10px;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    animation: slideIn 0.3s ease;
}

.notification-info {
    background: #4a90e2;
}

.notification-success {
    background: #28a745;
}

.notification-warning {
    background: #ffc107;
    color: #333;
}

.notification-error {
    background: #dc3545;
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

// Add styles to document
const styleSheet = document.createElement('style');
styleSheet.textContent = notificationStyles;
document.head.appendChild(styleSheet);
