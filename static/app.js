class VideoManager {
    constructor() {
        this.videos = [];
        this.currentVideo = null;
        this.touchStartY = 0;
        this.touchEndY = 0;
        this.initTouchHandlers();
    }

    async fetchVideoMetadata(filename) {
        // Enhanced metadata for mobile display
        return {
            filename: filename,
            displayName: filename.replace(/\.[^/.]+$/, ""),
            extension: filename.split('.').pop().toUpperCase(),
            isVertical: this.isVerticalVideo(filename)
        };
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    formatDuration(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    }

    isVerticalVideo(filename) {
        // Simple heuristic for vertical video detection
        const verticalKeywords = ['vertical', 'portrait', 'tiktok', 'reel', 'story'];
        return verticalKeywords.some(keyword =>
            filename.toLowerCase().includes(keyword)
        );
    }

    initTouchHandlers() {
        // Add swipe gesture support for video navigation
        document.addEventListener('touchstart', (e) => {
            this.touchStartY = e.touches[0].clientY;
        }, { passive: true });

        document.addEventListener('touchend', (e) => {
            this.touchEndY = e.changedTouches[0].clientY;
            this.handleSwipeGesture();
        }, { passive: true });
    }

    handleSwipeGesture() {
        const swipeThreshold = 50;
        const swipeDistance = this.touchStartY - this.touchEndY;

        if (Math.abs(swipeDistance) > swipeThreshold) {
            // Vertical swipe detected
            if (swipeDistance > 0) {
                // Swiped up - could trigger specific actions
                this.onSwipeUp();
            } else {
                // Swiped down - could trigger specific actions
                this.onSwipeDown();
            }
        }
    }

    onSwipeUp() {
        // Custom implementation for swipe up
        console.log('Swipe up detected');
    }

    onSwipeDown() {
        // Custom implementation for swipe down
        console.log('Swipe down detected');
    }
}

// Face detection with enhanced mobile performance
class FaceDetector {
    constructor() {
        this.minConfidence = 0.7;
        this.detectedFaces = [];
        this.frameBuffer = [];
        this.maxFrameBuffer = 10;
        this.isProcessing = false;
    }

    async detectFacesInFrame(videoElement) {
        if (this.isProcessing) return false;

        this.isProcessing = true;

        try {
            // Enhanced face detection for mobile devices
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');

            canvas.width = Math.min(videoElement.videoWidth || 640, 640);
            canvas.height = Math.min(videoElement.videoHeight || 480, 480);

            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

            // Add frame to buffer for quality analysis
            this.frameBuffer.push(canvas.toDataURL('image/jpeg', 0.8));
            if (this.frameBuffer.length > this.maxFrameBuffer) {
                this.frameBuffer.shift();
            }

            return videoElement.readyState >= 2;
        } finally {
            this.isProcessing = false;
        }
    }

    validateFaceQuality(canvas) {
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        let brightness = 0;
        let contrast = 0;
        const pixels = data.length / 4;

        // Calculate brightness and basic contrast
        for (let i = 0; i < data.length; i += 4) {
            const pixel = (data[i] + data[i + 1] + data[i + 2]) / 3;
            brightness += pixel;
        }
        brightness = brightness / pixels;

        // Calculate contrast (simplified)
        for (let i = 0; i < data.length; i += 4) {
            const pixel = (data[i] + data[i + 1] + data[i + 2]) / 3;
            contrast += Math.abs(pixel - brightness);
        }
        contrast = contrast / pixels;

        const quality = {
            isGood: brightness > 50 && brightness < 200 && contrast > 10,
            brightness: Math.round(brightness),
            contrast: Math.round(contrast),
            message: this.getQualityMessage(brightness, contrast)
        };

        return quality;
    }

    getQualityMessage(brightness, contrast) {
        if (brightness < 50) return "Too dark - need better lighting";
        if (brightness > 200) return "Too bright - reduce lighting";
        if (contrast < 10) return "Low contrast - move closer or adjust lighting";
        return "Good quality";
    }

    // Mobile-optimized frame capture
    async captureOptimizedFrame(videoElement) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');

        // Optimize canvas size for mobile performance
        const maxWidth = 800;
        const maxHeight = 600;

        let { videoWidth, videoHeight } = videoElement;

        if (videoWidth > maxWidth || videoHeight > maxHeight) {
            const aspectRatio = videoWidth / videoHeight;
            if (videoWidth > videoHeight) {
                videoWidth = maxWidth;
                videoHeight = maxWidth / aspectRatio;
            } else {
                videoHeight = maxHeight;
                videoWidth = maxHeight * aspectRatio;
            }
        }

        canvas.width = videoWidth;
        canvas.height = videoHeight;

        ctx.drawImage(videoElement, 0, 0, videoWidth, videoHeight);

        return canvas.toDataURL('image/jpeg', 0.85);
    }
}

// Enhanced progress tracking with visual feedback
class ProgressTracker {
    constructor() {
        this.currentProgress = 0;
        this.stage = '';
        this.startTime = null;
        this.estimatedTimeRemaining = null;
        this.stages = [
            'Initializing',
            'Processing faces',
            'Analyzing video',
            'Generating frames',
            'Encoding video',
            'Finalizing'
        ];
        this.currentStageIndex = 0;
    }

    start() {
        this.startTime = Date.now();
        this.currentProgress = 0;
        this.currentStageIndex = 0;
        this.updateDisplay();
    }

    updateProgress(percent, stage = null) {
        this.currentProgress = Math.min(100, Math.max(0, percent));

        if (stage) {
            this.stage = stage;
            this.currentStageIndex = this.stages.indexOf(stage);
        }

        this.calculateETA();
        this.updateDisplay();
        this.addProgressAnimation();
    }

    calculateETA() {
        if (!this.startTime || this.currentProgress === 0) return;

        const elapsed = Date.now() - this.startTime;
        const rate = this.currentProgress / elapsed;
        const remaining = (100 - this.currentProgress) / rate;

        this.estimatedTimeRemaining = Math.round(remaining / 1000);
    }

    updateDisplay() {
        const progressBar = document.getElementById('progressFill');
        const statusMessage = document.getElementById('statusMessage');

        if (progressBar) {
            progressBar.style.width = `${this.currentProgress}%`;

            let displayText = `${Math.round(this.currentProgress)}%`;
            if (this.stage) {
                displayText += ` - ${this.stage}`;
            }
            if (this.estimatedTimeRemaining && this.estimatedTimeRemaining > 10) {
                displayText += ` (${this.formatTime(this.estimatedTimeRemaining)} remaining)`;
            }

            progressBar.textContent = displayText;
        }

        // Update stage indicator if available
        this.updateStageIndicator();
    }

    updateStageIndicator() {
        // Could add visual stage indicators for better UX
        const stageElements = document.querySelectorAll('.stage-indicator');
        stageElements.forEach((el, index) => {
            if (index <= this.currentStageIndex) {
                el.classList.add('completed');
            } else {
                el.classList.remove('completed');
            }
        });
    }

    addProgressAnimation() {
        const progressBar = document.getElementById('progressFill');
        if (progressBar) {
            progressBar.style.transition = 'width 0.5s ease-out';

            // Add pulse effect for active processing
            if (this.currentProgress > 0 && this.currentProgress < 100) {
                progressBar.style.animation = 'pulse 2s infinite';
            } else {
                progressBar.style.animation = 'none';
            }
        }
    }

    formatTime(seconds) {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = seconds % 60;

        if (minutes > 0) {
            return `${minutes}m ${remainingSeconds}s`;
        }
        return `${remainingSeconds}s`;
    }

    complete() {
        this.currentProgress = 100;
        this.stage = 'Complete';
        this.updateDisplay();

        const progressBar = document.getElementById('progressFill');
        if (progressBar) {
            progressBar.style.animation = 'none';
            setTimeout(() => {
                progressBar.style.background = 'linear-gradient(90deg, #11998e, #38ef7d)';
            }, 500);
        }
    }
}

// Enhanced notification system for mobile
class NotificationManager {
    constructor() {
        this.notifications = [];
        this.maxNotifications = 3;
        this.defaultDuration = 5000;
        this.container = this.createContainer();
    }

    createContainer() {
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            container.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                z-index: 10000;
                pointer-events: none;
            `;
            document.body.appendChild(container);
        }
        return container;
    }

    notify(title, message, icon = null, duration = this.defaultDuration) {
        // Use system notifications if available and not in fullscreen
        if ('Notification' in window && Notification.permission === 'granted' && !document.fullscreenElement) {
            new Notification(title, {
                body: message,
                icon: icon || '/static/favicon.ico',
                tag: 'rope-deepfake'
            });
        }

        // Also show in-app notification
        this.showInAppNotification(title, message, icon, duration);
    }

    showInAppNotification(title, message, icon, duration) {
        const notification = document.createElement('div');
        notification.className = 'mobile-notification';
        notification.style.cssText = `
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 15px 20px;
            border-radius: 12px;
            margin-bottom: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            pointer-events: auto;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            max-width: 300px;
            word-wrap: break-word;
        `;

        notification.innerHTML = `
            <div style="font-weight: 600; margin-bottom: 5px;">${title}</div>
            <div style="font-size: 0.9rem; opacity: 0.9;">${message}</div>
        `;

        this.container.appendChild(notification);

        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);

        // Auto remove
        setTimeout(() => {
            this.removeNotification(notification);
        }, duration);

        // Limit number of notifications
        this.notifications.push(notification);
        if (this.notifications.length > this.maxNotifications) {
            this.removeNotification(this.notifications[0]);
        }

        // Touch to dismiss
        notification.addEventListener('click', () => {
            this.removeNotification(notification);
        });
    }

    removeNotification(notification) {
        const index = this.notifications.indexOf(notification);
        if (index > -1) {
            this.notifications.splice(index, 1);
        }

        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }

    requestPermission() {
        if ('Notification' in window && Notification.permission === 'default') {
            Notification.requestPermission();
        }
    }

    notifySuccess(message) {
        this.notify(
            "✅ Success",
            message || "Operation completed successfully!",
            '/static/success.png'
        );
    }

    notifyError(error) {
        this.notify(
            "❌ Error",
            error || "An error occurred during processing.",
            '/static/error.png',
            8000 // Longer duration for errors
        );
    }

    notifyProcessing(message) {
        this.notify(
            "⚙️ Processing",
            message || "Processing your request...",
            '/static/processing.png',
            3000
        );
    }
}

// Enhanced keyboard shortcuts with touch fallbacks
class KeyboardShortcuts {
    constructor() {
        this.shortcuts = new Map();
        this.touchShortcuts = new Map();
        this.setupDefaultShortcuts();
        this.setupTouchShortcuts();
        this.listen();
    }

    setupDefaultShortcuts() {
        // Space: Start/stop recording
        this.shortcuts.set(' ', () => {
            const btn = document.getElementById('startRecording');
            if (btn && !btn.disabled) btn.click();
        });

        // Enter: Generate deepfake
        this.shortcuts.set('Enter', () => {
            const btn = document.getElementById('generateDeepfake');
            if (btn && !btn.disabled) btn.click();
        });

        // Backspace: Clear faces
        this.shortcuts.set('Backspace', () => {
            const btn = document.getElementById('clearFaces');
            if (btn) btn.click();
        });

        // Arrow keys: Navigate views
        this.shortcuts.set('ArrowLeft', () => {
            const btn = document.getElementById('prevBtn');
            if (btn && !btn.disabled) btn.click();
        });

        this.shortcuts.set('ArrowRight', () => {
            const btn = document.getElementById('nextBtn');
            if (btn && !btn.disabled) btn.click();
        });

        // ESC: Emergency stop
        this.shortcuts.set('Escape', () => {
            if (window.isRecording) {
                window.stopRecording();
            }
        });
    }

    setupTouchShortcuts() {
        // Long press actions for touch devices
        let longPressTimer;
        let isLongPress = false;

        document.addEventListener('touchstart', (e) => {
            longPressTimer = setTimeout(() => {
                isLongPress = true;
                this.handleLongPress(e);
            }, 800);
        });

        document.addEventListener('touchend', (e) => {
            clearTimeout(longPressTimer);
            if (isLongPress) {
                e.preventDefault();
                isLongPress = false;
            }
        });

        document.addEventListener('touchmove', () => {
            clearTimeout(longPressTimer);
            isLongPress = false;
        });
    }

    handleLongPress(event) {
        const target = event.target.closest('button');
        if (!target) return;

        // Long press actions for different buttons
        switch (target.id) {
            case 'startRecording':
                // Could trigger advanced recording options
                break;
            case 'generateDeepfake':
                // Could show quality options quickly
                break;
            case 'clearFaces':
                // Confirm action with haptic feedback
                if ('vibrate' in navigator) {
                    navigator.vibrate(100);
                }
                break;
        }
    }

    listen() {
        document.addEventListener('keydown', (e) => {
            // Ignore if typing in input fields
            if (e.target.tagName.toLowerCase() === 'input' ||
                e.target.tagName.toLowerCase() === 'textarea') {
                return;
            }

            const handler = this.shortcuts.get(e.key);
            if (handler) {
                e.preventDefault();
                handler();
            }
        });
    }

    // Add visual feedback for touch interactions
    addTouchFeedback(element) {
        element.addEventListener('touchstart', () => {
            element.style.transform = 'scale(0.95)';
            element.style.transition = 'transform 0.1s ease';
        });

        element.addEventListener('touchend', () => {
            element.style.transform = 'scale(1)';
        });
    }
}

// Quality presets optimized for mobile processing
const QUALITY_PRESETS = {
    mobile: {
        crf: 28,
        preset: 'veryfast',
        threads: 2,
        name: 'Mobile Optimized',
        description: 'Fast processing for mobile devices'
    },
    balanced: {
        crf: 23,
        preset: 'medium',
        threads: 4,
        name: 'Balanced',
        description: 'Good quality and reasonable speed'
    },
    quality: {
        crf: 18,
        preset: 'slow',
        threads: 4,
        name: 'High Quality',
        description: 'Better quality, slower processing'
    },
    maximum: {
        crf: 14,
        preset: 'veryslow',
        threads: 6,
        name: 'Maximum Quality',
        description: 'Best quality, very slow (not recommended for mobile)'
    }
};

// Apply quality preset with mobile detection
function applyQualityPreset(presetName) {
    const preset = QUALITY_PRESETS[presetName];
    if (!preset) return;

    // Auto-adjust for mobile devices
    const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
    if (isMobile && presetName === 'maximum') {
        // Fallback to quality preset for mobile
        applyQualityPreset('quality');
        return;
    }

    document.getElementById('qualitySlider').value = preset.crf;
    document.getElementById('qualityValue').textContent = preset.crf;
    document.getElementById('threadsSlider').value = preset.threads;
    document.getElementById('threadsValue').textContent = preset.threads;
    document.getElementById('presetSelect').value = preset.preset;
}

// Device capabilities detection
class DeviceCapabilities {
    constructor() {
        this.capabilities = this.detectCapabilities();
    }

    detectCapabilities() {
        return {
            isMobile: /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent),
            hasTouch: 'ontouchstart' in window,
            hasWebcam: navigator.mediaDevices && navigator.mediaDevices.getUserMedia,
            hasNotifications: 'Notification' in window,
            hasVibration: 'vibrate' in navigator,
            maxThreads: navigator.hardwareConcurrency || 2,
            screenSize: {
                width: window.screen.width,
                height: window.screen.height
            },
            isPortrait: window.screen.height > window.screen.width
        };
    }

    optimizeForDevice() {
        if (this.capabilities.isMobile) {
            // Apply mobile optimizations
            this.setMobileQualityDefaults();
            this.enableTouchOptimizations();
        }

        if (this.capabilities.isPortrait) {
            // Optimize for portrait orientation
            this.enablePortraitMode();
        }
    }

    setMobileQualityDefaults() {
        // Set conservative defaults for mobile
        document.getElementById('threadsSlider').value = Math.min(this.capabilities.maxThreads, 2);
        document.getElementById('threadsValue').textContent = Math.min(this.capabilities.maxThreads, 2);
        applyQualityPreset('mobile');
    }

    enableTouchOptimizations() {
        // Add touch-friendly interactions
        document.querySelectorAll('.btn').forEach(btn => {
            window.keyboardShortcuts.addTouchFeedback(btn);
        });

        // Enable pull-to-refresh style interactions
        this.enablePullToRefresh();
    }

    enablePortraitMode() {
        // Add portrait-specific styles
        document.body.classList.add('portrait-mode');
    }

    enablePullToRefresh() {
        let startY = 0;
        let pullDistance = 0;
        const threshold = 100;

        document.addEventListener('touchstart', (e) => {
            startY = e.touches[0].clientY;
        });

        document.addEventListener('touchmove', (e) => {
            const currentY = e.touches[0].clientY;
            pullDistance = currentY - startY;

            if (pullDistance > 0 && window.scrollY === 0) {
                // User is pulling down from top
                if (pullDistance > threshold) {
                    // Show refresh indicator
                    this.showRefreshIndicator();
                }
            }
        });

        document.addEventListener('touchend', () => {
            if (pullDistance > threshold) {
                // Trigger refresh action
                this.refreshVideoList();
            }
            this.hideRefreshIndicator();
            pullDistance = 0;
        });
    }

    showRefreshIndicator() {
        // Visual feedback for pull-to-refresh
        let indicator = document.getElementById('refresh-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'refresh-indicator';
            indicator.style.cssText = `
                position: fixed;
                top: 20px;
                left: 50%;
                transform: translateX(-50%);
                background: rgba(255, 255, 255, 0.9);
                padding: 10px 20px;
                border-radius: 20px;
                z-index: 1000;
            `;
            indicator.textContent = '🔄 Release to refresh';
            document.body.appendChild(indicator);
        }
        indicator.style.display = 'block';
    }

    hideRefreshIndicator() {
        const indicator = document.getElementById('refresh-indicator');
        if (indicator) {
            indicator.style.display = 'none';
        }
    }

    refreshVideoList() {
        // Refresh the video list
        if (typeof loadVideos === 'function') {
            loadVideos();
        }
    }
}

// Export utilities for use in main script
window.RopeUtils = {
    VideoManager,
    FaceDetector,
    ProgressTracker,
    NotificationManager,
    KeyboardShortcuts,
    DeviceCapabilities,
    QUALITY_PRESETS,
    applyQualityPreset
};

// Initialize utilities when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Initializing Rope Deepfake Mobile UI...');

    // Initialize managers
    window.videoManager = new VideoManager();
    window.faceDetector = new FaceDetector();
    window.progressTracker = new ProgressTracker();
    window.notificationManager = new NotificationManager();
    window.keyboardShortcuts = new KeyboardShortcuts();
    window.deviceCapabilities = new DeviceCapabilities();

    // Request notification permission
    window.notificationManager.requestPermission();

    // Optimize for current device
    window.deviceCapabilities.optimizeForDevice();

    // Add quality preset buttons if container exists
    const qualityContainer = document.querySelector('.quality-settings');
    if (qualityContainer) {
        const presetButtons = document.createElement('div');
        presetButtons.className = 'preset-buttons';
        presetButtons.style.cssText = `
            display: grid; 
            grid-template-columns: repeat(2, 1fr); 
            gap: 10px; 
            margin-bottom: 15px;
        `;

        Object.entries(QUALITY_PRESETS).forEach(([key, preset]) => {
            const btn = document.createElement('button');
            btn.textContent = preset.name;
            btn.title = preset.description;
            btn.className = 'btn';
            btn.style.cssText = `
                padding: 10px; 
                font-size: 0.9rem; 
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                color: white;
                border-radius: 8px;
                transition: all 0.3s ease;
            `;

            btn.onclick = () => {
                applyQualityPreset(key);
                // Visual feedback
                document.querySelectorAll('.preset-buttons button').forEach(b => {
                    b.style.background = 'rgba(255, 255, 255, 0.1)';
                });
                btn.style.background = 'rgba(102, 126, 234, 0.3)';
            };

            // Add touch feedback
            window.keyboardShortcuts.addTouchFeedback(btn);

            presetButtons.appendChild(btn);
        });

        qualityContainer.insertBefore(presetButtons, qualityContainer.firstChild);
    }

    // Add haptic feedback for supported devices
    if ('vibrate' in navigator) {
        document.querySelectorAll('.btn').forEach(btn => {
            btn.addEventListener('click', () => {
                navigator.vibrate(50); // Short vibration
            });
        });
    }

    // Handle orientation changes
    window.addEventListener('orientationchange', () => {
        setTimeout(() => {
            window.deviceCapabilities.optimizeForDevice();
        }, 500);
    });

    // Add service worker for offline capabilities (if needed)
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/sw.js').catch(() => {
            // Service worker not available, continue without offline support
        });
    }

    console.log('✅ Rope Deepfake Mobile UI initialized successfully');

    // Log device capabilities for debugging
    console.log('📱 Device capabilities:', window.deviceCapabilities.capabilities);
});