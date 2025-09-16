// Advanced video information fetcher
class VideoManager {
    constructor() {
        this.videos = [];
        this.currentVideo = null;
    }

    async fetchVideoMetadata(filename) {
        // In a real implementation, you could fetch additional metadata
        // For now, return basic info
        return {
            filename: filename,
            displayName: filename.replace(/\.[^/.]+$/, ""),
            extension: filename.split('.').pop().toUpperCase()
        };
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Face detection confidence helper
class FaceDetector {
    constructor() {
        this.minConfidence = 0.7;
        this.detectedFaces = [];
    }

    // Simulate face detection for preview (client-side)
    async detectFacesInFrame(videoElement) {
        // This would normally use a face detection library
        // For now, we'll just check if the video element has content
        return videoElement.readyState >= 2;
    }

    validateFaceQuality(canvas) {
        // Check if the image has good lighting and clarity
        const ctx = canvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;

        let brightness = 0;
        for (let i = 0; i < data.length; i += 4) {
            brightness += (data[i] + data[i + 1] + data[i + 2]) / 3;
        }
        brightness = brightness / (data.length / 4);

        // Check if brightness is in acceptable range (not too dark or bright)
        const quality = {
            isGood: brightness > 50 && brightness < 200,
            brightness: brightness,
            message: brightness < 50 ? "Too dark" : brightness > 200 ? "Too bright" : "Good lighting"
        };

        return quality;
    }
}

// Progress tracker with ETA calculation
class ProgressTracker {
    constructor() {
        this.startTime = null;
        this.currentProgress = 0;
        this.history = [];
    }

    start() {
        this.startTime = Date.now();
        this.currentProgress = 0;
        this.history = [];
    }

    update(progress) {
        this.currentProgress = progress;
        this.history.push({
            time: Date.now(),
            progress: progress
        });

        // Keep only last 10 updates for ETA calculation
        if (this.history.length > 10) {
            this.history.shift();
        }
    }

    getETA() {
        if (this.history.length < 2) return null;

        const recent = this.history.slice(-5);
        const timeElapsed = recent[recent.length - 1].time - recent[0].time;
        const progressMade = recent[recent.length - 1].progress - recent[0].progress;

        if (progressMade <= 0) return null;

        const rate = progressMade / timeElapsed;
        const remaining = 100 - this.currentProgress;
        const etaMs = remaining / rate;

        return this.formatTime(etaMs / 1000);
    }

    formatTime(seconds) {
        if (seconds < 60) return `${Math.round(seconds)}s`;
        if (seconds < 3600) return `${Math.round(seconds / 60)}m ${Math.round(seconds % 60)}s`;
        return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`;
    }

    getElapsedTime() {
        if (!this.startTime) return '0s';
        const elapsed = (Date.now() - this.startTime) / 1000;
        return this.formatTime(elapsed);
    }
}

// Notification manager
class NotificationManager {
    constructor() {
        this.permission = null;
        this.checkPermission();
    }

    async checkPermission() {
        if ("Notification" in window) {
            if (Notification.permission === "granted") {
                this.permission = true;
            } else if (Notification.permission !== "denied") {
                const permission = await Notification.requestPermission();
                this.permission = permission === "granted";
            }
        }
    }

    notify(title, body, icon = '/static/icon.png') {
        if (this.permission) {
            new Notification(title, {
                body: body,
                icon: icon,
                badge: icon,
                vibrate: [200, 100, 200]
            });
        }
    }

    notifyCompletion() {
        this.notify(
            "🎉 Deepfake Complete!",
            "Your video has been processed successfully. Click to download.",
            '/static/success.png'
        );
    }

    notifyError(error) {
        this.notify(
            "❌ Processing Error",
            error || "An error occurred during processing.",
            '/static/error.png'
        );
    }
}

// Keyboard shortcuts
class KeyboardShortcuts {
    constructor() {
        this.shortcuts = new Map();
        this.setupDefaultShortcuts();
        this.listen();
    }

    setupDefaultShortcuts() {
        // Ctrl/Cmd + R: Start recording
        this.shortcuts.set('r', () => {
            const btn = document.getElementById('startRecording');
            if (btn && !btn.disabled) btn.click();
        });

        // Ctrl/Cmd + G: Generate deepfake
        this.shortcuts.set('g', () => {
            const btn = document.getElementById('generateDeepfake');
            if (btn && !btn.disabled) btn.click();
        });

        // Ctrl/Cmd + C: Clear faces
        this.shortcuts.set('c', () => {
            const btn = document.getElementById('clearFaces');
            if (btn) btn.click();
        });

        // ESC: Stop recording
        this.shortcuts.set('Escape', () => {
            if (window.isRecording) {
                window.stopRecording();
            }
        });
    }

    listen() {
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                const handler = this.shortcuts.get(e.key);
                if (handler) {
                    e.preventDefault();
                    handler();
                }
            } else if (e.key === 'Escape') {
                const handler = this.shortcuts.get('Escape');
                if (handler) handler();
            }
        });
    }
}

// Quality presets
const QUALITY_PRESETS = {
    draft: {
        crf: 25,
        preset: 'ultrafast',
        threads: 2,
        name: 'Draft (Fast)',
        description: 'Quick preview, lower quality'
    },
    balanced: {
        crf: 18,
        preset: 'medium',
        threads: 4,
        name: 'Balanced',
        description: 'Good quality and speed'
    },
    high: {
        crf: 14,
        preset: 'slow',
        threads: 4,
        name: 'High Quality',
        description: 'Better quality, slower processing'
    },
    maximum: {
        crf: 10,
        preset: 'veryslow',
        threads: 8,
        name: 'Maximum Quality',
        description: 'Best quality, very slow'
    }
};

// Apply quality preset
function applyQualityPreset(presetName) {
    const preset = QUALITY_PRESETS[presetName];
    if (preset) {
        document.getElementById('qualitySlider').value = preset.crf;
        document.getElementById('qualityValue').textContent = preset.crf;
        document.getElementById('threadsSlider').value = preset.threads;
        document.getElementById('threadsValue').textContent = preset.threads;
        document.getElementById('presetSelect').value = preset.preset;
    }
}

// Export utilities for use in main script
window.RopeUtils = {
    VideoManager,
    FaceDetector,
    ProgressTracker,
    NotificationManager,
    KeyboardShortcuts,
    QUALITY_PRESETS,
    applyQualityPreset
};

// Initialize utilities when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Initialize managers
    window.videoManager = new VideoManager();
    window.faceDetector = new FaceDetector();
    window.progressTracker = new ProgressTracker();
    window.notificationManager = new NotificationManager();
    window.keyboardShortcuts = new KeyboardShortcuts();

    // Add quality preset buttons if container exists
    const qualityContainer = document.querySelector('.quality-settings');
    if (qualityContainer) {
        const presetButtons = document.createElement('div');
        presetButtons.className = 'preset-buttons';
        presetButtons.style.cssText = 'display: flex; gap: 5px; margin-bottom: 10px;';

        Object.entries(QUALITY_PRESETS).forEach(([key, preset]) => {
            const btn = document.createElement('button');
            btn.textContent = preset.name;
            btn.title = preset.description;
            btn.className = 'btn btn-sm';
            btn.style.cssText = 'padding: 5px 10px; font-size: 12px; flex: 1;';
            btn.onclick = () => applyQualityPreset(key);
            presetButtons.appendChild(btn);
        });

        qualityContainer.insertBefore(presetButtons, qualityContainer.firstChild);
    }

    console.log('🚀 Rope Deepfake UI utilities loaded');
});