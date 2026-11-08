/**
 * FaceAttend AI - Camera Streamer & Frame Capture Manager
 * Handles browser webcam acquisition, playback, and high-performance base64 frame capture.
 */

class CameraStreamer {
    constructor(videoElementId, options = {}) {
        this.video = document.getElementById(videoElementId);
        this.width = options.width || 640;
        this.height = options.height || 480;
        this.stream = null;
        this.isStreaming = false;

        // Hidden capture canvas for frame extraction
        this.captureCanvas = document.createElement('canvas');
        this.captureCanvas.width = this.width;
        this.captureCanvas.height = this.height;
        this.captureCtx = this.captureCanvas.getContext('2d');
    }

    async start() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Webcam API is not supported in this browser or insecure context (requires HTTPS or localhost).');
        }

        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: this.width },
                    height: { ideal: this.height },
                    facingMode: 'user'
                },
                audio: false
            });

            this.video.srcObject = this.stream;
            await this.video.play();
            this.isStreaming = true;
            return true;
        } catch (err) {
            console.error('Camera Access Error:', err);
            throw err;
        }
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        if (this.video) {
            this.video.srcObject = null;
        }
        this.isStreaming = false;
    }

    captureFrameBase64(quality = 0.85) {
        if (!this.isStreaming || this.video.readyState < 2) {
            return null;
        }
        this.captureCtx.drawImage(this.video, 0, 0, this.width, this.height);
        return this.captureCanvas.toDataURL('image/jpeg', quality);
    }
}

