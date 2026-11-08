/**
 * FaceAttend AI - Live Attendance Client Engine
 * Continuously streams video frames to the Flask CV backend, renders bounding boxes
 * and biometric badges onto the canvas, and manages live attendance toasts.
 */

class LiveAttendanceManager {
    constructor(config) {
        this.videoElementId = config.videoElementId || 'liveVideo';
        this.canvasElementId = config.canvasElementId || 'overlayCanvas';
        this.feedTableBodyId = config.feedTableBodyId || 'liveFeedBody';
        this.activeSessionId = config.sessionId || null;
        this.fps = config.fps || 10;

        this.camera = new CameraStreamer(this.videoElementId, { width: 640, height: 480 });
        this.canvas = document.getElementById(this.canvasElementId);
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;

        this.isRunning = false;
        this.isProcessing = false;
        this.intervalId = null;

        // Web Audio API Context for Chime Notifications
        this.audioCtx = null;
    }

    async init() {
        if (this.canvas) {
            this.canvas.width = 640;
            this.canvas.height = 480;
        }

        try {
            await this.camera.start();
            this.startLoop();
            return true;
        } catch (err) {
            alert('Could not start camera: ' + err.message);
            return false;
        }
    }

    stop() {
        this.isRunning = false;
        if (this.intervalId) {
            clearInterval(this.intervalId);
            this.intervalId = null;
        }
        this.camera.stop();
        if (this.ctx && this.canvas) {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        }
    }

    startLoop() {
        this.isRunning = true;
        const intervalMs = Math.round(1000 / this.fps);

        this.intervalId = setInterval(async () => {
            if (!this.isRunning || this.isProcessing) return;

            const frameBase64 = this.camera.captureFrameBase64(0.85);
            if (!frameBase64) return;

            this.isProcessing = true;
            try {
                const response = await fetch('/api/attendance/process-frame', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        image: frameBase64,
                        session_id: this.activeSessionId
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    if (result.success) {
                        this.renderOverlays(result.faces || []);
                        this.handleNewEvents(result.new_attendance_events || []);
                        this.updateStats(result.session_stats || {});
                    }
                }
            } catch (err) {
                console.error('Frame processing error:', err);
            } finally {
                this.isProcessing = false;
            }
        }, intervalMs);
    }

    renderOverlays(faces) {
        if (!this.ctx || !this.canvas) return;
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        faces.forEach(face => {
            const [x, y, w, h] = face.bbox;
            const color = face.badge_color || '#EF4444';

            // 1. Draw Bounding Box
            this.ctx.lineWidth = 3;
            this.ctx.strokeStyle = color;
            this.ctx.strokeRect(x, y, w, h);

            // 2. Draw 5 Facial Landmarks (small dots)
            if (face.landmarks && face.landmarks.length === 5) {
                this.ctx.fillStyle = '#38BDF8';
                face.landmarks.forEach(([lx, ly]) => {
                    this.ctx.beginPath();
                    this.ctx.arc(lx, ly, 3, 0, 2 * Math.PI);
                    this.ctx.fill();
                });
            }

            // 3. Draw Top Badge with Label
            const text = face.confidence > 0 
                ? `${face.label} (${Math.round(face.confidence * 100)}%)`
                : face.label;

            this.ctx.font = '600 13px Inter, sans-serif';
            const textMetrics = this.ctx.measureText(text);
            const badgeW = textMetrics.width + 16;
            const badgeH = 24;
            const badgeY = Math.max(0, y - badgeH - 6);

            // Badge Background
            this.ctx.fillStyle = color;
            this.ctx.beginPath();
            this.ctx.roundRect(x, badgeY, badgeW, badgeH, 6);
            this.ctx.fill();

            // Text
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.fillText(text, x + 8, badgeY + 16);
        });
    }

    handleNewEvents(events) {
        if (!events || events.length === 0) return;

        events.forEach(evt => {
            this.playChime();
            this.showToast(`Present: ${evt.name} (${evt.roll_number})`, 'success');

            // Prepend to live attendance table
            const tbody = document.getElementById(this.feedTableBodyId);
            if (tbody) {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="font-mono">${evt.roll_number}</td>
                    <td class="font-medium">${evt.name}</td>
                    <td>${evt.department}</td>
                    <td><span class="badge badge-success">Present</span></td>
                    <td class="text-subtle">${evt.timestamp.split(' ')[1]}</td>
                `;
                tbody.insertBefore(tr, tbody.firstChild);

                // Keep only top 15 rows
                if (tbody.children.length > 15) {
                    tbody.removeChild(tbody.lastChild);
                }
            }
        });
    }

    updateStats(stats) {
        const elPresent = document.getElementById('statPresent');
        const elTotal = document.getElementById('statTotal');
        const elPercent = document.getElementById('statPercent');

        if (elPresent && stats.present_students !== undefined) elPresent.textContent = stats.present_students;
        if (elTotal && stats.total_students !== undefined) elTotal.textContent = stats.total_students;
        if (elPercent && stats.attendance_percentage !== undefined) elPercent.textContent = `${stats.attendance_percentage}%`;
    }

    playChime() {
        try {
            if (!this.audioCtx) {
                this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            }
            if (this.audioCtx.state === 'suspended') {
                this.audioCtx.resume();
            }
            const osc = this.audioCtx.createOscillator();
            const gain = this.audioCtx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(880, this.audioCtx.currentTime); // A5
            osc.frequency.exponentialRampToValueAtTime(1760, this.audioCtx.currentTime + 0.15); // A6
            gain.gain.setValueAtTime(0.15, this.audioCtx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, this.audioCtx.currentTime + 0.25);
            osc.connect(gain);
            gain.connect(this.audioCtx.destination);
            osc.start();
            osc.stop(this.audioCtx.currentTime + 0.25);
        } catch (e) {
            // Audio policy blocked without interaction
        }
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.style.cssText = `
            background: #1E293B;
            border-left: 4px solid ${type === 'success' ? '#10B981' : '#6366F1'};
            color: #F8FAFC;
            padding: 0.75rem 1.25rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
            font-size: 0.875rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 0.75rem;
            animation: slideIn 0.3s ease;
        `;
        toast.innerHTML = `<i class="fa-solid fa-circle-check" style="color: #10B981;"></i> <span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(-10px)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }
}

