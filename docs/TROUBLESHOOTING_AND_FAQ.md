# FaceAttend AI: Troubleshooting, FAQ & Configuration Tuning Guide

This guide covers common questions, edge cases, camera permission fixes, and sensitivity tuning when running **FaceAttend AI** locally on Windows, macOS, or Linux.

---

## 1. Common Local Issues & Quick Fixes

### Issue 1: "Camera Access Error" or Black Screen in Web Browser
- **Cause**: Chrome/Edge blocks camera access if permissions were previously denied, or if another application is locking the webcam.
- **Fix**:
  1. Look at the right or left of your browser address bar (`http://localhost:5000`).
  2. Click the **Camera Icon** or the **Tune Icon (Site Settings)**.
  3. Ensure **Camera** is set to **"Allow"**.
  4. Ensure applications like **Zoom, MS Teams, Skype, Discord, or the Windows Camera app** are closed and not locking the webcam hardware.
  5. Refresh the page (`Ctrl + F5`).

---

### Issue 2: "Port 5000 is already in use" (Address already in use)
- **Cause**: A previous Flask instance or background Python process is still listening on port 5000.
- **Fix (Option A - Close previous process in PowerShell)**:
  ```powershell
  Get-Process python | Stop-Process -Force
  ```
- **Fix (Option B - Change Port in `config.py`)**:
  Open `config.py` and change the port from `5000` to `5001` or `8080`:
  ```python
  FLASK_CONFIG = {
      "HOST": "0.0.0.0",
      "PORT": 5001,  # Change here
      "DEBUG": True,
  }
  ```

---

### Issue 3: "Laplacian Blur Warning" on Low-Quality / Budget Webcams
- **Symptom**: During face enrollment (`/register`), the system warns *"Image is blurry"* even when you are holding still.
- **Cause**: Some built-in 480p laptop cameras have noisy sensors that produce lower high-frequency edge variance.
- **Fix**: Open `config.py` and adjust `blur_threshold` from `80.0` to `50.0` or `40.0`:
  ```python
  QUALITY_CONFIG = {
      "min_face_width": 80,
      "min_face_height": 80,
      "blur_threshold": 50.0,       # Lowered for budget laptop webcams
      "max_pose_angle_deg": 20.0,
      "samples_per_student": 5,
  }
  ```

---

### Issue 4: How to Reset the Database to a Clean Demo State
If you want to clear test students and restore the clean demonstration state:
```powershell
python scripts\seed_demo_data.py
```
This automatically resets the tables and repopulates the 10 demo students and 50 biometric vectors.

---

### Issue 5: How to Use an External USB Webcam or Phone Camera
If you plugged in a secondary USB camera or are using a phone webcam app (like DroidCam / Iriun):
1. **In the Web Browser**: Click the **Camera Icon** in Chrome's URL bar and choose your external webcam from the device dropdown.
2. **In Kiosk Mode (`run_kiosk.py`)**: Pass `--camera 1` (or device index `2`):
   ```powershell
   python scripts\run_kiosk.py --camera 1
   ```

---

## 2. Threshold Calibration & Sensitivity Tuning Guide

All Computer Vision sensitivity parameters are centrally defined in [`config.py`](file:///d:/FaceAttend%20AI/config.py):

| Goal | Parameter in `config.py` | Default Value | Recommended Adjustment |
| :--- | :--- | :--- | :--- |
| **Strict Anti-Impostor Security** | `cosine_similarity_threshold` | `0.637` | Increase to `0.70` or `0.72` (eliminates false accepts, requires clean frontal views). |
| **Lenient Matching in Dim Lighting** | `cosine_similarity_threshold` | `0.637` | Decrease to `0.60` or `0.58` (matches more easily under poor lighting). |
| **Faster Multi-Frame Consensus** | `temporal_consensus_count` | `3` | Reduce to `2` (locks attendance in 2 frames instead of 3). |
| **Longer Consensus Window** | `temporal_window_size` | `5` | Increase to `7` or `9` for noisy or high-motion environments. |
| **Higher Detection Sensitivity** | `score_threshold` | `0.80` | Reduce to `0.70` to detect smaller or further-away faces. |

---

## 3. Frequently Asked Questions (FAQ)

### Q: Does FaceAttend AI require an active internet connection to recognize faces?
> **No.** Once the ONNX models are downloaded to the `models/` folder, the entire Computer Vision engine, database, and web server run **100% offline locally on your laptop** with zero cloud latency.

### Q: Where are student photos and face vectors stored?
> Student biometric embeddings are stored in SQLite (`database/faceattend.db`) inside the `face_embeddings` table as compact **512-byte binary buffers** (`float32` arrays of 128 elements). We store mathematical feature embeddings rather than raw photo files for biometric privacy and high-speed in-memory retrieval.

### Q: Can FaceAttend AI detect multiple people in the camera at the same time?
> **Yes.** In **Live Attendance Mode (`/live`)**, YuNet detects all faces in the camera frame simultaneously, extracts embeddings for each face, matches them individually against the in-memory gallery, and draws distinct color-coded boxes and badges for each person.

