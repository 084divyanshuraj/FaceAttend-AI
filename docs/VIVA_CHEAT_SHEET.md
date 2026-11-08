# FaceAttend AI: Viva Voce Rapid Revision Cheat Sheet
## Essential Numbers, Formulas & 30-Second Technical Answers for Examination Defense

Keep this one-page summary handy for fast revision right before your project presentation or oral examination.

---

### 1. Key System Numbers & Calibrated Parameters (Memorize These!)

| Parameter | Exact Value | Mathematical / Technical Justification |
| :--- | :--- | :--- |
| **YuNet Model Size** | **227 KB** (ONNX) | Single-stage convolutional face detector; scale-invariant. |
| **SFace Model Size** | **36.9 MB** (ONNX) | Deep metric embedding model trained on millions of faces. |
| **Embedding Dimension** | **128 float32** | $128 \times 4\text{ bytes} = 512\text{ bytes}$ per face sample. |
| **Cosine Similarity Threshold** | **$\tau_{\text{sim}} \ge 0.637$** | Values $\ge 0.637$ denote the same identity; $< 0.637$ denotes "Unknown". |
| **Cosine Distance Threshold** | **$\tau_{\text{dist}} \le 0.363$** | Distance is $1.0 - \text{Similarity}$. Zero is an identical match. |
| **Laplacian Blur Threshold** | **$\ge 80.0$** | Focus sharpness metric: Modified Laplacian variance $\sigma^2(\nabla^2 I)$. |
| **Maximum Head Roll (Tilt)** | **$\le 20.0^\circ$** | Eye-to-eye angle relative to horizontal: $\arctan\left(\frac{\Delta y}{\Delta x}\right)$. |
| **Yaw Asymmetry Ratio** | **$[0.45, 2.2]$** | Ratio of nose tip to right eye distance vs left eye distance. |
| **Minimum Face Bounding Box** | **$80 \times 80\text{ px}$** | Ensures adequate pixel resolution before embedding extraction. |
| **Enrollment Samples** | **5 samples** | Frontal, slight left, slight right, neutral, smiling. |
| **Temporal Consensus Rule** | **3 of 5 frames** | Must be detected in $\ge 3$ consecutive frames within 3 seconds. |
| **Session Cooldown Window** | **60 seconds** | Prevents redundant database transactions after initial logging. |
| **LFW Benchmark Accuracy** | **99.60%** | SFace accuracy on standard Labeled Faces in the Wild dataset. |
| **Equal Error Rate (EER)** | **$\approx 3.2\%$** | Point where $\text{FAR}(\tau) = \text{FRR}(\tau)$ on calibrated test splits. |

---

### 2. Core Mathematical Formulas

#### Cosine Similarity on Unit-Normalized Vectors
Because all SFace embeddings are $L_2$-normalized to unit length ($\|\mathbf{e}\|_2 = 1.0$), cosine similarity simplifies to a simple vector dot product:
$$\text{Sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \mathbf{u} \cdot \mathbf{v}$$

#### Cosine Distance
$$D_{\text{cos}}(\mathbf{u}, \mathbf{v}) = 1.0 - \text{Sim}(\mathbf{u}, \mathbf{v})$$

#### In-Memory Gallery Vector Matching (Matrix Dot Product)
For $M$ registered student samples in normalized matrix $\mathbf{E} \in \mathbb{R}^{M \times 128}$ and query vector $\mathbf{e}_q \in \mathbb{R}^{128}$:
$$\mathbf{s} = \mathbf{E} \cdot \mathbf{e}_q \in [-1.0, 1.0]^M$$
- Global maximum: $\text{score}_{\max} = \max(\mathbf{s})$.
- If $\text{score}_{\max} \ge 0.637 \implies$ Student Identified.
- If $\text{score}_{\max} < 0.637 \implies$ Unknown Person.

#### Laplacian Blur Variance (Focus Sharpness)
$$\sigma^2(\nabla^2 I) = \frac{1}{N} \sum_{x, y} (L(x, y) - \bar{L})^2$$

---

### 3. Rapid-Fire 30-Second Examiner Answers

#### Q: "Why didn't you train your own model?"
> *"Because training a CNN classifier from scratch introduces an open-world scaling bottleneck: every time a new student registers or transfers, the entire model would have to be retrained. Instead, SFace uses Deep Metric Learning to project faces into a 128-D hyperspherical embedding space. New students are enrolled in milliseconds by simply saving their 128-D vector to SQLite and RAM without retraining."*

#### Q: "Why SFace over ArcFace?"
> *"ArcFace uses a heavy ResNet-50 backbone with 512-dimensional embeddings and requires a dedicated GPU server. SFace achieves near-identical accuracy (99.60% vs 99.83%) while reducing embedding memory by 75% (128-D vs 512-D) and running at 50+ FPS on standard laptop CPUs. SFace is the mathematically optimal choice for cost-effective edge deployment."*

#### Q: "How do you prevent duplicate attendance?"
> *"At the database level, SQLite enforces `UNIQUE(session_id, student_id)`. At the application level, our temporal state machine caches marked student IDs in memory for the active session, skipping redundant database writes while continuing to display the green 'Present' badge on the live video stream."*

#### Q: "How does the in-memory cache scale to 10,000 students?"
> *"Each student has 5 samples of 128 float32 numbers (512 bytes each). For 10,000 students, that's $50,000 \times 512\text{ bytes} \approx 25.6\text{ MB}$ of RAM. Computing the dot product across 50,000 vectors takes under 0.8 milliseconds using modern CPU BLAS libraries, easily sustaining 30 FPS real-time video."*

#### Q: "What database mode did you use and why?"
> *"We enabled SQLite Write-Ahead Logging (`PRAGMA journal_mode = WAL;`) and enforced Foreign Keys (`PRAGMA foreign_keys = ON;`). WAL allows concurrent readers and writers without database locking during live video streaming, and foreign keys guarantee cascading biometric deletion when a student is deleted."*

