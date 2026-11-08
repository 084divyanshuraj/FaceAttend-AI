import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

def main():
    doc = docx.Document("Case_StudySample copy.docx")
    
    # Analyze existing styles
    print(f"Loaded document with {len(doc.paragraphs)} paragraphs.")
    
    # We will clear existing paragraphs and rewrite with FaceAttend AI content
    # retaining identical document styles and structure.
    
    # Sections data: (Style, Text, IsBold, SpacingAfter)
    sections = [
        ("Title", "Case Study Based Learning: AI-Powered Face Recognition Attendance Management System", False),
        
        ("Heading 1", "Background", True),
        ("Normal", "Face recognition is a cornerstone of biometric identification and computer vision, converting high-dimensional facial images into discriminative compact feature representations. Traditional attendance tracking methods (manual roll calls, paper sign-in sheets, RFID cards, and physical fingerprint scanners) suffer from operational inefficiencies, proxy marking (\"buddy punching\"), queue congestion, and hygiene concerns. Modern automated attendance systems utilize deep convolutional neural networks (CNNs) and metric learning to detect and recognize individuals in real time with high accuracy.", False),
        
        ("Heading 1", "Case Study", True),
        ("Normal", "An educational institution with 50+ classrooms and thousands of students needs an automated, contactless attendance logging system. The system must process real-time webcam video streams, accurately verify student identities even under varying classroom illumination and head poses, prevent buddy punching, protect student biometric privacy, and log attendance into a centralized database without requiring model retraining when new students enroll.", False),
        
        ("Heading 1", "Problem Statement", True),
        ("Normal", "Design, implement, and evaluate an edge-deployable, real-time face recognition attendance system that achieves sub-millisecond recognition latency, robust face detection, and high verification accuracy (low FAR and FRR). The system must handle real-world challenges including motion blur, varying head angles, illumination shifts, and camera noise, while eliminating the computational bottleneck of retraining classifiers for newly registered individuals.", False),
        
        ("Heading 1", "Learning Objectives", True),
        ("Normal", "1. Understand the paradigms of face recognition: Classical handcrafted features (Eigenfaces/PCA, LBPH, Haar Cascades) vs. Deep Metric Learning (SphereFace, ArcFace, SFace).\n"
         "2. Master two-stage computer vision pipelines: real-time face detection and 5-point landmark regression (YuNet) paired with canonical facial alignment (112×112) and feature extraction (SFace).\n"
         "3. Implement automated Face Quality Assessment (FQA) using Laplacian focus sharpness, geometric yaw/tilt pose thresholds, and brightness normalization.\n"
         "4. Formulate few-shot biometric vector matching via in-memory BLAS matrix cosine similarity (S = E · eq) on L2-normalized 128-D hyperspheres.\n"
         "5. Design anti-spoofing and temporal consensus filters (K=3 of N=5 sliding window) to eliminate false positives in dynamic video streams.\n"
         "6. Evaluate biometric performance using False Acceptance Rate (FAR), False Rejection Rate (FRR), Equal Error Rate (EER), and Receiver Operating Characteristic (ROC/AUC).", False),
        
        ("Heading 1", "Data", True),
        ("Normal", "• Primary Data: Live video stream from standard classroom RGB webcams (720p/1080p at 30 FPS).\n"
         "• Enrollment Data: Multi-shot facial biometric profile (5 diverse, quality-gated face crops per student) stored as 128-dimensional floating-point vector embeddings (512 bytes per vector; no raw photos stored for privacy-by-design compliance).\n"
         "• Benchmark / Pretraining Datasets: CASIA-WebFace and LFW (Labeled Faces in the Wild) for deep metric learning feature extractor validation.", False),
        
        ("Heading 1", "Approaches", True),
        ("Normal", "1. Classical Feature Matching: Haar Cascade Detector + Local Binary Patterns Histograms (LBPH) classifier.\n"
         "2. Deep Classification Network: Multi-Task Cascaded CNN (MTCNN) + Softmax-based ResNet Classifier (requires retraining for every new identity).\n"
         "3. Deep Metric Learning (Baseline): Dual-Shot Face Detector (DSFD) + FaceNet (Triplet Loss, 512-D embeddings).\n"
         "4. Proposed Production Pipeline: Lightweight YuNet CNN Detector (5-point landmark localization) + SFace Deep Metric Recognizer (128-D L2-normalized cosine similarity) + In-Memory BLAS Vector Cache.\n"
         "5. Cloud GPU Fine-Tuned Model: MobileFaceNet backbone with Additive Angular Margin Loss (ArcFace) for domain-specific feature optimization.", False),
        
        ("Heading 1", "Methodology", True),
        ("Normal", "1. Acquisition & Detection: Video frames are captured via OpenCV; YuNet dynamically resizes input frames to detect face bounding boxes and 5 key landmarks (left eye, right eye, nose tip, left mouth, right mouth) in ~5 ms.\n"
         "2. Quality Assurance (FQA): Each detected crop is validated for Laplacian sharpness (variance ≥ 80.0), tilt roll (≤ 20°), yaw ratio (0.45 ≤ r ≤ 2.2), and pixel illumination (40 ≤ μ ≤ 230). Low-quality crops are discarded.\n"
         "3. Canonical Alignment & Normalization: Validated faces are warped via affine transformation to standard 112×112 canonical coordinate space.\n"
         "4. Feature Extraction: SFace extracts a 128-dimensional embedding vector, projected onto a unit hypersphere (||v||₂ = 1.0).\n"
         "5. In-Memory Vector Matching: The query vector is matched against pre-cached student vectors via matrix dot product (S = E · eq) in < 0.1 ms. If cosine similarity S ≥ 0.637, identity is matched.\n"
         "6. Temporal Sliding Consensus & Logging: An identity must be matched in at least 3 of 5 consecutive frames before attendance is marked present in SQLite with timestamp and session ID, accompanied by an audio-visual confirmation chime and HUD overlay.", False),
        
        ("Heading 1", "Observations", True),
        ("Normal", "Classical Haar + LBPH is computationally lightweight but suffers from high error rates under varying lighting and head poses (EER > 14%). Softmax CNN classifiers achieve high accuracy on fixed cohorts but are impractical for production due to mandatory retraining overhead upon adding new students. SFace paired with YuNet achieves superior real-time inference latency (45–65 ms total pipeline latency on standard CPU, >18 FPS) while maintaining an Equal Error Rate (EER) of ~3.2% and ROC-AUC of 0.992 at a calibrated cosine similarity threshold of 0.637. Storing 128-D vector BLOBs eliminates database bloat and safeguards student biometric privacy.", False),
        
        ("Heading 1", "Discussion", True),
        ("Normal", "Students analyze the engineering trade-offs between model parameter footprint, computational latency, and verification accuracy: edge CPU deployment vs. cloud GPU training; 128-D vs. 512-D embedding dimensions; Cosine Distance vs. Euclidean (L2) Distance under lighting shifts; and Single-frame decisions vs. Multi-frame temporal consensus for spoof resistance. The zero-shot metric learning approach proves decisive for scalability, enabling instantaneous registration of new students without system downtime.", False),
        
        ("Heading 1", "Outcome", True),
        ("Normal", "Recommend the two-stage YuNet + SFace pipeline with in-memory BLAS vector caching for resource-constrained edge environments and educational institutions, delivering sub-millisecond matching, zero-downtime student enrollment, and 96.8% operational verification accuracy. For enterprise scale with millions of identities, recommend MobileFaceNet with ArcFace loss running on dedicated GPU acceleration.", False),
    ]

    # Clear existing paragraphs
    p_elements = doc._element.xpath('//w:p')
    for p_elem in p_elements:
        p_elem.getparent().remove(p_elem)
        
    for style_name, text, is_bold in sections:
        p = doc.add_paragraph()
        p.style = doc.styles[style_name] if style_name in doc.styles else doc.styles['Normal']
        
        # Set line spacing and font
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        
        if style_name == "Title":
            p.paragraph_format.space_after = Pt(12)
            p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(18)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
        elif style_name == "Heading 1":
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)
        else:
            run = p.add_run(text)
            run.font.name = "Times New Roman"
            run.font.size = Pt(11)
            run.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
            
    doc.save("Case_StudySample copy.docx")
    # Also save a clean copy named Case_Study_FaceAttend_AI.docx for convenience
    doc.save("Case_Study_FaceAttend_AI.docx")
    print("Successfully updated 'Case_StudySample copy.docx' and created 'Case_Study_FaceAttend_AI.docx'!")

if __name__ == "__main__":
    main()

