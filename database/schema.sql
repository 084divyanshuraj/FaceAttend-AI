-- FaceAttend AI: Relational SQLite Schema
-- Configured for Foreign Key enforcement and Write-Ahead Logging (WAL)

PRAGMA foreign_keys = ON;

-- 1. Students Table
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_number TEXT NOT NULL UNIQUE,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    department TEXT NOT NULL,
    email TEXT UNIQUE,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Facial Embeddings Table (Stores 128-D float32 vectors as 512-byte binary blobs)
CREATE TABLE IF NOT EXISTS face_embeddings (
    embedding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id INTEGER NOT NULL,
    embedding_blob BLOB NOT NULL,
    sample_index INTEGER NOT NULL,
    quality_score REAL NOT NULL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE
);

-- 3. Attendance Sessions Table
CREATE TABLE IF NOT EXISTS sessions (
    session_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_code TEXT NOT NULL UNIQUE,
    course_code TEXT NOT NULL,
    course_name TEXT NOT NULL,
    instructor TEXT NOT NULL,
    session_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME,
    status TEXT NOT NULL DEFAULT 'ACTIVE' CHECK(status IN ('ACTIVE', 'COMPLETED', 'CANCELLED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Attendance Records Table
-- Note: UNIQUE(session_id, student_id) ensures strict idempotency and zero duplicate logs per session
CREATE TABLE IF NOT EXISTS attendance (
    attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    student_id INTEGER NOT NULL,
    marked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    confidence_score REAL NOT NULL,
    method TEXT NOT NULL DEFAULT 'AUTO_FACE' CHECK(method IN ('AUTO_FACE', 'MANUAL_ADMIN')),
    remarks TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE,
    FOREIGN KEY (student_id) REFERENCES students (student_id) ON DELETE CASCADE,
    UNIQUE(session_id, student_id)
);

-- 5. Audit Logs Table (Tracks administrative manual overrides and data modifications)
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'admin',
    action TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    target_id INTEGER,
    details TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Performance Optimization Indexes
CREATE INDEX IF NOT EXISTS idx_embeddings_student_id ON face_embeddings (student_id);
CREATE INDEX IF NOT EXISTS idx_attendance_session ON attendance (session_id);
CREATE INDEX IF NOT EXISTS idx_attendance_student ON attendance (student_id);
CREATE INDEX IF NOT EXISTS idx_students_roll ON students (roll_number);
CREATE INDEX IF NOT EXISTS idx_sessions_code ON sessions (session_code);

