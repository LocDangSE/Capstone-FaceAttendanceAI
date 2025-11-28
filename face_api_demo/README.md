# 🎓 Face Recognition API Service

A high-performance, modular Flask-based face recognition API for attendance systems. Uses DeepFace with Facenet512 model for accurate face detection and recognition.

## 🚀 Features

- ✅ **Fast Recognition**: 3-5x faster with embedding cache
- ✅ **Rate Limiting**: Prevents CPU overload (1 FPS max)
- ✅ **Type Safety**: Pydantic models for runtime validation
- ✅ **Structured Logging**: Rotating logs with proper levels
- ✅ **Cloud Storage**: Supabase integration for persistence
- ✅ **Session Tracking**: Track recognition sessions
- ✅ **Image Preprocessing**: Automatic optimization
- ✅ **Multi-Face Detection**: Recognize multiple people

## 📋 Prerequisites

- Python 3.11.9+
- Supabase account (optional, for cloud storage)

## 🔧 Installation

### 1. Clone the repository
```bash
cd d:\Capstone\Test\face_api_demo
```

### 2. Install dependencies
```bash
# Using the main project virtual environment
D:/Capstone/Test/.venv/Scripts/python.exe -m pip install -r requirements.txt
```

### 3. Configure environment
```bash
# Copy the example environment file
copy .env.example .env

# Edit .env with your configuration
notepad .env
```

Required environment variables:
```env
# Flask Configuration
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=true

# Supabase (optional)
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_BUCKET=student-photos
SUPABASE_ENABLED=true

# DeepFace Model Settings
DEEPFACE_MODEL=Facenet512
DEEPFACE_DETECTOR=opencv
CONFIDENCE_THRESHOLD=0.6
RECOGNITION_FPS_LIMIT=1.0
```

## 🚀 Running the Server

### Start the server
```bash
D:/Capstone/Test/.venv/Scripts/python.exe app.py
```

Server will start on `http://localhost:5000`

### Run tests
```bash
# Validation tests
D:/Capstone/Test/.venv/Scripts/python.exe test_refactoring.py

# API endpoint tests (requires running server)
D:/Capstone/Test/.venv/Scripts/python.exe test_api.py
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```
Returns service status and cache statistics.

### Detect Faces
```http
POST /api/face/detect
Content-Type: multipart/form-data

image: <file>
```
Detects faces in an uploaded image.

### Register Student
```http
POST /api/face/register
Content-Type: multipart/form-data

image: <file>
studentId: <string>
```
Registers a student's face for recognition.

### Recognize Faces
```http
POST /api/face/recognize
Content-Type: multipart/form-data

image: <file>
classId: <string> (optional)
```
Recognizes faces in an image against the database.

### Get Session Results
```http
GET /api/session/<session_id>/results
```
Retrieves results for a specific recognition session.

### Cache Statistics
```http
GET /api/cache/stats
```
Returns embedding cache statistics.

### Clear Cache
```http
POST /api/cache/clear
Content-Type: application/json

{
  "studentId": "<string>" // Optional, clears specific student or all if omitted
}
```

## 📁 Project Structure

```
face_api_demo/
├── app.py                      # Main Flask application (400 lines)
├── requirements.txt            # Python dependencies
├── .env.example               # Environment template
├── .env                       # Your configuration (gitignored)
│
├── config/                    # Configuration management
│   ├── __init__.py
│   ├── settings.py           # Pydantic settings
│   └── logging_config.py     # Structured logging
│
├── services/                  # Business logic
│   ├── __init__.py
│   ├── image_processor.py    # Image preprocessing
│   ├── embedding_cache.py    # Fast embedding lookup
│   ├── face_recognition_service.py  # Core recognition
│   └── supabase_service.py   # Cloud storage
│
├── models/                    # Data models
│   ├── __init__.py
│   └── schemas.py            # Pydantic request/response models
│
├── utils/                     # Utilities
│   ├── __init__.py
│   ├── file_handler.py       # File operations
│   └── validators.py         # Input validation
│
├── embeddings/               # Cached face embeddings (.npy files)
├── sessions/                 # Recognition session results
├── logs/                     # Application logs
├── temp/                     # Temporary uploads
├── uploads/                  # Uploaded files
├── face_database/           # Student face images
│
├── test_refactoring.py      # Validation tests
└── test_api.py              # API endpoint tests
```

## 🎯 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Student Registration | 1-2s | Single face detection + embedding |
| Face Recognition (cached) | 0.5-1s | Fast lookup from cache |
| Face Recognition (uncached) | 2-3s | Generates embedding on-demand |
| Memory Usage | ~400MB | With 100 cached embeddings |

## 🔐 Security Features

- ✅ File type validation (only jpg/jpeg/png)
- ✅ File size limits (5MB max)
- ✅ Filename sanitization
- ✅ UUID validation
- ✅ Student ID sanitization
- ✅ No sensitive data in logs

## 🧪 Testing

### Unit Tests
```bash
D:/Capstone/Test/.venv/Scripts/python.exe test_refactoring.py
```

Tests:
- Module imports
- Service initialization
- Directory structure
- Cache statistics

### API Tests
```bash
# Start server first
D:/Capstone/Test/.venv/Scripts/python.exe app.py

# In another terminal
D:/Capstone/Test/.venv/Scripts/python.exe test_api.py
```

## 📊 Configuration Options

### DeepFace Models
- `Facenet512` (recommended, most accurate)
- `VGG-Face`
- `Facenet`
- `OpenFace`
- `DeepFace`
- `ArcFace`

### Detectors
- `opencv` (default, fastest)
- `retinaface` (most accurate)
- `mtcnn`
- `ssd`
- `dlib`

### Distance Metrics
- `cosine` (recommended)
- `euclidean`
- `euclidean_l2`

## 🐛 Troubleshooting

### Server won't start
1. Check `.env` file exists and has correct values
2. Verify Python version: `python --version` (should be 3.11.9+)
3. Check logs in `logs/face_api_YYYYMMDD.log`

### No faces detected
1. Ensure good image quality (not blurry)
2. Face should be clearly visible
3. Try adjusting `CONFIDENCE_THRESHOLD` in `.env`

### Recognition too slow
1. Check if embedding cache is working: `/api/cache/stats`
2. Consider using faster detector (`opencv` instead of `retinaface`)
3. Reduce image size before uploading

### Supabase errors
1. Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
2. Check bucket exists and is accessible
3. Set `SUPABASE_ENABLED=false` to disable if not needed

## 📝 Logs

Logs are stored in `logs/face_api_YYYYMMDD.log` with:
- Automatic rotation (max 10MB per file)
- Keep 5 backup files
- Timestamps and log levels
- Request/response tracking

## 🔄 Updates

### Updating dependencies
```bash
D:/Capstone/Test/.venv/Scripts/python.exe -m pip install -r requirements.txt --upgrade
```

### Clearing cache
```bash
# Via API
curl -X POST http://localhost:5000/api/cache/clear

# Manually
rmdir /s embeddings
mkdir embeddings
```

## 🤝 Integration

### .NET Backend Integration
The API is 100% backward compatible with existing .NET backend. No changes required.

### Frontend Integration
Use the provided Postman collection (`Postman_Collection.json`) for testing.

## 📄 License

Part of Face-ID-Attendance capstone project.

## 👨‍💻 Development

### Code Style
- PEP8 compliant
- Type hints throughout
- Comprehensive docstrings
- Modular design (<400 lines per file)

### Architecture
- Clean Architecture principles
- Service layer pattern
- Dependency injection
- Single Responsibility Principle

## 🆘 Support

For issues or questions:
1. Check logs in `logs/` folder
2. Run validation tests
3. Review configuration in `.env`
4. Check service initialization messages

---

**Built with ❤️ using Flask, DeepFace, and Facenet512**
