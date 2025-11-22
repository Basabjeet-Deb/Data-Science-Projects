# OneMG Medicine Recommendation - Web Application

Flask-based backend for AI-powered medicine recommendations.

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:5000`

## API Endpoints

### POST /api/recommend
Get medicine recommendations

**Request:**
```json
{
  "symptom": "fever and pain",
  "num_recommendations": 5
}
```

**Response:**
```json
{
  "success": true,
  "query": "fever and pain",
  "recommendations": [...]
}
```

### GET /api/health
System health check

## Files

- `app.py` - Flask backend
- `requirements.txt` - Python dependencies
- `templates/index.html` - Main interface
- `templates/test.html` - API test console

## Author

Basabjeet Deb
