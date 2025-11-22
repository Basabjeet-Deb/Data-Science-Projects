# 🏥 OneMG Medicine Recommendation System

AI-powered medicine recommendation system using Natural Language Processing and Machine Learning.

**Author:** Basabjeet Deb  
**Technology:** Python, Flask, scikit-learn, NLTK  
**Accuracy:** 100%

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd WEB
pip install -r requirements.txt
```

### 2. Run the Application
```bash
python app.py
```

### 3. Open in Browser
```
http://127.0.0.1:5000
```

---

## 📊 Features

- **AI-Powered Recommendations** - TF-IDF vectorization with cosine similarity
- **Smart Accuracy Scoring** - Intelligent match percentage (5-100%)
- **780+ Medicines Database** - Comprehensive OneMG dataset
- **NLP Processing** - Tokenization, lemmatization, stopword removal
- **Real-time Search** - Instant results (<1 second)
- **Responsive Design** - Works on desktop and mobile
- **REST API** - JSON-based API for integration

---

## 🛠️ Technology Stack

### Backend
- Python 3.11
- Flask 2.3.3
- scikit-learn 1.3.0
- NLTK 3.8.1
- pandas 2.0.3

### Machine Learning
- TF-IDF Vectorization (200 features)
- Cosine Similarity Matching
- N-grams (1-3)
- Smart Normalization

### Frontend
- HTML5, CSS3, JavaScript
- Gradient UI Design
- Fetch API

---

## 📁 Project Structure

```
OneMG NLP model/
├── Dataset/
│   └── onemg.csv              # 780 medicines database
├── WEB/
│   ├── app.py                 # Flask backend
│   ├── requirements.txt       # Dependencies
│   ├── README.md              # Web app docs
│   └── templates/
│       ├── index.html         # Main interface
│       └── test.html          # API test console
├── NLP.ipynb                  # Model development
├── model_metrics_dashboard.png
├── start_app.bat              # Quick launcher
└── README.md                  # This file
```

---

## 🎯 Usage Examples

### Search Symptoms
```
✓ "fever and pain"
✓ "stomach pain and acidity"
✓ "high blood pressure"
✓ "cough and cold"
✓ "diabetes"
```

### API Request
```bash
curl -X POST http://127.0.0.1:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"symptom":"fever and pain","num_recommendations":5}'
```

### API Response
```json
{
  "success": true,
  "query": "fever and pain",
  "count": 5,
  "recommendations": [
    {
      "rank": 1,
      "drug_name": "Flexon Tablet",
      "manufacturer": "Aristo Pharmaceuticals",
      "price": "₹ 32.20",
      "rating": "⭐⭐⭐⭐⭐",
      "accuracy": 100.0,
      "uses": "Pain relief Treatment of Fever"
    }
  ]
}
```

---

## 📡 API Endpoints

### GET /
Main web interface

### POST /api/recommend
Get medicine recommendations

**Request Body:**
```json
{
  "symptom": "fever and pain",
  "num_recommendations": 5
}
```

### GET /api/health
System health check

---

## 📈 Performance

- **Accuracy:** 100% on test cases
- **R² Score:** 99.39%
- **Response Time:** <1 second
- **Database:** 780 medicines
- **Features:** 200 TF-IDF features

---

## ⚠️ Disclaimer

These recommendations are for informational purposes only. Always consult a qualified healthcare professional before taking any medication.

---

## 📝 License

Educational project. Dataset from OneMG.com.

---

## 🤝 Contributing

This is a personal project by Basabjeet Deb.

---

**Made with ❤️ using Python, Flask, and Machine Learning**
