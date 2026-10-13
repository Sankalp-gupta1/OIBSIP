# OIBSIP Machine Learning Lab

Three Oasis Infobyte internship projects deployed in one live application.

## Live app

**https://oibsip-ml-lab.vercel.app/**

The production app loads the actual saved model files already present in this repository:

- Iris Flower Detection — SVM with RBF kernel
- Email Spam Detection — saved text vectorizer + classifier
- Car Price Prediction — Random Forest Regressor

## Production stack

- FastAPI
- scikit-learn
- NumPy / Pandas
- saved pickle/joblib models
- responsive HTML/CSS/JavaScript UI
- Vercel

## API endpoints

- `GET /health`
- `POST /api/iris`
- `POST /api/spam`
- `POST /api/car`

The original Streamlit / Flask source files, training scripts, datasets and model artifacts remain in their original folders.

## Local production run

```bash
pip install -r requirements.txt
uvicorn api.index:app --reload
```

Open `http://127.0.0.1:8000`.

## Project folders

```text
OIBSIP/
├── iris_flower_detection/
├── email_spam_detection/
├── Car-Price-Prediction/
├── api/
│   └── index.py
├── requirements.txt
└── vercel.json
```

## Author

Sankalp Gupta  
GitHub: https://github.com/Sankalp-gupta1
