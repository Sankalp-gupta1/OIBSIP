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


## Run the web demo locally

The hosted portfolio version is a static web demo, so it does not need an API key or database for local preview.

### Step 1 — clone the repository

```bash
git clone https://github.com/Sankalp-gupta1/OIBSIP.git
cd OIBSIP
```

### Step 2 — open the deployable web folder

```bash
cd web
```

### Step 3 — start a small local server

```bash
python -m http.server 8000
```

### Step 4 — open the app

```text
http://localhost:8000
```

Production demo: **https://oibsip-ml-lab.vercel.app/**

The original project files are kept in the repository. The `web/` version is the browser-friendly deployment made for easy review.
