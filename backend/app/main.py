from app.services.model_service import metadata

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.prediction import router as prediction_router

app = FastAPI(
    title="RareTumor-Omics API",
    description="PCPG molecular subtype prediction using integrated genomics and epigenomics data.",
    version="1.0.0"
)

# Allow the React frontend to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Welcome to RareTumor-Omics API",
        "status": "running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": True
    }


@app.get("/models")
def available_models():
    return {
        "available_models": [
            "logistic_regression",
            "random_forest"
        ],
        "prediction_target": metadata["target"]
    }


app.include_router(prediction_router)
