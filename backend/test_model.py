from app.services.model_service import (
    logistic_model,
    random_forest_model,
    expected_features
)

print("Models loaded successfully!")

print("Logistic Regression:", type(logistic_model).__name__)
print("Random Forest:", type(random_forest_model).__name__)

print("Expected input features:", len(expected_features))

print("Logistic Regression classes:", logistic_model.classes_)
print("Random Forest classes:", random_forest_model.classes_)