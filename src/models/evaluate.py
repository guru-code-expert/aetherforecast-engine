"""Model evaluation utilities."""
from sklearn.model_selection import train_test_split
from src.models.train import train_best_model
from src.utils.helpers import evaluate_model


def full_train_evaluate(X, y):
    """Train and evaluate on hold-out set."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = train_best_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    evaluate_model(y_test, y_pred, "Random Forest (Tuned)")
    
    return model, X_test, y_test, y_pred