"""Model training and persistence."""
import joblib
from sklearn.ensemble import RandomForestRegressor
from config.settings import MODEL_PATH


BEST_PARAMS = {
    'bootstrap': True,
    'max_depth': 11,
    'max_features': 'log2',
    'min_samples_leaf': 4,
    'min_samples_split': 9,
    'n_estimators': 161,
    'random_state': 42,
    'n_jobs': -1
}


def train_best_model(X_train, y_train) -> RandomForestRegressor:
    """Train the final production model."""
    model = RandomForestRegressor(**BEST_PARAMS)
    model.fit(X_train, y_train)
    return model


def save_model(model: RandomForestRegressor, path: str = MODEL_PATH) -> None:
    """Persist trained model."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path: str = MODEL_PATH) -> RandomForestRegressor:
    """Load the trained model."""
    return joblib.load(path)