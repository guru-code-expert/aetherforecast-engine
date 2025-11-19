"""Utility functions."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred, model_name: str = "") -> None:
    """Print standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name}".ljust(30))
    print(f"  R²: {r2:.4f} | MAE: {mae:.2f}\n")


def plot_correlation_heatmap(df_num: pd.DataFrame) -> None:
    """Display correlation heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_num.corr(), annot=True, cmap="RdYlGn", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()