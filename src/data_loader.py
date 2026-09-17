"""
Load the credit card fraud dataset and perform a stratified train/test split.

IMPORTANT: The split happens BEFORE any resampling (SMOTE/undersampling).
Balancing before splitting leaks fraud examples into the test set indirectly
via synthetic neighbors, inflating test metrics. Always split first.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/creditcard.csv"
TARGET_COL = "Class"
RANDOM_STATE = 42


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV. Raises a clear error if the file is missing."""
    try:
        df = pd.read_csv(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Could not find {path}. Download creditcard.csv from "
            "https://www.kaggle.com/mlg-ulb/creditcardfraud and place it in data/."
        ) from e
    return df


def get_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified split on the target so both sets keep the same fraud rate."""
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test


def summarize_imbalance(y) -> dict:
    counts = y.value_counts()
    total = len(y)
    return {
        "total": total,
        "fraud_count": int(counts.get(1, 0)),
        "non_fraud_count": int(counts.get(0, 0)),
        "fraud_rate_pct": round(counts.get(1, 0) / total * 100, 4),
    }


if __name__ == "__main__":
    df = load_raw_data()
    print("Overall:", summarize_imbalance(df[TARGET_COL]))

    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print("Train:", summarize_imbalance(y_train))
    print("Test:", summarize_imbalance(y_test))
