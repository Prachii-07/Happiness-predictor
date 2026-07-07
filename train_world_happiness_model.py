import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_model(input_path: str, output_path: str, test_size: float = 0.2) -> dict:
    data = pd.read_csv(input_path)

    # Basic cleaning for the dataset.
    for col in ["Happiness Score", "Economy", "Family", "Health", "Freedom", "Generosity", "Corruption", "Dystopia", "Job Satisfaction"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Region"] = data["Region"].fillna("Unknown")

    target_col = "Happiness Score"
    feature_cols = [
        "Economy",
        "Family",
        "Health",
        "Freedom",
        "Generosity",
        "Corruption",
        "Dystopia",
        "Job Satisfaction",
        "Region",
    ]

    X = data[feature_cols]
    y = data[target_col]

    numeric_features = [
        "Economy",
        "Family",
        "Health",
        "Freedom",
        "Generosity",
        "Corruption",
        "Dystopia",
        "Job Satisfaction",
    ]
    categorical_features = ["Region"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_features),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "r2_score": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "rmse": mean_squared_error(y_test, predictions, squared=False),
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a world happiness prediction model")
    parser.add_argument("--input", default="World Happiness Report.csv", help="Path to the CSV dataset")
    parser.add_argument("--output", default="models/world_happiness_model.joblib", help="Where to save the trained model")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    metrics = build_model(args.input, args.output, test_size=args.test_size)

    print("Model trained successfully.")
    print(f"R2 Score: {metrics['r2_score']:.3f}")
    print(f"MAE: {metrics['mae']:.3f}")
    print(f"RMSE: {metrics['rmse']:.3f}")
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()
