import argparse
import joblib
import pandas as pd


def predict(input_path: str, model_path: str) -> None:
    model = joblib.load(model_path)
    data = pd.read_csv(input_path)

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

    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    predictions = model.predict(data[feature_cols])
    data["Predicted Happiness Score"] = predictions
    print(data[["Country", "Predicted Happiness Score"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict happiness scores with a trained model")
    parser.add_argument("--input", required=True, help="Path to a CSV file containing the feature values")
    parser.add_argument("--model", default="models/world_happiness_model.joblib", help="Path to the trained model")
    args = parser.parse_args()

    predict(args.input, args.model)


if __name__ == "__main__":
    main()
