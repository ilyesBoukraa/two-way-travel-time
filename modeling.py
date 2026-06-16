import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def preprocess_and_split_data(df):
    """Handles categorical encoding, isolates target from features, and splits data into

    train and test sets.
    """
    print("\n=== DATA PREPROCESSING FOR MACHINE LEARNING ===")

    # 1. Isolate and handle Categorical Columns via One-Hot Encoding
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    print(f"Categorical features to encode: {list(categorical_cols)}")

    df_encoded = pd.get_dummies(
        df, columns=categorical_cols, drop_first=True, dtype=int
    )

    # 2. Separate Target (y) and Features (X)
    y = df_encoded["TWT"]
    X = df_encoded.drop(columns=["TWT"], errors="ignore")

    print("Target variable isolated: TWT")
    print(
        f"Total feature matrix dimensions: {X.shape[0]} rows, {X.shape[1]} columns."
    )

    # 3. Perform Train-Test Split (80% Train, 20% Evaluation Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(
        f"Split complete: {X_train.shape[0]} training samples | {X_test.shape[0]} test samples."
    )
    print("-" * 50)

    return X_train, X_test, y_train, y_test

def run_predictive_modeling(X_train, X_test, y_train, y_test):
    """Trains ML models and compares them directly against the physical formula baseline

    (TWT = 2Z / V) scored on the exact same test partition.
    Saves both the performance matrix text and the cross-plot chart locally.
    """
    print("\n=== PHASE 2: PREDICTIVE MODELING TO PREDICT TWT ===")

    # 1. Define the machine learning models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "XGBoost Regressor": XGBRegressor(
            n_estimators=100, learning_rate=0.1, random_state=42
        ),
    }

    results = {}

    # Initialize a string to capture the table text for saving
    table_lines = []
    table_lines.append("=== PHASE 2: PREDICTIVE MODELING TO PREDICT TWT ===")
    table_lines.append("Evaluating Performance Matrix:")
    table_lines.append(
        f"{'Model Name':<25} | {'R2 Score':<10} | {'RMSE (ms)':<10} | {'MAE (ms)':<10}"
    )
    table_lines.append("-" * 65)

    # Print the header to console
    print(table_lines[1])
    print(table_lines[2])
    print(table_lines[3])

    # 2. Evaluate Machine Learning Models
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae}

        row_str = f"{name:<25} | {r2:<10.4f} | {rmse:<10.4f} | {mae:<10.4f}"
        print(row_str)
        table_lines.append(row_str)

    # 3. CALCULATE PHYSICAL FORMULA BASELINE
    v_field_avg = ((2 * np.abs(X_train["TVDSS"])) / (y_train / 1000)).mean()
    physical_predictions = (2 * np.abs(X_test["TVDSS"]) / v_field_avg) * 1000

    # Score analytical equation
    r2_phys = r2_score(y_test, physical_predictions)
    rmse_phys = np.sqrt(mean_squared_error(y_test, physical_predictions))
    mae_phys = mean_absolute_error(y_test, physical_predictions)

    phys_str = f"{'Physical Formula (2Z/V)':<25} | {r2_phys:<10.4f} | {rmse_phys:<10.4f} | {mae_phys:<10.4f}"
    print(phys_str)
    print("-" * 65)

    table_lines.append(phys_str)
    table_lines.append("-" * 65)

    # NEW LOGIC: Save the captured performance matrix to a text file
    txt_output_filename = "data/ml_vs_physics_metrics.txt"
    with open(txt_output_filename, "w", encoding="utf-8") as f:
        f.write("\n".join(table_lines))
    print(
        f"\n✔ Requirement #5 Performance Matrix table saved to '{txt_output_filename}'"
    )

    # 4. Plot Multi-Validation Comparison Chart
    plt.figure(figsize=(10, 6))
    plt.scatter(
        y_test,
        models["XGBoost Regressor"].predict(X_test),
        alpha=0.6,
        color="teal",
        label="XGBoost Predictions",
    )
    plt.scatter(
        y_test,
        physical_predictions,
        alpha=0.5,
        color="orange",
        marker="x",
        label="Physical Formula Baseline",
    )
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--",
        lw=2,
        label="Perfect Fit Line (Y=X)",
    )

    plt.xlabel("Actual TWT (ms)")
    plt.ylabel("Predicted TWT (ms)")
    plt.title(
        "Machine Learning Models vs. Geophysical Physical Formula Benchmark"
    )
    plt.legend()
    plt.tight_layout()

    img_output_filename = "data/ml_vs_physics_benchmark.png"
    plt.savefig(img_output_filename, dpi=300)
    print(f"✔ Requirement #6 Cross-Plot chart saved to '{img_output_filename}'")

    plt.show()