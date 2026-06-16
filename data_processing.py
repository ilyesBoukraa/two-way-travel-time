import numpy as np
import pandas as pd


def load_and_clean_data(filepath):
    """Loads the well picks CSV, drops specified columns, and renames coordinate headers."""
    try:
        df = pd.read_csv(filepath)
        print(f"✔ Data loaded successfully from '{filepath}'")
    except FileNotFoundError:
        print(f"❌ Error: Could not find the file at '{filepath}'.")
        return None

    # Drop unnecessary columns
    columns_to_drop = ["DIP", "AZI", "QLF", "INTRP", "OBS"]
    df = df.drop(columns=columns_to_drop, errors="ignore")

    # Rename coordinate columns
    df = df.rename(
        columns={"EASTING": "EASTING (X)", "NORTHING": "NORTHING (Y)"}
    )

    print(
        f"✔ Cleaning complete. Shape: {df.shape[0]} rows, {df.shape[1]} columns.\n"
    )
    return df


def check_remaining_nulls(df, stage_label="POST-CLEANING"):
    """Displays a summary of the columns and their null value counts using df.info()."""
    print(f"\n=== {stage_label} DATA SUMMARY (df.info) ===")
    df.info()
    print("-" * 50)


def log_geophysical_insights_and_drop_nulls(df):
    """Analyzes the raw relationship between TVDSS and TWT to log insights for the user,

    then cleanly drops the rows where the target TWT is missing to ensure maximum
    ML purity.
    """
    print("\n=== GEOPHYSICAL INSIGHTS & TARGET CLEANING ===")

    # Isolate complete rows to analyze the relationship
    complete_pairs = df[df["TWT"].notnull() & df["TVDSS"].notnull()]
    missing_twt_count = df["TWT"].isnull().sum()

    # Calculate the correlation to keep as documented insight
    raw_correlation = complete_pairs["TVDSS"].corr(complete_pairs["TWT"])

    print(
        f"[LOGGED INSIGHT] Raw correlation between TVDSS and TWT is: {raw_correlation:.4f}\n"
        "-> Rationale: This near-perfect negative linear correlation proves a strict physical link.\n"
        "   As sub-sea depth (TVDSS) gets deeper (more negative), acoustic travel time (TWT) increases.\n"
        f"-> Strategy: To keep the ML models perfectly unbiased, we are dropping the {missing_twt_count} rows\n"
        "   where the target TWT is missing instead of fabricating them."
    )

    # Cleanly drop the null target rows
    df_cleaned = df.dropna(subset=["TWT"]).copy()
    print(
        f"✔ Successfully dropped rows with missing targets. New shape: {df_cleaned.shape[0]} rows."
    )
    print("-" * 50)

    return df_cleaned


def show_velocity_and_correlation_logic(df):
    """Calculates Average Velocity isolated for demonstration and printing.

    This shows the underlying physics and codes users how the metrics
    correlate WITHOUT leaking data into the main modeling DataFrame.
    """
    print("\n=== GEOPHYSICAL DEMONSTRATION: SEISMIC VELOCITY & CORRELATION ===")

    # Calculate velocity in a temporary series for demonstration
    temp_v_avg = (2 * np.abs(df["TVDSS"])) / (df["TWT"] / 1000)

    print(
        "[EDUCATIONAL NOTE] We calculate Average Seismic Velocity using: V_avg = (2 * |TVDSS|) / (TWT / 1000)\n"
        "Even though V_avg is highly descriptive, it requires the target (TWT) to be calculated.\n"
        "Therefore, we show it here for structural understanding, but we DO NOT append it to the training features\n"
        "to prevent strict Data Leakage in the Machine Learning Phase.\n"
    )
    print("Summary statistics of the calculated field velocity profile (m/s):")
    print(temp_v_avg.describe())

    # Calculate correlation to prove the physical link explicitly
    corr_v_twt = temp_v_avg.corr(df["TWT"])
    corr_v_tvdss = temp_v_avg.corr(df["TVDSS"])
    print(f"\nCorrelation between calculated V_avg and TWT: {corr_v_twt:.4f}")
    print(f"Correlation between calculated V_avg and TVDSS: {corr_v_tvdss:.4f}")
    print("-" * 50)