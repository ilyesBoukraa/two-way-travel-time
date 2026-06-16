import numpy as np

# Import functions cleanly from your custom modules
from data_processing import (
    check_remaining_nulls,
    load_and_clean_data,
    log_geophysical_insights_and_drop_nulls,
    show_velocity_and_correlation_logic,
)
from eda import assess_data_structure, plot_visual_eda, summarize_statistics
from modeling import preprocess_and_split_data, run_predictive_modeling


def main():
    """
    Main orchestrator function that controls the loading, business logic,
    exploratory data analysis, and predictive execution workflow.
    """
    filepath = "data/volve_well_picks_modified.csv"

    print("\n" + "=" * 60)
    print("4.1. DATASET LOADING (Chargement du dataset)")
    print("=" * 60)

    # Step 1: Load and clean raw structural data
    df = load_and_clean_data(filepath)

    if df is None:
        print("❌ Pipeline stopped: Dataset could not be loaded.")
        return

    print("\n" + "=" * 60)
    print("QUESTIONS - PART 1 (Observing the Dataset)")
    print("=" * 60)

    # Q1. How many wells are in the dataset?
    num_wells = df["WELL"].nunique()
    print(f"Q1. How many wells does the dataset contain?")
    print(f"    👉 Answer: {num_wells} wells")
    print("-" * 50)

    # Q2. How many geological formations are present?
    num_formations = df["PICKS"].nunique()
    formation_names = df["PICKS"].unique()
    print(f"Q2. How many geological formations are present?")
    print(f"    👉 Answer: {num_formations} formations")
    print(f"    👉 Formations list: {list(formation_names)}")
    print("-" * 50)

    # Q3. What is the maximum TVD depth reached (in meters)?
    max_tvdss = np.abs(df["TVDSS"]).max()
    print(f"Q3. What is the maximum TVD depth reached (in meters)?")
    print(
        f"    👉 Answer: {max_tvdss:.2f} meters (True Vertical Depth Sub-Sea)"
    )
    print("-" * 50)

    # Q4. What is the range of TWT values (minimum and maximum, in ms)?
    twt_min = df["TWT"].min()
    twt_max = df["TWT"].max()
    print(f"Q4. What is the range of TWT values (minimum and maximum, in ms)?")
    print(f"    👉 Answer: Min = {twt_min:.2f} ms | Max = {twt_max:.2f} ms")
    print("-" * 50)

    # Q5. Are there missing values? If so, how do you treat them and why?
    print(
        "Q5. Y a-t-il des valeurs manquantes ? Si oui, comment les traitez-vous et pourquoi ?"
    )
    df = log_geophysical_insights_and_drop_nulls(df)
    print("-" * 50)

    # Step 4: Show velocity physics using only the remaining real target lines
    show_velocity_and_correlation_logic(df)

    # Step 5: Final validation check (Should verify 0 remaining null rows)
    check_remaining_nulls(df, stage_label="POST-CLEANING & NULL DROPPING")

    # Step 6: Complete Exploratory Data Analysis Pipeline
    assess_data_structure(df)
    summarize_statistics(df)
    plot_visual_eda(df)

    # Step 7: Preprocess and parse data into ML splits
    X_train, X_test, y_train, y_test = preprocess_and_split_data(df)

    # Step 8: Transition into predictive Machine Learning pipeline modeling
    run_predictive_modeling(X_train, X_test, y_train, y_test)


# -------------------------------------------------------------------
# EXECUTION PIPELINE CONTROL
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Call your newly wrapped main workflow function here
    main()