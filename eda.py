import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def assess_data_structure(df):
    """Prints basic structural information, data types, and missing values."""
    print("\n=== 1. SHAPE, DATA TYPES & UNIQUES ===")
    summary_df = pd.DataFrame(
        {"Data Type": df.dtypes, "Unique Values": df.nunique()}
    )
    print(summary_df)

    print("\n=== 2. FIRST 5 ROWS ===")
    print(df.head())


def summarize_statistics(df):
    """Provides statistical summaries for both numerical and categorical variables."""
    print("\n=== 3. NUMERICAL SUMMARY ===")
    print(df.describe().T)

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(categorical_cols) > 0:
        print("\n=== 4. CATEGORICAL SUMMARY ===")
        for col in categorical_cols:
            print(f"\nValue counts for '{col}' (Top 10):")
            print(df[col].value_counts().head(10))
    print("-" * 50)


def plot_visual_eda(df):
    """Generates distributions and correlation matrices for the dataset."""
    sns.set_theme(style="whitegrid")
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    if len(numerical_cols) > 0:
        cols_to_plot = numerical_cols[:6]
        df[cols_to_plot].hist(bins=30, figsize=(12, 8), layout=(-1, 3))
        plt.suptitle("Distributions of Numerical Features")
        plt.tight_layout()
        plt.show()

    if len(numerical_cols) > 1:
        plt.figure(figsize=(10, 8))
        corr = df[numerical_cols].corr()
        sns.heatmap(
            corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1
        )
        plt.title("Correlation Matrix (Target TWT vs Features)")
        plt.tight_layout()
        plt.show()