"""
Lesson 01: Handling Missing Numerical Values
============================================

Learning objectives
-------------------
1. Identify numerical columns containing missing values.
2. Compare three missing-value strategies using validation MAE:
   - Drop columns with missing values
   - Median imputation
   - Median imputation with missing-value indicators
3. Fit preprocessing only on the training split to avoid data leakage.
4. Save the experiment results as a CSV file.

Dataset
-------
Kaggle House Prices / Home Data for ML Course.

Example usage
-------------
python scripts/01_missing_values.py --train-path data/train.csv

This is a learning script inspired by Kaggle Learn's Intermediate
Machine Learning course. The organization, explanations, and experiment
reporting are adapted for a personal learning portfolio.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


TARGET = "SalePrice"
RANDOM_STATE = 0


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare methods for handling missing numerical values."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to the House Prices training CSV.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/01_missing_values_results.csv"),
        help="Where to save the comparison table.",
    )
    return parser.parse_args()


def load_numerical_data(train_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the dataset and return numerical predictors and the target."""
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Download train.csv and place it in data/, or provide "
            "--train-path with the correct location."
        )

    data = pd.read_csv(train_path, index_col="Id")
    data = data.dropna(subset=[TARGET])

    y = data[TARGET]
    X_full = data.drop(columns=[TARGET])

    # This lesson focuses only on missing numerical values.
    X = X_full.select_dtypes(exclude="object").copy()

    return X, y


def make_model() -> RandomForestRegressor:
    """Create the same model for every preprocessing approach."""
    return RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def score_dataset(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> float:
    """Train a model and return validation mean absolute error."""
    model = make_model()
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictions)


def compare_missing_value_methods(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> pd.DataFrame:
    """Evaluate three missing-value strategies on the same split."""
    results: list[dict[str, float | int | str]] = []

    # ------------------------------------------------------------------
    # Approach 1: Drop columns containing missing values
    # ------------------------------------------------------------------
    columns_with_missing = X_train.columns[X_train.isna().any()]

    drop_X_train = X_train.drop(columns=columns_with_missing)
    drop_X_valid = X_valid.drop(columns=columns_with_missing)

    results.append(
        {
            "method": "Drop columns with missing values",
            "feature_count": drop_X_train.shape[1],
            "validation_mae": score_dataset(
                drop_X_train,
                drop_X_valid,
                y_train,
                y_valid,
            ),
        }
    )

    # ------------------------------------------------------------------
    # Approach 2: Median imputation
    # ------------------------------------------------------------------
    median_imputer = SimpleImputer(strategy="median")

    median_X_train = pd.DataFrame(
        median_imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    median_X_valid = pd.DataFrame(
        median_imputer.transform(X_valid),
        columns=X_valid.columns,
        index=X_valid.index,
    )

    results.append(
        {
            "method": "Median imputation",
            "feature_count": median_X_train.shape[1],
            "validation_mae": score_dataset(
                median_X_train,
                median_X_valid,
                y_train,
                y_valid,
            ),
        }
    )

    # ------------------------------------------------------------------
    # Approach 3: Median imputation plus missing-value indicators
    # ------------------------------------------------------------------
    indicator_imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    indicator_X_train_array = indicator_imputer.fit_transform(X_train)
    indicator_X_valid_array = indicator_imputer.transform(X_valid)

    # The model accepts arrays, but DataFrames make the feature count clear.
    indicator_feature_names = indicator_imputer.get_feature_names_out(
        X_train.columns
    )
    indicator_X_train = pd.DataFrame(
        indicator_X_train_array,
        columns=indicator_feature_names,
        index=X_train.index,
    )
    indicator_X_valid = pd.DataFrame(
        indicator_X_valid_array,
        columns=indicator_feature_names,
        index=X_valid.index,
    )

    results.append(
        {
            "method": "Median imputation + indicators",
            "feature_count": indicator_X_train.shape[1],
            "validation_mae": score_dataset(
                indicator_X_train,
                indicator_X_valid,
                y_train,
                y_valid,
            ),
        }
    )

    return (
        pd.DataFrame(results)
        .sort_values("validation_mae")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()

    X, y = load_numerical_data(args.train_path)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("Lesson 01: Handling Missing Numerical Values")
    print("=" * 46)
    print(f"Training shape:   {X_train.shape}")
    print(f"Validation shape: {X_valid.shape}")

    missing_counts = (
        X_train.isna()
        .sum()
        .loc[lambda values: values > 0]
        .sort_values(ascending=False)
    )

    print("\nMissing values in the training split:")
    if missing_counts.empty:
        print("No numerical columns contain missing values.")
    else:
        print(missing_counts.to_string())

    results = compare_missing_value_methods(
        X_train,
        X_valid,
        y_train,
        y_valid,
    )

    print("\nMethod comparison:")
    print(
        results.to_string(
            index=False,
            formatters={"validation_mae": "{:,.2f}".format},
        )
    )

    best_method = results.iloc[0]
    print(
        "\nBest validation result: "
        f"{best_method['method']} "
        f"(MAE = {best_method['validation_mae']:,.2f})"
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_path, index=False)
    print(f"\nSaved results to: {args.results_path}")


if __name__ == "__main__":
    main()
