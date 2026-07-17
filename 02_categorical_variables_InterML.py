"""
Lesson 02: Handling Categorical Variables
=========================================

Learning objectives
-------------------
1. Identify numerical and categorical predictor columns.
2. Understand categorical cardinality.
3. Compare three categorical-data strategies using validation MAE:
   - Drop categorical columns
   - Ordinal encoding
   - One-hot encoding for low-cardinality columns
4. Handle categories appearing only in validation data.
5. Preserve row indexes and meaningful one-hot feature names.
6. Save the experiment results as a CSV file.

Dataset
-------
Kaggle House Prices / Home Data for ML Course.

Example usage
-------------
python scripts/02_categorical_variables.py --train-path data/train.csv

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
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


TARGET = "SalePrice"
RANDOM_STATE = 0
LOW_CARDINALITY_LIMIT = 10
MISSING_CATEGORY = "__MISSING__"


def parse_args() -> argparse.Namespace:
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare methods for handling categorical variables."
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
        default=Path("results/02_categorical_variables_results.csv"),
        help="Where to save the comparison table.",
    )
    return parser.parse_args()


def load_data(train_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load all predictors, including categorical columns."""
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Download train.csv and place it in data/, or provide "
            "--train-path with the correct location."
        )

    data = pd.read_csv(train_path, index_col="Id")
    data = data.dropna(subset=[TARGET])

    y = data[TARGET]
    X = data.drop(columns=[TARGET])

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


def impute_numerical_columns(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    numerical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median-impute numerical columns using training data only."""
    imputer = SimpleImputer(strategy="median")

    train_array = imputer.fit_transform(X_train[numerical_cols])
    valid_array = imputer.transform(X_valid[numerical_cols])

    train_frame = pd.DataFrame(
        train_array,
        columns=numerical_cols,
        index=X_train.index,
    )
    valid_frame = pd.DataFrame(
        valid_array,
        columns=numerical_cols,
        index=X_valid.index,
    )

    return train_frame, valid_frame


def fill_categorical_missing_values(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    categorical_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace missing categorical values with a visible category label."""
    train_frame = X_train[categorical_cols].fillna(MISSING_CATEGORY).copy()
    valid_frame = X_valid[categorical_cols].fillna(MISSING_CATEGORY).copy()
    return train_frame, valid_frame


def make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a dense one-hot encoder.

    sparse_output is used by newer scikit-learn versions.
    sparse is the equivalent parameter in older versions.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


def compare_categorical_methods(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate three categorical-data strategies on the same split."""
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()
    numerical_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    if not categorical_cols:
        raise ValueError(
            "No categorical columns were found. Make sure X was created "
            "from all predictors, not from numerical predictors only."
        )

    num_X_train, num_X_valid = impute_numerical_columns(
        X_train,
        X_valid,
        numerical_cols,
    )
    cat_X_train, cat_X_valid = fill_categorical_missing_values(
        X_train,
        X_valid,
        categorical_cols,
    )

    cardinality = pd.DataFrame(
        {
            "column": categorical_cols,
            "unique_training_values": [
                cat_X_train[col].nunique() for col in categorical_cols
            ],
        }
    ).sort_values("unique_training_values")

    low_cardinality_cols = cardinality.loc[
        cardinality["unique_training_values"] < LOW_CARDINALITY_LIMIT,
        "column",
    ].tolist()

    high_cardinality_cols = [
        col for col in categorical_cols if col not in low_cardinality_cols
    ]

    results: list[dict[str, float | int | str]] = []

    # ------------------------------------------------------------------
    # Approach 1: Drop all categorical columns
    # ------------------------------------------------------------------
    drop_X_train = num_X_train.copy()
    drop_X_valid = num_X_valid.copy()

    results.append(
        {
            "method": "Drop categorical columns",
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
    # Approach 2: Ordinal encoding
    # ------------------------------------------------------------------
    ordinal_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    ordinal_train_array = ordinal_encoder.fit_transform(cat_X_train)
    ordinal_valid_array = ordinal_encoder.transform(cat_X_valid)

    ordinal_train = pd.DataFrame(
        ordinal_train_array,
        columns=categorical_cols,
        index=X_train.index,
    )
    ordinal_valid = pd.DataFrame(
        ordinal_valid_array,
        columns=categorical_cols,
        index=X_valid.index,
    )

    ordinal_X_train = pd.concat(
        [num_X_train, ordinal_train],
        axis=1,
    )
    ordinal_X_valid = pd.concat(
        [num_X_valid, ordinal_valid],
        axis=1,
    )

    results.append(
        {
            "method": "Ordinal encoding",
            "feature_count": ordinal_X_train.shape[1],
            "validation_mae": score_dataset(
                ordinal_X_train,
                ordinal_X_valid,
                y_train,
                y_valid,
            ),
        }
    )

    # ------------------------------------------------------------------
    # Approach 3: One-hot encode low-cardinality columns
    # ------------------------------------------------------------------
    one_hot_encoder = make_one_hot_encoder()

    one_hot_train_array = one_hot_encoder.fit_transform(
        cat_X_train[low_cardinality_cols]
    )
    one_hot_valid_array = one_hot_encoder.transform(
        cat_X_valid[low_cardinality_cols]
    )

    one_hot_feature_names = one_hot_encoder.get_feature_names_out(
        low_cardinality_cols
    )

    one_hot_train = pd.DataFrame(
        one_hot_train_array,
        columns=one_hot_feature_names,
        index=X_train.index,
    )
    one_hot_valid = pd.DataFrame(
        one_hot_valid_array,
        columns=one_hot_feature_names,
        index=X_valid.index,
    )

    one_hot_X_train = pd.concat(
        [num_X_train, one_hot_train],
        axis=1,
    )
    one_hot_X_valid = pd.concat(
        [num_X_valid, one_hot_valid],
        axis=1,
    )

    # All generated one-hot names are already strings.
    assert all(isinstance(col, str) for col in one_hot_X_train.columns)
    assert list(one_hot_X_train.columns) == list(one_hot_X_valid.columns)

    results.append(
        {
            "method": (
                "One-hot encoding "
                f"(< {LOW_CARDINALITY_LIMIT} unique values)"
            ),
            "feature_count": one_hot_X_train.shape[1],
            "validation_mae": score_dataset(
                one_hot_X_train,
                one_hot_X_valid,
                y_train,
                y_valid,
            ),
        }
    )

    print("\nCategorical-column summary:")
    print(f"Total categorical columns: {len(categorical_cols)}")
    print(f"One-hot encoded columns:    {len(low_cardinality_cols)}")
    print(f"Dropped high-cardinality:   {len(high_cardinality_cols)}")

    if high_cardinality_cols:
        print("\nHigh-cardinality columns excluded from one-hot encoding:")
        print(", ".join(high_cardinality_cols))

    results_frame = (
        pd.DataFrame(results)
        .sort_values("validation_mae")
        .reset_index(drop=True)
    )

    return results_frame, cardinality


def main() -> None:
    args = parse_args()

    X, y = load_data(args.train_path)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("Lesson 02: Handling Categorical Variables")
    print("=" * 44)
    print(f"Training shape:   {X_train.shape}")
    print(f"Validation shape: {X_valid.shape}")

    results, cardinality = compare_categorical_methods(
        X_train,
        X_valid,
        y_train,
        y_valid,
    )

    print("\nCategorical cardinality:")
    print(cardinality.to_string(index=False))

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

    cardinality_path = args.results_path.with_name(
        "02_categorical_cardinality.csv"
    )
    cardinality.to_csv(cardinality_path, index=False)

    print(f"\nSaved comparison to:  {args.results_path}")
    print(f"Saved cardinality to: {cardinality_path}")


if __name__ == "__main__":
    main()
