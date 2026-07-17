"""
Lesson 03: Preprocessing Pipelines
==================================
** This hugely reduces the mistake space compared to the Lesson 2's way of coding. *****
Learning objectives
-------------------
1. Use ColumnTransformer for numerical and categorical columns.
2. Use Pipeline to connect preprocessing and model training.
3. Compare one-hot encoding and ordinal encoding fairly.
4. Handle missing and unseen categorical values safely.
5. Save the validation MAE results.

Dataset
-------
Kaggle House Prices / Home Data for ML Course.

Example
-------
python scripts/03_preprocessing_pipelines.py --train-path data/train.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


TARGET = "SalePrice"
RANDOM_STATE = 0
LOW_CARDINALITY_LIMIT = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare one-hot and ordinal preprocessing pipelines."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/train.csv"),
        help="Path to train.csv.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("results/03_pipeline_results.csv"),
        help="Where to save the MAE comparison.",
    )
    return parser.parse_args()


def load_data(train_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_path}\n"
            "Place train.csv in data/, or use --train-path."
        )

    data = pd.read_csv(train_path, index_col="Id")
    data = data.dropna(subset=[TARGET])

    y = data[TARGET]
    X = data.drop(columns=[TARGET])
    return X, y


def select_columns(
    X_train: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    categorical_cols = [
        column
        for column in X_train.columns
        if X_train[column].dtype == "object"
        and X_train[column].nunique() < LOW_CARDINALITY_LIMIT
    ]

    numerical_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    return numerical_cols, categorical_cols


def make_one_hot_encoder() -> OneHotEncoder:
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


def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_one_hot_pipeline(
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    numerical_transformer = SimpleImputer(strategy="constant")

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                make_one_hot_encoder(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", make_model()),
        ]
    )


def build_ordinal_pipeline(
    numerical_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    numerical_transformer = SimpleImputer(strategy="constant")

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", make_model()),
        ]
    )


def evaluate_pipeline(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
) -> dict[str, str | float]:
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_valid)

    return {
        "method": name,
        "validation_mae": mean_absolute_error(
            y_valid,
            predictions,
        ),
    }


def main() -> None:
    args = parse_args()
    X, y = load_data(args.train_path)

    X_train_full, X_valid_full, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    numerical_cols, categorical_cols = select_columns(X_train_full)
    selected_cols = categorical_cols + numerical_cols

    X_train = X_train_full[selected_cols].copy()
    X_valid = X_valid_full[selected_cols].copy()

    print("Lesson 03: Preprocessing Pipelines")
    print("=" * 38)
    print(f"Numerical columns:   {len(numerical_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    print(f"Training shape:      {X_train.shape}")
    print(f"Validation shape:    {X_valid.shape}")

    results = pd.DataFrame(
        [
            evaluate_pipeline(
                "One-hot encoding pipeline",
                build_one_hot_pipeline(
                    numerical_cols,
                    categorical_cols,
                ),
                X_train,
                X_valid,
                y_train,
                y_valid,
            ),
            evaluate_pipeline(
                "Ordinal encoding pipeline",
                build_ordinal_pipeline(
                    numerical_cols,
                    categorical_cols,
                ),
                X_train,
                X_valid,
                y_train,
                y_valid,
            ),
        ]
    ).sort_values("validation_mae").reset_index(drop=True)

    print("\nPipeline comparison:")
    print(
        results.to_string(
            index=False,
            formatters={"validation_mae": "{:,.2f}".format},
        )
    )

    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.results_path, index=False)
    print(f"\nSaved results to: {args.results_path}")


if __name__ == "__main__":
    main()
