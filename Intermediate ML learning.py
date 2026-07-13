"""Compare methods for handling missing numerical values.

Dataset: Kaggle House Prices / Home Data for ML Course
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_PATH = Path("../input/train.csv")
TEST_PATH = Path("../input/test.csv")
TARGET = "SalePrice"
RANDOM_STATE = 0


def score_dataset(X_train, X_valid, y_train, y_valid) -> float:
    """Train a random forest and return its validation MAE."""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)
    return mean_absolute_error(y_valid, predictions)


def main() -> None:
    # -----------------------------------------------------------------------
    # Load and prepare the data
    # -----------------------------------------------------------------------

    train_data = pd.read_csv(TRAIN_PATH, index_col="Id")
    test_data = pd.read_csv(TEST_PATH, index_col="Id")

    # Remove rows where the target is missing.
    train_data = train_data.dropna(subset=[TARGET])

    # Separate the target from the predictors.
    y = train_data[TARGET]
    X_full = train_data.drop(columns=[TARGET])

    # Keep only numerical predictors.
    X = X_full.select_dtypes(exclude="object").copy()

    # Use exactly the same predictor columns and order in the test data.
    X_test = test_data[X.columns].copy()

    # Split the training data into training and validation sets.
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------------------------
    # Inspect missing values
    # -----------------------------------------------------------------------

    print(f"Training shape: {X_train.shape}")

    missing_counts = X_train.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(
        ascending=False
    )

    print("\nMissing values by column:")
    print(missing_counts if not missing_counts.empty else "No missing values.")

    # -----------------------------------------------------------------------
    # Approach 1: Drop columns containing missing values
    # -----------------------------------------------------------------------

    columns_with_missing = X_train.columns[X_train.isna().any()]

    reduced_X_train = X_train.drop(columns=columns_with_missing)
    reduced_X_valid = X_valid.drop(columns=columns_with_missing)

    drop_mae = score_dataset(
        reduced_X_train,
        reduced_X_valid,
        y_train,
        y_valid,
    )

    print(f"\nMAE — drop missing-value columns: {drop_mae:,.2f}")

    # -----------------------------------------------------------------------
    # Approach 2: Impute missing values with each column's median
    # -----------------------------------------------------------------------

    median_imputer = SimpleImputer(strategy="median")

    imputed_X_train = median_imputer.fit_transform(X_train)
    imputed_X_valid = median_imputer.transform(X_valid)

    imputation_mae = score_dataset(
        imputed_X_train,
        imputed_X_valid,
        y_train,
        y_valid,
    )

    print(f"MAE — median imputation:          {imputation_mae:,.2f}")

    # -----------------------------------------------------------------------
    # Approach 3: Impute and add missing-value indicator columns
    # -----------------------------------------------------------------------

    indicator_imputer = SimpleImputer(
        strategy="median",
        add_indicator=True,
    )

    indicator_X_train = indicator_imputer.fit_transform(X_train)
    indicator_X_valid = indicator_imputer.transform(X_valid)

    indicator_mae = score_dataset(
        indicator_X_train,
        indicator_X_valid,
        y_train,
        y_valid,
    )

    print(f"MAE — imputation + indicators:    {indicator_mae:,.2f}")

    # -----------------------------------------------------------------------
    # Prepare the full training and test sets using the chosen method
    # -----------------------------------------------------------------------
    # Example using median imputation:
    final_imputer = SimpleImputer(strategy="median")
    final_X = final_imputer.fit_transform(X)
    final_X_test = final_imputer.transform(X_test)

    print(f"\nPrepared training shape: {final_X.shape}")
    print(f"Prepared test shape:     {final_X_test.shape}")


if __name__ == "__main__":
    main()
