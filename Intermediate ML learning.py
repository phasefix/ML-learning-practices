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


def score_dataset(X_train, X_valid, y_train, y_valid):
    """Train a model and return the validation MAE."""

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_valid)

    return mean_absolute_error(y_valid, predictions)


def main():
    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------

    train_data = pd.read_csv(TRAIN_PATH, index_col="Id")
    test_data = pd.read_csv(TEST_PATH, index_col="Id")

    # Remove rows where the target is missing.
    train_data = train_data.dropna(subset=[TARGET])

    # Separate target from predictors.
    y = train_data[TARGET]
    X_full = train_data.drop(columns=[TARGET])

    # Keep only numerical columns.
    X = X_full.select_dtypes(exclude="object").copy()

    # Ensure that test data uses exactly the same columns.
    X_test = test_data[X.columns].copy()

    # -----------------------------------------------------------------------
    # 2. Split training and validation data
    # -----------------------------------------------------------------------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=0.8,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_valid.shape)
    print("Test shape:", X_test.shape)

    # -----------------------------------------------------------------------
    # 3. Inspect missing values
    # -----------------------------------------------------------------------

    missing_counts = X_train.isna().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(
        ascending=False
    )

    print("\nMissing values by column:")
    print(
        missing_counts
        if not missing_counts.empty
        else "No missing values."
    )

    # -----------------------------------------------------------------------
    # 4. Approach 1: Drop columns with missing values
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

    print(f"\nMAE — drop columns:              {drop_mae:,.2f}")

    # -----------------------------------------------------------------------
    # 5. Approach 2: Median imputation
    # -----------------------------------------------------------------------

    median_imputer = SimpleImputer(strategy="median")

    imputed_X_train = pd.DataFrame(
        median_imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    imputed_X_valid = pd.DataFrame(
        median_imputer.transform(X_valid),
        columns=X_valid.columns,
        index=X_valid.index,
    )

    imputation_mae = score_dataset(
        imputed_X_train,
        imputed_X_valid,
        y_train,
        y_valid,
    )

    print(f"MAE — median imputation:         {imputation_mae:,.2f}")

    # -----------------------------------------------------------------------
    # 6. Approach 3: Imputation with missing indicators
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

    print(f"MAE — imputation + indicators:   {indicator_mae:,.2f}")

    # -----------------------------------------------------------------------
    # 7. Preprocess full training data and test data
    # -----------------------------------------------------------------------

    final_imputer = SimpleImputer(strategy="median")

    # Learn median values from the full training data.
    final_X = pd.DataFrame(
        final_imputer.fit_transform(X),
        columns=X.columns,
        index=X.index,
    )

    # Apply the same median values to the test data.
    final_X_test = pd.DataFrame(
        final_imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # Check preprocessing.
    assert final_X_test.isna().sum().sum() == 0
    assert len(final_X_test) == len(X_test)

    print("\nMissing values in final_X_test:")
    print(final_X_test.isna().sum().sum())

    print("Final training shape:", final_X.shape)
    print("Final test shape:", final_X_test.shape)

    # -----------------------------------------------------------------------
    # 8. Train final model using all training data
    # -----------------------------------------------------------------------

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(final_X, y)

    # -----------------------------------------------------------------------
    # 9. Generate test predictions
    # -----------------------------------------------------------------------

    preds_test = model.predict(final_X_test)

    print("Number of predictions:", len(preds_test))

    # -----------------------------------------------------------------------
    # 10. Create submission file
    # -----------------------------------------------------------------------

    submission = pd.DataFrame(
        {
            "Id": X_test.index,
            "SalePrice": preds_test,
        }
    )

    submission.to_csv("submission.csv", index=False)

    print("\nCreated submission.csv")
    print(submission.head())


if __name__ == "__main__":
    main()
