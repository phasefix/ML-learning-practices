"""Learning cross-validation.

When the dataset is small, cross-validation can give a more reliable
evaluation than using only one validation set.

With cross-validation, every row has a chance to be used for validation.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline


# Read the data
train_data = pd.read_csv("../input/train.csv", index_col="Id")
test_data = pd.read_csv("../input/test.csv", index_col="Id")


# Remove rows with missing target
train_data.dropna(
    axis=0,
    subset=["SalePrice"],
    inplace=True
)


# Separate target from predictors
y = train_data["SalePrice"]
train_data.drop(
    columns=["SalePrice"],
    inplace=True
)


# Select numerical columns only
numeric_cols = [
    column
    for column in train_data.columns
    if train_data[column].dtype in ["int64", "float64"]
]

X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()


# Define the function
def get_score(n_estimators):
    """Return the average MAE over 3 cross-validation folds."""

    my_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                SimpleImputer()
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=0
                )
            )
        ]
    )

    scores = -1 * cross_val_score(
        my_pipeline,
        X,
        y,
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    return scores.mean()


# Compare different numbers of trees
results = {}

for number_of_trees in [50, 100, 150, 200, 250, 300, 350, 400]:
    results[number_of_trees] = get_score(number_of_trees)

print(results)


# Plot the results
plt.plot(
    list(results.keys()),
    list(results.values()),
    marker="o"
)

plt.xlabel("Number of trees")
plt.ylabel("Average cross-validation MAE")
plt.title("Random Forest cross-validation results")
plt.show()
