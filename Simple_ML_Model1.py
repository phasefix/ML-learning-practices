# decision_tree_study.py
# ML concept: Decision Tree Regression using pandas and scikit-learn

import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """
    Train a DecisionTreeRegressor with a given max_leaf_nodes value
    and return the validation MAE.

    max_leaf_nodes controls the maximum number of final leaf nodes
    in the decision tree.
    """

    model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes,
        random_state=0
    )

    model.fit(train_X, train_y)

    predictions = model.predict(val_X)

    mae = mean_absolute_error(val_y, predictions)

    return mae


# Load data
iowa_file_path = "data/train.csv"
home_data = pd.read_csv(iowa_file_path)

# Target variable
y = home_data["SalePrice"]

# Features
features = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd"
]

X = home_data[features]

# Split data
train_X, val_X, train_y, val_y = train_test_split(
    X,
    y,
    random_state=1
)

# Baseline model
baseline_model = DecisionTreeRegressor(random_state=1)
baseline_model.fit(train_X, train_y)

baseline_predictions = baseline_model.predict(val_X)
baseline_mae = mean_absolute_error(val_y, baseline_predictions)

print("Baseline validation MAE: {:,.0f}".format(baseline_mae))


# Find best max_leaf_nodes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

best_tree_size = None
lowest_mae = float("inf")

for max_leaf_nodes in candidate_max_leaf_nodes:
    my_mae = get_mae(
        max_leaf_nodes,
        train_X,
        val_X,
        train_y,
        val_y
    )

    print(
        "Max leaf nodes: %d \t\t Mean Absolute Error: %d"
        % (max_leaf_nodes, my_mae)
    )

    if my_mae < lowest_mae:
        lowest_mae = my_mae
        best_tree_size = max_leaf_nodes

print("\nBest max_leaf_nodes:", best_tree_size)
print("Lowest validation MAE:", round(lowest_mae))


# Train final model using all data
final_model = DecisionTreeRegressor(
    max_leaf_nodes=best_tree_size,
    random_state=1
)

final_model.fit(X, y)

print("\nFinal model trained successfully.")
