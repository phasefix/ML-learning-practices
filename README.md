# 1 Simple ML Model

This project is based on the Kaggle Learn tutorial:
[Random Forests Tutorial](https://www.kaggle.com/code/dansbecker/random-forests/tutorial)

The goal of this project is to practice building a simple machine learning model using `pandas` and `scikit-learn`.

## What I Learned

### Machine Learning Concepts

* The difference between training data and test data
* How to split data into training data and validation data using `scikit-learn`
* The meaning of overfitting and underfitting
* The difference between a Decision Tree and a Random Forest
* A Random Forest uses many decision trees together to make more stable predictions
* `fit()` means training the model
* `predict()` means using the trained model to make predictions
* The target is the value we want to predict, usually called `y`
* Features are the input columns used to describe or predict the target, usually called `X`
* Evaluation can be many metrics, here we used Mean Absolute Error (MAE)

### Python Libraries Used

* `pandas` is used to load, manage, and prepare the data
* `scikit-learn` is used to build, train, evaluate, and use machine learning models

## File

`Simple_ML_Model1.py`

This file contains the code for training a machine learning model and making predictions using a Random Forest model.

## Notes

This is a beginner machine learning practice project.
The main purpose is to understand the basic machine learning workflow:

1. Load data
2. Select features and target
3. Split data into training and validation sets
4. Train a model
5. Make predictions
6. Evaluate model performance

# 2 Pandas Basic Practice

This coursework is based on the Kaggle Learn Pandas exercises.

The goal of this project is to practice basic data manipulation using `pandas`.

## What I Learned

* How to create a `DataFrame`
* How to create a `Series`
* How to read a CSV file using `pd.read_csv()`
* How to use `index_col` when loading a CSV file
* How to select columns and rows using `loc` and `iloc`
* How to filter data using conditions
* How to calculate summary statistics such as median and mean
* How to count unique values using `value_counts()`
* How to create new variables from existing columns
* How to use `map()` and lambda functions
* How to find the row with the highest value using `idxmax()`

## Tools Used

* Python
* pandas
* Kaggle Notebook

## File

`Pandas_Basic_Practice.py`

This file contains practice code for loading, selecting, filtering, and summarizing data with pandas.

## Main Topics Practiced

### Creating Data

Created simple `DataFrame` and `Series` objects manually.

### Reading Data

Loaded a wine reviews CSV file using:

```python
pd.read_csv()
```

### Selecting Data

Practiced selecting rows and columns with:

```python
loc
iloc
```

### Filtering Data

Filtered wine reviews by country, points, and index values.

### Summary Statistics

Calculated values such as median points, average price, and value counts by country.

### Data Transformation

Created new variables such as centered price, bargain wine, descriptor counts, and star ratings.

## Notes

This is a beginner pandas coursework project.
The purpose is to understand how pandas is used to manage, select, filter, and analyze tabular data.


