# Simple ML Model 1

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



