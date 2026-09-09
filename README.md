# california-housing-price-prediction
California Housing price prediction using data analysis and machine learning.
# California Housing Price Prediction

Machine Learning project focused on analyzing and predicting median house values using the California Housing dataset.

## Project Overview

The goal of this project is to explore the dataset, identify relationships between variables, clean anomalous observations, and compare different Machine Learning
models for predicting house values.

The project follows a complete Machine Learning workflow:

- Exploratory Data Analysis
- Data Cleaning
- Feature and Target preparation
- Train/Test split
- Model training
- Model evaluation
- Feature importance analysis
- Model comparison

## Dataset

The dataset contains information about California housing districts.

Each observation represents a census block group rather than an individual house.

Main features:

| Feature | Description |
|---|---|
| MedInc | Median income |
| HouseAge | Median house age |
| AveRooms | Average number of rooms |
| AveBedrms | Average number of bedrooms |
| Population | Population of the block group |
| AveOccup | Average household occupancy |
| Latitude | Geographic latitude |
| Longitude | Geographic longitude |
| MedHouseVal | Median house value |

The target variable is `MedHouseVal`.

The target is expressed in units of $100,000.

## Exploratory Data Analysis

The analysis focused on:

- Dataset structure and dimensions
- Descriptive statistics
- Missing values
- Duplicate observations
- Target distribution
- Relationship between income and house value
- Correlation between numerical variables
- Geographic distribution of house values

One of the strongest relationships identified was between median income (`MedInc`) and median house value (`MedHouseVal`).

Geographic variables also showed meaningful relationships with house values, highlighting the importance of location in the dataset.

## Data Cleaning

The dataset was checked for:

- Missing values
- Duplicate rows
- Extreme observations

A small number of extremely anomalous observations were identified in `AveOccup`.

Observations with:

`AveOccup > 50`

were removed.

Only 7 observations were removed from the original 20,640 rows.

The cleaned dataset was saved as:

`housing_clean.csv`

## Machine Learning

Three regression models were trained and evaluated:

1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor

The dataset was divided into:

- 80% training data
- 20% test data

A fixed `random_state=42` was used to make the results reproducible.

## Model Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 0.5096 | 0.7137 | 0.6180 |
| Random Forest | 0.3301 | 0.5079 | 0.8066 |
| Gradient Boosting | 0.3728 | 0.5409 | 0.7806 |

The Random Forest achieved the best overall performance.

### Best Model: Random Forest

The Random Forest achieved:

- **MAE:** 0.3301
- **RMSE:** 0.5079
- **R²:** 0.8066

Since the target is expressed in units of $100,000, the MAE corresponds to an average absolute error of approximately **$33,000**.

The model explains approximately **80.7% of the variance** in the target variable on the test set.

## Feature Importance

The Random Forest identified `MedInc` as the most important feature.

The main feature importances were:

| Feature | Importance |
|---|---:|
| MedInc | 0.5205 |
| AveOccup | 0.1385 |
| Longitude | 0.0917 |
| Latitude | 0.0914 |
| HouseAge | 0.0542 |

Feature importance represents the contribution of each feature to the model's predictions. It should not be interpreted as proof of causation.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Project Structure

```text
california-housing-price-prediction/
│
├── analisi.py
├── housing.csv
├── housing_clean.csv
└── README.md

# How to Run

Install the required libraries:

`pip install pandas numpy matplotlib seaborn scikit-learn`

Then run:

`python3 analisi.py`

The script performs the complete analysis, cleaning process, model training and evaluation.

# Conclusions

The analysis shows that non-linear ensemble models significantly outperform the Linear Regression baseline on this dataset.

Among the tested models, Random Forest achieved the best performance, with an R² of approximately 0.81.

The project demonstrates a complete introductory Machine Learning pipeline, from raw data exploration and cleaning to model comparison and interpretation.
