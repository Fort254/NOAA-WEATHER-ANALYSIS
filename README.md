# NOAA WEATHER DATA ANALYSIS
A consideration of weather condition to help predict the possibility of precipitations, which inolves using various local climatological variables, including temperature, wind speed, dew point and pressure. The data was collected by a NOAA weather station located at the John F. Kennedy International Airport in Queens, New York.

# Introduction

This project relates to the NOAA Weather Dataset - JFK Airport (New York). The dataset contains 114,546 hourly observations of 12 local climatological variables (such as temperature and wind speed) collected at JFK airport. This dataset can be obtained for free from the IBM Developer [Data Asset Exchange](https://developer.ibm.com/exchanges/data/all/jfk-weather-data/).
The end goal will be to predict the precipitation using some of the available features. In this project I read data files, preprocessed data, created models, improved models and evaluated them to ultimately choose the best model.

The end goal of this project will be to predict `HOURLYprecip` (precipitation) using a few other variables. Before I did this, I first needed to preprocess the dataset.

The first step in preprocessing is to select a subset of data columns and inspect the column types.

The key columns that we will explore in this project are:

* HOURLYRelativeHumidity
* HOURLYDRYBULBTEMPF
* HOURLYPrecip(target variable)
* HOURLYWindSpeed
* HOURLYStationPressure

Data Glossary:

* 'HOURLYRelativeHumidity' is the relative humidity given to the nearest whole percentage.
* 'HOURLYDRYBULBTEMPF' is the dry-bulb temperature and is commonly used as the standard air temperature reported. It is given here in whole degrees Fahrenheit.
* 'HOURLYPrecip' is the amount of precipitation in inches to hundredths over the past hour. For certain automated stations, precipitation will be reported at sub-hourly intervals (e.g. every 15 or 20 minutes) as an accumulated amount of all precipitation within the preceding hour. A “T” indicates a trace amount of precipitation.
* 'HOURLYWindSpeed' is the speed of the wind at the time of observation given in miles per hour (mph).
* 'HOURLYStationPressure' is the atmospheric pressure observed at the station during the time of observation. Given in inches of Mercury (in Hg).

## Project Goals

* Perform data cleaning and selection of key meteorological variables.

* Conduct Exploratory Data Analysis (EDA) with visualizations.

* Build Simple and Multiple Linear Regression Models to predict precipitation.

* Evaluate models using RMSE, R-squared, and MSE.

* Implement Lasso and Ridge Regression for regularization.

* Apply Cross-Validation to ensure model generalizability.

* Conduct Residual Analysis for model diagnostics.

* Use Stepwise Model Selection (AIC) to improve model parsimony

## Tools & Libraries

* tidyverse – Data wrangling and visualization

* ggplot2 – Plotting

* glmnet – Lasso/Ridge regression

tidymodels – Unified modeling interface

caret – Model training and cross-validation

MASS – Stepwise AIC model selection

## Model Performance

Model | RMSE (Train) | RMSE (Test)
Model 1: RH only | 0.03303 | 0.03311
Model 2: RH + SP | 0.03294 | 0.03301
Model 3: RH + SP + WS | 0.03280 | 0.03290

Where:

RH = Relative Humidity
SP = Station Pressure
WS = Wind Speed

## Insights
The multiple linear regression model with RH, SP, and WS had the lowest RMSE, indicating stronger predictive capability.

Cross-validation confirmed minimal overfitting.

Stepwise AIC selection affirmed the inclusion of all three predictors.

Residual plots showed good model behavior with randomness and near-normality.

### How to Run

* Clone the repository.

* Open NOAA WEATHER FULL DATASET ANALYSIS.R in RStudio.

* Run line-by-line or as a full script.

Note: Required packages will be installed automatically if not already present.

### Author
Fortunatus Ochieng
Data Scientist | BSc Statistic

### License
This project is open-source and free to use under the MIT License.
