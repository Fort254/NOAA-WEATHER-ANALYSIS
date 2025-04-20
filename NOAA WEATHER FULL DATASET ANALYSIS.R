#Loading the necessary packages and libraries
install.packages("tidyverse")
library(tidyverse)
install.packages("glmnet")
library(glmnet)
# Load parsnip directly
library(parsnip)
# Load tidymodels
library(tidymodels)
install.packages("glmnet")
library(glmnet)
# Load parsnip directly
library(parsnip)

url<-"https://dax-cdn.cdn.appdomain.cloud/dax-noaa-weather-data-jfk-airport/1.1.4/noaa-weather-data-jfk-airport.tar.gz"
download.file(url,destfile = "noaa-weather-data-jfk-airport.tar.gz")
untar("noaa-weather-data-jfk-airport.tar.gz",exdir = "extracted_file")

#Extracting and reading into the project
extracted_files <- list.files("extracted_file", full.names = TRUE)
noaa_sample_data <- read.csv("~/extracted_file/noaa-weather-data-jfk-airport/jfk_weather_cleaned.csv")

#Display the first few rows of the data
head(noaa_sample_data,10)

key_variables<-noaa_sample_data%>%select(HOURLYRelativeHumidity,HOURLYDRYBULBTEMPF,HOURLYPrecip,HOURLYWindSpeed,HOURLYStationPressure)

#Renaming the columns to meaningful column names and storing the new data frame
key_variables<-key_variables%>%rename(relative_humidity=HOURLYRelativeHumidity,dry_bulb_temp_f=HOURLYDRYBULBTEMPF,
                                      precip=HOURLYPrecip,wind_speed=HOURLYWindSpeed,station_pressure=HOURLYStationPressure)

#checking unique row entries for the column HOURLYPrecip
unique(key_variables$HOURLYPrecip)
typeof(key_variables$HOURLYPrecip)
#checking for count of missing values `NA` for each column
sum(is.na(key_variables$precip))
sum(is.na(key_variables$relative_humidity))
sum(is.na(key_variables$dry_bulb_temp_f))
sum(is.na(key_variables$wind_speed))
sum(is.na(key_variables$station_pressure))

#EXPLORATORY DATA ANALYSIS
#splitting the data into a training and testing set
#set seed for reproducibility
set.seed(1234)
#Get total number of rows
n <- nrow(key_variables)
#randomly select 80 percent of the data to be used for training
train_indices <- sample(1:n, size = 0.8 * n)  # Randomly select 80% indices

train_set <- key_variables[train_indices, ]  # Training data
test_set <- key_variables[-train_indices, ]

#CORRELATION ANALYSIS
cor_matrix <- cor(train_set)
print(cor_matrix) #simply prints  correlation significance
install.packages("Hmisc") #For rcorr use, must install this package
library(Hmisc)
rcorr(as.matrix(train_set)) # prints correlation significance(-1,1) and the p-values(significant at p<0.05)

#HISTOGRAMS AND BOXPLOTS OF VARIABLES FOR INITIAL LOOK AT THEIR DISTRIBUTIONS
# Histogram and boxplot for precip
ggplot(train_set, aes(x = precip)) +
  geom_histogram(binwidth = 0.1, fill = "green", color = "black") +
  labs(title = "Distribution of Precipitation",
       x = "Precipitation",
       y = "Frequency") +
  theme_minimal()
ggplot(train_set,aes(y=precip))+
  geom_boxplot(fill="steelblue",color="black")+
  labs(title="Boxplot of Precip",
       y="precip")+
  theme_minimal()

#Histogram and boxplot for relative_humidity
ggplot(train_set,aes(x=relative_humidity))+
  geom_histogram(binwidth=10,fill="blue",color="black")+
  labs(title="Histogram of Relative Humidity",
       x="Relative Humidity",
       y="Frequency")+
  theme_minimal()
ggplot(train_set,aes(x=relative_humidity))+
  geom_boxplot(fill="blue",color="black")+
  labs(title="Boxplot of Relative Humidity",
       x="Relative Humidity",
       y="Frequency")+
  theme_minimal()

#Histogram and boxplot of wind_speed
ggplot(train_set, aes(x=wind_speed))+
  geom_histogram(binwidth=5,fill="maroon",color="black")+
  labs(title="Histogram of wind_speed",
       x="wind_speed",
       y="frequency")
ggplot(train_set,aes(y=wind_speed))+
  geom_boxplot(fill="maroon",color="black")+
  labs(title="Boxplot of Wind_speed",
       y="wind_speed")+
  theme_minimal()

#SIMPLE LINEAR REGRESSION MODELS(Precip as the target/response variable and relative_humidity,wind_speed,dry_bulb_temp_f and station_pressure as predictor variables)
#SCATTER PLOTS 
#Model 1: precip ~ relative_humidity AND scatter plot
model_relative_humidity <- lm(precip ~ relative_humidity, data = train_set)
summary(model_relative_humidity)

ggplot(data=train_set,aes(x=relative_humidity,y=precip))+
  geom_point(alpha=2,color="green",size=3)+
  geom_smooth(method="lm",se = TRUE,color="black")+
  labs(title="Scatter plot of precip vs relative_humidity",
       x="relative_humidity",
       y="precip")

#Model 2: precip ~ dry_bulb_temp_f AND scatter plot(Although we have dropped this predictor variable)
model_dry_bulb_temp_f <- lm(precip ~ dry_bulb_temp_f, data = train_set)
summary(model_dry_bulb_temp_f)

ggplot(data=train_set,aes(x = dry_bulb_temp_f,y=precip))+
  geom_point(alpha = 1, color= "yellow", size=3)+
  geom_smooth(method="lm",se = TRUE,color="blue")+
  labs(title="Scatter plot of Precip Vs dry_bulb_temp_f linear Model",
       x="dry_bulb_temp_f",
       y="precip")+
  theme_minimal()

#Model 3: precip ~ wind_speed AND scatter plot
model_wind_speed <- lm(precip ~ wind_speed, data = train_set)
summary(model_wind_speed)

ggplot(data=train_set , aes(x=wind_speed,y = precip ))+
  geom_point(alpha=1,color="blue",size=3)+
  geom_smooth(method="lm", se = TRUE,color="yellow")+
  labs(title="Scatter plot of Precip vs Wind_speed",
       x="wind_speed",
       y="precip")+
  theme_minimal()

#Model 4: precip ~ station_pressure AND scatter plot
model_station_pressure <- lm(precip ~ station_pressure, data = train_set)
summary(model_station_pressure)

ggplot(data=train_set,aes(x=station_pressure,y=precip))+
  geom_point(alpha=1,color="yellow",size=3)+
  geom_smooth(method="lm",se= TRUE,color="blue")+
  labs(title="scatter plot of precip vs station_pressure",
       x="station_pressure",
       y="precip")+
  theme_minimal()

#MAIN MODELS
#a simple linear regression model of precip on relative_humidity that prints its r-squared
train_set_Model_1<- lm(precip ~ relative_humidity, data = train_set)
summary(train_set_Model_1)
r_squared <- summary(train_set_Model_1)$r.squared
print(paste("R-squared:", r_squared))

#a multiple linear regression model of precip on relative_humidity + station_pressure that prints its r-squared
train_set_Model_2<- lm(precip ~ relative_humidity+ station_pressure , data = train_set)
summary(train_set_Model_2)
r_squared <- summary(train_set_Model_2)$r.squared
print(paste("R-squared:", r_squared))

#a multiple linear regression model of precip on r_h + s_p + w_s that prints its r-squared
train_set_Model_3<- lm(precip ~ relative_humidity+ station_pressure + wind_speed, data = train_set)
summary(train_set_Model_3)
r_squared <- summary(train_set_Model_3)$r.squared
print(paste("R-squared:", r_squared))

#Training the three main models on the training set
train_pred_1 <- predict(train_set_Model_1, train_set)
train_pred_2 <- predict(train_set_Model_2, train_set)
train_pred_3 <- predict(train_set_Model_3, train_set)

#calculate MSE for the three models on the training set
#Predict on the training set
train_pred_1 <- predict(train_set_Model_1, newdata = train_set)
mse <- mean((train_set$precip - train_pred_1)^2)
print(paste("MSE:", mse))
rmse<-sqrt(mse)
print(paste("rmse:",rmse))

train_pred_2 <- predict(train_set_Model_2, newdata = train_set)
mse <- mean((train_set$precip - train_pred_2)^2)
print(paste("MSE:", mse))
rmse<-sqrt(mse)
print(paste("rmse:",rmse))

train_pred_3 <- predict(train_set_Model_3, newdata = train_set)
mse <- mean((train_set$precip - train_pred_3)^2)
print(paste("MSE:", mse))
rmse<-sqrt(mse)
print(paste("rmse:",rmse))

# Predict the three models on the testing set
test_pred_1 <- predict(train_set_Model_1, newdata=test_set)
test_pred_2 <- predict(train_set_Model_2, newdata=test_set)
test_pred_3 <- predict(train_set_Model_3, newdata=test_set)

#calculate MSE and RMSE for the three models on the test set
#Predict on the test set
#the `newdata` equated to the test set is important
test_pred_1 <- predict(train_set_Model_1, newdata=test_set)
mse <- mean((test_set$precip - test_pred_1)^2)
print(paste("MSE:", mse))
rmse<-sqrt(mse)
print(paste("rmse:",rmse))

test_pred_2 <- predict(train_set_Model_2, newdata=test_set)
mse <- mean((test_set$precip - test_pred_2)^2)
print(paste("MSE:", mse))
rmse<-sqrt(mse)
print(paste("rmse:",rmse))

test_pred_3 <- predict(train_set_Model_3, newdata=test_set)
mse <- mean((test_set$precip - test_pred_3)^2)
print(paste("MSE:", mse))
rmse<-sqrt(mse)
print(paste("rmse:",rmse))

model_names <- c("Model_1", "Model_2", "Model_3")
train_error_rmse <- c("0.03303", "0.03294", "0.03280")
test_error_rmse <- c("0.03311", "0.03301", "0.03290")
comparison_df <- data.frame(model_names, train_error_rmse, test_error_rmse)
print(comparison_df)
#Root Mean Squared Error tells us the average error a model makes in predicting the target 
#(in this case, precipitation), in the same units as the target.
range(key_variables$precip)#our target ranges from 0.00 to 2.41 so this model is off by about 0.033 units indicating strong prediction

#Both of these techniques would be necessary if predictors were highly correlated(ridge regression) or
#feature selection(too many irrelevant predictors) as in lasso regression
library(glmnet)
# Convert predictors to a matrix (exclude intercept)
x_train <- model.matrix(precip ~ relative_humidity + station_pressure + wind_speed, data = train_set)[, -1]
# Target variable
y_train <- train_set$precip
# Cross-validate to find optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
# Get best lambda value
best_lambda_ridge <- cv_ridge$lambda.min
# Predict using the best lambda
ridge_pred <- predict(cv_ridge, newx = x_train, s = "lambda.min")
# Calculate Ridge MSE
ridge_mse <- mean((y_train - ridge_pred)^2)
print(paste("Ridge MSE:", ridge_mse))

# Cross-validate to find optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)
# Get best lambda value
best_lambda_lasso <- cv_lasso$lambda.min
# Predict using the best lambda
lasso_pred <- predict(cv_lasso, newx = x_train, s = "lambda.min")
# Calculate Lasso MSE
lasso_mse <- mean((y_train - lasso_pred)^2)
print(paste("Lasso MSE:", lasso_mse))

#OR ALTERNATIVELY

ridge_pred <- predict(train_set_Model_3, newdata = train_set)
ridge_mse <- mean((train_set$precip - ridge_pred)^2)
print(paste("MSE:", ridge_mse))

lasso_pred <- predict(train_set_Model_3, newdata = train_set)
lasso_mse <- mean((train_set$precip - lasso_pred)^2)
print(paste("MSE:",lasso_mse))

#CROSS-VALIDATION
library(caret)
# Set up training control for 5-fold cross-validation
train_control <- trainControl(method = "cv", number = 5)
# Fit your model (e.g., linear regression or random forest)
# Replace 'precip ~ .' with your actual formula
model_cv <- train(
  precip ~ wind_speed + relative_humidity + station_pressure , 
  data = train_set, 
  method = "lm",  # or "rf", "xgbTree", etc.
  trControl = train_control,
  metric = "RMSE"
)
print(model_cv)#print model performance

# Predict on the test set
predictions <- predict(model_cv, newdata = test_set)
# Compare predicted vs actual and compute RMSE
actuals <- test_set$precip
# Calculate RMSE
rmse_test <- sqrt(mean((predictions - actuals)^2))
print(rmse_test)#you will notice that cross validation rmse is approximately equal to the normal rmse which means that
#our model generalizes well,there is no overfitting or underfitting and performance is stable across different data splits

# Predict on the test set
predictions <- predict(model_cv, newdata = test_set)
# Actual values
actuals <- test_set$precip  # Replace "precip" with your actual target variable if different
# Plot
plot(actuals, predictions,
     xlab = "Actual Precipitation",
     ylab = "Predicted Precipitation",
     main = "Actual vs Predicted Precipitation",
     col = "dodgerblue", pch = 10)
abline(0, 1, col = "red", lwd = 2)  # perfect prediction line
#A tight scatter around the red line = accurate and unbiased predictions.

#RESIDUAL ANALYSIS
# Get residuals
residuals <- actuals - predictions
# Plot residuals vs predicted
plot(predictions, residuals,
     xlab = "Predicted Precipitation",
     ylab = "Residuals",
     main = "Residuals vs Predicted",
     col = "purple", pch = 19)
abline(h = 0, col = "red", lwd = 2)
#we expect the points to be randomly scattered around 0
hist(residuals, breaks = 20, col = "skyblue",
     main = "Histogram of Residuals", xlab = "Residuals") #a check for residual normality

#Feature coeffecients
summary(model_cv$finalModel)

#STEPWISE MODEL SELECTION BASED ON AKAIKE INFORMATION CRITERION(AIC)
install.packages("MASS")
library(MASS)
step_model <- stepAIC(model_cv$finalModel, direction = "both")
summary(step_model)
#The model currently includes all three variables, and has the lowest (best) AIC at -410688.
#Removing any of the predictors increases the AIC, which means the model gets worse.
#Therefore, stepAIC() decides not to drop any variable — that’s why <none> is selected.

