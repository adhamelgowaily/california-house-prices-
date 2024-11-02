import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('California_Houses.csv', header = 0)

# Assume `data` is your DataFrame and `target_column` is the name of the column you're trying to predict.
# Separate the features and target variable
y = data['Median_House_Value']                 # Define the target
X = data.drop(columns=['Median_House_Value'])  # Drop the target column

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

# Second split: 15% validation, 15% test from the 30% temp
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

# Now we have:
# - X_train, y_train: 70% of the data for training
# - X_val, y_val: 15% of the data for validation
# - X_test, y_test: 15% of the data for testing

# Step 1: Initialize the Linear Regression model
linear_model = LinearRegression()

# Step 2: Fit the model to the training data
linear_model.fit(X_train, y_train)

# Step 3: Make predictions on the validation set
y_val_pred = linear_model.predict(X_val)

# Step 4: Evaluate the model using Mean Squared Error and Mean Absolute Error
linear_val_mse = mean_squared_error(y_val, y_val_pred)
linear_val_mae = mean_absolute_error(y_val, y_val_pred)

y_test_pred = linear_model.predict(X_test)

# Step 4: Evaluate the model using Mean Squared Error and Mean Absolute Error
linear_test_mse = mean_squared_error(y_test, y_test_pred)
linear_test_mae = mean_absolute_error(y_test, y_test_pred)


print("Model\tMSE(validations set)\tMAE(validation set)\tMSE(test set)\tMAE(test set)\tBest Alpha")
print("==================================================================================================")
print(f"Linear\t{linear_val_mse:.2f}\t\t{linear_val_mae:.2f}\t\t{linear_test_mse:.2f}\t{linear_test_mae:.2f}\t--")

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform it
X_train_scaled = scaler.fit_transform(X_train)

# Transform the validation and test data
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Now proceed with Lasso and Ridge regression using scaled features
alphas = np.logspace(-4, 1, 10)
lasso_mse = []
lasso_mae = []
ridge_mse = []
ridge_mae = []

for alpha in alphas:
    lasso_model = Lasso(alpha=alpha, max_iter=5000)
    lasso_model.fit(X_train_scaled, y_train)
    y_val_pred_lasso = lasso_model.predict(X_val_scaled)
    
    # Calculate errors for Lasso
    lasso_mse.append(mean_squared_error(y_val, y_val_pred_lasso))
    lasso_mae.append(mean_absolute_error(y_val, y_val_pred_lasso))

    # Ridge Regression
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train_scaled, y_train)
    y_val_pred_ridge = ridge_model.predict(X_val_scaled)
    
    # Calculate errors for Ridge
    ridge_mse.append(mean_squared_error(y_val, y_val_pred_ridge))
    ridge_mae.append(mean_absolute_error(y_val, y_val_pred_ridge))

# After this, you can analyze the results and select the best alpha

# Find the best alpha for Lasso
best_lasso_alpha = alphas[np.argmin(lasso_mse)]
lasso_val_mse = min(lasso_mse)
lasso_val_mae = lasso_mae[np.argmin(lasso_mse)]

# Find the best alpha for Ridge
best_ridge_alpha = alphas[np.argmin(ridge_mse)]
ridge_val_mse = min(ridge_mse)
ridge_val_mae = ridge_mae[np.argmin(ridge_mse)]




#Therefore using alpha min
# Train the model on the full training data (using the best alpha)
final_lasso_model = Lasso(alpha=best_lasso_alpha, max_iter=5000)
final_lasso_model.fit(X_train_scaled, y_train)

# Evaluate on test data
y_test_pred = final_lasso_model.predict(X_test_scaled)
lasso_test_mse = mean_squared_error(y_test, y_test_pred)
lasso_test_mae = mean_absolute_error(y_test, y_test_pred)




# Train the model on the full training data (using the best alpha for Ridge)
final_ridge_model = Ridge(alpha=best_ridge_alpha)
final_ridge_model.fit(X_train_scaled, y_train)

# Evaluate on test data
y_test_pred_ridge = final_ridge_model.predict(X_test_scaled)
ridge_test_mse = mean_squared_error(y_test, y_test_pred_ridge)
ridge_test_mae = mean_absolute_error(y_test, y_test_pred_ridge)


print(f"Lasso\t{lasso_val_mse:.2f}\t\t{lasso_val_mae:.2f}\t\t{lasso_test_mse:.2f}\t{lasso_test_mae:.2f}\t{best_lasso_alpha}")
print(f"Ridge\t{ridge_val_mse:.2f}\t\t{ridge_val_mae:.2f}\t\t{ridge_test_mse:.2f}\t{ridge_test_mae:.2f}\t{best_ridge_alpha}")
