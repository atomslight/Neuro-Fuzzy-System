import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the test data
data_test = pd.read_excel('testing-new.xlsx')

# Separate features and target variable (for evaluation purposes)
X_test = data_test[['N_HMAX', 'N_MMAX', 'N_LMIN', 'N_SLP', 'N_RH']]
y_test = data_test['Precipitation']

# Normalize the features and target (using the same scaler as in training)
scaler_X = MinMaxScaler()
X_test_normalized = scaler_X.fit_transform(X_test)

scaler_y = MinMaxScaler()
y_test_normalized = scaler_y.fit_transform(y_test.values.reshape(-1, 1)).flatten()

# Convert data to numpy arrays
X_test_np = X_test_normalized
y_test_np = y_test_normalized

# Load the trained model
delta_rule_nn = joblib.load('delta_rule_model.pkl')

# Make predictions
y_pred_normalized = delta_rule_nn.predict(X_test_np)

# Denormalize the predictions
y_pred = scaler_y.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()

# Save the predictions to a CSV file
results = pd.DataFrame({
    'Actual_current_yr_Precipitation': y_test,
    'Predicted_current_yr_Precipitation': y_pred
})

results.to_csv('predictions.csv', index=False)

print('Testing completed and results saved to predictions.csv')
