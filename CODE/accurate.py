import pandas as pd
import numpy as np
import joblib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load training data
data_train = pd.read_excel('training-dataset-ann-neuro-fuzzy.xlsx')

# Separate features and target variable
X_train = data_train[['N_HMAX', 'N_MMAX', 'N_LMIN', 'N_SLP', 'N_RH']]
y_train = data_train['Precipitation']

# Convert data to numpy arrays
X_train_np = X_train.values
y_train_np = y_train.values

# Define fuzzy variables and membership functions
N_HMAX = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'N_HMAX')
N_MMAX = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'N_MMAX')
N_LMIN = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'N_LMIN')
N_SLP = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'N_SLP')
N_RH = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'N_RH')

Precipitation = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'Precipitation')

# Define fuzzy membership functions
N_HMAX['low'] = fuzz.trimf(N_HMAX.universe, [0, 0, 0.5])
N_HMAX['medium'] = fuzz.trimf(N_HMAX.universe, [0, 0.5, 1])
N_HMAX['high'] = fuzz.trimf(N_HMAX.universe, [0.5, 1, 1])

N_MMAX['low'] = fuzz.trimf(N_MMAX.universe, [0, 0, 0.5])
N_MMAX['medium'] = fuzz.trimf(N_MMAX.universe, [0, 0.5, 1])
N_MMAX['high'] = fuzz.trimf(N_MMAX.universe, [0.5, 1, 1])

N_LMIN['low'] = fuzz.trimf(N_LMIN.universe, [0, 0, 0.5])
N_LMIN['medium'] = fuzz.trimf(N_LMIN.universe, [0, 0.5, 1])
N_LMIN['high'] = fuzz.trimf(N_LMIN.universe, [0.5, 1, 1])

N_SLP['low'] = fuzz.trimf(N_SLP.universe, [0, 0, 0.5])
N_SLP['medium'] = fuzz.trimf(N_SLP.universe, [0, 0.5, 1])
N_SLP['high'] = fuzz.trimf(N_SLP.universe, [0.5, 1, 1])

N_RH['low'] = fuzz.trimf(N_RH.universe, [0, 0, 0.5])
N_RH['medium'] = fuzz.trimf(N_RH.universe, [0, 0.5, 1])
N_RH['high'] = fuzz.trimf(N_RH.universe, [0.5, 1, 1])

# Define membership functions for Precipitation
Precipitation['low'] = fuzz.trimf(Precipitation.universe, [0, 0, 0.5])
Precipitation['medium'] = fuzz.trimf(Precipitation.universe, [0, 0.5, 1])
Precipitation['high'] = fuzz.trimf(Precipitation.universe, [0.5, 1, 1])

# Define fuzzy rules
rule1 = ctrl.Rule(N_HMAX['low'] & N_MMAX['low'] & N_LMIN['low'] & N_SLP['low'] & N_RH['low'], Precipitation['low'])
rule2 = ctrl.Rule(N_HMAX['medium'] & N_MMAX['medium'] & N_LMIN['medium'] & N_SLP['medium'] & N_RH['medium'], Precipitation['medium'])
rule3 = ctrl.Rule(N_HMAX['high'] & N_MMAX['high'] & N_LMIN['high'] & N_SLP['high'] & N_RH['high'], Precipitation['high'])

# Create control system and simulation
fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
fuzzy_sim = ctrl.ControlSystemSimulation(fuzzy_ctrl)

# Compute fuzzy outputs for the training set
def compute_fuzzy_outputs(X):
    fuzzy_outputs = []
    for row in X:
        fuzzy_sim.input['N_HMAX'] = row[0]
        fuzzy_sim.input['N_MMAX'] = row[1]
        fuzzy_sim.input['N_LMIN'] = row[2]
        fuzzy_sim.input['N_SLP'] = row[3]
        fuzzy_sim.input['N_RH'] = row[4]
        fuzzy_sim.compute()
        fuzzy_outputs.append([fuzzy_sim.output['Precipitation']])
    return np.array(fuzzy_outputs)

fuzzy_outputs = compute_fuzzy_outputs(X_train_np)

# Combine fuzzy outputs with original features
X_train_combined = np.hstack((X_train_np, fuzzy_outputs))

# Define and train the neural network
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create and train the neural network with combined features
nn_model = create_nn_model(X_train_combined.shape[1])
nn_model.fit(X_train_combined, y_train_np, epochs=3500, batch_size=10)

# Save the trained model to a .pkl file
joblib.dump(nn_model, 'delta_rule_model.pkl')

print('Model training completed and saved to delta_rule_model.pkl')
