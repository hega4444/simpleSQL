import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    # Disables excessive log
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Disables ODNN

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
from simpleSQL import SimpleSQL

# Load data and preprocess
with SimpleSQL("salaries", drop_after=True) as db:
    db.create_table_from_csv("data/SalaryData2.csv", table_name="salaries", create_pri_key=True)
    data = list(db.into_dict(sql_block="SELECT * FROM salaries").values())

df = pd.DataFrame(data)

print(df)

# Encoding 'Gender' to numerical values
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Male: 0, Female: 1

# Standardize 'Age' for better convergence
scaler = StandardScaler()
df['Age'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))

print(df)

# Specify random values
tf.random.set_seed(42)
np.random.seed(42)

# Prepare the data
X = df[['Age', 'Gender']].astype(np.float32)  # Input features (Age and Gender)
y = df['Salary'].astype(np.float32)  # Target variable (Salary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a more complex neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=[2]),  # Input shape is now [2]
    tf.keras.layers.Dense(64, activation='relu'),  # Additional hidden layers
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1)  # Single output neuron for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Redirect stdout and stderr to /dev/null
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=1)

print(X)

# Restore stdout and stderr to their original settings
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

# Make predictions
y_pred = model.predict(X_test)

# Let's say you want to predict salary for a 30-year-old female
# Age has been standardized, and Gender is encoded (Male: 0, Female: 1)
experience_to_predict = np.array([30, 1], dtype=np.float32).reshape(1, -1)
predicted_salary = model.predict(experience_to_predict)

print('Predicted Salary for a 30-year-old female:', predicted_salary[0][0])

# Evaluate the model
mae = tf.keras.metrics.mean_absolute_error(y_test, y_pred).numpy()
mse = tf.keras.metrics.mean_squared_error(y_test, y_pred).numpy()
rmse = np.sqrt(mse)

# Calculate R-squared
mean_y_test = np.mean(y_test)
total_sum_of_squares = np.sum(np.square(y_test - mean_y_test))
residual_sum_of_squares = np.sum(np.square(y_test - y_pred.flatten()))
r_squared = 1.0 - (residual_sum_of_squares / total_sum_of_squares)

# Print indicators
print('Mean Absolute Error (MAE):', mae.mean())
print('Mean Squared Error (MSE):', mse.mean())
print('Root Mean Squared Error (RMSE):', rmse.mean())
print('R-squared:', r_squared)

# Visualize the results
# Scatter plot for Age and Salary (Gender not visualized)
plt.scatter(X_test['Age'], y_test, color='blue', label='Actual')
plt.scatter(X_test['Age'], y_pred, color='red', label='Predicted')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()
