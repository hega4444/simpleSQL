import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'    #Disables excessive log
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'   # Disables ODNN (something related to the round-off operations, was causing some differences in results)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from simpleSQL import SimpleSQL


# Load data and preprocess
with SimpleSQL("salaries", drop_after=True) as db:
    db.create_table_from_csv("data/SalaryData.csv", table_name="salaries", create_pri_key=True)
    data = list(db.into_dict(sql_block="SELECT * FROM salaries").values())

df = pd.DataFrame(data)
print(df)

#Specify random values
tf.random.set_seed(42)
np.random.seed(42)

# Prepare the data
X = df[['Experience']].astype(np.float32)  # Input feature (Experience)
y = df['Salary'].astype(np.float32)  # Target variable (Salary)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=[1])
])

# Compile the modelR-squared: 0.5796248316764832
model.compile(optimizer='adam', loss='mean_squared_error')

# Redirect stdout and stderr to /dev/null
sys.stdout = open(os.devnull, 'w')
sys.stderr = open(os.devnull, 'w')


# Train the model
model.fit(X_train, y_train, epochs=50, verbose=1)

# Make predictions
y_pred = model.predict(X_test)

# Let's say you want to predict salary for an experience of 20 months
experience_to_predict = np.array([20], dtype=np.float32)  # Ensure it's in float32
predicted_salary = model.predict(experience_to_predict.reshape(-1, 1))

# Restore stdout and stderr to their original settings
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

print('Predicted Salary for 20 months of experience:', predicted_salary[0][0])

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

# ...


# Visualize the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
