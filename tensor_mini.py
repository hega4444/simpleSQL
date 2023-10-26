import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

# Create a toy dataset
data = pd.DataFrame({'Education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'PhD']})

# Apply one-hot encoding
encoder = OneHotEncoder(sparse=False)
X = encoder.fit_transform(data[['Education']])
encoded_data = pd.DataFrame(X, columns=encoder.get_feature_names_out(['Education']))

# Create the target variable (in this case, we'll generate random values)
y = np.random.rand(len(encoded_data))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(encoded_data, y, test_size=0.2, random_state=42)

# Define a simple neural network model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(encoded_data.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test)

print("Test Data:")
print(X)
print("Predictions:")
print(y_pred)
