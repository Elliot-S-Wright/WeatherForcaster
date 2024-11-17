# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Dataframe output adjustments
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Import the dataset into the dataframe
df = pd.read_csv(filepath_or_buffer=r"C:\Users\ellio\OneDrive\Documents\Python Projects\weather data.csv")

# Synthetic features
df["day sin"] = np.sin(2 * np.pi * df["day of year"] / 365)
df["day cos"] = np.cos(2 * np.pi * df["day of year"] / 365)
df["temp yesterday"] = df["mean temp"].shift(1)
df["temp last 7 days"] = df["mean temp"].rolling(window=7).mean()

# Normalize features
features_to_normalize = ["day cos", "day sin", "temp yesterday", "temp last 7 days"]
scaler = MinMaxScaler()
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# Build the model
def build_model(learning_rate, input_shape):
  # Create neural network
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(units=64, activation='relu', input_shape=(input_shape,)))
  model.add(tf.keras.layers.Dense(units=32, activation='relu'))  # Add a hidden layer with 32 units
  model.add(tf.keras.layers.Dense(units=1))  # Output layer for regression

  # Configure training to minimize mean squared error
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model

# Train the model
def train_model(model, df, features, label, epochs, batch_size):
  # Feed training set to the model
  training_set = df[7:584]
  history = model.fit(x=training_set[features], y=training_set[label], batch_size=batch_size, epochs=epochs)

  # Gather weights and bias
  trained_weight = model.get_weights()[0]
  trained_bias = model.get_weights()[1]

  # Track RMSE at each epoch
  epochs = history.epoch
  hist = pd.DataFrame(history.history)
  rmse = hist["root_mean_squared_error"]

  # Output model variables
  print("\nThe learned weight for your model is: \n", trained_weight)
  print("The learned bias for your model is: \n", trained_bias )

  return trained_weight, trained_bias, epochs, rmse

# Plot the model against the test set
def test_model(trained_weight, trained_bias, features, label):
  # Labels
  plt.title("Test Set Predictions (End of 2022)")
  plt.xlabel("Day of the Year")
  plt.ylabel("Mean Temp in Fahrenheit")

  # Make predictions
  test_set = df[584:730]
  x_test = test_set[features]
  y_test = test_set[label]
  predictions = model.predict(x_test)
  
  # Create a scatter plot for the test set
  x_axis = test_set["day of year"]
  plt.scatter(x_axis, y_test, label='Actual', c='b')  # Plot actual points (use one feature for x-axis)
  plt.scatter(x_axis, predictions, label='Predicted', c='r', marker= 'x')  # Plot predictions
  
  # Render
  plt.legend()
  plt.show()

# Plot loss vs epoch
def plot_the_loss_curve(epochs, rmse):
  plt.figure()
  plt.xlabel("Epoch")
  plt.ylabel("Root Mean Squared Error")

  plt.plot(epochs, rmse, label="Loss")
  plt.legend()
  plt.ylim([rmse.min()*0.97, rmse.max()])
  plt.show()

# Predict mean temperature based on a feature
def predict_temp(features, label):
  # Make predictions
  validation_set = df[730:1095]
  predicted_values = model.predict_on_batch(x=validation_set[features])

  # Labels
  plt.xlabel("Day of Year")
  plt.ylabel("Mean Temp in Fahrenheit")
  plt.title("Validation Set Predictions (2023)")

  #Plot predictions
  days_of_year = validation_set["day of year"]
  plt.scatter(days_of_year, validation_set[label], label='Actual', c='b')
  plt.scatter(days_of_year, predicted_values, label='Predicted', c='r', marker='x')
    
  # Render
  plt.legend()
  plt.tight_layout()
  plt.show()

# Hyperparameters
learning_rate = 0.001
epochs = 600
batch_size = 64

# Algorithm variables
features = ["day cos", "day sin", "temp yesterday", "temp last 7 days"]
label = "mean temp"
input_shape = len(features)

# Discard any pre-existing model
model = None

# Invoke functions
model = build_model(learning_rate, input_shape)
weight, bias, epochs, rmse = train_model(model, df, features, label, epochs, batch_size)

test_model(weight, bias, features, label)
plot_the_loss_curve(epochs, rmse)
    
predict_temp(features, label)
print(f"RMSE: {rmse.iloc[-1]} degrees Fahrenheit")