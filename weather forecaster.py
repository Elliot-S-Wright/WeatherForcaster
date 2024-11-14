# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Dataframe output adjustments
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# Import the dataset into the dataframe
df = pd.read_csv(filepath_or_buffer=r"C:\Users\ellio\OneDrive\Documents\Projects\weather data.csv")

# Define synthetic features
df["day cos"] = np.cos(2 * np.pi * df["day of year"] / 365)

# Build the model
def build_model(learning_rate):
  # Simple linear regression model
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

  # Configure training to minimize mean squared error
  model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate),
                loss="mean_squared_error",
                metrics=[tf.keras.metrics.RootMeanSquaredError()])

  return model

# Train the model
def train_model(model, df, feature, label, epochs, batch_size):
  print("Train the model by feeding it data.")
  training_set = df[0:584]
  history = model.fit(x=training_set[feature], y=training_set[label], batch_size=batch_size, epochs=epochs)

  # Gather weights and bias
  trained_weight = model.get_weights()[0][0]
  trained_bias = model.get_weights()[1]

  # Record epochs
  epochs = history.epoch

  # Isolate the error for each epoch.
  hist = pd.DataFrame(history.history)

  # Track RMSE at each epoch
  rmse = hist["root_mean_squared_error"]

  # Output model variables
  print("\nThe learned weight for your model is %.4f" % trained_weight)
  print("The learned bias for your model is %.4f\n" % trained_bias )

  return trained_weight, trained_bias, epochs, rmse

# Plot the model against the test set
def plot_the_model(trained_weight, trained_bias, feature, label):
  # Label the axes.
  plt.xlabel("cos(2pi * day of the year / 365)")
  plt.ylabel(label)

  # Create a scatter plot of the test set
  test_set = df[584:730]
  plt.scatter(test_set[feature], test_set[label])

  # Create a red line representing the model
  x0 = test_set[feature][584]
  y0 = trained_bias - trained_weight
  x1 = test_set[feature].max()
  y1 = trained_bias + (trained_weight * x1)
  plt.plot([x0, x1], [y0, y1], c='r')

  # Render
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
def predict_temp(feature, label):
  validation_set = df[730:1095]
  validation_feature = validation_set[feature]
  predicted_values = model.predict_on_batch(x=validation_feature)

  # Output predictions
  print("date   predicted      actual")
  print("            temp        temp")
  print("--------------------------------")
  for i in range(365):
    print("%2s-%2s-%4s %4.0f %10.0f" % (validation_set["MONTH"][730+i], validation_set["DAY"][730+i], validation_set["YEAR"][730+i], predicted_values[i][0], validation_set[label][730 + i]))

# Hyperparameters
learning_rate = 0.01
epochs = 1400
batch_size = 100

# Algorithm variables
feature = "day cos"
label="mean temp"

# Discard any pre-existing model
model = None

# Invoke functions
model = build_model(learning_rate)
weight, bias, epochs, rmse = train_model(model, df, feature, label, epochs, batch_size)

plot_the_model(weight, bias, feature, label)
plot_the_loss_curve(epochs, rmse)
    
predict_temp(feature, label)