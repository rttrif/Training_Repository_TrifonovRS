import tensorflow as tf
import numpy as np
import matplotlib.pyplot  as plt
from tensorflow.keras.utils import plot_model

# %%
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# %%
X = np.arange(-100, 100, 4)

y = np.arange(-90, 110, 4)

y = X + 10

X_train = X[:40]
y_train = y[:40]

X_test = X[40:]
y_test = y[40:]

plt.figure(figsize=(10, 7))
plt.scatter(X_train, y_train, c='b', label='Training data')
plt.scatter(X_test, y_test, c='g', label='Testing data')
plt.legend()
plt.show()
# %%
tf.random.set_seed(42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, input_shape=[1]),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

model.summary()

plot_model(model, show_shapes=True)

model.fit(X, y, epochs=50)

# %%

y_preds = model.predict(X_test)


# %%
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_preds):
    """
    Plots training data, test data and compares predictions.
    """
    plt.figure(figsize=(10, 7))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot the predictions in red (predictions were made on the test data)
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend()


# %%
plot_predictions(predictions=y_preds)
# %%
model.evaluate(X_test, y_test)