import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

epochs = 15
batch_size = 64
margin = 0.8  # margin for constrastive loss.

# function for checking learning curves


def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()

# contrastive loss


def loss(margin=1):

    def contrastive_loss(y_true, y_pred):

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


def euclidean_distance(vects):
    #  Euclidean distance between two vectors
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


# Model architecture
input = layers.Input((768, 1))
x = layers.Flatten()(input)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation="relu")(x)
embedding_network = keras.Model(input, x)


input_1 = layers.Input((768, 1))
input_2 = layers.Input((768, 1))

# share weights between two towers.
tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(normal_layer)
siamese = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

# read vectorized data
if __name__ == "__main__":

    # read vectorized data

    full_x1 = []
    with open('x1.txt') as f:
        lines = f.readlines()
        for line in lines:
            row = np.fromstring(line, dtype=float, sep=',')
            full_x1.append(row)
            # print(myarray.shape)
    full_x1 = np.array(full_x1)

    full_x2 = []
    with open('x2.txt') as f:
        lines = f.readlines()
        for line in lines:
            row = np.fromstring(line, dtype=float, sep=',')
            full_x2.append(row)
            # print(myarray.shape)
    full_x2 = np.array(full_x2)

    full_y = []
    with open('y.txt') as f:
        lines = f.readlines()
        for line in lines:
            row = np.fromstring(line, dtype=float, sep=',')[0]
            full_y.append(row)
    full_y = np.array(full_y)
    print(full_y.shape)

    # data was previously shuffled.
    x_train_1, x_val_1, x_test_1 = full_x1[:5000, :].copy(
    ), full_x1[5000:5500, :].copy(), full_x1[5500:, :].copy()
    print(x_train_1.shape, x_val_1.shape, x_test_1.shape)

    x_train_2, x_val_2, x_test_2 = full_x2[:5000, :].copy(
    ), full_x2[5000:5500, :].copy(), full_x2[5500:, :].copy()
    print(x_train_2.shape, x_val_2.shape, x_test_2.shape)

    labels_train, labels_val, labels_test = full_y[:5000].copy(
    ), full_y[5000:5500].copy(), full_y[5500:].copy()
    print(labels_train.shape, labels_val.shape, labels_test.shape)

    # del full_x1; del full_x2; del full_y save some RAMs

    ##################################################################
    # intialize model

    siamese.compile(loss=loss(margin=margin), optimizer="RMSprop", metrics=["accuracy"])

    siamese.summary()

    # start training
    history = siamese.fit(
        [x_train_1, x_train_2],
        labels_train,
        validation_data=([x_val_1, x_val_2], labels_val),
        batch_size=batch_size,
        epochs=epochs,
    )
    #  plot accuracy curve
    plt_metric(history=history.history, metric="accuracy", title="Model accuracy")

    # plot constrastive loss curve
    plt_metric(history=history.history, metric="loss", title="Constrastive Loss")

    # evluation on test set
    results = siamese.evaluate([x_test_1, x_test_2], labels_test)
    print("test loss, test acc:", results)
