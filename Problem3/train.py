import sys
sys.path.append("../")

from config_path import data_path
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import BATCH_SIZE, MAX_EPOCHS
from models import resnetV2
from Problem3.utils import data_tools

if __name__ == "__main__":

    X_train, X_test, X_valid, y_train, y_test, y_valid = data_tools.get_splitted_data(True)
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    # Create batch data
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
    train_batches = (
        train
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    validation_batches = (
        val
        .batch(2 * BATCH_SIZE)
    )

    # Declare model
    model = resnetV2.make_model(input_shape=input_shape)

    # Declare useful varibles and functions
    EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        verbose=0,
        patience=10,
        mode='min',
        restore_best_weights=True)

    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.BinaryAccuracy(name="acc")
    ]

    initial_learning_rate = 0.1
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.96,
        staircase=True)

    # Train
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    history = model.fit(
        train_batches,
        epochs=MAX_EPOCHS,
        validation_data=validation_batches,
        callbacks=[EARLY_STOPPING],
        verbose=1
    )