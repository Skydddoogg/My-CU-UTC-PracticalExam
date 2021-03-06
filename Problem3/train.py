import sys
sys.path.append("../")

from config_path import data_path
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from config import BATCH_SIZE, MAX_EPOCHS
from Problem3.utils import data_tools
from Problem3.models import resnetV2, model_utils
import argparse

def augment(image, label):

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.flip_left_right(image)
    image = tf.image.adjust_saturation(image, 3)
    image = tf.image.rot90(image)
    image = tf.image.random_brightness(image, max_delta=0.5)

    return image, label

if __name__ == "__main__":

    # Argument management
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_augment", default='no')
    parser.add_argument("--model_name")
    parser.add_argument("--use_pretrained", default='no')
    args = parser.parse_args()

    model_name = args.model_name

    X_train, X_test, X_valid, y_train, y_test, y_valid = data_tools.get_splitted_data(True)
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

    # Create batch data
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

    if args.use_augment == 'yes':
        print("Add augmentation")
        train_batches = (
            train
            .map(augment, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )
    else:
        train_batches = (
            train
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE)
        )

    validation_batches = (
        val
        .batch(2 * BATCH_SIZE)
    )

    # Declare model
    if args.use_pretrained == 'yes':
        weights = 'imagenet'
    else:
        weights = None

    base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=weights)

    base_model.trainable = True

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Declare useful varibles and functions
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.BinaryAccuracy(name="accuracy")
    ]

    EARLY_STOPPING = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        verbose=0,
        patience=10,
        mode='min',
        restore_best_weights=True)

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
        epochs=100,
        validation_data=validation_batches,
    )

    # Evaluate
    results = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    for name, value in zip(model.metrics_names, results):
        print('({0}) evaluating on test set: {1} = {2:.2f}'.format(model_name, name, value))

    # Make prediction
    # y_prob = model.predict(X_test)
    # y_prob = np.reshape(y_prob, (y_prob.shape[0],))

    # y_pred = y_prob.copy()
    # y_pred[y_prob >= 0.5] = 1
    # y_pred[y_prob < 0.5] = 0

    # Save results
    model_utils.save_history(history, model_name)
    # data_tools.save_results(y_test, y_pred, y_prob, model_name)
