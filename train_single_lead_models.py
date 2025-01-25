# Standard libraries

import math
import os
import sys

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'


# Third-party libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Keras
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Conv1D, Dropout, Activation, Flatten
from keras.metrics import AUC
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import Sequence

# TensorFlow
import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model

# Local imports
from models import Attia_et_al_CNN

# Environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from utils import split_train_val_test, load_X_y

# OUTCOME = 'AF'
BATCH_SIZE = 32

outcomes = ['SB', 'AF','ST', 'TWC', 'TWO', 'SA', 'AFIB', 
            'STDD', 'APB', 'STTC', '1AVB', 'AQW',
            'LVQRSAL', 'ARS', 'STE', 'IDC', 'SVT',
            'RBBB', 'LVH']

for OUTCOME in outcomes:

    lead_labels = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


    def generator(X, y, batch_size=8):
        row_nums = np.arange(X.shape[0])
        np.random.shuffle(row_nums)
        for i in range(0, len(row_nums), batch_size):
            current_idxs = row_nums[i:i+batch_size]

            yield X[current_idxs], y[current_idxs].reshape(-1)


    for LEAD_NAME in lead_labels:
        # load diagnostic_data.pickle
        X,y = load_X_y(outcome=OUTCOME, lead_name=LEAD_NAME)

        model = Attia_et_al_CNN().build(input_shape=(5000, 1))

        X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(X, y, train_size=0.7, val_size=0.15)
        del X, y

            
        output_signature = (
            tf.TensorSpec(shape=(None, 5000, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )

        train_ds = tf.data.Dataset.from_generator(generator=lambda: generator(X_train,y_train, BATCH_SIZE), output_signature=output_signature)
        val_ds = tf.data.Dataset.from_generator(generator=lambda: generator(X_val,y_val, BATCH_SIZE), output_signature=output_signature)

        learning_rate =1e-3
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # Monitor validation loss
            factor=0.5,  # Reduce learning rate by half when triggered
            patience=3,  # Number of epochs with no improvement to trigger the callback
            verbose=1,  # Print messages
            min_lr=1e-8  # Minimum learning rate
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=6, mode='min', restore_best_weights=True)
        model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate), metrics=['accuracy', AUC(name='auc')])
        # Training parameters
        EPOCHS = 100  # You can adjust based on your needs

        history = model.fit(train_ds,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=val_ds,
        callbacks=[reduce_lr, early_stopping],
        verbose=1)

        save_dir = f'models/single-lead/{OUTCOME}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save(f'{save_dir}/cnn_{OUTCOME}_{LEAD_NAME}.keras')
