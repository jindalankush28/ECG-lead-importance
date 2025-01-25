import tensorflow as tf
from keras import backend as K
from keras.layers import (Input, Dense, Conv1D, Dropout, MaxPooling1D, 
                          Activation, Lambda, BatchNormalization, Add,
                          Flatten, Attention)
from keras.optimizers import Adam
from keras.models import Model
from keras.metrics import AUC
from keras.models import Model, Sequential

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Conv2D, MaxPooling2D, \
    ReLU, Reshape, GlobalAveragePooling1D, Dense, Concatenate, Dropout, concatenate, LeakyReLU, SpatialDropout1D, Attention
import logging

# PAPER: Screening for cardiac contractile dysfunction using an artificial intelligence–enabled electrocardiogram
#        https://www.nature.com/articles/s41591-018-0240-2
# SOURCE REPO: https://github.com/chrisby/DeepCardiology
class Attia_et_al_CNN():
    def __init__(self, 
                 filter_numbers=[16, 16, 32, 32, 64, 64], 
                 kernel_widths=[5, 5, 5, 3, 3, 3], 
                 pool_sizes=[2, 2, 4, 2, 2, 4], 
                 spatial_num_filters=64, 
                 dense_dropout_rate=0.2, 
                 spatial_dropout_rate=0.2,
                 dense_units=[64, 32], 
                 use_spatial_layer=False,
                 verbose=1,
                 output_size=1):

        self.filter_numbers = filter_numbers
        self.kernel_widths = kernel_widths
        self.pool_sizes = pool_sizes
        self.spatial_num_filters = spatial_num_filters
        self.dense_dropout_rate = dense_dropout_rate
        self.spatial_dropout_rate = spatial_dropout_rate
        self.dense_units = dense_units
        self.use_spatial_layer = use_spatial_layer
        self.verbose = verbose
        self.output_size = output_size

        self.att = Attention()

        self.model = None

        if self.verbose == 0:
            return
        
        print("Attia et al. CNN model initialized with the following parameters:")
        print(f"  filter_numbers: {self.filter_numbers}")
        print(f"  kernel_widths: {self.kernel_widths}")
        print(f"  pool_sizes: {self.pool_sizes}")
        print(f"  spatial_num_filters: {self.spatial_num_filters}")
        print(f"  dense_dropout_rate: {self.dense_dropout_rate}")
        print(f"  spatial_dropout_rate: {self.spatial_dropout_rate}")
        print(f"  dense_units: {self.dense_units}")
        print(f"  use_spatial_layer: {self.use_spatial_layer}")
    
    def get_temporal_layer(self, N, k, p, input_layer):
        c = Conv1D(N, k, padding='same', kernel_initializer='he_normal')(input_layer)
        b = tf.keras.layers.BatchNormalization()(c)
        a = Activation('relu')(b)
        p = MaxPooling1D(pool_size=p)(a)
        do = SpatialDropout1D(self.spatial_dropout_rate)(p)
        return do
            
    def get_temporal_layer_with_residual(self, N, k, p, input_layer):
        # Main pathway
        c = Conv1D(N, k, padding='same', kernel_initializer='he_normal')(input_layer)
        b = BatchNormalization()(c)
        a = Activation('relu')(b)
        
        # Shortcut pathway
        # Ensure the shortcut matches the dimension of the main pathway's output, adjust filters and stride as necessary
        shortcut = Conv1D(N, 1, padding='same', kernel_initializer='he_normal')(input_layer)  # 1x1 conv for matching dimension
        shortcut = BatchNormalization()(shortcut)  # Optional, for matching feature-wise statistics
        
        # Merging the shortcut with the main pathway
        merged_output = Add()([a, shortcut])  # Element-wise addition

        p = MaxPooling1D(pool_size=p)(merged_output)
        do = SpatialDropout1D(self.spatial_dropout_rate)(p)
        
        return do
    
    def get_spatial_layer(self, kernel_size, input_layer):
        c = Conv1D(self.spatial_num_filters, kernel_size, padding='same', data_format="channels_first", kernel_initializer='he_normal')(input_layer)
        b = tf.keras.layers.BatchNormalization()(c)
        a = Activation('relu')(b)
        do = SpatialDropout1D(self.spatial_dropout_rate)(a)
        return do
    
    def get_fully_connected_layer(self, units, input_layer):
        d = Dense(units, kernel_initializer='he_normal')(input_layer)
        b = tf.keras.layers.BatchNormalization()(d)
        a = Activation('relu')(b)
        do = Dropout(self.dense_dropout_rate)(a)
        return do

    def build(self, input_shape=(5000, 12)):
        input_layer = Input(shape=input_shape)
        last_layer = input_layer
        
        for i in range(len(self.pool_sizes)):
            temp_layer = self.get_temporal_layer(self.filter_numbers[i], self.kernel_widths[i],
                                            self.pool_sizes[i], last_layer)
            last_layer = temp_layer
        
        if self.use_spatial_layer:
            last_layer = self.get_spatial_layer(input_shape[1], last_layer)

        last_layer = Flatten()(last_layer)

        for i in range(len(self.dense_units)):
            dense_layer = self.get_fully_connected_layer(self.dense_units[i], last_layer)
            last_layer = dense_layer

        output_layer = Dense(self.output_size, activation='sigmoid')(last_layer)
        self.model = Model(inputs=input_layer, outputs=output_layer)

        if self.verbose > 0:
            print(self.model.summary())
        return self.model