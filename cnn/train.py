import numpy as np
import tensorflow as tf # print(tf.__version__): version gpu-1.13.1
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from scripts.metrics import dice_coefficient, weighted_categorical_crossentropy
from scripts.cnn_models import unet_model
import time
from datetime import timedelta
import pandas as pd
import os
from os.path import join as jp

# Check available GPUs
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print("Please install GPU version of TF")

# Set seed for reproducibility
np.random.seed(256)
tf.set_random_seed(256)

# Set necessary variables
K.set_image_data_format('channels_last')

project_name = '3dUnet'
img_depth = 64
img_rows = 128
img_cols = 160
num_of_labels = 6

# Load + compile Unet model
model = unet_model(img_depth, img_rows, img_cols, num_of_labels)

model.compile(optimizer=Adam(lr=1e-4), loss=['categorical_crossentropy'], metrics=['accuracy', dice_coefficient])

# Callbacks
path = '/.../'  # set path for saving training results
if not os.path.exists(path):
    os.makedirs(path)

filepath = jp(path, 'epochs:{epoch:03d}-best_model.hdf5')

model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only = False, mode = 'min')
csv_logger = CSVLogger(os.path.join(path, project_name + '_epoch_results.txt'), separator=',', append=False)
callbacks_list = [model_checkpoint, csv_logger]

# Load train and validation data
train_imgs = np.load('/.../')
train_labels = np.load('/.../')

val_imgs = np.load('/.../')
val_labels = np.load('/.../')

# Train model
print('Start training ... ')
start_time = time.time()

history = model.fit(train_imgs, train_labels, batch_size=2, epochs=250, verbose=1, shuffle=True, validation_data=(val_imgs, val_labels), callbacks=callbacks_list)

elapsed_time = time.time() - start_time

print('Training finished')
print('\nElapsed time: ', str(timedelta(seconds=elapsed_time)))

# Save history as csv file
hist_df = pd.DataFrame(history.history)
hist_csv_file = jp(path, 'history.csv')
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

