import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow import keras

from dataset import Dataset
from model import build_model
from loss import my_loss

gpus = tf.config.list_physical_devices(device_type='GPU')
if gpus:
    tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)

batch_size = 2
input_path = 'train_data/sensor/2018-11-29.csv'
target_dir = 'train_data/heatMat/'
label_maps = pd.read_csv('data_name_idx.txt', header=None, sep="\t")
trian_gen = Dataset(batch_size, input_path, target_dir, label_maps,trian=True)
val_gen = Dataset(batch_size, input_path, target_dir, label_maps,val=True)

model = build_model()
model.summary()

initial_learning_rate = 1e-4
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=300, decay_rate=0.96, staircase=True
)

model.compile(loss=my_loss,optimizer=keras.optimizers.Adam(learning_rate=lr_schedule))
checkpoint_cb = keras.callbacks.ModelCheckpoint("model_5.h5", save_best_only=True)

epochs = 20
history = model.fit(
    trian_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=[checkpoint_cb],
    shuffle=True )
 
pd.DataFrame(history.history).plot()
plt.savefig('loss.png')