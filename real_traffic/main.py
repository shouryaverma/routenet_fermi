import os
import tensorflow as tf
from data_generator import input_fn

import sys

#sys.path.append('../../')
from delay_model import RouteNet_Fermi
# from delay_model_LSTM import RouteNet_Fermi
# from delay_model_RNN import RouteNet_Fermi

TRAIN_PATH = f'/home/verma198/Public/RouteNet-Fermi/data/real_traces/train/geant'
VALIDATION_PATH = f'/home/verma198/Public/RouteNet-Fermi/data/real_traces/test/geant'
TEST_PATH = f'/home/verma198/Public/RouteNet-Fermi/data/real_traces/test/geant'

ds_train = input_fn(TRAIN_PATH, shuffle=True)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
ds_train = ds_train.repeat()

ds_validation = input_fn(VALIDATION_PATH, shuffle=False)
ds_validation = ds_validation.prefetch(tf.data.experimental.AUTOTUNE)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model = RouteNet_Fermi()
ckpt_dir = './ckpt_dir_GRU'

loss_object = tf.keras.losses.MeanAbsolutePercentageError()

model.compile(loss=loss_object,
              optimizer=optimizer,
              run_eagerly=False)

# latest = tf.train.latest_checkpoint(ckpt_dir)
latest = None

if latest is not None:
    print("Found a pretrained model, restoring...")
    model.load_weights(latest)
else:
    print("Starting training from scratch...")

filepath = os.path.join(ckpt_dir, "{epoch:02d}-{val_loss:.2f}")

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=filepath,
    verbose=1,
    mode="min",
    monitor='val_loss',
    save_best_only=False,
    save_weights_only=True,
    save_freq='epoch')

model.fit(ds_train,
          epochs=50,
          steps_per_epoch=2000,
          validation_data=ds_validation,
          callbacks=[cp_callback],
          use_multiprocessing=True)

ds_test = input_fn(TEST_PATH, shuffle=False)
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

model.evaluate(ds_test)
