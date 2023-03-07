#!/usr/bin/env python
# coding: utf-8
import os
import datetime
import json
from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
from data_preprocess import get_key_frames_by_cluster

import joblib
import matplotlib.pyplot as plt


# Hyperparams
mirror = True
MAX_FRAMES = 5
MAX_SEQ_LENGTH = MAX_FRAMES
N_PTS = 21 * 2 # just left & right hands
N_DIMS = 2
NUM_FEATURES = N_PTS*N_DIMS
BATCH_SIZE = 64
EPOCHS = 500

X_npy_fname = f'X-only_hands-{MAX_FRAMES}_frames_key_cluster.npy'
y_npy_fname = f'y.npy'
masks_fname = f'all_masks-only_hands-{MAX_FRAMES}_key_cluster.npy'

COMP = os.environ.get('COMP_NAME', '?')

model_details = (
    f'model_rnn-only_hands'
    f'-key_frames_cluster'
    f'-{MAX_FRAMES}_frames'
    f'-{N_PTS}_pts_per_frame'
    f'-{N_DIMS}_dims'
    f'-mirror' if mirror else ''
    f'-{BATCH_SIZE}_batch_size'
)

DATA_ROOT = Path('data')
DF_TRAIN = DATA_ROOT / 'train.csv'
train = pd.read_csv(DF_TRAIN)

print("train.shape = {} rows, {} cols".format(*train.shape))


class CFG:
    data_path = "data/"
    quick_experiment = False
    is_training = True
    use_aggregation_dataset = True
    num_classes = 250
    rows_per_frame = 543 

def load_relevant_data_subset_with_imputation(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    data.replace(np.nan, 0, inplace=True)
    n_frames = int(len(data) / CFG.rows_per_frame)
    data = data.values.reshape(n_frames, CFG.rows_per_frame, len(data_columns))
    return data.astype(np.float32)

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / CFG.rows_per_frame)
    data = data.values.reshape(n_frames, CFG.rows_per_frame, len(data_columns))
    return data.astype(np.float32)

def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, "r") as f:
        dic = json.load(f)
    return dic


train = pd.read_csv(f"{CFG.data_path}train.csv")
label_index = read_dict(f"{CFG.data_path}sign_to_prediction_index_map.json")
index_label = {label_index[key]: key for key in label_index}
train["label"] = train["sign"].map(lambda sign: label_index[sign])


def load_frames(
    frames_data,
    max_frames: int = 20,
    crop_method: str = 'nearest',
    get_masks: bool = True,
):
    '''Take frames data shape=(n_frames, pts, dims) to fixed num of frames.
    
    - Pads data with fewer frames with zeros
    - Reduces max number of allowed frames by crop method:
        * 'nearest' (default): 
    '''

    if len(frames_data) < max_frames:
        diff = max_frames - len(frames_data)
        padding = np.zeros((diff, N_PTS, N_DIMS))
        frames = np.concatenate((frames_data, padding))
        if get_masks:
            # Only mask the padding
            masks = np.zeros(shape=(1, max_frames,), dtype='bool')
            masks[0,:len(frames_data)] = 1
            
    else:
        if crop_method == 'nearest':
            frames = tf.image.resize(
                frames_data,
                (max_frames, N_PTS),
                method='nearest',
            ).numpy()
        elif crop_method == 'cut':
            frames = frames_data[:max_frames]
        else:
            raise Exception(f'{crop_method=} not found')
        if get_masks:
            # Use all the frames
            masks = np.ones(shape=(1, max_frames,), dtype='bool')

    if get_masks:
        return frames, masks
    else:
        return frames



def compress_frames(frames):
    '''Make a video of shape (n_frames, pts, dims) --> (n_frames, pts*dims) '''
    n_frames = frames.shape[0]
    columns = frames.shape[1]*frames.shape[2]
    return frames.reshape(n_frames, columns)


if CFG.is_training:

    try:
        X = np.load(X_npy_fname)
        y = np.load(y_npy_fname)
        M = np.load(masks_fname)
        
    except:
        X = np.zeros((len(train), MAX_FRAMES, NUM_FEATURES))
        M = np.zeros(shape=(len(train), MAX_FRAMES), dtype='bool')
        y = np.zeros((len(train),))
        for i in tqdm(range(len(train))):
            y[i] = train.iloc[i].label
            path = f'{CFG.data_path}{train.iloc[i].path}'
            data = load_relevant_data_subset_with_imputation(path)
            hand_mask = np.zeros(data.shape[1], dtype='bool')
            hand_mask[468:489] = True
            hand_mask[522:] = True
            #
            data = data[:, hand_mask, :N_DIMS]
            ## Frame Aggregation
            data_key_frames = get_key_frames_by_cluster(
                data=data,
                n_frames=MAX_FRAMES,
                padding=False, # If not enough key frames, keep just those
            )
            #
            frames_reduced, masks = load_frames(
                data_key_frames,
                max_frames=MAX_FRAMES,
                crop_method='cut',
            )
            M[i] = masks
            data_resize = compress_frames(frames_reduced)
            X[i] = data_resize
            y[i] = train.iloc[i].label
        # Save number of frames of each training sample for data analysis
        np.save(X_npy_fname, X)
        np.save(y_npy_fname, y)
        np.save(masks_fname, M)

    print(X.shape, y.shape, M.shape)


# All masks as part of split
X_train, X_val, M_train, M_val, y_train, y_val = train_test_split(
    X, M, y,
    test_size=0.2,
    random_state=27,
    stratify=y,
)


# Double the training data by mirroring the coordinates over the x-axis
if mirror:
    # Mirror x-axis of features
    X_temp = np.zeros(
        shape=(X_train.shape[0]*2, *X_train.shape[1:]),
    )
    X_temp[:X_train.shape[0]] = X_train
    X_temp[X_train.shape[0]:] = X_train
    X_temp[X_train.shape[0]:,:,0] *= -1 
    X_train = X_temp
    #
    y_temp = np.zeros(
        shape=(y_train.shape[0]*2,),
    )
    y_temp[:y_train.shape[0]] = y_train
    y_temp[y_train.shape[0]:] = y_train
    y_train = y_temp
    # Simple copy
    M_temp = np.zeros(
        shape=(M_train.shape[0]*2, M_train.shape[1]),
    )
    M_temp[:M_train.shape[0]] = M_train
    M_temp[M_train.shape[0]:] = M_train
    M_train = M_temp
    


# # RNN Model 
# > https://keras.io/examples/vision/video_classification/


# Utility for our sequence model.
def get_sequence_model(max_frames: int, num_features: int):
    n_classes = 250

    frame_features_input = tf.keras.Input((max_frames, num_features))
    mask_input = tf.keras.Input((max_frames,), dtype='bool')

    # Why `mask`: https://keras.io/api/layers/recurrent_layers/gru/
    x = tf.keras.layers.GRU(32, return_sequences=True)(
        frame_features_input,
        mask=mask_input,
    )
    x = tf.keras.layers.GRU(32)(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(128)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
   
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(64)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(n_classes, activation='softmax')(x)

    rnn_model = tf.keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='nadam',
        metrics=['accuracy'],
    )
    return rnn_model


def run_experiment(
    model,
    train_data,
    train_labels,
    validation_data = None,
    validation_split: int = 0.2,
    model_path: str = 'temp',
    epochs: int = 10,
    batch_size: int = 128,
    monitor_metric: str = 'val_accuracy',
    patience: int = 6,
):
    from Slack import SlackCallback
    slack_callback = SlackCallback(
        token=os.environ['SLACK_BOT_TOKEN'],
        start_message=(
            f'Model starting on {COMP=} for {EPOCHS=}\n'
            f'{model_details=}'
        ),
    )

    print(model.summary())

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            save_weights_only=True,
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            patience=patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=(
                'logs/fit/'
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                + f'-{model_details}'
            ),
            histogram_freq=1,
        ),
        slack_callback,
    ]
        
    fit_params = dict(
        x=train_data,
        y=train_labels,
        epochs=epochs,
        batch_size=batch_size,
    )
    if validation_data:
        fit_params['validation_data'] = validation_data
    else:
        fit_params['validation_split'] = validation_split

    history = model.fit(
        **fit_params,
        callbacks=callbacks,
    )

    return model, history


# ## Experiment

model = get_sequence_model(max_frames=MAX_FRAMES, num_features=NUM_FEATURES)
print(model.summary())

model_details += f'-{model.count_params()}_model_params'
model, history = run_experiment(
    model=model,
    train_data=(X_train, M_train),
    train_labels=y_train,
    validation_data=((X_val, M_val), y_val),
    model_path=f'{model_details}.h5',
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    patience=16,
)

joblib.dump(history, f'history-{model_details}.gz')


# ### Plot Results
# summarize history for accuracy
fig, (ax_acc, ax_loss) = plt.subplots(ncols=2, figsize=(16,12))

ax_acc.plot(history.history['accuracy'])
ax_acc.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss
ax_loss.plot(history.history['loss'])
ax_loss.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

fig.savefig(f'acc_loss-{model_details}.png')
