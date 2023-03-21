#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import datetime
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

import joblib
import matplotlib.pyplot as plt
from IPython.display import HTML

from semisupervision import *
from visualize import animation_and_image
from sklearn.model_selection import train_test_split


# In[2]:
n_clusters = 100
# Hyperparams
mirror = False
MAX_FRAMES = 50
MAX_SEQ_LENGTH = MAX_FRAMES
N_PTS = 543
N_DIMS = 2
NUM_FEATURES = N_PTS*N_DIMS
RESIZE_METHOD = 'nearest'

START_FACE, END_FACE = (0, 468)
START_LHAND, END_LHAND = (468, 489)
START_POSE, END_POSE = (489, 522)
START_RHAND, END_RHAND = (522, 543)

LHAND_PTS = [i for i in range(START_LHAND, END_LHAND)]
RHAND_PTS = [i for i in range(START_RHAND, END_RHAND)]
FACE_PTS = [i for i in range(START_FACE, END_FACE)]
POSE_PTS = [i for i in range(START_POSE, END_POSE)]
LIPS_PTS = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 78, 191, 80, 81, 82, 13, 312, 311, 310,
    415, 308, 95, 88, 178, 87, 14, 317, 402, 318,
    324, 146, 91, 181, 84, 17, 314, 405, 321, 375,
]
PART_PTS = dict(
    lhand=LHAND_PTS,
    rhand=RHAND_PTS,
    face=FACE_PTS,
    pose=POSE_PTS,
    lips=LIPS_PTS,
)
PARTS_USED = ('lhand', 'rhand', 'lips')

PATIENCE = 16
BATCH_SIZE = 128
EPOCHS = 500

LR_START = 0.001
REDUCE_LR_PATIENCE = 4
REDUCE_LR_FACTOR = 0.2

X_npy_base = f'{MAX_FRAMES:03}_frames_key_frames_nearest_by_part.npy'
y_npy_fname = f'../y.npy'

COMP = os.environ.get('COMP_NAME', '?')
MODEL_DIR = '/'.join((__file__).split('/')[:-1])
print(f'{MODEL_DIR=}')
METRIC_STR = '_xx_val_acc-'
MODEL_NAME = MODEL_DIR.split(METRIC_STR)[-1]


DATA_ROOT = '../data/'
DF_TRAIN =  f'{DATA_ROOT}train.csv'
train = pd.read_csv(DF_TRAIN)

class CFG:
    data_path = DATA_ROOT
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

X_parts = dict(
    lhand=None,
    rhand=None,
    # face=None,
    # pose=None,
    # lips=None,
)

# %%
print(X_npy_base)

# In[3]:
try:
    y = np.load(y_npy_fname)
    
    for part_name in X_parts:
        X_parts[part_name] = np.load(
            f'X-{part_name}-{X_npy_base}',
        )
except:
    X_parts = dict(
        lhand=np.zeros(
            shape=(len(train), MAX_FRAMES, N_DIMS*(END_LHAND-START_LHAND)),
        ),
        rhand=np.zeros(
            shape=(len(train), MAX_FRAMES, N_DIMS*(END_RHAND-START_RHAND)),
        ),
    )
    y = np.zeros((len(train),))

    # 
    order_paths = []
    order_n_frames = []
    for i in tqdm(range(len(train))):
        y[i] = train.iloc[i].label
        path = f'{CFG.data_path}{train.iloc[i].path}'
        data = load_relevant_data_subset_with_imputation(path)
        order_paths.append(path)
        order_n_frames.append(len(data))
        ## Frame Aggregation per part
        for part_name in X_parts.keys():
            part_mask = np.zeros(data.shape[1], dtype='bool')
            part_mask[PART_PTS.get(part_name)] = True
            n_part_pts = part_mask.sum()

            data_part = data[:, part_mask, :N_DIMS]
            # Note dimensions are channels
            if len(data_part) > MAX_FRAMES:
                data_part_key_frames = tf.image.resize(
                    data_part,
                    size=(MAX_FRAMES, n_part_pts),
                    method='nearest',
                )
            else:
                data_part_key_frames = data_part[:MAX_FRAMES]

            n_frames = data_part_key_frames.shape[0]
            columns = data_part_key_frames.shape[1]*data_part_key_frames.shape[2]
            data_part_resize = tf.reshape(
                tensor=data_part_key_frames,
                shape=(n_frames, columns),
            )
            X_parts[part_name][i, :len(data_part_resize), :] = data_part_resize

    pd.DataFrame(
        data=(order_paths, order_n_frames),
        columns=('paths','nframes'),
    ).to_csv('data_info.csv')

    for part_name, X_data in X_parts.items():
        np.save(
            f'X-{part_name}-{X_npy_base}',
            X_data,
        )
    np.save(y_npy_fname, y)

# %% Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_parts['rhand'], y,
    test_size=0.10,
    random_state=27,
    stratify=y,
)


# %% Load data 
DATA_PART_NAME = 'rhand'
SIGN_NAME = 'ALL'
name_section = 'VictorDesktop'
curr_time = None
# fname = (
#     f'label'
#     f'-{DATA_PART_NAME}'
#     #TODO: Name for data source??
#     f'-sign_{SIGN_NAME}'
#     f'{name_section}'
#     '*'# f'-{curr_time}'
#     '.csv'
# )
fname = 'label-rhand-sign_ALL-user_VictorDesktop-20230321_120044.csv'
# frame_id,video,relative_frame,handshape
manual_labels_data = pd.read_csv(
    fname
)


# %% New training data
X_reshape = X_train.reshape(
    X_train.shape[0]*X_train.shape[1],
    X_train.shape[-1]
)

X_manual_label = X_reshape[manual_labels_data['frame_id'].values]
lookupTable, y_manual_label = np.unique(
    manual_labels_data['handshape'].values,
    return_inverse=True,
)

print(f'{X_manual_label.shape=}')
print(f'{y_manual_label.shape=}')



# %% Redo clustering
try:
    kmeans = joblib.load('_kmeans.pkl')
except:
    print(f'getting kmeans with {n_clusters}')
    kmeans = cluster_frames(X_reshape, k=n_clusters)
print('Get Distances')
kmeans_dist, kmeans_labels = get_distances_kmeans(
    frames=X_reshape,
    kmeans=kmeans,
)

# %% Kmeans distance save
try:
    kmeans_dist = np.load('kmeans_dist.npy')
except:
    np.save('kmeans_dist.npy', kmeans_dist)

# %% Propogate
prop_masks = dict(
    p10 = get_propogate_mask(X_reshape, kmeans=kmeans, percentile_closest=10),
    p20 = get_propogate_mask(X_reshape, kmeans=kmeans, percentile_closest=20),
)

prop_mask = prop_masks['p10']
temp_dist = kmeans_dist[prop_mask]
propogated_labels = np.argmin(kmeans_dist[prop_mask], axis=1)
# Get same name as the manually given labels
propogated_labels = np.vectorize(lambda x: y_manual_label[x])(propogated_labels)


# %%

model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(42,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(
            len(np.unique(y_manual_label)),
            activation="softmax",
        ),
    ]
)

print(model.summary())

# %%
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history_manual = model.fit(
    X_manual_label,
    y_manual_label.reshape(-1,1),
    # batch_size=batch_size,
    epochs=100,
    validation_split=0.1,
)

# %% Auto-Labeled ####################################
######################################################
X_prop = X_reshape[prop_mask]
y_prop = propogated_labels
print(f'{X_prop.shape=}')
print(f'{y_prop.shape=}')

# %%
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(42,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(
            len(np.unique(y_manual_label)-1),
            activation="softmax"
        ),
    ]
)

print(model.summary())

# %%
model.compile(
    # optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

X_mix = np.vstack(
    (X_manual_label,
    X_prop[y_prop != 0][4])
)
y_mix = np.hstack(
    (y_manual_label,
    y_prop[y_prop != 0][4])
)

history_autolabel = model.fit(
    X_prop[y_prop != 0],
    y_prop[y_prop != 0].reshape(-1,1),
    # X_mix,
    # y_mix.reshape(-1,1),
    batch_size=1024,
    epochs=5,
    validation_split=0.1,
)
# %% Eval
n_instance = np.random.choice(
    np.arange(y_val.shape[0])[y_val==label_index['donkey']]
)

X_val_reshape = X_val.reshape(
    X_val.shape[0]*X_val.shape[1],
    X_val.shape[-1]
)

print(f'{n_instance=}')
print(
    lookupTable[
        np.argmax(model.predict(
            X_val_reshape[MAX_FRAMES*n_instance:MAX_FRAMES*(n_instance+1)]
            ),axis=1
        )
    ]
)
a = animation_and_image(X_val[n_instance], sign_name=index_label[y_val[n_instance]])
from IPython.display import HTML
HTML(a.to_html5_video())
# %%
