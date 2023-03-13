#!/usr/bin/env python
# coding: utf-8
import os
import datetime
import json
from tqdm import tqdm

import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import joblib
import matplotlib.pyplot as plt
import math

# Hyperparams
mirror = False
MAX_FRAMES = 15
MAX_SEQ_LENGTH = MAX_FRAMES
N_PTS = 543
N_DIMS = 2
NUM_FEATURES = N_PTS*N_DIMS

START_FACE, END_FACE = (0, 468)
START_LHAND, END_LHAND = (468, 489)
START_POSE, END_POSE = (489, 522)
START_RHAND, END_RHAND = (522, 543)
LIPS_PTS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324, 146, 91, 181, 84, 17, 314, 405, 321, 375]

LANDMARK_PTS = len(LIPS_PTS) + (END_LHAND - START_LHAND) + (END_RHAND - START_RHAND)

avg_sets = [(START_FACE, END_FACE),(START_POSE, END_POSE)]

DATA_ROOT = 'data/'
DF_TRAIN =  f'{DATA_ROOT}train.csv'
X_npy_fname = f'X-15-keyframes-nearest-avg-std.npy'
y_npy_fname = f'y.npy'
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


def tf_nan_mean(x, axis=0):
    return tf.math.reduce_mean(x, axis, keepdims=True)

def tf_nan_std(x, axis=0):
    return tf.math.reduce_std(x, axis=axis, keepdims=True)

def flatten_means_and_stds(x, axis=0):
    # Get means and stds
    x_mean = tf_nan_mean(x, axis=0)
    x_std  = tf_nan_std(x, axis=0)

    x_out = tf.concat([x_mean, x_std], axis=0)
    x_out = tf.reshape(x_out, (1, INPUT_SHAPE[1]*2))
    x_out = tf.where(tf.math.is_finite(x_out), x_out, tf.zeros_like(x_out))
    return x_out

def convert_row(index_row):
    _, row = index_row

    path = f'{CFG.data_path}{row.path}'
    data = load_relevant_data_subset_with_imputation(path) # (N, 543, 3)
    data = data[:,:,:N_DIMS]
    
    x_list = [tf_nan_mean(data[:, avg_set[0]:avg_set[0]+avg_set[1],:],axis=1) for avg_set in avg_sets]
    x_list += ([tf_nan_std(data[:, avg_set[0]:avg_set[0]+avg_set[1],:],axis=1) for avg_set in avg_sets])
    x_list.append(tf.gather(data, LIPS_PTS, axis=1))
    x_list.append(data[:,START_LHAND:END_LHAND,:])
    x_list.append(data[:,START_RHAND:END_RHAND,:])
    x = tf.concat(x_list, 1)

    ## Frame Aggregation
    x = tf.image.resize(
        x, # tf.where(tf.math.is_finite(x), x, tf_nan_mean(x, axis=0)),
        size=(MAX_FRAMES, LANDMARK_PTS + len(avg_sets)*2),
        method='nearest', #DEFAULT
    )
    # (MAX_FRAMES, num_landmark_pts + avg_set*2, 2)
    x = tf.reshape(x, (MAX_FRAMES, x.shape[1] * x.shape[2]))
    y = row.label
    return x, y

if __name__ == '__main__':
    train = pd.read_csv(DF_TRAIN)
    label_index = read_dict(f"{CFG.data_path}sign_to_prediction_index_map.json")
    index_label = {label_index[key]: key for key in label_index}
    train["label"] = train["sign"].map(lambda sign: label_index[sign])

    X = np.zeros((len(train), MAX_FRAMES, (LANDMARK_PTS + (len(avg_sets)*2))*N_DIMS))
    y = np.zeros((len(train),))
    
#     for i in tqdm(range(100)):
#         _x, _y = convert_row((i, train.iloc[i]))
#         X[i] = _x
#         y[i] = _y
        
    with mp.Pool(processes=4) as pool:
        results = pool.imap_unordered(convert_row, train.iterrows(), chunksize=250)
        for i, (_x,_y) in tqdm(enumerate(results), total=len(train)):
            X[i] = _x
            y[i] = _y
    
    # Save number of frames of each training sample for data analysis
    np.save(X_npy_fname, X)
    np.save(y_npy_fname, y)
    print(X.shape, y.shape)
