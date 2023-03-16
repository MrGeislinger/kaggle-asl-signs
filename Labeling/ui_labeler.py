import streamlit as st

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from visualize import viz_hand, animation_and_image
from semisupervision import *

st.set_page_config(layout="wide")

##########################################
# Revtrieved via https://aslfont.github.io/Symbol-Font-For-ASL/asl/handshapes.html
handshapes = pd.read_csv('handshapes.csv')

##########################################

@st.cache_data
def load_data(part_name):
    X_npy_base = f'15_frames_key_resize_nearest_by_part.npy'
    X = np.load(
        f'../X-{part_name}-{X_npy_base}'
    )
    y = np.load('../y.npy')
    return X, y



@st.cache_data
def read_dict(file_path):
    path = os.path.expanduser(file_path)
    with open(path, 'r') as f:
        d = json.load(f)
    return d

st.title('Labeler')


# Load data
train = pd.read_csv(f'../data/train.csv')
label_index = read_dict(f'../data/sign_to_prediction_index_map.json')
index_label = {label_index[key]: key for key in label_index}
X_rhand, y = load_data('rhand')

# Button to perform KMeans
N_CLUSTERS = st.number_input(
    label='Number of clusters',
    min_value=5,
    max_value=100,
    value=20,
    step=5,
)

# Display Signs
SIGN_NAME = st.selectbox(
    'Select sign',
    label_index.keys(),
)
st.write('You selected:', SIGN_NAME)

# Process for the sign
N_FRAMES = X_rhand.shape[1]
mask = np.isin(y, [label_index[SIGN_NAME]])
N = mask.sum() # instances
X_sample = X_rhand[mask][:].reshape(N*N_FRAMES, X_rhand.shape[-1])
# Get sign label for each frame
y_all_frames = np.zeros((X_rhand.shape[0]*N_FRAMES))
for i,label in enumerate(y):
    y_all_frames[i*N_FRAMES:(i+1)*N_FRAMES] = label


@st.cache_data
def show_rep_images(X, subset_mask, n_clusters):
    X_subset = X[subset_mask]
    X_subset = X_subset.reshape(
        X_subset.shape[0]*X_subset.shape[1],
        X_subset.shape[-1]
    )
    kmeans = cluster_frames(X_subset, k=n_clusters)
    
    rep_frames, rep_frame_idx = get_representative_images(
        X,
        kmeans,
        frame_subset_mask=subset_mask,
    )
    rep_frame_labels = tuple(index_label[l] for l in y_all_frames[rep_frame_idx])

    base_size = 3
    fig = plt.figure(
        figsize=(5*base_size, (n_clusters//5)*base_size),
    )

    for j in range(n_clusters):
        ax = fig.add_subplot(n_clusters//5, 5, j+1)
        f_rhand = rep_frames[j].reshape(-1,2).copy()
        viz_hand(
            ax=ax,
            hand_frame=f_rhand,
            label=rep_frame_labels[j],
        )
    st.pyplot(fig)
    return rep_frame_idx

frame_index = show_rep_images(X_rhand, subset_mask=mask, n_clusters=N_CLUSTERS)
selection_containers = {}
# choice = {}


@st.cache_data()
def display_choice(_frame_data, frame_idx, sign_name, _col,):
    animation = animation_and_image(
        _frame_data,
        main_frame_idx=frame_idx,
        sign_name=sign_name,
    )
    with _col:
        st.components.v1.html(
            animation.to_html5_video(),
            width=300,
            height=400,
        )


results_container = st.container()

state = st.session_state
def write_labels_to_file():
    df = pd.DataFrame(
        data=[
            (
                f_idx,
                f_idx // 15,
                f_idx % 15,
                state[f_idx],
            )
            for f_idx in frame_index
        ],
        columns=('frame_id', 'video', 'relative_frame', 'handshape'),
    )
    results_container.write(df)
    # Write data to file
    fname = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    df.to_csv(f'label-{fname}.csv', index=False)
    # TODO: Confirm?

form = st.form('my_form', clear_on_submit=True)
with form:
    submitted = st.form_submit_button(
        'Save results',
        on_click=write_labels_to_file
    )
    for i,frame_idx in enumerate(frame_index):
        col1, col2, col3 = form.columns(3)
        display_choice(
            _frame_data=X_rhand[frame_idx//15].reshape(-1,21,2),
            frame_idx=frame_idx%15,
            sign_name=index_label[y_all_frames[frame_idx]],
            _col=col1,
        )
        col2.selectbox('handshape', handshapes['gloss'], key=frame_idx)
        col3.write(f'{frame_idx=}')

if submitted:
    st.write('submitted')
    