import streamlit as st

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from visualize import viz_hand, animation_plot, animation_and_image
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


# Button to perform KMeans
N_CLUSTERS = 20

@st.cache_data
def show_rep_images(X, subset_mask):
    X_subset = X[subset_mask]
    X_subset = X_subset.reshape(
        X_subset.shape[0]*X_subset.shape[1],
        X_subset.shape[-1]
    )
    kmeans = cluster_frames(X_subset, k=N_CLUSTERS)
    
    rep_frames, rep_frame_idx = get_representative_images(
        X,
        kmeans,
        frame_subset_mask=subset_mask,
    )
    rep_frame_labels = tuple(index_label[l] for l in y_all_frames[rep_frame_idx])

    base_size = 3
    fig = plt.figure(
        figsize=(5*base_size, (N_CLUSTERS//5)*base_size),
    )

    for j in range(N_CLUSTERS):
        ax = fig.add_subplot(N_CLUSTERS//5, 5, j+1)
        f_rhand = rep_frames[j].reshape(-1,2).copy()
        viz_hand(
            ax=ax,
            hand_frame=f_rhand,
            label=rep_frame_labels[j],
        )
    st.pyplot(fig)
    return rep_frame_idx

frame_index = show_rep_images(X_rhand, subset_mask=mask)
selection_containers = {}
choice = {}


# @st.cache_data()
def display_choice(frame_data, frame_idx, sign_name, key_id, _containers):
    animation = animation_and_image(
        frame_data,
        main_frame_idx=frame_idx,
        sign_name=sign_name,
    )
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.components.v1.html(animation.to_html5_video(), width=300, height=400,)

    # Select frame name
    _containers[key_id] = temp = col2.container()

    # Show examples of frames?
    col3.write('temp3')

with st.form('my_form'):
    instance_idx = 0
    submitted = st.form_submit_button("Submit")
    temp = st.container()
    for i,frame_idx in enumerate(frame_index):
        display_choice(
            frame_data=X_rhand[frame_idx//15].reshape(-1,21,2),
            frame_idx=frame_idx%15,
            sign_name=index_label[y_all_frames[frame_idx]],
            key_id=i,
            _containers=selection_containers,
        )
        st.write(len(selection_containers))
        # Every form must have a submit button.
        with selection_containers[i]:
            choice[i] = st.selectbox('handshape', handshapes['gloss'], key=i)
    if submitted:
       temp.write('submitted')
       temp.write(choice)
