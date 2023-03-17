import streamlit as st

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from visualize import viz_hand, animation_and_image
from semisupervision import *
from glob import glob
st.set_page_config(layout="wide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css('style.css')


##########################################
# Revtrieved via https://aslfont.github.io/Symbol-Font-For-ASL/asl/handshapes.html
handshapes = pd.read_csv('handshapes.csv')
NO_SELECTION_STR = '--SELECT--'
selection_list = [NO_SELECTION_STR] + handshapes['gloss'].to_list()
##########################################

st.title('Labeler')

user_name = st.text_input(
    label='Name of User (Labeler)',
    value=os.environ.get('COMP_NAME'),
)

@st.cache_data
def load_selection_images():
    images = {
        fname.split('/')[-1].split('.png')[0]: Image.open(fname)
        for fname in glob('handshape-images/*.png')
    }
    return images

@st.cache_data
def load_data(part_name):
    # TODO: Change this to a fixed value
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


# Load data
label_index = read_dict(f'../data/sign_to_prediction_index_map.json')
index_label = {label_index[key]: key for key in label_index}
DATA_PART_NAME = 'rhand'
X_rhand, y = load_data(DATA_PART_NAME)

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
            label=f'{rep_frame_labels[j]}\nframe_idx={rep_frame_idx[j]}',
        )
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
    st.pyplot(fig)
    return rep_frame_idx

frame_index = show_rep_images(X_rhand, subset_mask=mask, n_clusters=N_CLUSTERS)


@st.cache_data()
def display_choice(
    _frame_data: npt.ArrayLike, # Not cached
    frame_idx: npt.ArrayLike, # All frames index
    n_frames: int, # To get relative frame for vide 
    sign_name: str,
    _col, # Not cached
):
    rel_frame_idx = frame_idx % n_frames
    animation = animation_and_image(
        _frame_data,
        main_frame_idx=rel_frame_idx,
        sign_name=sign_name,
    )
    with _col:
        st.components.v1.html(
            animation.to_html5_video(),
            width=300,
            height=400,
        )

results_container = st.container()

def write_labels_to_file():
    df = pd.DataFrame(
        data=[
            (
                f_idx,
                f_idx // N_FRAMES,
                f_idx % N_FRAMES,
                st.session_state[f_idx],
            )
            for f_idx in frame_index
            if st.session_state[f_idx] != NO_SELECTION_STR
        ],
        columns=('frame_id', 'video', 'relative_frame', 'handshape'),
    )
    results_container.write(df)
    # Write data to file
    curr_time = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    if user_name == 'None':
        name_section = ''
    else:
        name_section = f'-user_{user_name}'
    fname = (
        f'label'
        f'-{DATA_PART_NAME}'
        #TODO: Name for data source??
        f'-sign_{SIGN_NAME}'
        f'{name_section}'
        f'-{curr_time}'
        '.csv'
    )
    df.to_csv(fname, index=False)
    # TODO: Confirm?
    # TODO: Combine into one CSV

def selection_to_image(_container, label):
    try:
        image = load_selection_images()[label]
        _container.image(image, f'{label=}')
    except:
        _container.write(f'{label=}')

# Read in (most recent) file with same sign name & populate selection value 
# if frame already defined
csvs = glob(f'label-*{SIGN_NAME}*.csv')
prev_signs = dict()
if csvs:
    # TODO: Decide how to handle multiple CSVs for a sign
    df_prev_signs = pd.read_csv(csvs[0], index_col=False,)
    prev_signs = pd.Series(
        df_prev_signs.handshape.values,
        index=df_prev_signs.frame_id,
    ).to_dict()

form = st.form('my_form', clear_on_submit=True)
with form:
    submitted = st.form_submit_button(
        'Save results',
        on_click=write_labels_to_file
    )
    for i,frame_idx in enumerate(frame_index):
        col1, col2, col3 = st.columns(3)
        display_choice(
            _frame_data=X_rhand[frame_idx//N_FRAMES].reshape(-1,21,2),
            frame_idx=frame_idx,
            n_frames=N_FRAMES,
            sign_name=index_label[y_all_frames[frame_idx]],
            _col=col1,
        )
        col1.write(f'Frame Index: {frame_idx:_}')
        col2.selectbox(
            'handshape',
            selection_list,
            index=selection_list.index(prev_signs.get(frame_idx, NO_SELECTION_STR)),
            key=frame_idx,
        )
        col3.write(f'{frame_idx=}')

if submitted:
    st.write('submitted')
    