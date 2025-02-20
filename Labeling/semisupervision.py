from __future__ import annotations
import numpy.typing as npt

import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
import matplotlib.pyplot as plts





def cluster_frames(
    frames: npt.ArrayLike,
    k: int,
    **cluster_kwargs,
):
    '''
    '''
    if 'random_state' not in cluster_kwargs:
        cluster_kwargs['random_state'] = 27
    if 'random_state' not in cluster_kwargs:
        cluster_kwargs['n_init'] = 'auto'
    if 'n_clusters' in cluster_kwargs:
        raise Exception(f'`n_clusters` already defined: {cluster_kwargs.get("n_clusters")}')
    else:
        cluster_kwargs['n_clusters'] = k

    kmeans = KMeans(**cluster_kwargs)
    kmeans.fit(frames)

    return kmeans


def get_distances_kmeans(
        frames: npt.ArrayLike, 
        kmeans = None,
        refit: bool = False,
        **cluster_kwargs,
) -> npt.ArrayLike:
    if kmeans is None:
        if 'random_state' not in cluster_kwargs:
            cluster_kwargs['random_state'] = 27
        if 'random_state' not in cluster_kwargs:
            cluster_kwargs['n_init'] = 'auto'
        # User defines number of clusters still
        if 'n_clusters' not in cluster_kwargs:
            raise Exception(f'`n_clusters` must be passed to function')
        kmeans = KMeans(**cluster_kwargs)
        kmeans_dist = kmeans.fit_transform(frames)
    # Whether to refit the KMeans clustering
    if refit:
        kmeans_dist = kmeans.fit_transform(frames)
    else:
        kmeans_dist = kmeans.transform(frames)

    return kmeans_dist, kmeans.labels_


def get_representative_images(
    frames: npt.ArrayLike,
    kmeans = None,
    frame_subset_mask = None,
    **cluster_kwargs,
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    '''Get representative images from frames after clustering.

    Defaults to KMeans to be calculated but `n_clusters` must be defined first
    '''
    if frame_subset_mask is None:
        frame_subset_mask = np.ones(frames.shape[0], dtype='bool')

    frames_subset = frames[frame_subset_mask]
    frames_subset = frames_subset.reshape(frames_subset.shape[0]*frames_subset.shape[1], frames_subset.shape[-1])

    kmeans_dist, _ = get_distances_kmeans(
        frames=frames_subset,
        kmeans=kmeans,
        **cluster_kwargs,
    )
    representative_frame_idx = np.argmin(kmeans_dist, axis=0)
    #
    all_frame_idx = []
    for i,s in enumerate(frame_subset_mask):
        if s: 
            all_frame_idx.extend([
                    i * frames.shape[1] + j
                    for j in range(frames.shape[1])
            ])
    all_frame_idx = np.array(all_frame_idx)
    subset_frame_idx = all_frame_idx[representative_frame_idx]
    
    representative_frames = frames_subset[representative_frame_idx]

    return representative_frames, subset_frame_idx


def get_propagate_mask(
    frames: npt.ArrayLike,
    kmeans,
    percentile_closest: float = 20,
):
    '''
    frames: All frames (not just labelled ones)
    '''
    kmeans_dist, kmeans_labels = get_distances_kmeans(
        frames=frames,
        kmeans=kmeans,
    )
    k = kmeans_dist.shape[1] # number of clusters
    #
    #
    cluster_distances = kmeans_dist[np.arange(len(frames)), kmeans.labels_]

    for i in range(k):
        in_cluster = (kmeans_labels == i)
        cluster_dist = cluster_distances[in_cluster]
        cutoff_dist = np.percentile(cluster_dist, percentile_closest)
        above_cutoff = (cluster_distances > cutoff_dist)
        cluster_distances[in_cluster & above_cutoff] = -1

    partially_propagated_mask = (cluster_distances != -1)
    return partially_propagated_mask




