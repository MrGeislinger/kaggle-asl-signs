from __future__ import annotations
from typing import Callable
import numpy.typing as npt
import numpy as np
from sklearn.cluster import KMeans

def get_disparate_key_frames(
    data: npt.ArrayLike,
    n_frames: int = 3,
    initial_frame_idx: int = 0,
    distance: Callable[[int, int], float] | None = None,
    padding: bool = False,
) -> npt.ndarray:
    '''Use first and most n disparate key frames by a distance metric function.
    
    n_frames: number of total key frames to (ideally) return
    initial_frame_idx: index of initial key frame to search from and against
    distance: function takes in 2 frame indices and returns distance metric
    padding: whether to pad result with "zero frames"
    '''
    frames = [initial_frame_idx]
    curr_frame_idx = frames[0]
    frame_diffs = {}
    
    if distance is None:
        distance = lambda f0,f1: np.linalg.norm(data[f1] - data[f0])
    
    # TODO: Skip for a high percent of NaNs (set to all zeros)
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = 0
    

    while len(frames) < n_frames:
        # Get most different curr_frame
        most_different_frame_rel_idx = np.argmax([
            distance(curr_frame_idx, fi) # Find all distances from curr_frame
            for fi in range(curr_frame_idx, data.shape[0])
            if not np.all(data[fi] == 0) # Skip if all points are 0s (Had NaNs)
        ])
        most_different_frame_idx = (
            most_different_frame_rel_idx # relative position from frame
            + curr_frame_idx # Offset by the current frame index
        )
        frames.append(most_different_frame_idx)
        curr_frame_idx = most_different_frame_idx
        # Possible not enough different frames
        if curr_frame_idx + 1 >= data.shape[0]:
            # TODO: Log that there weren't enough frames
            #print(
            #    f'Want {n_frames} frames; Have {len(frames)} '
            #)
            # All leftover frames are just missing from final
            # Padding option: rest of frames default all zeros
            # TODO: Copy rest of frames to end?
            break

    # Output: same shape after ignoring the number of frames
    # Only the n frames used
    new_shape = (len(frames), *data.shape[1:])
    # Pad missing frames with zeros
    if padding:
        new_shape = (n_frames, *new_shape[1:])    
     
    data_new = np.zeros(new_shape) 
    for i,f in enumerate(frames):
        data_new[i] = data[f,:,:]

    return data_new

def get_key_frames_by_cluster(
    data: npt.ArrayLike,
    n_frames: int = 3,
    distance: Callable[[npt.ArrayLike, npt.ArrayLike], float] | None = None,
    padding: bool = False,
    unique: bool = True,
    return_mask: bool = False,
    **kmeans_kwargs,
) -> npt.ArrayLike:
    '''Use frames closest to KMeans' centroids by distance metric function.
    
    If fewer frames requested (centroids/clusters), returns full data (with
    zeros for NaNs).
    
    Frames are always returned in sequential order as given.
    
    n_frames: number of total key frames to (ideally) return
    initial_frame_idx: index of initial key frame to search from and against
    distance: function takes in 2 frames and returns distance metric
        default to euclidean distance between frames
    padding: whether to pad result with "zero frames"
        default to False
    unique: whether key frames returned should be unique. Looks for next 
      closest frame to centroid.
        default to True
    return_mask: whether to return the mask instead of the key frames
        default to False
    '''
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = 0
    if len(data) <= n_frames:
        # Return a mask instead
        if return_mask:
            mask_frames = np.ones(shape=data.shape[0], dtype='bool')
            # Allow for padding (bigger than original data)
            if padding:
                mask_frames_new = np.zeros(shape=(n_frames,), dtype='bool')
                mask_frames_new[:mask_frames.shape[0]] = mask_frames
                mask_frames = mask_frames_new
            return mask_frames
        # Padding will make the result biggesr than origina
        if padding:
            new_shape = (n_frames, *new_shape[1:])
            data_new = np.zeros(new_shape)
            data_new[:data.shape[0]] = data
            data = data_new
        return data

    reshaped = (data.shape[0], np.prod(data.shape[1:]))    
    kmeans_params = dict(
        n_clusters=n_frames,
        random_state=27,
        n_init='auto',
    )
    kmeans_params |= kmeans_kwargs
    kmeans = KMeans(**kmeans_params).fit(data.reshape(reshaped))

    frames = []
    if distance is None:
        distance = lambda f0,f1: np.linalg.norm(f1 - f0)
    for c in kmeans.cluster_centers_:
        # When unique flag is True, don't consider used frames
        frame_distances = {
            i: distance(f,c)
            for i,f in enumerate(data.reshape(reshaped))
            if (not unique) or (unique and i not in frames)
        }
        try:
            i = min(frame_distances, key=lambda i: frame_distances[i])
            frames.append(i)
        except ValueError: # In case there are no frames left
            pass

    frames.sort() # Keep the frames in sequential order
    if return_mask:
        mask_frames = np.zeros(shape=data.shape[0], dtype='bool')
        for fi in frames:
            mask_frames[fi] = True
        return mask_frames
    key_frames = data[frames]         

    return key_frames