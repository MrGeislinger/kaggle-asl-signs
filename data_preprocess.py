from __future__ import annotations
from typing import Callable
import numpy.typing as npt
import numpy as np

def first_and_disparate_key_frames(
    data: npt.ArrayLike,
    n_frames: int = 3,
    frame0: int = 0,
    find_dist: Callable[[int, int], float] | None = None,
) -> npt.ndarray:
    '''Use first and most n disparate key frames by a distance metric function.
    '''
    frames = [frame0] # Always include first frame (as defined by input params)
    curr_frame_idx = frames[0]
    frame_diffs = {}
    
    if find_dist is None:
        find_dist = lambda f0,f1: np.linalg.norm(data[f1] - data[f0])
    
    # TODO: Skip for a high percent of NaNs (set to all zeros)
    where_are_NaNs = np.isnan(data)
    data[where_are_NaNs] = 0

    

    while len(frames) < n_frames:
        # Get most different curr_frame
        most_different_frame_rel_idx = np.argmax([
            find_dist(curr_frame_idx, fi) # Find all distances from curr_frame
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
        if curr_frame_idx >= data.shape[0]:
            print(
                f'not enough frames: '
                f'Want {n_frames} frames; Have {len(frames)} '
            )
            # All leftover frames are just the default all zeros
            # TODO: Copy rest of frames to end?

    data_new = np.zeros((n_frames,*data.shape[1:])) # Same shape after n_frames
    for i,f in enumerate(frames):
        data_new[i] = data[f,:,:]

    return data_new