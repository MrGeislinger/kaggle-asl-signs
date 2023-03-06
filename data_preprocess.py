from __future__ import annotations
from typing import Callable
import numpy.typing as npt
import numpy as np

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
    
    if find_dist is None:
        find_dist = lambda f0,f1: np.linalg.norm(data[f1] - data[f0])
    
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
            #    f'not enough frames: '
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