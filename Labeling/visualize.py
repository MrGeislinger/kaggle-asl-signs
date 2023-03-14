import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_hand_points(hand):
    x = [
         np.array([hand[0][0],  hand[1][0],  hand[2][0],  hand[3][0],  hand[4][0]],),
         np.array([hand[5][0],  hand[6][0],  hand[7][0],  hand[8][0]],),
         np.array([hand[9][0],  hand[10][0], hand[11][0], hand[12][0]],),
         np.array([hand[13][0], hand[14][0], hand[15][0], hand[16][0]],),
         np.array([hand[17][0], hand[18][0], hand[19][0], hand[20][0]],),
         np.array([hand[0][0],  hand[5][0],  hand[9][0],  hand[13][0], hand[17][0], hand[0][0]],),
    ]

    y = [
         np.array([hand[0][1], hand[1][1], hand[2][1], hand[3][1], hand[4][1]],),
         np.array([hand[5][1], hand[6][1], hand[7][1], hand[8][1]],),
         np.array([hand[9][1], hand[10][1], hand[11][1], hand[12][1]],),
         np.array([hand[13][1], hand[14][1], hand[15][1], hand[16][1]],),
         np.array([hand[17][1], hand[18][1], hand[19][1], hand[20][1]],),
         np.array([hand[0][1], hand[5][1], hand[9][1], hand[13][1], hand[17][1], hand[0][1]],),
    ] 
    return x, y


def viz_hand(ax, hand_frame, label='hand', axis_min=None, axis_max=None):
    hand = hand_frame.reshape(-1,2).copy()
    # Flip y
    hand[:,1] = 1 - hand[:,1]

    rx, ry = get_hand_points(hand)

    for i in range(len(rx)):
        ax.plot(rx[i], ry[i], )

    axis_min = axis_min if axis_min else np.nanmin(hand) - 0.2
    xmin = axis_min
    ymin = axis_min
    axis_max = axis_max if axis_max else np.nanmax(hand) + 0.2
    xmax = axis_max
    ymax = axis_max

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(label)


def animation_plot(data):
    fig, ax = plt.subplots()

    axis_min=data.min()
    axis_max=data.max()

    def animation_frame(f_idx):
        ax.cla()
        f = data[f_idx]
        viz_hand(
            ax=ax,
            hand_frame=f,
            label='test',
            axis_min=axis_min,
            axis_max=axis_max,
        )
        

    animation = FuncAnimation(
        fig,
        func=animation_frame,
        frames=list(range(data.shape[0])),
    )

    return animation

def animation_and_image(
    data,
    main_frame_idx=0,
    sign_name='?',
):
    fig, (ax_img, ax_anim) = plt.subplots(nrows=2, figsize=(2,4))
    # Remove the axis ticks to make it clearer to read
    ax_anim.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )
    ax_img.tick_params(
        axis='both',
        which='both',
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False,
    )

    viz_hand(
            ax=ax_img,
            hand_frame=data[main_frame_idx],
            label=f'Frame in question #{main_frame_idx}',
        )
    # Make the animations at the same scale
    axis_min=data.min()
    axis_max=data.max()

    def animation_frame(f_idx):
        ax_anim.cla()
        f = data[f_idx]
        viz_hand(
            ax=ax_anim,
            hand_frame=f,
            label=f'Animation of `{sign_name}`',
            axis_min=axis_min,
            axis_max=axis_max,
        )    

    animation = FuncAnimation(
        fig,
        func=animation_frame,
        frames=list(range(data.shape[0])),
    )

    return animation