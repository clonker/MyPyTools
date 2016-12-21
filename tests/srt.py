import time

import numpy as np
import super_resolution_tools as srt


def insert_trans_rot_frame(vf, f_pad, dx, dy, th):
    sin_th = np.sin(th)
    cos_th = np.cos(th)

    cx = int(np.floor(0.5 * f_pad.shape[0]) + dx)
    cy = int(np.floor(0.5 * f_pad.shape[1]) + dy)

    xr = cos_th * vf[:, 0] - sin_th * vf[:, 1]
    yr = sin_th * vf[:, 0] + cos_th * vf[:, 1]

    for i in range(vf.shape[0]):
        ind_x = int(np.floor(xr[i]))
        ind_y = int(np.floor(yr[i]))

        frac_x = xr[i] - ind_x
        frac_y = yr[i] - ind_y

        # f_pad[int (cx + ind_x), int (cy - ind_y)] += vf[i, 2]

        corn1_x = (cx + ind_x) % f_pad.shape[0]
        corn1_y = (cy - ind_y) % f_pad.shape[1]

        corn2_x = (cx + ind_x + 1) % f_pad.shape[0]
        corn2_y = (cy - ind_y - 1) % f_pad.shape[1]

        # bilinear interpolation
        f_pad[corn1_x, corn1_y] += (1.0 - frac_x) * (1.0 - frac_y) * vf[i, 2]
        f_pad[corn2_x, corn1_y] += frac_x * (1.0 - frac_y) * vf[i, 2]
        f_pad[corn1_x, corn2_y] += (1.0 - frac_x) * frac_y * vf[i, 2]
        f_pad[corn2_x, corn2_y] += frac_x * frac_y * vf[i, 2]


vframes = np.load('/home/mho/Dropbox/Minisymposium/Super-Resolution Microscopy/149septin_vframes.npy')
dim = np.load('/home/mho/Dropbox/Minisymposium/Super-Resolution Microscopy/149septin_dim.npy')
gains = np.load('/home/mho/Dropbox/Minisymposium/Super-Resolution Microscopy/149septin_gain.npy')

lxmax = np.max(dim[:, 0])
lymax = np.max(dim[:, 1])

lmax = int(np.round(max(lxmax, lymax) * 1.5))

n_frames = len(vframes)

d = np.zeros([n_frames, n_frames])
th = np.zeros([n_frames, n_frames])
dx = np.zeros([n_frames, n_frames])
dy = np.zeros([n_frames, n_frames])

for i in range(10):
    f1_padded = np.zeros([lmax, lmax])
    f1_padded_control = np.zeros([lmax, lmax])

    start = time.time()
    srt.insert_trans_rot_frame (vframes[i], f1_padded, 0.0, 0.0, 0.0)
    end = time.time()
    print("srt: %s" % (end - start))
    start = time.time()
    insert_trans_rot_frame(vframes[i], f1_padded_control, 0.0, 0.0, 0.0)
    end = time.time()
    print("py: %s" % (end - start))

    np.testing.assert_array_almost_equal(f1_padded, f1_padded_control)

    #print f1_padded
    #print "---"*30
    #print f1_padded_control