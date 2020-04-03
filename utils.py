import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import json

cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#FFFFFF'])
norm = colors.Normalize(vmin=0, vmax=10)


def seed_everything(seed=42):
    print(f'setting everything to seed {seed}')
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_batch(task_paths, out_rows, out_cols):
    x_batch = []
    y_batch = []

    x_test_batch = []
    y_test_batch = []
    for task_file in task_paths:
        with open(task_file, 'r') as f:
            task = json.load(f)

        input_im1, output_im1, not_valid = pad_im(task, out_rows,
                                                  out_cols, mode='train')
        if not_valid:
            continue

        input_im, output_im, not_valid = pad_im(task, out_rows,
                                                out_cols, mode='test')
        if not_valid:
            continue

        x_batch.extend(input_im1[None])
        y_batch.extend(output_im1[None])
        x_test_batch.extend(input_im[None])
        y_test_batch.extend(output_im[None])
    return x_batch, y_batch, x_test_batch, y_test_batch


def pad_im(task, out_rows, out_cols, mode='train', cval=10):

    ip = []
    op = []
    num_pairs = len(task[mode])
    input_im = np.zeros((num_pairs, 1, out_rows, out_cols))
    output_im = np.zeros(
        (num_pairs, 1, out_rows, out_cols), dtype=np.long)
    for task_num in range(num_pairs):
        im = np.array(task[mode][task_num]['input'])
        nrows, ncols = im.shape
        if (nrows > out_rows) or (ncols > out_cols):
            return 0, 0, 1
        im = np.pad(im, ((out_rows-nrows, 0), (out_cols-ncols, 0)), mode='constant',
                    constant_values=(cval, cval))

        input_im[task_num, 0] = im
        im = np.array(task[mode][task_num]['output'])
        nrows, ncols = im.shape
        if (nrows > out_rows) or (ncols > out_cols):
            return 0, 0, 1
        im = np.pad(im, ((out_rows-nrows, 0), (out_cols-ncols, 0)), mode='constant',
                    constant_values=(cval, cval))
        output_im[task_num, 0] = im
    ip.extend(input_im)
    op.extend(output_im)

    return np.vstack(ip), np.vstack(op), 0


def plot_figure(x_spt, y_spt, x_qry,
                pred_q, im_num, img_sz=30):

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(x_spt[0].cpu().numpy().reshape(img_sz, img_sz),
               cmap=cmap, norm=norm)
    plt.subplot(2, 2, 2)
    plt.imshow(y_spt[:img_sz*img_sz].cpu().numpy().reshape(img_sz, img_sz),
               cmap=cmap, norm=norm)

    plt.subplot(2, 2, 3)
    plt.imshow(x_qry[0].cpu().numpy().reshape(img_sz, img_sz),
               cmap=cmap, norm=norm)

    # do visualization only for the first input.
    pred_q = pred_q[0, :img_sz*img_sz].cpu().numpy().reshape(img_sz, img_sz)
    frow = np.nonzero(np.count_nonzero(pred_q-10, axis=1))[0][0]
    fcol = np.nonzero(np.count_nonzero(pred_q-10, axis=0))[0][0]
    a = np.copy(pred_q[frow:, fcol:])
    a[a == 10] = 0
    plt.subplot(2, 2, 4)
    plt.imshow(a,
               cmap=cmap, norm=norm)

    plt.savefig(f'./model_preds/epoch_30_preds_{im_num}.png')
    plt.close()
