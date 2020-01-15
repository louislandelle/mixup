
import numpy as np
import torch
import PIL
import matplotlib.pyplot as plt
import os
import time
import datetime

def show_np_arr(arr, ax=None):
    """ Shows an np array of floats """
    np_img_255 = np.uint8(arr*255)
    im = PIL.Image.fromarray(np.rollaxis(np_img_255.T, axis=1))
    if ax:
        ax.imshow(im)
    else:
        plt.imshow(im)
        
def get_print(save_raw=False, res_dir="results", fname="0"):
    """ Makes print function to allow keeping output on disk """
    def print_(*a, **kwa):
        if save_raw:
            with open(os.path.join(res_dir, fname), "a") as f:
                t = datetime.datetime.now()
                f.write(str(t) + '    ' + ' '.join([str(x) for x in a]) + '\n')
        print(*a, **kwa)
    return print_

def no_overlap_perms_random(n, length, max_seconds=10, return_elapsed=False):
    """
    Returns n random permutations of range(length) without overlaps using randomness (potential infinite runtime)
    """
    start = time.time()
    no_overlap_perms = []

    for depth in range(n):
        while ...:
            rand_perm = np.random.permutation(np.arange(length))
            if all(not 0 in p - rand_perm for p in no_overlap_perms):
                break
            elif max_seconds:
                if time.time() - start > max_seconds:
                    print("non overlapping ordering search took too long")
                    return ([], time.time() - start) if return_elapsed else []
        no_overlap_perms.append(rand_perm)
    return (no_overlap_perms, time.time() - start) if return_elapsed else no_overlap_perms

def mixup_data_nmix(n_ways, X, y, lam_, device):
    """
    Mixes data n ways with itself shuffled without overlaps, returns X_mixed and an array of every y shuffling
    Example:
        mixup_data(3, X, y, (.5, .5, 0)) will return X_mixed, y[permutation0], y[permutation1], y[permutation2]
        and X_mixed will veriy X_mixed = .5*X[permutation0] + .5*X[permutation1] + 0*X[permutation2]
    X: tensor of shape (batch_size, w, h, 3)
    y: tensor of shape (batch_size,)
    lam_: tensor of shape (n_ways, 1, 1, 1, 1)
    """
    assert lam_.shape[0] == n_ways
    assert X.shape[0] == y.shape[0]
    assert torch.eq(lam_.sum(), torch.tensor([1], dtype=torch.float, device=device)) or\
        torch.isclose(lam_.sum(), torch.tensor([1], dtype=torch.float, device=device)),\
        "Needs to sum to one :" + str(lam_.sum())
    perms = no_overlap_perms_random(n_ways, X.shape[0])
    X_permutations = torch.stack([X[perm] for perm in perms], 0)
    X_mixed = (X_permutations * lam_).sum(axis=0)
    ys = [y[perm] for perm in perms]
    return X_mixed, ys, perms

def mixup_criterion_nmix(criterion, preds, ys, lambda_v):
    return sum(lambda_v[i] * criterion(preds, ys[i]) for i in range(lambda_v.shape[0]))