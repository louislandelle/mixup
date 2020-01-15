
import numpy as np
import pandas as pd
import datetime


def load_losses(fpath):
    dfs = {}

    with open(fpath, "r") as fp:
        lines = fp.readlines()
        
    # Split lines for each experiment in res_split
    res_split = []
    lines_acc = []
    for l in res:
        if 'Trying' in l and len(lines_acc)>2:
            res_split.append(lines_acc)
            lines_acc = []
        lines_acc.append(l)
    res_split.append(lines_acc)

    # Foreach experiment
    for exp_lines in res_split:
        n_batch = None
        n_epoch = None
        batch_step = None
        lambda_str = None

        for l in exp_lines:
            if 'Trying lambda=' in l or 'Trying pr=' in l:
                lambda_str = l.split('= ')[1].replace('\n', '')
            if 'Epoch[' in l:
                n_epoch = int(l.split('Epoch[')[1].split('/')[1].split(']')[0])
            if not n_batch:
                if 'Batch[' in l:
                    n_batch = int(l.split('Batch[')[1].split('/')[1].split(']')[0])
            else:
                if 'Batch[' in l:
                    batch_step = int(l.split('Batch[')[1].split('/')[0])        
            if n_batch and n_epoch and batch_step:
                break

        lines_per_epoch = n_batch // batch_step

        df = pd.DataFrame(
            columns=["epoch", "batch", "train loss", "val loss"],
            index=range(n_epoch*lines_per_epoch))

        epochs_indices = np.arange(1, n_epoch+1)
        batch_indices = np.arange(0, n_batch, batch_step)
        df["epoch"] = np.repeat(epochs_indices, lines_per_epoch)
        df["batch"] = np.tile(batch_indices, n_epoch)

        tr_losses = []
        val_losses = []

        get_epoch = lambda l: int(l.split('Epoch[')[1].split('/')[0])
        get_batch = lambda l: int(l.split('Batch[')[1].split('/')[0])

        for l in exp_lines:
            if 'train loss:' in l:
                tr_loss = l.split('train loss:')[1].split(' Epoch')[0]
                tr_losses.append(float(tr_loss))
            elif 'val loss:' in l:
                val_loss = l.split('val loss:')[1].split(' Epoch')[0]
                val_losses.append(float(val_loss))


        tr_losses = np.array(tr_losses)
        val_losses = np.array(val_losses[1:])

        df["train loss"] = tr_losses
        df["val loss"] = np.repeat(val_losses, lines_per_epoch)

        dfs[lambda_str]=df
        
    return dfs

def load_testaccs(fpath):
    """ Used to load testaccs with unique lambdas as keys """
    with open(fpath, "r") as fp:
        lines = fp.readlines()
    current_lam = None
    accs = {}
    for l in lines:
        if "Trying lambda" in l or "Trying pr" in l:
            current_lam = l.split('= ')[1]                    
        if "Test Accuracy" in l:
            accs[current_lam[:-1]] = float(l[-8:-3])
    return accs

def load_testaccs_duplicates(fpath):
    """ Used to load testaccs when there are multiple results for a single lambda key """
    with open(fpath, "r") as fp:
        lines = fp.readlines()
    current_lam = None
    accs = []
    for l in lines:
        if "Trying lambda" in l or "Trying pr" in l:
            current_lam = l.split('= ')[1]                    
        if "Test Accuracy" in l:
            accs.append(float(l[-8:-3]))
    return current_lam, accs

def runtime(fpath):
    """ Returns the str repr. of the running time of results file at fpath """
    with open(fpath, "r") as fp:
        lines = fp.readlines()
    fmt = '%Y-%m-%d %H:%M:%S.%f'
    dt = lambda idx: datetime.datetime.strptime(' '.join(lines[idx].split(' ')[:2]), fmt)
    return str(dt(-1) - dt(0))