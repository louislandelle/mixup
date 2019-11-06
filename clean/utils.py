import sys
sys.path.append("/home/landelle/pytorch-cifar/models")
PATH_TO_EFFICIENT_NET = "/home/landelle/EfficientNet-PyTorch"
sys.path.append(PATH_TO_EFFICIENT_NET)

import time
import random
import numpy as np
import pandas as pd
import torch
import torchvision
import datetime
import PIL
import matplotlib.pyplot as plt

# Local files
import resnet
import main


STATE_DICTS_PATH = "/home/landelle/state_dicts/"
STATE_DICTS_PREFIX = "state_dict_resnet18"

def mixup_data_nmix(n_ways, X, y, lam_):
    """
    Mixes data n ways with itself shuffled without overlaps, returns X_mixed and an array of every y shuffling
    Example:
        mixup_data(3, X, y, (.5, .5, 0)) will return X_mixed, y[permutation0], y[permutation1], y[permutation2]
        and X_mixed will veriy X_mixed = .5*X[permutation0] + .5*X[permutation1] + 0*X[permutation2]
    X: np.arr of shape (batch_size, w, h)
    y: np.arr of shape (batch_size,)
    lam_: np.arr of shape (n_ways, )
    """
    assert lam_.shape[0] == n_ways
    assert X.shape[0] == y.shape[0]
    assert np.isclose(lam_.sum(), 1)
    lam_rs = lam_.reshape((n_ways, 1, 1, 1, 1))
    perms = no_overlap_perms_random(n_ways, X.shape[0])
    X_permutations = np.array([X[perm].numpy() for perm in perms])
    X_mixed = (X_permutations * lam_rs).sum(axis=0)
    ys = [y[perm] for perm in perms]
    return torch.from_numpy(X_mixed).float(), ys, perms

def mixup_criterion_nmix(criterion, preds, ys, lambda_v):
    return [lambda_v[i] * criterion(preds[i], ys[i]) for i in range(lambda_v.shape[0])]

def get_model():
    """ Get model is used to remake the model between lambda attempts """
    return resnet.ResNet18()

def show_np_array(arr):
    np_img_255 = np.uint8(arr*255)
    im = PIL.Image.fromarray(np.rollaxis(np_img_255.T, axis=1))
    plt.imshow(im)

def get_print(save_raw=False, f_path="results"):
    """ Makes print function to allow keeping output on disk """
    def print_(*a, **kwa):
        if save_raw:
            with open(f_path, "a") as f:
                t = datetime.datetime.now()
                f.write(str(t) + '    ' + ' '.join([str(x) for x in a]) + '\n')
        print(*a, **kwa)
    return print_


def train(device, model, optimizer, criterion, lam, trainloader, *, 
          n_epochs=None, n_batches=None,
          save_raw=False, save_state_dicts="", save_tensorboard="",
         LR=.01, TRAIN_EPOCHS=50, N_BATCHES_IN_TRAIN_SET=500):
    """
    save_raw: save raw results (also print to external file) 
    save_state_dicts: No|Override|Separate, saves state_dicts after an epoch
    save_tensorboard: saves tensorboard data
    """
    print_ = get_print(save_raw)
    
    # switch to train mode
    model.train()
    n_epochs = n_epochs if n_epochs else TRAIN_EPOCHS
    for epoch in range(1, n_epochs + 1):
        print_("Epoch[{}/{}]".format(epoch, n_epochs))
        for batch_id, (images, labels) in enumerate(trainloader):
            if n_batches and batch_id > n_batches:
                break            
            labels, images = labels.to(device), images.to(device)
            images, labels_a, labels_b, lam = mixup_data(images, labels, lam)
            outputs = model(images)
            loss = main.mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 50 == 0:
                print_('Loss :{:.4f} Epoch[{}/{}] Batch[{}/{}] batch_shape:{}'.format(
                    loss.item(), epoch, n_epochs, batch_id, N_BATCHES_IN_TRAIN_SET, images.shape))
        # Each epoch if enabled: save state dicts
        if save_state_dicts=="Separate":
            torch.save(model.state_dict(), STATE_DICTS_DIR + STATE_DICTS_PREFIX + "_epoch_" + str(epoch))
        elif save_state_dicts=="Override":
            torch.save(model.state_dict(), STATE_DICTS_DIR + STATE_DICTS_PREFIX + "_override")

def train_nmix(device, model, optimizer, criterion, lam_vec, trainloader, *, 
          n_epochs=None, n_batches=None,
          save_raw=False, save_state_dicts="", save_tensorboard="",
         LR=.01, TRAIN_EPOCHS=50, N_BATCHES_IN_TRAIN_SET=500):
    """
    save_raw: save raw results (also print to external file) 
    save_state_dicts: No|Override|Separate, saves state_dicts after an epoch
    save_tensorboard: saves tensorboard data
    """
    print_ = get_print(save_raw)
    NMIX = lam_vec.shape[0]
    print_("Performing n-mixup with N-mix:", NMIX)
    
    # switch to train mode
    model.train()
    n_epochs = n_epochs if n_epochs else TRAIN_EPOCHS
    for epoch in range(1, n_epochs + 1):
        print_("Epoch[{}/{}]".format(epoch, n_epochs))
        for batch_id, (images, labels) in enumerate(trainloader):
            if n_batches and batch_id > n_batches:
                break
            labels, images = labels.to(device), images.to(device)
            images, ys, perms = mixup_data_nmix(NMIX, images, labels, lam_vec)
            outputs = model(images.float())
            loss = mixup_criterion_nmix(criterion, outputs, ys, lam_vec)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 50 == 0:
                print_('Loss :{:.4f} Epoch[{}/{}] Batch[{}/{}] batch_shape:{}'.format(
                    loss.item(), epoch, n_epochs, batch_id, N_BATCHES_IN_TRAIN_SET, images.shape))
        # Each epoch if enabled: save state dicts
        if save_state_dicts=="Separate":
            torch.save(model.state_dict(), STATE_DICTS_DIR + STATE_DICTS_PREFIX + "_epoch_" + str(epoch))
        elif save_state_dicts=="Override":
            torch.save(model.state_dict(), STATE_DICTS_DIR + STATE_DICTS_PREFIX + "_override")

def test(device, model, testloader, *, save_raw=False):
    print_ = get_print(save_raw)
        
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Output test accuracy
        test_acc = 100 * correct / total
        print_('Test Accuracy of the model on the test images: {} %'.format(test_acc))
        return test_acc
    
def collect_results(trainloader, testloader, *, model_getter=get_model, n_epochs=10, n_batches=None, half=False, no_mixup=False,
                    save_raw=False, save_state_dicts=False, save_tensorboard=False, custom_lambdas=[], 
         LR=.01, TRAIN_EPOCHS=50, N_BATCHES_IN_TRAIN_SET=500, use_cuda=True):
    """
    n_batches: None trains on whole dataset otherwise it trains on n_batches of BATCH_SIZE per epoch    
    
    """
    USE_CUDA = use_cuda
    print_ = get_print(save_raw)

    test_accs = {}
    for lambda_ in custom_lambdas if custom_lambdas else [x*.1 for x in range(11)]:
        if half and lambda_>.5:
            break
        if no_mixup and lambda_>0:
            break
        print_("Trying lambda=", lambda_)        

        device = torch.device('cuda' if USE_CUDA else 'cpu')
        #model = models.My_Model(NUM_CLASS).to(device)
        #model = EfficientNet.from_pretrained('efficientnet-b0').to(device)
        model = model_getter().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()

        train(device, model, optimizer, criterion, lambda_, trainloader,
              n_epochs=n_epochs, n_batches=n_batches,
              save_raw=save_raw, save_state_dicts=save_state_dicts, save_tensorboard=save_tensorboard,
             LR=LR, TRAIN_EPOCHS=TRAIN_EPOCHS, N_BATCHES_IN_TRAIN_SET=N_BATCHES_IN_TRAIN_SET)
        test_accs[lambda_] = test(device, model, testloader, save_raw=save_raw)

        del device, model, optimizer, criterion
    test_accs_dict = {str(k)[:3]:[v] for k, v in test_accs.items()}
    test_accs_df = pd.DataFrame.from_dict(test_accs_dict).transpose()
    test_accs_df.reset_index(inplace=True)
    test_accs_df.rename(columns={0:'Test accuracy', 'index':'Lambda'}, inplace=True)
    print_(test_accs_df)
    return test_accs_df
 

def no_overlap_perms_random(n, length, max_seconds=10, return_elapsed=False):
    """ Returns n random permutations of range(length) without overlaps using randomness (potential infinite runtime) """
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

def mixup_data(n_ways, X, y, lam_):
    """
    Mixes data n ways with itself shuffled without overlaps, returns X_mixed and an array of every y shuffling
    Example:
        mixup_data(3, X, y, (.5, .5, 0)) will return X_mixed, y[permutation0], y[permutation1], y[permutation2]
        and X_mixed will veriy X_mixed = .5*X[permutation0] + .5*X[permutation1] + 0*X[permutation2]
    X: np.arr of shape (batch_size, w, h)
    y: np.arr of shape (batch_size,)
    lam_: np.arr of shape (n_ways, )
    """
    assert lam_.shape[0] == n_ways
    assert X.shape[0] == y.shape[0]
    assert lam_.sum() == 1
    lam_rs = lam_.reshape((n_ways, 1, 1, 1))
    perms = util.no_overlap_perms_random(n_ways, X.shape[0])
    X_permutations = np.array([X[perm] for perm in perms])
    X_mixed = (X_permutations * lam_rs).sum(axis=0)
    ys = [y[perm] for perm in perms]
    return X_mixed, ys
    
    
