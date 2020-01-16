
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
import PIL
import os
import time
import datetime

# For dataset loaders
BATCH_SIZE = 100
NUM_WORKERS = 8

# For training
USE_CUDA = True
N_EPOCHS = 100
N_BATCHES = None
LR = 0.001

def get_print(save_raw=False, res_dir="results", fname="0"):
    """ Makes print function to allow keeping output on disk """
    def print_(*a, **kwa):
        if save_raw:
            with open(os.path.join(res_dir, fname), "a") as f:
                t = datetime.datetime.now()
                f.write(str(t) + '    ' + ' '.join([str(x) for x in a]) + '\n')
        print(*a, **kwa)
    return print_

def get_sets(vision_dataset, root='/data/landelle/datasets'):
    """ Creates trainset and testset for vision_dataset """
    transform = torchvision.transforms.transforms.Compose([torchvision.transforms.transforms.ToTensor()])
    trainset = getattr(torchvision.datasets, vision_dataset)(root=root, train=True, download=True, transform=transform)
    testset = getattr(torchvision.datasets, vision_dataset)(root=root, train=False, download=True, transform=transform)
    return trainset, testset

def get_loader(dset, *, shuffle=True, num_workers=None, batch_size=None):
    """ Creates (DataLoader, N_BATCHES) for the given dataset, N_BATCHES is the # of batches in the loader """
    batch_size = batch_size if batch_size else BATCH_SIZE
    num_workers = num_workers if num_workers else NUM_WORKERS
    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers),\
        len(dset) // batch_size

def show_np_arr(arr, ax=None):
    """ Shows an np array of floats """
    np_img_255 = np.uint8(arr*255)
    im = PIL.Image.fromarray(np.rollaxis(np_img_255.T, axis=1))
    if ax:
        ax.imshow(im)
    else:
        plt.imshow(im)

def show_loader_cf(loader):
    """ Shows the first eight samples of loader (CIFAR10/100) for visual checks """
    fig, axes = plt.subplots(1, 8, figsize=(15, 2))
    for images, labels in loader:
        for i in range(8):
            show_np_arr(images[i], ax=axes[i])
            #axes[i].set_title(trainset.classes[labels[i]])
        break

def show_loader_fm(loader):
    """ Shows the first eight samples of loader (FMNIST) for visual checks """
    fig, axes = plt.subplots(1, 8, figsize=(15, 2))
    for i in range(8):
        show_np_arr(next(iter(loader))[0][i][0, :, :], ax=axes[i])
        
def no_overlap_perms_random(n, length, max_seconds=10, return_elapsed=False):
    """
    Returns n random permutations of range(length) without overlaps using randomness
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

def pixelgraft_data(n_ways, X, y, pr_vec, device):
    """
    
    """
    pr = pr_vec[0]
    assert X.shape[0] == y.shape[0]
    assert 0 <= pr <= 1
    perms = no_overlap_perms_random(2, X.shape[0])
    X_permutations = torch.stack([X[perm] for perm in perms], 0)
    
    # compute the amount of pixels in image
    img_flat_size = X.shape[2] * X.shape[3]
    
    # create a mask that select 100pr% pixels of image 0 and 100(1-pr)% of image 1.
    graft_idx = (torch.rand(img_flat_size).to(device) < pr).int()
        
    lam_ = torch.stack((graft_idx, 1-graft_idx))
    lam_ = lam_.reshape((2, 1, 1, 32, 32)).float()
    
    X_mixed = (X_permutations * lam_).sum(axis=0)
    ys = [y[perm] for perm in perms]
    
    return X_mixed, ys, perms

def train_with_lambdas(model, device, criterion, optimizer, lambda_, trainloader, testloader, train_losses, val_losses, 
                       *, N_BATCHES_TRAIN, print_=print, model_getter=None, n_epochs=None, n_batches=None,
                      mixup_f=mixup_data_nmix):
    """ Computes test accuracy of our model on testloader
    Args:
        model -- the model instance which was trained
        device -- the device used
        criterion -- the loss criterion
        optimizer -- optimization algorithm instance from torch.optim
        trainloader -- DataLoader for the training dataset
        testloader -- DataLoader for the testing dataset
        train_losses -- dict str->[float] where the training losses were recorded into
        val_losses -- dict str->[float] where the validation losses were recorded into
        N_BATCHES_TRAIN -- number of batches within the training DataLoader
        print_ -- print function for prints
        model_getter -- function () -> torch.nn.Module for the ML model used
        n_epochs -- Limit for the # of epochs for training, or 100 by default
        n_batches -- Limit for the # of batches trained on each epoch, or None to train on all of them
        mixup_f -- Function (n, images, labels, lam_rs_tensor, device) -> (images_mx, ys, perms) to mixup samples
    Returns:
        loss_value -- the loss value of this validation to serialize
    """ 
    
    n_batches = n_batches if n_batches else N_BATCHES
    n_epochs = n_epochs if n_epochs else N_EPOCHS
    
    # We convert the lambda_ vector into a correctly shaped tensor, and compute NMIX the dimension of lambda_
    lam_rs = lambda_.reshape((-1, 1, 1, 1, 1))
    lam_rs_tensor = torch.from_numpy(lam_rs).float().to(device)
    NMIX = lam_rs_tensor.shape[0]
    print_("Performing n-mixup with N-mix:", NMIX)   
    
    # Train mode
    model.train()
    for epoch in range(1, n_epochs + 1):
        print_("Epoch[{}/{}]".format(epoch, n_epochs))
        
        for batch_id, (images, labels) in enumerate(trainloader):
            # Stop training at n_batches if specified (useful for debugging etc.)
            if n_batches and batch_id > n_batches:
                break
            labels, images = labels.to(device), images.to(device)
            # Perform mixup
            images_mx, ys, perms = mixup_f(NMIX, images, labels, lam_rs_tensor, device)            
            outputs = model(images_mx)
            loss = mixup_criterion_nmix(criterion, outputs, ys, lam_rs_tensor)
            loss_value = loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record training loss
            train_losses[str(lambda_)].append(loss_value)
            
            # Every 50 batches, print progress
            if batch_id % 50 == 0:
                print_('train loss:{:.4f} Epoch[{}/{}] Batch[{}/{}]'.format(
                    loss_value, epoch, n_epochs, batch_id, N_BATCHES_TRAIN))
                
        # Compute validation loss for this epoch
        # Note: due to a (fixed) code mistake, the val loss is wrong for the results that were generated.
        # As such, the val loss was not used/discussed within the report
        val_loss = validate_with_lambdas(model, device, criterion, testloader, epoch, n_epochs, print_=print_)
        val_losses[str(lambda_)].append(val_loss)
        
    return train_losses, val_losses, images_mx, ys, perms
    
def validate_with_lambdas(model, device, criterion, testloader, epoch, n_epochs, *, print_=print):
    """ Computes test accuracy of our model on testloader
    Args:
        model -- the model instance which was trained
        device -- the device used
        criterion -- the loss criterion
        testloader -- DataLoader for the testing dataset
        epoch -- int for the current epoch validating
        n_epochs -- int for the total # of epochs of this training
        print_ -- print function for prints
    Returns:
        loss_value -- the loss value of this validation to serialize
    """ 
    
    # Validating on one batch    
    for images, labels in testloader:
        correct = 0
        total = 0
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        loss = criterion(outputs, labels)
        loss_value = loss.item()

        print_('val loss:{:.4f} Epoch[{}/{}]'.format(
                loss_value, epoch, n_epochs))
        print_('val acc:{:.4f} Epoch[{}/{}]'.format(
                val_acc, epoch, n_epochs))
        return loss_value

def test(model, device, lambda_, testloader, test_accs, *, print_=print):
    """ Computes test accuracy of our model on testloader
    Args:
        model -- the model instance which was trained
        device -- the device used
        lambda_ -- the lambda_ used to train model, to use as key in test_accs
        testloader -- DataLoader for the testing dataset
        test_accs -- dict str->float mapping lambda_ str repr. to resulting test accuracy
        print_ -- print function for prints
    Returns:
        test_accs -- dict str->float where the test accuracy was recorded into
    """ 
    # Test mode
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
        test_accs[str(lambda_)] = test_acc
    return test_accs

def perform_experiment(trainloader, testloader, lambdas, results_fname, *,
                       N_BATCHES_TRAIN, model_getter, n_epochs=100, n_batches=None,
                       mixup_f=mixup_data_nmix):
    """
    Performs an experiment with loaders on every lambda_ in lambdas,
    and serializes results in results/results_fname
    Args:
        trainloader -- DataLoader for the training dataset
        testloader -- DataLoader for the training dataset
        lambdas -- DataLoader for the training dataset
        results_fname -- filename for the results file (results/ needs to exist)
        N_BATCHES_TRAIN -- number of batches within the training DataLoader
        model_getter -- function () -> torch.nn.Module for the ML model used
        n_epochs -- Limit for the # of epochs for training, or 100 by default
        n_batches -- Limit for the # of batches trained on each epoch, or None to train on all of them
        mixup_f -- Function (n, images, labels, lam_rs_tensor, device) -> (images_mx, ys, perms) to mixup samples
    Returns:
        test_accs -- dict str->float where the test accuracies were recorded into
        train_losses -- dict str->[float] where the train losses were recorded into
        val_losses -- dict str->[float] where the validation losses were recorded into
        images_mx -- sample image mixed samples of the last batch
        ys -- sample train targets of the last batch
        perms -- sample indices permutations for the last batch
    """
    # Generates the print_ function to dump outputs in a result file for later analysis
    print("Saving results in", results_fname)
    print_ = get_print(True, fname=results_fname)
    
    # Makes dicts to collect losses and accuracies for this experiment
    test_accs = {}
    train_losses = {str(k):[] for k in lambdas}
    val_losses = {str(k):[] for k in lambdas}    
    
    for lambda_ in lambdas:
        print_("Trying lambda=", lambda_)
        
        # device, model, optimizer and criterion remain during this lambda
        device = torch.device('cuda' if USE_CUDA else 'cpu')
        model = model_getter().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = torch.nn.CrossEntropyLoss()
        
        
        train_losses, val_losses, images_mx, ys, perms = train_with_lambdas(
            model, device, criterion, optimizer,
            lambda_, trainloader, testloader, train_losses, val_losses,
            N_BATCHES_TRAIN=N_BATCHES_TRAIN, print_=print_,
            model_getter=model_getter, n_epochs=n_epochs, n_batches=n_batches,
            mixup_f=mixup_f)
        
        test_accs = test(model, device, lambda_, testloader, test_accs, print_=print_)

        # Now we delete instances just to enforce that no data leak between lambdas
        # We want to ensure the model for the next lambdas is not pretrained by the previous lambda
        del device, model, optimizer, criterion
        
    return test_accs, train_losses, val_losses, images_mx, ys, perms

