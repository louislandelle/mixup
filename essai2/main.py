

# Get dataloaders
import dataset
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BATCH_SIZE = 50
LR = 0.005
NUM_CLASS = 10
IMAGE_SIZE = 28
CHANNEL = 1
Train_epoch = 5

def get_dataloader(csv_fpath):
    df = pd.read_csv(csv_fpath)
    My_transform = transforms.Compose([transforms.ToTensor(),])
    t = dataset.MyDataset
    data = t(df, transform=My_transform)
    dataloader = DataLoader(dataset=data, batch_size=BATCH_SIZE, shuffle=True)
    return dataloader


# Train model on pre-gen dataset
import models
from torch import nn
import numpy as np

def mixup_data(x, y, lam):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

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
    assert lam_.sum() == 1
    lam_rs = lam_.reshape((n_ways, 1, 1, 1, 1))
    perms = util.no_overlap_perms_random(n_ways, X.shape[0])
    X_permutations = np.array([X[perm] for perm in perms])
    X_mixed = (X_permutations * lam_rs).sum(axis=0)
    ys = [y[perm] for perm in perms]
    return X_mixed, ys, perms
    
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def mixup_criterion_nmix(criterion, preds, ys, lambda_v):
    return [lambda_v[i] * criterion(preds[i], ys[i]) for i in range(lambda_v.shape[0])]

def train(device, model, optimizer, criterion, lam, verbose=True):
    print("start1")
    for epoch in range(1, Train_epoch + 1):
        if verbose:
            print("epoch", epoch)
            print("epoch", epoch)
        MAX_BATCH_ID = 60000 // 50
        for batch_id, (labels, images) in enumerate(train_dataloader):
            if batch_id == MAX_BATCH_ID: break
            labels, images = labels.to(device), images.to(device)

            images, labels_a, labels_b, lam = mixup_data(images, labels, lam)
            #images, labels_a, labels_b = map(Variable, (images, labels_a, labels_b))
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if verbose and batch_id % 100 == 0:
                print('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, Train_epoch))

# Test model
def test(device, model, verbose=True):
    with torch.no_grad():
        correct = 0
        total = 0
        for labels, images in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        # Output test accuracy
        test_acc = 100 * correct / total
        if verbose:
            print('Test Accuracy of the model on the test images: {} %'.format(test_acc))
        return test_acc

def main():
    test_accs = {}
    for lambda_ in [x*.1 for x in range(11)]:
        print("Trying lambda =", lambda_)

        device = torch.device('cpu')
        model = models.My_Model(NUM_CLASS).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss() # vs Adams??

        train(device, model, optimizer, criterion, lambda_)
        test_accs[lambda_] = test(device, model)

        del device, model, optimizer, criterion

    print(test_accs)
if __name__=="__main__":
    train_dataloader = dataset.get_dataloader("fashion-mnist_train.csv")
    test_dataloader = dataset.get_dataloader("fashion-mnist_test.csv")
    main()
