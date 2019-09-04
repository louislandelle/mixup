

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

train_dataloader = dataset.get_dataloader("fashion-mnist_train.csv")
test_dataloader = dataset.get_dataloader("fashion-mnist_test.csv")

# Train model on pre-gen dataset
import models
from torch import nn

def mixup_data(x, y, lam):
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(device, model, optimizer, criterion, lam, verbose=True):
    print("start")
    for epoch in range(1, Train_epoch + 1):
        if verbose:
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
