import torch
import os
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset, random_split
from parameters import *


def generate_data_loaders(train_data, train_targets, test_data, test_targets, batch_size):
    '''
    Normalize data, split training and validation dataset,
    generate training, test, validation Datasets and Dataloaders.

    Parameters
    ----------
    train_data: FloatTensor
        digits tensors for the training data
    train_targets: FloatTensor
        targets for the training data
    test_data: FloatTensor
        digits tensors for the test data
    test_targets: FloatTensor
        targets for the test data
    batch_size: int
        size of batches

    Returns
    -------
    train_loader: DataLoader
        training DataLoader
    val_loader: DataLoader
        validation DataLoader
    test_loader: DataLoader
        test DataLoader
    '''
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    train_ds = TensorDataset(train_data, train_targets)
    test_ds = TensorDataset(test_data, test_targets)

    train_ds, val_ds= random_split(train_ds, [N_SAMPLES - N_VAL, N_VAL])

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size = N_VAL, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def generate_data_loaders_AL(train_data, train_targets, train_classes,
                             test_data, test_targets, test_classes, batch_size):
    '''
    Normalize data, split training and validation dataset, include individual digits classes,
    generate training, test, validation Datasets and Dataloaders
    for using auxiliary losses.

    Parameters
    ----------
    train_data: FloatTensor
        digits tensors for the training data
    train_targets: FloatTensor
        targets for the training data
    train_classes: FloatTensor
        2 digits classes
    test_data: FloatTensor
        digits tensors for the test data
    test_targets: FloatTensor
        targets for the test data
    batch_size: int
        size of batches

    Returns
    -------
    train_loader: DataLoader
        training DataLoader
    val_loader: DataLoader
        validation DataLoader
    test_loader: DataLoader
        test DataLoader
    '''
    train_data = normalize_data(train_data)
    test_data = normalize_data(test_data)

    train_targets = torch.cat((train_targets[:, None], train_classes), 1)
    test_targets = torch.cat((test_targets[:, None], test_classes), 1)

    train_ds = TensorDataset(train_data, train_targets)
    test_ds = TensorDataset(test_data, test_targets)

    train_ds, val_ds= random_split(train_ds, [N_SAMPLES - N_VAL, N_VAL])

    train_loader = DataLoader(train_ds, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size = N_VAL, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size = batch_size, shuffle=True)

    return train_loader, val_loader, test_loader


def normalize_data(data):
    '''
    Normalize tensor along axis 0.

    Parameters
    ----------
    data: FloatTensor
        tensor to be normalized
    '''
    mu, std = data.mean(), data.std()
    return data.sub_(mu).div_(std)


def mnist_to_pairs(nb, input, target): # taken as-is from dlc_practical_prologue.py
    '''
    Generate the pairs of input tensors,
    generate the labels referring to the comparison between individual targets.

    Parameters
    ----------
    nb: int
        number of pairs of images
    input: FloatTensor
        input tensors
    target: FloatTensor
        target tensors

    Returns
    -------
    input: FloatTensor
        new input with concatenated pairs
    target:FloatTensor
        new target with comparison label
    classes:
        individual labels

    '''
    input = torch.functional.F.avg_pool2d(input, kernel_size = 2)
    a = torch.randperm(input.size(0))
    a = a[:2 * nb].view(nb, 2)
    input = torch.cat((input[a[:, 0]], input[a[:, 1]]), 1)
    classes = target[a]
    target = (classes[:, 0] <= classes[:, 1]).long()
    return input, target, classes


def generate_pair_sets(nb): # adapted from dlc_practical_prologue.py
    '''
    Generate pairs of digits from the MNIST dataset.

    Parameters
    ----------
    nb: int
        size of set

    '''
    data_dir = os.environ.get('PYTORCH_DATA_DIR')
    if data_dir is None:
        data_dir = './data'

    train_set = datasets.MNIST(data_dir + '/mnist/', train = True, download = True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets

    test_set = datasets.MNIST(data_dir + '/mnist/', train = False, download = True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets

    return mnist_to_pairs(nb, train_input, train_target) + \
           mnist_to_pairs(nb, test_input, test_target)