import torch
from parameters import *
from generate_data import *
from models import *

def train_val(model, train_loader, val_loader, nb_epochs, optimizer, criterion, scheduler, device, rho, verbose = False):
    ''' Trains a model using one (or many) optimization criterion (criteria)
    for a certain number of epochs.
    Returns validation and training performance indicators over epochs.

    Parameters
    ----------
    model: torch.nn.Module
        model to be trained
    train_loader: DataLoader
        Training loader
    val_loader: DataLoader
        Validation loader
    nb_epochs: int
        Number of training epochs
    optimizer: PyTorch optimizer
        PyTorch optimizer
    criterion: PyTorch criterion
        PyTorch criterion
    scheduler: PyTorch learning rate scheduler
        PyTorch learning rate scheduler
    device: PyTorch device
        PyTorch device
    rho: list(float)
        Loss function weights

    Returns
    -------
    loss_train: list(FloatTensor)
        training loss over epochs
    loss_val: list(FloatTensor)
        validation loss over epochs
    acc_val: list(FloatTensor)
        validation accuracy over epochs
    '''
    loss_train = []
    loss_val = []
    acc_val = []
    for e in range(num_epochs):
        for feat, target in train_loader:
            output = model(feat.to(device))
            target = target.to(device)
            if len(criterion) > 1:
                loss = sum([rho[j] * criterion[j](x, target[:, j]) for j, x in enumerate(output)])
            else:
                loss = criterion[0](output, target.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc, loss_v = validate(model, val_loader, criterion, device, rho)
        if verbose:
            print(f'Epoch {e}: Train Loss {loss:.4f}, Validation Loss {loss_v:.4f}, Validation Accuracy {acc:.2f}')
        loss_train.append(loss)
        loss_val.append(loss_v)
        acc_val.append(acc)
        scheduler.step()

    return loss_train, loss_val, acc_val

def validate(model, val_loader, criterion, device, rho = None):
    '''
    Evaluate the model on the validation dataset.

    Parameters
    ----------
    model: torch.nn.Module
        model to be evaluated
    val_loader: DataLoader
        Validation loader
    criterion: PyTorch criterion
        PyTorch criterion
    device: PyTorch device
        PyTorch device
    rho: list(float)
        Loss function weights

    Returns
    -------
    acc: FloatTensor
        validation accuracy
    loss: FloatTensor
        validation loss
    '''
    correct = 0
    total = 0
    for feats, target in val_loader:
        with torch.no_grad():
            outputs = model(feats.to(device))
            target = target.to(device)
        if len(criterion) > 1:
            loss = sum([rho[j] * criterion[j](x, target[:, j]) for j, x in enumerate(outputs)])
            _, predicted = torch.max(outputs[0].data, 1)
            correct += (predicted == target[:, 0]).sum()
        else:
            loss = criterion[0](outputs, target)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target).sum()
        total += target.size(0)
    acc = torch.true_divide(100 * correct, total)
    return acc, loss


def repeat_train(n_rounds, train_input, train_target, train_classes,
                 test_input, test_target, test_classes, arch, kernel_conv, fc_size,
                 device, rho = None, struc = 'None'):
    ''' Repeats training and testing for a certain number of rounds.

    Parameters
    ----------
    model: torch.nn.Module
        model to be trained
    train_input: FloatTensor
        training input
    train_target: FloatTensor
        training comparison targets
    train_classes: FloatTensor
        training individual classes
    test_input: FloatTensor
        test input
    test_target: FloatTensor
        test comparison targets
    test_classes: FloatTensor
        test individual classes
    arch : list (int)
        a list of int values representing the number of channels
    kernel_conv : list (int)
        a list of int values representing the size of the convolution kernel
    fc_size: int
        size of the fully connected layer
    device: PyTorch device
        PyTorch device
    rho: list(float)
        Loss function weights
    struc: string
        'None' for two independent CNNs, 'WS' to include weight sharing, 'AL' to include auxiliary losses.

    Returns
    -------
    loss_train_t: FloatTensor
        2D Tensor with training loss computed over epochs and repetitions
    loss_val_t: FloatTensor
        2D Tensor with validation loss computed over epochs and repetitions
    acc_val_t: FloatTensor
        2D Tensor with validation accuracy computed over epochs and repetitions
    acc_test_t: FloatTensor
        1D Tensor with test accuracy computed over repetitions
    loss_test_t: FloatTensor
        1D Tensor with test loss computed over repetitions
    model: torch.nn.Module
        trained model
    '''
    loss_train_t, loss_val_t, acc_val_t = torch.zeros(n_rounds, num_epochs), \
                                          torch.zeros(n_rounds, num_epochs), torch.zeros(n_rounds, num_epochs)
    loss_test_t, acc_test_t = torch.zeros(n_rounds), torch.zeros(n_rounds)

    for i in range(n_rounds):
        if struc == 'None':
            model = ConvNet2(arch, kernel_conv, fc_size)
        elif struc == 'WS':
            model = ConvNet_WS(arch, kernel_conv, fc_size)
        else:
            model = ConvNet_WS_AL(arch, kernel_conv, fc_size)

        if struc == 'AL':
            train_loader, val_loader, test_loader = generate_data_loaders_AL(train_input, train_target, train_classes,
                                                                             test_input, test_target,
                                                                             test_classes, batch_size)
            criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]

        else:
            train_loader, val_loader, test_loader = generate_data_loaders(train_input, train_target, test_input,
                                                                      test_target, batch_size)
            criterion = [nn.CrossEntropyLoss()]

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        loss_train, loss_val, acc_val = train_val(model.to(device), train_loader, val_loader, num_epochs, optimizer,
                                                  criterion, scheduler, device, rho)
        acc_test, loss_test = validate(model.to('cpu'), test_loader, criterion, 'cpu', rho)
        print(f'Architecture {arch} and Kernel {kernel_conv[0]}, Run {i+1}: '
                  f'Train Loss = {loss_train[-1]:.4f}, '
                  f'Val. Accuracy = {acc_val[-1]:.2f}%, '
                  f'Val. Loss = {loss_val[-1]:.4f}')


        loss_train_t[i] = torch.tensor(loss_train)
        loss_val_t[i] = torch.tensor(loss_val)
        acc_val_t[i] = torch.tensor(acc_val)
        loss_test_t[i] = loss_test
        acc_test_t[i] = acc_test

    return loss_train_t, loss_val_t, acc_val_t, acc_test_t, loss_test_t, model

def compute_fc_size(arch, kernel_conv):
    '''
    Compute fully connected layer size given the chosen architecture and convolution kernel size.

    Parameters
    ----------
    arch: list(int)
    kernel_conv: list(int)

    Returns
    -------
    : int
        size of fully connected layer.
    '''
    s = img_size
    for i in range(len(arch) - 1):
        s = (s - kernel_conv[i] + 1)
    return arch[-1] * s ** 2