from parameters import *
from generate_data import *
from train_test import *
from models import *
import itertools as it
import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Project 1: Training and Testing Models.')

    parser.add_argument('--train',
                        action='store_true', default=False,
                        help='Train (default False)')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train:  # train all models
        if not os.path.exists('./stats/'):
            os.makedirs('./stats/')
        if not os.path.exists('./models/'):
            os.makedirs('./models/')

        torch.manual_seed(SEED)
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N_SAMPLES)
        print('### First Strategy: Training Models with CNNs ###')  # First type of network structure
        for i, (arch, kern) in enumerate(it.product(arch_setup, kernel_setup)):
            kernel_conv = [kern] * (len(arch) - 1)
            fc_size = compute_fc_size(arch, kernel_conv)
            loss_train, loss_val, acc_val, acc_test, loss_test, model = repeat_train(
                num_rounds, train_input, train_target, train_classes, test_input, test_target, test_classes,
                arch, kernel_conv, fc_size, device)
            print(f'Architecture {arch} and Kernel {kern}, Average Performance: '
                  f'Train Loss = {loss_train.mean(0)[-1]:.4f}, '
                  f'Val. Accuracy = {acc_val.mean(0)[-1]:.2f}%, '
                  f'Val. Loss = {loss_val.mean(0)[-1]:.4f}')
            torch.save((loss_train, loss_val, acc_val, acc_test, loss_test), f'./stats/stats_{i}.pkl')
            torch.save(model.state_dict(), f'./models/cnn_{i}.pkl')

        torch.manual_seed(SEED)
        print(
            '### Second Strategy: Training Models with CNNs and Weight Sharing ###')  # Second type of network structure
        for i, (arch, kern) in enumerate(it.product(arch_setup, kernel_setup)):
            kernel_conv = [kern] * (len(arch) - 1)
            fc_size = compute_fc_size(arch, kernel_conv)
            loss_train, loss_val, acc_val, acc_test, loss_test, model = repeat_train(
                num_rounds, train_input, train_target, train_classes, test_input, test_target, test_classes,
                arch, kernel_conv, fc_size, device, struc='WS')
            print(f'Architecture {arch} and Kernel {kern}, Average Performance: '
                  f'Train Loss = {loss_train.mean(0)[-1]:.4f}, '
                  f'Val. Accuracy = {acc_val.mean(0)[-1]:.2f}%, '
                  f'Val. Loss = {loss_val.mean(0)[-1]:.4f}')
            torch.save((loss_train, loss_val, acc_val, acc_test, loss_test), f'./stats/stats_ws_{i}.pkl')
            torch.save(model.state_dict(), f'./models/cnn_ws_{i}.pkl')

        torch.manual_seed(SEED)

        print(
            '### Third Strategy: Training Models with CNNs, Weight Sharing and Auxiliary Losses ###')  # Third type of network structure
        for i, (arch, kern) in enumerate(it.product(arch_setup, kernel_setup)):
            kernel_conv = [kern] * (len(arch) - 1)
            fc_size = compute_fc_size(arch, kernel_conv)
            loss_train, loss_val, acc_val, acc_test, loss_test, model = repeat_train(
                num_rounds, train_input, train_target, train_classes, test_input, test_target, test_classes,
                arch, kernel_conv, fc_size, device, rho, struc='AL')
            print(f'Architecture {arch} and Kernel {kern}, Average Performance: '
                  f'Train Loss = {loss_train.mean(0)[-1]:.4f}, '
                  f'Val. Accuracy = {acc_val.mean(0)[-1]:.2f}%, '
                  f'Val. Loss = {loss_val.mean(0)[-1]:.4f}')
            torch.save((loss_train, loss_val, acc_val, acc_test, loss_test), f'./stats/stats_ws_al_{i}.pkl')
            torch.save(model.state_dict(), f'./models/cnn_ws_al_{i}.pkl')


    else:  # train and test best model
        arch = arch_setup[-1]  # best architecture [1, 32, 64, 128]
        kernel_conv = [kernel_setup[1]] * (len(arch) - 1)  # best kernel 5
        fc_size = compute_fc_size(arch, kernel_conv)

        print(f'### Training Best Model: Architecture {arch}, Kernel Size {kernel_conv[0]} ###')

        model = ConvNet_WS_AL(arch, kernel_conv, fc_size)  # best model

        torch.manual_seed(SEED)
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(N_SAMPLES)
        train_loader, val_loader, test_loader = generate_data_loaders_AL(train_input, train_target, train_classes,
                                                                         test_input, test_target, test_classes,
                                                                         batch_size)

        criterion = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

        loss_train, loss_val, acc_val = train_val(model.to(device), train_loader, val_loader, num_epochs, optimizer,
                                                  criterion, scheduler, device, rho, verbose=True)

        print(f'### Testing Best Model: Architecture {arch}, Kernel Size {kernel_conv[0]} ###')

        acc_test, loss_test = validate(model.to('cpu'), test_loader, criterion, 'cpu', rho)

        print(f'Best trained model: Test loss {loss_test:.4f}, Test accuracy: {acc_test:.2f}%')






