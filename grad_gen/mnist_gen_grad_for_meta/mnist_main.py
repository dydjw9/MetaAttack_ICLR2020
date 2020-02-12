import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import mnist_model
from utils import train, test, save_gradient, save_gradient_zoo
import os
import numpy as np
import pdb
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def main():
    parser = argparse.ArgumentParser(description = 'test different models on Mnist')
    parser.add_argument('--batch_size', type = int, default = 100,
                        help = 'input batch size for training (default: 100)')
    parser.add_argument('--test_batch_size', type = int, default = 100,
                        help = 'input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type = int, default = 50, 
                        help = 'number of epochs to train (default: 15)')
    parser.add_argument('--lr', type = float, default = 0.01,
                        help = 'learning rate (default: 0.01)')
    parser.add_argument('--momentum', type = float, default = 0.9,
                        help = 'SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action = 'store_true', default = False,
                        help = 'disable CUDA training')
    parser.add_argument('--seed', type = int, default = 1, 
                        help = 'random seed (default: 1)')
    parser.add_argument('--log_interval', type = int, default = 10,
                        help = 'how many batches to wait before logging training status')
    parser.add_argument('--save_model', action = 'store_true', default = False,
                        help = 'for saving the current model')
    parser.add_argument('--resume', action = 'store_true', default = True,
                        help = 'resume from existing models directly without training')

    args = parser.parse_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', 
                train = True, 
                download = True, 
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])),
                batch_size = args.batch_size, 
                shuffle = False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', 
                train = False, 
                download = True, 
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])),
                batch_size = args.test_batch_size, 
                shuffle = False, **kwargs)

    #models = ['LeNet5', 'Net1', 'Net2', 'Net3', 'Net4', 'Net5', 'Net6']
    models = ['LeNet5', 'Net2', 'Net3']
    for i in range(len(models)):
        model = mnist_model.__dict__[models[i]]()
        model = model.to(device)
        
        if args.resume:
            model_checkpoint_path = './checkpoint/mnist/' + 'mnist_' + models[i] + '.pt'
            assert os.path.exists(model_checkpoint_path)
            model.load_state_dict(torch.load(model_checkpoint_path))    
            test(args, model, device, test_loader, criterion)
            
            start = time.time()
#           save_gradient_zoo(args, model, device, train_loader, criterion, mode = 'train')
            print('Time is', time.time()-start)
            save_gradient(args, model, device, test_loader, criterion, mode = 'test')

        else:
            optimizer = optim.SGD(model.parameters(), lr = args.lr, weight_decay = 1e-6, momentum = args.momentum, nesterov = True)
            
            for epoch in range(args.epochs):
                train(args, model, device, train_loader, optimizer, epoch, criterion)
                test(args, model, device, test_loader, criterion)
            
            save_gradient(args, model, device, train_loader, criterion, mode = 'train')
            save_gradient(args, model, device, test_loader, criterion, mode = 'test')
            
            if args.save_model:

                model_file_path = './checkpoint/mnist'
                if not os.path.exists(model_file_path):
                    os.makedirs(model_file_path)
            
                file_name = 'mnist_'+ models[i] + '.pt'
                save_model_path = os.path.join(model_file_path, file_name)
                torch.save(model.state_dict(), save_model_path)

if __name__ == '__main__':
    main()
