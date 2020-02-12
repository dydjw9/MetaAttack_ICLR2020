from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import pdb
import os
import argparse
import imagenet_models
from tiny_imagenet import TinyImageNet200
from utils import save_gradient
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

cudnn.benchmark = True
torch.manual_seed(1)
if __name__ == '__main__':
    # Load checkpoints and generate gradient map.
    models_name = ['vgg13_bn', 'vgg16_bn', 'vgg_19_bn', 'resnet18', 'resnet34', 'resnet50']
    
    transform_train = transforms.Compose([
                      transforms.ToTensor(),
                      ])

    transform_test = transforms.Compose([
                      transforms.ToTensor(),
                      ])

    trainset = TinyImageNet200(root='~/dataset', type = 'train', transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 100, shuffle = True, num_workers = 2)

    testset = TinyImageNet200(root='~/dataset', type = 'val', transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 100, shuffle = False, num_workers = 2)

    for i in range(len(models_name)):
        model_name = models_name[i]
        net = imagenet_models.__dict__[model_name]()
        print('==> Resuming from checkpoint..')
        #pdb.set_trace() 
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        
        model_checkpoint_path = './checkpoint/' + model_name + '_adam' + '/model_best.pth.tar'
        print('Loading checkpoint from %s' % model_checkpoint_path)
        checkpoint = torch.load(model_checkpoint_path)
        net.load_state_dict(checkpoint['state_dict'])
        net = net.to(device)
        best_acc = checkpoint['best_prec1']
        print('Best test accuracy', best_acc.data)
        save_gradient(net, device, trainloader, model_name, mode = 'train')
        save_gradient(net, device, testloader, model_name, mode = 'test')   
