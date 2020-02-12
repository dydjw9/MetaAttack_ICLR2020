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

from cifar_models import *
from utils import save_gradient
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default = 0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action = 'store_true', default = False, help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if not args.resume:
    print('==> Preparing data..')
    class LeNormalize(object):
        '''
            normalize -1 to 1
        '''
        def __call__(self, tensor):
            for t in tensor:
                t.sub_(0.5).mul_(2.0)
            return tensor

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #LeNormalize(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #LeNormalize(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 128, shuffle = True, num_workers = 2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size = 100, shuffle = False, num_workers = 2)

# Model
print('==> Building model..')
#net = VGG('VGG16')
#net = ResNet34()
#net = PreActResNet18()
#net = GoogLeNet()
#net = DenseNet121()
#net = ResNeXt29_2x64d()
#net = MobileNet()
#net = MobileNetV2()
#net = DPN92()
#net = ShuffleNetG2()
#net = SENet18()
#net = ShuffleNetV2(1)
net = zoo_attack()

net = net.to(device)

if device == 'cuda':
    #net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = 0.9, weight_decay = 1e-6, nesterov=True)
torch.manual_seed(1)
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = './checkpoint/' + net.__class__.__name__ + '_ckpt.t7'
        print('save path', save_path)
        torch.save(state, save_path)
        best_acc = acc

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':

    if args.resume:
        #models_name = ['VGG("VGG16")', 'ResNet18()', 'MobileNet()', 'MobileNetV2()', 'SENet18()', 'GoogLeNet()', 'PreActResNet18()']
        #models_name = ['VGG("VGG11")', 'VGG("VGG13")', 'VGG("VGG16")', 'VGG("VGG19")', 'zoo_attack()']
        models_name = ['zoo_attack()']
        transform_train = transforms.Compose([
                          transforms.ToTensor(),
                          ])

        transform_test = transforms.Compose([
                          transforms.ToTensor(),
                          ])

        trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size = 100, shuffle = True, num_workers = 2)

        testset = torchvision.datasets.CIFAR10(root='./data', train = False, download = True, transform = transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size = 100, shuffle = False, num_workers = 2)

        for i in range(len(models_name)):
            net = eval(models_name[i])
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            if net.__class__.__name__ == 'VGG':
                m_name = models_name[i].split('"')[1]
                model_checkpoint_path = './checkpoint/' + m_name + '_ckpt.t7'
            else:
                m_name = ''
                model_checkpoint_path = './checkpoint/' + net.__class__.__name__ + '_ckpt.t7'
            print('Loading checkpoint from %s' % model_checkpoint_path)
            checkpoint = torch.load(model_checkpoint_path)
            net.load_state_dict(checkpoint['net'])
            net = net.to(device)
            best_acc = checkpoint['acc']
            print('Best test accuracy', best_acc)
            save_gradient(net, device, trainloader, m_name, mode = 'train')
            save_gradient(net, device, testloader, m_name, mode = 'test')   
    else:
        for epoch in range(start_epoch, start_epoch + 250):
            if epoch > 50 and epoch < 150:
                adjust_learning_rate(optimizer, 0.01)
            elif epoch >= 150:
                adjust_learning_rate(optimizer, 0.001)
            train(epoch)
            test(epoch)
