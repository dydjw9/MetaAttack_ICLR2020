'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def cw_loss(output, target):
    #pdb.set_trace()
    num_classes = 10
    target_onehot = torch.zeros(target.size() + (num_classes,))
    if torch.cuda.is_available():
        target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = torch.autograd.Variable(target_onehot, requires_grad = False)
    
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = torch.clamp(torch.log(real + 1e-30) - torch.log(other + 1e-30), min = 0.)
    loss = torch.sum(0.5 * loss) / len(target)
    return loss
    
def save_gradient(model, device, train_loader, model_name, mode = 'train'):
    model.eval()
    correct  = 0
    loss_avg = 0
    
    process_data = dict()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        data.requires_grad_()
        model.zero_grad()
        
        output = model(data)
        #loss = criterion(output, target)
        loss = cw_loss(F.softmax(output, dim = 1), target)
        grad = torch.autograd.grad(loss, data)[0]

        loss_avg += loss.item()
        pred = output.argmax(dim=1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        grad = grad.data.cpu().numpy()
        data = data.data.cpu().numpy()
        target = target.cpu().numpy()
        
        process_data[str(batch_idx)] = [data, grad, target]

    loss_avg /= batch_idx

    print('Average Loss: {:4f}%, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss_avg, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    #save_path = './grad_mnist/' + mode
    save_path = './zoo_cw_grad_cifar/' + mode
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    if model.__class__.__name__ == 'VGG':
        save_file_path = save_path + '/' + model_name + '_cifar.npy'
    else:
        save_file_path = save_path + '/' + model.__class__.__name__ + '_cifar.npy'
    np.save(save_file_path, process_data)
