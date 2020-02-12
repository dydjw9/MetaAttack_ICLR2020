import os
import sys
import time
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

def cw_loss(output, target):
    num_classes = 200
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
        loss = cw_loss(F.softmax(output, dim = 1), target)
        grad = torch.autograd.grad(loss, data)[0]

        loss_avg += loss.item()
        pred = output.argmax(dim=1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        grad = grad.data.cpu().numpy()
        data = data.data.cpu().numpy()
        target = target.cpu().numpy()
        
        process_data[str(batch_idx)] = [data, grad, target]
        
        if batch_idx >= 100:
            break
    loss_avg /= batch_idx

    print('Average Loss: {:4f}%, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss_avg, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    #save_path = './grad_mnist/' + mode
    save_path = './zoo_cw_grad_imagenet/' + mode
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_file_path = save_path + '/' + model_name + '_imagnet.npy'
    np.save(save_file_path, process_data)
