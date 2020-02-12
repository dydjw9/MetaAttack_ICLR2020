import torch
import numpy as np 
import os
import pdb
import torch.nn.functional as F
import time

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\t Loss {:6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), train_loss / args.log_interval))
            train_loss = 0

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim = True)
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= batch_idx
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'.format(
                    test_loss, correct, len(test_loader.dataset),
                    100. * correct / len(test_loader.dataset)))

def cw_loss(output, target, FLAG=True):
    num_classes = 10
    target_onehot = torch.zeros(target.size() + (num_classes,))
    if torch.cuda.is_available():
        target_onehot = target_onehot.cuda()
    target_onehot.scatter_(1, target.unsqueeze(1), 1.)
    target_var = torch.autograd.Variable(target_onehot, requires_grad = False)
    
    real = (target_var * output).sum(1)
    other = ((1. - target_var) * output - target_var * 10000.).max(1)[0]
    loss = torch.clamp(torch.log(real + 1e-30) - torch.log(other + 1e-30), min = 0.)
    if FLAG:
        loss = torch.sum(0.5 * loss) / len(target)
    return loss
    
def save_gradient(args, model, device, train_loader, criterion, mode = 'train'):
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

    loss_avg /= batch_idx

    print('Average Loss: {:4f}%, Accuracy: {}/{} ({:.0f}%)\n'.format(
            loss_avg, correct, len(train_loader.dataset),
            100. * correct / len(train_loader.dataset)))
    #save_path = './grad_mnist/' + mode
    save_path = './zoo_cw_grad_mnist/' + mode
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_file_path = save_path + '/' + model.__class__.__name__ + '_mnist.npy'
    np.save(save_file_path, process_data)

def save_gradient_zoo(args, model, device, train_loader, criterion, mode = 'train'):
    model.eval()
    correct  = 0
    loss_avg = 0
    
    process_data = dict()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        grad = torch.zeros(data.shape).cuda()
        for j in range(data.shape[0]):
            tmp_grad = torch.zeros(28*28, dtype = torch.float32).cuda()
            var = data[j].clone().repeat(28*28*2+1, 1, 1, 1)
            for i in range(28*28):
                var[i*2 + 1].reshape(-1)[i] += 0.0001
                var[i*2 + 2].reshape(-1)[i] -= 0.0001
            with torch.no_grad():
                output = model(var)
            loss = cw_loss(F.softmax(output, dim = 1), target[j].unsqueeze(0).repeat(28*28*2+1), FLAG=False)
            for i in range(28*28):
                tmp_grad[i] = (loss[i*2+1] - loss[i*2+2]) / 0.0002
            grad[j] = tmp_grad.reshape(28,28).unsqueeze(0)
            
        grad = grad.data.cpu().numpy()
        data = data.data.cpu().numpy()
        target = target.cpu().numpy()
        process_data[str(batch_idx)] = [data, grad, target]
    save_path = './zoo_cw_grad_mnist_zoo_estimate/' + mode
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    save_file_path = save_path + '/' + model.__class__.__name__ + '_mnist.npy'
    np.save(save_file_path, process_data)

