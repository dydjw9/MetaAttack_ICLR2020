import os
import sys

class Logger(object):
    def __init__(self, filepath = './log.txt', mode = 'w', stdout = None):
        if stdout == None:
            self.terminal = sys.stdout
        else:
            self.terminal = stdout
        self.log = open(filepath, mode)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        os.fsync(self.log)
    def flush(self):
        pass
def save_gradient(model, device, train_loader, mode = 'train'):
    model.eval()
    correct  = 0
    loss_avg = 0
    
    process_data = dict()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        
        output = model(data)
        #loss = criterion(output, target)
        pred = output.argmax(dim=1, keepdim = True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    return correct


