import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import numpy as np
import pandas as pd 
from learner import Learner
from copy import deepcopy

class Meta(nn.Module):
    def __init__(self, args, config):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test

        self.meta_optim_choose = args.meta_optim_choose

        self.net = Learner(config, args.imgc, args.imgsz)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)
    def loss_function(self,logits,target,label):        
        loss = F.mse_loss(logits,target)
        return loss

    def forward(self, batch_data,device):
        task_num = len(batch_data)
        _,__, c_, h, w = batch_data[0][0].size()

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 2)]
        querysz = 1 

        def computer_diff(weights):
            '''
            this method is to computer the diifference between received weights
            and the self.net.parameters(), it will return the difference
            '''
            dis = []
            for i,each in enumerate(self.net.parameters()):
                dis.append((each-weights[i]).clone().detach())
            return dis
        def computer_meta_weight(weights):
            '''
            meta optim for reptile
            this method will update the self.net.parameters according to the 
            reveived weights, which is the updating directions. The updata learning 
            rate is self.update.lr
            '''
            dic = self.net.state_dict()
            keys = list(dic.keys())
            for i,each in enumerate(weights[0]):
                diff = torch.zeros_like(each)
                for j in range(task_num):
                    diff += weights[j][i]
                diff /= task_num
                dic[keys[i]] -=  diff
            self.net.load_state_dict(dic)

        task_diff_weights = []   # task diff weights is the list to store all weights diiffs in all tasks
        for i in range(task_num):
            x_spt, y_spt, label_spt, x_qry, y_qry, label_qry = [x.reshape([-1,c_,h,w]).to(device) if i % 3 <=1 else x.reshape([-1]).to(device) for i,x in enumerate(batch_data[i])]

            fast_weights = []
            for each in self.net.parameters():
                #we need to use variable reassgin the weights so that 
                #it will not be considered as non-leaf variable
                pp = torch.autograd.Variable(each.clone(), requires_grad=True)
                fast_weights.append(pp)
            #the optimizer for each sub-task
            if self.meta_optim_choose == "reptile":
                cur_task_optim = optim.Adam(fast_weights, lr = self.update_lr)

            if self.meta_optim_choose == "maml":
                cur_task_optim = optim.SGD(fast_weights, lr = self.update_lr)

            logits = self.net(x_spt, fast_weights, bn_training=True)
            loss = self.loss_function(logits, y_spt,label_spt)
            cur_task_optim.zero_grad()
            loss.backward()
            cur_task_optim.step()
            # this is the loss and accuracy before first update
            with torch.no_grad():
                logits_q = self.net(x_spt, self.net.parameters(), bn_training=True)
                loss_q = self.loss_function(logits_q, y_spt, label_spt)
                losses_q[0] += loss_q
                correct = loss_q.sum()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                loss_q = self.loss_function(logits_q, y_qry, label_qry)
                losses_q[1] += loss_q
                correct = loss_q.sum()
                corrects[1] = corrects[1] + correct
            for k in range(1, self.update_step):
                logits = self.net(x_spt, fast_weights, bn_training=True)
                loss = self.loss_function(logits, y_spt, label_spt)
                cur_task_optim.zero_grad()
                loss.backward()
                cur_task_optim.step()

                with torch.no_grad():
                    logits_q = self.net(x_spt, fast_weights, bn_training=True)
                    loss_q = self.loss_function(logits_q, y_spt, label_spt)
                    correct = loss_q.sum()
                    corrects[k + 1] = corrects[k + 1] + correct

            current_task_diff = computer_diff(fast_weights)
            task_diff_weights.append(current_task_diff)

            if self.meta_optim_choose == "maml":
                fast_weights = list(map(lambda p: p[1] - p[0], zip(current_task_diff, self.net.parameters())))

                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                loss_q = self.loss_function(logits_q, y_qry, label_qry)
                losses_q[k + 1] += loss_q
        # end of all tasks
        # sum over all losses on query set across all tasks
        if self.meta_optim_choose == "maml":
            loss_q = losses_q[-1] / task_num
            # optimize theta parameters
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()

        if self.meta_optim_choose == "reptile":
            computer_meta_weight(task_diff_weights)

        logits_q = self.net(x_spt, bn_training=True)
        loss_q = self.loss_function(logits_q, y_spt, label_spt)
        correct = loss_q.sum()
        corrects[k + 2] = correct
        
        for i in range(len(corrects)):
            corrects[i] = corrects[i].item()

        accs = np.array(corrects) / (querysz * task_num)
        return accs

    def finetunning(self, test_data,model,device,step_size=0.3):
        # method attack to measure the attack success rate
        def attack(x,y,label,model,step_size):
            orilabel = model(x).argmax(dim=1)
            x_fool = x + torch.sign(y)*step_size
            x_fool = torch.clamp(x_fool,0,1)
            fool_label = model(x_fool).argmax(dim=1)
            acc = (orilabel==fool_label).cpu().numpy().sum()/len(orilabel)
            return acc
        _,__, c_, h, w = test_data[0].size()
        x_spt, y_spt, label_spt, x_qry, y_qry, label_qry = [x.reshape([-1,c_,h,w]).to(device) if i % 3 <=1 else x.reshape([-1]).to(device).long() for i, x in enumerate(test_data)]
        assert len(x_spt.shape) == 4

        corrects = [[0,0] for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        net.eval()
        if self.meta_optim_choose == "reptile":
            cur_task_optim = optim.Adam(net.parameters(), lr=self.update_lr)

        if self.meta_optim_choose == "maml":
            cur_task_optim = optim.SGD(net.parameters(), lr=self.update_lr)
        logits_q = net(x_qry,bn_training=True)
        correct = [attack(x_qry, logits_q, label_qry, model, step_size), attack(x_qry, y_qry, label_qry, model, step_size)]
        corrects[0] = correct
        
        cur_task_optim.zero_grad()
        logits = net(x_spt)
        loss = self.loss_function(logits, y_spt, label_spt)
        loss.backward()
        cur_task_optim.step()
        
        logits_q = net(x_qry,bn_training=True)
        logits_q = net(x_qry,bn_training=True)
        correct = [attack(x_qry, logits_q, label_qry, model, step_size), attack(x_qry, y_qry, label_qry, model, step_size)]
        corrects[1] = correct

        for k in range(1, self.update_step_test):
            logits = net(x_spt, bn_training=True)
            loss = self.loss_function(logits, y_spt, label_spt)
            cur_task_optim.zero_grad()
            loss.backward()
            cur_task_optim.step()
            
            logits_q = net(x_qry, bn_training=True)
            loss_q = self.loss_function(logits_q, y_qry, label_qry)
            
            logits_q = net(x_qry,bn_training=True)
            correct = [attack(x_qry, logits_q, label_qry, model, step_size), attack(x_qry, y_qry, label_qry, model, step_size)]
            corrects[k + 1] = correct

        del net
        accs = np.array(corrects) 
        return accs
def main():
    pass

if __name__ == '__main__':
    main()
