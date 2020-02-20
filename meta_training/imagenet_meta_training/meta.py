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
        """
        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :batch_data: [task_number,x_spt, y_spt, x_qry, y_qry]
        :return:
        """
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
            x_spt, y_spt, label_spt, x_qry, y_qry, label_qry = [x.reshape([-1,c_,h,w]).to(device) if i % 3 <=1 else  x.reshape([-1]).to(device) for i,x in enumerate(batch_data[i])]
            fast_weights = []
            for each in self.net.parameters():
                '''
                we need to use variable reassgin the weights so that 
                it will not be considered as non-leaf variable
                '''
                pp = torch.autograd.Variable(each.clone(), requires_grad=True)
                fast_weights.append(pp)
            '''
            the optimizer for each sub-task
            '''
            if self.meta_optim_choose == "reptile":
                cur_task_optim = optim.Adam(fast_weights, lr=self.update_lr)

            if self.meta_optim_choose == "maml":
                cur_task_optim = optim.SGD(fast_weights, lr=self.update_lr)

            logits = self.net(x_spt, fast_weights, bn_training=True)
            loss = self.loss_function(logits, y_spt,label_spt)
            cur_task_optim.zero_grad()
            loss.backward()
            cur_task_optim.step()
            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_spt, self.net.parameters(), bn_training=True)
                loss_q = self.loss_function(logits_q, y_spt,label_spt)
                losses_q[0] += loss_q
                correct = loss_q.sum()
                corrects[0] = corrects[0] + correct

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                loss_q = self.loss_function(logits_q, y_qry,label_qry)
                losses_q[1] += loss_q
                correct = loss_q.sum()
                corrects[1] = corrects[1] + correct
            for k in range(1, self.update_step):
                logits = self.net(x_spt, fast_weights, bn_training=True)
                loss = self.loss_function(logits, y_spt,label_spt)
                cur_task_optim.zero_grad()
                loss.backward()
                cur_task_optim.step()

                with torch.no_grad():
                    logits_q = self.net(x_spt, fast_weights, bn_training=True)
                    loss_q = self.loss_function(logits_q, y_spt,label_spt)
                    
                    correct = loss_q.sum()
                    corrects[k + 1] = corrects[k + 1] + correct

            current_task_diff = computer_diff(fast_weights)
            task_diff_weights.append(current_task_diff)

            if self.meta_optim_choose == "maml":
                fast_weights = list(map(lambda p: p[1] - p[0], zip(current_task_diff, self.net.parameters())))

                logits_q = self.net(x_qry, fast_weights, bn_training=True)
                loss_q = self.loss_function(logits_q, l_qry,label_qry)
                losses_q[k + 1] += loss_q
        
        if self.meta_optim_choose == "maml":
            loss_q = losses_q[-1] / task_num
            self.meta_optim.zero_grad()
            loss_q.backward()
            self.meta_optim.step()

        if self.meta_optim_choose == "reptile":
            computer_meta_weight(task_diff_weights)

        logits_q = self.net(x_spt, bn_training=True)
        loss_q = self.loss_function(logits_q, y_spt,label_spt)
        correct = loss_q.sum()
        corrects[k + 2] = correct
        accs = np.array(corrects) / (querysz * task_num)
        return accs

    def finetunning(self, test_data,model,device,step_size=0.3):
        # method attack to measure the attack success rate
        def attack(x,y,label,model,step_size):
            orilabel = model(x).argmax(dim=1)
            x_fool = x + torch.sign(y)*step_size
            x_fool = torch.clamp(x_fool,0,1)
            fool_label = model(x_fool).argmax(dim=1)
            acc =(orilabel==fool_label).cpu().numpy().sum()/len(orilabel)
            return acc

        def save_csv(name,dic):
            save_file = pd.DataFrame(data=dic)
            save_file.to_csv("../csv/"+name+".csv")
        def issuccess(model,data,label):

            logits = model(data)
            pred = logits.argmax(dim=1)
            max = logits.max(dim=1)
            if pred !=label:
                return True
            else:
                print(max)
                return False

        def gradients(model,data,label,logits_true,isCWloss=False):
            data.requires_grad_()
            model.zero_grad()
            output = model(data)
            if isCWloss:
                lt,l_posi = logits_true.topk(2,dim=1)
                lt = torch.gather(output,1,l_posi)
                loss1 = lt[:,0]
                loss2 = lt[:,1]
                loss = loss1 
                loss = loss.sum() *-1
            else:
                loss = output.max(dim=1)
                loss = loss[0].sum() * -1
            loss.backward()
            grad = data.grad
            return grad.clone().detach()

        def attacking_fintune(model,x,y,optim):
            model.train()
            print("training loss")
            for i in range(1):
                y_pred = model(x)
                loss = self.loss_function(y_pred,y,y)
                yy = torch.argmax(y,dim=1)
                yy_pred = torch.argmax(y_pred,dim=1)
                acc = (yy==yy_pred).sum().float()/yy.size(0)
                print("acc:%f"%(acc))
                optim.zero_grad()
                loss.backward(retain_graph=True)
                print(loss)
                optim.step()
            print("training loss")
            model.eval()
        def adam_attack(data,target,meta_model,targeted_model,lr,step_size = 0.3):
            '''
            the iterative attack process like zoo-attack.
            for each test sample, the meta model will generate the gradients map
            and feed it in to the adam to generate noise.
            '''
            from adam import Adam
            count_sum = np.zeros(data.shape[0])
            loss_sum = np.zeros(data.shape[0])
            zero_count = 0

            mask = torch.tensor(np.load("di.npy")).float().cuda()
            mask[mask<0.11] = 0
            mask[mask>0.11] = 1

            item_l2 = []
            item_count = []
            training_x = torch.FloatTensor(3,1,28,28).cuda()
            training_y = torch.FloatTensor(3,10).cuda()
            training_count = 0

            for i in range(len(data)): 
                #copy meta model for each sample
                model = deepcopy(meta_model)
                training_count = 0

                if self.meta_optim_choose == "reptile":
                    cur_task_optim = optim.Adam(model.parameters(), lr=self.update_lr)

                if self.meta_optim_choose == "maml":
                    cur_task_optim = optim.SGD(model.parameters(), lr=self.update_lr)

                opt = Adam(0.01,data[0].size(),beta1=0.9,beta2=0.99)
                x = data[i]
                label = target[i]
                count = 0
                original_sample = x.clone().detach()
                x = x.view(1,1,28,28)
                x = x.clone().detach()
                update = torch.zeros_like(x)
                label = label.view(1)

                adv_sample = (x+update).clone().detach()
                while True:
                    count +=1
                    logits_true = targeted_model(adv_sample)
                    grad = gradients(model,adv_sample,label,logits_true,isCWloss=True)

                    update += opt.update(grad)
                    update = torch.clamp(update,-1*step_size,step_size)
                    update = update * mask
                    adv_sample = (x+update).clone().detach()
                    adv_sample = torch.clamp(adv_sample,0,1)

                    if issuccess(targeted_model,adv_sample,label):
                        if count < 1000:
                            count_sum[i] = count
                            loss_sum[i] =np.sum((adv_sample.view(1,28,28)-x).detach().cpu().numpy()**2)**0.5
                            item_l2.append(loss_sum[i])
                            item_count.append(count)
                            print("queries:%d,l2:%f"%(count_sum[i],loss_sum[i]))
                        else:
                            zero_count += 1
                        break
                    else:
                        if training_count <3:
                            training_x[training_count] = adv_sample.view(1,28,28)
                            training_y[training_count] = targeted_model(adv_sample).view(-1)
                        training_count += 1
                        if training_count >=3 and count <300:
                            training_count = 0
                            attacking_fintune(model,training_x,training_y,cur_task_optim)
                        if count >1000:
                            zero_count += 1
                            item_l2.append(loss_sum[i])
                            item_count.append(count)
                            print("failed")
                            break
            effctive_count = len(loss_sum) - zero_count
            print("avrg query times:%f, avrg l2 loss %f,zero_count:%d"%(count_sum.sum()/effctive_count,loss_sum.sum()/effctive_count,zero_count))
            save_csv("adam_attack",{"count":item_count,"l2":item_l2})
        _,__, c_, h, w = test_data[0].size()
        x_spt, y_spt, label_spt,x_qry, y_qry, label_qry = [x.reshape([-1,c_,h,w]).to(device) if i % 3 <=1 else  x.reshape([-1]).to(device).long() for i,x in enumerate(test_data)]
        assert len(x_spt.shape) == 4
        """
        initiallize an random mask for the optimization
        """
        random_point_number = 256
        weight_mask = np.zeros((784))
        picked_points = np.random.choice(728,random_point_number)
        weight_mask[picked_points] = 1
        weight_mask = weight_mask.reshape((28,28))
        weight_mask = torch.tensor(weight_mask).float().to(device)
        querysz = x_qry.size(0)

        corrects = [[0,0] for _ in range(self.update_step_test + 1)]
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)
        net.eval()
        if self.meta_optim_choose == "reptile":
            cur_task_optim = optim.Adam(net.parameters(), lr=self.update_lr)

        if self.meta_optim_choose == "maml":
            cur_task_optim = optim.SGD(net.parameters(), lr=self.update_lr)

        logits_q = net(x_qry,bn_training=True)
        correct = [attack(x_qry,logits_q,label_qry,model,step_size),attack(x_qry,y_qry,label_qry,model,step_size)]
        corrects[0] = correct

        cur_task_optim.zero_grad()
        logits = net(x_spt)
        loss = self.loss_function(logits, y_spt,label_spt)
        loss.backward()
        cur_task_optim.step()

        # this is the loss and accuracy after the first update
        logits_q = net(x_qry,bn_training=True)
        # logits_q = gradients(net,x_qry,label_qry,l_qry,isCWloss=True)
        logits_q = net(x_qry,bn_training=True)
        correct = [attack(x_qry,logits_q,label_qry,model,step_size),attack(x_qry,y_qry,label_qry,model,step_size)]
        corrects[1] = correct

        for k in range(1, self.update_step_test):
            logits = net(x_spt, bn_training=True)
            loss = self.loss_function(logits, y_spt,label_spt)
            cur_task_optim.zero_grad()
            loss.backward()
            cur_task_optim.step()

            logits_q = net(x_qry, bn_training=True)
            loss_q = self.loss_function(logits_q, y_qry,label_qry)
            logits_q = net(x_qry,bn_training=True)
            correct = [attack(x_qry,logits_q,label_qry,model,step_size),attack(x_qry,y_qry,label_qry,model,step_size)]
            corrects[k + 1] = correct

        print(loss_q.detach().cpu().numpy())
        del net
        accs = np.array(corrects) 
        return accs
def main():
    pass

if __name__ == '__main__':
    main()
