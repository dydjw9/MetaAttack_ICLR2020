#python scripts
__author__='Du Jiawei NUS/IHPC'
__email__='dujw@ihpc.a-star.edu'
#Descrption:
import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

class Adam:
    def __init__(self,lr,grad_shape,beta1=0.9,beta2=0.99):
        self.beta1 = torch.tensor(beta1)
        self.beta2 = torch.tensor(beta2)
        self.lr = lr

        self.indices = 0

        self.mt = torch.zeros(grad_shape).cuda()
        self.nt = torch.zeros_like(self.mt)

        self.mt_modi = torch.zeros_like(self.mt)
        self.nt_modi = torch.zeros_like(self.mt)



    def update(self,gt):
        self.indices +=1 

        b1 = self.beta1
        b2 = self.beta2

        self.mt = b1 * self.mt + (1-b1) * gt
        self.nt = b2 * self.nt + (1-b2) * (gt * gt)

        self.mt_modi = self.mt/(1-b1.pow(self.indices))
        self.nt_modi = self.nt/(1-b2.pow(self.indices))

        updates = self.lr * self.mt_modi/(self.nt_modi.pow(0.5)+1e-8)

        return updates

        

    


