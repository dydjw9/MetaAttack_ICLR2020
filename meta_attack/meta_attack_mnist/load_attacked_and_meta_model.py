import torch
import mnist_model
from learner import Learner
import torch.nn as nn
# v1 config
config = [
        ('conv2d', [16, 1, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [16]),
        ('conv2d', [32, 16, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('convt2d', [64, 32, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [32]),
        ('convt2d', [32, 16, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [16]),
        ('convt2d', [16, 8, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [8]),
        ('convt2d', [8, 1, 3, 3, 1, 1]),
    ]

MODELS  = ['LeNet5', 'Net1', 'Net2', 'Net3', 'Net4', 'Net5','Net6']
def load_attacked_model(args, device):
    return get_target_model(args.attacked_model).to(device)

def get_target_model(index):
    i = index
    models = MODELS
    model_checkpoint_path = '../../checkpoints/targeted_model/mnist/' + 'mnist_' + models[i] + '.pt'
    model = mnist_model.__dict__[models[i]]()
    model.load_state_dict(torch.load(model_checkpoint_path)) 
    model.eval()
    return model

class Meta(nn.Module):
    def __init__(self):
        super(Meta, self).__init__() 
        self.net = Learner(config, 1, 28)
    def forward(self,x):
        x = self.net(x)
        return x

def load_meta_model(args, meta_model_path, device):
    # the function to load the meta attacker 
    meta_model = Meta()
    pretrained_dict = torch.load(meta_model_path)
    meta_model.eval()
    return meta_model.to(device)


