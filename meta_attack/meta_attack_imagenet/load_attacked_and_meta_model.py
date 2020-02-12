import torch
import mnist_model
import imagenet_models
from learner import Learner
from models import *
# '''
config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [128, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('conv2d', [256, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [256]),
        ('convt2d', [256, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('convt2d', [128, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('convt2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('convt2d', [32, 3, 3, 3, 1, 1]),
   ]
'''
config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [128, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('conv2d', [128, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('conv2d', [256, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [256]),
        ('convt2d', [256, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('convt2d', [128, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('convt2d', [128, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('convt2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('convt2d', [32, 3, 3, 3, 1, 1]),
   ]
'''


MODELS = ['vgg13_bn', 'vgg16_bn', 'resnet18', 'resnet34', 'vgg19_bn','resnet50']
def load_attacked_model(args, index, device):
    i = index
    model_name = MODELS[i]
    net = imagenet_models.__dict__[model_name]()
    model_checkpoint_path = '/data/home/dujw/attack/imagenet_grad/meta-zoo/checkpoint/' + model_name + '_adam' + '/model_best.pth.tar'
    print('Loading checkpoint from %s' % model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path)
    net.load_state_dict(checkpoint['state_dict'])

    return net.to(device)

class Meta(nn.Module):
    def __init__(self):
        super(Meta, self).__init__() 
        self.net = Learner(config, 3, 32)
    def forward(self,x):
        return x

def load_meta_model(args, meta_model_path, device):
    # meta_model = Learner(config, 3, 32)
    meta_model = Meta()
    meta_model = nn.DataParallel(meta_model)
    pretrained_dict = torch.load(meta_model_path)
    pretrained_dict = {"module."+k: v for k, v in pretrained_dict.items()}
    meta_model.load_state_dict(pretrained_dict)
    meta_model.module.net.eval()
    # meta_model.eval()
    return meta_model.module.net.to(device)
    return meta_model.net.to(device)


