import torch
import mnist_model
from learner import Learner
from models import *
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
        ('conv2d', [16, 3, 3, 3, 1, 1]),
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
        ('convt2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('convt2d', [32, 16, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [16]),
        ('convt2d', [16, 8, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [8]),
        ('convt2d', [8, 3, 3, 3, 1, 1]),
    ]
'''
MODELS= ['VGG("VGG13")','VGG("VGG16")','VGG("VGG19")','VGG("VGG11")', 'ResNet18()', 'MobileNetV2()', 'SENet18()', 'GoogLeNet()', 'PreActResNet18()', 'MobileNet()']
def load_attacked_model(args, index, device):
    i = index
    net = eval(MODELS[i])
    if MODELS[i].startswith("VGG"):
        model_checkpoint_path = 'data/home/dujw/attack/cifar_grad/meta-zoo/train_cifar_model/checkpoint/' + net.__class__.__name__ + MODELS[i][-4:-2]+'_ckpt.t7'
    else:
        model_checkpoint_path = '/data/home/dujw/attack/cifar_grad/meta-zoo/train_cifar_model/checkpoint/' + net.__class__.__name__ +'_ckpt.t7'
 
    print('Loading checkpoint from %s' % model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path)
    net.load_state_dict(checkpoint['net'])

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
    pretrained_dict = torch.load(meta_model_path)
#       pretrained_dict = {k[4:]: v for k, v in pretrained_dict.items()}
    meta_model.load_state_dict(pretrained_dict)
    meta_model.net.eval()
    return meta_model.net.to(device)


