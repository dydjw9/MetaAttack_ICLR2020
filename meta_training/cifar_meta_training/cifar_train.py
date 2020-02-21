#python scripts
__author__='Du Jiawei NUS/IHPC'
__email__='dujw@ihpc.a-star.edu'
#Descrption:
import torch, os
import os.path as osp
import numpy as np
from cifar import cifar
import scipy.stats
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import random, sys, pickle
import argparse
import sys 
from models import *
from meta import Meta

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)

MODELS= ['VGG("VGG13")', 'VGG("VGG16")', 'GoogLeNet()', 'VGG("VGG11")', 'ResNet18()', 'MobileNetV2()', 'SENet18()', 'GoogLeNet()', 'PreActResNet18()', 'MobileNet()']
cifar_grads = ["VGG13", "VGG16", "VGG19", "VGG11", "ResNet", "MobileNetV2", "SENet", "GoogLeNet", "PreActResNet", "MobileNet"]

def get_target_model(index):
    i = index
    net = eval(MODELS[i])
    if MODELS[i].startswith("VGG"):
        model_checkpoint_path = './checkpoint/' + net.__class__.__name__ + MODELS[i][-4:-2] + '_ckpt.t7'
    else:
        model_checkpoint_path = './checkpoint/' + net.__class__.__name__ +'_ckpt.t7'
    print('Loading checkpoint from %s' % model_checkpoint_path)
    checkpoint = torch.load(model_checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    return net

def swapaxis(*input):
    result = []
    for each in input:
        each = each.transpose(0,1)
        result.append(each)
    return result
def main():
    #print(args)
    TARGET_MODEL = 3
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

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    
    # initiate different datasets 
    minis = []
    for i in range(args.task_num):
        path = osp.join("./zoo_cw_grad_cifar/train/", cifar_grads[i] + "_cifar.npy")
        mini = cifar(path, 
                    mode='train', 
                    n_way=args.n_way, 
                    k_shot=args.k_spt, 
                    k_query=args.k_qry, 
                    batchsz=100, 
                    resize=args.imgsz)
        db = DataLoader(mini,args.batchsize, shuffle=True, num_workers=0, pin_memory=True)
        minis.append(db)

    path_test = osp.join("./zoo_cw_grad_cifar/test/", cifar_grads[TARGET_MODEL] + "_cifar.npy")
    mini_test = cifar(path_test, 
                    mode='test', 
                    n_way=1, 
                    k_shot=args.k_spt,
                    k_query=args.k_qry,
                    batchsz=100, 
                    resize=args.imgsz)

    mini_test = DataLoader(mini_test, 10, shuffle=True, num_workers=0, pin_memory=True)
    
    # start training
    step_number = len(minis[0])
    test_step_number = len(mini_test)
    BEST_ACC = 1.5
    target_model = get_target_model(TARGET_MODEL).to(device)
    def save_model(model, acc):
        model_file_path = './checkpoint/meta'
        if not os.path.exists(model_file_path):
            os.makedirs(model_file_path)
    
        file_name = str(acc) + 'cifar_' + MODELS[TARGET_MODEL] + '.pt'
        save_model_path = os.path.join(model_file_path, file_name)
        torch.save(model.state_dict(), save_model_path)
    def load_model(model,acc):
        model_checkpoint_path = './checkpoint/meta/' + str(acc)+'cifar_' + MODELS[TARGET_MODEL] + '.pt'
        assert os.path.exists(model_checkpoint_path)
        model.load_state_dict(torch.load(model_checkpoint_path))        
        return model

    for epoch in range(args.epoch//100):
        minis_iter = []
        for i in range(len(minis)):
            minis_iter.append(iter(minis[i]))
        mini_test_iter = iter(mini_test)

        if args.resume:
            maml = load_model(maml, "0.74789804")
        for step in range(step_number):
            batch_data = []
            for each in minis_iter:
                batch_data.append(each.next())

            accs = maml(batch_data, device)
            if (step + 1) % step_number  == 0:
                print('Training acc:', accs)
                if accs[0] < BEST_ACC:
                    BEST_ACC = accs[0]
                    save_model(maml, BEST_ACC)

            if (epoch  + 1) % 15 == 0 and step ==0:  # evaluation
                accs_all_test = []
                for i in range(3):
                    test_data = mini_test_iter.next()
                    accs = maml.finetunning(test_data,target_model,device)
                    accs_all_test.append(accs)

                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=254600)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=10)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=10)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=32)
    argparser.add_argument('--imgc', type=int, help='imgc', default=3)
    argparser.add_argument('--batchsize', type=int, help='batchsize', default=64)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-2)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=1e-2)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=20)
    argparser.add_argument('--meta_optim_choose', help='reptile or maml', default="reptile")
    argparser.add_argument('--resume', action = 'store_true',help='load model or not', default=False)

    args = argparser.parse_args()

    main()
