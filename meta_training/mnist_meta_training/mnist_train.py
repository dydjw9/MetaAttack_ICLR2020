#python scripts
__author__='Du Jiawei NUS/IHPC'
__email__='dujw@ihpc.a-star.edu'
#Descrption:
import torch, os
import os.path as osp
import numpy as np
import scipy.stats
import random, pickle
import argparse
import sys 
from mnist import mnist
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import mnist_model
from meta import Meta

torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
 
MODELS  = ['LeNet5', 'Net2', 'Net3', 'Net4']

np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def get_target_model(index):
    i = index
    models = MODELS
    model_checkpoint_path = 'checkpoint/mnist/' + 'mnist_' + models[i] + '.pt'
    model = mnist_model.__dict__[models[i]]()
    model.load_state_dict(torch.load(model_checkpoint_path)) 
    return model

def mean_confidence_interval(accs, confidence=0.95):
    n = accs.shape[0]
    m, se = np.mean(accs), scipy.stats.sem(accs)
    h = se * scipy.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h

def swapaxis(*input):
    result = []
    for each in input:
        each = each.transpose(0, 1)
        result.append(each)
    return result

def main():
    #print(args)
    TARGET_MODEL = 3

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

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    #print(maml)
    #print('Total trainable tensors:', num)

    # initiate different datasets 
    minis = []
    for i in range(args.task_num):
        path = osp.join("./zoo_cw_grad_mnist/train", MODELS[i] + "_mnist.npy")
        mini = mnist(path,
                    mode='train', 
                    n_way=args.n_way, 
                    k_shot=args.k_spt,
                    k_query=args.k_qry,
                    batchsz=100, 
                    resize=args.imgsz)
        db = DataLoader(mini, args.batchsize, shuffle=True, num_workers=0, pin_memory=True)
        minis.append(db)

    path_test = osp.join("./zoo_cw_grad_mnist/test", MODELS[TARGET_MODEL] + "_mnist.npy")
    mini_test = mnist(path_test, 
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
    BEST_ACC = 1.0
    target_model = get_target_model(TARGET_MODEL).to(device)
    def save_model(model,acc):
        model_file_path = './checkpoint/mnist'
        if not os.path.exists(model_file_path):
            os.makedirs(model_file_path)
    
        file_name = str(acc) + 'mnist_'+ MODELS[TARGET_MODEL] + '.pt'
        save_model_path = os.path.join(model_file_path, file_name)
        torch.save(model.state_dict(), save_model_path)
    def load_model(model,acc):
        model_checkpoint_path = './checkpoint/mnist/' + str(acc) + 'mnist_' + MODELS[TARGET_MODEL] + '.pt'
        assert os.path.exists(model_checkpoint_path)
        model.load_state_dict(torch.load(model_checkpoint_path))        
        return model

    for epoch in range(args.epoch//100):
        minis_iter = []
        for i in range(len(minis)):
            minis_iter.append(iter(minis[i]))
        mini_test_iter = iter(mini_test)
        if args.resume:
            maml = load_model(maml,"0.7231071")
        for step in range(step_number):
            batch_data = []
            for each in minis_iter:
                batch_data.append(each.next())
            accs = maml(batch_data, device)

            if (step + 1) % step_number == 0:
                print('Training acc:', accs)
                if accs[0] < BEST_ACC:
                    BEST_ACC = accs[0]
                    save_model(maml, BEST_ACC)

            if (epoch + 1) % 15 == 0 and step ==0:  # evaluation
                accs_all_test = []
                for i in range(3):
                    test_data = mini_test_iter.next()
                    accs = maml.finetunning(test_data, target_model, device)
                    accs_all_test.append(accs)

                accs = np.array(accs_all_test).mean(axis=0).astype(np.float16)
                print('Test acc:', accs)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=54600)
    argparser.add_argument('--n_way', type=int, help='n way', default=5)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=25)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=5)
    argparser.add_argument('--imgsz', type=int, help='imgsz', default=28)
    argparser.add_argument('--imgc', type=int, help='imgc', default=1)
    argparser.add_argument('--batchsize', type=int, help='batchsize', default=32)
    argparser.add_argument('--task_num', type=int, 
                            help='meta batch size, namely task num', default=3)
    argparser.add_argument('--meta_lr', type=float, 
                            help='meta-level outer learning rate', default=1e-2)
    argparser.add_argument('--update_lr', type=float, 
                            help='task-level inner update learning rate', default=1e-2)
    argparser.add_argument('--update_step', type=int, 
                            help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, 
                            help='update steps for finetunning', default=20)
    argparser.add_argument('--meta_optim_choose', help='reptile or maml', default="reptile")
    argparser.add_argument('--resume', action = 'store_true', default=False, 
                            help='load model or not')

    args = argparser.parse_args()

    main()
