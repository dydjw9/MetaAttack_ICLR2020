#python scripts
__author__='Du Jiawei NUS/IHPC'
__email__='dujw@ihpc.a-star.edu'
#Descrption:
import os
import os.path as osp
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import numpy as np
import collections
from PIL import Image
import csv
import random
import pdb

class mnist(Dataset):

    def __init__(self, root, mode, batchsz, n_way, k_shot, k_query, resize, startidx=0):
        """

        :param root: root path of gradients file
        :param mode: train, val or test
        :param batchsz: batch size of sets, not batch of imgs
        :param n_way:
        :param k_shot:
        :param k_query: num of qeruy imgs per class
        :param resize: resize to
        :param startidx: start to index label from startidx
        """

        self.batchsz = batchsz  # batch of set, not batch of imgs
        self.n_way = n_way  # n-way
        self.k_shot = k_shot  # k-shot
        self.k_query = k_query  # for evaluation
        self.root = root 
        

        # setsz and querysz are n_way
        self.setsz = self.n_way   # num of samples per set
        self.querysz = self.n_way   # number of samples per set for evaluation

        
        self.resize = resize  # resize to
        self.startidx = startidx  # index label not from 0, but from startidx
        self.mode = mode
        print('shuffle DB :%s, b:%d, %d-way, %d-shot, %d-query, resize:%d' % (
        mode, batchsz, n_way, k_shot, k_query, resize))


        # rescale makes the output to be within(0 - 1)
        def rescale(x):
            low, high = x.min(), x.max()
            x_rescaled = (x - low) / (high - low)
            return x_rescaled
        self.transform = transforms.Compose([torch.from_numpy])

        self.n_way = 1 

        images = {}
        grads = {}
        labels = {}

        #load the data
        path = osp.join(root)
        data = np.load(path)


        images = data[()]['0'][0]
        grads = data[()]['0'][1]
        labels = data[()]['0'][2]

        for batch in range(1, batchsz):
            bz = str(batch)

            images = np.concatenate((images, data[()][bz][0]), axis=0)
            grads = np.concatenate((grads, data[()][bz][1]), axis=0)
            labels = np.concatenate((labels, data[()][bz][2]), axis=0)
            
        #pdb.set_trace()
        order = np.arange(images.shape[0])
        np.random.shuffle(order)

        images = images[order]
        grads = grads[order]
        labels = labels[order]
        
        #pdb.set_trace()
        std = grads.std(axis=(1, 2, 3))
        
        std = std.reshape((-1, 1, 1, 1))
        grads = grads / (std + 10e-23)

        self.images = images
        self.grads = grads
        self.labels = labels

        #cutoff is the index to divide support and query 
        self.cutoff = self.k_shot/(self.k_shot+self.k_query) * self.images.shape[0]
        self.cutoff = int(self.cutoff)
        self.maximum_index = self.cutoff // self.k_shot


    def __getitem__(self, index):
        """
        index means index of sets, 0<= index <= batchsz-1
        :param index:
        :return:
        """
        # [setsz,  resize, resize]
        support_x = torch.FloatTensor(self.k_shot,1,  self.resize, self.resize)
        # [setsz]
        support_y = torch.FloatTensor(self.k_shot,1,  self.resize, self.resize)
        # [querysz,  resize, resize]

        query_x = torch.FloatTensor(self.k_query,1,  self.resize, self.resize)
        # [querysz]
        query_y = torch.FloatTensor(self.k_query, 1, self.resize, self.resize)

        if self.mode == "test":
            support_label  = torch.LongTensor(self.k_shot)
            query_label = torch.LongTensor(self.k_query)

        bias = self.cutoff
        for i in range(self.k_shot):
            support_x[i] = self.transform(self.images[i + index * self.k_shot])
            support_y[i] = self.transform(self.grads[i + index * self.k_shot])
            if self.mode == "test":
                support_label[i] = torch.Tensor(np.array(self.labels[i + index * self.k_shot], np.float32)).long()
        for i in range(self.k_query):
            query_x[i] = self.transform(self.images[i + bias+index * self.k_query])
            query_y[i] = self.transform(self.grads[i + bias+index * self.k_query])
            #return labels for test mode
            if self.mode == "test":
                query_label[i] = torch.Tensor(np.array(self.labels[i + bias + index * self.k_query], np.float32)).long()

        if self.mode == "test":
            return support_x, support_y, support_label, query_x, query_y, query_label 


        return support_x, support_y, query_x, query_y 

    def __len__(self):
        # as we have built up to batchsz of sets, you can sample some small batch size of sets.
        return self.maximum_index


if __name__ == '__main__':
    # the following episode is to view one set of images via tensorboard.
    from torchvision.utils import make_grid
    from matplotlib import pyplot as plt
    from tensorboardX import SummaryWriter
    import time
    from    torch.utils.data import DataLoader

    plt.ion()

    tb = SummaryWriter('runs', 'mini-imagenet')
    mini = mnist('../grad_mnist/Net5_mnist.npy', mode='train', n_way=5, k_shot=5, k_query=15, batchsz=100, resize=28)
    db = DataLoader(mini,batch_size=64, shuffle=True, num_workers=1, pin_memory=True)

    aa = []
    for i, set_ in enumerate(db):
        # support_x: [k_shot*n_way, 3, 84, 84]
        support_x, support_y, query_x, query_y = set_
        aa.append(support_x)
        print(i)
    import pdb
    pdb.set_trace()

    tb.close()
