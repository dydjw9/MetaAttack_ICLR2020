
from __future__ import print_function
import os
import os.path
import errno
import numpy as np
import sys
import cv2
from PIL import Image

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s : s.strip(), r.readlines()))

    #classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx

def loadPILImage(path):
    trans_img = Image.open(path).convert('RGB')
    return trans_img

def loadCVImage(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    trans_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    IMG = Image.fromarray(trans_img.astype('uint8'), 'RGB')
    return IMG

def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    if is_image_file(imgname):
                        path = os.path.join(cls_imgs_path, imgname)
                        item = (path, class_to_idx[fname])
                        images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s : s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            if is_image_file(imgname):
                path = os.path.join(imgs_path, imgname)
                item = (path, class_to_idx[cls_map[imgname]])
                images.append(item)

    return images

class TinyImageNet200(data.Dataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``tiny-imagenet-200`` exists.
        type (string): train or val or all
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    base_folder = 'tiny-imagenet-200'
    download_fname = "tiny-imagenet-200.zip"
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root, type='train',
                 transform=None, target_transform=None,
                 download=False, loader = 'pil'):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.type = type  # training set or test set
        self.fpath = os.path.join(root, self.download_fname)
        self.loader = loader

        if download:
            self.download()

#        if not check_integrity(self.fpath, self.md5):
#            raise RuntimeError('Dataset not found or corrupted.' +
#                               ' You can use download=True to download it')

        _, class_to_idx = find_classes(os.path.join(self.root, self.base_folder, 'wnids.txt'))
        self.class_to_idx = class_to_idx
        print("class_to_idx: ", len(class_to_idx))
        if type == 'train':
            self.data_info = make_dataset(self.root, self.base_folder, 'train', class_to_idx)
        elif type == 'val':
            self.data_info = make_dataset(self.root, self.base_folder, 'val', class_to_idx)
        elif type == 'all':
            self.data_info = make_dataset(self.root, self.base_folder, 'train', class_to_idx) + \
                             make_dataset(self.root, self.base_folder, 'val', class_to_idx)
        else:
            raise NotImplementedError

        if len(self.data_info) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img_path, target) where target is index of the target class.
        """

        img_path, target = self.data_info[index][0], self.data_info[index][1]
        if self.loader == 'pil':
            img = loadPILImage(img_path)
        else:
            img = loadCVImage(img_path)

        if self.transform is not None:
            result_img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
#        print('result_img size and type:', result_img.data[2][1])
        return result_img, target

    def __len__(self):
        return len(self.data_info)

    def download(self):
        import zipfile

        if check_integrity(self.fpath, self.md5):
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.base_folder, self.md5)

        # extract file
        dataset_zip = zipfile.ZipFile(self.fpath)
        dataset_zip.extractall()
        dataset_zip.close
