from builtins import breakpoint
import os
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
from .augmentations import RandAugment
from tools.function import get_pkl_rootpath
import torchvision.transforms as T
import torch


class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        # assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
        #     f'dataset name {args.dataset} is not exist'
        if args.dataset == 'RAP':
        # if 1:
            if split == 'test':
                path = '/home/zhexuan_wh/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition-master/data/rap/test.txt'
            else:
                path = '/home/zhexuan_wh/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition-master/data/rap/train.txt'
            with open(path) as f:
                lines = f.read().split('\n')
                lines = lines[0: len(lines) - 1]

            img_id = []
            label = []
            for line in lines:
                index = line.find('png')
                img_id.append(line[0:index+3])
                l = line[index+3: len(line)].split()
                l = np.array([int(x) for x in l]).astype('uint8')
                label.append(l)
            self.img_id = img_id
            self.label = np.array(label) 
            self.transform = transform
            self.target_transform = target_transform
            self.attr_num = 51
            self.dataset = 'RAP'
            self.root_path = '/home/zhexuan_wh/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition-master/data/rap/images'
        else:
            data_path = get_pkl_rootpath(args.dataset)
            
            breakpoint()
            dataset_info = pickle.load(open(data_path, 'rb+'))

            img_id = dataset_info.image_name
            attr_label = dataset_info.label

            assert split in dataset_info.partition.keys(), f'split {split} is not exist'

            self.dataset = args.dataset
            self.transform = transform
            self.target_transform = target_transform

            self.root_path = dataset_info.root

            self.attr_id = dataset_info.attr_name
            self.attr_num = len(self.attr_id)

            self.img_idx = dataset_info.partition[split]

            if isinstance(self.img_idx, list):
                self.img_idx = self.img_idx[0]  # default partition 0
            self.img_num = self.img_idx.shape[0]
            self.img_id = [img_id[i] for i in self.img_idx]
            self.label = attr_label[self.img_idx]

    def __getitem__(self, index):

        imgname, gt_label = self.img_id[index], self.label[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)

class AddGaussianNoise(object):
    def __init__(self, mean=[0., 0., 0.], std=[1., 1., 1.], p=0.5):
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.p = p
        
    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            return tensor + torch.randn_like(tensor).to(tensor.device) * self.std + self.mean
        return tensor
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, p={2})'.format(self.mean, self.std, self.p)

def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # normalize = T.Normalize(mean=[0.357, 0.323, 0.328], std=[0.252, 0.242, 0.239])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        # AddGaussianNoise([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], 0.5),  # 添加高斯噪声
        normalize,
    ])
    if args.randAug:
        print('Use Rand Augmentation', args.n, args.m)
        train_transform.transforms.insert(1, RandAugment(args.n, args.m))
        

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform

if __name__ == '__main__':
    pass
    # d = AttrDataset('test', None)