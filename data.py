# coding: utf-8
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader,Dataset
from scipy.io import loadmat
import os 
import numpy as np
from torch import tensor

input_size = 224

class CarDataset(Dataset):
    def __init__(self, mat_path = 'data/cars_annos.mat',transform = None,test = False):
        #read data
        data = loadmat(mat_path)
        annotations = data['annotations']
        self.test = test

        #匿名函数，由于读取的数据都用array包起来了（eg :array(['car_ims/000001.jpg'], dtype='<U18')），所以把它取出来
        get_elem = lambda elem: elem[0] if elem.shape == (1,) else elem[0,0]

        self.relative_im_path = np.array(list(map(get_elem,annotations['relative_im_path'].reshape(-1))))
        self.bbox_x1 = tensor(np.array(list(map(get_elem,annotations['bbox_x1'].reshape(-1)))).astype('int32')).long()
        self.bbox_x2 = tensor(np.array(list(map(get_elem,annotations['bbox_x2'].reshape(-1)))).astype('int32')).long()
        self.bbox_y1 = tensor(np.array(list(map(get_elem,annotations['bbox_y1'].reshape(-1)))).astype('int32')).long()
        self.bbox_y1 = tensor(np.array(list(map(get_elem,annotations['bbox_y1'].reshape(-1)))).astype('int32')).long()
        self.labels = tensor(np.array(list(map(get_elem,annotations['class'].reshape(-1)))).astype('int32')-1).long() #-1 for (1-196) convert to (0,195)
        self.is_test =np.array(list(map(get_elem,annotations['test'].reshape(-1))))
        #read class_names
        class_names = list(map(get_elem,data['class_names'][0].reshape(-1)))

        ##pack to a dictionary
        self.class_dict = {}
        for i,elem in enumerate(list(class_names)):
            self.class_dict[elem] = i
        
        self.transform = transform

    def __getitem__(self, index):
        path = self.relative_im_path[self.is_test == self.test][index]
        label = self.labels[self.is_test == self.test][index]

        img = Image.open(os.path.join('data',path)).convert('RGB') 
        if self.transform is not None:
            img = self.transform(img) 
        
        return img, label

    def __len__(self):
        return sum(self.is_test) if self.test else sum(1-self.is_test)

def load_data(batch_size = 32,input_size = 224,data_augmentation = 0):
    train_transform = { 0 : transforms.Compose(
            [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),#0.5,0.5,0.5,0.5
                # transforms.RandomChoice([transforms.MotionBlur(blur_limit=3),transforms.MedianBlur(blur_limit=3), transforms.GaussianBlur(blur_limit=3),], p=0.5,),
                transforms.RandomAffine(degrees = 0,translate=(0.1,0.1)),#
                transforms.RandomRotation(40),
                transforms.RandomGrayscale(p=0.3),
                # transforms.RandomChoice([transforms.GaussianBlur(11),transforms.GaussianBlur(5),transforms.RandomGrayscale(p=0.3),transforms.RandomEqualize(p=0.3)]),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomResizedCrop(input_size),
                transforms.Resize(size = (input_size,input_size)),
                transforms.CenterCrop(size = (input_size,input_size)),
                transforms.ToTensor(),
                # transforms.RandomErasing(p = 0.5,value = 'random'),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
        #https://github.com/0chaoxin1/Car_Recogniztion/blob/main/train.py
        #https://github.com/eqy/PyTorch-Stanford-Cars-Baselines/blob/54dd3ccbc25cd4e0a3893bbbe23af4d83f0dab80/main.py
        1: transforms.Compose(
            [
                # transforms.RandomRotation(20),
                # transforms.RandomAffine(degrees = 0,translate=(0.1,0.1)),#
                # transforms.RandomGrayscale(p=0.3),
                # transforms.RandomChoice([transforms.GaussianBlur(11),transforms.GaussianBlur(5),transforms.RandomGrayscale(p=0.3),transforms.RandomEqualize(p=0.3)]),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.Resize(size = (input_size,input_size)),
                # transforms.CenterCrop(size = (input_size,input_size)),
                transforms.ToTensor(),
                # transforms.RandomErasing(p = 0.5,value = 'random'),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        ),
    }
    test_transform = transforms.Compose(
        [
            transforms.Resize(size = (input_size,input_size)),
            transforms.CenterCrop(size = (input_size,input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
        )
    
    
    train_dataset = CarDataset(transform = train_transform[data_augmentation])
    test_dataset = CarDataset(transform = test_transform,test= True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers =2 ,pin_memory = False,drop_last  = False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,num_workers = 2 ,pin_memory= False ,drop_last= False)

    return train_loader,test_loader