import os
import random
from pathlib import Path
import pandas as pd
import torch
from config.conf import cfg
from torch.utils.data import Dataset, DataLoader
# from robustbench.data import load_cifar10c, load_cifar100c
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# def create_df(cfg, corruption=None, mode='train'):
#     filelist = []
#     assert os.path.isdir(cfg.DATA_DIR), 'Invalid path to dataset is inserted'
#     for root, _ , files in os.walk(cfg.DATA_DIR):
#         for file in files:
#             filelist.append(os.path.normpath(os.path.join(root,file)).split(os.path.sep))
#     columns = ['root','severity','corruption','mode','class','sample']
#     df = pd.DataFrame(filelist, columns=columns)
        
#     if corruption is not None:
#         corruption = [corruption] if not isinstance(corruption, list) else corruption
#         df = df[df['corruption'].isin(corruption)]
            
#     df ['path'] = (df['root'].astype(str) + '/' + 
#                         df['severity'].astype(str) + '/' +
#                         df['corruption'].astype(str) + '/' +
#                         df['mode'].astype(str) + '/' +
#                         df['class'].astype(str) + '/' +
#                         df['sample'].astype(str)  )
    
#     df = df[df['mode']==mode]  
#     df['target'] = df['class'].map(lambda x: cfg.DATA.CLASSES.index(x))
#     df['corruption'] = df['corruption'].map(lambda x: cfg.CORRUPTION.TYPE.index(x)) # change here
#     df = df.sample(frac=1,random_state=cfg.BASE.SEED).reset_index(drop=True)   
#     return df

# class Dataset_Corrupted(Dataset):
#     def __init__(self, data_list, corruption=None,transform=None,mode='test') -> None:
#         #self.data_dir = data_dir
#         #self.annotations = create_df_cifar10(filepath=data_dir,noise=noise,mode=mode)
#         self.annotations = data_list
#         self.n_classes = len(self.annotations['target'].unique())
#         self.n_domain = len(self.annotations['corruption'].unique())
#         self.transform = transform
#         self.mode=mode
#         to_tensor = []
#         ## Resize the tensor here if needed
#         to_tensor += [transforms.ToTensor()]
#         to_tensor += [transforms.Normalize(
#             mean = (0.4914, 0.4822, 0.4465),
#             std = (0.2470, 0.2435, 0.2616)) # (0.2470, 0.2435, 0.2616) (0.2023, 0.1994, 0.2010)
#             ]
#         # to_tensor += [transforms.Normalize(
#         #    mean =[0.4914, 0.4822, 0.4465], std =[0.2470, 0.2435, 0.2616]) # (0.2470, 0.2435, 0.2616) (0.2023, 0.1994, 0.2010)
#         #     ]
#         self.to_tensor = transforms.Compose(to_tensor)
        
        
#     def __len__(self):
#         return len(self.annotations)
        
#     def __getitem__(self, index):
#         data_path = self.annotations.iloc[index]['path'] 
#         data = Image.open(data_path)
#         #data = torch.load(data_path)
#         if self.transform and self.mode=='train':
#             data = self.transform(data)
#         data = self.to_tensor(data)
        
#         # data = torch.tensor(data) 
#         # data = torch.permute(data,(2,0,1))
#         label = torch.tensor(self.annotations.iloc[index]['target']) 
#         domain = torch.tensor(self.annotations.iloc[index]['corruption']) 
#         sample = {'data': data, 'label': label, 'domain':domain}
#         return sample


# def get_loader(cfg, mode= 'test', corelated=False, gamma=0.1, corruption=None):

#     data_list = create_df(cfg, corruption=corruption, mode=mode)
#     assert len(data_list) !=0 , "No data found in the data list"
#     dataset = Dataset_Corrupted(
#         data_list=data_list,
#         corruption=corruption,
#         mode = mode
#         )
    
#     shuffle = True if mode == 'train' else False
#     #shuffle = True
#     data_loader = DataLoader(dataset=dataset, 
#                               batch_size = cfg.DATA.BATCH_SIZE, 
#                               shuffle=shuffle,
#                               num_workers=cfg.BASE.NUM_WORKERS)
#     # data_loader = DataLoader(dataset=dataset, 
#     #                           batch_size = args['batch_size'], 
#     #                           shuffle=False,
#     #                           sampler= DistributedSampler(dataset),
#     #                           num_workers=8)
#     if corelated and mode =='test':
#         data_loader = DataLoader(dataset=dataset, 
#                               batch_size = cfg.DATA.BATCH_SIZE, 
#                               shuffle=shuffle,
#                               #sampler=CorrelatedSampler(data_list=data_list,gamma=gamma), #gamma value
#                               num_workers=cfg.BASE.NUM_WORKERS)
        
#     return data_loader


def get_fisher_loader(cfg,
                file_name = 'GOLD_XYZ_OSC.0001_1024.hdf5',
                snr=10,
                batch_size = 64,
                train=True,
                num_workers=4,
                shuffle = False,
                ):

    g = torch.Generator()
    g.manual_seed(cfg.BASE.SEED)    
    dataset = RadioML_Datset(file_name=file_name, snr=snr, train=train)
    dataset_size = len(dataset)
    dataset_indices = list(range(dataset_size))
    np.random.seed(0)
    np.random.shuffle(dataset_indices)
    fisher_split_index = int(np.floor(0.1*dataset_size))
    fisher_idx = dataset_indices[:fisher_split_index]
    fisher_sampler = SubsetRandomSampler(fisher_idx)

    fisher_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, 
                            sampler=fisher_sampler,worker_init_fn=seed_worker, shuffle = shuffle,
                            generator=g, num_workers=num_workers)
    return fisher_dataloader


# def sanity_check(dataset='cifar10c'):

#     dataloader = get_loader(cfg,corruptions='fog')
#     for i,data in enumerate(dataloader):
#         x , y = data['img'], data['label']
#         print(f'data shape: {x.size()}')
#         print(f'label shape: {y.size()}')
#         break

# if __name__ == '__main__':

#     sanity_check('cifar10c')
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(cfg.BASE.SEED)
    random.seed(cfg.BASE.SEED)

def get_transform(mode='test'):
    transform = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        ]),
        'test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        ])
    }
    return transform[mode] 

def get_corrupted_dataset(cfg, corruption=None):
    data_dir = Path(cfg.DATA.PATH) / f"severity_{cfg.DATA.SEVERITY}"/ corruption / cfg.DATA.MODE
    transform = get_transform(mode=cfg.DATA.MODE)
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    return dataset 

def get_loader(cfg, corruption):
    g = torch.Generator()
    g.manual_seed(cfg.BASE.SEED)

    dataloader = torch.utils.data.DataLoader(
        dataset = get_corrupted_dataset(cfg, corruption),
        batch_size = cfg.DATA.BATCH_SIZE,
        generator = g,
        shuffle = True, # for test we also need shuffle to generate diversified batch of data
        num_workers = cfg.BASE.NUM_WORKERS
    )
    return dataloader



    
# dataloader = get_loader(cfg,corruptions='fog')
    
# for i,data in enumerate(dataloader):
#     x , y = data['img'], data['label']
#     print(f'data shape: {x.size()}')
#     print(f'label shape: {y.size()}')
        
        


