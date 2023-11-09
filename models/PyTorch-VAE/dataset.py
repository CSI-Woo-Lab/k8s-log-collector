import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
from PIL import Image

import json
import cv2
import time
import sys
import logging
import numpy as np

#================================================================================
# Please change following path to your OWN
LOCAL_ROOT = '/home/shjeong/deepops/workloads/examples/k8s/datasets/CelebA_spoof/CelebA_Spoof/'
# LOCAL_IMAGE_LIST_PATH_TEST = 'metas/intra_test/test_label.json'
# LOCAL_IMAGE_LIST_PATH_TRAIN = 'metas/intra_test/train_label.json'
#================================================================================


def read_image(image_path):
    """
    Read an image from input path

    params:
        - image_local_path (str): the path of image.
    return:
        - image: Required image.
    """

    image_path = LOCAL_ROOT + image_path

    img = cv2.imread(image_path)
    # Get the shape of input image
    real_h,real_w,c = img.shape
    assert os.path.exists(image_path[:-4] + '_BB.txt'),'path not exists' + ' ' + image_path
    
    with open(image_path[:-4] + '_BB.txt','r') as f:
        material = f.readline()
        try:
            x,y,w,h,score = material.strip().split(' ')
        except:
            logging.info('Bounding Box of' + ' ' + image_path + ' ' + 'is wrong')   

        try:
            w = int(float(w))
            h = int(float(h))
            x = int(float(x))
            y = int(float(y))
            w = int(w*(real_w / 224))
            h = int(h*(real_h / 224))
            x = int(x*(real_w / 224))
            y = int(y*(real_h / 224))

            # Crop face based on its bounding box
            y1 = 0 if y < 0 else y
            x1 = 0 if x < 0 else x 
            y2 = real_h if y1 + h > real_h else y + h
            x2 = real_w if x1 + w > real_w else x + w
            img = img[y1:y2,x1:x2,:]

        except:
            logging.info('Cropping Bounding Box of' + ' ' + image_path + ' ' + 'goes wrong')   

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    # img = transforms.ToTensor()(img)
    return img


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class CelebASpoof(Dataset):
    """
    URL = https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z
    """
    def __init__(self,  
                data_path: str,
                transform: Callable,
                **kwargs):      
        self.transforms = transform
        self.data_path = data_path
        with open(self.data_path) as f:
            self.image_list = json.load(f)
        self.image_id_list = list(self.image_list) 
        self.base_folder = '/home/shjeong/deepops/workloads/examples/k8s/datasets/CelebA_spoof/CelebA_Spoof/'
        self.len = len(self.image_list)
    def __len__(self):
        # print("first:", len(self.image_list))
        # print("second_length:", len(self.image_list.keys()))
        return self.len
    
    def __getitem__(self, idx):
        image_id = self.image_id_list[idx]
        image = Image.open(os.path.join(self.base_folder, image_id))
        # image = default_loader(LOCAL_ROOT + image_id)
        # image = default_loader(image)
        image = self.transforms(image)
        # th_image = image
        # print('image:', th_image)
        return image, 1.0

    # def __iter__(self):
    #     """
    #     This function returns a iterator of image.
    #     It is used for local test of participating algorithms.
    #     Each iteration provides a tuple of (image_id, image), each image will be in RGB color format with array shape of (height, width, 3)
        
    #     return: tuple(image_id: str, image: numpy.array)
    #     """
    #     with open(self.data_path) as f:
    #         image_list = json.load(f)
    #     logging.info("got local image list, {} image".format(len(image_list.keys())))
    #     Batch_size = self.batch_size
    #     logging.info("Batch_size=, {}".format(Batch_size))
    #     n = 0
    #     final_image = []
    #     final_image_id = []
    #     for idx,image_id in enumerate(image_list):
    #         # get image from local file
    #         image = read_image(image_id)
    #         th_image = self.transforms(image)
            
    #         yield th_image, image_id
            # try:
            #     image = read_image(image_id)
            #     final_image.append(image)
            #     final_image_id.append(image_id)
            #     n += 1
            # except:
            #     logging.info("Failed to read image: {}".format(image_id))
            #     raise
            
            # yield th_final_image, final_image_id

            # if n == Batch_size or idx == len(image_list) - 1:
            #     np_final_image_id = np.array(final_image_id)
            #     np_final_image = np.array(final_image)
            #     th_final_image = self.tranform(torch.from_numpy(np_final_image))
            #     n = 0
            #     final_image = []
            #     final_image_id = []
            #     yield th_final_image, final_image_id
        
        

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path_train: str,
        data_path_test: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir_train = data_path_train
        self.data_dir_test = data_path_test
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.CenterCrop(148),
                                              transforms.Resize(self.patch_size),
                                              transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.CenterCrop(148),
                                            transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        

        # self.train_dataset = MyCelebA(
        #     self.data_dir_train,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )
        
        # # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        #     self.data_dir_test,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        # )
        
        self.train_dataset = CelebASpoof(
            self.data_dir_train,
            transform=train_transforms,
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = CelebASpoof(
            self.data_dir_test,
            transform=val_transforms,
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     