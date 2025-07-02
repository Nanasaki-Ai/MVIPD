import os
import torch
import argparse
import pickle
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class DataSetReader(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.viewpoint = args.viewpoint
        self.data = args.data

        label_file = 'train_labels.pkl' if mode == 'train' else 'test_labels.pkl'
        label_path = os.path.join(args.label_path, args.benchmark, label_file)
        with open(label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # Path setting
        folder_map = {
            'ori': ('train/images', 'valid/images'),
            'yolo_fci': ('fci_train', 'yolov11n_fci'),
        }
        
        if self.data == 'main_auxiliary':
            self.data_path_main = os.path.join(args.data_path, folder_map['yolo_fci'][0] if mode == 'train' else folder_map['yolo_fci'][1])
            self.data_path_auxiliary = os.path.join(args.data_path, folder_map['ori'][0] if mode == 'train' else folder_map['ori'][1])
        else:
            folder = folder_map[self.data][0] if mode == 'train' else folder_map[self.data][1]
            self.data_path = os.path.join(args.data_path, folder)

        self.transform = transforms.Compose([
            transforms.Resize((180, 320)),
            #transforms.Resize((270, 480)),
            transforms.ToTensor(),
        ])
        
        self.transform_320 = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.label)

    def _load_and_transform_image(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return self.transform(Image.open(path).convert('RGB'))
    
    def _load_and_transform_image_320(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")
        return self.transform_320(Image.open(path).convert('RGB'))
    
    
    def _get_image_paths(self, sample_name):
        return [os.path.join(self.data_path, p) for p in sample_name.split()]
    
    def _get_formulanetl_image_paths(self, sample_name):
        return [os.path.join(self.data_path_main, p) for p in sample_name.split()], [os.path.join(self.data_path_auxiliary, p) for p in sample_name.split()]

    def __getitem__(self, index):
        label = int(self.label[index])
        sample_name = self.sample_name[index]

        if self.viewpoint == 'sv':
            path = os.path.join(self.data_path, sample_name)
            view = self._load_and_transform_image(path)
            return view, label, sample_name
        
        if self.viewpoint in ['mv', 'mv3', 'mv3_320', 'mv3_cat']:
            paths = self._get_image_paths(sample_name)
            if self.viewpoint == 'mv3_320':
                views = [self._load_and_transform_image_320(p) for p in paths]
            else:
                views = [self._load_and_transform_image(p) for p in paths]
        if self.viewpoint == 'formulanet':
            paths_main, paths_auxiliary = self._get_formulanetl_image_paths(sample_name)
            views_main = [self._load_and_transform_image(p) for p in paths_main]
            views_auxiliary = [self._load_and_transform_image(p) for p in paths_auxiliary]            
        
        if self.viewpoint == 'mv':
            return views[0], views[1], label, sample_name
        elif self.viewpoint in ['mv3', 'mv3_320']:
            return views[0], views[1], views[2], label, sample_name
        elif self.viewpoint == 'mv3_cat':
            view_cat = torch.cat(views, dim=1)  # 
            return view_cat, label, sample_name
        elif self.viewpoint == 'formulanet':
            main_cat = torch.cat(views_main, dim=1)
            auxiliary_cat = torch.cat(views_auxiliary, dim=1)
            return main_cat, auxiliary_cat, label, sample_name

        else:
            raise ValueError(f"Unsupported viewpoint type: {self.viewpoint}")
            