import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
import os
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import random
from detectron2.data import transforms as T
import cv2
class ShapeDataset(Dataset):
    def __init__(self, data_dir, label_path=None, label_path_r=None, split='train', transform=None, val_size=0.2):
        self.data_dir = os.path.join(data_dir, "image_data")
        self.transform = transform
        self.label_path = label_path
        # self.test_size = test_size
        self.val_size = val_size
        self.split = split
        self.label_path_reverse = label_path_r
        self.image_files, self.labels = self.load_data(self.split)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        # print(img_name)
        img_path = os.path.join(self.split_path, img_name)
        image = cv2.imread(img_path)
        # cv2_imshow(image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_transformed = T.ResizeShortestEdge(800, 1333).get_transform(image).apply_image(image)
        # cv2_imshow(im_transformed)
        im_transformed = cv2.cvtColor(im_transformed, cv2.COLOR_BGR2RGB)
        # batched_inputs = torch.as_tensor(im_transformed).permute(2, 0, 1)

        image_pil = Image.fromarray(im_transformed)
        tensor = self.transform(image_pil)
        # print(tensor.shape)
        label = self.labels[idx]
        return tensor, label

    def load_data(self, split):
      with open('/content/labels_mappings.json', 'r') as json_labels:
        label_mapping = json.load(json_labels)
      if split == 'train':
        self.split_path = os.path.join(self.data_dir, 'train')
        image_files = [x for x in os.listdir(os.path.join(self.data_dir, 'train'))]
        random.shuffle(image_files)
        labels = []
        for img in image_files:
          label_name = img.split('_')[0]
          label = label_mapping[label_name]
          labels.append(label)
        return image_files, labels
      elif split == 'val':
        self.split_path = os.path.join(self.data_dir, 'val')
        image_files = [x for x in os.listdir(os.path.join(self.data_dir, 'val'))]
        random.shuffle(image_files)
        labels = []
        for img in image_files:
          label_name = img.split('_')[0]
          label = label_mapping[label_name]
          labels.append(label)
        return image_files, labels
      else:
        self.split_path = os.path.join(self.data_dir, 'test')
        image_files = [x for x in os.listdir(os.path.join(self.data_dir, 'test'))]
        random.shuffle(image_files)
        labels = []
        for img in image_files:
          label_name = img.split('_')[0]
          label = label_mapping[label_name]
          labels.append(label)
        return image_files, labels

