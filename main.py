import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2

import torch.nn as nn
import torch.optim as optim

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")
import os
# import PointRend project
from detectron2.projects import point_rend
from rcnn_modified import GeneralizedRCNNModified
from roi_heads_modified import StandardROIHeadsModified
from rpn import RPN_Modified
from mask_head_pointrend import PointRendMaskHeadModified
from utils import parse_data, extract_coarse_mask_instance, pointrend_head
device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

from dataloader import ShapeDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import json
import wandb

print(device)



cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"

cfg.MODEL.DEVICE= device
cfg.MODEL.META_ARCHITECTURE = 'GeneralizedRCNNModified'
cfg.MODEL.PROPOSAL_GENERATOR.NAME = 'RPN_Modified'
cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeadsModified'
cfg.MODEL.ROI_MASK_HEAD.NAME = 'PointRendMaskHeadModified'
torch.cuda.empty_cache()
predictor = DefaultPredictor(cfg)

transform = transforms.Compose([
transforms.PILToTensor(),
transforms.Resize((512, 512)),
])
train_dataset = ShapeDataset(data_dir="/home/dark/tahira", split='train', transform=transform)
val_dataset = ShapeDataset(data_dir='/home/dark/tahira', split = 'val', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=14, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=14)


model = predictor.model
#for now the segmenter works as it works during inference, so model.train() works for both. I will separate model.train and model.eval, a little later
model.train()
dataiter = iter(train_dataloader)
images, labels = next(dataiter)

batched_input = []
print(images.shape)


import random
#this is to fill the undetected instances when in batch training
model = predictor.model
def fill_undetected_instances(labels, undetected_inds, batched_final_masks):
  if undetected_inds !=[]:
    choose = None
    r = len(undetected_inds)
    for i, e in enumerate(labels):
      if e not in undetected_inds:
        choose = i
        break
    # print(f"choose: {choose}")
    if choose != None:
      tensor_ind = batched_final_masks[choose]
      for e in undetected_inds:
        batched_final_masks[e] =  tensor_ind
        labels[e] = labels[choose]
      return batched_final_masks, labels
    else:
      return None




batch_size = 14
learning_rate = 0.0001
patience = 3
LOG_DIR = 'checkpoints'
device = 'cuda'
LOAD_CHKPT = True
epochs=20
LOG_DIR = '/home/dark/tahira/checkpoints'


transform_classifier = transforms.Compose([
    # transforms.ToTensor(),
    transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
classifier = torchvision.models.resnet50(weights=None)
num_features = classifier.fc.in_features
num_classes = 20
classifier.fc = nn.Linear(num_features, num_classes)




criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(classifier.parameters())+list(model.backbone.parameters()), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience= patience, factor=0.1)
for param in model.roi_heads.parameters():
      param.requires_grad = False
for param in model.proposal_generator.parameters():
      param.requires_grad = False

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience= patience, factor=0.1)

# Move model to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
classifier.to(device)


config = {
    'epoch': epochs,
    'criterion': criterion,
    'train_loader': train_dataloader,
    'val_loader': val_dataloader,
    'device': device,
    'log_dir': LOG_DIR,
    'scheduler': scheduler,
    'patience': patience
}


if LOAD_CHKPT:
    print('Loading the classifier from the checkpoint')
    checkpoint_path = os.path.join('/home/dark/tahira/Silhouette_based_recog/checkpoints', 'checkpoint_class.pt')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    classifier.load_state_dict(checkpoint)

if LOAD_CHKPT:
    print('Loading the model from the checkpoint')
    checkpoint_path = os.path.join('/home/dark/tahira/Silhouette_based_recog/checkpoints', 'checkpoint_seg.pt')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    model.load_state_dict(checkpoint)

import json
def train(config, model, classifier):
    train_dataloader = config['train_loader']
    val_dataloader = config['val_loader']
    device = config['device']
    scheduler = config['scheduler']
    patience = config['patience']
    LOG_DIR = config['log_dir']
    best_val_loss = float('inf')
    early_stopping_counter = 0

    with open("/home/dark/tahira/Silhouette_based_recog/labels_mappings_reversed.json", 'r') as json_map:
      reverse_mapping = json.load(json_map)
    for epoch in range(epochs):
          running_loss = 0.0
          classifier.train()
          for data in train_dataloader:

                image, label = data
                if torch.cuda.is_available():
                    image, label = image.cuda(), label.cuda()
                optimizer.zero_grad()
                batched_input = []
                for t in range(image.shape[0]):
                    batched_input.append({"image": image[t]})
                outputs = extract_coarse_mask_instance(model, batched_input)
                res = parse_data(outputs)
                if res is None:
                  print("no instance detected")
                  continue
                batched_outputs, pred_classes, pred_scores = res
                res = pointrend_head(model, batched_outputs, batched_input, label, device)

                batched_final_masks, undetected_inds = res
                # print(f"batched_final_mask shape before filter: {batched_final_masks.shape}")
                un_result = fill_undetected_instances(label, undetected_inds, batched_final_masks)
                if un_result is None:
                  continue
                batched_final_masks, label = un_result
                # print(f"batched_final_mask shape after filter: {batched_final_masks.shape}")
                im = transform_classifier(batched_final_masks)
                if torch.cuda.is_available():
                    im = im.cuda()
                preds = classifier(im)
                loss = criterion(preds, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * image.size(0)
          epoch_loss = running_loss / len(train_dataloader.dataset)
          print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")


          classifier.eval()
          # model.eval()
          val_loss = 0.0
          total = 0
          correct = 0
          with torch.no_grad():
            for image, label in val_dataloader:
                if torch.cuda.is_available():
                    image, label = image.cuda(), label.cuda()
                # images, labels = images.cuda(), labels.cuda()
                batched_input = []
                for t in range(image.shape[0]):
                  batched_input.append({"image": image[t]})
                outputs = extract_coarse_mask_instance(model, batched_input)
                res = parse_data(outputs)
                if res is None:
                  print("no instance detected")
                  continue

                batched_outputs, pred_classes, pred_scores = res
                res = pointrend_head(model, batched_outputs, batched_input, label, device)

                batched_final_masks, undetected_inds = res
                un_result = fill_undetected_instances(label, undetected_inds, batched_final_masks)
                if un_result is None:
                    continue
                batched_final_masks, label = un_result
                im = transform_classifier(batched_final_masks)
                if torch.cuda.is_available():
                    im = im.cuda()
                    label = label.cuda()
                preds = classifier(im)
                # if device =='cuda':
                #   preds = preds.cuda()
                loss = criterion(preds, label)
                val_loss += loss.item() * image.size(0)
                _, prediction = torch.max(preds, 1)
                total += label.size(0)
                correct += (prediction == label).sum().item()
            val_loss_total = val_loss / len(val_dataloader.dataset)
            accuracy = correct / total
            print(f"Validation Loss: {val_loss_total:.4f}, Accuracy: {accuracy:.2%}")
            print(f"validation dataloader size {len(val_dataloader.dataset)}")
            scheduler.step(val_loss_total)

            new_learning_rate = scheduler.get_last_lr()
            print("New learning rate:", new_learning_rate)
            print("Last learning rate:", scheduler._last_lr)
            print("Number of epochs since last improvement:", scheduler.num_bad_epochs)
            wandb.log({"acc": accuracy, "train_loss": epoch_loss, "val_loss": val_loss_total, "learning_rate": new_learning_rate})
            if val_loss_total < best_val_loss:
                best_val_loss = val_loss_total
                early_stopping_counter = 0
                print('Saving the best model, end of epoch %d' % (epoch+1))
                if not os.path.exists(LOG_DIR):
                    os.makedirs(LOG_DIR)
                torch.save(classifier.state_dict(), os.path.join(LOG_DIR,'checkpoint_class.pt'))
                torch.save(model.state_dict(), os.path.join(LOG_DIR,'checkpoint_seg.pt'))
            else:
                early_stopping_counter += 1
            if early_stopping_counter >= 10:
                print("Validation loss has not improved for {} epochs. Early stopping...".format(patience))
                break



if __name__ == '__main__':
    wandb.login()
    run = 5
    wandb.init(
        project="Shape_Pointrend_20_classifier_backbone_train",
        name=f"experiment_class20_{run}_lr_{learning_rate}_batch_{batch_size}",
        config={
        "learning_rate": learning_rate,
        "architecture": "ResNet50",
        "dataset": "CO3D_poinrendMasks",
        "epochs": epochs,
        })
    train(config, model, classifier)
    wandb.finish()