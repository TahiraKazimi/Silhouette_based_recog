from ast import parse
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
# import wandb


print(device)




cfg = get_cfg()
# Add PointRend-specific config
point_rend.add_pointrend_config(cfg)
# Load a config from file
cfg.merge_from_file("/home/tahira/detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
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
transforms.Resize((800, 800)),
])
train_dataset = ShapeDataset(data_dir="/home/tahira/downloads_old", split='train', transform=transform)
val_dataset = ShapeDataset(data_dir='/home/tahira/downloads_old', split = 'val', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1)


model = predictor.model
#for now the segmenter works as it works during inference, so model.train() works for both. I will separate model.train and model.eval, a little later
model.train()
dataiter = iter(train_dataloader)
images, labels = next(dataiter)
batched_input = []
print(images.shape)
for t in range(images.shape[0]):
    batched_input.append({"image": images[t]})

transform = T.ToPILImage()
pil_img = transform(images[0])
# plt.imshow(pil_img)
# plt.show()
# plt.imsave("/home/downloads_old/Silhouette_based_recog")

from utils import parse_data, extract_coarse_mask_instance, pointrend_head

images= model.preprocess_image(batched_input)
print(images.tensor.dim())
features_2 = model.backbone(images.tensor)


#region proposal 
proposals, _ = model.proposal_generator(images, features_2, None)


features = [features_2[f] for f in model.roi_heads.box_in_features]
box_features = model.roi_heads.box_pooler(features, [x.proposal_boxes for x in proposals])
print(box_features.shape)
box_features = model.roi_heads.box_head(box_features)
print(box_features.shape)
predictions = model.roi_heads.box_predictor(box_features)
prediction_scores, prediction_deltas = predictions
##thresholding the maximum predicted regions
pred_instances, inds = model.roi_heads.box_predictor.inference(predictions, proposals)
print(f"prediction scores len: {prediction_scores.shape}")
if inds:
    print("instances detected")
    detected_class_logits = prediction_scores[inds[0]]
    pred_classes = prediction_scores.argmax(dim=1)[inds[0]]
    print(f"detected_logits: {detected_class_logits.shape}")
    
    print("coarse mask prediction")
    res = model.roi_heads.mask_head(features_2, pred_instances)
    
    print(f"res : {res[0].shape}")
    res = parse_data((pred_instances,res))
    
    if res is None:
        print("no instance detected")
    batched_outputs, _, _ = res
    res = pointrend_head(model, batched_outputs, batched_input, labels, device, write_enabled=True)
    print(f"pred_masks {pred_instances}")
print(f"pred classes : {pred_classes}")
print(inds)
print(pred_instances[0].pred_masks.shape)

# print(coco_metadata.thing_classes)