import torch
from detectron2.layers import interpolate
from detectron2.projects.point_rend.mask_head import calculate_uncertainty
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_on_grid,
    point_sample,
    point_sample_fine_grained_features,
)
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference, mask_rcnn_loss
from detectron2.modeling.postprocessing import detector_postprocess
import matplotlib.pyplot as plt
import numpy as np
import json
import cv2
import os

from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

def parse_data(outputs):
  batched_outputs = []
  instances = outputs[0]
  masks = outputs[1][0] # (N, 80, 7, 7)
  pred_class_list = []
  pred_scores = []
  mask_index = 0
  for j, inst in enumerate(instances):
    one_output = {}
    one_output['instance'] = inst
    is_empty = []
    for i, el in enumerate(inst.pred_classes):
      is_empty.append(i)
    # print(is_empty)
    if is_empty == []:
      one_output['instance'] = None
      one_output['masks'] = None
      batched_outputs.append(one_output)
      # print("None append")
      continue
    pred_class_list.append(inst.pred_classes)
    pred_classes = (inst.pred_classes)
    pred_scores.append(inst.scores)
    one_input_masks = {}
    all_masks = []

    for i, e in enumerate(pred_classes):
      # print(j)
      # print(mask_index)
      all_masks.append(masks[mask_index+i])
    mask_index += 1
    all_masks = torch.stack(all_masks)
    one_output['masks'] = all_masks # make this a tensor of (N, C, H, W)
    batched_outputs.append(one_output)
  # if batched_outputs == []:
  #   return None
  return batched_outputs, pred_class_list, pred_scores


def pointrend_head(model, batched_outputs, batched_images, gt_labels, device):
  batched_final_masks = []
  ind = 0
  undetected_inds = []
  for i , element in enumerate(batched_outputs):
        coarse_masks_logits = element['masks']
        #No detection boxes 
        if coarse_masks_logits is None:
          if device == 'cuda':
             zero_tensor = torch.zeros(3, 512, 512).to('cuda')
          else:
             zero_tensor = torch.zeros(3, 512, 512)
          batched_final_masks.append(zero_tensor)
          undetected_inds.append(i)
          continue

        single_input = [batched_images[i]]
        image = model.preprocess_image(single_input)
        img_height, img_width = image.image_sizes[0]
        features = model.backbone(image.tensor)
        mask_features_list = [
          features[k] for k in model.roi_heads.mask_head.mask_point_in_features
        ]
        features_scales = [
          model.roi_heads.mask_head._feature_scales[k]
          for k in model.roi_heads.mask_head.mask_point_in_features
        ]
        num_subdivision_steps = 5
        num_subdivision_points = 28 * 28

        mask_logits = coarse_masks_logits
        pred_boxes = element['instance'].pred_boxes
        pred_classes = element['instance'].pred_classes
        for subdivions_step in range(num_subdivision_steps):
              # Upsample mask prediction
              mask_logits = interpolate(
                  mask_logits, scale_factor=2, mode="bilinear", align_corners=False
              )
              # If `num_subdivision_points` is larger or equalt to the
              # resolution of the next step, then we can skip this step
              H, W = mask_logits.shape[-2:]
              if (
                num_subdivision_points >= 4 * H * W
                and subdivions_step < num_subdivision_steps - 1
              ):
                continue

              uncertainty_map = calculate_uncertainty(mask_logits, pred_classes)
              point_indices, point_coords = get_uncertain_point_coords_on_grid(
                  uncertainty_map,
                  num_subdivision_points
              )

              # extract fine-grained and coarse features for the points
              fine_grained_features, _ = point_sample_fine_grained_features(
                mask_features_list, features_scales, [pred_boxes], point_coords
              )
              coarse_features = point_sample(coarse_masks_logits, point_coords, align_corners=False)

              point_logits = model.roi_heads.mask_head.point_head(fine_grained_features, coarse_features)
              # put mask point predictions to the right places on the upsampled grid.
              R, C, H, W = mask_logits.shape
              x = (point_indices[0] % W).to("cpu")
              y = (point_indices[0] // W).to("cpu")
              point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
              mask_logits = (
                mask_logits.reshape(R, C, H * W)
                .scatter_(2, point_indices, point_logits)
                .view(R, C, H, W)
              )
        detected_instances = [element['instance']]
        mask_rcnn_inference(mask_logits, detected_instances)
        results = detector_postprocess(detected_instances[0], img_height, img_width, 0.85) # for classifier
        binary_masks = results.pred_masks
        confidence_score = 0.0
        num_of_masks = binary_masks.shape[0]

        masked_array = get_relevant_mask(results, coco_metadata, gt_labels[i])
        masked_array = masked_array.cpu().numpy()
        masked_image = (masked_array * 255).astype(np.uint8)
        masked_image_bgr = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(os.path.join('/content/', f'mask_{ind + 1}_mask_binary.png'), masked_image_bgr)
        print(f"foreground mask writte to {os.path.join('/content/', f'mask_{ind + 1}_mask_binary.png')}")
        ind += 1
        if device == 'cuda':
           masked_image = torch.tensor(masked_image_bgr).to('cuda')
        else:
           masked_image = torch.tensor(masked_image_bgr)
        masked_image = masked_image.to(torch.float32)
        masked_image = masked_image.permute(2, 0, 1)
        batched_final_masks.append(masked_image) # 224x224 for classifier
  batched_final_masks = torch.stack(batched_final_masks)
  return batched_final_masks, undetected_inds




def get_relevant_mask(results, coco_metadata, gt_labels):
  with open("/content/Silhouette_based_recog/labels_mappings_reversed.json", 'r') as f:
    labels_names = json.load(f)
  with open("/content/Silhouette_based_recog/co3d_to_coco.json", 'r') as fd:
    coco_mapping = json.load(fd)
  pred_class = results.pred_classes
  l = str(int(gt_labels))
#   for i, pred_c in enumerate(pred_class):
#     if coco_metadata.thing_classes[pred_c] =='donut' and coco_mapping[labels_names[l]] =='cake':
#       return results.pred_masks[i, :, :]
#     if coco_metadata.thing_classes[pred_c] =='cake' and coco_mapping[labels_names[l]] =='donut':
#       return results.pred_masks[i, :, :]
#     if coco_metadata.thing_classes[pred_c] == coco_mapping[labels_names[l]]:
#       return results.pred_masks[i, :, :]
  max_value, max_index = torch.max(results.scores, dim=0)
  masked_array = results.pred_masks[max_index,:,:]
  return masked_array



def extract_coarse_mask_instance(model, batched_input):
  for param in model.roi_heads.parameters():
      param.requires_grad = False
  outputs = model(batched_input)
  return outputs
