# print(outputs)
def extract_pred_classes(pred_classes,scores, meta_data):
  pred_class_names = []
  for i, pred_tensor in enumerate(pred_classes):
    for j, element in enumerate(pred_tensor):
      temp = {}
      int_label = int(element)
      # contig_label = meta_data.get("thing_dataset_id_to_contiguous_id")[int_label]
      label_name = meta_data.get("thing_classes")[int_label]
      # print(f"predicted class: {label_name}")
      temp["class_name"] = label_name
      temp["score"] = scores[i][j]
      pred_class_names.append(temp)
  return pred_class_names
def extract_labels_json(labels, reverse_mapping):
  names = []
  for label in labels:
    l = str(int(label))
    names.append(reverse_mapping[l])
  return names