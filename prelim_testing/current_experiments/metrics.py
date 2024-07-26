import os
import heapq
import torch
import torch.nn as nn
import numpy as np
import clip
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
CLIP_prompts = ["a dog", "an innocent person", "a cat", "a person holding a gun"]

class ImageDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_list[idx])
        image = preprocess(Image.open(img_name))

        return image # [3,224,224]

class ImageFeaturesDataset(Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.image_list = os.listdir(folder_path)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.folder_path, self.image_list[idx])
        image1_preprocess = preprocess(Image.open(img_name)).unsqueeze(0).to(device)
        image1_features = model.encode_image(image1_preprocess)

        return image1_features # [1,512]

def getImageTensorsFrom(folder_path):
  dataset = ImageDataset(folder_path)
  data_loader = DataLoader(dataset, batch_size=1, shuffle=False)#, collate_fn=custom_collate)

  return data_loader

def getFeatureTensorsFrom(folder_path):
  dataset = ImageFeaturesDataset(folder_path)
  data_loader = DataLoader(dataset, batch_size=1, shuffle=False)#, collate_fn=custom_collate)

  return data_loader

def getCLIPPromptScores(generated_images, prompt=0):
  metrics = ["CLIP_promptsimilarity"]
  all_scores = []

  if prompt == 0:
    labels = CLIP_prompts

  metric_values = {metric: np.zeros(len(labels)) for metric in metrics}

  for _, generated_tensor in enumerate(generated_images):

    image = generated_tensor.to(device)
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
      logits_per_image, logits_per_text = model(image, text)
      probs = (logits_per_image.softmax(dim=-1).cpu().numpy())[0]

      metric_values["CLIP_promptsimilarity"] += probs
      all_scores.append(probs)

  for metric in metric_values:
    metric_values[metric] /= (len(generated_images))

  return metric_values, all_scores

# takes in a dataloader for original image features and generated image features (each image has a feature entry with shape [1,512])
def getCLIPSimilarityScores(og_images, generated_images):
  metrics = ["CLIP_imgsimilarity"]
  metric_values = {metric: 0 for metric in metrics}
  all_scores = []

  cos = torch.nn.CosineSimilarity(dim=0)

  for _, og_tensor in enumerate(og_images):
    image1_features = (og_tensor.squeeze(0))

    for _, generated_tensor in enumerate(generated_images):
      image2_features = (generated_tensor.squeeze(0))
      print('TENSOR SIZES', image1_features.size(), image2_features.size())
      similarity = cos(image1_features[0],image2_features[0]).item()
      similarity = (similarity+1)/2
      metric_values["CLIP_imgsimilarity"] += similarity
      all_scores.append(similarity)

  for metric in metric_values:
    metric_values[metric] /= (len(og_images) * len(generated_images))

  return metric_values, all_scores

def CLIP_similarity_metric(original_path, cloaked_path, uncloaked_path):
    og_images = getFeatureTensorsFrom(original_path)
    uncloaked_images = getFeatureTensorsFrom(uncloaked_path)
    cloaked_images = getFeatureTensorsFrom(cloaked_path)

    uncloaked_metrics, _= getCLIPSimilarityScores(og_images, uncloaked_images)
    cloaked_metrics, _ = getCLIPSimilarityScores(og_images, cloaked_images)
    return uncloaked_metrics, cloaked_metrics

def CLIP_prompt_metric(cloaked_path, uncloaked_path):
    uncloaked_images = getImageTensorsFrom(uncloaked_path)
    cloaked_images = getImageTensorsFrom(cloaked_path)

    uncloaked_metrics, _ = getCLIPPromptScores(uncloaked_images)
    cloaked_metrics, _ = getCLIPPromptScores(cloaked_images)
    return uncloaked_metrics, cloaked_metrics

