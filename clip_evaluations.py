import argparse

import torch
import clip
import os
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

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
      similarity = cos(image1_features[0],image2_features[0]).item()
      similarity = (similarity+1)/2
      metric_values["CLIP_imgsimilarity"] += similarity
      all_scores.append(similarity)

  for metric in metric_values:
    metric_values[metric] /= (len(og_images) * len(generated_images))

  return metric_values, all_scores

# Image-Caption Pair similarity using CLIP

def getCLIPPromptScores(generated_images, prompt=0):
  metrics = ["CLIP_promptsimilarity"]
  all_scores = []

  if prompt == 0:
    labels = ["a dog", "an innocent person", "a cat", "a person holding a gun"]

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

  for metric in metric_values:
    metric_values[metric] = np.max(metric_values[metric])
    
  return metric_values, all_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run CLIP metrics')
    parser.add_argument('--celeb', type=int, help='Description of the argument')
    args = parser.parse_args()
    experiment_directory = "experiments/experiment_" + str(args.celeb)


    og_images = getImageTensorsFrom("CelebA/" + str(args.celeb))
    uncloaked_images = getImageTensorsFrom(experiment_directory + "/output-real-p0")
    cloaked_images = getImageTensorsFrom(experiment_directory + "/output-pgd-p0")
    og_features = getFeatureTensorsFrom("CelebA/" + str(args.celeb))
    uncloaked_features = getFeatureTensorsFrom(experiment_directory + "/output-real-p0")
    cloaked_features = getFeatureTensorsFrom(experiment_directory + "/output-pgd-p0")
    np.set_printoptions(suppress=True)
    
    uncloaked_metrics, _ = getCLIPPromptScores(uncloaked_images)
    uncloaked_metrics2, _ = getCLIPSimilarityScores(og_features, uncloaked_features)
    csv_file = experiment_directory + '/clip_metrics_uncloaked_p0.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in uncloaked_metrics.items():
            writer.writerow([key, value])
        for key, value in uncloaked_metrics2.items():
            writer.writerow([key, value])
    
    cloaked_metrics, _ = getCLIPPromptScores(cloaked_images)
    cloaked_metrics2, _ = getCLIPSimilarityScores(og_features, cloaked_features) 
    csv_file = experiment_directory + '/clip_metrics_pgd_p0.csv'
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in cloaked_metrics.items():
            writer.writerow([key, value])
        for key, value in cloaked_metrics2.items():
            writer.writerow([key, value])
            
    print("Unloaked metrics:", uncloaked_metrics)
    print("Cloaked metrics:", cloaked_metrics)
    print("Unloaked metrics:", uncloaked_metrics2)
    print("Cloaked metrics:", cloaked_metrics2)
