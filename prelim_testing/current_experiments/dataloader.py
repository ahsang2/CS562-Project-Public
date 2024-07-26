import os, shutil, zipfile
import pandas as pd

# Removing classes with less than 5 images
def remove_classes():
  for class_dir in os.listdir('CelebA'):
    class_path = os.path.join('CelebA', class_dir)
    if os.path.isdir(class_path) and len(os.listdir(class_path)) < 5:
      os.system(f'rm -rf {class_path}')

def load_celeba():
  if os.path.exists('CelebA-HQ'):
    remove_classes()
    return
  # Download CelebA-HQ dataset and identities
  os.system("wget -q https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/Eb37jNPPA7hHl0fmktYqcV8B-qmPLx-ZKYQ1eFk4UPBV_A?download=1 -O CelebA-HQ.zip")
  os.system("wget -q https://postechackr-my.sharepoint.com/:t:/g/personal/dongbinna_postech_ac_kr/EVRoUY8_txRFv56-KWvZrksBDWbD6adkjBxwwRN7qAC6bg?download=1 -O CelebA-HQ-identities")

  # Extract dataset
  with zipfile.ZipFile("CelebA-HQ.zip") as f:
    f.extractall()

  # Extract images and remove unnecessary files
  os.system("mv CelebAMask-HQ/CelebA-HQ-img CelebA-HQ")
  os.system("rm -r CelebAMask-HQ")

  # Create identities and copy images to corresponding directory
  with open("CelebA-HQ-identities") as f:
    for line in f:
      img, i = line.rstrip("\n").split(" ")

      out_dir = os.path.join("CelebA", i)
      os.makedirs(out_dir, exist_ok=True)

      in_path = os.path.join("CelebA-HQ", img)
      out_path = os.path.join(out_dir, img)
      shutil.copy(in_path, out_path)
  remove_classes()

load_celeba()
