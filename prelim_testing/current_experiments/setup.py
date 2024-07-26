import os, sys
import os, shutil, zipfile
import numpy as np
import pandas as pd
import json
import torch

USING_GDRIVE = False
malicious_activities = ['holding_gun', 'drinking_liquor']

def install_modules():
    if not os.path.exists('train_dreambooth.py'):
        os.system('wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py')
    if not os.path.exists('convert_diffusers_to_original_stable_diffusion.py'):
        os.system('wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py')
    if USING_GDRIVE:
        os.system('pip install -qq git+https://github.com/ShivamShrirao/diffusers')
        os.system('pip install -q -U --pre triton')
        os.system('pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers')
        os.system('pip install -U xformers --index-url https://download.pytorch.org/whl/cu118')

def add_huggingface_token():
  os.system('mkdir -p ~/.huggingface')
  os.system('echo -n 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 > ~/.huggingface/token')
  
def download_celeba():
    if os.path.exists('CelebA-HQ'):
      return
    os.system('wget -q https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/Eb37jNPPA7hHl0fmktYqcV8B-qmPLx-ZKYQ1eFk4UPBV_A?download=1 -O CelebA-HQ.zip')
    os.system('wget -q https://postechackr-my.sharepoint.com/:t:/g/personal/dongbinna_postech_ac_kr/EVRoUY8_txRFv56-KWvZrksBDWbD6adkjBxwwRN7qAC6bg?download=1 -O CelebA-HQ-identities')

    # Extract dataset
    with zipfile.ZipFile("CelebA-HQ.zip") as f:
        f.extractall()

    # Select images and remove unnecessary files
    os.system('mv CelebAMask-HQ/CelebA-HQ-img CelebA-HQ')
    os.system('rm -r CelebAMask-HQ')

    # Read identities as array
    ids = pd.read_csv("identity_CelebA.txt", sep=" ", header=None, usecols=[1]).to_numpy().reshape(-1)

    # Create identities and copy images to corresponding directory
    with open("CelebA-HQ-identities") as f:
        for line in f:
            img, i = line.rstrip("\n").split(" ")

            out_dir = os.path.join("CelebA", i)
            os.makedirs(out_dir, exist_ok=True)

            in_path = os.path.join("CelebA-HQ", img)
            out_path = os.path.join(out_dir, img)
            shutil.copy(in_path, out_path)

def download_action_photos():
  os.system('mkdir -p data')
  if USING_GDRIVE:
    for malicious_activity in malicious_activities:
      os.system(f'cp -r /content/drive/MyDrive/{malicious_activity} /content/data/{malicious_activity}')

install_modules()
add_huggingface_token()
#download_celeba()
if USING_GDRIVE:
    download_action_photos()

# Move Mr Beast photos
if USING_GDRIVE:
  os.system('cp -r /content/drive/MyDrive/mr_beast /content/celeba/mr_beast')