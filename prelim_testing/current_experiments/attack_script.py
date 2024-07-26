# This file is run as a script to execute attacks in attacks.property
# Copying code form Vaibhav notebook: https://colab.research.google.com/drive/1e5V6OiuxlhQHuIcLUSscKg9AQetir4mt#scrollTo=sv4w79HQevUU

from current_experiments.attacks import adv_loss
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
from argparse import Namespace
import numpy as np
from PIL import Image
import glob

WEIGHT_DIR = "weights"

# Create inference pipeline
model_path = os.path.join("/content", WEIGHT_DIR, "800")
pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
g_cuda = None

def generate_attacked_faces(img_path, model, attack, save_path):
    #load images from img_path
    valid_path = "{}/*.*".format(img_path)
    images = map(Image.open, glob.glob(valid_path))
    for image in images:
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = 2.0 * image - 1.0
        adv_image = adv_loss(model, image, attack=attack, eps=16/255, alpha=0.02, steps=1000)
        adv_image = (adv_image / 2 + 0.5).clamp(0, 1)
        #save adv_image to img_path
        adv_image = adv_image[0].detach().cpu().numpy()
        img = Image.fromarray((adv_image * 255).astype(np.uint8))
        img.save(save_path)

IN_DIR = os.path.join("CelebA", str(id))
OUT_DIR = os.path.join("cloaked", str(id))
if not os.path.exists(OUT_DIR):
    os.system(f'mkdir -p {OUT_DIR}')

generate_attacked_faces(IN_DIR, pipe, "pgd", OUT_DIR)