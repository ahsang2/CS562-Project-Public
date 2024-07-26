import os, sys, json, shutil, zipfile
import numpy as np
import pandas as pd
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torchvision

USING_GDRIVE = False
MODEL_NAME = "runwayml/stable-diffusion-v1-5"
OUTPUT_DIR = "dreambooth-weights"
os.system(f'mkdir -p {OUTPUT_DIR}')

# Parsing arguments
if len(sys.argv) < 2:
    raise Exception('Invalid arguments')
celeb_id = sys.argv[1]
malicious_activity = "holding gun"
if len(sys.argv) == 3:
  malicious_activity = sys.argv[2]

def create_concept_list():
    concepts_list = [
        {
            "instance_prompt":      "photo of sks person",
            "class_prompt":         "photo of a person",
            "instance_data_dir":    os.path.join("CelebA", str(celeb_id)),
            "class_data_dir":       "person"
        },
        {
            "instance_prompt":      f"photo of person {malicious_activity}",
            "class_prompt":         "photo of a person",
            "instance_data_dir":    f"training-photos/{malicious_activity.replace(' ', '_')}",
            "class_data_dir":       "person"
        }
    ]
    # `class_data_dir` contains regularization images
    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

def finetune_dreambooth():
    finetune_command = f'python3 train_dreambooth.py \
    --pretrained_model_name_or_path={MODEL_NAME} \
    --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
    --output_dir={OUTPUT_DIR} \
    --revision="fp16" \
    --with_prior_preservation --prior_loss_weight=1.0 \
    --seed=1337 \
    --resolution=512 \
    --train_batch_size=1 \
    --train_text_encoder \
    --mixed_precision="fp16" \
    --use_8bit_adam \
    --gradient_accumulation_steps=1 \
    --learning_rate=1e-6 \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --num_class_images=50 \
    --sample_batch_size=4 \
    --max_train_steps=800 \
    --save_interval=10000 \
    --save_sample_prompt="photo of sks person" \
    --concepts_list="concepts_list.json"'
    os.system(finetune_command)

def generate_mimicry_images():
    model_path = "/content/dreambooth-weights/800"             # If you want to use previously trained model saved in gdrive, replace this with the full path of model in gdrive
    pipe = StableDiffusionPipeline.from_pretrained(model_path, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    g_cuda = torch.Generator(device='cuda')
    seed = 52362 #@param {type:"number"}
    g_cuda.manual_seed(seed)
    prompt = f"photo of sks person {malicious_activity}"
    negative_prompt = ""
    num_samples = 10 
    guidance_scale = 7.5 #@param {type:"number"}
    num_inference_steps = 24 #@param {type:"number"}
    height = 512 #@param {type:"number"}
    width = 512 #@param {type:"number"}

    with autocast("cuda"), torch.inference_mode():
        images = pipe(
            prompt,
            height=height,
            width=width,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_samples,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=g_cuda
        ).images

    os.system('mkdir -p celeba_mimics')
    os.system(f'mkdir -p celeba_mimics/{celeb_id}')
    i = 0
    for img in images:
        img.save(f"celeba_mimics/{celeb_id}/{malicious_activity.replace(' ', '_')}{i}.jpg")
        i += 1

create_concept_list()
finetune_dreambooth()
generate_mimicry_images()



